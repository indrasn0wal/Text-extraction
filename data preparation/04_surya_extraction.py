"""
Step 3: Extract text line crops from GT pages using Surya layout detection.

Input:  GT pages folder (data/gt_pages/)
Output:
    - data/lines_detected_surya/  — cropped line images organized by source/page
    - data/layout_maps.json       — layout metadata for all pages

Pipeline per page:
    1. Run Surya FoundationPredictor + LayoutPredictor → detect text blocks
    2. Filter blocks (remove marginalia, captions, footnotes)
    3. Handle drop caps (ornate letters kept if far left)
    4. For section headers/titles → save entire block as one image
    5. For text blocks → run DetectionPredictor → merge nearby lines → save crops

Note:
    After extraction, line crops were manually verified and matched to
    ground truth transcriptions to create train_lines.json and test_lines.json.

Usage:
    python 03_surya_extraction.py \
        --gt_dir data/gt_pages \
        --output_dir data/lines_detected_surya
"""

import os
import gc
import torch
import json
import argparse
from PIL import Image
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.foundation import FoundationPredictor
from surya.settings import settings

# --- MEMORY CONFIGURATION ---
os.environ["DETECTOR_BATCH_SIZE"] = "2"
os.environ["LAYOUT_BATCH_SIZE"] = "2"
Image.MAX_IMAGE_PIXELS = None

# --- CONFIG ---
DEFAULT_GT_PAGES_PATH = "data/gt_pages"
DEFAULT_SURYA_LINES_PATH = "data/lines_detected_surya"
DEFAULT_LAYOUT_MAP_PATH = "data/layout_maps.json"

# Labels considered as main text content
TARGET_LABELS = [
    "text", "sectionheader", "title",
    "section-header", "section_header", "pageheader"
]


def clear_memory():
    """Clear Python garbage collector and CUDA cache to prevent OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_layout_map(image_path):
    """
    Identifies structural blocks using Surya FoundationPredictor + LayoutPredictor.
    Clears memory immediately after to prevent RAM crashes.

    Returns:
        List of layout blocks sorted by reading position
    """
    img = Image.open(image_path).convert("RGB")
    fp = FoundationPredictor(
        checkpoint=settings.LAYOUT_MODEL_CHECKPOINT
    )
    lp = LayoutPredictor(fp)

    layout_results = lp([img])[0]

    # Sort by 'position' to maintain reading flow
    sorted_blocks = sorted(
        layout_results.bboxes,
        key=lambda x: x.position
    )

    # Explicit cleanup to prevent RAM crashes
    del lp, fp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sorted_blocks


def filter_layout_blocks(image_path, raw_blocks):
    """
    Filters blocks to eliminate marginalia and decorative elements.
    Handles drop cap (ornate capital letter) exception.

    Filtering rules:
        - Keep only TARGET_LABELS
        - Reject blocks starting after 80% of page width (marginalia)
        - Reject blocks narrower than 15% of page width
        - Reject caption, footnote, formula labels
        - Exception: keep 'figure' blocks in far left (drop caps)

    Returns:
        accepted_blocks: List of block dicts
        rejected_blocks: List of block dicts
    """
    img = Image.open(image_path)
    width, height = img.size

    BOTTOM_PADDING = 20

    accepted_blocks = []
    rejected_blocks = []

    for block in raw_blocks:
        x1, y1, x2, y2 = block.bbox

        # Normalize label
        clean_label = (
            block.label.lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        # Convert Surya object to dict for modification
        block_dict = block.model_dump()

        # Rule 1 — Reject blocks too far right (marginalia)
        is_not_too_far_right = x1 < (width * 0.80)

        # Rule 2 — Reject blocks too narrow (sidenotes)
        is_wide_enough = (x2 - x1) > (width * 0.15)

        # Rule 3 — Reject marginal label types
        is_not_marginal_label = clean_label not in [
            "caption", "footnote", "formula"
        ]

        # Exception — Drop cap: ornate figure on far left
        is_drop_cap = (
            clean_label == "figure" and x1 < (width * 0.25)
        )

        # Final decision
        if (
            clean_label in [l.replace("-", "").replace("_", "")
                            for l in TARGET_LABELS]
            and is_not_too_far_right
            and is_wide_enough
            and is_not_marginal_label
        ) or is_drop_cap:
            # Apply bottom padding to avoid clipping descenders
            block_dict['bbox'][3] = min(height, y2 + BOTTOM_PADDING)
            accepted_blocks.append(block_dict)
        else:
            rejected_blocks.append(block_dict)

    print(f"  Total Raw Blocks: {len(raw_blocks)}")
    print(f"  Accepted Blocks:  {len(accepted_blocks)}")
    print(f"  Rejected Blocks:  {len(rejected_blocks)}")

    return accepted_blocks, rejected_blocks


def merge_nearby_lines(bboxes, x_threshold=100):
    """
    Groups bboxes on the same vertical level that are close horizontally.
    Fixes split titles like 'INFINITAMENTE    AMABLE' → one box.

    Args:
        bboxes: List of Surya bbox objects
        x_threshold: Max horizontal gap to merge (pixels)

    Returns:
        List of merged [x1, y1, x2, y2] boxes
    """
    if not bboxes:
        return []

    # Sort by top Y then left X
    sorted_bboxes = sorted(
        bboxes,
        key=lambda b: (b.bbox[1], b.bbox[0])
    )

    merged = []
    current = list(sorted_bboxes[0].bbox)

    for next_box in sorted_bboxes[1:]:
        nx1, ny1, nx2, ny2 = next_box.bbox
        vertical_mid = (ny1 + ny2) / 2

        # Merge if vertically overlapping and horizontally close
        if (
            current[1] <= vertical_mid <= current[3]
            and (nx1 - current[2]) < x_threshold
        ):
            current[0] = min(current[0], nx1)
            current[1] = min(current[1], ny1)
            current[2] = max(current[2], nx2)
            current[3] = max(current[3], ny2)
        else:
            merged.append(current)
            current = list(next_box.bbox)

    merged.append(current)
    return merged


def extract_lines_from_blocks(image_path, accepted_blocks, output_folder):
    """
    Extract individual line crops from accepted layout blocks.

    Logic:
        - Section headers / titles → save entire block as one image
        - Text blocks → run Surya DetectionPredictor → merge nearby
          lines → save individual line crops

    Args:
        image_path: Path to full page image
        accepted_blocks: Filtered block dicts from filter_layout_blocks
        output_folder: Where to save line crop images

    Returns:
        Number of line crops saved
    """
    full_img = Image.open(image_path).convert("RGB")
    width, height = full_img.size

    det_predictor = DetectionPredictor()
    os.makedirs(output_folder, exist_ok=True)

    line_counter = 0

    for block in accepted_blocks:
        crop_box = block['bbox']
        block_crop = full_img.crop(crop_box)

        # Normalize label
        label = (
            block.get('label', 'text')
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        if "sectionheader" in label or "title" in label:
            # Save entire block as one image — no line detection needed
            file_name = f"line_{line_counter:04d}_{label}_full.png"
            block_crop.save(os.path.join(output_folder, file_name))
            line_counter += 1

        else:
            # Run line detection on text block
            line_results = det_predictor([block_crop])[0]

            # Merge split words using 15% of page width threshold
            merged_bboxes = merge_nearby_lines(
                line_results.bboxes,
                x_threshold=width * 0.15
            )

            for bbox in merged_bboxes:
                lx1, ly1, lx2, ly2 = bbox
                l_width = lx2 - lx1
                l_height = ly2 - ly1

                # Skip small symbols and noise
                if l_width < 40 and l_height < 40:
                    continue

                line_img = block_crop.crop(bbox)
                file_name = f"line_{line_counter:04d}_{label}.png"
                line_img.save(os.path.join(output_folder, file_name))
                line_counter += 1

    # Cleanup
    del det_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return line_counter


def run_extraction(
    gt_dir=DEFAULT_GT_PAGES_PATH,
    output_dir=DEFAULT_SURYA_LINES_PATH,
    layout_map_path=DEFAULT_LAYOUT_MAP_PATH
):
    """
    Run full Surya extraction pipeline on all GT pages.

    Args:
        gt_dir: Directory with GT page images organized by source
        output_dir: Output directory for line crops
        layout_map_path: Path to save layout metadata JSON
    """
    os.makedirs(output_dir, exist_ok=True)

    all_layout_maps = {}
    total_lines = 0

    for source_folder in sorted(os.listdir(gt_dir)):
        source_in = os.path.join(gt_dir, source_folder)
        if not os.path.isdir(source_in):
            continue

        print(f"\n{'='*50}")
        print(f"Source: {source_folder}")
        print(f"{'='*50}")

        for fname in sorted(os.listdir(source_in)):
            if not fname.endswith('.jpg'):
                continue

            # Create sub-subfolder named after image file
            # e.g. 'p2_full.jpg' → folder 'p2_full'
            file_slug = fname.replace('.jpg', '')
            page_out = os.path.join(output_dir, source_folder, file_slug)
            os.makedirs(page_out, exist_ok=True)

            image_path = os.path.join(source_in, fname)
            page_key = f"{source_folder}/{fname}"
            print(f"\nProcessing: {page_key}")

            # Step 1 — Layout detection
            raw_layout_blocks = get_layout_map(image_path)
            clear_memory()

            # Step 2 — Filter blocks
            accepted_blocks, rejected_blocks = filter_layout_blocks(
                image_path, raw_layout_blocks
            )

            # Save layout metadata
            all_layout_maps[page_key] = {
                "accepted": accepted_blocks,
                "rejected": [
                    b.model_dump() if hasattr(b, 'model_dump') else b
                    for b in rejected_blocks
                ],
                "total_raw": len(raw_layout_blocks),
                "total_accepted": len(accepted_blocks),
                "total_rejected": len(rejected_blocks)
            }

            # Step 3 — Extract lines
            n_lines = extract_lines_from_blocks(
                image_path, accepted_blocks,
                output_folder=page_out
            )
            print(f"  Lines extracted: {n_lines}")
            total_lines += n_lines

            clear_memory()

    # Save all layout maps
    with open(layout_map_path, 'w', encoding='utf-8') as f:
        json.dump(all_layout_maps, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Total lines extracted: {total_lines}")
    print(f"Layout maps saved to:  {layout_map_path}")
    print(f"Line crops saved to:   {output_dir}")
    print(f"{'='*50}")

    return total_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract line crops from GT pages using Surya"
    )
    parser.add_argument(
        "--gt_dir",
        default=DEFAULT_GT_PAGES_PATH,
        help="GT pages directory"
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_SURYA_LINES_PATH,
        help="Output directory for line crops"
    )
    parser.add_argument(
        "--layout_map_path",
        default=DEFAULT_LAYOUT_MAP_PATH,
        help="Path to save layout metadata JSON"
    )
    args = parser.parse_args()

    run_extraction(
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        layout_map_path=args.layout_map_path
    )