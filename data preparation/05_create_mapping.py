"""
Step 4: Create image-text mapping from Surya line crops and docx transcriptions.

Input:
    - data/lines_detected_surya/  — line crops from step 3
    - this is saved as surya_lines folder, shown in 
    - data/docx_lines.json        — ground truth text lines from .docx files

Output:
    - data/image_text_mapping.json — full mapping with metadata

Format of docx_lines.json:
    {
        "Buendia - Instruccion": {
            "p2_full": ["line1 text", "line2 text", ...],
            "p3_full": ["line1 text", ...],
            ...
        },
        ...
    }

Format of image_text_mapping.json:
    {
        "Buendia - Instruccion/p2_full": {
            "n_images": 25,
            "n_texts": 25,
            "n_pairs": 25,
            "pairs": [
                {
                    "image": "data/lines_detected_surya/.../line_0000.png",
                    "text": "guro diſſeño de ſu edad : la Reli-",
                    "source": "Buendia - Instruccion",
                    "page": "p2_full"
                },
                ...
            ]
        },
        ...
    }

Usage:
    python 04_create_mapping.py \
        --surya_dir data/lines_detected_surya \
        --docx_lines data/docx_lines.json \
        --output data/image_text_mapping.json
"""

import os
import json
import argparse

# --- CONFIG ---
DEFAULT_SURYA_DIR    = "data/surya_lines"
DEFAULT_DOCX_LINES   = "data/docx_lines.json"
DEFAULT_MAPPING_PATH = "data/image_text_mapping.json"


def create_mapping(surya_dir, docx_lines):
    """
    Match surya line crop images to docx text lines 1-to-1.

    Matching strategy:
        - Images sorted alphabetically (line_0000, line_0001 ...)
        - Text lines in order from docx extraction (top to bottom)
        - Pair up to min(n_images, n_texts)
        - Extra images or texts beyond min are discarded

    Args:
        surya_dir: Root directory of surya line crops
        docx_lines: Dict loaded from docx_lines.json
                    Format: {source: {page: [line1, line2, ...]}}

    Returns:
        mapping dict with full metadata per source/page
    """
    mapping = {}

    for source_folder in sorted(os.listdir(surya_dir)):
        if source_folder.startswith('.'):
            continue
        source_path = os.path.join(surya_dir, source_folder)
        if not os.path.isdir(source_path):
            continue

        for page_folder in sorted(os.listdir(source_path)):
            if page_folder.startswith('.'):
                continue
            page_path = os.path.join(source_path, page_folder)
            if not os.path.isdir(page_path):
                continue

            # Get sorted image files
            images = sorted([
                os.path.join(page_path, f)
                for f in os.listdir(page_path)
                if f.endswith('.png') and not f.startswith('.')
            ])

            # Get text lines for this source/page
            docx_text_lines = (
                docx_lines
                .get(source_folder, {})
                .get(page_folder, [])
            )

            # 1-to-1 pairing up to min length
            pairs = []
            for img, text in zip(images, docx_text_lines):
                text = text.strip()
                if not text:
                    continue
                pairs.append({
                    "image": img,
                    "text": text,
                    "source": source_folder,
                    "page": page_folder
                })

            full_key = f"{source_folder}/{page_folder}"
            mapping[full_key] = {
                "n_images": len(images),
                "n_texts":  len(docx_text_lines),
                "n_pairs":  len(pairs),
                "pairs":    pairs
            }

            print(
                f"{full_key}: "
                f"{len(images)} images | "
                f"{len(docx_text_lines)} texts | "
                f"{len(pairs)} pairs"
            )

    return mapping


def run(surya_dir, docx_lines_path, output_path):
    """
    Run mapping creation and save to JSON.

    Args:
        surya_dir: Surya line crops directory
        docx_lines_path: Path to docx_lines.json
        output_path: Where to save image_text_mapping.json
    """
    print(f"Loading docx lines from: {docx_lines_path}")
    with open(docx_lines_path, 'r', encoding='utf-8') as f:
        docx_lines = json.load(f)

    print(f"\nCreating mapping...")
    mapping = create_mapping(surya_dir, docx_lines)

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # Summary
    total_pairs = sum(v['n_pairs'] for v in mapping.items()
                      if isinstance(v, dict) and 'n_pairs' in v)
    total_pairs = sum(v['n_pairs'] for v in mapping.values())

    print(f"\n{'='*50}")
    print(f"Total source/page keys: {len(mapping)}")
    print(f"Total pairs:            {total_pairs}")
    print(f"Saved to:               {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create image-text mapping from Surya crops and docx lines"
    )
    parser.add_argument(
        "--surya_dir",
        default=DEFAULT_SURYA_DIR,
        help="Surya line crops directory"
    )
    parser.add_argument(
        "--docx_lines",
        default=DEFAULT_DOCX_LINES,
        help="Path to docx_lines.json"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_MAPPING_PATH,
        help="Output path for image_text_mapping.json"
    )
    args = parser.parse_args()

    run(
        surya_dir=args.surya_dir,
        docx_lines_path=args.docx_lines,
        output_path=args.output
    )