"""
Step 2: Split double-column GT pages for PORCONES.23.5.

PORCONES.23.5 has double-column pages that need to be split
into left and right halves before line extraction.

p1 (index 0) → single full page — no split needed
p2, p3, p4   → double column   — split into left and right

Input:  gt_pages/PORCONES.23.5_-_1628/p*_full.jpg
Output: gt_pages/PORCONES.23.5_-_1628/p*_left.jpg + p*_right.jpg
        (original p*_full.jpg removed after split)

Usage:
    python 02_split_double_column_pages.py
"""

import os
import gc
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --- CONFIG ---
PORCONES_GT_PATH = "/kaggle/working/gt_pages/PORCONES.23.5_-_1628"

# Pages to split — original name → (left name, right name)
# p1_full.jpg is single column — not in this list
PAGES_TO_SPLIT = {
    "p2_full.jpg": ("p2_left.jpg",  "p2_right.jpg"),
    "p3_full.jpg": ("p3_left.jpg",  "p3_right.jpg"),
    "p4_full.jpg": ("p4_left.jpg",  "p4_right.jpg"),
}


def split_page(original_path, left_path, right_path):
    """
    Split a double-column page image at the horizontal midpoint.

    Args:
        original_path: Path to full page image
        left_path: Output path for left half
        right_path: Output path for right half
    """
    img = Image.open(original_path)
    mid = img.width // 2

    img.crop((0, 0, mid, img.height)).save(left_path, quality=85)
    img.crop((mid, 0, img.width, img.height)).save(right_path, quality=85)

    img.close()
    del img
    gc.collect()


def run():
    for original_name, (left_name, right_name) in PAGES_TO_SPLIT.items():
        original_path = os.path.join(PORCONES_GT_PATH, original_name)

        if not os.path.exists(original_path):
            print(f"Not found: {original_name} — skipping")
            continue

        left_path  = os.path.join(PORCONES_GT_PATH, left_name)
        right_path = os.path.join(PORCONES_GT_PATH, right_name)

        split_page(original_path, left_path, right_path)

        # Remove original full image — no longer needed
        os.remove(original_path)
        print(f"Split {original_name} → {left_name}, {right_name}")

    print("\nAll splits done. Current files:")
    for f in sorted(os.listdir(PORCONES_GT_PATH)):
        print(f"  {f}")


if __name__ == "__main__":
    run()