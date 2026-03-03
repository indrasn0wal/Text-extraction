"""
Step 1: Setup directories and convert PDFs to page images.

Uses PyMuPDF (fitz) for PDF conversion — NOT pdf2image.
GT pages and unlabeled pages are separated during conversion.

Input:  PDF files in SCANS_PATH
Output:
    - gt_pages/          — ground truth page images per source
    - unlabeled_pages/   — all other pages for MAE pretraining
    - gt_lines/          — (empty, filled in later by surya)
    - transcriptions/    — (empty, filled in later)

Usage:
    python 01_setup_and_pdf_to_images.py
"""

import os
import gc
import fitz
from PIL import Image

# --- CONFIG ---
SCANS_PATH = "/kaggle/input/datasets/indrasn0wal/humanai-print-ocr/printed/Text Scans"
TRANSCRIPTIONS_PATH = "/kaggle/input/datasets/indrasn0wal/humanai-print-ocr/printed/Transcription source text"
OUTPUT_PATH = "/kaggle/working"

# Ground truth page mapping (0-indexed PDF pages)
# PORCONES.23.5 has double column pages — handled in step 2
GROUND_TRUTH_PAGE_INDICES = {
    "Guardiola - Tratado nobleza.pdf":  [11, 12, 13],
    "PORCONES.23.5 - 1628.pdf":         [0, 1, 2, 3],
    "PORCONES.228.38 – 1646.pdf":       [0, 1, 2, 3, 4],
    "PORCONES.748.6 – 1650.pdf":        [0, 1, 2, 3],
    "Covarrubias - Tesoro lengua.pdf":  [6, 7, 8],
    "Buendia - Instruccion.pdf":        [1, 2, 3],
}

TRANSCRIPTION_FILES = {
    "Guardiola - Tratado nobleza.pdf":  "Guardiola - Tratado nobleza transcription.docx",
    "PORCONES.23.5 - 1628.pdf":         "PORCONES.23.5 - 1628 transcription.docx",
    "PORCONES.228.38 – 1646.pdf":       "PORCONES.228.38 - 1646 transcription.docx",
    "PORCONES.748.6 – 1650.pdf":        "PORCONES.748.6 – 1650 Transcription.docx",
    "Covarrubias - Tesoro lengua.pdf":  "Covarrubias - Tesoro lengua transcription.docx",
    "Buendia - Instruccion.pdf":        "Buendia - Instruccion transcription.docx",
}


def setup_directories(output_path):
    """Create all required output directories."""
    dirs = [
        f"{output_path}/gt_pages",
        f"{output_path}/unlabeled_pages",
        f"{output_path}/gt_lines",
        f"{output_path}/transcriptions",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created: {d}")
    print("\nDirectories ready.")


def extract_pages(pdf_filename, gt_indices, output_path, dpi=300):
    """
    Convert a single PDF to page images.
    GT pages → gt_pages/source_name/
    All other pages → unlabeled_pages/source_name/

    Args:
        pdf_filename: PDF filename
        gt_indices: List of 0-indexed page numbers that are GT
        output_path: Root output directory
        dpi: Resolution for conversion
    """
    pdf_path = os.path.join(SCANS_PATH, pdf_filename)
    if not os.path.exists(pdf_path):
        print(f"Skipping: {pdf_filename} not found.")
        return

    # Source name — replace spaces and special chars
    source_name = (
        pdf_filename
        .replace(".pdf", "")
        .replace(" ", "_")
        .replace("–", "-")
    )

    gt_dir = os.path.join(output_path, "gt_pages", source_name)
    unlabeled_dir = os.path.join(output_path, "unlabeled_pages", source_name)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(unlabeled_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        target_dir = gt_dir if page_num in gt_indices else unlabeled_dir
        img.save(
            os.path.join(target_dir, f"p{page_num + 1}_full.jpg"),
            quality=85
        )

        # Clear memory
        del pix
        img.close()
        del img
        if page_num % 10 == 0:
            gc.collect()

    doc.close()
    print(f"Finished: {pdf_filename}")


def run():
    # Step 1 — Create directories
    setup_directories(OUTPUT_PATH)

    # Step 2 — Convert PDFs
    print("\nConverting PDFs to images...")
    for pdf_file, gt_pages in GROUND_TRUTH_PAGE_INDICES.items():
        extract_pages(pdf_file, gt_pages, OUTPUT_PATH)
        gc.collect()

    print("\nExtraction complete.")


if __name__ == "__main__":
    run()