"""
Step 3: Parse transcription .docx files into line-level JSON.

Input:  .docx transcription files per source
Output: data/docx_lines.json

Docx format:
    Each page section starts with a label like:
        "PDF p3 - left"
        "PDF p3 – right"  
        "PDF p1"
    Followed by transcription text lines.
    Section ends at next page label or "END OF EXTRACT".

Output format:
    {
        "Buendia_-_Instruccion": {
            "p2_full": ["line1 text", "line2 text", ...],
            "p3_full": ["line1 text", ...],
        },
        "PORCONES.23.5_-_1628": {
            "p1_full":  ["line1", ...],
            "p2_left":  ["line1", ...],
            "p2_right": ["line1", ...],
            ...
        },
        ...
    }

Usage:
    python 03_parse_transcriptions.py \
        --transcriptions_dir /path/to/transcriptions_fixed \
        --output data/docx_lines.json
"""

import json
import os
import re
import argparse
import docx

# --- CONFIG ---
DEFAULT_TRANSCRIPTIONS_PATH = "/kaggle/working/transcriptions_fixed"
DEFAULT_OUTPUT_PATH          = "data/docx_lines.json"
MANUAL_REVIEW_PATH           = "/kaggle/working/manual_review"

# Maps source folder name → docx filename
DOCX_MAP = {
    "Guardiola_-_Tratado_nobleza": "Guardiola - Tratado nobleza transcription.docx",
    "PORCONES.23.5_-_1628":        "PORCONES.23.5 - 1628 transcription.docx",
    "PORCONES.228.38_-_1646":      "PORCONES.228.38 - 1646 transcription.docx",
    "PORCONES.748.6_-_1650":       "PORCONES.748.6 – 1650 Transcription.docx",
    "Covarrubias_-_Tesoro_lengua": "Covarrubias - Tesoro lengua transcription.docx",
    "Buendia_-_Instruccion":       "Buendia - Instruccion transcription.docx",
}

PAGE_LABEL_PATTERN = re.compile(
    r"^PDF\s+p(\d+)(?:\s*[-–]\s*(left|right))?$",
    re.IGNORECASE
)

# Only stop at END OF EXTRACT — NOTES: appears before page labels so handled separately
STOP_PATTERNS = ["END OF EXTRACT"]


def parse_docx_lines(docx_path):
    """
    Parse a transcription docx into page-keyed line lists.

    Returns:
        {
            "p1_full":  ["line1", "line2", ...],
            "p2_left":  ["line1", "line2", ...],
            "p3_right": ["line1", "line2", ...],
            ...
        }
    """
    doc = docx.Document(docx_path)
    pages = {}
    current_key = None
    found_first_page = False

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Stop at END OF EXTRACT — only after we've started reading pages
        if found_first_page and any(
            text.startswith(s) for s in STOP_PATTERNS
        ):
            break

        # Skip NOTES block — appears before any page label in some docx files
        if not found_first_page and text.startswith("NOTES:"):
            continue

        # Check if it's a page label
        match = PAGE_LABEL_PATTERN.match(text)
        if match:
            page_num = match.group(1)
            side = match.group(2)
            current_key = (
                f"p{page_num}_{side.lower()}"
                if side else
                f"p{page_num}_full"
            )
            pages[current_key] = []
            found_first_page = True

        elif current_key:
            # Regular text — split by newlines and clean
            for line in text.split('\n'):
                cleaned = ' '.join(line.split())
                if cleaned:
                    pages[current_key].append(cleaned)

    return pages


def run(transcriptions_dir, output_path):
    """
    Parse all transcription docx files and save as JSON.

    Args:
        transcriptions_dir: Directory containing .docx files
        output_path: Output path for docx_lines.json
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    docx_splits = {}

    for source_folder, docx_file in DOCX_MAP.items():
        docx_path = os.path.join(transcriptions_dir, docx_file)

        if not os.path.exists(docx_path):
            print(f"WARNING: Not found — {docx_path}")
            continue

        parsed = parse_docx_lines(docx_path)

        # Ensure all values are clean line lists
        fixed = {}
        for page_key, content in parsed.items():
            if isinstance(content, str):
                lines = [l.strip() for l in content.split('\n') if l.strip()]
            else:
                lines = [l.strip() for l in content if l.strip()]
            fixed[page_key] = lines

        docx_splits[source_folder] = fixed

        # Preview
        print(f"\n=== {source_folder} ===")
        for page_key, lines in fixed.items():
            print(f"  {page_key}: {len(lines)} lines")
            for i, line in enumerate(lines[:5]):
                print(f"    [{i:03d}] {line}")

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(docx_splits, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {output_path}")
    print(f"Sources: {len(docx_splits)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse transcription docx files into line-level JSON"
    )
    parser.add_argument(
        "--transcriptions_dir",
        default=DEFAULT_TRANSCRIPTIONS_PATH,
        help="Directory containing .docx transcription files"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for docx_lines.json"
    )
    args = parser.parse_args()

    run(
        transcriptions_dir=args.transcriptions_dir,
        output_path=args.output
    )