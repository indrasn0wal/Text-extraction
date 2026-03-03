# RenAIssance OCR — GSoC 2025 Evaluation Test

Automated transcription pipeline for 17th century Spanish printed documents.
Built as part of the RenAIssance GSoC 2025 evaluation test at CERN.

---

## Pipeline Overview
```
PDFs
 ↓
Data Preparation (data_preparation/)
 ├── PDF → Page Images (PyMuPDF)
 ├── GT / Unlabeled page separation
 ├── Double-column page splitting (PORCONES.23.5)
 ├── Transcription parsing (.docx → JSON)
 ├── Surya layout detection → line crops
 └── Image-text mapping → train/test split
 ↓
Training (notebooks/submission-humanai.ipynb)
 ├── MAE Pretraining (SSL experiment — unlabeled pages)
 ├── Synthetic data generation (TRDG)
 ├── Stage 1: TrOCR fine-tune on synthetic pairs
 └── Stage 2: TrOCR fine-tune on real pairs
 ↓
Inference Pipeline
 ├── CRAFT text detection
 ├── Layout analysis (projection profile)
 ├── Adaptive sidenote filtering (positional + statistical)
 ├── Line merging (fragmented headlines)
 ├── TrOCR recognition
 └── Gemini VLM post-correction
 ↓
Gradio Web App
```

---

## Dataset

6 historical Spanish sources from the 17th century:

| Source | GT Pages |
|--------|----------|
| Buendia - Instruccion | 3 |
| Covarrubias - Tesoro lengua | 3 |
| Guardiola - Tratado nobleza | 3 |
| PORCONES.228.38 | 5 |
| PORCONES.23.5 | 5 |
| PORCONES.748.6 | 4 |
| **Total** | **25** |

- **Training pairs:** 550 (line image + transcription text)
- **Test pairs:** 195 (manually verified)
- **Synthetic Training pairs:** 4932 (line image + transcription text)
- **Synthetic Data Link:** [Google Drive](https://drive.google.com/drive/folders/1d0T-uVZ_Ygi9cagTXVY78FZOd1guImMg?usp=sharing)
- **Data Containing the real Training and Testing images:** [Google Drive](https://drive.google.com/drive/folders/1qzgRcTdak-lyqYcl5Cl2Su6RvGkE_NGh?usp=sharing)
- **Train and Test JSON Pairs folder:** [Google Drive](https://drive.google.com/drive/folders/1FTjT3T7-H8V9zvB5ED-ryR_rHhsO7tWJ?usp=sharing)
- **Unlabelled and ground truth data:** [Kaggle Dataset](https://www.kaggle.com/datasets/indrasn0wal/maetaskocr/data)
- Only ```gt_pages``` and ```unlabeled_pages``` are relevant. 
---

## Results

### TrOCR Fine-tuned (Final Model)

| Metric | Score |
|--------|-------|
| CER | 11.33% |
| WER | 26.99% |

**Test result:** [Google Drive](https://drive.google.com/file/d/1k0N9NRx1NaL8Od3lKxmyVbRTTWyk4Tqw/view?usp=sharing)

### LLM Post-Processing (Gemini)

| | CER | WER |
|--|-----|-----|
| TrOCR only | 11.33% | 27.40% |
| + Gemini (line-level LLM) | 12.05% | 26.99% |

**Finding:** Line-level LLM correction degraded CER due to insertion of
modern accent marks not present in ground truth. Upgraded to VLM approach
(full page image + text) for contextual correction including drop cap
identification.

---

## Training Strategy

### Stage 1 — Synthetic Pretraining (TRDG)
- Generated 5,000 synthetic line images mimicking 17th century Spanish text
- Used TRDG (Text Recognition Data Generator)
- Teaches TrOCR basic character patterns: long-s (ſ), u/v swaps, f/s swaps
- Result: CER ~4% on synthetic validation

### Stage 2 — Real Data Fine-tuning
- Fine-tuned Stage 1 model on 550 real line pairs
- Real pairs extracted from 25 GT pages using Surya layout detection
- Manually verified for quality
- Result: CER 11.33% on 195 held-out test pairs

### SSL Experiments (MAE + DINOv2)
Attempted self-supervised pretraining on 1,373 unlabeled pages to improve
TrOCR encoder representations before fine-tuning. See README section below.

---

## SSL Experiments

### MAE (Masked Autoencoder)
- Pretrained custom MAE encoder on 1,373 unlabeled pages
- 30 epochs, loss: 0.0891 → 0.0053
- Attempted injection into TrOCR encoder
- **Result: Failed** — CER stuck at 74-79% after injection

### DINOv2 Injection
- Attempted replacing TrOCR encoder with DINOv2
- **Result: Failed** — architecture mismatch
  - `position_embeddings`: 1370 vs 577
  - `patch_embeddings`: 14×14 vs 16×16

### Conclusion
Domain adaptation via SSL injection requires matching architecture from
the start (e.g. SeqCLR approach). Fine-tuning vanilla TrOCR directly
on domain data proved more effective for this dataset size.

---

## Inference Pipeline Details

### Layout Analysis & Pre-processing

* **Marginal Crop:** Removes a hard outer margin (7%) and top/bottom edges (3%) to eliminate scanner noise and blank borders before detection.
* **Two-Column Detection:** Detects page layout using a **Vertical Projection Profile** to identify central "white valleys" without requiring heavy model inference.
* **Automated Splitting:** Programmatically splits two-page or two-column scans into independent segments to ensure correct chronological reading order.

### Text Detection — CRAFT

* **Character-Level Detection:** Utilizes the **CRAFT (Character Region Awareness for Text Detection)** model to identify precise line-level bounding boxes.
* **Historical Robustness:** Provides better spatial accuracy than standard region-level detectors for dense, 17th-century printed layouts.
* **Columnar Execution:** Runs independently on each detected column or page segment to maintain structural integrity.

### Geometric Filtering & "The Fence"

* **Positional Sidenote Filter:** Calculates a dynamic "geometric fence" using the median left and right edges of the column.
* **Marginalia Rejection:** Identifies and rejects boxes that are narrow AND sit outside the median margins (e.g., citations like "Marci. 10").
* **Sparse Text Preservation:** Accurately preserves short main-body lines (e.g., "Amen.") by verifying their alignment with the column's primary left margin.
* **Noise Suppression:** Filters out decorative symbols (page-top crosses), tiny ink artifacts, and ink blobs using aspect ratio and pixel density checks.

### Neural Transcription & Post-Filtering

* **Historical TrOCR:** Inputs each line crop into a fine-tuned **TrOCR (Transformer-based OCR)** model optimized for early-modern Spanish typography.
* **Catchword Filtering:** Analyzes the final line of every segment to programmatically remove isolated words or hyphenated fragments (e.g., "crian-").
* **Deterministic Integrity:** Ensures that single words in the middle of paragraphs are safely preserved while only targeting column-end fragments.

### Line Merging

* **Headline Reconstruction:** Merges fragmented headlines (e.g., "INFINITAMENTE AMABLE") into single cohesive bounding boxes.
* **Merge Heuristics:** Joins boxes sharing the same vertical level with a horizontal gap below a defined pixel threshold.

### Gemini VLM Post-Correction

* **Multimodal Refinement:** Sends the full-page image and concatenated raw OCR text to **Gemini 2.5 Flash** for final orchestration.
* **Drop-Cap Integration:** Identifies decorative woodcut letters in the image and prepends them to the first word of the paragraph.
* **Diplomatic Transcription:** Enforces strict line integrity, preventing the model from joining hyphenated line-endings or modernizing vocabulary.
* **Zero-Hallucination Mode:** Operates at `temperature: 0.0` to ensure the model does not "rescue" marginalia already removed by the geometric filters.


---

## Repository Structure
```
renaissance-ocr/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data_preparation/
│   ├── README.md
│   ├── 01_setup_and_pdf_to_images.py
│   ├── 02_split_double_column_pages.py
│   ├── 03_parse_transcriptions.py
│   ├── 04_surya_extraction.py
│   ├── 05_create_mapping.py
│   └── 06_create_train_test_split.py
│
└── notebooks/
    └── submission-humanai.ipynb
```



---

## Data Preparation
```bash
# Step 1 — Convert PDFs to images
python data_preparation/01_setup_and_pdf_to_images.py

# Step 2 — Split double column pages (PORCONES.23.5)
python data_preparation/02_split_double_column_pages.py

# Step 3 — Parse transcription docx files
python data_preparation/03_parse_transcriptions.py

# Step 4 — Extract line crops using Surya
python data_preparation/04_surya_extraction.py

# Step 5 — Create image-text mapping
python data_preparation/05_create_mapping.py

# Step 6 — Create train/test split
python data_preparation/06_create_train_test_split.py
```

---

## Training

See `notebooks/submission-humanai.ipynb` for full training pipeline
with outputs including loss, CER per epoch, and evaluation results.

---

## Running the Gradio App

The inference app is included at the end of `submission-humanai.ipynb`.

Requirements:
- Fine-tuned TrOCR model (link below)
- Gemini API key
- CRAFT pretrained weights (auto-downloaded)

**Model weights TrOCR:** [Drive Link](https://drive.google.com/file/d/1UgH0RczClJmuDLCIix3SIZZksBxP2N8c/view?usp=sharing)