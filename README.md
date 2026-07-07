# SATD Embedding Comparison

Comparing modern LLM embedding models (Qwen3-Embedding, E5-Large) against the
BiLSTM/BERT pipeline from Sutoyo et al. (2024) for detecting and categorizing
Self-Admitted Technical Debt (SATD) across four software artifact types.

**Reference paper:**
Sutoyo, E., Avgeriou, P., & Capiluppi, A. (2024).
*Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt.*
arXiv:2410.15804

---

## Repository structure

```
satd-embedding-comparison/
├── notebooks/
│   └── full_pipeline.ipynb       full pipeline — Phases 1 through 6
├── scripts/                      standalone .py version of each phase
├── data/
│   ├── keywords/                 8 scored SATD keyword files (for future Phase 8)
│   └── README.md                 links to Google Drive for large data files
├── results/
│   ├── phase5_results_summary.csv
│   ├── phase6_identification_comparison.csv
│   ├── phase6_categorization_comparison.csv
│   └── SATD_Findings_Summary.pdf
├── docs/
│   ├── pipeline_overview.md      phase-by-phase technical overview
│   └── teammate_guide.md         how to replicate the full pipeline
├── requirements.txt
└── README.md
```

---

## Pipeline

| Phase | Description | Status |
|---|---|---|
| 1 — Clean | Merge 31 sources, tag artifact types, standardize labels | Done |
| 2 — Balance | Downsample Not-SATD, augment minority SATD classes (T5 paraphraser) | Done |
| 3 — Split | Stratified 80/10/10 train/val/test split (seed=42) | Done |
| 4 — Embed | Generate Qwen3-Embedding-0.6B and E5-Large-v2 vectors | Done |
| 5 — Train | XGBoost + Logistic Regression on each embedding (32 models total) | Done |
| 6 — Compare | Compare vs. paper's BiLSTM+AugGPT / BERT+AugGPT baseline | Done |
| 7 — Code embeddings | Evaluate CodeBERT, GraphCodeBERT, UniXcoder, CodeT5 | Future work |
| 8 — Keywords | Add weighted SATD keyword signal features | Future work |
| 9 — Retrain | Retrain classifiers with keywords concatenated to embeddings | Future work |

---

## Key results

**Best overall combination: Qwen3-Embedding + XGBoost (avg macro F1 = 0.8648)**

| Task | Finding |
|---|---|
| Identification (SATD vs Not-SATD) | Embeddings underperform BiLSTM baseline across all artifact types |
| Categorization (C/D, DOC, TES, REQ) | Embeddings match or beat BERT for 3 of 4 artifact types |
| Classifier comparison | XGBoost beats Logistic Regression in every single comparison (32/32) |

### Best overall result

**Qwen3 + XGBoost** achieved the best performance when averaged across all
artifact types and both tasks: **macro F1 = 0.8648**.

### Identification task (SATD vs. Not-SATD)

Our embedding approach underperformed the paper's BiLSTM baseline in every case.

| Metric | Ours | Paper (BiLSTM) |
|---|---|---|
| Best artifact type | Code comments (0.915) | Code comments (0.939) |

### Categorization task (SATD type)

For **3 out of 4** artifact types, our embeddings (E5/Qwen3 + XGBoost) beat the
paper's fine-tuned BERT:

| Artifact | Our best (E5/Qwen3 + XGBoost) | Paper's BERT | Outcome |
|---|---|---|---|
| Code comments | 0.904 | 0.882 | Won |
| Issues | 0.941 | 0.899 | Won |
| Pull requests | 0.936 | 0.876 | Won |
| Commit messages | 0.959 | 0.980 | Lost |

### Headline finding

> Frozen LLM embeddings + XGBoost cannot beat a purpose-trained BiLSTM for
> detecting SATD, but they outperform fine-tuned BERT for categorizing SATD types
> in 3 out of 4 artifact types — at far lower computational cost, since no
> fine-tuning is required.

Full results by artifact type are in `results/`. A summary PDF is at
`results/SATD_Findings_Summary.pdf`.

---

## Label scheme

| Label | Meaning |
|---|---|
| Not-SATD | Not technical debt |
| C/D | Code or design debt |
| REQ | Requirement debt |
| TES | Test debt |
| DOC | Documentation debt |

Defect, Architecture, and Build debt labels were excluded — not used in the
reference study.

---

## Data and models

Large files (embeddings, processed CSVs, trained models) are on Google Drive:

**[SATD Project — Data and Models](https://drive.google.com/drive/folders/1E-jzrNGE2NyKEsrI8Ud9Phk3gx_8a2dD?usp=sharing)**

```
SATD Project - Data & Models/
├── 01_raw_sources/        original 31 source files (zipped)
├── 02_processed_data/     satd_with_artifact_type.csv, satd_balanced.csv,
│                          satd_train/val/test.csv, label CSVs
├── 03_embeddings/         qwen3_*.npy, e5_*.npy (shape: n x 1024 each)
└── 04_trained_models/     32 trained classifier .joblib files
```

---

## How to run

See `docs/teammate_guide.md` for full setup and re-run instructions.

Quick start:
```bash
git clone https://github.com/YOUR_USERNAME/satd-embedding-comparison.git
cd satd-embedding-comparison
pip install -r requirements.txt
```

Then open `notebooks/full_pipeline.ipynb` on Kaggle (GPU T4 x2 recommended,
Internet ON, HF_TOKEN set as a Kaggle Secret).

---

## Future work

Three extensions are planned. Input files for Phase 8 are already prepared
in `data/keywords/`. See `docs/pipeline_overview.md` for full technical details
on each.

**Phase 7 — Code-specific embedding models**
Evaluate CodeBERT, GraphCodeBERT, UniXcoder, and CodeT5 as drop-in replacements
for Qwen3/E5, using the same classifier and evaluation setup. Code-specific
pre-training may improve SATD detection in code comments and commit messages.

**Phase 8 — SATD keyword signal features**
Build 9 weighted numeric features from 8 scored SATD keyword lists (stored in
`data/keywords/`), capturing how strongly each text matches known SATD patterns
by artifact type and debt category. Feature vectors are then concatenated onto
the embedding arrays before classifier training.

**Phase 9 — Combined embedding + keyword retraining**
Retrain XGBoost and Logistic Regression on the combined feature vectors
(embedding + keyword features) and compare against the Phase 5/6 baseline to
measure the isolated contribution of keyword signal on top of embeddings.

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
transformers>=4.40.0
torch>=2.0.0
joblib>=1.3.0
```

GPU (CUDA) required for Phases 2 and 4. All other phases run on CPU.