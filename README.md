# SATD Embedding Comparison

Frozen text embeddings for **identifying** and **categorizing** Self-Admitted Technical Debt (SATD) in four software artifacts: code comments, issues, commit messages, and pull requests.

This repository compares modern embedding models (**Qwen3-Embedding-0.6B**, **E5-Large-v2**, **BGE-M3**) with XGBoost or logistic regression against the BiLSTM / BERT + AugGPT setup of Sutoyo, Avgeriou, and Capiluppi (2024). Two data protocols are evaluated so architectural effects can be distinguished from distribution effects.

| | |
|---|---|
| **Baseline paper** | Sutoyo, E., Avgeriou, P., & Capiluppi, A. (2024). *Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt.* [arXiv:2410.15804](https://arxiv.org/abs/2410.15804) |
| **Progress report** | [docs/SATD_Research_Progress.md](docs/SATD_Research_Progress.md) |
| **Data and models** | [Google Drive](https://drive.google.com/drive/folders/1E-jzrNGE2NyKEsrI8Ud9Phk3gx_8a2dD?usp=sharing) |
| **Repository** | [A-Piyas-04/SATD-embedding-DP2](https://github.com/A-Piyas-04/SATD-embedding-DP2) |

---

## Findings

Pipeline A and Pipeline B must **not** be averaged into a single headline score. They use different corpora and augmentation policies.

**Stable across both pipelines**

- Frozen embeddings + XGBoost outperform fine-tuned BERT on SATD **categorization** for code comments, issues, and pull requests.
- Commit-message categorization remains below the BERT baseline (supervisor replication: 0.9804).
- XGBoost outperforms logistic regression in every comparison that was recorded.

**Sensitive to data protocol**

- **Pipeline A** (31-source rebuild, T5 class equalization): identification loses to BiLSTM on all four artifacts. Best overall average on this protocol: Qwen3 + XGBoost, macro F1 **0.8648**.
- **Pipeline B** (paper AugGPT files, Not-SATD majority retained): identification is competitive—Qwen3 beats BiLSTM on comments (0.9664 vs 0.939); E5 beats it on pull requests (0.8703 vs 0.862). Best combined average with XGBoost: BGE-M3 **0.9261**, Qwen3 **~0.9248**, E5 **0.9206**.

Classifier choice matters more than swapping among the three frozen encoders. The largest identification shift is Pipeline A versus Pipeline B, not encoder size.

Full tables, caveats, and research questions: [docs/SATD_Research_Progress.md](docs/SATD_Research_Progress.md).

---

## Experimental design

| | Identification | Categorization |
|---|---|---|
| **Task** | SATD vs Not-SATD | C/D, REQ, TES, DOC (SATD rows only) |
| **Baseline** | GloVe + BiLSTM + AugGPT | Fine-tuned BERT + AugGPT |
| **This work** | Frozen embeddings + XGBoost / logistic regression | Same |

Models are trained **separately per artifact**. The primary metric is **macro F1** on a held-out test set. Encoders are frozen (mean-pooled, L2-normalized); no BERT or BiLSTM fine-tuning is performed here.

**Labels:** Not-SATD, C/D (code or design), REQ, TES, DOC. Defect, Architecture, and Build labels are excluded, matching the reference study.

---

## Pipelines

| Pipeline | Data | Status |
|---|---|---|
| **[A — from-scratch](docs/pipeline_a_from_scratch.md)** | Merge of 31 sources, T5 paraphrasing to equalize five classes, split after augmentation | Phases 1–6 complete |
| **[B — paper data](docs/pipeline_b_paper_data.md)** | Four AugGPT CSVs from the paper; Not-SATD majority kept; split before extra augmentation | Qwen3, E5, and BGE-M3 complete |

Pipeline A includes the paper’s own `data-augmentation-*` files among the 31 sources. It is a multi-source rebuild, not a raw-only corpus. Splitting after paraphrasing is a leakage risk (synthetic siblings across splits).

---

## Data and models

Embeddings, processed CSVs, and trained classifiers are not stored in git. Download them from:

**[SATD Project — Data and Models (Google Drive)](https://drive.google.com/drive/folders/1E-jzrNGE2NyKEsrI8Ud9Phk3gx_8a2dD?usp=sharing)**

File inventory: [`data/README.md`](data/README.md).

```
SATD Project - Data & Models/
├── 01_raw_sources/        original 31 source files (zipped)
├── 02_processed_data/     cleaned, balanced, and split CSVs
├── 03_embeddings/         qwen3_*.npy, e5_*.npy
└── 04_trained_models/     classifier .joblib files
```

---

## Repository layout

```
satd-embedding-comparison/
├── notebooks/full_pipeline.ipynb    Pipeline A, phases 1–6
├── navid-experiment/                Pipeline B notebook and reports
├── scripts/                         standalone scripts for each phase
├── data/keywords/                   KeyBERT SATD keyword lists (unused in reported models)
├── results/                         Pipeline A phase 5/6 CSVs and summary PDF
├── docs/
│   ├── SATD_Research_Progress.md    progress report (Markdown)
│   ├── SATD_Research_Progress.pdf   same report (PDF)
│   ├── pipeline_a_from_scratch.md
│   ├── pipeline_b_paper_data.md
│   └── thesis_research_narrative.md
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/A-Piyas-04/SATD-embedding-DP2.git
cd SATD-embedding-DP2
pip install -r requirements.txt
```

Place Drive downloads on the paths expected by the notebooks. GPU (CUDA) is required for embedding generation and Pipeline A paraphrasing; classifier training runs on CPU.

| Dependency | Version |
|---|---|
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| scikit-learn | ≥ 1.3 |
| xgboost | ≥ 2.0 |
| transformers | ≥ 4.40 |
| torch | ≥ 2.0 |
| joblib | ≥ 1.3 |

---

## Documentation

| Document | Purpose |
|---|---|
| [SATD_Research_Progress.md](docs/SATD_Research_Progress.md) | Problem, RQs, chronology, both pipelines, limitations |
| [pipeline_a_from_scratch.md](docs/pipeline_a_from_scratch.md) | Pipeline A methods and phases |
| [pipeline_b_paper_data.md](docs/pipeline_b_paper_data.md) | Pipeline B methods |
| [thesis_research_narrative.md](docs/thesis_research_narrative.md) | Longer narrative reconstructed from the repo |
| [navid-experiment/satd_report.md](navid-experiment/satd_report.md) | Pipeline B Qwen3 / E5 report |
| [navid-experiment/satd_report_bge.md](navid-experiment/satd_report_bge.md) | Pipeline B BGE-M3 report |

---

## Planned work

Keyword-feature phases (7–9: code-specific encoders, keyword signals, combined retraining) remain future work. Keyword lists are already in `data/keywords/`. Broader thesis directions include SATD in Agile practice and links between SATD, developer emotion, and issue resolution time.
