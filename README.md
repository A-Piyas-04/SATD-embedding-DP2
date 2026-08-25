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

Scores are **macro F1** on a held-out test set (higher is better). Two jobs are measured separately:

| Job | Question the model answers |
|---|---|
| **Identification** | Is this text SATD or not? (compared to the paper’s **BiLSTM**) |
| **Categorization** | If it is SATD, which type? C/D, requirement, test, or documentation (compared to the paper’s **BERT**) |

We ran the same embedding + classifier recipe twice, on **different data recipes**. Those two runs are not interchangeable: do not average them into one “our F1.”

| | Pipeline A | Pipeline B |
|---|---|---|
| **What we trained on** | 31 SATD sources merged, then T5 paraphrases so all five labels (including Not-SATD) have similar counts | The paper’s four AugGPT files; Not-SATD stays the majority (~65–82%) |
| **What that means in practice** | The mix of sources and class balance is **not** the paper’s training distribution | Closer to how the baseline numbers were produced |

XGBoost beat logistic regression in every recorded pair. Tables below use the **XGBoost** head (the stronger, cheaper recipe). Commit identification/categorization baselines marked with \* are the **supervisor replication** (BiLSTM 0.91, BERT 0.9804), not necessarily the published paper row.

### Identification (SATD vs not)

On Pipeline A, frozen embeddings **never** beat BiLSTM. The worst drop is commits (−0.16). On Pipeline B, the same stack is **close or better** on comments and pull requests; issues and commits stay slightly below BiLSTM.

| Artifact | BiLSTM baseline | A — best of Qwen3/E5 | B — Qwen3 | B — E5 | B — BGE-M3 |
|---|---:|---:|---:|---:|---:|
| Code comments | 0.939 | 0.915 (Qwen3) | **0.966** | 0.959 | 0.959 |
| Issues | 0.878 | 0.798 (Qwen3) | 0.866 | 0.863 | 0.853 |
| Pull requests | 0.862 | 0.792 (Qwen3) | 0.864 | **0.870** | 0.858 |
| Commits | 0.910\* | 0.751 (E5) | 0.892 | 0.882 | 0.901 |

**Read this as:** “Can we detect SATD at all?” depends more on **how the dataset was built** than on which of Qwen3 / E5 / BGE we pick. Equalizing Not-SATD with T5 (Pipeline A) hurts identification; keeping the paper’s Not-SATD majority (Pipeline B) largely recovers it.

### Categorization (SATD type)

This result is **stable**. Frozen embeddings + XGBoost beat BERT on comments, issues, and pull requests in **both** pipelines. Commits are the exception: BERT stays ahead.

| Artifact | BERT baseline | A — best of Qwen3/E5 | B — Qwen3 | B — E5 | B — BGE-M3 |
|---|---:|---:|---:|---:|---:|
| Code comments | 0.882 | **0.904** (E5) | 0.947 | 0.958 | 0.956 |
| Issues | 0.899 | **0.941** (Qwen3) | 0.944 | 0.950 | **0.956** |
| Pull requests | 0.876 | **0.936** (E5) | 0.955 | 0.934 | **0.964** |
| Commits | 0.980\* | 0.959 (Qwen3) | 0.964 | 0.949 | 0.960 |

Pipeline B margins over BERT are larger than Pipeline A (e.g. pull requests: BGE-M3 **0.964** vs BERT **0.876**). No encoder we tested reaches the 0.980 commit BERT number.

### What to take away

1. **Typing SATD** (C/D vs REQ vs TES vs DOC) is where frozen embeddings shine: cheaper than fine-tuning BERT, and stronger on three of four artifacts.
2. **Finding SATD vs ordinary text** is not a property of the encoder alone. It tracks the data protocol. Pipeline B identification averages with XGBoost sit together (Qwen3 0.897, E5 0.894, BGE-M3 0.893).
3. **XGBoost vs logistic regression** is a bigger lever than swapping Qwen3, E5, or BGE-M3 on the same data. BGE-M3 + logistic regression is much weaker (combined ~0.85) than BGE-M3 + XGBoost (**0.926** across both jobs on Pipeline B).
4. Headline averages (only useful **inside** one pipeline): Pipeline A Qwen3+XGB **0.8648**; Pipeline B XGBoost combined BGE-M3 **0.9261**, Qwen3 **~0.9248**, E5 **0.9206**.

Limitations (split-after-paraphrase leakage in A, AugGPT files inside the 31-source merge, missing BGE training notebook) are in [docs/SATD_Research_Progress.md](docs/SATD_Research_Progress.md).

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
