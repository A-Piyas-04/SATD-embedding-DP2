# SATD Embedding Comparison

> **Do frozen modern embeddings beat purpose-trained BiLSTM / BERT for Self-Admitted Technical Debt — and when does the data protocol matter more than the encoder?**

Frozen text embeddings for **identifying** (is this SATD?) and **categorizing** (what kind of debt?) Self-Admitted Technical Debt across four software artifacts — **code comments, issues, commit messages, and pull requests** — compared against the BiLSTM / BERT + AugGPT setup of Sutoyo, Avgeriou, and Capiluppi (2024).

This repository now covers the full arc from **controlled experiments (Pipelines A/B)** to **large-scale deployment**: the trained Pipeline A models have been run over **458,232 unseen GitHub-style issues** (title / description / combined) with an optimized 2×T4 Kaggle pipeline, and the resulting classifications are published as Parquet.

| | |
|---|---|
| **Baseline paper** | Sutoyo, E., Avgeriou, P., & Capiluppi, A. (2024). *Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt.* [arXiv:2410.15804](https://arxiv.org/abs/2410.15804) |
| **Progress report** | [docs/SATD_Research_Progress.md](docs/SATD_Research_Progress.md) |
| **Data and models** | [Google Drive — SATD Project Data & Models](https://drive.google.com/drive/folders/1E-jzrNGE2NyKEsrI8Ud9Phk3gx_8a2dD?usp=sharing) |
| **New-issue input (458k rows)** | [issue_202608272234.parquet](https://drive.google.com/file/d/1e9ZiR2VOglBilEEnJIfSWA_UkNudjEym/view?usp=sharing) |
| **New-issue output (classified)** | [satd_classification_results.parquet](https://drive.google.com/file/d/1BTzQmgwanxi5YdCh4zQUGOf1XT5CR5g-/view?usp=drive_link) |
| **Inference notebook (V2, executed)** | [`musaddiq_rafi/satd-classifications-optimized V2.ipynb`](musaddiq_rafi/satd-classifications-optimized%20V2.ipynb) |
| **Inference report (MD + PDF)** | [docs/satd_new_issue_inference_v2.md](docs/satd_new_issue_inference_v2.md) · [SATD_New_Issue_Inference_V2_Report.pdf](musaddiq_rafi/SATD_New_Issue_Inference_V2_Report.pdf) |
| **Repository** | [A-Piyas-04/SATD-embedding-DP2](https://github.com/A-Piyas-04/SATD-embedding-DP2) |

---

## Contents

- [Why this project exists](#why-this-project-exists)
- [Findings](#findings)
- [Large-scale inference on 458k new issues (V2 — executed)](#large-scale-inference-on-458k-new-issues-v2--executed)
- [Experimental design](#experimental-design)
- [Pipelines](#pipelines)
- [Data and models](#data-and-models)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Reproduce the 458k inference](#reproduce-the-458k-inference)
- [Documentation](#documentation)
- [Limitations](#limitations)
- [Planned work](#planned-work)
- [Citation](#citation)

---

## Why this project exists

Developers routinely leave explicit traces of compromise in text: *"TODO: hack, fix later"*, *"tests missing for this module"*, *"docs outdated"*. That signal is **Self-Admitted Technical Debt (SATD)** — valuable because the author has already flagged that work remains, painful because no team can read every comment, issue, commit, and pull request by hand.

Two questions matter in practice, and this repo treats them as **separate jobs**:

| Job | Question | Baseline it replaces |
|---|---|---|
| **Identification** | Is this text SATD or ordinary text? | GloVe + BiLSTM + AugGPT |
| **Categorization** | Given SATD, is it code/design (C/D), requirement (REQ), test (TES), or documentation (DOC) debt? | Fine-tuned BERT + AugGPT |

The core hypothesis: **frozen general-purpose embeddings + a lightweight XGBoost / logistic-regression head** can match or beat purpose-trained sequence models at a fraction of the training cost — but only if we are honest about *which data recipe* produced the numbers. Hence two pipelines (A and B) that share the same classifiers but differ in training distribution.

**Encoders tested:** `Qwen/Qwen3-Embedding-0.6B`, `intfloat/e5-large-v2`, `BAAI/bge-m3` (Pipeline B only). All are frozen, mean-pooled, L2-normalized. No BERT/BiLSTM fine-tuning is done here.

---

## Findings

Scores are **macro F1** on a held-out test set (higher is better). Models are trained **separately per artifact**. The same embedding + classifier recipe is evaluated under **two data protocols** — do not average A and B into one "our F1".

| | Pipeline A | Pipeline B |
|---|---|---|
| **Trained on** | 31 SATD sources merged, then T5 paraphrases so all five labels (incl. Not-SATD) have similar counts | The paper's four AugGPT files; Not-SATD stays the majority (~65–82%) |
| **Interpretation** | Multi-source rebuild — source mix and balance are **not** the paper's distribution | Closest to how the baseline numbers were produced |

XGBoost beat logistic regression in every recorded pair. Tables below use the **XGBoost** head. Commit baselines marked \* are the **supervisor replication** (BiLSTM 0.91, BERT 0.9804), not necessarily the published paper row.

### Identification (SATD vs not)

On Pipeline A, frozen embeddings **never** beat BiLSTM (worst drop: commits, −0.16). On Pipeline B the same stack is **close or better** on comments and pull requests; issues and commits stay slightly below BiLSTM.

| Artifact | BiLSTM baseline | A — best of Qwen3/E5 | B — Qwen3 | B — E5 | B — BGE-M3 |
|---|---:|---:|---:|---:|---:|
| Code comments | 0.939 | 0.915 (Qwen3) | **0.966** | 0.959 | 0.959 |
| Issues | 0.878 | 0.798 (Qwen3) | 0.866 | 0.863 | 0.853 |
| Pull requests | 0.862 | 0.792 (Qwen3) | 0.864 | **0.870** | 0.858 |
| Commits | 0.910\* | 0.751 (E5) | 0.892 | 0.882 | 0.901 |

**Reading:** "can we detect SATD at all?" tracks the **data protocol** more than the encoder choice. Equalizing Not-SATD with T5 (A) hurts identification; keeping the paper's Not-SATD majority (B) largely recovers it.

### Categorization (SATD type)

This result is **stable and practically important**. Frozen embeddings + XGBoost beat BERT on comments, issues, and pull requests in **both** pipelines. Commits are the exception: BERT stays ahead.

| Artifact | BERT baseline | A — best of Qwen3/E5 | B — Qwen3 | B — E5 | B — BGE-M3 |
|---|---:|---:|---:|---:|---:|
| Code comments | 0.882 | **0.904** (E5) | 0.947 | 0.958 | 0.956 |
| Issues | 0.899 | **0.941** (Qwen3) | 0.944 | 0.950 | **0.956** |
| Pull requests | 0.876 | **0.936** (E5) | 0.955 | 0.934 | **0.964** |
| Commits | 0.980\* | 0.959 (Qwen3) | 0.964 | 0.949 | 0.960 |

Pipeline B margins over BERT are larger than Pipeline A (e.g. pull requests: BGE-M3 **0.964** vs BERT **0.876**). No tested encoder reaches the 0.980 commit BERT number.

### What to take away

1. **Typing SATD (C/D vs REQ vs TES vs DOC) is where frozen embeddings shine** — cheaper than fine-tuning BERT, stronger on three of four artifacts in both pipelines.
2. **Finding SATD vs ordinary text is distribution-sensitive.** Pipeline B identification with XGBoost clusters tightly (Qwen3 0.897, E5 0.894, BGE-M3 0.893 mean across artifacts).
3. **XGBoost vs logistic regression matters more than Qwen3 vs E5 vs BGE-M3** on the same data. BGE-M3 + logistic regression (~0.85 combined) trails BGE-M3 + XGBoost (**0.926** combined on Pipeline B) by a wide margin.
4. **Headline averages (only inside one pipeline):** A Qwen3+XGB **0.8648**; B XGBoost combined — BGE-M3 **0.9261**, Qwen3 **~0.9248**, E5 **0.9206**.

---

## Large-scale inference on 458k new issues (V2 — executed)

> The controlled experiments answer "how accurate are we?". This section answers "what happens when we actually run the best issue model over nearly half a million real unseen issues?" — **this run is complete, and the output is published.**

**Notebook:** [`musaddiq_rafi/satd-classifications-optimized V2.ipynb`](musaddiq_rafi/satd-classifications-optimized%20V2.ipynb) — optimized rewrite of the original single-GPU inference notebook, executed end-to-end on Kaggle (2× Tesla T4, free tier).

**What was classified:** `issue_202608272234.parquet` — **458,232 rows**, columns `ID, Title, Description`.

| Input profile (from the notebook's inspection cell) | Value |
|---|---|
| Rows | **458,232** |
| Missing descriptions | 29,128 (~6.4%; treated as empty string) |
| Mean title length | ~58 chars (~14 tokens) |
| Mean description length | ~929 chars, std ~4,701, max ~1.39M chars (heavy tail) |
| Descriptions exceeding the 256-token cap (~1,024 chars) | **16.3% truncated** |

**Models applied (Pipeline A issue heads):**

| Role | File (in `04_trained_models/`) | Pipeline A issue macro F1 |
|---|---|---|
| Identification | `identification_issue_qwen3_xgboost.joblib` | 0.798 |
| Categorization | `categorization_issue_qwen3_xgboost.joblib` | 0.941 |

Two-stage logic per text: identification first; rows predicted SATD are passed to the 4-class categorizer. Encoder: `Qwen/Qwen3-Embedding-0.6B`, frozen, `normalize_embeddings=True`, FP16, `max_seq_length=256`. Pooling/prompt handling kept identical to training so predictions stay comparable.

**Why V2 is fast enough for Kaggle free tier:**

| Optimization | Effect |
|---|---|
| One worker process per GPU (interleaved shards `rows[shard::2]`) | ~2× throughput, load-balanced |
| FP16 weights | ~1.5–2× faster on T4 tensor cores |
| 256-token cap | Removes the long-description attention tail |
| Length-sorted batching (`batch_size=128`) | Cuts padding waste — the biggest win on mixed lengths |
| Per-batch classify-and-discard | Never holds 3 × 458k × 1024 floats in RAM |
| Per-shard Parquet checkpoints + log streaming | Crash/timeout no longer loses the run |
| 4,000-row benchmark cell with runtime projection | Validates settings before the full run |

**Output (published):** [`satd_classification_results.parquet`](https://drive.google.com/file/d/1BTzQmgwanxi5YdCh4zQUGOf1XT5CR5g-/view?usp=drive_link) — **458,232 / 458,232 rows classified (100% coverage)**, merged from 2 shards with `row_pos`/`ID` alignment checks.

| Column | Content |
|---|---|
| `ID`, `Title`, `Description` | Echoed from input (NaN descriptions → `""`) |
| `Title_SATD` | Prediction from title alone |
| `Description_SATD` | Prediction from description alone |
| `Title_Description_SATD` | Prediction from `Title + " " + Description` |

**Observed label distribution in the executed notebook:**

| View | `0` (Not-SATD) | `1` (SATD) | SATD rate |
|---|---|---:|---:|
| Title only | 283,019 | 175,213 | **38.2%** |
| Description only | 188,418 | 269,814 | **58.9%** |
| Title + Description | 205,454 | 252,778 | **55.2%** |

> **Encoding note (important for reuse):** the executed notebook's summary cells show binary `0/1` values rather than the `Not-SATD / C/D / REQ / TES / DOC` strings described in the task spec. Treat `1` as SATD and `0` as Not-SATD for this artifact; the per-type breakdown (C/D, REQ, TES, DOC) is not present in the saved summary. Re-mapping or re-running the categorizer head with string labels is listed under [Limitations](#limitations). The notebook's own `SATD total = (col != "Not-SATD").sum()` line therefore overcounts — use the table above instead.

**What the distribution suggests:** descriptions carry substantially more debt signal than titles alone (+20.7 pp), while title+description sits between the two — consistent with issues where the title is a neutral feature request and the debt admission lives in the body. Full discussion, charts, and reproduction steps are in [docs/satd_new_issue_inference_v2.md](docs/satd_new_issue_inference_v2.md) and the [PDF report](musaddiq_rafi/SATD_New_Issue_Inference_V2_Report.pdf).

---

## Experimental design

| | Identification | Categorization |
|---|---|---|
| **Task** | SATD vs Not-SATD | C/D, REQ, TES, DOC (SATD rows only) |
| **Baseline** | GloVe + BiLSTM + AugGPT | Fine-tuned BERT + AugGPT |
| **This work** | Frozen embeddings + XGBoost / logistic regression | Same |

Models are trained **separately per artifact**. Primary metric: **macro F1** on a held-out test set. Encoders are frozen (mean-pooled, L2-normalized).

**Labels:** Not-SATD, C/D (code or design), REQ, TES, DOC. Defect, Architecture, and Build labels are excluded, matching the reference study.

---

## Pipelines

| Pipeline | Data | Status |
|---|---|---|
| **[A — from-scratch](docs/pipeline_a_from_scratch.md)** | Merge of 31 sources, T5 paraphrasing to equalize five classes, split after augmentation | Phases 1–6 complete; issue heads deployed to 458k inference |
| **[B — paper data](docs/pipeline_b_paper_data.md)** | Four AugGPT CSVs from the paper; Not-SATD majority kept; split before extra augmentation | Qwen3, E5, and BGE-M3 complete |
| **[New-issue inference V2](docs/satd_new_issue_inference_v2.md)** | 458k unseen issues through Pipeline A Qwen3+XGBoost issue heads (3 text views) | **Executed — output published (Parquet + PDF report)** |

Pipeline A includes the paper's own `data-augmentation-*` files among the 31 sources. Splitting after paraphrasing is a leakage risk (synthetic siblings across splits) — see Limitations.

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
└── 04_trained_models/     classifier .joblib files (32 files)
```

**Large-scale inference files (separate Drive links):**

| File | Rows | Link |
|---|---|---|
| `issue_202608272234.parquet` (input: `ID, Title, Description`) | 458,232 | [Download](https://drive.google.com/file/d/1e9ZiR2VOglBilEEnJIfSWA_UkNudjEym/view?usp=sharing) |
| `satd_classification_results.parquet` (output: + `Title_SATD, Description_SATD, Title_Description_SATD`) | 458,232 | [Download](https://drive.google.com/file/d/1BTzQmgwanxi5YdCh4zQUGOf1XT5CR5g-/view?usp=drive_link) |

---

## Repository layout

```
satd-embedding-comparison/
├── notebooks/full_pipeline.ipynb              Pipeline A, phases 1–6
├── navid-experiment/                          Pipeline B notebook and reports
├── musaddiq_rafi/
│   ├── satd-classifications-optimized V2.ipynb  458k-issue inference (executed on 2×T4)
│   ├── SATD_New_Issue_Inference_V2_Report.pdf   458k inference report (PDF)
│   ├── README.md                                task brief + run log for the inference
│   └── models/                                  32 trained .joblib heads (mirrored)
├── scripts/                                   standalone scripts for each phase
├── data/keywords/                             KeyBERT SATD keyword lists (future Phase 8)
├── results/                                   Pipeline A phase 5/6 CSVs and summary PDF
└── docs/
    ├── SATD_Research_Progress.md              problem, RQs, chronology, both pipelines
    ├── satd_new_issue_inference_v2.md         458k inference report (Markdown)
    ├── pipeline_a_from_scratch.md
    ├── pipeline_b_paper_data.md
    └── thesis_research_narrative.md
```

---

## Setup

```bash
git clone https://github.com/A-Piyas-04/SATD-embedding-DP2.git
cd SATD-embedding-DP2
pip install -r requirements.txt
```

Place Drive downloads on the paths expected by the notebooks. GPU (CUDA) is required for embedding generation and Pipeline A paraphrasing; classifier training runs on CPU. The 458k inference requires a **2×T4 Kaggle session with internet on** (first run downloads the encoder from Hugging Face).

| Dependency | Version |
|---|---|
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| scikit-learn | ≥ 1.3 |
| xgboost | ≥ 2.0 |
| transformers / sentence-transformers | ≥ 4.40 |
| torch | ≥ 2.0 |
| joblib / pyarrow | ≥ 1.3 / ≥ 14 (Parquet I/O) |

---

## Reproduce the 458k inference

1. Kaggle → Settings → Accelerator **GPU T4 x2**, Internet **on**.
2. Add two datasets: `issue_202608272234.parquet` and the `04_trained_models/` folder.
3. Open `musaddiq_rafi/satd-classifications-optimized V2.ipynb`, set `INPUT_PATH` / `MODEL_DIR`.
4. Run top-to-bottom: inspect → pre-download encoder → benchmark (4k rows, ~minutes) → full run → merge shards → summary.
5. Output lands at `/kaggle/working/satd_classification_results.parquet`. The published copy is linked above.

```python
# expected output schema
# ID, Title, Description, Title_SATD, Description_SATD, Title_Description_SATD
import pandas as pd
df = pd.read_parquet("satd_classification_results.parquet")
print(df[["Title_SATD","Description_SATD","Title_Description_SATD"]].apply(pd.Series.value_counts))
```

---

## Documentation

| Document | Purpose |
|---|---|
| [SATD_Research_Progress.md](docs/SATD_Research_Progress.md) | Problem, RQs, chronology, both pipelines, limitations |
| [satd_new_issue_inference_v2.md](docs/satd_new_issue_inference_v2.md) | 458k inference: method, optimizations, results, reuse |
| [SATD_New_Issue_Inference_V2_Report.pdf](musaddiq_rafi/SATD_New_Issue_Inference_V2_Report.pdf) | Same inference report as PDF |
| [pipeline_a_from_scratch.md](docs/pipeline_a_from_scratch.md) | Pipeline A methods and phases |
| [pipeline_b_paper_data.md](docs/pipeline_b_paper_data.md) | Pipeline B methods |
| [thesis_research_narrative.md](docs/thesis_research_narrative.md) | Longer narrative reconstructed from the repo |
| [navid-experiment/satd_report.md](navid-experiment/satd_report.md) | Pipeline B Qwen3 / E5 report |
| [navid-experiment/satd_report_bge.md](navid-experiment/satd_report_bge.md) | Pipeline B BGE-M3 report |
| [musaddiq_rafi/README.md](musaddiq_rafi/README.md) | Inference task brief and run log |

---

## Limitations

- **Split-after-paraphrase leakage (A):** synthetic siblings can straddle train/val/test; identification numbers in A should be read with this caveat.
- **AugGPT files inside the 31-source merge:** Pipeline A is a multi-source rebuild, not a raw-only corpus.
- **16.3% of descriptions truncated** at 256 tokens in the 458k run (512 tokens is more faithful, ~2× slower).
- **Output encoding:** the executed run's summary shows `0/1`, not per-type `C/D / REQ / TES / DOC` strings — verify the saved Parquet dtypes before citing type breakdowns.
- **No ground truth for the 458k issues:** the inference output is a model-labeled resource for triage/analysis, not an accuracy benchmark.
- **Single encoder deployed:** only Pipeline A Qwen3+XGBoost issue heads were run at scale; E5/BGE and Pipeline B heads remain un-deployed.

Full limitation discussion: [docs/SATD_Research_Progress.md](docs/SATD_Research_Progress.md).

---

## Planned work

- [x] Controlled A/B experiments (Qwen3, E5, BGE-M3 + XGBoost/LR)
- [x] Large-scale inference over 458k unseen issues (V2, 2×T4) + published Parquet + PDF report
- [ ] Reconcile `0/1` vs string-label encoding; re-emit per-type (C/D, REQ, TES, DOC) columns if needed
- [ ] Deploy E5 / BGE-M3 and Pipeline B heads over the same 458k issues for a deployment-side comparison
- [ ] Keyword-feature phases (7–9: code-specific encoders, keyword signals, combined retraining) — lists already in `data/keywords/`
- [ ] Broader thesis directions: SATD in Agile practice; links between SATD, developer emotion, and issue resolution time

---

## Citation

```bibtex
@misc{satd-embedding-dp2-2026,
  title  = {SATD Embedding Comparison: Frozen Embeddings vs BiLSTM/BERT with Application to 458k Unseen Issues},
  author = {SATD-embedding-DP2 contributors},
  year   = {2026},
  note   = {Qwen3-Embedding-0.6B + XGBoost (Pipeline A) deployed over 458,232 issues; three text views; Parquet + report published}
}
@misc{sutoyo2024deep,
  title  = {Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt},
  author = {Sutoyo, E. and Avgeriou, P. and Capiluppi, A.},
  year   = {2024},
  eprint = {2410.15804},
  archivePrefix = {arXiv}
}
```
