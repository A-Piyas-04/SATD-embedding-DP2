# Pipeline B — paper data (Sutoyo AugGPT files)

Technical overview of the second experiment. Sibling: [Pipeline A — from-scratch](pipeline_a_from_scratch.md). Full research narrative: [thesis_research_narrative.md](thesis_research_narrative.md).

| | |
|---|---|
| **Code** | [`navid-experiment/glm_pipeline_artifacts_separated.ipynb`](../navid-experiment/glm_pipeline_artifacts_separated.ipynb) |
| **Scores** | [`navid-experiment/glm_pipeline_results.txt`](../navid-experiment/glm_pipeline_results.txt), [`navid-experiment/satd_report.md`](../navid-experiment/satd_report.md) (3 Aug 2026), [`navid-experiment/satd_report_bge.md`](../navid-experiment/satd_report_bge.md) (12 Aug 2026) |
| **BGE training notebook** | Not in this repo (results only) |
| **Status** | Qwen3 vs E5 done; BGE-M3 done as an **extension of this pipeline**, not a third pipeline. |

---

## Goal

Hold the **paper’s augmented distribution** (almost) fixed and change only the **encoder + classifier**. If Pipeline A loses identification, this pipeline asks whether that is the embeddings or the 31-file / T5 rebuild.

Same Sutoyo et al. (2024) baselines: **BiLSTM+AugGPT** (identification), **BERT+AugGPT** (categorization). Commit numbers: supervisor replication (0.910 / 0.980).

---

## Shared setup (same scientific skeleton as A)

**Artifacts:** code comments, issues, commits, pull requests (separate models).

**Tasks:** identification (SATD vs Not-SATD); categorization (C/D, REQ, TES, DOC on SATD only).

**Labels:** Option A (Not-SATD, C/D, REQ, TES, DOC). Drop Defect / Architecture / Build / `without_classification`.

**Metric:** macro F1.

**Heads:** XGBoost (`n_estimators=200`, `max_depth=6`, `learning_rate=0.1` where recorded) and logistic regression (`max_iter=1000`, **C = 0.5**).

**Encoders:** mean-pool, L2-normalize. E5 uses `query:` / `passage:` prefixes.

| Run | Embeddings | Combinations |
|---|---|---|
| Qwen vs E5 | `Qwen/Qwen3-Embedding-0.6B`, `intfloat/e5-large-v2` | 32 models |
| + BGE | `BAAI/bge-m3` dense | +16 models |

---

## Data

Four **semicolon-separated** AugGPT files from the paper:

| File | Artifact |
|---|---|
| `data-augmentation-code_comments.csv` | code_comment |
| `data-augmentation-issues.csv` | issue |
| `data-augmentation-commit-messages.csv` | commit |
| `data-augmentation-pull-requests.csv` | pull_request |

About **109k** rows → **95,704** after drop NA, ≤2-word text, unmapped labels.

- Four SATD subtypes already **equalized per artifact**.
- **Not-SATD stays 65–82%** (paper-like). No T5 five-class equalization.
- Column `classification` (or `label` / `class`) mapped to Option A; `artifact_type` tagged from the filename.

This is **not** the 31-file merge. Pipeline A’s merge also *contains* these augmentation files as a subset; Pipeline B uses **only** these four.

---

## Protocol (step by step)

### 1. Load and clean

Notebook `file_artifact_map` loads the four CSVs, concatenates, maps labels, drops junk. See [`glm_pipeline_artifacts_separated.ipynb`](../navid-experiment/glm_pipeline_artifacts_separated.ipynb).

### 2. Split before extra augmentation

Stratified **80 / 10 / 10** on the cleaned paper data **before** any extra paraphrase. This avoids train/test leakage from synthetic siblings (the opposite of Pipeline A).

### 3. Optional extra paraphrase (notebook only)

The notebook includes a **Qwen2.5-1.5B-Instruct** persona prompt (“rephrase like a programmer”). Reported Qwen / E5 / BGE tables are described as using the **paper’s pre-augmented set** without Pipeline A’s T5 five-class balance. Treat extra Qwen2.5 aug as **available in the notebook**, not as the default reported protocol unless you re-run and document it.

### 4. Embed

Same recipe as Pipeline A Phase 4 for Qwen3 and E5. Later: BGE-M3 dense vectors through the same pooling/normalize path.

### 5. Train per artifact and task

Filter by `artifact_type`. Identification uses all rows; categorization uses SATD rows only (`id_label == SATD` in the notebook). Save classifiers; evaluate on test.

### 6. Compare

Hardcoded baselines in the notebook: identification 0.939 / 0.878 / 0.862 / 0.910; categorization 0.882 / 0.899 / 0.876 / 0.980 (comments, issues, PRs, commits).

---

## Results — Qwen3 vs E5 (3 August 2026)

Source: [`glm_pipeline_results.txt`](../navid-experiment/glm_pipeline_results.txt). Best head is XGBoost unless noted.

### Identification vs BiLSTM+AugGPT

| Artifact | Baseline | Qwen3 XGB | E5 XGB | Outcome (best XGB) |
|---|---|---|---|---|
| Code comments | 0.939 | **0.9664** | 0.9587 | **Won** |
| Issues | 0.878 | 0.8656 | 0.8632 | Lost (small) |
| Pull requests | 0.862 | 0.8642 | **0.8703** | **Won** |
| Commits | 0.910 | 0.8917 | 0.8817 | Lost (small) |

### Categorization vs BERT+AugGPT

| Artifact | Baseline | Qwen3 XGB | E5 XGB | Outcome |
|---|---|---|---|---|
| Code comments | 0.882 | 0.9470 | **0.9581** | **Won** |
| Issues | 0.899 | 0.9442 | **0.9503** | **Won** |
| Pull requests | 0.876 | **0.9548** | 0.9335 | **Won** |
| Commits | 0.980 | **0.9639** | 0.9487 | Lost |

Averages over artifacts and tasks: E5 **0.8988**, Qwen3 **0.8953**. Best combo: **Qwen3 + XGBoost 0.9247**. XGBoost beat LogReg for both embeddings.

---

## Extension — BGE-M3 (12 August 2026)

Same four files, same split policy, same heads. Only the encoder changes.

XGBoost identification: BGE closest on **commits** (**0.9014** vs 0.910). Still below BiLSTM on issues (0.8533) and PRs (0.8580 vs 0.862). Qwen3 still best on comments (0.9664).

XGBoost categorization: BGE best overall average **0.9592**; largest BERT gap **+0.0884** on PRs (0.9644 vs 0.876). No encoder reaches commit **0.980**.

Overall (both tasks): **BGE-M3 + XGBoost 0.9261** > Qwen3+XGB 0.9248 > E5+XGB 0.9206. BGE + LogReg is weak (~0.85 combined): use a tree head with BGE.

Full tables: [`satd_report_bge.md`](../navid-experiment/satd_report_bge.md).

---

## How this differs from Pipeline A

| | Pipeline B | Pipeline A |
|---|---|---|
| Why it exists | Isolate encoder/classifier on the paper’s data | Rebuild + rebalance from 31 files |
| Not-SATD | Majority kept | Downsampled to SATD class size |
| Identification | Competitive; wins comments and PRs | Lost all 4 |
| Categorization 3/4 win | Yes, larger margins | Yes |
| Extra encoder | BGE-M3 | — |

**Takeaway for this pipeline:** frozen embeddings + XGBoost can **match or beat BiLSTM on identification when the training distribution is the paper’s AugGPT data**. That does **not** automatically transfer to Pipeline A’s T5-equalized set. Categorization wins on comments / issues / PRs **do** transfer.

---

## Compute

Kaggle / Colab T4-class GPU for embeddings. Classifiers on CPU. Frozen encoders only (0.6B Qwen3, E5-large, BGE-M3). No BERT/BiLSTM fine-tuning.

Larger models are untested here. On this data, swapping Qwen / E5 / BGE moved identification by less than half a point (XGBoost averages). Data protocol (A vs B) moved identification much more.

---

## Planned follow-ons (same as A if you keep these splits)

Code embeddings, keyword features, and concatenated retrain are specified under [Pipeline A future phases](pipeline_a_from_scratch.md#future-phases-not-implemented). They are not implemented on Pipeline B either. If you run them, report **A and B separately**.
