# Pipeline A — from-scratch (31 sources)

Technical overview of the first experiment. Sibling: [Pipeline B — paper data](pipeline_b_paper_data.md). Full research narrative: [thesis_research_narrative.md](thesis_research_narrative.md).

| | |
|---|---|
| **Code** | [`notebooks/full_pipeline.ipynb`](../notebooks/full_pipeline.ipynb) |
| **Scores** | [`results/phase5_results_summary.csv`](../results/phase5_results_summary.csv), [`results/phase6_identification_comparison.csv`](../results/phase6_identification_comparison.csv), [`results/phase6_categorization_comparison.csv`](../results/phase6_categorization_comparison.csv) |
| **Large files** | [Google Drive](https://drive.google.com/drive/folders/1E-jzrNGE2NyKEsrI8Ud9Phk3gx_8a2dD) |
| **Status** | Phases 1–6 done. Phases 7–9 planned (see below). |

---

## Goal

Test whether frozen embeddings (**Qwen3-Embedding-0.6B**, **E5-Large-v2**) plus XGBoost / logistic regression can identify and categorize Self-Admitted Technical Debt (SATD) better than Sutoyo et al. (2024): **GloVe+BiLSTM+AugGPT** (identification) and **BERT+AugGPT** (categorization).

This pipeline **rebuilds the corpus** from 31 source files, **equalizes class counts with T5 paraphrases**, then embeds and classifies. It is **not** the same experiment as Pipeline B (paper AugGPT files only).

**Baseline paper:** Sutoyo, Avgeriou, and Capiluppi (2024), [arXiv:2410.15804](https://arxiv.org/abs/2410.15804). Commit-message baselines in the CSVs are the **supervisor replication** (BiLSTM 0.91, BERT 0.9804).

---

## Shared setup (this pipeline)

**Artifacts:** code comments, issues, commit messages, pull requests (separate models each).

**Tasks**

1. Identification — SATD vs Not-SATD
2. Categorization — C/D, REQ, TES, DOC on SATD rows only

**Labels (Option A)**

| Label | Meaning |
|---|---|
| Not-SATD | Not technical debt |
| C/D | Code or design (merged CODE / DESIGN / CODE/DESIGN) |
| REQ | Requirement debt |
| TES | Test debt |
| DOC | Documentation debt |

Dropped: Defect, Architecture, Build (not used in the reference study).

**Metric:** macro F1 on the held-out test set.

**Models trained:** 2 embeddings × 2 classifiers × 4 artifacts × 2 tasks = **32** (not 16).

---

## Phase 1 — Clean

**Done.** Notebook input: Kaggle `shahriarpias/satd-sources-cleaned`. Combined dump also seen as `satd_combined.csv` (**820,885** rows before mapping).

1. Load 31 CSVs; tag `artifact_type` from the filename.
2. Standardize text/label column names (`satd` / `commenttext` / `text`; `classification` / `annotation` / `label` / …).
3. Map labels to Option A; drop Defect / Architecture / Build / unmapped.
4. Drop blank or ≤2-word text.
5. Drop duplicates on `(text, artifact_type)`.
6. Write `satd_with_artifact_type.csv`.

**Caveat:** the 31 files include the paper’s own `data-augmentation-*` CSVs. This is a multi-source merge, not a purely unaugmented raw corpus.

### Source files

**Code comments (21):** `apache_ant_cleaned.csv`, `apache_jmeter_cleaned.csv`, `argouml_cleaned.csv`, `jfreechart_cleaned.csv`, `jruby_cleaned.csv`, `cppsatd_cleaned_cleaned.csv`, `mlsatd_cleaned.csv`, `manual_annotations_cleaned_cleaned.csv`, `labeled_dataset_cleaned.csv`, `OBrien_789_v2_cleaned.csv`, `satd-dataset-code_comments_cleaned.csv`, `data-augmentation-code_comments_cleaned.csv`, `maldonado_corrected_cleaned.csv`, `duplicate_satd_comment_cleaned.csv`, `unique_satd_comment_cleaned.csv`, `SSATD_COMMENTS_cleaned.csv`, `satd-comments-manual-subclass_cleaned.csv`, `ownership argouml_cleaned.csv`, `ownership_ant_cleaned.csv`, `ownership_jmeter_cleaned.csv`, `ownership_jruby_cleaned.csv`.

**Issues (4):** `issue-satd_cleaned.csv`, `satd-dataset-issues_cleaned.csv`, `data-augmentation-issues_cleaned.csv`, `SSATD_ISSUES_cleaned.csv`.

**Commits (3):** `satd-dataset-commit_messages_cleaned.csv`, `data-augmentation-commit-messages_cleaned.csv`, `SSATD_COMMITS_cleaned.csv`.

**Pull requests (3):** `satd-dataset-pull_requests_cleaned.csv`, `data-augmentation-pull-requests_cleaned.csv`, `SSATD_PULL_cleaned.csv`.

### Rows after cleaning (~386,646)

| Artifact | Rows | C/D | REQ | TES | DOC | Not-SATD |
|---|---|---|---|---|---|---|
| code_comment | 353,009 | 11,600 | 5,406 | 1,786 | 1,410 | 332,753 |
| issue | 23,128 | 2,189 | 814 | 965 | 1,081 | 18,079 |
| commit | 5,616 | 487 | 390 | 372 | 392 | 3,975 |
| pull_request | 4,893 | 502 | 155 | 247 | 380 | 3,609 |

---

## Phase 2 — Balance

**Done.** Without this step a model can always predict Not-SATD (comments: 332,753 vs 1,410 DOC).

Per artifact type:

- Target size = size of the **largest SATD class**.
- **Downsample** Not-SATD to the target.
- **Paraphrase up** smaller SATD classes with T5 `humarin/chatgpt_paraphraser_on_T5_base` (Kaggle GPU).
- Pull-request REQ (155 rows): cap at about **3×** (~465) to limit repetitive synthetic text.

Qwen3 is **not** used here; it is reserved for embeddings (Phase 4).

This **equalizes all five classes**, including Not-SATD. Sutoyo keep a Not-SATD majority. Output: `satd_balanced.csv`.

---

## Phase 3 — Split

**Done.** Stratified **80 / 10 / 10** train / val / test, seed 42, **after** augmentation (same ratio as the paper). Drive shapes: **59,105 / 7,390 / 7,395**.

Outputs: `satd_train.csv`, `satd_val.csv`, `satd_test.csv` plus aligned label CSVs.

**Risk:** paraphrases of the same original can appear in more than one split.

---

## Phase 4 — Embeddings

**Done.** Kaggle GPU T4 × 2. Set `HF_TOKEN` as a Kaggle secret.

| Model | Hugging Face id | Dim |
|---|---|---|
| Qwen3 | `Qwen/Qwen3-Embedding-0.6B` | 1024 |
| E5 | `intfloat/e5-large-v2` | 1024 |

Procedure: batch 32–64; E5 `query:` / `passage:` prefix; mean-pool; L2-normalize; save `.npy` **row-aligned** with the CSVs. Embed all text in one pass; filter by `artifact_type` at train time.

Outputs: `qwen3_{train,val,test}.npy`, `e5_{train,val,test}.npy`.

---

## Phase 5 — Classifiers

**Done.** CPU is enough.

| Setting | Value (as recorded) |
|---|---|
| XGBoost | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, multi-class `multi:softprob` |
| Logistic regression | `max_iter=1000` |
| Scope | One model per (task, artifact, embedding, classifier) |

Save `{task}_{artifact}_{embedding}_{classifier}.joblib` (32 files on Drive).

---

## Phase 6 — Compare

**Done.** Test-set macro F1 vs Sutoyo / supervisor.

**Identification:** lost all four artifacts vs BiLSTM (best: comments Qwen3+XGB **0.915** vs **0.939**; worst: commits E5+XGB **0.751** vs **0.91**).

**Categorization:** XGBoost beat BERT on comments (**0.904** vs 0.882), issues (**0.941** vs 0.899), PRs (**0.936** vs 0.876); lost commits (**0.959** vs 0.980).

**Classifier:** XGBoost beat LogReg in **32/32**. README overall: Qwen3+XGB average macro F1 **0.8648**.

Full tables: [`results/`](../results/) and [thesis narrative §7](thesis_research_narrative.md).

---

## How this differs from Pipeline B

| | Pipeline A | Pipeline B |
|---|---|---|
| Input | 31 files | Four paper AugGPT CSVs |
| Class balance | Five classes equal (T5 + downsample) | Not-SATD majority 65–82% |
| Split | After augmentation | Before extra augmentation |
| Identification vs BiLSTM | Lost all 4 | Won comments and PRs |
| Extra encoder | Qwen3, E5 | Qwen3, E5, then BGE-M3 |

---

## Future phases (not implemented)

Keyword lists for Phase 8 already live in [`data/keywords/`](../data/keywords/). These steps can reuse Pipeline A splits without re-running 1–6.

### Phase 7 — Code-specific embeddings

General-purpose Qwen3/E5 may under-read SATD in comments and commits. Planned drop-ins (same Phase 4 pooling, all four artifacts):

- CodeBERT — `microsoft/codebert-base`
- GraphCodeBERT — `microsoft/graphcodebert-base`
- UniXcoder — `microsoft/unixcoder-base` (`trust_remote_code=True` if needed)
- CodeT5 — `Salesforce/codet5-base` via `T5EncoderModel`, not `AutoModel`

Add scores to the Phase 6 tables.

### Phase 8 — Keyword signal features

Cheap SATD cues (`todo`, `hack`, `flaky`, …) may be underweighted by embeddings.

1. Parse eight KeyBERT lists; keep top 100 per file; drop punctuation-only / empty tuples.
2. Normalize scores to 0–1 within each file.
3. For each text, eight weighted match scores (word-boundary regex) plus `kw_total_signal`.
4. Save `keyword_features_{train,val,test}.csv` / `.npy` aligned to Phase 3.

Lists: comments, issues, commits, PRs, C/D, DOC, TES, REQ (see `data/keywords/` folder names).

### Phase 9 — Retrain with concatenated features

`[embedding | 9 keyword scores]` → shape `(n, 1033)`. Retrain the same 32 models. Compare macro F1 to Phase 5/6 to isolate the keyword contribution.
