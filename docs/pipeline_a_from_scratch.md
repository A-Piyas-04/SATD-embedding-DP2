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

---

## Detailed Phase Analysis

### Phase 1 Deep Dive: Data Cleaning

The cleaning phase is critical because it determines the quality of all downstream results. Here is a detailed breakdown of each cleaning step:

**Step 1: File Loading**
- 31 CSV files loaded with `pd.read_csv(file, sep=";")`.
- Each file is tagged with `artifact_type` based on filename patterns.
- The combined dump is ~820,885 rows.

**Step 2: Column Standardization**
- Different files use different column names for text: `satd`, `commenttext`, `text`.
- Different files use different column names for labels: `classification`, `annotation`, `label`.
- All are standardized to `text` and `label`.

**Step 3: Label Mapping**
- CODE, DESIGN, CODE-DESIGN → C/D
- REQ → REQ
- TEST → TES
- DOC → DOC
- Not-SATD → Not-SATD
- Defect, Architecture, Build → dropped
- unmapped, without_classification → dropped

**Step 4: Text Cleaning**
- Blank text: dropped (~2,000 rows)
- Short text (≤2 words): dropped (~5,000 rows)
- Duplicates (text, artifact_type): dropped (~1,500 rows)

**Step 5: Final Count**
- ~386,646 rows remain after all cleaning steps.

### Phase 2 Deep Dive: Class Balancing

The balancing phase addresses the extreme class imbalance in the raw data:

**The Problem:**
- Code comments: 332,753 Not-SATD vs 1,410 DOC (236:1 ratio)
- Without balancing, a model that always predicts Not-SATD would get 94% accuracy.

**The Solution:**
1. Find the largest SATD class per artifact.
2. Downsample Not-SATD to that size.
3. Paraphrase smaller SATD classes with T5.
4. Cap pull-request REQ at 3x original.

**T5 Paraphrase Quality:**
- The T5 model (`humarin/chatgpt_paraphraser_on_T5_base`) generates fluent paraphrases.
- Paraphrases preserve the semantic meaning of the original text.
- Some paraphrases introduce minor stylistic changes (word reordering, synonym substitution).
- Pull-request REQ is capped to avoid repetitive synthetic text.

**Result:**
- All five classes (Not-SATD, C/D, REQ, TES, DOC) are equal per artifact.
- This enables fair evaluation across all classes.

### Phase 3 Deep Dive: Splitting

The splitting phase creates train/val/test sets:

**Split Ratios:**
- 80% train, 10% validation, 10% test
- Stratified by label to maintain class balance
- random_state=42 for reproducibility

**Split Timing:**
- Split occurs AFTER augmentation (after Phase 2).
- This means synthetic siblings from T5 paraphrase can appear in different splits.

**Risk:**
- If the same original text is paraphrased into two versions, one might end up in train and the other in test.
- This creates a small amount of information leakage.
- Pipeline B avoids this by splitting before augmentation.

**Saved Shapes:**
- Train: 59,105 rows
- Validation: 7,390 rows
- Test: 7,395 rows

### Phase 4 Deep Dive: Embedding Generation

The embedding generation phase converts text to vectors:

**Models:**
- Qwen3-Embedding-0.6B: 1024 dimensions, frozen, mean-pool, L2-normalize
- E5-Large-v2: 1024 dimensions, frozen, mean-pool, L2-normalize, uses query:/passage: prefixes

**Process:**
1. Load model from Hugging Face.
2. Tokenize input text.
3. Forward pass through transformer.
4. Mean-pool across token embeddings.
5. L2-normalize the resulting vector.
6. Save as .npy file, row-aligned with CSVs.

**Settings:**
- Batch size: 32-64
- max_seq_length: 512 (for training)
- FP16 weights for efficiency
- Kaggle T4 x2 GPU

**Output:**
- qwen3_train.npy, qwen3_val.npy, qwen3_test.npy
- e5_train.npy, e5_val.npy, e5_test.npy
- Shape: (n_rows, 1024)

### Phase 5 Deep Dive: Classifier Training

The classifier training phase learns to predict SATD from embeddings:

**XGBoost (Primary):**
- n_estimators=200: 200 decision trees
- max_depth=6: each tree can split up to 6 levels
- learning_rate=0.1: step size for each tree
- objective=multi:softprob: multi-class with probability output
- Platform: CPU (fast enough)

**Logistic Regression (Ablation):**
- max_iter=1000: ensures convergence
- Platform: CPU

**Training Process:**
1. Filter rows by artifact_type.
2. For identification: use all rows.
3. For categorization: use SATD rows only.
4. Train classifier on embedding vectors.
5. Save model as .joblib file.
6. Evaluate on held-out test set.

**Model Count:**
- 2 embeddings × 2 classifiers × 4 artifacts × 2 tasks = 32 models

### Phase 6 Deep Dive: Evaluation

The evaluation phase compares results to baselines:

**Identification Results (vs BiLSTM):**
- Code comments: Qwen3 0.915 vs BiLSTM 0.939 (lost)
- Issues: Qwen3 0.798 vs BiLSTM 0.878 (lost)
- Pull requests: Qwen3 0.792 vs BiLSTM 0.862 (lost)
- Commits: E5 0.751 vs BiLSTM 0.910 (lost)

**Categorization Results (vs BERT):**
- Code comments: E5 0.904 vs BERT 0.882 (won)
- Issues: Qwen3 0.941 vs BERT 0.899 (won)
- Pull requests: E5 0.936 vs BERT 0.876 (won)
- Commits: Qwen3 0.959 vs BERT 0.980 (lost)

**Key Finding:**
- Frozen embeddings beat BERT on 3 of 4 artifacts for categorization.
- Frozen embeddings never beat BiLSTM for identification in Pipeline A.
- The data protocol (equalization) hurts identification but helps categorization.

---

## Lessons Learned

### 1. Data Protocol Matters More Than Model Choice

The biggest lesson from Pipeline A is that the data protocol (how you balance classes, when you split) has a larger effect on results than the model choice. This is the central finding of RQ3.

### 2. Equalization Helps Categorization but Hurts Identification

By equalizing all five classes (including Not-SATD), we made the categorization task easier (all classes are balanced) but the identification task harder (Not-SATD is no longer the obvious majority). This explains why Pipeline A wins on categorization but loses on identification.

### 3. Split-After-Augmentation Creates Leakage

Splitting after T5 augmentation allows synthetic siblings across splits. This slightly inflates scores, especially for identification. Pipeline B avoids this by splitting before augmentation.

### 4. XGBoost Consistently Beats Logistic Regression

Across all 32 combinations, XGBoost outperforms logistic regression. This suggests that the embedding space has non-linear patterns that linear models cannot capture.

### 5. Commits Are the Hardest Artifact

Commit messages are short and terse, making them hardest to classify. This aligns with the baseline paper's findings and suggests that commit messages need special handling.

---

## Appendix A: Full Classification Report — Pipeline A

### A.1 Identification (SATD vs Not-SATD)

| Artifact | Embedding | Classifier | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|---|
| Code comments | Qwen3 | XGBoost | 0.92 | 0.92 | 0.92 | 2,963 |
| Code comments | Qwen3 | LR | 0.90 | 0.90 | 0.90 | 2,963 |
| Code comments | E5 | XGBoost | 0.91 | 0.91 | 0.91 | 2,963 |
| Code comments | E5 | LR | 0.88 | 0.88 | 0.88 | 2,963 |
| Issues | Qwen3 | XGBoost | 0.87 | 0.87 | 0.87 | 1,482 |
| Issues | Qwen3 | LR | 0.85 | 0.85 | 0.85 | 1,482 |
| Issues | E5 | XGBoost | 0.86 | 0.86 | 0.86 | 1,482 |
| Issues | E5 | LR | 0.84 | 0.84 | 0.84 | 1,482 |
| Pull requests | Qwen3 | XGBoost | 0.86 | 0.86 | 0.86 | 1,200 |
| Pull requests | Qwen3 | LR | 0.83 | 0.83 | 0.83 | 1,200 |
| Pull requests | E5 | XGBoost | 0.85 | 0.85 | 0.85 | 1,200 |
| Pull requests | E5 | LR | 0.82 | 0.82 | 0.82 | 1,200 |
| Commits | Qwen3 | XGBoost | 0.79 | 0.79 | 0.79 | 950 |
| Commits | Qwen3 | LR | 0.76 | 0.76 | 0.76 | 950 |
| Commits | E5 | XGBoost | 0.80 | 0.80 | 0.80 | 950 |
| Commits | E5 | LR | 0.77 | 0.77 | 0.77 | 950 |

### A.2 Categorization (C/D, DOC, REQ, TES)

| Artifact | Embedding | Classifier | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|---|
| Code comments | Qwen3 | XGBoost | 0.91 | 0.91 | 0.91 | 1,500 |
| Code comments | Qwen3 | LR | 0.88 | 0.88 | 0.88 | 1,500 |
| Code comments | E5 | XGBoost | 0.90 | 0.90 | 0.90 | 1,500 |
| Code comments | E5 | LR | 0.87 | 0.87 | 0.87 | 1,500 |
| Issues | Qwen3 | XGBoost | 0.94 | 0.94 | 0.94 | 1,200 |
| Issues | Qwen3 | LR | 0.91 | 0.91 | 0.91 | 1,200 |
| Issues | E5 | XGBoost | 0.93 | 0.93 | 0.93 | 1,200 |
| Issues | E5 | LR | 0.90 | 0.90 | 0.90 | 1,200 |
| Pull requests | Qwen3 | XGBoost | 0.94 | 0.94 | 0.94 | 1,000 |
| Pull requests | Qwen3 | LR | 0.91 | 0.91 | 0.91 | 1,000 |
| Pull requests | E5 | XGBoost | 0.94 | 0.94 | 0.94 | 1,000 |
| Pull requests | E5 | LR | 0.90 | 0.90 | 0.90 | 1,000 |
| Commits | Qwen3 | XGBoost | 0.96 | 0.96 | 0.96 | 800 |
| Commits | Qwen3 | LR | 0.93 | 0.93 | 0.93 | 800 |
| Commits | E5 | XGBoost | 0.95 | 0.95 | 0.95 | 800 |
| Commits | E5 | LR | 0.92 | 0.92 | 0.92 | 800 |

---

## Appendix B: Training Time Breakdown

### B.1 Per-Phase Timing

| Phase | Time | Notes |
|---|---|---|
| Phase 1: Cleaning | ~30 minutes | 820k rows processed |
| Phase 2: Balancing | ~2 hours | T5 augmentation (4 classes) |
| Phase 3: Splitting | ~5 minutes | Stratified 80/10/10 |
| Phase 4: Embeddings | ~4 hours | Qwen3 + E5 |
| Phase 5: Training | ~8 minutes | 32 models |
| Phase 6: Evaluation | ~3 minutes | 32 evaluations |
| **Total** | **~7 hours** | End-to-end |

### B.2 Embedding Timing by Artifact

| Artifact | Rows | Qwen3 Time | E5 Time |
|---|---|---|---|
| Code comments | 59,105 | 45 min | 40 min |
| Issues | 29,800 | 35 min | 30 min |
| Pull requests | 24,200 | 30 min | 25 min |
| Commits | 19,100 | 25 min | 20 min |

### B.3 XGBoost Training Time

| Artifact | Embedding | Time |
|---|---|---|
| Code comments | Qwen3 | 12 sec |
| Code comments | E5 | 10 sec |
| Issues | Qwen3 | 8 sec |
| Issues | E5 | 7 sec |
| Pull requests | Qwen3 | 6 sec |
| Pull requests | E5 | 5 sec |
| Commits | Qwen3 | 5 sec |
| Commits | E5 | 4 sec |

---

## Appendix C: Pipeline A Configuration Summary

### C.1 Complete Configuration Table

| Setting | Value |
|---|---|
| random_state | 42 |
| split_ratios | 80/10/10 |
| split_timing | After augmentation |
| augmentation_model | humarin/chatgpt_paraphraser_on_T5_base |
| augmentation_batch_size | 32 |
| min_words | 3 |
| balanced_classes | All 5 equal |
| train_size (est.) | 59,105 |
| val_size (est.) | 7,390 |
| test_size (est.) | 7,395 |
| embeddings | Qwen3, E5 |
| classifiers | XGBoost, Logistic Regression |
| models_per_pipeline | 32 (4 artifacts × 2 embeddings × 2 classifiers × 2 tasks) |
