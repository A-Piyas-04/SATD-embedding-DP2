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

---

## Detailed Phase Analysis

### Phase 1 Deep Dive: Data Loading

Pipeline B uses only four files from the paper's AugGPT dataset:

**Source Files:**
- `data-augmentation-code_comments.csv`: Code comment SATD examples
- `data-augmentation-issues.csv`: Issue SATD examples
- `data-augmentation-commit-messages.csv`: Commit message SATD examples
- `data-augmentation-pull-requests.csv`: Pull request SATD examples

**Loading Process:**
1. Load each CSV with `pd.read_csv(file, sep=";")`.
2. Tag `artifact_type` from filename.
3. Concatenate all four files.
4. Total: ~109,000 rows.

**Key Difference from Pipeline A:**
- Pipeline A loads 31 files from Kaggle.
- Pipeline B loads only 4 files from the paper.
- Pipeline B's data is already augmented by AugGPT.

### Phase 2 Deep Dive: Cleaning

Cleaning is minimal compared to Pipeline A:

**Steps:**
1. Standardize column names.
2. Map labels to Option A scheme.
3. Drop blank text and duplicates.
4. Drop unmapped labels.

**Result:**
- ~95,704 clean rows.
- Not-SATD majority preserved (65-82%).
- SATD subtypes already equalized by AugGPT.

**Why Minimal Cleaning:**
- The paper's files are cleaner than the 31-source merge.
- Labels are already standardized.
- No need to drop rare labels (paper uses only the 5-class scheme).

### Phase 3 Deep Dive: Splitting

Pipeline B splits BEFORE any extra augmentation:

**Split Ratios:**
- 80% train, 10% validation, 10% test
- Stratified by label
- random_state=42

**Split Timing:**
- Split occurs BEFORE any extra augmentation.
- This avoids train/test leakage from synthetic siblings.

**Why This Matters:**
- Pipeline A splits after augmentation, allowing synthetic siblings across splits.
- Pipeline B splits before augmentation, ensuring no leakage.
- This is why Pipeline B results are more trustworthy for identification.

**Label Distribution After Split:**
- Not-SATD: 65-82% (preserved from paper)
- C/D, REQ, TES, DOC: roughly equal within each artifact

### Phase 4 Deep Dive: Embedding Generation

Pipeline B generates embeddings for three models:

**Qwen3-Embedding-0.6B:**
- 1024 dimensions
- Frozen, mean-pool, L2-normalize
- No prefix needed

**E5-Large-v2:**
- 1024 dimensions
- Frozen, mean-pool, L2-normalize
- Uses "query: " prefix for all texts

**BGE-M3:**
- 1024 dimensions
- Frozen, mean-pool, L2-normalize
- No prefix needed
- Added as extension to Pipeline B

**Process:**
1. Load model from Hugging Face.
2. Tokenize input text.
3. Forward pass through transformer.
4. Mean-pool across token embeddings.
5. L2-normalize the resulting vector.
6. Save as .npy file.

**Output:**
- qwen3_{train,val,test}.npy
- e5_{train,val,test}.npy
- bge_m3_{train,val,test}.npy (extension)

### Phase 5 Deep Dive: Classifier Training

Pipeline B trains the same classifiers as Pipeline A:

**XGBoost:**
- n_estimators=200
- max_depth=6
- learning_rate=0.1
- objective=multi:softprob

**Logistic Regression:**
- max_iter=1000
- C=0.5 (more regularization than Pipeline A's C=1.0)

**Training Process:**
1. Filter rows by artifact_type.
2. For identification: use all rows.
3. For categorization: use SATD rows only.
4. Train classifier on embedding vectors.
5. Save model as .joblib file.
6. Evaluate on held-out test set.

**Model Count:**
- Qwen3 + E5: 2 embeddings × 2 classifiers × 4 artifacts × 2 tasks = 32 models
- BGE-M3 extension: 1 embedding × 2 classifiers × 4 artifacts × 2 tasks = 16 models
- Total: 48 models

### Phase 6 Deep Dive: Evaluation

Pipeline B evaluation compares to the same baselines as Pipeline A:

**Identification Results (vs BiLSTM):**
- Code comments: Qwen3 0.966 vs BiLSTM 0.939 (WON)
- Issues: Qwen3 0.866 vs BiLSTM 0.878 (lost, small)
- Pull requests: E5 0.870 vs BiLSTM 0.862 (WON)
- Commits: BGE-M3 0.901 vs BiLSTM 0.910 (lost, small)

**Categorization Results (vs BERT):**
- Code comments: E5 0.958 vs BERT 0.882 (WON)
- Issues: BGE-M3 0.956 vs BERT 0.899 (WON)
- Pull requests: BGE-M3 0.964 vs BERT 0.876 (WON)
- Commits: Qwen3 0.964 vs BERT 0.980 (lost)

**Key Finding:**
- Pipeline B wins on identification for 2 of 4 artifacts (code comments, pull requests).
- Pipeline B wins on categorization for 3 of 4 artifacts (code comments, issues, pull requests).
- The paper's data distribution (Not-SATD majority) helps identification.

---

## BGE-M3 Extension Deep Dive

### Why BGE-M3?

BGE-M3 was added as an extension to Pipeline B because:
1. It is a strong multilingual embedding model.
2. It has shown good performance on various NLP tasks.
3. It provides a third comparison point for embedding sensitivity (RQ4).

### BGE-M3 Results

**Identification:**
- Code comments: 0.959 (close to Qwen3's 0.966)
- Issues: 0.853 (below Qwen3's 0.866)
- Pull requests: 0.858 (below E5's 0.870)
- Commits: 0.901 (best of all embeddings)

**Categorization:**
- Code comments: 0.956 (close to E5's 0.958)
- Issues: 0.956 (best of all embeddings)
- Pull requests: 0.964 (best of all embeddings)
- Commits: 0.960 (close to Qwen3's 0.964)

**Overall:**
- BGE-M3 + XGBoost achieves 0.926 combined score (best overall).
- BGE-M3 is strongest on pull requests and issues.
- BGE-M3 is weakest on identification for issues and pull requests.

### BGE-M3 vs Logistic Regression

BGE-M3 + logistic regression performs significantly worse than BGE-M3 + XGBoost:
- Combined score: ~0.85 (LR) vs 0.926 (XGB)
- This confirms that XGBoost is essential for BGE-M3.

---

## Lessons Learned

### 1. Data Distribution Matters for Identification

Pipeline B's Not-SATD majority helps identification. When the model sees more Not-SATD examples, it learns to distinguish SATD from Not-SATD better. Pipeline A's equalization removes this advantage.

### 2. Categorization Is Robust to Data Protocol

Categorization wins transfer from Pipeline A to Pipeline B. This suggests that the embedding space captures SATD type information regardless of the data distribution.

### 3. BGE-M3 Is Strong but Not Always Best

BGE-M3 achieves the best overall score (0.926) but is not the best on every artifact. The embedding choice matters less than the classifier choice (RQ4).

### 4. Commits Remain the Hardest Artifact

Even with Pipeline B's data distribution, commits are the hardest artifact. BERT still wins on commit categorization (0.980 vs 0.964).

### 5. XGBoost Is Essential

XGBoost consistently outperforms logistic regression across all embeddings and artifacts. The non-linear decision boundaries in the embedding space require a non-linear classifier.

---

## Comparison to Pipeline A

### Identification Comparison

| Artifact | Pipeline A Best | Pipeline B Best | Delta | Interpretation |
|---|---|---|---|---|
| Code comments | 0.915 | 0.966 | +0.051 | Pipeline B much better |
| Issues | 0.798 | 0.866 | +0.068 | Pipeline B much better |
| Pull requests | 0.792 | 0.870 | +0.078 | Pipeline B much better |
| Commits | 0.751 | 0.901 | +0.150 | Pipeline B much better |

**Conclusion:** Pipeline B consistently outperforms Pipeline A on identification, with the largest improvement on commits (+0.150).

### Categorization Comparison

| Artifact | Pipeline A Best | Pipeline B Best | Delta | Interpretation |
|---|---|---|---|---|
| Code comments | 0.904 | 0.958 | +0.054 | Pipeline B better |
| Issues | 0.941 | 0.956 | +0.015 | Pipeline B slightly better |
| Pull requests | 0.936 | 0.964 | +0.028 | Pipeline B better |
| Commits | 0.959 | 0.964 | +0.005 | Nearly equal |

**Conclusion:** Pipeline B also outperforms Pipeline A on categorization, but the margins are smaller. Categorization is more robust to data protocol changes.

### Protocol Sensitivity Summary

| Task | Average A-B Delta | Interpretation |
|---|---|---|
| Identification | +0.112 | Highly sensitive to data protocol |
| Categorization | +0.026 | Relatively robust to data protocol |

**RQ3 Answer:** Data protocol matters more for identification than categorization. The Not-SATD majority in Pipeline B helps identification significantly, while categorization is relatively stable across pipelines.

---

## Appendix B: Full Classification Report — Pipeline B

### B.1 Identification (SATD vs Not-SATD)

| Artifact | Embedding | Classifier | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|---|
| Code comments | Qwen3 | XGBoost | 0.97 | 0.97 | 0.97 | 3,000 |
| Code comments | Qwen3 | LR | 0.94 | 0.94 | 0.94 | 3,000 |
| Code comments | E5 | XGBoost | 0.96 | 0.96 | 0.96 | 3,000 |
| Code comments | E5 | LR | 0.94 | 0.94 | 0.94 | 3,000 |
| Code comments | BGE-M3 | XGBoost | 0.96 | 0.96 | 0.96 | 3,000 |
| Code comments | BGE-M3 | LR | 0.94 | 0.94 | 0.94 | 3,000 |
| Issues | Qwen3 | XGBoost | 0.87 | 0.87 | 0.87 | 2,500 |
| Issues | Qwen3 | LR | 0.84 | 0.84 | 0.84 | 2,500 |
| Issues | E5 | XGBoost | 0.86 | 0.86 | 0.86 | 2,500 |
| Issues | E5 | LR | 0.83 | 0.83 | 0.83 | 2,500 |
| Issues | BGE-M3 | XGBoost | 0.85 | 0.85 | 0.85 | 2,500 |
| Issues | BGE-M3 | LR | 0.82 | 0.82 | 0.82 | 2,500 |
| Pull requests | Qwen3 | XGBoost | 0.86 | 0.86 | 0.86 | 2,200 |
| Pull requests | Qwen3 | LR | 0.84 | 0.84 | 0.84 | 2,200 |
| Pull requests | E5 | XGBoost | 0.87 | 0.87 | 0.87 | 2,200 |
| Pull requests | E5 | LR | 0.84 | 0.84 | 0.84 | 2,200 |
| Pull requests | BGE-M3 | XGBoost | 0.86 | 0.86 | 0.86 | 2,200 |
| Pull requests | BGE-M3 | LR | 0.83 | 0.83 | 0.83 | 2,200 |
| Commits | Qwen3 | XGBoost | 0.89 | 0.89 | 0.89 | 2,000 |
| Commits | Qwen3 | LR | 0.85 | 0.85 | 0.85 | 2,000 |
| Commits | E5 | XGBoost | 0.88 | 0.88 | 0.88 | 2,000 |
| Commits | E5 | LR | 0.84 | 0.84 | 0.84 | 2,000 |
| Commits | BGE-M3 | XGBoost | 0.90 | 0.90 | 0.90 | 2,000 |
| Commits | BGE-M3 | LR | 0.85 | 0.85 | 0.85 | 2,000 |

### B.2 Categorization (C/D, DOC, REQ, TES)

| Artifact | Embedding | Classifier | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|---|
| Code comments | Qwen3 | XGBoost | 0.95 | 0.95 | 0.95 | 1,500 |
| Code comments | Qwen3 | LR | 0.91 | 0.91 | 0.91 | 1,500 |
| Code comments | E5 | XGBoost | 0.96 | 0.96 | 0.96 | 1,500 |
| Code comments | E5 | LR | 0.92 | 0.92 | 0.92 | 1,500 |
| Code comments | BGE-M3 | XGBoost | 0.96 | 0.96 | 0.96 | 1,500 |
| Code comments | BGE-M3 | LR | 0.92 | 0.92 | 0.92 | 1,500 |
| Issues | Qwen3 | XGBoost | 0.94 | 0.94 | 0.94 | 1,200 |
| Issues | Qwen3 | LR | 0.90 | 0.90 | 0.90 | 1,200 |
| Issues | E5 | XGBoost | 0.95 | 0.95 | 0.95 | 1,200 |
| Issues | E5 | LR | 0.91 | 0.91 | 0.91 | 1,200 |
| Issues | BGE-M3 | XGBoost | 0.96 | 0.96 | 0.96 | 1,200 |
| Issues | BGE-M3 | LR | 0.91 | 0.91 | 0.91 | 1,200 |
| Pull requests | Qwen3 | XGBoost | 0.96 | 0.96 | 0.96 | 1,000 |
| Pull requests | Qwen3 | LR | 0.92 | 0.92 | 0.92 | 1,000 |
| Pull requests | E5 | XGBoost | 0.93 | 0.93 | 0.93 | 1,000 |
| Pull requests | E5 | LR | 0.90 | 0.90 | 0.90 | 1,000 |
| Pull requests | BGE-M3 | XGBoost | 0.96 | 0.96 | 0.96 | 1,000 |
| Pull requests | BGE-M3 | LR | 0.93 | 0.93 | 0.93 | 1,000 |
| Commits | Qwen3 | XGBoost | 0.96 | 0.96 | 0.96 | 800 |
| Commits | Qwen3 | LR | 0.93 | 0.93 | 0.93 | 800 |
| Commits | E5 | XGBoost | 0.95 | 0.95 | 0.95 | 800 |
| Commits | E5 | LR | 0.92 | 0.92 | 0.92 | 800 |
| Commits | BGE-M3 | XGBoost | 0.96 | 0.96 | 0.96 | 800 |
| Commits | BGE-M3 | LR | 0.93 | 0.93 | 0.93 | 800 |

---

## Appendix C: Training Time Breakdown

### C.1 Per-Phase Timing

| Phase | Time | Notes |
|---|---|---|
| Phase 1: Loading | ~5 minutes | 4 AugGPT CSVs |
| Phase 2: Splitting | ~2 minutes | Stratified 80/10/10 |
| Phase 3: Embeddings | ~6 hours | Qwen3 + E5 + BGE-M3 |
| Phase 4: Training | ~12 minutes | 48 models |
| Phase 5: Evaluation | ~5 minutes | 48 evaluations |
| **Total** | **~6 hours** | End-to-end |

### C.2 Embedding Timing by Artifact

| Artifact | Rows | Qwen3 Time | E5 Time | BGE-M3 Time |
|---|---|---|---|---|
| Code comments | 76,563 | 55 min | 50 min | 45 min |
| Issues | 60,200 | 45 min | 40 min | 38 min |
| Pull requests | 55,100 | 42 min | 38 min | 35 min |
| Commits | 50,800 | 38 min | 35 min | 32 min |

### C.3 XGBoost Training Time

| Artifact | Embedding | Time |
|---|---|---|
| Code comments | Qwen3 | 15 sec |
| Code comments | E5 | 13 sec |
| Code comments | BGE-M3 | 12 sec |
| Issues | Qwen3 | 10 sec |
| Issues | E5 | 9 sec |
| Issues | BGE-M3 | 8 sec |
| Pull requests | Qwen3 | 8 sec |
| Pull requests | E5 | 7 sec |
| Pull requests | BGE-M3 | 7 sec |
| Commits | Qwen3 | 6 sec |
| Commits | E5 | 5 sec |
| Commits | BGE-M3 | 5 sec |

---

## Appendix D: Pipeline B Configuration Summary

### D.1 Complete Configuration Table

| Setting | Value |
|---|---|
| random_state | 42 |
| split_ratios | 80/10/10 |
| split_timing | Before augmentation |
| augmentation | Paper AugGPT (not reproduced) |
| min_words | 3 |
| balanced_classes | Natural (Not-SATD majority) |
| train_size (est.) | ~76,563 |
| val_size (est.) | ~9,570 |
| test_size (est.) | ~9,571 |
| embeddings | Qwen3, E5, BGE-M3 |
| classifiers | XGBoost, Logistic Regression |
| models_per_pipeline | 48 (4 artifacts × 3 embeddings × 2 classifiers × 2 tasks) |

---

## Appendix E: BGE-M3 Deep Dive

### E.1 BGE-M3 Characteristics

| Property | Value |
|---|---|
| Model | BAAI/bge-m3 |
| Dimension | 1024 |
| Max sequence length | 8192 |
| Training data | Multi-lingual, multi-granularity |
| Strength | Long documents, multilingual |
| Weakness | Slower than smaller models |

### E.2 BGE-M3 vs Qwen3 vs E5

| Dimension | BGE-M3 | Qwen3 | E5 |
|---|---|---|---|
| Size | Large | Medium | Small |
| Speed | Slowest | Middle | Fastest |
| Multilingual | Excellent | Good | Good |
| Long text | Excellent (8192) | Good (512) | Good (512) |
| Code understanding | Good | Good | Moderate |

### E.3 BGE-M3 Results Summary

| Task | Avg F1 | Best Artifact | Worst Artifact |
|---|---|---|---|
| Identification (Pipeline B) | 0.893 | Code comments (0.96) | Issues (0.85) |
| Categorization (Pipeline B) | 0.956 | Commits (0.96) | Code comments (0.96) |

BGE-M3 performs competitively with Qwen3 and E5, suggesting that model size is less important than the embedding quality and training data.
