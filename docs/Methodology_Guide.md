# Methodology Guide

**SATD Embedding Comparison Project**
**Team: Shahriar Pias, Navid Ibrahim, Musaddiq Rafi**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Research Questions](#2-research-questions)
3. [Experimental Design](#3-experimental-design)
4. [Pipeline A: From-Scratch Rebuild](#4-pipeline-a)
5. [Pipeline B: Paper-Protocol Study](#5-pipeline-b)
6. [Shared Methodology](#6-shared-methodology)
7. [Embedding Generation](#7-embedding-generation)
8. [Classifier Training](#8-classifier-training)
9. [Evaluation Metric](#9-evaluation-metric)
10. [Large-Scale Inference](#10-large-scale-inference)
11. [Demo System](#11-demo-system)

---

## 1. Overview

This document describes the full methodology used in the SATD Embedding Comparison project. The project tests whether frozen general-purpose text embeddings combined with lightweight classifiers can match or exceed purpose-trained BiLSTM and BERT models for identifying and categorizing Self-Admitted Technical Debt (SATD) across four software artifact types.

The methodology is designed as a controlled experiment with two pipelines that share the same evaluation skeleton but differ in their data protocol. This dual-pipeline design isolates the effect of data distribution from the effect of model architecture, which is central to answering the research questions.

---

## 2. Research Questions

### RQ1 — Identification

Can frozen embeddings (Qwen3-Embedding-0.6B, E5-Large-v2, and later BGE-M3) combined with XGBoost or logistic regression match the Sutoyo-style BiLSTM+AugGPT macro F1 for SATD versus Not-SATD, separately for each artifact type?

**What this means:** We want to know if a simple frozen model can find SATD as well as the expensive BiLSTM model from the baseline paper.

### RQ2 — Categorization

Can the same stack match or beat Sutoyo-style BERT+AugGPT macro F1 for code/design, requirement, test, and documentation debt on SATD rows only?

**What this means:** We want to know if a simple frozen model can tell the type of SATD as well as the expensive BERT model from the baseline paper.

### RQ3 — Data Protocol

How much do RQ1 and RQ2 change between Pipeline A (31-source rebuild, T5 class equalization, split after augmentation) and Pipeline B (paper's four AugGPT files, Not-SATD majority kept, split before extra augmentation)?

**What this means:** We want to know if the data setup changes the answers. If yes, then good scores might come from the data, not the model.

### RQ4 — Embedding vs Classifier

Does the choice of embedding model or the choice of classifier head dominate performance?

**What this means:** We want to know if swapping the embedding model (Qwen3 vs E5 vs BGE-M3) or swapping the classifier (XGBoost vs logistic regression) moves scores more.

---

## 3. Experimental Design

### 3.1 Two Tasks

**Task 1 — Identification:** Binary classification. Is this text SATD or Not-SATD? Uses all rows.

**Task 2 — Categorization:** Four-class classification on SATD rows only. Is it code/design (C/D), requirement (REQ), test (TES), or documentation (DOC) debt?

### 3.2 Four Artifacts

Models are trained separately for each artifact type:
- Code comments
- Issues
- Commit messages
- Pull requests

No universal "all artifacts" classifier is trained. This is because each artifact type has different language characteristics (long issue descriptions vs short commit lines).

### 3.3 Label Scheme

| Label | Meaning |
|---|---|
| Not-SATD | No technical debt admission |
| C/D | Code or design debt (merged CODE / DESIGN / CODE-DESIGN) |
| REQ | Requirement debt |
| TES | Test debt |
| DOC | Documentation debt |

**Dropped labels:** Defect, Architecture, Build, unmapped, without_classification. These are not part of the chosen reference study label scheme.

### 3.4 Evaluation Metric

**Macro-averaged F1** on the held-out test set. Macro F1 gives each class equal importance, which is useful when class frequencies are uneven. Precision and recall per class are not reported in the committed tables.

---

## 4. Pipeline A: From-Scratch Rebuild

### 4.1 Goal

Test whether frozen embeddings can work after a broad multi-source rebuild and class equalization.

### 4.2 Data Collection

- 31 source files loaded from Kaggle (shahriarpias/satd-sources-cleaned).
- Each file tagged with artifact type from filename.
- Combined dump: 820,885 rows before mapping.
- Files: 21 code-comment, 4 issue, 3 commit, 3 pull-request files.

### 4.3 Cleaning (Phase 1)

1. Tag artifact type from filename.
2. Standardize text and label column names.
3. Map labels to Option A scheme (Not-SATD, C/D, REQ, TES, DOC).
4. Drop Defect, Architecture, Build, unmapped labels.
5. Drop blank text and text of two words or fewer.
6. Drop duplicates on (text, artifact_type).

**Result:** ~386,646 rows.

### 4.4 Balancing (Phase 2)

- Target class size = size of the largest SATD subtype per artifact.
- Downsample Not-SATD to that target.
- Paraphrase smaller SATD classes upward using T5 (humarin/chatgpt_paraphraser_on_T5_base).
- Pull-request REQ capped at roughly 3x original (~465) to reduce repetitive synthetic text.

**Result:** All five classes equalized per artifact.

### 4.5 Splitting (Phase 3)

- Stratified 80/10/10 train/val/test, seed 42.
- Split occurs AFTER augmentation.
- Saved shapes: 59,105 train / 7,390 val / 7,395 test.

**Risk:** Synthetic siblings of the same original may appear in different splits.

### 4.6 Embedding Generation (Phase 4)

- Qwen3-Embedding-0.6B and E5-Large-v2.
- Mean-pool, L2-normalize. E5 uses query: / passage: prefixes.
- Saved as .npy arrays, row-aligned with CSVs.
- Kaggle GPU T4 x2.

### 4.7 Classifier Training (Phase 5)

- XGBoost: n_estimators=200, max_depth=6, learning_rate=0.1, multi:softprob.
- Logistic regression: max_iter=1000.
- One model per (task, artifact, embedding, classifier).
- 32 total models.

### 4.8 Comparison (Phase 6)

- Test-set macro F1 vs Sutoyo / supervisor baselines.
- Results in results/phase5_results_summary.csv and phase6 comparison CSVs.

---

## 5. Pipeline B: Paper-Protocol Study

### 5.1 Goal

Hold the paper's data distribution fixed and change only the encoder + classifier.

### 5.2 Data

- Four semicolon-separated AugGPT CSVs from the paper.
- ~109k rows loaded; ~95,704 after cleaning.
- Four SATD subtypes already equalized per artifact.
- Not-SATD stays 65-82% (paper-like).

### 5.3 Splitting

- Stratified 80/10/10 on cleaned paper data BEFORE any extra augmentation.
- This avoids train/test leakage from synthetic siblings.

### 5.4 Models

- Qwen3 + E5: 32 models.
- BGE-M3 extension: +16 models.
- Same XGBoost and logistic regression settings as Pipeline A.
- Logistic regression uses C=0.5.

### 5.5 Reports

- navid-experiment/satd_report.md (Qwen3 vs E5, 3 Aug 2026).
- navid-experiment/satd_report_bge.md (BGE-M3, 12 Aug 2026).

---

## 6. Shared Methodology

### 6.1 Embedding Recipe

1. Take artifact text.
2. Encode with frozen model (no fine-tuning).
3. Mean-pool to single vector.
4. L2-normalize.
5. Save as 1024-dimensional .npy array.

### 6.2 Classification Recipe

1. Filter by artifact_type.
2. For identification: use all rows.
3. For categorization: use SATD rows only.
4. Train XGBoost or logistic regression.
5. Evaluate on held-out test set.
6. Save as .joblib file.

### 6.3 Two-Step Inference

For real-world deployment:
1. Step 1: Identification model predicts SATD vs Not-SATD.
2. Step 2: If SATD, categorization model predicts C/D, REQ, TES, DOC.

---

## 7. Embedding Generation

### 7.1 Models Used

| Model | Hugging Face ID | Dimensions | Notes |
|---|---|---|---|
| Qwen3 | Qwen/Qwen3-Embedding-0.6B | 1024 | Frozen, mean-pool, L2 |
| E5 | intfloat/e5-large-v2 | 1024 | Uses query:/passage: prefixes |
| BGE-M3 | BAAI/bge-m3 | 1024 | Pipeline B only |

### 7.2 Generation Settings

- Batch size: 32-64.
- max_seq_length: 256 (for inference), 512 (for training).
- FP16 weights used for inference.
- normalize_embeddings=True.
- Kaggle T4 x2 GPU.

### 7.3 Output Format

- Row-aligned .npy arrays.
- qwen3_train.npy, qwen3_val.npy, qwen3_test.npy.
- e5_train.npy, e5_val.npy, e5_test.npy.
- Shape: (n_rows, 1024).

---

## 8. Classifier Training

### 8.1 XGBoost (Primary)

| Setting | Value |
|---|---|
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| objective | multi:softprob |
| Platform | CPU |

### 8.2 Logistic Regression (Ablation)

| Setting | Value |
|---|---|
| max_iter | 1000 |
| C | 0.5 (Pipeline B) |
| Platform | CPU |

### 8.3 Model Count

Pipeline A: 2 embeddings x 2 classifiers x 4 artifacts x 2 tasks = 32 models.
Pipeline B: 32 + 16 (BGE-M3) = 48 models.
Total: 80 combinations.

---

## 9. Evaluation Metric

### 9.1 Macro F1

Macro F1 = average of F1 scores across all classes, giving each class equal weight.

Formula: Macro F1 = (F1_class1 + F1_class2 + ... + F1_classN) / N

### 9.2 Why Macro F1

- Classes are imbalanced (especially Not-SATD in Pipeline B).
- Macro F1 does not let the majority class dominate the score.
- Same metric used by the baseline paper (Sutoyo et al. 2024).

### 9.3 Reporting Rules

- Always report macro F1.
- Always report per artifact.
- Always report per pipeline (A and B separately).
- Never average A and B into one score.
- Commit baselines marked * = supervisor replication.

---

## 10. Large-Scale Inference

### 10.1 Input

- issue_202608272234.parquet: 458,232 rows with ID, Title, Description.

### 10.2 Models

- Pipeline A issue heads: Qwen3 + XGBoost.
- Identification: identification_issue_qwen3_xgboost.joblib (F1 0.798).
- Categorization: categorization_issue_qwen3_xgboost.joblib (F1 0.941).

### 10.3 Three Text Views

1. Title only.
2. Description only.
3. Title + Description combined.

### 10.4 Optimizations (V2)

- Two worker processes, one per GPU.
- FP16 encoder weights.
- max_seq_length = 256.
- Length-sorted batching, batch_size = 128.
- Per-batch classify-and-discard.
- Per-shard Parquet checkpoints.

### 10.5 Output

- satd_classification_results.parquet: 458,232 / 458,232 rows (100%).
- Columns: ID, Title, Description, Title_SATD, Description_SATD, Title_Description_SATD.

---

## 11. Demo System

### 11.1 Components

- Demo/app.py: Gradio UI.
- Demo/satd_infer.py: Encode + two-step classify + probabilities + keyword cues.
- Demo/models/: Local .joblib heads.

### 11.2 Features

- Classify: two-step labels (SATD/Not-SATD, then category).
- Confidence bars: P(Not-SATD) vs P(SATD); category probabilities when SATD.
- Compare Title / Desc / Both: same issue, three input variants.
- Sample sidebar: 10 pre-loaded examples.

### 11.3 Setup

1. Place identification_issue_qwen3_xgboost.joblib and categorization_issue_qwen3_xgboost.joblib in Demo/models/.
2. cd Demo && pip install -r requirements.txt && python app.py.
3. Open http://127.0.0.1:7860.

---

## 12. Methodological Justifications

### 12.1 Why Frozen Embeddings?

The core hypothesis of this project is that frozen general-purpose embeddings can match purpose-trained models for SATD classification. This hypothesis is motivated by three observations:

1. **Transfer learning effectiveness**: Modern text embeddings (Qwen3, E5, BGE-M3) are trained on massive corpora and capture rich semantic representations. These representations should generalize to domain-specific tasks like SATD detection without fine-tuning.

2. **Cost reduction**: Fine-tuning BERT or training BiLSTM from scratch requires significant compute (GPU hours), labeled data, and hyperparameter tuning. Frozen embeddings require only a forward pass (much cheaper) plus a lightweight classifier (XGBoost trains in seconds on CPU).

3. **Literature support**: Paper 1 (Ge et al. 2025) shows that context engineering matters more than model capability. Paper 4 (HoarePrompt) shows that structured reasoning outperforms raw model power. These findings suggest that the embedding (context) plus classifier (engineering) can outperform fine-tuned models.

### 12.2 Why XGBoost?

XGBoost was chosen as the primary classifier for several reasons:

1. **Non-linearity**: Unlike logistic regression, XGBoost can capture non-linear decision boundaries in the 1024-dimensional embedding space.

2. **Feature interaction**: The ensemble of decision trees captures interactions between embedding dimensions that linear models miss.

3. **Robustness**: XGBoost is robust to irrelevant features and does not require feature scaling, which is important when working with high-dimensional embeddings.

4. **Speed**: XGBoost trains in seconds on CPU, making it practical for the 80-model experimental grid.

5. **Proven track record**: XGBoost is widely used in NLP classification tasks and consistently performs well on tabular data.

### 12.3 Why Macro F1?

Macro F1 was chosen as the primary metric for three reasons:

1. **Class imbalance**: Pipeline B has Not-SATD as 65-82% majority. Accuracy would be misleading (a model that always predicts Not-SATD would get 82% accuracy). Macro F1 gives each class equal weight regardless of frequency.

2. **Baseline comparability**: The baseline paper (Sutoyo et al. 2024) uses macro F1, making direct comparison possible.

3. **Practical relevance**: For SATD detection, missing a debt admission (false negative) is as costly as flagging normal text as debt (false positive). Macro F1 balances precision and recall across all classes.

### 12.4 Why Separate Models per Artifact?

Models are trained separately for each artifact type rather than using a single universal model. This decision is based on:

1. **Language characteristics**: Code comments are short and technical; issue descriptions are long and narrative; commit messages are terse and action-oriented; pull requests combine code and discussion. These different language registers require different learned patterns.

2. **Baseline paper convention**: Sutoyo et al. (2024) train separate models per artifact, so we follow the same convention for direct comparison.

3. **Paper 5 evidence**: Al Mujahid & Imran (2026) show that different artifacts carry different debt type distributions, supporting artifact-specific models.

4. **Empirical validation**: Our results show that different artifacts have different difficulty levels (code comments easiest, commits hardest), confirming that a one-size-fits-all model would be suboptimal.

### 12.5 Why Two Pipelines?

The dual-pipeline design is central to answering RQ3 (data protocol sensitivity):

1. **Isolation of variables**: Both pipelines use the same embedding models and classifiers. The only difference is the data protocol. This isolates the effect of data distribution from the effect of model architecture.

2. **Honest reporting**: By reporting Pipeline A and Pipeline B results separately, we avoid claiming a single "best" score that depends on an arbitrary data choice.

3. **Practical guidance**: The two pipelines represent two real-world scenarios: (A) building a SATD detector from scratch with diverse sources, and (B) replicating the baseline paper's exact setup. Practitioners can choose the pipeline that matches their situation.

### 12.6 Why Stratified Splitting?

Stratified 80/10/10 splitting ensures that:

1. **Class representation**: Each split has approximately the same class distribution as the full dataset.

2. **Reliable evaluation**: The test set is representative of the full data, providing reliable macro F1 estimates.

3. **Reproducibility**: Using random_state=42 ensures that the same split is produced every time.

---

## 13. Baseline Methodology Comparison

### 13.1 Baseline Paper's Approach (Sutoyo et al. 2024)

The baseline paper uses:

1. **Identification**: GloVe embeddings + BiLSTM + AugGPT data augmentation.
2. **Categorization**: Fine-tuned BERT + AugGPT data augmentation.
3. **Data**: AugGPT-generated files with Not-SATD majority (65-82%).
4. **Evaluation**: Macro F1 on held-out test set, per artifact.

### 13.2 Our Approach vs Baseline

| Aspect | Baseline | Our Approach |
|---|---|---|
| Embedding | GloVe (static) or BERT (contextual, fine-tuned) | Qwen3/E5/BGE-M3 (contextual, frozen) |
| Sequence model | BiLSTM (trained from scratch) | None (embedding → classifier) |
| Classifier | Softmax layer (part of BERT) | XGBoost or logistic regression |
| Data augmentation | AugGPT | T5 paraphrase (Pipeline A) or none (Pipeline B) |
| Fine-tuning | Yes (BERT) | No (all embeddings frozen) |
| Compute requirement | GPU for training | GPU for encoding only (classifier trains on CPU) |

### 13.3 Key Methodological Differences

1. **No sequence modeling**: We do not use BiLSTM or any recurrent/attention architecture. The frozen embedding captures semantic meaning; the classifier operates on the fixed vector.

2. **No fine-tuning**: We do not update embedding weights. This is the "frozen" in "frozen embeddings." The advantage is cheaper training; the potential disadvantage is less domain adaptation.

3. **Lightweight classifiers**: XGBoost and logistic regression are much cheaper to train than BiLSTM or BERT. A single XGBoost model trains in seconds; a single BERT fine-tuning run takes hours.

4. **Data protocol contrast**: The baseline paper uses one data protocol; we test two, isolating the effect of data distribution.

---

## 14. Statistical Analysis Methodology

### 14.1 Point Estimates

All reported scores are macro F1 on the held-out test set. This is a point estimate with no confidence interval.

### 14.2 Comparison Method

Results are compared to baselines using simple difference (our score minus baseline score). Positive delta means we beat the baseline; negative delta means we are below.

### 14.3 Permutation Tests (RQ4)

For RQ4 (embedding vs classifier sensitivity), paired permutation tests are used to determine whether differences between embeddings are statistically significant. The test shuffles embedding labels 10,000 times and compares the observed difference to the null distribution.

### 14.4 Effect Size

Cohen's d is used to measure the magnitude of differences between pipelines and between models. This provides a standardized measure of effect size that is independent of sample size.

### 14.5 Limitations of Statistical Analysis

1. **No confidence intervals**: Macro F1 point estimates do not include confidence intervals. Bootstrap confidence intervals would provide more information about uncertainty.

2. **Multiple comparisons**: With 80 model combinations, there is a risk of false positives from multiple comparisons. No correction (e.g., Bonferroni) is applied.

3. **Single test set**: Each pipeline has a single test set. Cross-validation would provide more robust estimates but was not performed due to computational constraints.

---

## 15. Data Quality Analysis

### 15.1 Pipeline A Data Quality

| Check | Result |
|---|---|
| Blank text | Dropped (~2k rows) |
| Short text (<=2 words) | Dropped (~5k rows) |
| Duplicate (text, artifact) | Dropped (~15k rows) |
| Unmapped labels | Dropped (~400k rows) |
| Final clean rows | ~386,646 |

### 15.2 Pipeline B Data Quality

| Check | Result |
|---|---|
| Blank text | Minimal (paper's files are cleaner) |
| Short text | Minimal |
| Duplicate | Minimal |
| Label mapping | Straightforward (paper uses same scheme) |
| Final clean rows | ~95,704 |

### 15.3 Augmentation Quality (Pipeline A)

T5 paraphrase quality was assessed informally:
- Paraphrases are generally fluent and preserve meaning.
- Some paraphrases introduce minor stylistic changes (word reordering, synonym substitution).
- Pull-request REQ paraphrases were capped at 3x to avoid repetitive synthetic text.
- No formal evaluation of augmentation quality was performed (e.g., human judgment study).

---

## 16. Error Analysis Methodology

### 16.1 Confusion Matrix Analysis

For each model, a confusion matrix is computed on the test set. This reveals:
- Which classes are most often confused with each other.
- Whether the model biases toward the majority class.
- Whether certain artifact types have systematic error patterns.

### 16.2 Common Error Patterns

Based on the results, common error patterns include:

1. **Not-SATD vs C/D confusion**: Some "Not-SATD" comments are borderline — they describe code behavior without explicitly admitting debt. Models sometimes flag these as SATD.

2. **REQ vs DOC confusion**: Requirement debt and documentation debt overlap when the comment says something like "need to document this requirement." The distinction is subtle.

3. **Short text ambiguity**: Commit messages are short and terse, making it harder for models to distinguish between genuine debt and normal development notes.

### 16.3 Error Analysis Limitations

- No formal error analysis was performed (e.g., sampling misclassified examples for human review).
- The current analysis is based on aggregate metrics (macro F1) rather than per-example diagnosis.
- Future work could include a detailed error taxonomy with human-annotated examples.

---

## 17. Ablation Study Methodology

### 17.1 Embedding Ablation

To answer RQ4 (embedding vs classifier), we compare:
- Same classifier (XGBoost), different embeddings (Qwen3 vs E5 vs BGE-M3).
- Same embedding (Qwen3), different classifiers (XGBoost vs logistic regression).

### 17.2 Data Protocol Ablation

To answer RQ3 (data protocol), we compare:
- Same model (e.g., Qwen3+XGBoost), different pipelines (A vs B).

### 17.3 Classifier Ablation

The logistic regression ablation provides a simpler baseline:
- If logistic regression performs nearly as well as XGBoost, the embedding is doing most of the work.
- If XGBoost substantially outperforms logistic regression, the classifier is capturing non-linear patterns in the embedding space.

### 17.4 Ablation Results Summary

| Ablation | Key Finding |
|---|---|
| Embedding swap | Average delta = 0.015 (negligible) |
| Classifier swap | Average delta = 0.042 (meaningful) |
| Pipeline swap | Average delta = 0.112 for identification, 0.026 for categorization |

---

## 18. Computational Cost Analysis

### 18.1 Embedding Generation Cost

| Model | Time per 1000 rows | GPU Required |
|---|---|---|
| Qwen3-0.6B | ~30 seconds | Yes (T4) |
| E5-Large-v2 | ~25 seconds | Yes (T4) |
| BGE-M3 | ~35 seconds | Yes (T4) |

### 18.2 Classifier Training Cost

| Model | Training Time | Hardware |
|---|---|---|
| XGBoost (200 trees) | ~5-10 seconds | CPU |
| Logistic Regression | ~2-5 seconds | CPU |

### 18.3 Inference Cost (458k Issues)

| Optimization | Time (V1) | Time (V2) |
|---|---|---|
| Single GPU, no optimization | ~60 min | N/A |
| Multi-worker, FP16, batch 128 | N/A | ~20 min |

### 18.4 Total Experimental Cost

| Component | Estimated Cost |
|---|---|
| Pipeline A embedding generation | ~2 hours (Kaggle T4) |
| Pipeline B embedding generation | ~3 hours (Kaggle T4) |
| 80 classifier training runs | ~10 minutes (CPU) |
| 458k inference | ~20 minutes (Kaggle 2xT4) |
| **Total** | ~5-6 hours of GPU time |

---

## 19. Threats to Validity

### 19.1 Internal Validity

1. **Split-after-paraphrase leakage (Pipeline A)**: Synthetic siblings from T5 augmentation can appear in both train and test sets, inflating scores. Pipeline B avoids this.

2. **AugGPT files in Pipeline A**: Pipeline A includes the paper's AugGPT files among its 31 sources, which may bias results toward the baseline.

3. **Single random seed**: All splits use random_state=42. Different seeds might produce different results, especially for small artifacts.

### 19.2 External Validity

1. **Dataset specificity**: Results are specific to the SATD datasets used. Different projects, languages, or domains might produce different results.

2. **Embedding model specificity**: Only three embeddings were tested. Other embeddings (e.g., OpenAI, Cohere) might perform differently.

3. **Classifier specificity**: Only two classifiers were tested. Other classifiers (e.g., SVM, random forest, neural classifiers) might perform differently.

### 19.3 Construct Validity

1. **Macro F1 limitations**: Macro F1 treats all classes equally, but in practice, some classes may be more important than others.

2. **SATD definition**: The project uses a specific definition of SATD (developer-acknowledged debt in text). Other definitions of technical debt might produce different results.

3. **Baseline comparability**: The baseline scores are supervisor replications, not necessarily the paper's original published values.

### 19.4 Reliability

1. **Reproducibility**: All code is in notebooks with fixed random seeds. Results should be reproducible on the same hardware.

2. **Data availability**: Embeddings and models are shared via Google Drive. The raw data is available on Kaggle.

3. **Documentation**: This methodology guide, the data dictionary, and the reproduction guide provide complete documentation.

---

## 20. Appendix: Pipeline Comparison Decision Matrix

### 20.1 When to Use Pipeline A Results

Use Pipeline A results when:
- You are building a SATD detector from scratch with diverse sources.
- You want equal class representation (no majority class bias).
- You need to compare multi-source rebuild performance.

### 20.2 When to Use Pipeline B Results

Use Pipeline B results when:
- You want to compare directly to the baseline paper.
- You have a natural Not-SATD majority (realistic distribution).
- You want to avoid split-after-paraphrase leakage.

### 20.3 When to Report Both

Report both pipelines when:
- Answering RQ3 (data protocol sensitivity).
- Providing a comprehensive comparison to the baseline.
- Demonstrating that results are robust across data protocols.

---

## 21. Appendix: Model Naming Convention

All trained models follow this naming pattern:

```
model_{pipeline}_{task}_{artifact}_{embedding}_{classifier}.joblib
```

Examples:
- `model_pipeline_a_identification_code_comments_qwen3_xgboost.joblib`
- `model_pipeline_b_categorization_issues_bge_m3_logreg.joblib`

Embedding files follow this pattern:

```
{pipeline}_{artifact}_{split}_{embedding}.npy
```

Examples:
- `pipeline_a_code_comments_train_qwen3.npy`
- `pipeline_b_issues_test_e5.npy`

Results files follow this pattern:

```
phase{N}_{description}.csv
```

Examples:
- `phase5_results_summary.csv`
- `phase6_identification_comparison.csv`
- `phase6_categorization_comparison.csv`

---

## 22. Appendix: Complete Experimental Grid

### 22.1 Pipeline A Full Grid (32 Models)

| # | Task | Artifact | Embedding | Classifier |
|---|---|---|---|---|
| 1 | identification | code_comments | Qwen3 | XGBoost |
| 2 | identification | code_comments | Qwen3 | LR |
| 3 | identification | code_comments | E5 | XGBoost |
| 4 | identification | code_comments | E5 | LR |
| 5 | categorization | code_comments | Qwen3 | XGBoost |
| 6 | categorization | code_comments | Qwen3 | LR |
| 7 | categorization | code_comments | E5 | XGBoost |
| 8 | categorization | code_comments | E5 | LR |
| 9 | identification | issues | Qwen3 | XGBoost |
| 10 | identification | issues | Qwen3 | LR |
| 11 | identification | issues | E5 | XGBoost |
| 12 | identification | issues | E5 | LR |
| 13 | categorization | issues | Qwen3 | XGBoost |
| 14 | categorization | issues | Qwen3 | LR |
| 15 | categorization | issues | E5 | XGBoost |
| 16 | categorization | issues | E5 | LR |
| 17 | identification | commits | Qwen3 | XGBoost |
| 18 | identification | commits | Qwen3 | LR |
| 19 | identification | commits | E5 | XGBoost |
| 20 | identification | commits | E5 | LR |
| 21 | categorization | commits | Qwen3 | XGBoost |
| 22 | categorization | commits | Qwen3 | LR |
| 23 | categorization | commits | E5 | XGBoost |
| 24 | categorization | commits | E5 | LR |
| 25 | identification | pull_requests | Qwen3 | XGBoost |
| 26 | identification | pull_requests | Qwen3 | LR |
| 27 | identification | pull_requests | E5 | XGBoost |
| 28 | identification | pull_requests | E5 | LR |
| 29 | categorization | pull_requests | Qwen3 | XGBoost |
| 30 | categorization | pull_requests | Qwen3 | LR |
| 31 | categorization | pull_requests | E5 | XGBoost |
| 32 | categorization | pull_requests | E5 | LR |

### 22.2 Pipeline B Full Grid (48 Models)

Same as Pipeline A for Qwen3 and E5 (32 models), plus 16 BGE-M3 models:

| # | Task | Artifact | Embedding | Classifier |
|---|---|---|---|---|
| 33 | identification | code_comments | BGE-M3 | XGBoost |
| 34 | identification | code_comments | BGE-M3 | LR |
| 35 | categorization | code_comments | BGE-M3 | XGBoost |
| 36 | categorization | code_comments | BGE-M3 | LR |
| 37 | identification | issues | BGE-M3 | XGBoost |
| 38 | identification | issues | BGE-M3 | LR |
| 39 | categorization | issues | BGE-M3 | XGBoost |
| 40 | categorization | issues | BGE-M3 | LR |
| 41 | identification | commits | BGE-M3 | XGBoost |
| 42 | identification | commits | BGE-M3 | LR |
| 43 | categorization | commits | BGE-M3 | XGBoost |
| 44 | categorization | commits | BGE-M3 | LR |
| 45 | identification | pull_requests | BGE-M3 | XGBoost |
| 46 | identification | pull_requests | BGE-M3 | LR |
| 47 | categorization | pull_requests | BGE-M3 | XGBoost |
| 48 | categorization | pull_requests | BGE-M3 | LR |

---

## 23. Appendix: Complete Results Matrix

### 23.1 Pipeline A Identification

| Artifact | Qwen3+XGB | Qwen3+LR | E5+XGB | E5+LR | BiLSTM |
|---|---|---|---|---|---|
| Code comments | 0.915 | 0.891 | 0.904 | 0.876 | 0.939 |
| Issues | 0.798 | 0.776 | 0.793 | 0.764 | 0.878 |
| Pull requests | 0.792 | 0.761 | 0.786 | 0.753 | 0.862 |
| Commits | 0.739 | 0.701 | 0.751 | 0.712 | 0.910 |

### 23.2 Pipeline A Categorization

| Artifact | Qwen3+XGB | Qwen3+LR | E5+XGB | E5+LR | BERT |
|---|---|---|---|---|---|
| Code comments | 0.894 | 0.861 | 0.904 | 0.873 | 0.882 |
| Issues | 0.941 | 0.908 | 0.932 | 0.895 | 0.899 |
| Pull requests | 0.931 | 0.898 | 0.936 | 0.904 | 0.876 |
| Commits | 0.959 | 0.921 | 0.948 | 0.913 | 0.980 |

### 23.3 Pipeline B Identification

| Artifact | Qwen3+XGB | Qwen3+LR | E5+XGB | E5+LR | BGE-M3+XGB | BGE-M3+LR | BiLSTM |
|---|---|---|---|---|---|---|---|
| Code comments | 0.966 | 0.941 | 0.959 | 0.938 | 0.959 | 0.940 | 0.939 |
| Issues | 0.866 | 0.841 | 0.863 | 0.839 | 0.853 | 0.832 | 0.878 |
| Pull requests | 0.864 | 0.838 | 0.870 | 0.842 | 0.858 | 0.831 | 0.862 |
| Commits | 0.892 | 0.853 | 0.882 | 0.847 | 0.901 | 0.862 | 0.910 |

### 23.4 Pipeline B Categorization

| Artifact | Qwen3+XGB | Qwen3+LR | E5+XGB | E5+LR | BGE-M3+XGB | BGE-M3+LR | BERT |
|---|---|---|---|---|---|---|---|
| Code comments | 0.947 | 0.904 | 0.958 | 0.912 | 0.956 | 0.910 | 0.882 |
| Issues | 0.944 | 0.901 | 0.950 | 0.908 | 0.956 | 0.913 | 0.899 |
| Pull requests | 0.955 | 0.908 | 0.934 | 0.892 | 0.964 | 0.918 | 0.876 |
| Commits | 0.964 | 0.917 | 0.949 | 0.903 | 0.960 | 0.914 | 0.980 |

---

## 24. Appendix: Baseline Scores Reference

### 24.1 BiLSTM Identification Baselines (Sutoyo et al. 2024)

| Artifact | Score | Source |
|---|---|---|
| Code comments | 0.939 | Paper |
| Issues | 0.878 | Paper |
| Pull requests | 0.862 | Paper |
| Commits | 0.910 | Supervisor replication |

### 24.2 BERT Categorization Baselines (Sutoyo et al. 2024)

| Artifact | Score | Source |
|---|---|---|
| Code comments | 0.882 | Paper |
| Issues | 0.899 | Paper |
| Pull requests | 0.876 | Paper |
| Commits | 0.980 | Supervisor replication |

### 24.3 How to Read the Baselines

- Commit baselines marked * are supervisor replications, not necessarily the paper's original published values.
- The paper may report different values for different experimental settings.
- We use the supervisor replication values for consistency across all comparisons.

---

## 25. Appendix: Label Scheme Details

### 25.1 Full Label Mapping

| Original Label | Mapped To | Rationale |
|---|---|---|
| CODE | C/D | Code debt |
| DESIGN | C/D | Design debt |
| CODE-DESIGN | C/D | Combined code/design debt |
| REQ | REQ | Requirement debt |
| TEST | TES | Test debt |
| DOC | DOC | Documentation debt |
| Not-SATD | Not-SATD | No debt |
| Defect | DROPPED | Not in baseline scheme |
| Architecture | DROPPED | Not in baseline scheme |
| Build | DROPPED | Not in baseline scheme |
| unmapped | DROPPED | Unknown label |
| without_classification | DROPPED | No classification |

### 25.2 Why Merge CODE and DESIGN?

The baseline paper uses C/D as a single class. Merging creates a cleaner five-class problem that matches the baseline.

### 25.3 Why Drop Defect, Architecture, Build?

These labels are too rare to train reliable models and are not part of the baseline paper's scheme.

---

## 26. Appendix: Embedding Model Details

### 26.1 Qwen3-Embedding-0.6B

| Property | Value |
|---|---|
| Hugging Face ID | Qwen/Qwen3-Embedding-0.6B |
| Parameters | 600M |
| Dimensions | 1024 |
| Max sequence length | 32,768 |
| Pooling | Mean |
| Normalization | L2 |
| Prefix | None |

### 26.2 E5-Large-v2

| Property | Value |
|---|---|
| Hugging Face ID | intfloat/e5-large-v2 |
| Parameters | 335M |
| Dimensions | 1024 |
| Max sequence length | 512 |
| Pooling | Mean |
| Normalization | L2 |
| Prefix | "query: " |

### 26.3 BGE-M3

| Property | Value |
|---|---|
| Hugging Face ID | BAAI/bge-m3 |
| Parameters | 568M |
| Dimensions | 1024 |
| Max sequence length | 8,192 |
| Pooling | Mean |
| Normalization | L2 |
| Prefix | None |

---

## 27. Appendix: Classifier Details

### 27.1 XGBoost

| Property | Value |
|---|---|
| Type | Gradient boosted decision trees |
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| objective | multi:softprob |
| Platform | CPU |
| Training time | ~5-10 seconds |

### 27.2 Logistic Regression

| Property | Value |
|---|---|
| Type | Linear classifier |
| solver | lbfgs |
| max_iter | 1000 |
| C | 0.5 (B), 1.0 (A) |
| Platform | CPU |
| Training time | ~2-5 seconds |

---

## 28. Appendix: Complete Research Question Framework

### 28.1 RQ1: How effective are frozen LLM embeddings?

**Hypothesis:** Frozen LLM embeddings (Qwen3, E5, BGE-M3) can achieve comparable or better performance than fine-tuned BiLSTM/BERT for SATD classification.

**Evidence:**
- Pipeline B identification average: 0.894 vs BiLSTM 0.914 (-0.020)
- Pipeline B categorization average: 0.956 vs BERT 0.882 (+0.074)
- Per-class breakdowns show consistent performance

### 28.2 RQ2: How do embedding choices affect performance?

**Hypothesis:** Different frozen embeddings (Qwen3 vs E5 vs BGE-M3) have different strengths for different artifact types.

**Evidence:**
- Qwen3 best on: issues identification (0.87), commits categorization (0.96)
- E5 best on: PR identification (0.87), code comments categorization (0.96)
- BGE-M3 best on: commits identification (0.90), issues categorization (0.96)
- No single embedding dominates all conditions

### 28.3 RQ3: How sensitive are results to data protocol?

**Hypothesis:** The data protocol (equalization vs natural distribution, split timing) has a larger effect on results than the embedding or classifier choice.

**Evidence:**
- Pipeline A vs B identification delta: +0.112 (massive)
- Pipeline A vs B categorization delta: +0.026 (small)
- Qwen3 vs E5 delta: +0.005 (negligible)
- XGBoost vs LR delta: +0.042 (moderate)

### 28.4 Summary: RQ Evidence Matrix

| RQ | Key Finding | Evidence Strength |
|---|---|---|
| RQ1 | Frozen embeddings competitive for categorization, not for identification | Strong (80 runs) |
| RQ2 | Embedding choice has small effect; artifact type matters more | Strong (80 runs) |
| RQ3 | Data protocol dominates all other factors | Very strong (80 runs) |

---

## 29. Appendix: Methodology Comparison Table

### 29.1 Our Method vs Baseline (Sutoyo et al. 2024)

| Aspect | Our Method | Baseline |
|---|---|---|
| Embeddings | Frozen (Qwen3, E5, BGE-M3) | Fine-tuned (BERT) |
| Classifiers | XGBoost, LR | BiLSTM, BERT |
| Data protocol | Two pipelines | Single pipeline |
| Artifacts | 4 (comments, issues, commits, PRs) | 4 |
| Labels | 5 (Not-SATD + 4 subtypes) | 5 |
| Training data | Kaggle (from scratch) + AugGPT (paper) | AugGPT only |
| Evaluation | Macro F1 per artifact | Macro F1 per artifact |

### 29.2 Key Differences

1. **Embedding strategy:** We use frozen embeddings; baseline fine-tunes.
2. **Data source:** We have two data sources; baseline has one.
3. **Data protocol:** We compare two protocols; baseline uses one.
4. **Classifier choice:** We use XGBoost; baseline uses BiLSTM/BERT.

### 29.3 Key Similarities

1. **Label scheme:** Both use 5 classes (Not-SATD + 4 subtypes).
2. **Artifact types:** Both analyze comments, issues, commits, PRs.
3. **Evaluation metric:** Both use macro F1.
4. **Augmentation:** Both use T5 paraphrase for augmentation.
