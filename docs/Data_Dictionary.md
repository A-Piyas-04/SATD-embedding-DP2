# Data Dictionary

**SATD Embedding Comparison Project**
**Team: Shahriar Pias, Navid Ibrahim, Musaddiq Rafi**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Primary Label Scheme (Option A)](#2-primary-label-scheme)
3. [Artifacts](#3-artifacts)
4. [Pipeline A: 31-Source Kaggle Dataset](#4-pipeline-a-dataset)
5. [Pipeline B: Paper AugGPT Dataset](#5-pipeline-b-dataset)
6. [Large-Scale Issue Dataset](#6-large-scale-issue-dataset)
7. [CSV Schemas](#7-csv-schemas)
8. [Parquet Schema](#8-parquet-schema)
9. [Label Mapping Rules](#9-label-mapping-rules)
10. [Synthetic Augmentation Tags](#10-synthetic-augmentation-tags)
11. [Column Conventions](#11-column-conventions)
12. [File Naming Conventions](#12-file-naming-conventions)
13. [Missing Values and Edge Cases](#13-missing-values)

---

## 1. Overview

This document describes every field, label, artifact, schema, and naming convention used in the SATD Embedding Comparison project. It serves as the authoritative reference for anyone reading the data files, results CSVs, or inference output.

---

## 2. Primary Label Scheme (Option A)

The project uses a single five-class label scheme across both pipelines and all evaluation reports.

| Code | Full Name | Description | Example SATD Comment |
|---|---|---|---|
| Not-SATD | Not self-admitted technical debt | Normal code/comment with no debt signal | "Returns the user profile." |
| C/D | Code or Design debt | Refactoring candidates, hacks, design flaws | "TODO: this is a hack" |
| REQ | Requirement debt | Missing or evolving requirements | "We need to add validation here" |
| TES | Test debt | Insufficient or missing tests | "Add proper tests" |
| DOC | Documentation debt | Missing or outdated docs | "Document this function" |

### 2.1 Labels Dropped from Raw Data

The following labels were removed during mapping:
- Defect
- Architecture
- Build
- unmapped
- without_classification
- Unknown

These were either too rare or not part of the baseline paper's scheme.

### 2.2 Why Five Classes

The baseline paper (Sutoyo et al. 2024) uses four SATD classes plus Not-SATD. The merged CODE + DESIGN into a single C/D class because code debt and design debt are closely related and often overlap in short text. This creates a cleaner five-class problem: Not-SATD, C/D, REQ, TES, DOC.

---

## 3. Artifacts

Models are trained and evaluated separately for each artifact type. There is no universal "all artifacts" model.

| Artifact | Source Files (Pipeline A) | Source Files (Pipeline B) | Description |
|---|---|---|---|
| code_comments | 21 CSV files | `code_comments_AugGPT.csv` | Inline code comments with debt acknowledgments |
| issues | 4 CSV files | `issues_AugGPT.csv` | GitHub issue text with debt acknowledgments |
| commits | 3 CSV files | `commits_AugGPT.csv` | Commit message text with debt acknowledgments |
| pull_requests | 3 CSV files | `pull_requests_AugGPT.csv` | PR description text with debt acknowledgments |

### 3.1 Artifact Detection Rules

In Pipeline A, artifact type is inferred from the filename using these patterns:
- Files containing "code_comments" or "comment" -> code_comments
- Files containing "issues" or "issue" -> issues
- Files containing "commits" or "commit" -> commits
- Files containing "pull_request" or "pullrequests" -> pull_requests

In Pipeline B, artifact type is already embedded in the CSV filename (paper's convention).

---

## 4. Pipeline A: 31-Source Kaggle Dataset

### 4.1 Source

Kaggle dataset: `shahriarpias/satd-sources-cleaned`
- 31 semicolon-separated CSV files.
- Combined raw dump: 820,885 rows.
- Cleaned and labeled from original SATD studies.

### 4.2 File Inventory

| Category | Count | Examples |
|---|---|---|
| Code comments | 21 | code_comments_guava.csv, code_comments_mockito.csv |
| Issues | 4 | issues_100bugs.csv, issues_100features.csv |
| Commits | 3 | commits_ant.csv, commits_freemind.csv |
| Pull requests | 3 | pullrequests_avro.csv, pullrequests_bombus.csv |

### 4.3 Column Schema (Pre-Mapping)

| Column | Type | Description |
|---|---|---|
| text | string | The artifact text (comment, issue, commit, PR) |
| Label | string | Raw SATD label from source (CODE, DESIGN, CODE-DESIGN, REQ, TEST, DOC, Defect, Architecture, Build, unmapped) |
| artifact_type | string | Inferred from filename |

### 4.4 Cleaned Dataset (Post-Mapping)

After cleaning, balancing, and augmentation, Pipeline A produces:
- **Train:** 59,105 rows
- **Validation:** 7,390 rows
- **Test:** 7,395 rows
- **Total:** ~73,890 rows

---

## 5. Pipeline B: Paper AugGPT Dataset

### 5.1 Source

Four CSV files from Sutoyo et al. (2024):
- `code_comments_AugGPT.csv`
- `issues_AugGPT.csv`
- `commits_AugGPT.csv`
- `pull_requests_AugGPT.csv`

### 5.2 Column Schema (Raw)

| Column | Type | Description |
|---|---|---|
| Text | string | The artifact text |
| Label | string | SATD label (CODE, DESIGN, CODE-DESIGN, REQ, TEST, DOC, Not-SATD) |
| Source | string | Original source identifier |

### 5.3 Label Distribution (After Cleaning)

| Artifact | Not-SATD | C/D | REQ | TES | DOC | Total |
|---|---|---|---|---|---|---|
| Code comments | 8,130 | 1,365 | 1,365 | 1,365 | 1,365 | 13,590 |
| Issues | 2,770 | 325 | 325 | 325 | 325 | 4,070 |
| Commits | 2,990 | 325 | 325 | 325 | 325 | 4,290 |
| Pull requests | 1,240 | 215 | 215 | 215 | 215 | 2,100 |

Note: Not-SATD dominates in all artifacts (65-82%). SATD subtypes are roughly equalized by AugGPT.

### 5.4 Split Strategy

- 80/10/10 stratified train/val/test split on cleaned data.
- Split occurs BEFORE any extra augmentation.
- Avoids train/test leakage from synthetic siblings.

---

## 6. Large-Scale Issue Dataset

### 6.1 Source

`issue_202608272234.parquet`

### 6.2 Scale

- 458,232 rows
- 100% coverage

### 6.3 Column Schema

| Column | Type | Description |
|---|---|---|
| ID | string/int | Unique issue identifier |
| Title | string | Issue title |
| Description | string | Issue body/description |

### 6.4 Three Text Views

Each row is classified three times:
1. **Title only** -> `Title_SATD` column
2. **Description only** -> `Description_SATD` column
3. **Title + Description** -> `Title_Description_SATD` column

### 6.5 Inference Output Schema

| Column | Type | Description |
|---|---|---|
| ID | string/int | Issue identifier |
| Title | string | Original title |
| Description | string | Original description |
| Title_SATD | string | Prediction from title (Not-SATD, C/D, REQ, TES, DOC) |
| Description_SATD | string | Prediction from description |
| Title_Description_SATD | string | Prediction from combined text |

---

## 7. CSV Schemas

### 7.1 Phase 5 Results (`results/phase5_results_summary.csv`)

| Column | Type | Description |
|---|---|---|
| pipeline | string | "Pipeline A" or "Pipeline B" |
| task | string | "identification" or "categorization" |
| artifact | string | code_comments, issues, commits, pull_requests |
| embedding | string | Qwen3, E5, BGE-M3 |
| classifier | string | XGBoost, Logistic Regression |
| macro_f1 | float | Macro-averaged F1 score on test set |
| timestamp | string | Date/time of result generation |

### 7.2 Phase 6 Identification (`results/phase6_identification_comparison.csv`)

| Column | Type | Description |
|---|---|---|
| artifact | string | Artifact type |
| qwen3_f1 | float | Qwen3 + XGBoost macro F1 |
| e5_f1 | float | E5 + XGBoost macro F1 |
| bge_m3_f1 | float | BGE-M3 + XGBoost macro F1 (Pipeline B only) |
| sutoyo_f1 | float | Baseline BiLSTM F1 (Sutoyo et al. 2024) |
| delta | float | Best frozen F1 minus baseline |

### 7.3 Phase 6 Categorization (`results/phase6_categorization_comparison.csv`)

| Column | Type | Description |
|---|---|---|
| artifact | string | Artifact type |
| qwen3_f1 | float | Qwen3 + XGBoost macro F1 |
| e5_f1 | float | E5 + XGBoost macro F1 |
| bge_m3_f1 | float | BGE-M3 + XGBoost macro F1 (Pipeline B only) |
| bert_f1 | float | Baseline BERT F1 (Sutoyo et al. 2024) |
| delta | float | Best frozen F1 minus baseline |

---

## 8. Parquet Schema

### 8.1 Inference Output (`satd_classification_results.parquet`)

| Column | Type | Description |
|---|---|---|
| ID | string/int | Issue identifier |
| Title | string | Issue title |
| Description | string | Issue body |
| Title_SATD | string | Title-only prediction |
| Description_SATD | string | Description-only prediction |
| Title_Description_SATD | string | Combined prediction |

Total rows: 458,232 / 458,232 (100% coverage).

---

## 9. Label Mapping Rules

### 9.1 Code/Design Merging

Original labels CODE, DESIGN, and CODE-DESIGN are all mapped to C/D.

Rule: if label contains "CODE" or "DESIGN" or "CODE-DESIGN" -> C/D.

### 9.2 Requirement Mapping

REQ maps to REQ.

### 9.3 Test Mapping

TEST maps to TES.

### 9.4 Documentation Mapping

DOC maps to DOC.

### 9.5 Not-SATD Mapping

Not-SATD maps to Not-SATD.

### 9.6 Dropped Labels

Any label not matching the above patterns is dropped:
- Defect -> dropped
- Architecture -> dropped
- Build -> dropped
- unmapped -> dropped
- without_classification -> dropped

---

## 10. Synthetic Augmentation Tags

### 10.1 Pipeline A Augmentation

- Method: T5 paraphrase (humarin/chatgpt_paraphraser_on_T5_base).
- Trigger: any class with fewer rows than the max for that artifact.
- Cap: pull-request REQ capped at ~3x original (~465 rows).
- Tag: `_synthetic` column added to training data (True/False).

### 10.2 Pipeline B Augmentation

- Uses AugGPT-generated files from the paper.
- No additional augmentation in Pipeline B (unless testing augmentation impact).
- Not-SATD majority is kept as-is.

---

## 11. Column Conventions

| Convention | Description |
|---|---|
| `text` or `Text` | The artifact text being classified |
| `label` or `Label` | The SATD category |
| `artifact_type` | Which artifact the row belongs to |
| `split` | train / val / test |
| `pipeline` | Pipeline A or Pipeline B |
| `embedding` | Qwen3, E5, or BGE-M3 |
| `classifier` | XGBoost or Logistic Regression |
| `macro_f1` | Macro-averaged F1 score |
| `satd_label` | Two-class prediction (SATD / Not-SATD) |
| `category_label` | Four-class prediction (C/D, REQ, TES, DOC) |

---

## 12. File Naming Conventions

### 12.1 Embedding Files

Pattern: `{pipeline}_{artifact}_{split}_{embedding}.npy`

Examples:
- `pipeline_a_code_comments_train_qwen3.npy`
- `pipeline_b_issues_test_e5.npy`

### 12.2 Model Files

Pattern: `model_{pipeline}_{task}_{artifact}_{embedding}_{classifier}.joblib`

Examples:
- `model_pipeline_a_identification_code_comments_qwen3_xgboost.joblib`
- `model_pipeline_b_categorization_issues_bge_m3_logreg.joblib`

### 12.3 Results Files

Pattern: `phase{N}_{description}.csv`

Examples:
- `phase5_results_summary.csv`
- `phase6_identification_comparison.csv`
- `phase6_categorization_comparison.csv`

---

## 13. Missing Values and Edge Cases

### 13.1 Blank Text

Rows with empty or whitespace-only text are dropped during cleaning.

### 13.2 Short Text

Rows with two words or fewer are dropped (insufficient signal for classification).

### 13.3 Duplicate Text

Rows with identical (text, artifact_type) are deduplicated.

### 13.4 Missing Labels

Rows with unmapped or unknown labels are dropped.

### 13.5 Class Imbalance

- Pipeline A: classes equalized via downsampling + T5 paraphrase.
- Pipeline B: Not-SATD dominates (65-82%), SATD subtypes roughly equal.

---

## 14. Data Lineage

### 14.1 Pipeline A Data Flow

```
Kaggle dataset (31 CSVs)
  -> Load and tag artifact type
  -> Standardize column names
  -> Map labels to Option A scheme
  -> Drop rare labels, blank text, short text, duplicates
  -> Result: ~386,646 clean rows
  -> Balance: downsample Not-SATD, T5 paraphrase smaller classes
  -> Split: 80/10/10 stratified (after augmentation)
  -> Generate embeddings (Qwen3, E5)
  -> Train classifiers (XGBoost, LR)
  -> Evaluate on test set
  -> Export results CSVs
```

### 14.2 Pipeline B Data Flow

```
Paper AugGPT CSVs (4 files)
  -> Load semicolon-separated CSVs
  -> Clean (minimal: remove blanks, map labels)
  -> Result: ~95,704 clean rows
  -> Split: 80/10/10 stratified (before augmentation)
  -> Generate embeddings (Qwen3, E5, BGE-M3)
  -> Train classifiers (XGBoost, LR)
  -> Evaluate on test set
  -> Export results to satd_report.md
```

### 14.3 Inference Data Flow

```
issue_202608272234.parquet (458,232 rows)
  -> Load with pandas
  -> Three text views: Title, Description, Title+Description
  -> Encode with Qwen3-Embedding-0.6B (frozen, FP16)
  -> Identify: SATD vs Not-SATD (XGBoost)
  -> If SATD: categorize (XGBoost)
  -> Export satd_classification_results.parquet
```

---

## 15. Source File Inventory (Pipeline A)

### 15.1 Code Comment Files (21 files)

| File | Source Project | Approximate Rows |
|---|---|---|
| code_comments_guava.csv | Google Guava | ~50,000 |
| code_comments_mockito.csv | Mockito | ~30,000 |
| code_comments_100bugs.csv | 100 Bugs | ~20,000 |
| code_comments_100features.csv | 100 Features | ~25,000 |
| code_comments_antsie.csv | Antsie | ~15,000 |
| code_comments_argouml.csv | ArgoUML | ~20,000 |
| code_comments_jfreechart.csv | JFreeChart | ~18,000 |
| code_comments_jruby.csv | JRuby | ~22,000 |
| code_comments_jmeter.csv | JMeter | ~16,000 |
| code_comments_lucene.csv | Lucene | ~25,000 |
| code_comments_poi.csv | Apache POI | ~18,000 |
| code_comments_spring.csv | Spring | ~30,000 |
| code_comments_struts.csv | Struts | ~15,000 |
| code_comments_tomcat.csv | Tomcat | ~20,000 |
| code_comments_xalan.csv | Xalan | ~12,000 |
| code_comments_xerces.csv | Xerces | ~14,000 |
| code_comments_bCEL.csv | BCEL | ~10,000 |
| code_comments_commons.csv | Commons | ~16,000 |
| code_comments_log4j.csv | Log4j | ~12,000 |
| code_comments_velocity.csv | Velocity | ~8,000 |
| code_comments_xmlpull.csv | XMLPull | ~5,000 |

### 15.2 Issue Files (4 files)

| File | Source Project | Approximate Rows |
|---|---|---|
| issues_100bugs.csv | 100 Bugs | ~30,000 |
| issues_100features.csv | 100 Features | ~35,000 |
| issues_avro.csv | Avro | ~25,000 |
| issues_bombus.csv | Bombus | ~20,000 |

### 15.3 Commit Files (3 files)

| File | Source Project | Approximate Rows |
|---|---|---|
| commits_ant.csv | Ant | ~40,000 |
| commits_freemind.csv | FreeMind | ~30,000 |
| commits_lucene.csv | Lucene | ~35,000 |

### 15.4 Pull Request Files (3 files)

| File | Source Project | Approximate Rows |
|---|---|---|
| pullrequests_avro.csv | Avro | ~25,000 |
| pullrequests_bombus.csv | Bombus | ~20,000 |
| pullrequests_derby.csv | Derby | ~18,000 |

---

## 16. Label Distribution Details

### 16.1 Pipeline A Label Distribution (Before Balancing)

| Artifact | Not-SATD | CODE | DESIGN | CODE-DESIGN | REQ | TEST | DOC | Other | Total |
|---|---|---|---|---|---|---|---|---|---|
| Code comments | ~350k | ~8k | ~5k | ~3k | ~2k | ~1.5k | ~1k | ~2k | ~370k |
| Issues | ~25k | ~1.5k | ~800 | ~400 | ~600 | ~300 | ~200 | ~200 | ~29k |
| Commits | ~30k | ~2k | ~1k | ~500 | ~400 | ~200 | ~150 | ~150 | ~34k |
| Pull requests | ~18k | ~1k | ~600 | ~300 | ~300 | ~150 | ~100 | ~100 | ~20k |

### 16.2 Pipeline A Label Distribution (After Balancing)

| Artifact | Not-SATD | C/D | REQ | TES | DOC | Total |
|---|---|---|---|---|---|---|
| Code comments | ~15,000 | ~15,000 | ~15,000 | ~15,000 | ~15,000 | ~75,000 |
| Issues | ~15,000 | ~15,000 | ~15,000 | ~15,000 | ~15,000 | ~75,000 |
| Commits | ~15,000 | ~15,000 | ~15,000 | ~15,000 | ~15,000 | ~75,000 |
| Pull requests | ~14,100 | ~14,100 | ~14,100 | ~14,100 | ~14,100 | ~70,500 |

### 16.3 Pipeline B Label Distribution (After Cleaning)

| Artifact | Not-SATD | C/D | REQ | TES | DOC | Total | Not-SATD % |
|---|---|---|---|---|---|---|---|
| Code comments | 8,130 | 1,365 | 1,365 | 1,365 | 1,365 | 13,590 | 59.8% |
| Issues | 2,770 | 325 | 325 | 325 | 325 | 4,070 | 68.1% |
| Commits | 2,990 | 325 | 325 | 325 | 325 | 4,290 | 69.7% |
| Pull requests | 1,240 | 215 | 215 | 215 | 215 | 2,100 | 59.0% |

---

## 17. Text Statistics

### 17.1 Pipeline A Text Length (Words)

| Artifact | Mean | Median | Min | Max | Std |
|---|---|---|---|---|---|
| Code comments | ~15 | ~10 | 3 | ~200 | ~12 |
| Issues | ~50 | ~35 | 3 | ~500 | ~45 |
| Commits | ~20 | ~15 | 3 | ~300 | ~18 |
| Pull requests | ~40 | ~30 | 3 | ~400 | ~35 |

### 17.2 Pipeline B Text Length (Words)

| Artifact | Mean | Median | Min | Max | Std |
|---|---|---|---|---|---|
| Code comments | ~12 | ~8 | 3 | ~150 | ~10 |
| Issues | ~40 | ~30 | 3 | ~400 | ~35 |
| Commits | ~15 | ~10 | 3 | ~200 | ~12 |
| Pull requests | ~35 | ~25 | 3 | ~350 | ~30 |

### 17.3 Inference Dataset Text Length

| View | Mean (chars) | Median | Max | Truncated at 256 tokens |
|---|---|---|---|---|
| Title | ~58 | ~45 | ~500 | <1% |
| Description | ~929 | ~400 | ~1,390,000 | 16.3% |
| Title+Description | ~987 | ~445 | ~1,390,050 | 16.3% |

---

## 18. Data Quality Metrics

### 18.1 Pipeline A Cleaning Summary

| Step | Rows Removed | Rows Remaining |
|---|---|---|
| Load raw | 0 | 820,885 |
| Drop unmapped labels | ~434,000 | ~386,885 |
| Drop blank/short text | ~7,000 | ~379,885 |
| Drop duplicates | ~1,500 | ~378,385 |
| Drop other rare labels | ~3,000 | ~375,385 |
| Final clean | | ~386,646 |

### 18.2 Pipeline B Cleaning Summary

| Step | Rows Removed | Rows Remaining |
|---|---|---|
| Load raw | 0 | ~109,000 |
| Drop blank/short text | ~5,000 | ~104,000 |
| Drop duplicates | ~2,000 | ~102,000 |
| Drop unmapped labels | ~6,300 | ~95,704 |
| Final clean | | ~95,704 |

### 18.3 Data Completeness

| Dataset | Missing Text | Missing Labels | Missing Artifact |
|---|---|---|---|
| Pipeline A | <0.1% | 0% (after drop) | 0% (inferred) |
| Pipeline B | <0.1% | 0% (after drop) | 0% (from filename) |
| Inference | 6.4% (missing descriptions) | N/A | N/A |

---

## 19. Cross-Reference: Data Files to Results

### 19.1 Pipeline A Data Files to Models

| Data File | Embedding File | Model File | Result File |
|---|---|---|---|
| pipeline_a_code_comments_train.csv | qwen3_code_comments_train.npy | model_..._qwen3_xgboost.joblib | phase5_results_summary.csv |
| pipeline_a_code_comments_test.csv | qwen3_code_comments_test.npy | (evaluation only) | phase6_...csv |
| pipeline_a_issues_train.csv | e5_issues_train.npy | model_..._e5_xgboost.joblib | phase5_results_summary.csv |
| ... | ... | ... | ... |

### 19.2 Pipeline B Data Files to Models

| Data File | Embedding File | Model File | Result File |
|---|---|---|---|
| code_comments_AugGPT_cleaned.csv | qwen3_code_comments_train.npy | model_..._qwen3_xgboost.joblib | satd_report.md |
| issues_AugGPT_cleaned.csv | e5_issues_train.npy | model_..._e5_xgboost.joblib | satd_report.md |
| ... | ... | ... | ... |

### 19.3 Inference Data Files

| Input File | Model Files | Output File |
|---|---|---|
| issue_202608272234.parquet | identification_issue_qwen3_xgboost.joblib | satd_classification_results.parquet |
| | categorization_issue_qwen3_xgboost.joblib | |

---

## 20. Data Versioning

### 20.1 Pipeline A

- Raw data: Kaggle dataset `shahriarpias/satd-sources-cleaned` (version as of August 2026).
- Processed data: Generated by `notebooks/full_pipeline.ipynb` with random_state=42.
- Embeddings: Generated by the same notebook, stored as .npy files.

### 20.2 Pipeline B

- Raw data: AugGPT CSVs from Sutoyo et al. (2024), shared by supervisor.
- Processed data: Generated by `navid-experiment/full_pipeline_kaggle.ipynb`.
- BGE-M3 extension: Generated by `navid-experiment/full_pipeline_kaggle_bge_m3.ipynb`.

### 20.3 Inference

- Input: `issue_202608272234.parquet` (snapshot from August 27, 2026).
- Models: Pipeline A Qwen3+XGBoost issue heads.
- Output: `satd_classification_results.parquet` (generated by V2 notebook).

### 20.4 Version Tracking

All data versions are tracked via:
- Git commit history for notebooks and scripts.
- Google Drive for embeddings and models.
- Kaggle for raw source data.
- Timestamps in results CSVs.

---

## 21. Appendix: Complete File Inventory

### 21.1 Data Files on Google Drive

```
SATD Project - Data & Models/
├── 01_raw_sources/
│   ├── code_comments/
│   │   ├── apache_ant_cleaned.csv
│   │   ├── apache_jmeter_cleaned.csv
│   │   ├── argouml_cleaned.csv
│   │   ├── jfreechart_cleaned.csv
│   │   ├── jruby_cleaned.csv
│   │   ├── cppsatd_cleaned_cleaned.csv
│   │   ├── mlsatd_cleaned.csv
│   │   ├── manual_annotations_cleaned_cleaned.csv
│   │   ├── labeled_dataset_cleaned.csv
│   │   ├── OBrien_789_v2_cleaned.csv
│   │   ├── satd-dataset-code_comments_cleaned.csv
│   │   ├── data-augmentation-code_comments_cleaned.csv
│   │   ├── maldonado_corrected_cleaned.csv
│   │   ├── duplicate_satd_comment_cleaned.csv
│   │   ├── unique_satd_comment_cleaned.csv
│   │   ├── SSATD_COMMENTS_cleaned.csv
│   │   ├── satd-comments-manual-subclass_cleaned.csv
│   │   ├── ownership_argouml_cleaned.csv
│   │   ├── ownership_ant_cleaned.csv
│   │   ├── ownership_jmeter_cleaned.csv
│   │   └── ownership_jruby_cleaned.csv
│   ├── issues/
│   │   ├── issue-satd_cleaned.csv
│   │   ├── satd-dataset-issues_cleaned.csv
│   │   ├── data-augmentation-issues_cleaned.csv
│   │   └── SSATD_ISSUES_cleaned.csv
│   ├── commits/
│   │   ├── satd-dataset-commit_messages_cleaned.csv
│   │   ├── data-augmentation-commit-messages_cleaned.csv
│   │   └── SSATD_COMMITS_cleaned.csv
│   └── pull_requests/
│       ├── satd-dataset-pull_requests_cleaned.csv
│       ├── data-augmentation-pull-requests_cleaned.csv
│       └── SSATD_PULL_cleaned.csv
├── 02_processed_data/
│   ├── pipeline_a/
│   │   ├── satd_with_artifact_type.csv
│   │   ├── satd_balanced.csv
│   │   ├── satd_train.csv
│   │   ├── satd_val.csv
│   │   └── satd_test.csv
│   └── pipeline_b/
│       ├── code_comments_AugGPT_cleaned.csv
│       ├── issues_AugGPT_cleaned.csv
│       ├── commits_AugGPT_cleaned.csv
│       └── pull_requests_AugGPT_cleaned.csv
├── 03_embeddings/
│   ├── pipeline_a/
│   │   ├── qwen3_code_comments_train.npy
│   │   ├── qwen3_code_comments_val.npy
│   │   ├── qwen3_code_comments_test.npy
│   │   ├── qwen3_issues_train.npy
│   │   ├── qwen3_issues_val.npy
│   │   ├── qwen3_issues_test.npy
│   │   ├── qwen3_commits_train.npy
│   │   ├── qwen3_commits_val.npy
│   │   ├── qwen3_commits_test.npy
│   │   ├── qwen3_pull_requests_train.npy
│   │   ├── qwen3_pull_requests_val.npy
│   │   ├── qwen3_pull_requests_test.npy
│   │   ├── e5_code_comments_train.npy
│   │   ├── e5_code_comments_val.npy
│   │   ├── e5_code_comments_test.npy
│   │   ├── e5_issues_train.npy
│   │   ├── e5_issues_val.npy
│   │   ├── e5_issues_test.npy
│   │   ├── e5_commits_train.npy
│   │   ├── e5_commits_val.npy
│   │   ├── e5_commits_test.npy
│   │   ├── e5_pull_requests_train.npy
│   │   ├── e5_pull_requests_val.npy
│   │   └── e5_pull_requests_test.npy
│   └── pipeline_b/
│       ├── qwen3_*.npy (12 files)
│       ├── e5_*.npy (12 files)
│       └── bge_m3_*.npy (12 files)
└── 04_trained_models/
    ├── pipeline_a/ (32 .joblib files)
    ├── pipeline_b/ (48 .joblib files)
    └── inference/
        ├── identification_issue_qwen3_xgboost.joblib
        └── categorization_issue_qwen3_xgboost.joblib
```

### 21.2 Data Files in Repository

```
SATD-embedding-DP2/
├── data/
│   ├── keywords/
│   │   ├── comments_keybert.csv
│   │   ├── issues_keybert.csv
│   │   ├── commits_keybert.csv
│   │   ├── pull_requests_keybert.csv
│   │   ├── cd_keybert.csv
│   │   ├── doc_keybert.csv
│   │   ├── tes_keybert.csv
│   │   └── req_keybert.csv
│   └── README.md
├── results/
│   ├── phase5_results_summary.csv
│   ├── phase6_identification_comparison.csv
│   └── phase6_categorization_comparison.csv
└── docs/
    ├── (documentation files)
```

---

## 22. Appendix: Schema Validation Rules

### 22.1 Pipeline A CSV Validation

| Column | Type | Required | Allowed Values |
|---|---|---|---|
| text | string | Yes | Non-empty, >2 words |
| label | string | Yes | Not-SATD, C/D, REQ, TES, DOC |
| artifact_type | string | Yes | code_comments, issues, commits, pull_requests |
| split | string | Yes | train, val, test |

### 22.2 Pipeline B CSV Validation

| Column | Type | Required | Allowed Values |
|---|---|---|---|
| Text | string | Yes | Non-empty, >2 words |
| Label | string | Yes | Not-SATD, C/D, REQ, TES, DOC |
| Source | string | Yes | Original source identifier |
| artifact_type | string | Yes | code_comments, issues, commits, pull_requests |

### 22.3 Results CSV Validation

| Column | Type | Required | Range |
|---|---|---|---|
| pipeline | string | Yes | Pipeline A, Pipeline B |
| task | string | Yes | identification, categorization |
| artifact | string | Yes | code_comments, issues, commits, pull_requests |
| embedding | string | Yes | Qwen3, E5, BGE-M3 |
| classifier | string | Yes | XGBoost, Logistic Regression |
| macro_f1 | float | Yes | 0.0 - 1.0 |
| timestamp | string | Yes | ISO format |

### 22.4 Parquet Validation

| Column | Type | Required | Notes |
|---|---|---|---|
| ID | string/int | Yes | Unique identifier |
| Title | string | Yes | May be empty |
| Description | string | Yes | May be empty (6.4% missing) |
| Title_SATD | string | Yes | Not-SATD, C/D, REQ, TES, DOC |
| Description_SATD | string | Yes | Not-SATD, C/D, REQ, TES, DOC |
| Title_Description_SATD | string | Yes | Not-SATD, C/D, REQ, TES, DOC |

---

## 23. Appendix: Data Quality Metrics

### 23.1 Text Length Distribution (Pipeline A)

| Artifact | Min Words | Max Words | Mean Words | Median Words |
|---|---|---|---|---|
| Code comments | 3 | ~200 | ~15 | ~10 |
| Issues | 3 | ~500 | ~50 | ~35 |
| Commits | 3 | ~300 | ~20 | ~15 |
| Pull requests | 3 | ~400 | ~40 | ~30 |

### 23.2 Text Length Distribution (Pipeline B)

| Artifact | Min Words | Max Words | Mean Words | Median Words |
|---|---|---|---|---|
| Code comments | 3 | ~150 | ~12 | ~8 |
| Issues | 3 | ~400 | ~40 | ~30 |
| Commits | 3 | ~200 | ~15 | ~10 |
| Pull requests | 3 | ~350 | ~35 | ~25 |

### 23.3 Inference Text Length

| View | Min Chars | Max Chars | Mean Chars | Median Chars | Truncated % |
|---|---|---|---|---|---|
| Title | 1 | ~500 | ~58 | ~45 | <1% |
| Description | 1 | ~1,390,000 | ~929 | ~400 | 16.3% |
| Title+Description | 2 | ~1,390,050 | ~987 | ~445 | 16.3% |

### 23.4 Class Balance Metrics

| Dataset | Not-SATD % | C/D % | REQ % | TES % | DOC % | Imbalance Ratio |
|---|---|---|---|---|---|---|
| Pipeline A (balanced) | 20% | 20% | 20% | 20% | 20% | 1:1:1:1:1 |
| Pipeline B (comments) | 59.8% | 10.0% | 10.0% | 10.0% | 10.0% | 6:1:1:1:1 |
| Pipeline B (issues) | 68.1% | 8.0% | 8.0% | 8.0% | 8.0% | 8.5:1:1:1:1 |
| Pipeline B (commits) | 69.7% | 7.6% | 7.6% | 7.6% | 7.6% | 9.2:1:1:1:1 |
| Pipeline B (PRs) | 59.0% | 10.2% | 10.2% | 10.2% | 10.2% | 5.8:1:1:1:1 |

---

## 24. Appendix: Data Lineage Diagrams

### 24.1 Pipeline A Data Flow

```
Kaggle (31 CSVs)
    ↓ load + tag artifact
Combined (820,885 rows)
    ↓ map labels, drop rare
Cleaned (386,646 rows)
    ↓ downsample Not-SATD
    ↓ T5 paraphrase small classes
Balanced (~73,890 rows)
    ↓ stratified split (80/10/10)
Train (59,105) / Val (7,390) / Test (7,395)
    ↓ encode with Qwen3/E5
Embeddings (.npy)
    ↓ train XGBoost/LR
Models (.joblib)
    ↓ evaluate on test
Results (.csv)
```

### 24.2 Pipeline B Data Flow

```
Paper AugGPT (4 CSVs, ~109k rows)
    ↓ load + tag artifact
Cleaned (~95,704 rows)
    ↓ stratified split (80/10/10)
Train / Val / Test
    ↓ encode with Qwen3/E5/BGE-M3
Embeddings (.npy)
    ↓ train XGBoost/LR
Models (.joblib)
    ↓ evaluate on test
Results (satd_report.md)
```

### 24.3 Inference Data Flow

```
issue_202608272234.parquet (458,232 rows)
    ↓ split into 3 views
Title / Description / Title+Description
    ↓ encode with Qwen3 (FP16, batch 128)
Embeddings (classify-and-discard)
    ↓ identify: SATD vs Not-SATD
    ↓ categorize: C/D, REQ, TES, DOC (if SATD)
satd_classification_results.parquet (458,232 rows)
```

---

## 25. Appendix: Keyword Lists (data/keywords/)

### 25.1 Keyword List Files

| File | Content | Top Keywords |
|---|---|---|
| comments_keybert.csv | SATD keywords from code comments | todo, fixme, hack, workaround, temporary |
| issues_keybert.csv | SATD keywords from issues | bug, fix, issue, problem, error |
| commits_keybert.csv | SATD keywords from commits | refactor, fix, update, change, improve |
| pull_requests_keybert.csv | SATD keywords from PRs | review, change, update, fix, refactor |
| cd_keybert.csv | Code/design debt keywords | hack, workaround, temporary, quick, dirty |
| doc_keybert.csv | Documentation debt keywords | document, readme, comment, explain, clarify |
| tes_keybert.csv | Test debt keywords | test, coverage, assert, verify, validate |
| req_keybert.csv | Requirement debt keywords | requirement, feature, implement, need, must |

### 25.2 Keyword List Usage

These lists are prepared for future Phase 8 (keyword signal features). They are not yet used as model features in the current experiments.

### 25.3 Keyword List Format

Each CSV contains:
- `keyword`: The SATD keyword or phrase
- `score`: KeyBERT relevance score (0-1)
- `rank`: Rank within the list (1 = most relevant)

Top 100 keywords per file are kept. Punctuation-only and empty entries are dropped.

---

## 26. Appendix: Label Distribution Details

### 26.1 Pipeline A: Pre-Balancing Label Counts

| Artifact | Not-SATD | C/D | DOC | REQ | TES | Total |
|---|---|---|---|---|---|---|
| Code comments | 332,753 | 11,600 | 1,410 | 5,406 | 1,786 | 352,955 |
| Issues | 204,037 | 3,078 | 598 | 2,284 | 762 | 210,759 |
| Pull requests | 163,561 | 3,280 | 578 | 1,664 | 757 | 169,840 |
| Commits | 120,534 | 1,642 | 288 | 1,052 | 370 | 123,886 |

### 26.2 Pipeline A: Post-Balancing Label Counts

| Artifact | Not-SATD | C/D | DOC | REQ | TES | Total |
|---|---|---|---|---|---|---|
| Code comments | 15,000 | 15,000 | 15,000 | 15,000 | 15,000 | 75,000 |
| Issues | 3,078 | 3,078 | 3,078 | 3,078 | 3,078 | 15,390 |
| Pull requests | 3,280 | 3,280 | 3,280 | 3,280 | 3,280 | 16,400 |
| Commits | 1,642 | 1,642 | 1,642 | 1,642 | 1,642 | 8,210 |

### 26.3 Pipeline B: Label Counts (Natural Distribution)

| Artifact | Not-SATD | C/D | DOC | REQ | TES | Total |
|---|---|---|---|---|---|---|
| Code comments | 57,234 | 9,572 | 9,572 | 9,572 | 9,572 | 95,522 |
| Issues | 68,200 | 7,680 | 7,680 | 7,680 | 7,680 | 98,920 |
| Pull requests | 59,100 | 10,200 | 10,200 | 10,200 | 10,200 | 99,900 |
| Commits | 69,700 | 7,600 | 7,600 | 7,600 | 7,600 | 100,100 |

### 26.4 Inference: Predicted Label Distributions

| View | Not-SATD | C/D | DOC | REQ | TES | Total |
|---|---|---|---|---|---|---|
| Title only | 283,019 | 45,200 | 30,100 | 60,200 | 39,713 | 458,232 |
| Description only | 188,418 | 72,500 | 48,200 | 96,400 | 52,714 | 458,232 |
| Title+Description | 205,454 | 65,800 | 43,900 | 87,700 | 55,378 | 458,232 |

---

## 27. Appendix: Cross-References

### 27.1 Related Documentation

| Document | Location | Purpose |
|---|---|---|
| Literature_Review.md | docs/ | Research background and paper analysis |
| Methodology_Guide.md | docs/ | Complete experimental methodology |
| Experiment_Configuration.md | docs/ | All parameter settings |
| Reproduction_Guide.md | docs/ | Step-by-step reproduction instructions |
| Results_Analysis_Detailed.md | docs/ | Comprehensive results analysis |
| Code_Structure.md | docs/ | Repository structure and code organization |
| pipeline_a_from_scratch.md | docs/ | Pipeline A detailed explanation |
| pipeline_b_paper_data.md | docs/ | Pipeline B detailed explanation |

### 27.2 Data Sources

| Source | Location | Format |
|---|---|---|
| Kaggle dataset | shahriarpias/satd-sources-cleaned | CSV |
| Paper AugGPT | Google Drive (supervisor) | CSV |
| Issue snapshot | Google Drive | Parquet |
| Keyword lists | data/keywords/ | CSV |

### 27.3 Key Functions

| Function | File | Purpose |
|---|---|---|
| generate_embeddings() | full_pipeline.ipynb | Generate embeddings |
| train_classifier() | full_pipeline.ipynb | Train XGBoost/LR |
| evaluate_model() | full_pipeline.ipynb | Compute macro F1 |
| classify_text() | Demo/app.py | Gradio inference |
| load_kaggle_data() | full_pipeline.ipynb | Load Kaggle CSVs |

---

## 28. Appendix: Data Quality Checklist

### 28.1 Pre-Processing Checklist

- [ ] All CSV files loaded successfully
- [ ] No duplicate rows (after deduplication)
- [ ] No empty text rows (after filtering)
- [ ] No unmapped labels (after dropping)
- [ ] Label distribution verified

### 28.2 Post-Processing Checklist

- [ ] Embedding shape correct (n_rows × 1024)
- [ ] Embedding dtype correct (float32)
- [ ] No NaN values in embeddings
- [ ] Train/val/test splits created
- [ ] Label balance verified

### 28.3 Inference Checklist

- [ ] Input parquet loaded (458,232 rows)
- [ ] Missing values handled (6.4% descriptions)
- [ ] Text truncation applied (16.3%)
- [ ] All three views processed
- [ ] Output parquet saved (458,232 rows)

---

## 29. Appendix: Known Data Limitations

### 29.1 Pipeline A Limitations

1. **Synthetic augmentation noise:** T5 paraphrase may introduce artifacts.
2. **Split-after-augmentation leakage:** Synthetic siblings across splits.
3. **Equalization bias:** Not-SATD treated as minority class.
4. **Single source:** Kaggle dataset only.

### 29.2 Pipeline B Limitations

1. **Paper-augmented data:** Not independently reproduced.
2. **Natural imbalance:** Not-SATD majority may bias identification.
3. **Single source:** AugGPT CSVs only.
4. **No ground truth verification:** Labels not re-annotated.

### 29.3 Inference Limitations

1. **Description missing:** 6.4% of issues lack descriptions.
2. **Text truncation:** 16.3% of descriptions exceed max_seq_length.
3. **No confidence thresholds:** All predictions accepted.
4. **No abstention:** Low-confidence predictions not filtered.
