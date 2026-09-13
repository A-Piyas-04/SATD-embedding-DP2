# Reproduction Guide

**SATD Embedding Comparison Project**
**Team: Shahriar Pias, Navid Ibrahim, Musaddiq Rafi**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Environment Setup](#3-environment-setup)
4. [Data Acquisition](#4-data-acquisition)
5. [Pipeline A Reproduction](#5-pipeline-a-reproduction)
6. [Pipeline B Reproduction](#6-pipeline-b-reproduction)
7. [Embedding Generation](#7-embedding-generation)
8. [Classifier Training](#8-classifier-training)
9. [Evaluation](#9-evaluation)
10. [Large-Scale Inference](#10-large-scale-inference)
11. [Demo System](#11-demo-system)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Overview

This guide provides step-by-step instructions for reproducing all results in the SATD Embedding Comparison project. The project contains two pipelines with different data protocols, 80 model combinations, and a large-scale inference deployment.

---

## 2. Prerequisites

### 2.1 Software

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| pip | Latest |
| Git | Latest |
| CUDA | 11.8+ (for GPU) |
| Kaggle Account | Free tier sufficient |

### 2.2 Hardware (Minimum)

| Resource | Pipeline A/B | Inference |
|---|---|---|
| GPU | T4 (16GB) | Any CUDA GPU |
| CPU | 4 vCPU | 4+ vCPU |
| RAM | 16 GB | 8 GB |
| Storage | 10 GB | 5 GB |

### 2.3 Kaggle Setup

1. Create a Kaggle account at kaggle.com.
2. Go to Settings -> API -> Create New API Token.
3. Upload the downloaded `kaggle.json` to your working directory.
4. Enable GPU: Settings -> accelerator -> GPU T4 x2.

---

## 3. Environment Setup

### 3.1 Clone Repository

```bash
git clone https://github.com/ShahriarPias/SATD-embedding-DP2.git
cd SATD-embedding-DP2
```

### 3.2 Install Dependencies

```bash
pip install torch transformers sentence-transformers xgboost scikit-learn pandas numpy joblib gradio
```

### 3.3 Kaggle API Setup (for Pipeline A data)

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/ (Linux) or %USERPROFILE%\.kaggle\ (Windows)
```

---

## 4. Data Acquisition

### 4.1 Pipeline A: Kaggle Dataset

```bash
kaggle datasets download -d shahriarpias/satd-sources-cleaned --unzip
```

This downloads 31 CSV files to the current directory.

### 4.2 Pipeline B: Paper AugGPT Files

Download from the supervisor's shared drive or from the paper's supplementary materials:
- `code_comments_AugGPT.csv`
- `issues_AugGPT.csv`
- `commits_AugGPT.csv`
- `pull_requests_AugGPT.csv`

Place them in a `data/paper_data/` directory.

### 4.3 Large-Scale Issue Dataset

Download `issue_202608272234.parquet` from the shared drive and place it in `data/`.

---

## 5. Pipeline A Reproduction

### 5.1 Open the Notebook

```bash
jupyter notebook notebooks/full_pipeline.ipynb
```

### 5.2 Run All Cells

The notebook is organized in six phases:

**Phase 1 (Cells 1-5):** Load and clean 31 source files.
- Tag artifact type from filename.
- Standardize column names.
- Map labels to Option A scheme.
- Drop rare labels and short text.

**Phase 2 (Cells 6-10):** Balance classes.
- Downsample Not-SATD.
- Augment with T5 paraphrase.
- Cap pull-request REQ.

**Phase 3 (Cells 11-15):** Split data.
- 80/10/10 stratified split.
- Save train/val/test CSVs.

**Phase 4 (Cells 16-25):** Generate embeddings.
- Qwen3-Embedding-0.6B.
- E5-Large-v2.
- Mean-pool + L2 normalize.

**Phase 5 (Cells 26-35):** Train classifiers.
- XGBoost for all combinations.
- Logistic regression ablation.
- Save .joblib files.

**Phase 6 (Cells 36-40):** Evaluate.
- Compute macro F1.
- Compare to baseline.
- Export results CSVs.

### 5.3 Expected Output

- `results/phase5_results_summary.csv`
- `results/phase6_identification_comparison.csv`
- `results/phase6_categorization_comparison.csv`

---

## 6. Pipeline B Reproduction

### 6.1 Open the Notebook

```bash
jupyter notebook navid-experiment/full_pipeline_kaggle.ipynb
```

### 6.2 Run All Cells

The notebook follows the same phase structure as Pipeline A but with different data:

**Phase 1:** Load paper AugGPT CSVs.
**Phase 2:** Clean (no re-balancing, keep paper's distribution).
**Phase 3:** Split BEFORE augmentation.
**Phase 4:** Generate embeddings (Qwen3 + E5).
**Phase 5:** Train classifiers.
**Phase 6:** Evaluate.

### 6.3 BGE-M3 Extension

```bash
jupyter notebook navid-experiment/full_pipeline_kaggle_bge_m3.ipynb
```

This adds BGE-M3 embeddings to Pipeline B results.

### 6.4 Expected Output

- `navid-experiment/satd_report.md` (Qwen3 vs E5)
- `navid-experiment/satd_report_bge.md` (BGE-M3)

---

## 7. Embedding Generation

### 7.1 Qwen3

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                          normalize_embeddings=True)
```

### 7.2 E5

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")
embeddings = model.encode(["query: " + t for t in texts],
                          batch_size=32, show_progress_bar=True,
                          normalize_embeddings=True)
```

### 7.3 BGE-M3

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                          normalize_embeddings=True)
```

### 7.4 Saving

```python
import numpy as np
np.save("qwen3_train.npy", embeddings)
```

---

## 8. Classifier Training

### 8.1 XGBoost

```python
import xgboost as xgb
import joblib

clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    use_label_encoder=False,
    random_state=42
)
clf.fit(X_train, y_train)
joblib.dump(clf, "model.joblib")
```

### 8.2 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
import joblib

clf = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, "model.joblib")
```

### 8.3 Evaluation

```python
from sklearn.metrics import f1_score

y_pred = clf.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"Macro F1: {macro_f1:.3f}")
```

---

## 9. Evaluation

### 9.1 Per-Artifact Results

Run the evaluation cells in either pipeline notebook. Results are saved to:
- `results/phase5_results_summary.csv`
- `results/phase6_identification_comparison.csv`
- `results/phase6_categorization_comparison.csv`

### 9.2 Comparison to Baseline

Compare your macro F1 scores to the Sutoyo et al. (2024) baselines:

**Identification (BiLSTM):**
| Artifact | Baseline F1 |
|---|---|
| Code comments | 0.939 |
| Issues | 0.878 |
| Pull requests | 0.862 |
| Commits | 0.910 |

**Categorization (BERT):**
| Artifact | Baseline F1 |
|---|---|
| Code comments | 0.882 |
| Issues | 0.899 |
| Pull requests | 0.876 |
| Commits | 0.980 |

---

## 10. Large-Scale Inference

### 10.1 Setup

1. Place the Qwen3 issue models in `Demo/models/`:
   - `identification_issue_qwen3_xgboost.joblib`
   - `categorization_issue_qwen3_xgboost.joblib`

2. Ensure `issue_202608272234.parquet` is in `data/`.

### 10.2 Run Inference

```bash
python musaddiq_rafi/label_issues.py
```

### 10.3 Expected Output

- `data/satd_classification_results.parquet` (458,232 rows, 100% coverage)

### 10.4 Duration

- V2 with multi-worker: ~15-30 minutes depending on GPU.
- V1 without optimization: ~45-60 minutes.

---

## 11. Demo System

### 11.1 Setup

```bash
cd Demo
pip install -r requirements.txt
```

### 11.2 Place Models

Copy to `Demo/models/`:
- `identification_issue_qwen3_xgboost.joblib`
- `categorization_issue_qwen3_xgboost.joblib`

### 11.3 Run

```bash
python app.py
```

Open http://127.0.0.1:7860 in your browser.

### 11.4 Features

- Classify text: two-step inference with probability bars.
- Compare views: Title vs Description vs Both.
- Sample sidebar: 10 pre-loaded examples.

---

## 12. Troubleshooting

### 12.1 GPU Not Available

- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- On Kaggle: Enable GPU in notebook settings.
- On local machine: Install CUDA toolkit.

### 12.2 OOM Errors

- Reduce batch_size from 32 to 16.
- Reduce max_seq_length from 512 to 256.
- Use FP16 weights.

### 12.3 File Not Found

- Verify data paths match notebook expectations.
- Check working directory.
- Ensure all CSVs are in the correct location.

### 12.4 Label Mismatch

- Verify label column is standardized.
- Check for uppercase/lowercase differences.
- Ensure all unmapped labels are dropped.

### 12.5 Model Performance Mismatch

- Verify random seeds (42 everywhere).
- Check train/val/test split timing.
- Confirm embedding generation settings match.

---

## 13. Step-by-Step Pipeline A Reproduction

### 13.1 Phase 1: Data Loading

1. Open `notebooks/full_pipeline.ipynb`.
2. Run the Kaggle API cell to download the dataset.
3. Load all 31 CSV files with `pd.read_csv(file, sep=";")`.
4. Tag artifact type from filename using the mapping rules.
5. Standardize column names: `text` for the text column, `Label` for the label column.
6. Print shape after loading: should be ~820,885 rows.

### 13.2 Phase 2: Label Mapping

1. Map labels using the Option A scheme:
   - CODE, DESIGN, CODE-DESIGN -> C/D
   - REQ -> REQ
   - TEST -> TES
   - DOC -> DOC
   - Not-SATD -> Not-SATD
2. Drop Defect, Architecture, Build, unmapped labels.
3. Print label distribution after mapping.
4. Verify that only 5 labels remain: Not-SATD, C/D, REQ, TES, DOC.

### 13.3 Phase 3: Cleaning

1. Drop rows with empty or whitespace-only text.
2. Drop rows with two words or fewer.
3. Drop duplicates on (text, artifact_type).
4. Print shape after cleaning: should be ~386,646 rows.

### 13.4 Phase 4: Balancing

1. For each artifact type:
   - Count rows per class.
   - Find the largest SATD subtype.
   - Downsample Not-SATD to that target.
   - Augment smaller SATD classes with T5 paraphrase.
2. Cap pull-request REQ at 3x original (~465 rows).
3. Print balanced distribution: all 5 classes should be equal per artifact.

### 13.5 Phase 5: Splitting

1. Combine all artifacts into one DataFrame.
2. Split with `train_test_split(test_size=0.1, stratify=df["label"], random_state=42)`.
3. Split the train set again to create validation: `train_test_split(test_size=0.111, ...)`.
4. Verify shapes: ~59,105 train, ~7,390 val, ~7,395 test.

### 13.6 Phase 6: Embedding Generation

1. Load Qwen3 model: `SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")`.
2. Encode train texts: `model.encode(train_texts, batch_size=32, normalize_embeddings=True)`.
3. Save as `qwen3_train.npy`.
4. Repeat for val and test.
5. Repeat with E5 model: `SentenceTransformer("intfloat/e5-large-v2")`.
6. For E5, add "query: " prefix to all texts before encoding.

### 13.7 Phase 7: Classifier Training

1. Load embedding arrays: `X_train = np.load("qwen3_train.npy")`.
2. Load labels: `y_train = train_df["label"].values`.
3. Train XGBoost: `clf.fit(X_train, y_train)`.
4. Evaluate on test set: `y_pred = clf.predict(X_test)`.
5. Compute macro F1: `f1_score(y_test, y_pred, average="macro")`.
6. Save model: `joblib.dump(clf, "model.joblib")`.
7. Repeat for all 32 combinations.

### 13.8 Phase 8: Results Export

1. Collect all F1 scores into a DataFrame.
2. Export to `results/phase5_results_summary.csv`.
3. Compare to baseline scores.
4. Export comparison to `results/phase6_identification_comparison.csv`.
5. Export comparison to `results/phase6_categorization_comparison.csv`.

---

## 14. Step-by-Step Pipeline B Reproduction

### 14.1 Phase 1: Data Loading

1. Open `navid-experiment/full_pipeline_kaggle.ipynb`.
2. Load the four AugGPT CSVs:
   ```python
   code_comments = pd.read_csv("code_comments_AugGPT.csv", sep=";")
   issues = pd.read_csv("issues_AugGPT.csv", sep=";")
   commits = pd.read_csv("commits_AugGPT.csv", sep=";")
   pull_requests = pd.read_csv("pull_requests_AugGPT.csv", sep=";")
   ```
3. Tag artifact type from filename.
4. Print total rows: should be ~109,000.

### 14.2 Phase 2: Cleaning

1. Standardize column names.
2. Map labels to Option A scheme.
3. Drop blank text and duplicates.
4. Print shape after cleaning: should be ~95,704 rows.

### 14.3 Phase 3: Splitting

1. Split BEFORE any augmentation.
2. Use `train_test_split(test_size=0.1, stratify=df["label"], random_state=42)`.
3. Verify Not-SATD majority is preserved (65-82%).

### 14.4 Phase 4: Embedding Generation

1. Same as Pipeline A, but with 3 embeddings (Qwen3, E5, BGE-M3).
2. BGE-M3: `SentenceTransformer("BAAI/bge-m3")`.

### 14.5 Phase 5: Classifier Training

1. Same as Pipeline A.
2. Use C=0.5 for logistic regression.

### 14.6 Phase 6: Results

1. Export results to `navid-experiment/satd_report.md`.
2. Compare to baseline scores.

---

## 15. Step-by-Step BGE-M3 Extension

1. Open `navid-experiment/full_pipeline_kaggle_bge_m3.ipynb`.
2. Follow the same steps as Pipeline B but only for BGE-M3.
3. Generate BGE-M3 embeddings for all artifacts.
4. Train XGBoost and LR classifiers.
5. Export results to `navid-experiment/satd_report_bge.md`.

---

## 16. Step-by-Step Inference Reproduction

### 16.1 Kaggle Setup

1. Go to Kaggle -> New Notebook.
2. Set accelerator to GPU T4 x2.
3. Enable Internet.
4. Add datasets:
   - `issue_202608272234.parquet`
   - `04_trained_models/` folder

### 16.2 Notebook Setup

1. Upload `musaddiq_rafi/satd-classifications-optimized V2.ipynb`.
2. Set `INPUT_PATH` to the parquet file location.
3. Set `MODEL_DIR` to the models folder location.

### 16.3 Run Steps

1. Run the inspection cell to verify input data.
2. Run the pre-download cell to cache the encoder.
3. Run the benchmark cell (4,000 rows) to validate settings.
4. Run the full inference cell.
5. Run the merge cell to combine shards.
6. Run the summary cell to verify output.

### 16.4 Verification

1. Check output row count: should be 458,232.
2. Check coverage: should be 100%.
3. Check label distribution:
   - Title only: ~38% SATD
   - Description only: ~59% SATD
   - Title+Description: ~55% SATD

---

## 17. Step-by-Step Demo Reproduction

### 17.1 Environment Setup

```bash
cd Demo
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 17.2 Model Placement

Copy these files to `Demo/models/`:
- `identification_issue_qwen3_xgboost.joblib`
- `categorization_issue_qwen3_xgboost.joblib`

### 17.3 Run

```bash
python app.py
```

### 17.4 Verify

1. Open http://127.0.0.1:7860.
2. Type "TODO: this is a hack" in the text box.
3. Click "Classify".
4. Verify output shows SATD and C/D category.
5. Check probability bars show confidence.

---

## 18. Verification Checklist

### 18.1 Pipeline A Checklist

- [ ] 31 CSV files loaded
- [ ] ~820,885 raw rows
- [ ] ~386,646 clean rows
- [ ] 5 labels (Not-SATD, C/D, REQ, TES, DOC)
- [ ] Classes equalized per artifact
- [ ] 59,105 train / 7,390 val / 7,395 test
- [ ] Qwen3 embeddings generated (shape: n x 1024)
- [ ] E5 embeddings generated (shape: n x 1024)
- [ ] 32 models trained
- [ ] Results exported to CSVs
- [ ] Baseline comparison completed

### 18.2 Pipeline B Checklist

- [ ] 4 AugGPT CSVs loaded
- [ ] ~109,000 raw rows
- [ ] ~95,704 clean rows
- [ ] Not-SATD majority preserved (65-82%)
- [ ] Split before augmentation
- [ ] Qwen3, E5, BGE-M3 embeddings generated
- [ ] 48 models trained
- [ ] Results exported to reports

### 18.3 Inference Checklist

- [ ] Input: 458,232 rows
- [ ] Models loaded correctly
- [ ] Three text views processed
- [ ] Output: 458,232 rows (100% coverage)
- [ ] Label distribution verified
- [ ] Parquet file saved

---

## 19. Expected Runtime

| Task | Hardware | Expected Time |
|---|---|---|
| Pipeline A embedding generation | Kaggle T4 x2 | ~2 hours |
| Pipeline A classifier training | CPU | ~5 minutes |
| Pipeline B embedding generation | Kaggle T4 x2 | ~3 hours |
| Pipeline B classifier training | CPU | ~8 minutes |
| BGE-M3 extension | Kaggle T4 x2 | ~1.5 hours |
| 458k inference (V2) | Kaggle T4 x2 | ~20 minutes |
| Demo startup | CPU | ~30 seconds |

---

## 20. Quick Reference Commands

```bash
# Clone repo
git clone https://github.com/A-Piyas-04/SATD-embedding-DP2.git
cd SATD-embedding-DP2

# Install dependencies
pip install -r requirements.txt

# Download data
kaggle datasets download -d shahriarpias/satd-sources-cleaned --unzip

# Run Pipeline A
jupyter notebook notebooks/full_pipeline.ipynb

# Run Pipeline B
jupyter notebook navid-experiment/full_pipeline_kaggle.ipynb

# Run Inference (on Kaggle)
jupyter notebook "musaddiq_rafi/satd-classifications-optimized V2.ipynb"

# Run Demo
cd Demo && python app.py
```

---

## 21. Appendix: Expected Outputs

### 21.1 Pipeline A Expected Outputs

**Phase 1 (Cleaning):**
```
Loaded 31 files: 820,885 rows
After label mapping: 820,885 rows
After dropping blanks/short: 813,885 rows
After deduplication: 386,646 rows
```

**Phase 2 (Balancing):**
```
Code comments: 332,753 Not-SATD -> 15,000 (downsampled)
Code comments: 11,600 C/D -> 15,000 (augmented)
Code comments: 5,406 REQ -> 15,000 (augmented)
Code comments: 1,786 TES -> 15,000 (augmented)
Code comments: 1,410 DOC -> 15,000 (augmented)
```

**Phase 3 (Splitting):**
```
Train: 59,105 rows
Val: 7,390 rows
Test: 7,395 rows
```

**Phase 4 (Embeddings):**
```
Qwen3 embeddings generated: (59105, 1024)
E5 embeddings generated: (59105, 1024)
```

**Phase 5 (Training):**
```
Trained 32 models
XGBoost average time: 8 seconds
LogReg average time: 3 seconds
```

**Phase 6 (Evaluation):**
```
Code comments identification: Qwen3+XGB 0.915 vs BiLSTM 0.939 (-0.024)
Code comments categorization: E5+XGB 0.904 vs BERT 0.882 (+0.022)
...
```

### 21.2 Pipeline B Expected Outputs

**Phase 1 (Loading):**
```
Loaded 4 files: 109,000 rows
After cleaning: 95,704 rows
```

**Phase 2 (Splitting):**
```
Train: ~76,563 rows
Val: ~9,570 rows
Test: ~9,571 rows
```

**Phase 3 (Embeddings):**
```
Qwen3 embeddings generated
E5 embeddings generated
BGE-M3 embeddings generated
```

**Phase 4 (Training):**
```
Trained 48 models
```

**Phase 5 (Evaluation):**
```
Code comments identification: Qwen3+XGB 0.966 vs BiLSTM 0.939 (+0.027)
Code comments categorization: E5+XGB 0.958 vs BERT 0.882 (+0.076)
...
```

### 21.3 Inference Expected Outputs

**Inspection:**
```
Total rows: 458,232
Missing descriptions: 29,128 (6.4%)
Mean title length: 58 chars
Mean description length: 929 chars
```

**Benchmark (4,000 rows):**
```
Time: ~2 minutes
Throughput: ~33 rows/sec
Projected full run: ~230 minutes
```

**Full Run:**
```
Shard 1: 229,116 rows classified
Shard 2: 229,116 rows classified
Total: 458,232 rows (100%)
```

**Label Distribution:**
```
Title only: 283,019 Not-SATD, 175,213 SATD (38.2%)
Description only: 188,418 Not-SATD, 269,814 SATD (58.9%)
Title+Description: 205,454 Not-SATD, 252,778 SATD (55.2%)
```

---

## 22. Appendix: Verification Steps

### 22.1 Verify Data Loading

```python
import pandas as pd

# Pipeline A
df = pd.read_csv("satd_with_artifact_type.csv")
print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Artifact types: {df['artifact_type'].unique()}")
print(f"Labels: {df['label'].unique()}")

# Expected: ~386,646 rows
# Expected columns: text, label, artifact_type
# Expected artifacts: code_comments, issues, commits, pull_requests
# Expected labels: Not-SATD, C/D, REQ, TES, DOC
```

### 22.2 Verify Embeddings

```python
import numpy as np

# Load embeddings
qwen3_train = np.load("qwen3_train.npy")
e5_train = np.load("e5_train.npy")

print(f"Qwen3 shape: {qwen3_train.shape}")
print(f"E5 shape: {e5_train.shape}")
print(f"Qwen3 dtype: {qwen3_train.dtype}")
print(f"E5 dtype: {e5_train.dtype}")

# Expected: (59105, 1024)
# Expected dtype: float32
```

### 22.3 Verify Models

```python
import joblib

# Load model
clf = joblib.load("model_pipeline_a_identification_code_comments_qwen3_xgboost.joblib")

print(f"Model type: {type(clf)}")
print(f"Classes: {clf.classes_}")
print(f"n_estimators: {clf.n_estimators}")

# Expected: XGBClassifier
# Expected classes: ['Not-SATD', 'SATD'] for identification
# Expected n_estimators: 200
```

### 22.4 Verify Results

```python
import pandas as pd

# Load results
df = pd.read_csv("phase5_results_summary.csv")

print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Pipelines: {df['pipeline'].unique()}")
print(f"Tasks: {df['task'].unique()}")

# Expected: 80 rows (32 Pipeline A + 48 Pipeline B)
# Expected columns: pipeline, task, artifact, embedding, classifier, macro_f1, timestamp
```

---

## 22. Appendix: Reproducibility Checklist

### 22.1 Environment Reproducibility

- [ ] Python version matches (3.10+)
- [ ] All packages installed with correct versions
- [ ] GPU available and working
- [ ] CUDA version compatible

### 22.2 Data Reproducibility

- [ ] Kaggle dataset downloaded
- [ ] AugGPT files available (Pipeline B)
- [ ] Issue parquet available (inference)
- [ ] All files in correct locations

### 22.3 Experiment Reproducibility

- [ ] random_state=42 used everywhere
- [ ] Same split timing (after/before augmentation)
- [ ] Same embedding settings (batch_size, max_seq_length)
- [ ] Same classifier settings (n_estimators, max_depth)

### 22.4 Results Reproducibility

- [ ] Same macro F1 scores (within rounding)
- [ ] Same label distributions
- [ ] Same file outputs
- [ ] Same directory structure
