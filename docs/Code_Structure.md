# Code Structure Guide

**SATD Embedding Comparison Project**
**Team: Shahriar Pias, Navid Ibrahim, Musaddiq Rafi**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Layout](#2-repository-layout)
3. [Notebook Files](#3-notebook-files)
4. [Script Files](#4-script-files)
5. [Demo Application](#5-demo-application)
6. [Results Files](#6-results-files)
7. [Data Files](#7-data-files)
8. [Documentation Files](#8-documentation-files)
9. [Key Classes and Functions](#9-key-classes-and-functions)
10. [File Dependencies](#10-file-dependencies)
11. [Adding New Components](#11-adding-new-components)

---

## 1. Overview

This document describes the repository structure, key files, and how components relate to each other. It is the authoritative reference for navigating the codebase.

---

## 2. Repository Layout

```
SATD-embedding-DP2/
├── notebooks/                          Pipeline A notebook
│   └── full_pipeline.ipynb
├── navid-experiment/                   Pipeline B notebooks and reports
│   ├── full_pipeline_kaggle.ipynb
│   ├── full_pipeline_kaggle_bge_m3.ipynb
│   ├── paper_experiment.ipynb
│   ├── paper_experiment_2.ipynb
│   ├── paper_experiment_3.ipynb
│   ├── satd_bench.ipynb
│   ├── satd_report.md
│   └── satd_report_bge.md
├── musaddiq_rafi/                      Inference task
│   ├── satd-classifications-optimized V2.ipynb
│   ├── SATD_New_Issue_Inference_V2_Report.pdf
│   ├── label_issues.py
│   ├── README.md
│   └── models/                         Trained .joblib heads (mirrored)
├── Demo/                               Gradio demo application
│   ├── app.py
│   ├── satd_infer.py
│   ├── requirements.txt
│   ├── README.md
│   └── models/                         Local .joblib heads
├── scripts/                            Standalone phase scripts
├── data/                               Data files (not in git)
│   ├── keywords/                       KeyBERT keyword lists
│   └── README.md
├── results/                            Phase 5/6 CSVs and summary PDF
│   ├── phase5_results_summary.csv
│   ├── phase6_identification_comparison.csv
│   └── phase6_categorization_comparison.csv
├── docs/                               Documentation
│   ├── SATD_Research_Progress.md
│   ├── satd_new_issue_inference_v2.md
│   ├── pipeline_a_from_scratch.md
│   ├── pipeline_b_paper_data.md
│   ├── thesis_research_narrative.md
│   ├── Slide-Text.txt
│   └── (additional docs)
├── README.md                           Main project README
├── requirements.txt                    Python dependencies
└── .gitignore
```

---

## 3. Notebook Files

### 3.1 Pipeline A: `notebooks/full_pipeline.ipynb`

**Purpose:** End-to-end Pipeline A implementation.

**Phases:**
1. Load 31 source files from Kaggle.
2. Clean and map labels.
3. Balance classes with T5 paraphrase.
4. Split into train/val/test.
5. Generate embeddings (Qwen3, E5).
6. Train classifiers (XGBoost, logistic regression).
7. Evaluate and export results.

**Dependencies:** Kaggle API, GPU (T4), transformers, xgboost.

**Output:** 32 trained models, embedding .npy files, results CSVs.

### 3.2 Pipeline B: `navid-experiment/full_pipeline_kaggle.ipynb`

**Purpose:** Pipeline B with Qwen3 and E5.

**Key difference from Pipeline A:** Uses paper's AugGPT files, splits before augmentation, keeps Not-SATD majority.

**Dependencies:** GPU (T4), transformers, xgboost.

**Output:** 32 trained models, results in `satd_report.md`.

### 3.3 Pipeline B Extension: `navid-experiment/full_pipeline_kaggle_bge_m3.ipynb`

**Purpose:** Add BGE-M3 embeddings to Pipeline B.

**Dependencies:** GPU (T4), transformers, xgboost.

**Output:** 16 additional models, results in `satd_report_bge.md`.

### 3.4 Early Experiments

- `paper_experiment.ipynb`: First trial, all three embeddings.
- `paper_experiment_2.ipynb`: T5 paraphrase experiment.
- `paper_experiment_3.ipynb`: Final after supervisor revision.
- `satd_bench.ipynb`: Qwen3 vs E5 on Qwen3 validation set.

### 3.5 Inference: `musaddiq_rafi/satd-classifications-optimized V2.ipynb`

**Purpose:** Run Pipeline A Qwen3+XGBoost issue heads over 458,232 unseen issues.

**Optimizations:** Multi-worker, FP16, length-sorted batching, per-batch classify-and-discard, per-shard Parquet checkpoints.

**Output:** `satd_classification_results.parquet` (458,232 rows, 100% coverage).

---

## 4. Script Files

### 4.1 `musaddiq_rafi/label_issues.py`

**Purpose:** Original inference script (V1).

**Usage:** Run standalone or import as module.

**Dependencies:** torch, transformers, joblib, pandas.

**Note:** Superseded by V2 notebook for production runs.

---

## 5. Demo Application

### 5.1 `Demo/app.py`

**Purpose:** Gradio web UI for SATD classification.

**Features:**
- Classify text: two-step inference with probability bars.
- Compare views: Title vs Description vs Both.
- Sample sidebar: 10 pre-loaded examples.

**Dependencies:** gradio, torch, transformers, joblib.

**Usage:**
```bash
cd Demo
python app.py
# Open http://127.0.0.1:7860
```

### 5.2 `Demo/satd_infer.py`

**Purpose:** Inference logic with probabilities and keyword cues.

**Key functions:**
- `classify_text(text)`: Two-step classification.
- `get_probabilities(text)`: Return class probabilities.
- `get_keyword_cues(text)`: Return matching SATD keywords.

**Dependencies:** torch, transformers, joblib.

### 5.3 `Demo/requirements.txt`

**Purpose:** Pin demo dependencies.

**Contents:** gradio, torch, transformers, sentence-transformers, joblib.

### 5.4 `Demo/models/`

**Purpose:** Local storage for trained .joblib heads.

**Required files:**
- `identification_issue_qwen3_xgboost.joblib`
- `categorization_issue_qwen3_xgboost.joblib`

---

## 6. Results Files

### 6.1 `results/phase5_results_summary.csv`

**Purpose:** All macro F1 scores across pipelines, embeddings, classifiers, artifacts, and tasks.

**Schema:** pipeline, task, artifact, embedding, classifier, macro_f1, timestamp.

### 6.2 `results/phase6_identification_comparison.csv`

**Purpose:** Identification results compared to BiLSTM baseline.

**Schema:** artifact, qwen3_f1, e5_f1, bge_m3_f1, sutoyo_f1, delta.

### 6.3 `results/phase6_categorization_comparison.csv`

**Purpose:** Categorization results compared to BERT baseline.

**Schema:** artifact, qwen3_f1, e5_f1, bge_m3_f1, bert_f1, delta.

---

## 7. Data Files

### 7.1 Kaggle Sources (Pipeline A)

- 31 CSV files from `shahriarpias/satd-sources-cleaned`.
- Semicolon-separated.
- Loaded and combined in `full_pipeline.ipynb`.

### 7.2 Paper AugGPT Files (Pipeline B)

- `code_comments_AugGPT.csv`
- `issues_AugGPT.csv`
- `commits_AugGPT.csv`
- `pull_requests_AugGPT.csv`

### 7.3 Issue Dataset (Inference)

- `issue_202608272234.parquet`: 458,232 rows with ID, Title, Description.

### 7.4 Embedding Files

- `{pipeline}_{artifact}_{split}_{embedding}.npy`: Row-aligned embedding vectors.
- Shape: (n_rows, 1024).

### 7.5 Model Files

- `model_{pipeline}_{task}_{artifact}_{embedding}_{classifier}.joblib`: Trained classifier.
- Stored in `musaddiq_rafi/models/` and `Demo/models/`.

---

## 8. Documentation Files

### 8.1 Main README

- `README.md`: Project overview, findings, setup, limitations.

### 8.2 Research Progress

- `docs/SATD_Research_Progress.md`: Full research chronology, RQs, both pipelines, limitations.

### 8.3 Pipeline Documentation

- `docs/pipeline_a_from_scratch.md`: Pipeline A methods and phases.
- `docs/pipeline_b_paper_data.md`: Pipeline B methods.

### 8.4 Inference Report

- `docs/satd_new_issue_inference_v2.md`: 458k inference report.
- `musaddiq_rafi/SATD_New_Issue_Inference_V2_Report.pdf`: Same as PDF.

### 8.5 Narrative

- `docs/thesis_research_narrative.md`: Longer narrative reconstructed from the repo.

### 8.6 Slide Text

- `Slide-Text.txt`: 18-slide presentation text for user-facing slides.

---

## 9. Key Classes and Functions

### 9.1 Embedding Generation

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True)
```

### 9.2 Classifier Training

```python
import xgboost as xgb

clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob"
)
clf.fit(X_train, y_train)
```

### 9.3 Two-Step Inference

```python
import joblib

id_model = joblib.load("identification.joblib")
cat_model = joblib.load("categorization.joblib")

id_pred = id_model.predict(embedding)
if id_pred == "SATD":
    cat_pred = cat_model.predict(embedding)
else:
    cat_pred = "Not-SATD"
```

### 9.4 Demo Inference

```python
from Demo.satd_infer import classify_text

result = classify_text("TODO: this is a hack")
# Returns: {"satd_label": "SATD", "category_label": "C/D", "probabilities": {...}}
```

---

## 10. File Dependencies

### 10.1 Pipeline A Flow

```
Kaggle CSVs -> full_pipeline.ipynb -> embeddings (.npy) -> models (.joblib) -> results (.csv)
```

### 10.2 Pipeline B Flow

```
AugGPT CSVs -> full_pipeline_kaggle.ipynb -> embeddings (.npy) -> models (.joblib) -> satd_report.md
```

### 10.3 Inference Flow

```
issue_202608272234.parquet + models (.joblib) -> V2 notebook -> satd_classification_results.parquet
```

### 10.4 Demo Flow

```
models (.joblib) + Demo/satd_infer.py -> Demo/app.py -> Gradio UI
```

---

## 11. Adding New Components

### 11.1 Adding a New Embedding

1. Add encoding block to `full_pipeline_kaggle.ipynb`.
2. Generate .npy files.
3. Add training blocks for XGBoost and LR.
4. Update results CSVs.
5. Update comparison tables in README.

### 11.2 Adding a New Classifier

1. Add training block to notebook.
2. Evaluate on test set.
3. Export .joblib file.
4. Update results CSVs.

### 11.3 Adding a New Artifact

1. Ensure CSV file is in correct format.
2. Add artifact type inference logic.
3. Update label mapping if needed.
4. Run through full pipeline.

### 11.4 Adding a New Pipeline

1. Create new notebook.
2. Define data loading and cleaning.
3. Define splitting strategy.
4. Run embedding generation and training.
5. Create results report.
6. Update README with new pipeline documentation.

---

## 12. Detailed File Descriptions

### 12.1 `notebooks/full_pipeline.ipynb`

**Cell count:** ~120 cells
**Execution time:** ~3 hours (including embedding generation)
**Output files:** 32 .joblib models, 8 .npy embedding files, 3 results CSVs

**Cell organization:**
- Cells 1-5: Imports and configuration
- Cells 6-15: Data loading and cleaning (Phase 1)
- Cells 16-25: Label mapping and balancing (Phase 2)
- Cells 26-35: Splitting (Phase 3)
- Cells 36-55: Qwen3 embedding generation (Phase 4)
- Cells 56-75: E5 embedding generation (Phase 4)
- Cells 76-95: XGBoost training (Phase 5)
- Cells 96-110: Logistic regression training (Phase 5)
- Cells 111-120: Evaluation and results export (Phase 6)

### 12.2 `navid-experiment/full_pipeline_kaggle.ipynb`

**Cell count:** ~100 cells
**Execution time:** ~4 hours (including 3 embeddings)
**Output files:** 32 .joblib models, 12 .npy embedding files, 1 report

**Cell organization:**
- Cells 1-5: Imports and configuration
- Cells 6-15: Data loading (4 AugGPT CSVs)
- Cells 16-25: Cleaning and label mapping
- Cells 26-35: Splitting
- Cells 36-55: Qwen3 embedding generation
- Cells 56-75: E5 embedding generation
- Cells 76-95: XGBoost and LR training
- Cells 96-100: Evaluation and report export

### 12.3 `navid-experiment/full_pipeline_kaggle_bge_m3.ipynb`

**Cell count:** ~60 cells
**Execution time:** ~1.5 hours
**Output files:** 16 .joblib models, 4 .npy embedding files, 1 report

**Cell organization:**
- Cells 1-5: Imports and configuration
- Cells 6-15: Data loading (same as Pipeline B)
- Cells 16-25: BGE-M3 embedding generation
- Cells 26-45: XGBoost and LR training
- Cells 46-60: Evaluation and report export

### 12.4 `musaddiq_rafi/satd-classifications-optimized V2.ipynb`

**Cell count:** ~40 cells
**Execution time:** ~20 minutes (with V2 optimizations)
**Output files:** 1 Parquet file (458,232 rows)

**Cell organization:**
- Cells 1-5: Imports and configuration
- Cells 6-10: Input inspection and validation
- Cells 11-15: Encoder pre-download and caching
- Cells 16-20: Benchmark (4,000 rows)
- Cells 21-30: Full inference (458k rows)
- Cells 31-35: Shard merging
- Cells 36-40: Summary and verification

### 12.5 `Demo/app.py`

**Lines:** ~150
**Dependencies:** gradio, torch, transformers, joblib, Demo/satd_infer.py

**Key components:**
- `classify_text(text)`: Main classification function
- `get_probabilities(text)`: Returns class probabilities
- `compare_views(title, desc)`: Compares three text views
- Gradio interface with text input, output display, and sample sidebar

### 12.6 `Demo/satd_infer.py`

**Lines:** ~200
**Dependencies:** torch, transformers, joblib, numpy

**Key functions:**
- `load_models()`: Load identification and categorization models
- `encode_text(text)`: Encode text with Qwen3
- `classify_single(text)`: Two-step inference
- `get_keyword_cues(text)`: Match against SATD keyword lists
- `format_output(prediction, probabilities)`: Format for display

---

## 13. API Documentation

### 13.1 Embedding Generation API

```python
from sentence_transformers import SentenceTransformer

def generate_embeddings(
    texts: list[str],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 32,
    max_seq_length: int = 512,
    normalize: bool = True,
    prefix: str = None  # "query: " for E5
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of strings to embed
        model_name: HuggingFace model ID
        batch_size: Number of texts per batch
        max_seq_length: Maximum token length
        normalize: Whether to L2-normalize
        prefix: Optional prefix for each text (e.g., "query: " for E5)
    
    Returns:
        np.ndarray of shape (len(texts), 1024)
    """
    model = SentenceTransformer(model_name)
    if prefix:
        texts = [prefix + t for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize
    )
    return embeddings
```

### 13.2 Classifier Training API

```python
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import joblib

def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier: str = "xgboost",
    **kwargs
) -> object:
    """
    Train a classifier on embedding vectors.
    
    Args:
        X_train: Training embeddings (n_samples, 1024)
        y_train: Training labels
        classifier: "xgboost" or "logreg"
        **kwargs: Additional parameters for the classifier
    
    Returns:
        Trained classifier object
    """
    if classifier == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            objective="multi:softprob",
            random_state=42
        )
    elif classifier == "logreg":
        clf = LogisticRegression(
            max_iter=kwargs.get("max_iter", 1000),
            C=kwargs.get("C", 1.0),
            random_state=42
        )
    clf.fit(X_train, y_train)
    return clf

def save_model(clf, path):
    joblib.dump(clf, path)

def load_model(path):
    return joblib.load(path)
```

### 13.3 Inference API

```python
def classify_text(
    text: str,
    id_model_path: str,
    cat_model_path: str,
    encoder_name: str = "Qwen/Qwen3-Embedding-0.6B"
) -> dict:
    """
    Classify a single text through two-step inference.
    
    Args:
        text: Input text to classify
        id_model_path: Path to identification model
        cat_model_path: Path to categorization model
        encoder_name: HuggingFace encoder model
    
    Returns:
        dict with keys: satd_label, category_label, probabilities
    """
    # Step 1: Encode
    encoder = SentenceTransformer(encoder_name)
    embedding = encoder.encode([text], normalize_embeddings=True)
    
    # Step 2: Identify
    id_model = joblib.load(id_model_path)
    id_pred = id_model.predict(embedding)[0]
    
    # Step 3: Categorize if SATD
    if id_pred == "SATD":
        cat_model = joblib.load(cat_model_path)
        cat_pred = cat_model.predict(embedding)[0]
    else:
        cat_pred = "Not-SATD"
    
    # Step 4: Get probabilities
    id_proba = id_model.predict_proba(embedding)[0]
    cat_proba = cat_model.predict_proba(embedding)[0] if id_pred == "SATD" else None
    
    return {
        "satd_label": id_pred,
        "category_label": cat_pred,
        "id_probabilities": dict(zip(id_model.classes_, id_proba)),
        "cat_probabilities": dict(zip(cat_model.classes_, cat_proba)) if cat_proba else None
    }
```

---

## 14. Code Style and Conventions

### 14.1 Python Style

- Follow PEP 8 for code formatting.
- Use type hints where practical.
- Use descriptive variable names.
- Keep functions focused and short.

### 14.2 Notebook Style

- Use markdown cells for section headers.
- Use code cells for executable code.
- Print shapes and distributions after major operations.
- Include comments for non-obvious logic.

### 14.3 File Naming

- Use lowercase with underscores for Python files.
- Use descriptive names that indicate purpose.
- Include version numbers where relevant (e.g., V2).

### 14.4 Documentation Style

- Use markdown for all documentation files.
- Include tables for structured data.
- Use code blocks for examples.
- Keep paragraphs short and focused.

---

## 15. Testing Strategy

### 15.1 Current Testing

- No formal unit tests are implemented.
- Testing is done through notebook execution and result verification.
- Each pipeline notebook is run end-to-end to verify correctness.

### 15.2 Verification Steps

1. Run notebook end-to-end without errors.
2. Check output shapes match expected values.
3. Verify label distributions are correct.
4. Compare F1 scores to baseline expectations.
5. Inspect a sample of predictions for sanity.

### 15.3 Future Testing

- Add unit tests for key functions (encode, train, classify).
- Add integration tests for the full pipeline.
- Add regression tests to catch performance degradation.
- Add validation tests for input data quality.

---

## 16. Performance Benchmarks

### 16.1 Embedding Generation Speed

| Model | Batch Size | Rows/sec | Time for 100k rows |
|---|---|---|---|
| Qwen3-0.6B | 32 | ~33 | ~50 min |
| Qwen3-0.6B | 64 | ~42 | ~40 min |
| E5-Large-v2 | 32 | ~40 | ~42 min |
| BGE-M3 | 32 | ~28 | ~60 min |

### 16.2 Classifier Training Speed

| Classifier | Rows | Features | Time |
|---|---|---|---|
| XGBoost (200 trees) | 59,105 | 1024 | ~8 sec |
| XGBoost (200 trees) | 10,872 | 1024 | ~2 sec |
| Logistic Regression | 59,105 | 1024 | ~3 sec |
| Logistic Regression | 10,872 | 1024 | ~1 sec |

### 16.3 Inference Speed

| Configuration | Batch Size | Rows/sec | Time for 458k rows |
|---|---|---|---|
| V1 (single GPU) | 32 | ~130 | ~58 min |
| V2 (multi-worker, FP16) | 128 | ~380 | ~20 min |

### 16.4 Memory Usage

| Operation | Peak GPU RAM | Peak CPU RAM |
|---|---|---|
| Qwen3 encoding (batch 32) | ~4 GB | ~2 GB |
| E5 encoding (batch 32) | ~3 GB | ~2 GB |
| XGBoost training | N/A | ~1 GB |
| 458k inference (V2) | ~6 GB (per GPU) | ~4 GB |

---

## 17. Known Issues and Workarounds

### 17.1 Split-After-Paraphrase Leakage

**Issue:** Pipeline A splits after augmentation, allowing synthetic siblings across splits.
**Workaround:** Use Pipeline B results for authoritative comparisons.
**Fix:** Split before augmentation in future versions.

### 17.2 Output Encoding in Inference

**Issue:** Inference output shows 0/1 instead of string labels.
**Workaround:** Map 0 -> Not-SATD, 1 -> SATD manually.
**Fix:** Re-run inference with string label output.

### 17.3 Description Truncation

**Issue:** 16.3% of descriptions truncated at 256 tokens.
**Workaround:** Accept information loss for speed.
**Fix:** Use 512 tokens (slower) or chunk long descriptions.

### 17.4 Missing Description Handling

**Issue:** 6.4% of issues have missing descriptions.
**Workaround:** Treat as empty string.
**Fix:** Flag and report separately.

---

## 18. Future Development Roadmap

### 18.1 Short-Term (1-2 months)

- [ ] Add unit tests for key functions.
- [ ] Fix output encoding in inference.
- [ ] Deploy E5 and BGE-M3 on 458k issues.
- [ ] Add confidence thresholds and abstention.

### 18.2 Medium-Term (3-6 months)

- [ ] Add keyword-feature integration (Phase 7-9).
- [ ] Add longitudinal SATD tracking.
- [ ] Add cross-project transfer experiments.
- [ ] Add error taxonomy with human annotation.

### 18.3 Long-Term (6-12 months)

- [ ] Add real-time SATD monitoring in CI/CD.
- [ ] Add multi-modal SATD detection (code + text).
- [ ] Add verification pipeline integration.
- [ ] Add cognitive and intent debt categorization.

---

## 19. Appendix: Complete Code Examples

### 19.1 Full Pipeline A Example

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
import joblib

# Phase 1: Load and clean
df = pd.read_csv("satd_with_artifact_type.csv")
print(f"Loaded {len(df)} rows")

# Phase 2: Balance
# (simplified - actual balancing involves T5 augmentation)
# df_balanced = balance_classes(df)

# Phase 3: Split
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

# Phase 4: Generate embeddings
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
X_train = model.encode(train_df["text"].tolist(), batch_size=32, normalize_embeddings=True)
X_test = model.encode(test_df["text"].tolist(), batch_size=32, normalize_embeddings=True)

# Phase 5: Train classifier
y_train = train_df["label"].values
y_test = test_df["label"].values

clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    random_state=42
)
clf.fit(X_train, y_train)

# Phase 6: Evaluate
y_pred = clf.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"Macro F1: {macro_f1:.3f}")

# Save model
joblib.dump(clf, "model.joblib")
```

### 19.2 Full Inference Example

```python
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_parquet("issue_202608272234.parquet")
print(f"Loaded {len(df)} issues")

# Load models
id_model = joblib.load("identification_issue_qwen3_xgboost.joblib")
cat_model = joblib.load("categorization_issue_qwen3_xgboost.joblib")
encoder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Three text views
views = {
    "Title": df["Title"].fillna(""),
    "Description": df["Description"].fillna(""),
    "Title_Description": df["Title"].fillna("") + " " + df["Description"].fillna("")
}

# Classify each view
for view_name, texts in views.items():
    print(f"Classifying {view_name}...")
    embeddings = encoder.encode(texts.tolist(), batch_size=128, normalize_embeddings=True)
    
    # Step 1: Identify
    id_preds = id_model.predict(embeddings)
    
    # Step 2: Categorize SATD
    cat_preds = []
    for i, pred in enumerate(id_preds):
        if pred == "SATD":
            cat_pred = cat_model.predict(embeddings[i:i+1])[0]
            cat_preds.append(cat_pred)
        else:
            cat_preds.append("Not-SATD")
    
    df[f"{view_name}_SATD"] = cat_preds

# Save results
df.to_parquet("satd_classification_results.parquet")
print(f"Saved {len(df)} classified issues")
```

### 19.3 Full Demo Example

```python
import gradio as gr
import joblib
from sentence_transformers import SentenceTransformer

# Load models
encoder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
id_model = joblib.load("identification_issue_qwen3_xgboost.joblib")
cat_model = joblib.load("categorization_issue_qwen3_xgboost.joblib")

def classify_text(text):
    # Encode
    embedding = encoder.encode([text], normalize_embeddings=True)
    
    # Identify
    id_pred = id_model.predict(embedding)[0]
    id_proba = id_model.predict_proba(embedding)[0]
    
    # Categorize if SATD
    if id_pred == "SATD":
        cat_pred = cat_model.predict(embedding)[0]
        cat_proba = cat_model.predict_proba(embedding)[0]
    else:
        cat_pred = "Not-SATD"
        cat_proba = None
    
    return {
        "SATD": id_pred,
        "Category": cat_pred,
        "Confidence": float(max(id_proba))
    }

# Create interface
demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(label="Enter text"),
    outputs=gr.JSON(label="Result"),
    title="SATD Classifier"
)

demo.launch()
```

---

## 20. Appendix: Code Organization Patterns

### 20.1 Notebook Organization

Each notebook follows this pattern:
1. **Imports**: All required libraries
2. **Configuration**: Paths, parameters, settings
3. **Data Loading**: Load raw data files
4. **Data Processing**: Clean, map, balance
5. **Embedding Generation**: Generate embeddings
6. **Classifier Training**: Train models
7. **Evaluation**: Compute metrics
8. **Export**: Save results and models

### 20.2 Model Naming Convention

All models follow this pattern:
```
{type}_{pipeline}_{task}_{artifact}_{embedding}_{classifier}.joblib
```

Examples:
- `model_pipeline_a_identification_code_comments_qwen3_xgboost.joblib`
- `model_pipeline_b_categorization_issues_bge_m3_logreg.joblib`

### 20.3 Results File Convention

All results follow this pattern:
```
phase{N}_{description}.csv
```

Examples:
- `phase5_results_summary.csv`
- `phase6_identification_comparison.csv`

### 20.4 Embedding File Convention

All embeddings follow this pattern:
```
{pipeline}_{artifact}_{split}_{embedding}.npy
```

Examples:
- `pipeline_a_code_comments_train_qwen3.npy`
- `pipeline_b_issues_test_e5.npy`

---

## 21. Appendix: Error Handling Patterns

### 21.1 Data Loading Errors

```python
try:
    df = pd.read_csv(file_path, sep=";")
except pd.errors.ParserError:
    print(f"Failed to parse {file_path}")
    continue
except FileNotFoundError:
    print(f"File not found: {file_path}")
    continue
```

### 21.2 Embedding Generation Errors

```python
try:
    embeddings = model.encode(texts, batch_size=32)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Reduce batch size
        embeddings = model.encode(texts, batch_size=16)
    else:
        raise
```

### 21.3 Model Training Errors

```python
try:
    clf.fit(X_train, y_train)
except ValueError as e:
    print(f"Training failed: {e}")
    # Check for NaN values
    if np.isnan(X_train).any():
        print("X_train contains NaN values")
    if len(np.unique(y_train)) < 2:
        print("y_train has fewer than 2 classes")
```

### 21.4 Inference Errors

```python
try:
    prediction = clf.predict(embedding)
except Exception as e:
    print(f"Prediction failed: {e}")
    prediction = "Unknown"
```

---

## 22. Appendix: Performance Optimization Patterns

### 22.1 Batch Processing

```python
# Process in batches to avoid memory issues
batch_size = 128
all_embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeddings = model.encode(batch, batch_size=batch_size)
    all_embeddings.append(embeddings)
all_embeddings = np.vstack(all_embeddings)
```

### 22.2 Length-Sorted Batching

```python
# Sort by length to reduce padding waste
lengths = [len(t) for t in texts]
sorted_indices = np.argsort(lengths)
sorted_texts = [texts[i] for i in sorted_indices]

# Process in batches
embeddings = model.encode(sorted_texts, batch_size=128)

# Restore original order
embeddings[sorted_indices] = embeddings
```

### 22.3 Classify-and-Discard

```python
# Never store all embeddings at once
for batch in batches:
    embeddings = model.encode(batch)
    predictions = clf.predict(embeddings)
    # Save predictions, discard embeddings
    save_predictions(predictions)
```

### 22.4 Multi-Worker Processing

```python
import multiprocessing as mp

def process_shard(shard_id, texts, model_path, model_path):
    model = SentenceTransformer(model_path)
    clf = joblib.load(model_path)
    embeddings = model.encode(texts)
    predictions = clf.predict(embeddings)
    return predictions

# Split data across workers
shards = np.array_split(texts, num_workers)
with mp.Pool(num_workers) as pool:
    results = pool.starmap(process_shard, enumerate(shards))
```

---

## 23. Appendix: Testing Patterns

### 23.1 Unit Test Example

```python
import unittest
import numpy as np

class TestEmbeddingGeneration(unittest.TestCase):
    def test_output_shape(self):
        embeddings = generate_embeddings(["test text"], "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(embeddings.shape, (1, 1024))
    
    def test_normalization(self):
        embeddings = generate_embeddings(["test text"], "Qwen/Qwen3-Embedding-0.6B")
        norm = np.linalg.norm(embeddings[0])
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_batch_consistency(self):
        emb1 = generate_embeddings(["test"], "Qwen/Qwen3-Embedding-0.6B")
        emb2 = generate_embeddings(["test"], "Qwen/Qwen3-Embedding-0.6B")
        np.testing.assert_array_almost_equal(emb1, emb2)
```

### 23.2 Integration Test Example

```python
def test_full_pipeline():
    # Load data
    df = pd.read_csv("test_data.csv")
    
    # Generate embeddings
    embeddings = generate_embeddings(df["text"].tolist())
    
    # Train model
    clf = train_classifier(embeddings, df["label"].values)
    
    # Evaluate
    predictions = clf.predict(embeddings)
    f1 = f1_score(df["label"].values, predictions, average="macro")
    
    # Assert
    assert f1 > 0.5, f"F1 too low: {f1}"
```
