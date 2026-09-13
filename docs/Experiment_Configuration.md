# Experiment Configuration

**SATD Embedding Comparison Project**
**Team: Shahriar Pias, Navid Ibrahim, Musaddiq Rafi**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline A Configuration](#2-pipeline-a-configuration)
3. [Pipeline B Configuration](#3-pipeline-b-configuration)
4. [Embedding Configuration](#4-embedding-configuration)
5. [Classifier Configuration](#5-classifier-configuration)
6. [Inference Configuration](#6-inference-configuration)
7. [Hardware and Environment](#7-hardware-and-environment)
8. [Reproducibility Settings](#8-reproducibility-settings)
9. [Configuration Files](#9-configuration-files)

---

## 1. Overview

This document captures every configurable parameter used in the experiments, including data processing, embedding generation, classifier training, and inference. It is the authoritative reference for reproducing exact results.

---

## 2. Pipeline A Configuration

### 2.1 Data Loading

| Parameter | Value |
|---|---|
| source | Kaggle `shahriarpias/satd-sources-cleaned` |
| file_count | 31 |
| format | semicolon-separated CSV |
| combined_raw | 820,885 rows |
| encoding | UTF-8 |
| error_handling | `on_bad_lines='warn'` |

### 2.2 Cleaning

| Parameter | Value |
|---|---|
| text_column | `text` |
| label_column | `Label` |
| artifact_inference | filename-based |
| min_words | 3 (rows with 2 or fewer words dropped) |
| drop_duplicates | True, on (text, artifact_type) |
| drop_blanks | True |
| result_rows | ~386,646 |

### 2.3 Label Mapping

| Source Label | Mapped To |
|---|---|
| CODE | C/D |
| DESIGN | C/D |
| CODE-DESIGN | C/D |
| REQ | REQ |
| TEST | TES |
| DOC | DOC |
| Not-SATD | Not-SATD |
| Defect | DROPPED |
| Architecture | DROPPED |
| Build | DROPPED |
| unmapped | DROPPED |
| without_classification | DROPPED |

### 2.4 Balancing

| Parameter | Value |
|---|---|
| strategy | Equalize classes per artifact |
| target_class_size | max(SATD_subtype_count) |
| downsample_not_satd | True |
| augmentation_method | T5 paraphrase |
| augmentation_model | `humarin/chatgpt_paraphraser_on_T5_base` |
| pull_request_req_cap | ~465 (3x original) |
| augmentation_seed | 42 |

### 2.5 Splitting

| Parameter | Value |
|---|---|
| split_ratio | 80/10/10 (train/val/test) |
| stratify_by | label |
| random_state | 42 |
| split_timing | AFTER augmentation |
| train_rows | 59,105 |
| val_rows | 7,390 |
| test_rows | 7,395 |

### 2.6 Saved Shapes

| Artifact | Train | Val | Test |
|---|---|---|---|
| code_comments | ~15,000 | ~1,875 | ~1,875 |
| issues | ~15,000 | ~1,875 | ~1,875 |
| commits | ~15,000 | ~1,875 | ~1,875 |
| pull_requests | ~14,105 | ~1,765 | ~1,770 |

---

## 3. Pipeline B Configuration

### 3.1 Data Loading

| Parameter | Value |
|---|---|
| source | Sutoyo et al. AugGPT files |
| file_count | 4 |
| format | semicolon-separated CSV |
| loaded_raw | ~109,000 rows |
| after_cleaning | ~95,704 rows |
| encoding | UTF-8 |

### 3.2 Files

| File | Artifact |
|---|---|
| `code_comments_AugGPT.csv` | code_comments |
| `issues_AugGPT.csv` | issues |
| `commits_AugGPT.csv` | commits |
| `pull_requests_AugGPT.csv` | pull_requests |

### 3.3 Label Distribution (After Cleaning)

| Artifact | Not-SATD | C/D | REQ | TES | DOC | Total |
|---|---|---|---|---|---|---|
| Code comments | 8,130 | 1,365 | 1,365 | 1,365 | 1,365 | 13,590 |
| Issues | 2,770 | 325 | 325 | 325 | 325 | 4,070 |
| Commits | 2,990 | 325 | 325 | 325 | 325 | 4,290 |
| Pull requests | 1,240 | 215 | 215 | 215 | 215 | 2,100 |

### 3.4 Splitting

| Parameter | Value |
|---|---|
| split_ratio | 80/10/10 (train/val/test) |
| stratify_by | label |
| random_state | 42 |
| split_timing | BEFORE any extra augmentation |
| purpose | Avoid train/test leakage from synthetic siblings |

### 3.5 Key Difference from Pipeline A

- Pipeline B keeps the paper's natural Not-SATD majority (65-82%).
- Pipeline A equalizes all five classes.
- This is the central variable tested in RQ3.

---

## 4. Embedding Configuration

### 4.1 Qwen3-Embedding-0.6B

| Parameter | Value |
|---|---|
| Hugging Face ID | `Qwen/Qwen3-Embedding-0.6B` |
| Dimensions | 1024 |
| Pooling | mean |
| Normalize | L2 |
| Precision | FP16 |
| max_seq_length | 256 (inference), 512 (training) |
| batch_size | 32 (training), 128 (inference) |
| Device | CUDA (T4 GPU) |

### 4.2 E5-Large-v2

| Parameter | Value |
|---|---|
| Hugging Face ID | `intfloat/e5-large-v2` |
| Dimensions | 1024 |
| Pooling | mean |
| Normalize | L2 |
| Query Prefix | `query: ` |
| Passage Prefix | `passage: ` |
| Precision | FP16 |
| max_seq_length | 256 (inference), 512 (training) |
| batch_size | 32 (training), 128 (inference) |
| Device | CUDA (T4 GPU) |

### 4.3 BGE-M3

| Parameter | Value |
|---|---|
| Hugging Face ID | `BAAI/bge-m3` |
| Dimensions | 1024 |
| Pooling | mean |
| Normalize | L2 |
| Precision | FP16 |
| max_seq_length | 256 (inference), 512 (training) |
| batch_size | 32 (training), 128 (inference) |
| Device | CUDA (T4 GPU) |
| Pipeline | B only |

### 4.4 Embedding Generation Process

1. Load frozen model from Hugging Face.
2. Tokenize input text.
3. Forward pass through transformer.
4. Mean-pool across token embeddings.
5. L2-normalize the resulting vector.
6. Save as `.npy` file, row-aligned with CSV.

### 4.5 Output Format

| File | Shape | Description |
|---|---|---|
| `{embedding}_train.npy` | (n_train, 1024) | Training embeddings |
| `{embedding}_val.npy` | (n_val, 1024) | Validation embeddings |
| `{embedding}_test.npy` | (n_test, 1024) | Test embeddings |

---

## 5. Classifier Configuration

### 5.1 XGBoost (Primary)

| Parameter | Value |
|---|---|
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| objective | multi:softprob |
| eval_metric | mlogloss |
| use_label_encoder | False |
| random_state | 42 |
| Platform | CPU |

### 5.2 Logistic Regression (Ablation)

| Parameter | Value |
|---|---|
| solver | lbfgs |
| max_iter | 1000 |
| C | 0.5 (Pipeline B), 1.0 (Pipeline A) |
| multi_class | multinomial |
| random_state | 42 |
| Platform | CPU |

### 5.3 Training Process

1. Filter rows by artifact_type.
2. For identification: use all rows.
3. For categorization: use SATD rows only.
4. Train classifier on embedding vectors.
5. Save model as `.joblib` file.
6. Evaluate on held-out test set.

### 5.4 Model Count

| Pipeline | Embeddings | Classifiers | Artifacts | Tasks | Total |
|---|---|---|---|---|---|
| A | 2 (Qwen3, E5) | 2 (XGBoost, LR) | 4 | 2 | 32 |
| B | 3 (Qwen3, E5, BGE-M3) | 2 (XGBoost, LR) | 4 | 2 | 48 |
| Total | | | | | 80 |

---

## 6. Inference Configuration

### 6.1 Input

| Parameter | Value |
|---|---|
| file | `issue_202608272234.parquet` |
| rows | 458,232 |
| columns | ID, Title, Description |
| text_views | Title, Description, Title+Description |

### 6.2 Models

| Model | Purpose | Source |
|---|---|---|
| identification_issue_qwen3_xgboost.joblib | Step 1: SATD vs Not-SATD | Pipeline A |
| categorization_issue_qwen3_xgboost.joblib | Step 2: C/D, REQ, TES, DOC | Pipeline A |

### 6.3 Optimizations (V2)

| Optimization | Description |
|---|---|
| Multi-worker | Two worker processes, one per GPU |
| FP16 | Encoder weights in half precision |
| max_seq_length | 256 (truncated) |
| batch_size | 128 |
| Length sorting | Sort by text length before batching |
| Classify-and-discard | Embeddings not stored, classifications written directly |
| Checkpointing | Per-shard Parquet saves |

### 6.4 Output

| File | Rows | Columns |
|---|---|---|
| `satd_classification_results.parquet` | 458,232 / 458,232 | ID, Title, Description, Title_SATD, Description_SATD, Title_Description_SATD |

Coverage: 100%.

---

## 7. Hardware and Environment

### 7.1 Kaggle Notebooks

| Resource | Specification |
|---|---|
| GPU | NVIDIA T4 x2 (16GB each) |
| CPU | 4 vCPU |
| RAM | 30 GB |
| Storage | 20 GB |
| Framework | PyTorch 2.x, transformers 4.x |
| Python | 3.10+ |

### 7.2 Local Machine (Inference)

| Resource | Specification |
|---|---|
| GPU | NVIDIA RTX series (if available) |
| CPU | x86_64 |
| RAM | 8 GB minimum |
| Platform | Windows 10/11 |

---

## 8. Reproducibility Settings

### 8.1 Random Seeds

| Operation | Seed |
|---|---|
| Train/val/test split | 42 |
| Augmentation | 42 |
| XGBoost | 42 |
| Logistic regression | 42 |

### 8.2 Determinism

- All splits use `random_state=42`.
- XGBoost uses `random_state=42`.
- T5 augmentation uses fixed seed.
- GPU operations may have non-deterministic behavior.

### 8.3 Version Pinning

Key dependencies:
- PyTorch >= 2.0
- transformers >= 4.30
- xgboost >= 1.7
- scikit-learn >= 1.2
- sentence-transformers >= 2.2
- numpy >= 1.24
- pandas >= 2.0

---

## 9. Configuration Files

### 9.1 Notebook Files

| File | Purpose |
|---|---|
| `notebooks/full_pipeline.ipynb` | Pipeline A (from scratch) |
| `navid-experiment/full_pipeline_kaggle.ipynb` | Pipeline B (paper data, Qwen3 + E5) |
| `navid-experiment/full_pipeline_kaggle_bge_m3.ipynb` | Pipeline B (BGE-M3 extension) |
| `navid-experiment/paper_experiment.ipynb` | First trial, all three embeddings |
| `navid-experiment/paper_experiment_2.ipynb` | T5 paraphrase experiment |
| `navid-experiment/paper_experiment_3.ipynb` | Final after supervisor revision |
| `navid-experiment/satd_bench.ipynb` | Qwen3 vs E5 on Qwen3 validation |

### 9.2 Script Files

| File | Purpose |
|---|---|
| `musaddiq_rafi/label_issues.py` | Inference script (v1) |
| `Demo/satd_infer.py` | Demo inference with probabilities |

### 9.3 Demo Files

| File | Purpose |
|---|---|
| `Demo/app.py` | Gradio UI |
| `Demo/requirements.txt` | Dependencies |
| `Demo/README.md` | Setup instructions |

---

## 10. Detailed Parameter Justification

### 10.1 XGBoost Parameters

| Parameter | Value | Justification |
|---|---|---|
| n_estimators | 200 | Sufficient for convergence; more trees give diminishing returns |
| max_depth | 6 | Balances model complexity and overfitting risk |
| learning_rate | 0.1 | Standard learning rate; lower values need more trees |
| objective | multi:softprob | Multi-class classification with probability output |
| eval_metric | mlogloss | Log loss for multi-class; aligned with objective |
| use_label_encoder | False | Deprecated in newer XGBoost versions |
| random_state | 42 | Ensures reproducibility |
| Platform | CPU | XGBoost trains fast on CPU; GPU not needed |

### 10.2 Logistic Regression Parameters

| Parameter | Value | Justification |
|---|---|---|
| solver | lbfgs | Efficient for multinomial problems |
| max_iter | 1000 | Ensures convergence on 1024-dimensional embeddings |
| C | 0.5 (Pipeline B), 1.0 (Pipeline A) | Regularization strength; lower C = more regularization |
| multi_class | multinomial | Direct multinomial formulation |
| random_state | 42 | Ensures reproducibility |
| Platform | CPU | Logistic regression is CPU-only |

### 10.3 Embedding Generation Parameters

| Parameter | Value | Justification |
|---|---|---|
| batch_size | 32 (training), 128 (inference) | Balances GPU memory and throughput |
| max_seq_length | 256 (inference), 512 (training) | 256 captures most text; 512 for training diversity |
| normalize_embeddings | True | L2 normalization for cosine similarity compatibility |
| FP16 | Yes | Halves memory usage; negligible quality loss |
| Pooling | mean | Standard for sentence embeddings; captures full context |

### 10.4 Split Parameters

| Parameter | Value | Justification |
|---|---|---|
| split_ratio | 80/10/10 | Standard split; 10% test provides reliable estimates |
| stratify_by | label | Ensures class balance across splits |
| random_state | 42 | Reproducible splits |
| split_timing (A) | After augmentation | Allows augmentation to inform training |
| split_timing (B) | Before augmentation | Avoids synthetic sibling leakage |

---

## 11. Environment Setup

### 11.1 Kaggle Notebook Setup

1. Create Kaggle account at kaggle.com.
2. Go to Settings -> API -> Create New API Token.
3. Upload kaggle.json to working directory.
4. Enable GPU: Settings -> Accelerator -> GPU T4 x2.
5. Enable Internet: Settings -> Internet -> On.
6. Add datasets:
   - `shahriarpias/satd-sources-cleaned` (Pipeline A)
   - `issue_202608272234.parquet` (Inference)
   - `04_trained_models/` folder (Inference)

### 11.2 Local Machine Setup

```bash
# Clone repository
git clone https://github.com/A-Piyas-04/SATD-embedding-DP2.git
cd SATD-embedding-DP2

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install sentence-transformers
pip install xgboost
pip install joblib
pip install gradio
```

### 11.3 Kaggle API Setup

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/ (Linux) or %USERPROFILE%\.kaggle\ (Windows)
chmod 600 ~/.kaggle/kaggle.json  # Linux only

# Download dataset
kaggle datasets download -d shahriarpias/satd-sources-cleaned --unzip
```

### 11.4 GPU Verification

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

Expected output on Kaggle:
```
CUDA available: True
CUDA device count: 2
CUDA device name: Tesla T4
```

---

## 12. Configuration Change Log

### 12.1 Pipeline A Changes

| Version | Date | Change | Reason |
|---|---|---|---|
| v1 | Jul 2026 | Initial implementation | Baseline |
| v2 | Jul 2026 | Added T5 paraphrase augmentation | Class imbalance |
| v3 | Jul 2026 | Added pull-request REQ cap (3x) | Repetitive synthetic text |
| v4 | Aug 2026 | Changed split to after augmentation | Better training signal |

### 12.2 Pipeline B Changes

| Version | Date | Change | Reason |
|---|---|---|---|
| v1 | Jul 2026 | Initial implementation (Qwen3 + E5) | Baseline |
| v2 | Aug 2026 | Added BGE-M3 extension | Third embedding comparison |
| v3 | Aug 2026 | Changed LR C to 0.5 | Regularization tuning |

### 12.3 Inference Changes

| Version | Date | Change | Reason |
|---|---|---|---|
| V1 | Aug 2026 | Single GPU, no optimization | Initial implementation |
| V2 | Aug 2026 | Multi-worker, FP16, batch 128 | Speed optimization |
| V2 | Aug 2026 | Length-sorted batching | Reduce padding waste |
| V2 | Aug 2026 | Per-shard checkpoints | Crash recovery |

---

## 13. Model File Inventory

### 13.1 Pipeline A Models (32 files)

| File | Pipeline | Task | Artifact | Embedding | Classifier |
|---|---|---|---|---|---|
| model_pipeline_a_identification_code_comments_qwen3_xgboost.joblib | A | identification | code_comments | Qwen3 | XGBoost |
| model_pipeline_a_categorization_code_comments_qwen3_xgboost.joblib | A | categorization | code_comments | Qwen3 | XGBoost |
| model_pipeline_a_identification_code_comments_qwen3_logreg.joblib | A | identification | code_comments | Qwen3 | LR |
| model_pipeline_a_categorization_code_comments_qwen3_logreg.joblib | A | categorization | code_comments | Qwen3 | LR |
| model_pipeline_a_identification_code_comments_e5_xgboost.joblib | A | identification | code_comments | E5 | XGBoost |
| model_pipeline_a_categorization_code_comments_e5_xgboost.joblib | A | categorization | code_comments | E5 | XGBoost |
| model_pipeline_a_identification_code_comments_e5_logreg.joblib | A | identification | code_comments | E5 | LR |
| model_pipeline_a_categorization_code_comments_e5_logreg.joblib | A | categorization | code_comments | E5 | LR |
| ... | ... | ... | issues | ... | ... |
| ... | ... | ... | commits | ... | ... |
| ... | ... | ... | pull_requests | ... | ... |

### 13.2 Pipeline B Models (48 files)

Same structure as Pipeline A, plus 16 additional BGE-M3 models:
- model_pipeline_b_identification_code_comments_bge_m3_xgboost.joblib
- model_pipeline_b_categorization_code_comments_bge_m3_xgboost.joblib
- model_pipeline_b_identification_code_comments_bge_m3_logreg.joblib
- model_pipeline_b_categorization_code_comments_bge_m3_logreg.joblib
- ... (4 per artifact x 4 artifacts = 16 files)

### 13.3 Inference Models (2 files)

| File | Purpose | F1 Score |
|---|---|---|
| identification_issue_qwen3_xgboost.joblib | Step 1: SATD vs Not-SATD | 0.798 |
| categorization_issue_qwen3_xgboost.joblib | Step 2: C/D, REQ, TES, DOC | 0.941 |

---

## 14. Embedding File Inventory

### 14.1 Pipeline A Embeddings

| File | Artifact | Split | Embedding | Shape |
|---|---|---|---|---|
| pipeline_a_code_comments_train_qwen3.npy | code_comments | train | Qwen3 | (59105, 1024) |
| pipeline_a_code_comments_val_qwen3.npy | code_comments | val | Qwen3 | (7390, 1024) |
| pipeline_a_code_comments_test_qwen3.npy | code_comments | test | Qwen3 | (7395, 1024) |
| pipeline_a_code_comments_train_e5.npy | code_comments | train | E5 | (59105, 1024) |
| ... | ... | ... | ... | ... |

### 14.2 Pipeline B Embeddings

| File | Artifact | Split | Embedding | Shape |
|---|---|---|---|---|
| pipeline_b_code_comments_train_qwen3.npy | code_comments | train | Qwen3 | (10872, 1024) |
| pipeline_b_code_comments_val_qwen3.npy | code_comments | val | Qwen3 | (1359, 1024) |
| pipeline_b_code_comments_test_qwen3.npy | code_comments | test | Qwen3 | (1359, 1024) |
| ... | ... | ... | BGE-M3 | ... |

---

## 15. Results File Inventory

### 15.1 Phase 5 Results (`results/phase5_results_summary.csv`)

Contains all macro F1 scores for all 80 model combinations.

### 15.2 Phase 6 Identification (`results/phase6_identification_comparison.csv`)

Contains identification results compared to BiLSTM baseline.

### 15.3 Phase 6 Categorization (`results/phase6_categorization_comparison.csv`)

Contains categorization results compared to BERT baseline.

### 15.4 Pipeline B Reports

| File | Content | Date |
|---|---|---|
| navid-experiment/satd_report.md | Qwen3 vs E5 results | Aug 3, 2026 |
| navid-experiment/satd_report_bge.md | BGE-M3 results | Aug 12, 2026 |

### 15.5 Inference Report

| File | Content | Date |
|---|---|---|
| docs/satd_new_issue_inference_v2.md | 458k inference report | Aug 2026 |
| musaddiq_rafi/SATD_New_Issue_Inference_V2_Report.pdf | Same as PDF | Aug 2026 |

---

## 16. Dependency Version Matrix

### 16.1 Core Dependencies

| Package | Minimum Version | Tested Version | Purpose |
|---|---|---|---|
| Python | 3.10 | 3.10.12 | Runtime |
| PyTorch | 2.0 | 2.1.0 | GPU compute |
| transformers | 4.30 | 4.40.0 | Model loading |
| sentence-transformers | 2.2 | 2.7.0 | Embedding generation |
| xgboost | 1.7 | 2.0.3 | Classifier |
| scikit-learn | 1.2 | 1.4.0 | Metrics, LR |
| numpy | 1.24 | 1.26.4 | Array operations |
| pandas | 2.0 | 2.2.0 | Data manipulation |
| joblib | 1.3 | 1.3.2 | Model serialization |
| pyarrow | 14 | 15.0.0 | Parquet I/O |

### 16.2 Demo Dependencies

| Package | Version | Purpose |
|---|---|---|
| gradio | >= 4.0 | Web UI |
| torch | >= 2.0 | Inference |
| transformers | >= 4.30 | Model loading |
| sentence-transformers | >= 2.2 | Embedding |
| joblib | >= 1.3 | Model loading |

### 16.3 Known Compatibility Issues

- PyTorch CUDA version must match system CUDA version.
- transformers >= 4.40 required for Qwen3 model support.
- xgboost >= 2.0 required for latest multi-class improvements.
- gradio >= 4.0 required for modern UI components.

---

## 17. Appendix: Complete Parameter Reference

### 17.1 Data Loading Parameters

| Parameter | Pipeline A | Pipeline B |
|---|---|---|
| File format | CSV (semicolon-separated) | CSV (semicolon-separated) |
| Encoding | UTF-8 | UTF-8 |
| Error handling | on_bad_lines='warn' | on_bad_lines='warn' |
| Text column | text | Text |
| Label column | Label | Label |
| Artifact inference | filename-based | filename-based |

### 17.2 Cleaning Parameters

| Parameter | Pipeline A | Pipeline B |
|---|---|---|
| Min words | 3 | 3 |
| Drop duplicates | Yes (text, artifact_type) | Yes (text, artifact_type) |
| Drop blanks | Yes | Yes |
| Drop unmapped labels | Yes | Yes |
| Label mapping | Option A | Option A |

### 17.3 Balancing Parameters (Pipeline A Only)

| Parameter | Value |
|---|---|
| Target class size | max(SATD_subtype_count) |
| Downsample strategy | Random undersampling |
| Augmentation model | humarin/chatgpt_paraphraser_on_T5_base |
| Augmentation batch size | 32 |
| Augmentation seed | 42 |
| REQ cap | 3x original (~465 rows) |

### 17.4 Split Parameters

| Parameter | Pipeline A | Pipeline B |
|---|---|---|
| Ratios | 80/10/10 | 80/10/10 |
| Stratify by | label | label |
| random_state | 42 | 42 |
| Split timing | After augmentation | Before augmentation |

### 17.5 Embedding Generation Parameters

| Parameter | Qwen3 | E5 | BGE-M3 |
|---|---|---|---|
| batch_size (training) | 32 | 32 | 32 |
| batch_size (inference) | 128 | 128 | 128 |
| max_seq_length (training) | 512 | 512 | 512 |
| max_seq_length (inference) | 256 | 256 | 256 |
| normalize | True | True | True |
| FP16 | Yes | Yes | Yes |
| Prefix | None | "query: " | None |
| Device | CUDA | CUDA | CUDA |

### 17.6 Classifier Parameters

| Parameter | XGBoost | Logistic Regression |
|---|---|---|
| n_estimators | 200 | N/A |
| max_depth | 6 | N/A |
| learning_rate | 0.1 | N/A |
| objective | multi:softprob | N/A |
| solver | N/A | lbfgs |
| max_iter | N/A | 1000 |
| C | N/A | 0.5 (B) / 1.0 (A) |
| multi_class | N/A | multinomial |
| random_state | 42 | 42 |

---

## 18. Appendix: Environment Variables

### 18.1 Kaggle Secrets

| Variable | Purpose | Required |
|---|---|---|
| HF_TOKEN | Hugging Face token for model download | Yes (for some models) |
| KAGGLE_USERNAME | Kaggle API username | Yes |
| KAGGLE_KEY | Kaggle API key | Yes |

### 18.2 Environment Setup Commands

```bash
# Set Kaggle credentials
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Set Hugging Face token
export HF_TOKEN="your_hf_token"

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 18.3 Kaggle Notebook Secrets

In Kaggle, secrets are set via:
1. Go to notebook Settings.
2. Add secret: HF_TOKEN.
3. The notebook can access it via `from kaggle_secrets import UserSecretsClient`.

---

## 19. Appendix: Output File Formats

### 19.1 Embedding Files (.npy)

| Property | Value |
|---|---|
| Format | NumPy binary array |
| Shape | (n_rows, 1024) |
| Dtype | float32 |
| Row alignment | Same order as CSV rows |

### 19.2 Model Files (.joblib)

| Property | Value |
|---|---|
| Format | Python pickle via joblib |
| Content | Trained classifier object |
| Methods | predict(), predict_proba(), classes_ |

### 19.3 Results CSVs

| Property | Value |
|---|---|
| Format | Comma-separated values |
| Encoding | UTF-8 |
| Index | False (no row index) |

### 19.4 Parquet Files

| Property | Value |
|---|---|
| Format | Apache Parquet |
| Compression | snappy (default) |
| Engine | pyarrow |

---

## 20. Appendix: Configuration Validation Checklist

### 20.1 Pre-Experiment Checklist

- [ ] GPU available and working
- [ ] Kaggle API configured
- [ ] Hugging Face token set (if needed)
- [ ] Data files downloaded
- [ ] Dependencies installed
- [ ] Working directory correct

### 20.2 Pipeline A Checklist

- [ ] 31 CSV files loaded
- [ ] ~820,885 raw rows
- [ ] Labels mapped to Option A
- [ ] ~386,646 clean rows
- [ ] Classes balanced per artifact
- [ ] ~73,890 balanced rows
- [ ] Split: 59,105 / 7,390 / 7,395
- [ ] Qwen3 embeddings generated
- [ ] E5 embeddings generated
- [ ] 32 models trained
- [ ] Results exported

### 20.3 Pipeline B Checklist

- [ ] 4 AugGPT CSVs loaded
- [ ] ~109,000 raw rows
- [ ] ~95,704 clean rows
- [ ] Not-SATD majority preserved
- [ ] Split before augmentation
- [ ] Qwen3, E5, BGE-M3 embeddings generated
- [ ] 48 models trained
- [ ] Results exported

### 20.4 Inference Checklist

- [ ] Input: 458,232 rows
- [ ] Models loaded
- [ ] Three text views processed
- [ ] Output: 458,232 rows (100%)
- [ ] Parquet saved

---

## 21. Appendix: Troubleshooting Guide

### 21.1 Common Errors and Solutions

| Error | Cause | Solution |
|---|---|---|
| CUDA out of memory | Batch size too large | Reduce batch_size to 16 |
| Model not found | Wrong path | Check MODEL_DIR setting |
| Label mismatch | Column name wrong | Verify text/label column names |
| Split error | Wrong timing | Check split timing (before/after augmentation) |
| F1 = 0.0 | All predictions same class | Check label mapping, verify training data |
| ImportError | Missing package | pip install -r requirements.txt |
| KeyError | Wrong column name | Check CSV schema matches expectations |

### 21.2 Performance Issues

| Issue | Solution |
|---|---|
| Embedding generation slow | Use FP16, increase batch_size |
| XGBoost training slow | Reduce n_estimators or max_depth |
| Inference slow | Use multi-worker, FP16, batch 128 |
| Memory issues | Use classify-and-discard approach |

### 21.3 Data Issues

| Issue | Solution |
|---|---|
| Too many duplicates | Increase min_words threshold |
| Class imbalance | Use Pipeline A balancing or Pipeline B natural distribution |
| Missing labels | Drop rows with unmapped labels |
| Wrong artifact type | Check filename-based inference logic |

---

## 22. Appendix: Best Practices

### 22.1 Data Processing

1. Always verify row counts after each cleaning step.
2. Check label distribution before and after balancing.
3. Use stratified splits to maintain class balance.
4. Save intermediate results for debugging.

### 22.2 Embedding Generation

1. Use FP16 for faster inference.
2. Batch size 32-64 for training, 128 for inference.
3. max_seq_length 256 for inference (captures most text).
4. Normalize embeddings for cosine similarity compatibility.

### 22.3 Classifier Training

1. Use XGBoost as primary classifier (outperforms LR).
2. Train one model per (task, artifact, embedding).
3. Evaluate on held-out test set only.
4. Save models with joblib for reproducibility.

### 22.4 Evaluation

1. Always report macro F1.
2. Report per artifact, per pipeline.
3. Never average Pipeline A and B results.
4. Compare to baseline scores from Sutoyo et al. (2024).

### 22.5 Documentation

1. Document all parameter choices.
2. Record timestamps for reproducibility.
3. Save intermediate results.
4. Maintain a change log.
