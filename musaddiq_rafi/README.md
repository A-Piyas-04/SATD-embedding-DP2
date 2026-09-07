# Musaddiq Rafi - SATD Classification Task

## What Was Told to Do

From the team chat (28/08/2026 - 07/09/2026):

1. Download the provided file from Google Drive (3 columns: `ID`, `Title`, `Description`)
2. Run the trained model on **3 variations**:
   - Title only
   - Description only
   - Title + Description
3. **For SATD Identification:** Use the baseline model (best output so far)
4. **For SATD Categorization:** Use our trained model
5. **Use Pipeline A** (uses more dataset than authors only)
6. **Output:** Parquet file with columns:
   - `ID`, `Title`, `Description`, `Title_SATD`, `Description_SATD`, `Title_Description_SATD`
7. **SATD Categories:** Not-SATD, C/D, TEST, Requirement, DOC

## What We Are Doing

We are using the trained Pipeline A models (Qwen3-Embedding + XGBoost) to classify SATD in the provided issue data. The pipeline follows a **two-step classification**:

1. **Identification** - Is this SATD or Not-SATD? (binary classification)
2. **Categorization** - If SATD, what type? (C/D, REQ, TES, DOC - 4-class classification)

### Models Used
- **Embedding:** Qwen3-Embedding-0.6B (frozen encoder)
- **Classifier:** XGBoost (best performer from Pipeline A)
- **Best F1 Scores for Issues:**
  - Identification: 0.798
  - Categorization: 0.941

## Files

| File | Description |
|------|-------------|
| `satd_classification.ipynb` | Notebook to run classification |
| `models/` | 32 trained .joblib classifier files |
| `README.md` | This file |

### Input Data (Not in Git)

The input parquet file is not in the repository due to size. Download from:

**[Download issue_202608272234.parquet](https://drive.google.com/file/d/1e9ZiR2VOglBilEEnJIfSWA_UkNudjEym/view?usp=sharing)**

- 458,232 rows
- Columns: `ID`, `Title`, `Description`

## How to Run (Kaggle)

### Step 1: Upload Datasets

1. Go to Kaggle → Datasets → New Dataset
2. Upload `issue_202608272234.parquet` as a dataset
3. Upload the `models/` folder as another dataset

### Step 2: Create Notebook

1. Create a new notebook on Kaggle
2. Add both datasets to the notebook:
   - Click "Add data" → Search for your uploaded datasets
3. Update the paths in the notebook:

```python
INPUT_PATH = "/kaggle/input/your-parquet-dataset-name/issue_202608272234.parquet"
MODEL_DIR = "/kaggle/input/your-models-dataset-name/models"
```

### Step 3: Run All Cells

The notebook will:
1. Load the input data (458,232 rows)
2. Load Qwen3 embedding model from HuggingFace
3. Generate embeddings for Title, Description, and Title+Description
4. Run classification (Identification → Categorization)
5. Save results

### Step 4: Save Results

Results are saved to:
```
/kaggle/working/satd_classification_results.parquet
```

Download from Kaggle → Output → Click download button.

## Output Format

The output parquet file contains:

| Column | Description |
|--------|-------------|
| `ID` | Issue ID |
| `Title` | Original title text |
| `Description` | Original description text |
| `Title_SATD` | SATD classification for Title |
| `Description_SATD` | SATD classification for Description |
| `Title_Description_SATD` | SATD classification for Title+Description |

### SATD Categories
- `Not-SATD` - Not technical debt
- `C/D` - Code or Design debt
- `REQ` - Requirement debt
- `TES` - Test debt
- `DOC` - Documentation debt

## Status

- [x] Input data uploaded
- [x] Models downloaded (32 .joblib files)
- [x] Notebook created
- [ ] Run on Kaggle
- [ ] Upload results
