# Musaddiq Rafi — SATD Classification on New Issue Data (V2, executed)

## Task (from team chat, 28/08/2026 – 07/09/2026)

1. Download the provided file from Google Drive (3 columns: `ID`, `Title`, `Description`)
2. Run the trained model on **3 variations**: Title only · Description only · Title + Description
3. **Identification:** baseline/best model so far · **Categorization:** our trained model
4. **Use Pipeline A** (broader than the authors' data alone)
5. **Output:** Parquet with `ID, Title, Description, Title_SATD, Description_SATD, Title_Description_SATD`
6. **Categories:** Not-SATD, C/D, TEST, Requirement, DOC

## What was actually run

**Notebook:** [`satd-classifications-optimized V2.ipynb`](satd-classifications-optimized%20V2.ipynb) — optimized rewrite of `satd-classifications-on-new-issue-data.ipynb`, **executed end-to-end on Kaggle (2× Tesla T4)**.

| Item | Detail |
|---|---|
| Input | `issue_202608272234.parquet` — **458,232 rows** ([Download](https://drive.google.com/file/d/1e9ZiR2VOglBilEEnJIfSWA_UkNudjEym/view?usp=sharing)) |
| Encoder | `Qwen/Qwen3-Embedding-0.6B`, frozen, FP16, `max_seq_length=256`, `normalize_embeddings=True` |
| Heads | `identification_issue_qwen3_xgboost.joblib` (F1 0.798) → `categorization_issue_qwen3_xgboost.joblib` (F1 0.941), both Pipeline A |
| Parallelism | One worker per GPU, interleaved shards, length-sorted batching (`batch_size=128`), classify-and-discard per batch |
| Checkpoints | Per-shard Parquet in `shards/`, streamed logs in `logs/`, 4k-row benchmark before the full run |
| Output | `satd_classification_results.parquet` — **458,232 / 458,232 rows (100%)** ([Download](https://drive.google.com/file/d/1BTzQmgwanxi5YdCh4zQUGOf1XT5CR5g-/view?usp=drive_link)) |

## Results (from the executed summary cells)

| View | `0` (Not-SATD) | `1` (SATD) | SATD rate |
|---|---|---:|---:|
| `Title_SATD` | 283,019 | 175,213 | 38.2% |
| `Description_SATD` | 188,418 | 269,814 | 58.9% |
| `Title_Description_SATD` | 205,454 | 252,778 | 55.2% |

> **Encoding note:** the executed notebook prints binary `0/1`, not the `Not-SATD / C/D / REQ / TES / DOC` strings in the original spec. Treat `1` = SATD, `0` = Not-SATD for this artifact. The notebook's own `SATD total = (col != "Not-SATD").sum()` line overcounts because of this — use the table above. Check dtypes in the downloaded Parquet before citing per-type breakdowns.

Input text profile: titles ~58 chars mean; descriptions ~929 chars mean with a heavy tail (max ~1.39M chars); 29,128 missing descriptions (~6.4%, filled with `""`); **16.3% of descriptions exceed the 256-token cap and were truncated**.

## Files

| File | Description |
|------|-------------|
| `satd-classifications-optimized V2.ipynb` | Executed V2 notebook (benchmark → full run → merge → summary) |
| `models/` | 32 trained `.joblib` heads (mirrored from `04_trained_models/`) |
| `issue_202608272234.parquet` | Local copy of the input (if present; otherwise use the Drive link above) |
| `README.md` | This file |

Full analysis, charts, and reuse guide: [`../docs/satd_new_issue_inference_v2.md`](../docs/satd_new_issue_inference_v2.md) and [`SATD_New_Issue_Inference_V2_Report.pdf`](SATD_New_Issue_Inference_V2_Report.pdf).

## How to re-run (Kaggle)

1. Settings → Accelerator **GPU T4 x2**, Internet **on**.
2. Add datasets: `issue_202608272234.parquet` + the `models/` folder.
3. Update `INPUT_PATH` and `MODEL_DIR` in the config cell.
4. Run all cells: inspect → pre-download encoder → benchmark → full run → merge → summary.
5. Download `/kaggle/working/satd_classification_results.parquet` from Output.

## Status

- [x] Input data uploaded (458,232 rows)
- [x] Models attached (32 `.joblib` files)
- [x] V2 notebook created **and executed** (2 shards merged, 100% coverage)
- [x] Results published ([Parquet on Drive](https://drive.google.com/file/d/1BTzQmgwanxi5YdCh4zQUGOf1XT5CR5g-/view?usp=drive_link))
- [ ] Reconcile `0/1` vs string-label encoding for per-type (C/D, REQ, TES, DOC) analysis
