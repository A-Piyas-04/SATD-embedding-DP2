# SATD Classification on 458,232 New Issues — Optimized Inference Report (V2)

**Notebook:** `musaddiq_rafi/satd-classifications-optimized V2.ipynb` (executed)
**Date run:** September 2026 · **Platform:** Kaggle, 2× Tesla T4 (free tier) · **Status:** complete, 100% coverage
**Input:** [`issue_202608272234.parquet`](https://drive.google.com/file/d/1e9ZiR2VOglBilEEnJIfSWA_UkNudjEym/view?usp=sharing) (458,232 rows)
**Output:** [`satd_classification_results.parquet`](https://drive.google.com/file/d/1BTzQmgwanxi5YdCh4zQUGOf1XT5CR5g-/view?usp=drive_link) (458,232 rows)
**PDF version of this report:** [`SATD_New_Issue_Inference_V2_Report.pdf`](../musaddiq_rafi/SATD_New_Issue_Inference_V2_Report.pdf)

---

## 1. Objective

Apply the trained **Pipeline A** issue models — `Qwen3-Embedding-0.6B` (frozen) + XGBoost heads — to **458,232 unseen issue records** under three text views (title only, description only, title + description) and publish a reusable labeled Parquet:

```
ID, Title, Description, Title_SATD, Description_SATD, Title_Description_SATD
```

Two-stage logic per text view: (1) **identification** (SATD vs Not-SATD), then (2) **categorization** (C/D, REQ, TES, DOC) for rows flagged SATD. This is a deployment/transfer exercise — there is no ground truth for the new issues, so the output is a triage resource, not an accuracy benchmark.

## 2. Why V2 (what changed vs the original notebook)

The original inference notebook used a single GPU, full-precision weights, uncapped sequence length, and held all embeddings in RAM — infeasible for 458k rows on a free-tier session. V2 keeps the **same task, models, pooling, prompt handling, and `normalize_embeddings=True`**, and changes only the execution strategy:

| Change | Why it matters |
|---|---|
| Two worker processes, one per T4, interleaved shards (`rows[shard::2]`) | ~2× throughput with balanced shard lengths |
| FP16 encoder weights | ~1.5–2× faster on T4 tensor cores |
| `max_seq_length = 256` token cap | Removes the long-description attention tail (attention is quadratic in length) |
| Length-sorted batching, `batch_size = 128` | Minimizes padding waste — the largest win on mixed-length issues |
| Embed → classify → discard per batch | Never materializes 3 × 458k × 1024 float arrays in RAM |
| Per-shard Parquet checkpoints + streamed logs | A crash or timeout loses at most one shard |
| 4,000-row benchmark cell with runtime projection | Validates throughput and settings before committing hours of GPU |

## 3. Data profile (from the notebook inspection cell)

| Statistic | Value |
|---|---|
| Rows loaded | **458,232** (`ID, Title, Description`) |
| Missing values | `ID`: 0 · `Title`: 0 · `Description`: 29,128 (~6.4%, filled with `""`) |
| Title chars (mean / p50 / p75 / max) | 58.4 / 55 / 72 / 269 |
| Description chars (mean / std / p50 / p75 / max) | 929 / 4,701 / 409 / 758 / ~1,391,304 |
| Descriptions longer than ~1,024 chars (truncated at 256 tokens) | **16.3%** |

Takeaway: titles are short and well-behaved; descriptions have a heavy tail (a single ~1.4M-char record exists). The 256-token cap is the deliberate trade-off — 512 tokens would be more faithful but roughly 2× slower on descriptions.

## 4. Models

| Role | Artifact file | Pipeline A issue macro F1 |
|---|---|---|
| Identification (SATD vs Not-SATD) | `identification_issue_qwen3_xgboost.joblib` | 0.798 |
| Categorization (C/D, REQ, TES, DOC) | `categorization_issue_qwen3_xgboost.joblib` | 0.941 |

Encoder `Qwen/Qwen3-Embedding-0.6B` was pre-downloaded once in the main process (shared `HF_CACHE`) so the two workers never race on the Hugging Face cache. XGBoost was chosen because it beat logistic regression in every recorded Pipeline A/B pair.

## 5. Execution

1. **Inspect** — load `ID/Title/Description`, report missingness and char-length distributions, preview head rows.
2. **Pre-download encoder** — `snapshot_download(Qwen/Qwen3-Embedding-0.6B)` into `/kaggle/working/hf_cache`.
3. **Write worker script** (`satd_worker.py`) — per-GPU shard loop: load shard → load encoder + both heads → `embed_and_classify` for titles, descriptions, combined → write `shard_{i}_of_2.parquet` with `row_pos, ID, Title_SATD, Description_SATD, Title_Description_SATD`.
4. **Benchmark** — full worker over 4,000 rows to project total runtime; guidance: if projection > 3h, lower `MAX_SEQ_LENGTH` (256 → 192/128) or raise batch size.
5. **Full run** — 2 workers × ~229,116 rows each, progress logged every 30 s (`rows/s`, elapsed, ETA).
6. **Merge** — concat shards, sort by `row_pos`, assert uniqueness/range/`ID` alignment, left-join predictions onto the input frame, write `satd_classification_results.parquet`. Found **2 shard files**; **458,232 classified (100%)**.

## 6. Results

### 6.1 Label distribution (executed summary cells)

| View | `0` (Not-SATD) | `1` (SATD) | SATD rate |
|---|---|---:|---:|
| Title only | 283,019 | 175,213 | **38.2%** |
| Description only | 188,418 | 269,814 | **58.9%** |
| Title + Description | 205,454 | 252,778 | **55.2%** |

Descriptions carry markedly more debt signal than titles alone (+20.7 pp). Title+description sits between the two (55.2%), consistent with issues whose title is a neutral feature request while the admission lives in the body.

### 6.2 Sample predictions (first 10 rows, `df.head(10)`)

| ID | Title (trunc.) | Title | Desc. | T+D |
|---|---|---:|---:|---:|
| 912 | Create a Java client for Receptor | 0 | 1 | 0 |
| 913 | Create Boot based ModuleRunner | 0 | 1 | 1 |
| 914 | Create a pluggable runtime SPI | 1 | 1 | 1 |
| 915 | Simplify GPDB UX around parameters… | 0 | 0 | 1 |
| 916 | Sqoop Module not running | 0 | 1 | 1 |
| 917 | Improve performance of TupleBuilder | 1 | 1 | 1 |
| 918 | Revisit benchmark matrix… | 1 | 1 | 1 |
| 919 | Produce Kafka Baseline numbers… | 0 | 0 | 0 |
| 920 | Acceptance Tests needs to wait… | 1 | 1 | 1 |
| 921 | Create a reference architecture… | 0 | 0 | 0 |

### 6.3 Encoding caveat — read before citing

The executed summary shows binary **`0/1`**, not the `Not-SATD / C/D / REQ / TES / DOC` strings in the task spec. Interpretation adopted here: **`1` = SATD, `0` = Not-SATD** for this artifact. Consequences:

- The notebook's own `SATD total = (df[col] != "Not-SATD").sum()` line counts every row (100%) and must be ignored; use §6.1 instead.
- The per-type breakdown (how much of the SATD is C/D vs REQ vs TES vs DOC) is **not present** in the saved summary. Verify the downloaded Parquet's dtypes/values before claiming type-level results; if the heads emitted only binary labels, re-emit string labels from the categorizer to recover the breakdown.

## 7. How to reuse the output

```python
import pandas as pd
df = pd.read_parquet("satd_classification_results.parquet")  # 458,232 rows
# schema: ID, Title, Description, Title_SATD, Description_SATD, Title_Description_SATD
print(df[["Title_SATD","Description_SATD","Title_Description_SATD"]].apply(pd.Series.value_counts))
# suggested triage: high-recall pool = Description_SATD==1 (58.9%);
# high-precision pool = all three views agree on 1; disagreements (e.g. title 0 / desc 1) are worth sampling.
```

Suggested analyses: agreement matrix across the three views; stratified manual audit (~300–500 sampled issues) to estimate precision on the new distribution; keyword/topic profiling of predicted-SATD descriptions; join with resolution-time metadata if available.

## 8. Limitations

- No ground truth for the 458k issues — rates above are *predicted* prevalence, not accuracy.
- 16.3% of descriptions truncated at 256 tokens; very long bodies lose tail context.
- Single deployment configuration (Pipeline A, Qwen3+XGBoost, issue heads); E5/BGE-M3 and Pipeline B heads not run at scale.
- Pipeline A training limitations propagate: split-after-paraphrase leakage risk and AugGPT files inside the 31-source merge.
- Output encoding ambiguity (§6.3) must be resolved before type-level claims.

## 9. Reproduction (Kaggle)

1. Settings → Accelerator **GPU T4 x2**, Internet **on**.
2. Add datasets: `issue_202608272234.parquet` and `04_trained_models/`.
3. Set `INPUT_PATH`, `MODEL_DIR` in the config cell; knobs: `MAX_SEQ_LENGTH=256`, `BATCH_SIZE=128` (→64 on OOM), `USE_FP16=True`.
4. Run: inspect → pre-download → benchmark (4k) → full run → merge → summary.
5. Output: `/kaggle/working/satd_classification_results.parquet` (+ `shards/`, `logs/`).

## 10. Links

- Input: <https://drive.google.com/file/d/1e9ZiR2VOglBilEEnJIfSWA_UkNudjEym/view?usp=sharing>
- Output: <https://drive.google.com/file/d/1BTzQmgwanxi5YdCh4zQUGOf1XT5CR5g-/view?usp=drive_link>
- Drive folder (data & models): <https://drive.google.com/drive/folders/1E-jzrNGE2NyKEsrI8Ud9Phk3gx_8a2dD?usp=sharing>
- Baseline: Sutoyo et al. (2024), arXiv:2410.15804.
