# SATD Gradio Demo

Live demo UI for presentation day. Paste (or pick) an issue → **Identification** → **Categorization**, with confidence bars, Title / Description / Both compare, and keyword cues.

**Model path:** same as `notebooks/satd-classifications-optimized V2.ipynb`  
Pipeline A · **issue** artifact · **Qwen3-Embedding-0.6B** + **XGBoost** heads.

This UI is a **working artifact** of the research (live inference). F1 tables, Pipeline A vs B narrative, and baseline comparisons stay on the **slides** — not duplicated here.

---

## Layout

| Side | Contents |
|---|---|
| **Left** | Input, Classify / Compare, results, confidence bars, Title vs Desc vs Both table, keyword cues |
| **Right (sticky sidebar)** | 10 demo samples always visible — click to fill the input |

---

## Features

| Feature | What it shows |
|---|---|
| **Classify** | Two-step labels: SATD / Not-SATD → C/D, DOC, REQ, TES |
| **Confidence bars** | P(Not-SATD) vs P(SATD); category probabilities when SATD |
| **Compare Title / Desc / Both** | Same issue, three input variants in one table (applied experiment) |
| **Keyword cues** | Highlights like `TODO`, `FIXME`, `hack` (display only — not used by the model) |
| **Sample sidebar** | Human-obvious samples with expected answers in the button labels |

### Labels

| Field | Format | Examples |
|---|---|---|
| Step 1 — Identification | Short + full name | `SATD — Self-Admitted Technical Debt` |
| Step 2 — Category | Short code only | `C/D`, `DOC`, `REQ`, `TES`, or `—` |
| Final label | Short + full name | `C/D — Code / Design debt` |

Identification heads use numeric classes `0` / `1` (Not-SATD / SATD); categorization uses `0`–`3` (C/D, DOC, REQ, TES). The demo maps these to readable labels.

---

## Demo samples (sidebar)

| # | Focus | Easy human answer |
|---|---|---|
| 1 | Feature request | Not-SATD |
| 2 | Hack / ugly code | SATD → C/D |
| 3 | Missing tests | SATD → TES |
| 4 | Outdated docs | SATD → DOC |
| 5 | Unbuilt requirement | SATD → REQ |
| 6 | Shortcut / design debt | SATD → C/D |
| 7 | Flaky tests | SATD → TES |
| 8 | Missing README | SATD → DOC |
| 9 | Incomplete requirement | SATD → REQ |
| 10 | Typo fix | Not-SATD |

---

## Timing (keep under 1 minute per click)

| Phase | Time |
|---|---|
| First start (download/load Qwen3 + joblibs) | Several minutes — do this **before** the talk |
| Each **Classify** or **Compare** | Usually a few seconds (GPU); often well under 60s on CPU |

Classifies **one sample** (or three short variants), not the 458k batch job.

---

## Setup

1. Place these files in `Demo/models/`:

   - `identification_issue_qwen3_xgboost.joblib`
   - `categorization_issue_qwen3_xgboost.joblib`

   From Kaggle: `satd-pipeline-a-models/04_trained_models/`  
   Or the project Google Drive `04_trained_models` folder.

2. Install and run:

```bash
cd Demo
pip install -r requirements.txt
python app.py
```

3. Open http://127.0.0.1:7860

Optional — point at another model folder:

```powershell
$env:SATD_MODEL_DIR = "D:\path\to\04_trained_models"
python app.py
```

---

## Demo-day checklist (~3 minutes after slides)

1. Start `python app.py` early; wait until **System status** says `Ready`
2. Click a sample on the **right sidebar**
3. **Classify** → show Step 1 / Step 2 / Final label + confidence bars
4. Optionally **Compare Title / Desc / Both**
5. Point at keyword cues if useful  
6. One line for judges: *“Slides showed where we beat or lose the baseline; this is the trained pipeline running live on new issue text.”*

---

## Files

| File | Role |
|---|---|
| `app.py` | Gradio UI (left workspace + right sample sidebar) |
| `satd_infer.py` | Encode + two-step classify, probabilities, 3-mode compare, keyword cues |
| `models/` | Local `.joblib` heads (not in git) |
| `models/README.md` | What to put in `models/` |
| `requirements.txt` | Demo dependencies |

---

## What this UI is / is not

| Is | Is not |
|---|---|
| Live inference for Pipeline A **issue** Qwen3+XGB | Full Pipeline A vs B result explorer |
| Applied Title / Desc / Both demo | Replacement for slide F1 tables |
| Confidence on the sample you just ran | Proof by itself that you beat BiLSTM/BERT |
