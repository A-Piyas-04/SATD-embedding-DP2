# SATD embedding comparison: research narrative

This document reconstructs the thesis project from what is actually in this repository: notebooks, result CSVs, reports, keyword lists, and README/docs. It is written so a first-time researcher can see the problem, the questions, the chronology, both pipelines, the numbers, and the limits of the claims.

**How this note was built.** Numbers come from [`results/phase5_results_summary.csv`](../results/phase5_results_summary.csv), [`results/phase6_identification_comparison.csv`](../results/phase6_identification_comparison.csv), [`results/phase6_categorization_comparison.csv`](../results/phase6_categorization_comparison.csv), [`navid-experiment/glm_pipeline_results.txt`](../navid-experiment/glm_pipeline_results.txt), [`navid-experiment/satd_report.md`](../navid-experiment/satd_report.md), and [`navid-experiment/satd_report_bge.md`](../navid-experiment/satd_report_bge.md). Methods come from [`notebooks/full_pipeline.ipynb`](../notebooks/full_pipeline.ipynb), [`docs/pipeline_a_from_scratch.md`](pipeline_a_from_scratch.md), [`docs/pipeline_b_paper_data.md`](pipeline_b_paper_data.md), and [`navid-experiment/glm_pipeline_artifacts_separated.ipynb`](../navid-experiment/glm_pipeline_artifacts_separated.ipynb). Large processed CSVs, embeddings, and `.joblib` models are **not** in git; they live on [Google Drive](https://drive.google.com/drive/folders/1E-jzrNGE2NyKEsrI8Ud9Phk3gx_8a2dD).

**What this note does not do.** It does not invent experiments. It does not treat unread papers as if they were cited. Background SATD literature that is **not named in this repo** is marked as *field context*. The only paper cited by authors, year, and title in the repo is Sutoyo, Avgeriou, and Capiluppi (2024).

---

## 1. The practical problem (narrative)

Software developers routinely take shortcuts: a hack that “works for now,” a missing test, a stale comment, a requirement that was never implemented. When they **admit** that shortcut in writing—in a source comment, an issue tracker ticket, a commit message, or a pull-request discussion—that admission is called **Self-Admitted Technical Debt (SATD)**.

SATD is useful because it is an explicit signal. It is also a management problem:

- Left untracked, SATD accumulates as extra cost: harder changes, more defects, slower onboarding.
- Manual scanning of comments, issues, commits, and pull requests does not scale across large projects.
- Debt is not one thing. A `TODO` that is a missing test is not the same as a `FIXME` that is a design smell or a documentation gap. Detection (is this SATD?) and **categorization** (what kind?) are different jobs.
- SATD appears in **four artifact types** with different language and length: code comments, issues, commit messages, and pull requests. A method that works on comments may fail on short commit lines.

The industry-facing story is therefore: developers leave SATD in the artifacts they already write; teams need automatic identification and typing of that debt so they can prioritize repayment. The research-facing story in **this** project is narrower: a recent deep-learning paper already reports strong numbers on those four artifacts using **task-specific** BiLSTM and BERT plus GPT-style augmentation. Those models are relatively expensive to train. This project asks whether **frozen modern text embeddings** plus cheap classifiers can match or beat that stack, and whether the answer depends on **which dataset and augmentation recipe** you use.

---

## 2. Field context vs sources used in this project

### 2.1 Sources used in this project (in-repo)

**Primary comparison paper**

Sutoyo, E., Avgeriou, P., & Capiluppi, A. (2024). *Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt.* [arXiv:2410.15804](https://arxiv.org/abs/2410.15804).

In this repo that paper is the replication target. The comparison is:

| Task | Sutoyo / supervisor setup (as used here) | This project |
|---|---|---|
| Identification (SATD vs Not-SATD) | GloVe + BiLSTM, with AugGPT augmentation | Frozen embeddings + XGBoost / logistic regression |
| Categorization (C/D, REQ, TES, DOC) | Fine-tuned BERT, with AugGPT | Same frozen embeddings + same heads |

Commit-message baselines in the CSVs are often labeled **supervisor’s replication** (BiLSTM 0.91, BERT 0.9804), not necessarily the paper’s published commit row. Comments, issues, and pull-request baselines are labeled as the paper (BiLSTM+AugGPT / BERT+AugGPT).

**Keyword lists.** Eight scored SATD keyword files in [`data/keywords/`](../data/keywords/) were extracted with KeyBERT in a replication / supervisor GitHub resource. They are **prepared for a future phase**, not yet used as model features.

**Hugging Face models actually run.** `Qwen/Qwen3-Embedding-0.6B`, `intfloat/e5-large-v2`, `BAAI/bge-m3` (Pipeline B only), T5 paraphraser `humarin/chatgpt_paraphraser_on_T5_base` (Pipeline A), and (in the Pipeline B notebook) optional `Qwen/Qwen2.5-1.5B-Instruct` rephrasing.

### 2.2 Field context (not bibliographic entries in this repo)

The following names appear **in dataset filenames**, not as formal citations: Maldonado-style comment corpora (Apache Ant, JMeter, ArgoUML, JFreeChart, JRuby), O’Brien, CppSATD, MLSATD, SSATD, and multi-artifact files named `satd-dataset-*` (issues, commits, pull requests). Standard SATD literature (for example Potdar & Shihab on the SATD concept, Maldonado et al. on comment datasets, later work on issues/commits/PRs) is the usual background for a thesis introduction, but **this repository does not cite those papers by title**. Do not list them in a related-work chapter unless they are actually read and added as citations.

---

## 3. Problem statement

**Problem.** Automatic SATD identification and categorization across four software artifacts currently depends, in the chosen baseline, on **purpose-trained sequence models and GPT-style data augmentation**. It is unclear (1) whether **frozen** general-purpose embeddings plus lightweight classifiers can replace that stack at much lower training cost, and (2) whether published baseline numbers are properties of the **architecture** or of the **paper’s own augmented data**.

**Why this is a research problem rather than an engineering ticket.** If frozen embeddings match BERT on debt *type* without fine-tuning, that is a cheaper operational recipe. If they only match BiLSTM when trained on the paper’s AugGPT files, then “beating the paper” is not a general SATD result—it is a result on that distribution.

---

## 4. Research questions

These RQs match what was actually measured. They were not numbered in the original README; they are stated here for the thesis.

- **RQ1 (identification).** Can frozen embeddings (Qwen3-Embedding-0.6B, E5-Large-v2, and later BGE-M3) plus XGBoost or logistic regression match Sutoyo-style BiLSTM+AugGPT **macro F1** for SATD vs Not-SATD, separately for each artifact type?
- **RQ2 (categorization).** Can the same stack match or beat Sutoyo-style BERT+AugGPT **macro F1** for C/D vs REQ vs TES vs DOC (SATD rows only), per artifact?
- **RQ3 (data protocol).** How much do RQ1 and RQ2 change between **Pipeline A** (31-source rebuild, T5 class equalization, split after augmentation) and **Pipeline B** (paper’s four AugGPT files, Not-SATD majority kept, split before extra augmentation)?
- **RQ4 (embedding vs head).** Does the choice of embedding model or the choice of classifier dominate performance?

---

## 5. Chronology of what was done

The work is two **similar but not identical** experiments, then an embedding swap on the second.

| When (from git, report dates, file mtimes) | What |
|---|---|
| ~20 June 2026 | [`results/SATD_Findings_Summary.pdf`](../results/SATD_Findings_Summary.pdf) — Pipeline A summary |
| 7 July 2026 | Pipeline A notebook, keyword files, phase 5/6 CSVs, README (from-scratch story). GitHub: `https://github.com/A-Piyas-04/SATD-embedding-DP2.git`. Kaggle datasets under Shahriar Pias (`shahriarpias/satd-sources-cleaned`, `combined-text-label`) |
| 3 August 2026 | Pipeline B: [`navid-experiment/glm_pipeline_artifacts_separated.ipynb`](../navid-experiment/glm_pipeline_artifacts_separated.ipynb), [`satd_report.md`](../navid-experiment/satd_report.md) — paper-data vs from-scratch tables |
| 12 August 2026 | Same Pipeline B protocol, **BGE-M3** added; [`satd_report_bge.md`](../navid-experiment/satd_report_bge.md). No BGE training notebook is in the repo; PDFs of both reports are committed |

**Phases planned but not implemented:** 7 (code embeddings: CodeBERT, GraphCodeBERT, UniXcoder, CodeT5), 8 (keyword numeric features from the eight lists), 9 (retrain concatenating keywords onto embeddings).

---

## 6. Shared experimental design

Both pipelines use the same scientific skeleton.

### 6.1 Tasks

1. **Identification** — binary: SATD vs Not-SATD.
2. **Categorization** — four-class on SATD rows only: code/design (C/D), requirement (REQ), test (TES), documentation (DOC).

Models are trained **separately per artifact type** (code comments, issues, commits, pull requests). There is no single “all artifacts” classifier in the reported tables.

### 6.2 Label scheme (“Option A”)

Aligned to the replication study:

| Label | Meaning |
|---|---|
| Not-SATD | Not technical debt |
| C/D | Code or design debt (merged from CODE, DESIGN, CODE/DESIGN, etc.) |
| REQ | Requirement debt |
| TES | Test debt |
| DOC | Documentation debt |

**Dropped** (not used in the reference study): Defect, Architecture, Build, and unmapped / `without_classification` rows.

### 6.3 Metric

**Macro-averaged F1** on the test set, same as the comparison paper. Precision/recall per class were specified in the overview but the committed comparison tables report macro F1.

### 6.4 Embedding and classification (shared recipe)

- Encode text with a frozen encoder; **mean-pool**; **L2-normalize**. E5 uses the `query:` / `passage:` prefix required by that model.
- Vector size **1024** for Qwen3 and E5 (Drive arrays).
- Train **XGBoost** (`n_estimators=200`, `max_depth=6`, `learning_rate=0.1` where recorded) and **logistic regression** (`max_iter=1000`; Pipeline B uses `C=0.5`).
- No fine-tuning of BERT or BiLSTM in this project. Compute was Kaggle T4-class GPU for embedding/augmentation; classifiers on CPU.

Pipeline A: 2 embeddings × 2 classifiers × 4 artifacts × 2 tasks = **32 models**.

Pipeline B Qwen/E5: same 32. Adding BGE-M3 adds another 16 (2 classifiers × 4 artifacts × 2 tasks).

---

## 7. Pipeline A — from-scratch (31 sources)

**Location:** [`notebooks/full_pipeline.ipynb`](../notebooks/full_pipeline.ipynb), [`docs/pipeline_a_from_scratch.md`](pipeline_a_from_scratch.md), [`results/`](../results/), root [`README.md`](../README.md).

**Goal:** Rebuild a corpus from original files, balance classes, embed, classify, compare to Sutoyo / supervisor numbers.

### 7.1 Data: 31 source files

Input directory in the notebook: Kaggle `satd-sources-cleaned`. Each file is tagged with an artifact type from its name. A combined dump `satd_combined.csv` is described as **820,885** rows with many raw label strings before mapping.

Files mapped in the notebook:

**Code comments (21 files)**  
`apache_ant_cleaned.csv`, `apache_jmeter_cleaned.csv`, `argouml_cleaned.csv`, `jfreechart_cleaned.csv`, `jruby_cleaned.csv`, `cppsatd_cleaned_cleaned.csv`, `mlsatd_cleaned.csv`, `manual_annotations_cleaned_cleaned.csv`, `labeled_dataset_cleaned.csv`, `OBrien_789_v2_cleaned.csv`, `satd-dataset-code_comments_cleaned.csv`, `data-augmentation-code_comments_cleaned.csv`, `maldonado_corrected_cleaned.csv`, `duplicate_satd_comment_cleaned.csv`, `unique_satd_comment_cleaned.csv`, `SSATD_COMMENTS_cleaned.csv`, `satd-comments-manual-subclass_cleaned.csv`, `ownership argouml_cleaned.csv`, `ownership_ant_cleaned.csv`, `ownership_jmeter_cleaned.csv`, `ownership_jruby_cleaned.csv`.

**Issues (4)**  
`issue-satd_cleaned.csv`, `satd-dataset-issues_cleaned.csv`, `data-augmentation-issues_cleaned.csv`, `SSATD_ISSUES_cleaned.csv`.

**Commits (3)**  
`satd-dataset-commit_messages_cleaned.csv`, `data-augmentation-commit-messages_cleaned.csv`, `SSATD_COMMITS_cleaned.csv`.

**Pull requests (3)**  
`satd-dataset-pull_requests_cleaned.csv`, `data-augmentation-pull-requests_cleaned.csv`, `SSATD_PULL_cleaned.csv`.

Text columns tried: `satd`, `commenttext`, `comment text`, `text`. Label columns tried: `classification`, `annotation`, `manual_annotation`, `software td type`, `label`, `debt`.

**Important caveat.** The 31-file merge **includes** the paper’s own `data-augmentation-*` files. Pipeline A is “from scratch” in the sense of merging many sources and re-balancing; it is **not** a purely unaugmented raw-only corpus.

### 7.2 Phase 1 — clean

- Tag artifact type from filename.
- Standardize columns and map labels to Option A; drop Defect / Architecture / Build / unmapped.
- Drop text with **two words or fewer**.
- Drop duplicates on `(text, artifact_type)`.

After cleaning ([`pipeline_a_from_scratch.md`](pipeline_a_from_scratch.md)): **~386,646** rows.

| Artifact | Rows | C/D | REQ | TES | DOC | Not-SATD |
|---|---|---|---|---|---|---|
| code_comment | 353,009 | 11,600 | 5,406 | 1,786 | 1,410 | 332,753 |
| issue | 23,128 | 2,189 | 814 | 965 | 1,081 | 18,079 |
| commit | 5,616 | 487 | 390 | 372 | 392 | 3,975 |
| pull_request | 4,893 | 502 | 155 | 247 | 380 | 3,609 |

Output: `satd_with_artifact_type.csv` (on Drive).

### 7.3 Phase 2 — balance

**Why.** Not-SATD dominates (especially comments: 332,753 vs 1,410 DOC). A classifier can ignore SATD.

**How (per artifact).** Target size = size of the **largest SATD class**. Not-SATD is **downsampled** to that size. Smaller SATD classes are **paraphrased up** with T5 `humarin/chatgpt_paraphraser_on_T5_base` on Kaggle GPU. Pull-request REQ (155 examples) is capped at about **3×** original (~465) to avoid repetitive synthetic text.

Output: `satd_balanced.csv`. This **equalizes all five classes** (including Not-SATD). That is a **deliberate deviation** from Sutoyo, who keep a Not-SATD majority.

### 7.4 Phase 3 — split

Stratified **80% / 10% / 10%** train / val / test, seed 42, **after** augmentation. Drive shapes: train **59,105**, val **7,390**, test **7,395** (aligned embedding rows). Same ratio as the paper, but **split after synthetic text** risks leakage (paraphrases of the same original can land in train and test).

### 7.5 Phases 4–6 — embed, train, compare

Qwen3-Embedding-0.6B and E5-Large-v2 vectors saved as `.npy` (Drive). 32 classifiers saved as `{task}_{artifact}_{embedding}_{classifier}.joblib`.

### 7.6 Pipeline A results (macro F1)

Source: [`results/phase5_results_summary.csv`](../results/phase5_results_summary.csv) and phase 6 comparison CSVs. README overall average: **Qwen3 + XGBoost = 0.8648** across artifacts and both tasks.

#### Identification vs BiLSTM+AugGPT (or supervisor BiLSTM on commits)

| Artifact | Baseline | Qwen3 XGB | Qwen3 LogReg | E5 XGB | E5 LogReg | Best ours vs baseline |
|---|---|---|---|---|---|---|
| Code comments | 0.939 | 0.9152 | 0.9079 | 0.8911 | 0.8708 | **Lost** (−0.0238) |
| Issues | 0.878 | 0.7981 | 0.7858 | 0.7779 | 0.7656 | **Lost** (−0.0799) |
| Pull requests | 0.862 | 0.7924 | 0.7476 | 0.7170 | 0.6562 | **Lost** (−0.0696) |
| Commits | 0.910 (supervisor) | 0.6861 | 0.5822 | 0.7508 | 0.6050 | **Lost** (−0.1592) |

**RQ1 on Pipeline A:** frozen embeddings **underperform BiLSTM on every artifact**.

#### Categorization vs BERT+AugGPT (or supervisor BERT on commits)

| Artifact | Baseline | Qwen3 XGB | Qwen3 LogReg | E5 XGB | E5 LogReg | Best ours vs baseline |
|---|---|---|---|---|---|---|
| Code comments | 0.882 | 0.9000 | 0.8556 | **0.9041** | 0.8554 | **Won** (+0.0221) |
| Issues | 0.899 | **0.9410** | 0.8679 | 0.9385 | 0.8851 | **Won** (+0.0420) |
| Pull requests | 0.876 | 0.9270 | 0.8634 | **0.9360** | 0.9264 | **Won** (+0.0600) |
| Commits | 0.9804 (supervisor) | **0.9589** | 0.9117 | 0.9378 | 0.9127 | **Lost** (−0.0215) |

**RQ2 on Pipeline A:** embeddings + XGBoost **beat BERT on 3 of 4 artifacts**; commits remain below 0.980.

**RQ4 on Pipeline A:** XGBoost beat logistic regression in **32 / 32** comparisons.

**Anomaly:** commit identification (0.58–0.75) is far worse than commit categorization (0.94–0.96). Distinguishing debt *types* while failing SATD vs Not-SATD is more consistent with **synthetic Not-SATD / T5 quality** than with “embeddings cannot represent debt types.”

README headline for this pipeline only: frozen embeddings + XGBoost cannot beat a purpose-trained BiLSTM for **detecting** SATD, but they outperform fine-tuned BERT for **categorizing** SATD in 3 of 4 artifact types, without fine-tuning.

---

## 8. Pipeline B — paper’s pre-augmented data

**Location:** [`navid-experiment/`](../navid-experiment/), [`docs/pipeline_b_paper_data.md`](pipeline_b_paper_data.md).

**Goal:** Hold the **paper’s data distribution** (almost) fixed and only change the representation and classifier. This answers RQ3: if we lose identification on Pipeline A, is that the embeddings or the rebuild?

BGE-M3 is an **extension of Pipeline B**, not a third pipeline: same files, same split policy, same heads; only the encoder changes.

### 8.1 Data: four AugGPT files

Semicolon-separated CSVs:

- `data-augmentation-code_comments.csv`
- `data-augmentation-issues.csv`
- `data-augmentation-commit-messages.csv`
- `data-augmentation-pull-requests.csv`

About **109k** rows; **95,704** after dropping NA, short text, and unmapped labels. Four SATD subtypes are already equalized **per artifact**; **Not-SATD remains 65–82%** (paper-like). No T5 five-class equalization.

Stratified 80/10/10 **before** extra augmentation (leakage-aware). Logistic regression `C = 0.5`. The notebook also contains optional **Qwen2.5-1.5B-Instruct** persona rephrasing (“rephrase like a programmer”); the reported Qwen/E5/BGE tables are described as using the paper’s pre-augmented set without extra T5 balancing.

### 8.2 Pipeline B results — Qwen3 vs E5 (3 August 2026)

Source: [`navid-experiment/glm_pipeline_results.txt`](../navid-experiment/glm_pipeline_results.txt) and [`satd_report.md`](../navid-experiment/satd_report.md).

#### Identification (macro F1)

| Artifact | Baseline | Qwen3 XGB | Qwen3 LogReg | E5 XGB | E5 LogReg |
|---|---|---|---|---|---|
| Code comments | 0.939 | **0.9664 (+0.0274)** | 0.9336 | 0.9587 (+0.0197) | 0.9139 |
| Issues | 0.878 | 0.8656 (−0.0124) | 0.8381 | 0.8632 (−0.0148) | 0.8426 |
| Pull requests | 0.862 | 0.8642 (+0.0022) | 0.8180 | **0.8703 (+0.0083)** | 0.8312 |
| Commits | 0.910 | 0.8917 (−0.0183) | 0.8357 | 0.8817 (−0.0283) | 0.8374 |

**RQ1 on Pipeline B:** identification is **competitive**. **Win** on comments (Qwen3) and pull requests (E5). **Loss** on issues and commits (small gaps vs Pipeline A’s collapse).

#### Categorization (macro F1)

| Artifact | Baseline | Qwen3 XGB | Qwen3 LogReg | E5 XGB | E5 LogReg |
|---|---|---|---|---|---|
| Code comments | 0.882 | 0.9470 (+0.0650) | 0.8694 | **0.9581 (+0.0761)** | 0.8638 |
| Issues | 0.899 | 0.9442 (+0.0452) | 0.8748 | **0.9503 (+0.0513)** | 0.8790 |
| Pull requests | 0.876 | **0.9548 (+0.0788)** | 0.8578 | 0.9335 (+0.0575) | 0.9242 |
| Commits | 0.980 | **0.9639 (−0.0161)** | 0.8996 | 0.9487 (−0.0313) | 0.9240 |

**RQ2 on Pipeline B:** again **win 3 of 4**; commits still short of 0.980; **margins larger** than Pipeline A.

Averages over artifacts and tasks: E5 **0.8988**, Qwen3 **0.8953**. Best combination: **Qwen3 + XGBoost 0.9247** vs E5 + XGBoost 0.9206; both XGBoost heads beat both LogReg heads (~0.87).

### 8.3 Pipeline B extension — BGE-M3 (12 August 2026)

Source: [`navid-experiment/satd_report_bge.md`](../navid-experiment/satd_report_bge.md). Training notebook for BGE is **not** in the repo.

XGBoost only in the published comparison tables (LogReg is weaker, especially for BGE):

#### Identification

| Artifact | Baseline | Qwen3 | E5 | BGE-M3 |
|---|---|---|---|---|
| Code comments | 0.939 | **0.9664** | 0.9587 | 0.9594 |
| Issues | 0.878 | **0.8656** | 0.8632 | 0.8533 |
| Pull requests | 0.862 | 0.8642 | **0.8703** | 0.8580 |
| Commits | 0.910 | 0.8917 | 0.8817 | **0.9014** |

Identification averages with XGBoost: Qwen3 0.8970, E5 0.8935, BGE-M3 0.8930 (spread under 0.5 points). BGE is closest to BiLSTM on **commits** (0.9014 vs 0.910).

#### Categorization

| Artifact | Baseline | Qwen3 | E5 | BGE-M3 |
|---|---|---|---|---|
| Code comments | 0.882 | 0.9470 | **0.9581** | 0.9558 |
| Issues | 0.899 | 0.9442 | 0.9503 | **0.9562** |
| Pull requests | 0.876 | 0.9548 | 0.9335 | **0.9644** |
| Commits | 0.980 | **0.9639** | 0.9487 | 0.9603 |

Categorization averages with XGBoost: BGE-M3 **0.9592**, Qwen3 0.9525, E5 0.9477. Largest BERT margin in the project: BGE on pull requests **+0.0884**. No encoder hits commit categorization **0.980**.

Overall (both tasks): **BGE-M3 + XGBoost 0.9261**, Qwen3 + XGBoost 0.9248, E5 + XGBoost 0.9206. BGE + LogReg is much worse (combined ~0.8489): BGE “wants” a tree head.

---

## 9. Head-to-head: what is robust vs what flips

This is the answer to **RQ3**. Do not merge the two pipelines into one “our F1.”

| Claim | Pipeline A (31 files, T5, 5-class equal, split after aug) | Pipeline B (paper AugGPT files, Not-SATD majority, split before extra aug) |
|---|---|---|
| Beat BERT on comments / issues / PRs (categorization, XGB) | Yes | Yes (larger gaps) |
| Beat BERT on commits (categorization) | No | No |
| Beat BiLSTM on identification | **Never** | **Yes on comments and PRs**; no on issues and commits |
| XGBoost vs LogReg | XGB wins 32/32 | XGB wins for Qwen, E5, and BGE |
| Qwen3 vs E5 | Mixed; README picks Qwen3 overall 0.8648 | Near tie (~0.90 average); BGE slightly best overall on this data |

**Robust finding.** Frozen embeddings + XGBoost are a **cheap alternative for SATD categorization** on comments, issues, and pull requests. That held on two different data protocols.

**Data-dependent finding.** Identification against BiLSTM **is not a property of Qwen vs E5 vs BGE**. On the paper’s augmented distribution it is close (sometimes better). On the T5-equalized rebuild it collapses, worst on commits. Equalizing all five classes did **not** reproduce the paper. The reports interpret this as evidence that Sutoyo identification numbers are tied to **AugGPT data**, not only to BiLSTM—**suggestive**, because Pipeline A also has a larger, mixed 31-file corpus (confound).

**RQ4.** Classifier choice dominates embedding choice. Switching Qwen / E5 / BGE moves scores by tenths of a point (with XGBoost). Switching XGBoost vs LogReg moves more. Switching Pipeline A vs B moves identification the most.

---

## 10. Compute limits and larger models (honest claim)

Work used **frozen** encoders in the **0.6B / few-hundred-million** range, Kaggle T4-class GPUs, and no SATD-specific transformer fine-tuning. That was a resource constraint, not a claim that 0.6B is optimal.

**What the results already show (not speculation):**

- Three different frozen embeddings on the **same** Pipeline B data are nearly interchangeable for identification.
- Categorization is already **above** fine-tuned BERT on three artifacts with these small frozen models.
- The large identification drop is aligned with **dataset and augmentation protocol**, not with “we lacked a 7B encoder.”

**What remains an open empirical question:**

- A larger embedding model, or **fine-tuning** BERT/BiLSTM-scale models on Pipeline A, **might** raise identification F1, especially on small artifacts (commits, PRs).
- It **might not**. If the paper’s identification scores are mostly AugGPT-distribution effects, a bigger encoder on T5-equalized data could stay weak. The project must not promise that “more parameters will make results more accurate.”

Future work that is already specified in-repo (Phases 7–9) is a better next step than an unbounded “train a huge model”: code-specific encoders on the **same** splits, then keyword features concatenated to the 1024-d vectors, then a controlled retrain.

---

## 11. Limitations and gaps in the repository

- **Two stories in the README vs navid reports.** The root README identification headline is Pipeline A only. Pipeline B identification wins on comments/PRs are in `navid-experiment/`. A thesis must present both.
- **Pipeline A leakage risk:** split after paraphrase.
- **Pipeline A contamination:** paper `data-augmentation-*` files sit inside the 31-file list.
- **Commit baselines** mixed paper vs supervisor replication.
- **`scripts/` is empty** though the README mentions per-phase Python scripts.
- **`docs/teammate_guide.md` is cited in the README and is missing.**
- **BGE-M3 code is not checked in**; only the report/PDF.
- **Raw CSVs, `.npy`, `.joblib`** are on Drive, not in git.
- **Phases 7–9** not run. Keyword files are ready.
- **Macro F1 only** in committed tables; error analysis / per-class confusion matrices are not in the repo.
- Formal related work is thin: one arXiv paper plus filename-level dataset lineage.

---

## 12. Future work (from the repo)

| Phase | Plan | Status |
|---|---|---|
| 7 | Drop-in CodeBERT, GraphCodeBERT, UniXcoder, CodeT5 embeddings; same classifiers and splits | Not started (`full_pipeline.ipynb` Phase 7 cell empty) |
| 8 | Parse eight KeyBERT lists; top-100 phrases; 8 weighted match scores + `kw_total_signal`; save aligned `keyword_features_{train,val,test}` | Input files present under `data/keywords/` |
| 9 | Concatenate 9 keyword features onto 1024-d embeddings (shape n × 1033); retrain 32 models; compare to Phase 5/6 | Not started |

Plus the open question in §10: larger or fine-tuned encoders, reported **per pipeline**, not as a guaranteed accuracy jump.

---

## 13. Short answers a thesis committee can use

**What developers do / why SATD matters.** They leave written admissions of debt in comments, issues, commits, and PRs. Those admissions are signals for repayment; missing them lets cost grow. Manual review does not scale; types of debt differ.

**What we compared against.** Sutoyo et al. (2024): BiLSTM+AugGPT for detection, BERT+AugGPT for typing, four artifacts.

**What we implemented.** Two pipelines with the same tasks and frozen-embedding + XGBoost/LogReg stack. Pipeline A rebuilt from 31 files with T5 balancing. Pipeline B trained on the paper’s four augmented CSVs; later the same protocol with BGE-M3.

**What we found.** Categorization: we beat BERT on three artifacts in **both** pipelines. Identification: we lose everywhere on Pipeline A; we win comments and PRs on Pipeline B. XGBoost always beat LogReg. Embedding brand mattered less than data protocol and classifier. Commits are the hard artifact. Bigger models are untested; existing ablations do **not** imply they will automatically win.

---

## Appendix A — File map

| Path | Supports |
|---|---|
| [`README.md`](../README.md) | Citation; Pipeline A headlines; Drive link; phase list |
| [`docs/pipeline_a_from_scratch.md`](pipeline_a_from_scratch.md) | Pipeline A methods, 31 files, Phases 1–9 |
| [`docs/pipeline_b_paper_data.md`](pipeline_b_paper_data.md) | Pipeline B methods, four AugGPT files, BGE |
| [`data/README.md`](../data/README.md) | Drive layout; embedding shapes; keyword files |
| [`notebooks/full_pipeline.ipynb`](../notebooks/full_pipeline.ipynb) | Pipeline A implementation; 31-file map |
| [`results/phase5_results_summary.csv`](../results/phase5_results_summary.csv) | All 32 Pipeline A macro F1s |
| [`results/phase6_identification_comparison.csv`](../results/phase6_identification_comparison.csv) | Pipeline A vs BiLSTM |
| [`results/phase6_categorization_comparison.csv`](../results/phase6_categorization_comparison.csv) | Pipeline A vs BERT |
| [`results/SATD_Findings_Summary.pdf`](../results/SATD_Findings_Summary.pdf) | Pipeline A PDF summary |
| [`navid-experiment/glm_pipeline_artifacts_separated.ipynb`](../navid-experiment/glm_pipeline_artifacts_separated.ipynb) | Pipeline B code; four-file load; optional Qwen2.5 aug |
| [`navid-experiment/glm_pipeline_results.txt`](../navid-experiment/glm_pipeline_results.txt) | Pipeline B Qwen/E5 vs baseline |
| [`navid-experiment/satd_report.md`](../navid-experiment/satd_report.md) | Dual-pipeline interpretation (3 Aug 2026) |
| [`navid-experiment/satd_report_bge.md`](../navid-experiment/satd_report_bge.md) | BGE-M3 on Pipeline B (12 Aug 2026) |
| [`data/keywords/`](../data/keywords/) | Future Phase 8 inputs |

## Appendix B — Drive artifacts (not in git)

From [`data/README.md`](../data/README.md):

- `01_raw_sources/satd_sources.zip` — original CSV/XLSX
- `02_processed_data/` — `satd_with_artifact_type.csv`, `satd_balanced.csv`, `satd_{train,val,test}.csv`, label CSVs
- `03_embeddings/` — `qwen3_*.npy`, `e5_*.npy` (n × 1024)
- `04_trained_models/` — 32 `.joblib` files

Folder: [SATD Project — Data and Models](https://drive.google.com/drive/folders/1E-jzrNGE2NyKEsrI8Ud9Phk3gx_8a2dD).
