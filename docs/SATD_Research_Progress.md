# SATD Embedding Comparison

**Research Progress and Experimental Findings**  
25 August 2026

Frozen text embeddings for identifying and categorizing Self-Admitted Technical Debt across software artifacts.

**Primary comparison:** Sutoyo, Avgeriou, and Capiluppi (2024)

---

## Contents

1. [The practical problem](#1-the-practical-problem)
2. [Research basis and field context](#2-research-basis-and-field-context)
3. [Problem statement](#3-problem-statement)
4. [Research questions](#4-research-questions)
5. [Chronology of the work](#5-chronology-of-the-work)
6. [Shared experimental design](#6-shared-experimental-design)
7. [Pipeline A — 31-source rebuild](#7-pipeline-a--from-scratch-rebuild-from-31-sources)
8. [Pipeline B — paper AugGPT data](#8-pipeline-b--paper-pre-augmented-data)
9. [Head-to-head interpretation](#9-head-to-head-interpretation-what-is-robust-and-what-changes)
10. [Compute limits and larger models](#10-compute-limits-and-larger-models)
11. [Limitations and reproducibility](#11-limitations-and-reproducibility-considerations)
12. [Future research directions](#12-future-research-directions)

---

## 1. The practical problem

Software developers often accept temporary compromises: a workaround that is good enough for now, a missing test, outdated documentation, a design shortcut, or a requirement that still needs to be implemented. When developers explicitly acknowledge such debt in text—for example in a source-code comment, issue, commit message, or pull-request discussion—that admission is called **Self-Admitted Technical Debt (SATD)**.

SATD is valuable because the developer has already supplied a direct signal that some work remains. At the same time, it creates a management problem: large projects generate too much text for people to inspect manually, and different forms of debt require different responses.

- If SATD is not tracked, it can accumulate and make future changes harder, increase defect risk, and slow onboarding.
- Manually scanning comments, issues, commits, and pull requests does not scale across large repositories.
- Identification and categorization are different tasks. First we ask whether text contains SATD; then, for SATD text, we ask what type of debt it represents.
- Language differs across artifact types. A long issue description and a short commit message do not provide the same amount or style of context.

**Research motivation.** The baseline paper reports strong SATD results with task-specific BiLSTM and BERT models plus GPT-style data augmentation. This project tests whether frozen modern text embeddings with lightweight classifiers can provide comparable performance at lower training cost—and whether the conclusion changes when the data protocol changes.

---

## 2. Research basis and field context

### 2.1 Primary baseline study

The replication target is Sutoyo, E., Avgeriou, P., and Capiluppi, A. (2024), *Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt* ([arXiv:2410.15804](https://arxiv.org/abs/2410.15804)). In this project, its reported or replicated baselines are used as the reference point for both SATD identification and categorization.

| Task | Baseline setup used here | This project |
|---|---|---|
| Identification | GloVe + BiLSTM + AugGPT | Frozen embeddings + XGBoost / logistic regression |
| Categorization | Fine-tuned BERT + AugGPT | Same frozen embeddings + same classifier heads |

For commit messages, the comparison CSVs often use the supervisor replication values (BiLSTM 0.91 and BERT 0.9804). The comments, issues, and pull-request rows are labeled as the paper baselines. This distinction matters when presenting the final thesis tables.

### 2.2 Models and supporting resources used

- Frozen encoders: `Qwen/Qwen3-Embedding-0.6B` and `intfloat/e5-large-v2` in both pipelines; `BAAI/bge-m3` as an extension of Pipeline B.
- Pipeline A augmentation: `humarin/chatgpt_paraphraser_on_T5_base`.
- Pipeline B notebook: optional `Qwen/Qwen2.5-1.5B-Instruct` paraphrasing is present, but the reported Qwen/E5/BGE tables are described as using the paper pre-augmented set.
- Eight KeyBERT-derived SATD keyword lists exist in `data/keywords/` for a planned future feature phase; they have not yet been used in the reported models.

### 2.3 Dataset lineage and related-work scope

Several established SATD dataset families are represented in the collected source files, including Maldonado-style comment corpora, O’Brien, CppSATD, MLSATD, SSATD, and multi-artifact SATD datasets. These names establish dataset lineage; formal bibliographic treatment will be added only after the corresponding papers are reviewed and verified.

---

## 3. Problem statement

The chosen baseline relies on purpose-trained sequence models and GPT-style augmentation. It is unclear whether frozen general-purpose embeddings plus lightweight classifiers can replace that stack with much less training, and whether published performance is mainly a property of the model architecture or strongly tied to the paper’s augmented data distribution.

The study is intended to separate architectural effects from data-distribution effects. If frozen embeddings can match or exceed BERT for debt categorization without task-specific fine-tuning, they may provide a computationally lighter alternative. If identification is competitive only when the AugGPT distribution is retained, the result should be interpreted as distribution-specific rather than as a general superiority claim for SATD detection.

---

## 4. Research questions

**RQ1 — Identification.** Can frozen embeddings (Qwen3-Embedding-0.6B, E5-Large-v2, and later BGE-M3) combined with XGBoost or logistic regression match the Sutoyo-style BiLSTM+AugGPT macro F1 for SATD versus Not-SATD, separately for each artifact type?

**RQ2 — Categorization.** Can the same stack match or beat Sutoyo-style BERT+AugGPT macro F1 for code/design, requirement, test, and documentation debt on SATD rows only?

**RQ3 — Data protocol.** How much do the conclusions change between Pipeline A, which rebuilds and rebalances 31 source files, and Pipeline B, which keeps the paper’s four AugGPT files and its Not-SATD majority?

**RQ4 — Embedding vs classifier.** Does performance change more when the embedding model changes, or when the classifier head changes?

---

## 5. Chronology of the work

The experimental work developed into two related pipelines. Pipeline A evaluates a broad 31-source reconstruction and rebalancing strategy. Pipeline B keeps the baseline study’s augmented data distribution substantially closer to its original form, allowing the effect of the representation and classifier to be examined with fewer data-protocol changes. BGE-M3 was subsequently evaluated as an encoder extension of Pipeline B.

| Date | Milestone |
|---|---|
| ~20 Jun 2026 | Pipeline A findings summary PDF appears in `results/`. |
| 7 Jul 2026 | Pipeline A notebook, keyword files, Phase 5/6 result CSVs, and README document the from-scratch experiment. |
| 3 Aug 2026 | Pipeline B notebook and report compare the paper-data protocol with Pipeline A using Qwen3 and E5. |
| 12 Aug 2026 | BGE-M3 is added to the same Pipeline B protocol; report committed, but the BGE training notebook is not in the repository. |

---

## 6. Shared experimental design

Despite their different data protocols, both pipelines use the same experimental skeleton. This makes the contrast between them meaningful: the task definitions, artifact separation, evaluation metric, embedding strategy, and classifier families remain aligned.

### 6.1 Tasks and artifact separation

1. **Identification** — binary classification: SATD versus Not-SATD.
2. **Categorization** — four-class classification on SATD rows only: code/design (C/D), requirement (REQ), test (TES), and documentation (DOC).

A separate model is trained for each artifact type: code comments, issues, commit messages, and pull requests. The reported results do not use one universal “all artifacts” classifier.

### 6.2 Label scheme

| Label | Meaning |
|---|---|
| Not-SATD | No technical debt admission |
| C/D | Code or design debt; CODE / DESIGN / CODE-DESIGN variants merged |
| REQ | Requirement debt |
| TES | Test debt |
| DOC | Documentation debt |

Defect, Architecture, Build, and unmapped or `without_classification` rows are excluded because they are not part of the chosen reference-study label scheme.

### 6.3 Evaluation metric

The primary metric is **macro-averaged F1** on the held-out test set. Macro F1 gives each class equal importance, which is useful when class frequencies are uneven. The committed comparison tables focus on macro F1 rather than complete per-class precision and recall.

### 6.4 Shared embedding and classifier recipe

- Text is encoded by a frozen model, mean-pooled, and L2-normalized. E5 uses its `query:` / `passage:` prefixes.
- Qwen3 and E5 embeddings are 1024-dimensional in the saved arrays.
- Classifier heads are XGBoost (`n_estimators=200`, `max_depth=6`, `learning_rate=0.1` where recorded) and logistic regression (`max_iter=1000`; Pipeline B records `C=0.5`).
- No BERT or BiLSTM fine-tuning is performed in this project. Embedding and augmentation used T4-class GPU compute; classifier training is CPU-suitable.

**Model count.** Pipeline A: 2 embeddings × 2 classifiers × 4 artifacts × 2 tasks = 32 models. Pipeline B with Qwen3/E5: another 32. Adding BGE-M3 to Pipeline B adds 16 more combinations.

---

## 7. Pipeline A — from-scratch rebuild from 31 sources

Pipeline A asks whether a new corpus can be rebuilt from many available SATD sources, rebalanced, embedded, and classified strongly enough to match the reference study. It is “from scratch” in the sense of the merge-and-rebalance process; it is not a purely raw corpus because the 31 inputs also contain the paper’s data-augmentation files.

### 7.1 Data collection and cleaning

The notebook loads 31 cleaned CSV files and infers artifact type from each filename. A combined dump is described as containing **820,885** rows before label mapping. The 31 files consist of 21 code-comment files, four issue files, three commit files, and three pull-request files.

**Caveat.** Pipeline A includes `data-augmentation-code_comments`, `data-augmentation-issues`, `data-augmentation-commit-messages`, and `data-augmentation-pull-requests` among the 31 sources. Therefore, it should be described as a **multi-source rebuild**, not as an unaugmented raw-only corpus.

Cleaning steps:

- Tag each row with artifact type from the source filename.
- Standardize text and label column names, then map labels to the five-class Option A scheme.
- Drop Defect, Architecture, Build, unmapped labels, blank text, and text of two words or fewer.
- Drop duplicates on the pair `(text, artifact_type)`.

After cleaning, approximately **386,646** rows remain.

| Artifact | Rows | C/D | REQ | TES | DOC | Not-SATD |
|---|---:|---:|---:|---:|---:|---:|
| Code comments | 353,009 | 11,600 | 5,406 | 1,786 | 1,410 | 332,753 |
| Issues | 23,128 | 2,189 | 814 | 965 | 1,081 | 18,079 |
| Commits | 5,616 | 487 | 390 | 372 | 392 | 3,975 |
| Pull requests | 4,893 | 502 | 155 | 247 | 380 | 3,609 |

### 7.2 Phase 2 — class balancing

The cleaned corpus is heavily dominated by Not-SATD, especially in code comments. Pipeline A therefore deliberately changes the class distribution before training.

- For each artifact, set the target class size to the largest SATD subtype.
- Downsample Not-SATD to that target.
- Paraphrase smaller SATD classes upward using `humarin/chatgpt_paraphraser_on_T5_base`.
- For pull-request REQ, cap augmentation at roughly three times the original count (about 465) to reduce repetitive synthetic text.

**Key methodological difference.** Pipeline A equalizes all five classes, including Not-SATD. The Sutoyo-style data keep Not-SATD as the majority class. This deliberate difference later becomes central to the interpretation of identification performance.

### 7.3 Phase 3 — train/validation/test split

A stratified 80/10/10 split with seed 42 is applied **after** augmentation. Saved shapes are 59,105 training rows, 7,390 validation rows, and 7,395 test rows. Because paraphrasing happens before the split, synthetic siblings of the same original may appear in different splits; this is a potential leakage risk.

### 7.4 Phases 4–6 — embeddings, classifiers, comparison

Qwen3-Embedding-0.6B and E5-Large-v2 embeddings are generated and saved as row-aligned arrays. For each task, artifact, embedding, and classifier combination, a separate model is trained and evaluated. The result tables below report held-out macro F1 against the paper or supervisor baseline.

### 7.5 Pipeline A results — identification

| Artifact | Baseline | Qwen3 XGB | Qwen3 LR | E5 XGB | E5 LR | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Code comments | 0.939 | 0.9152 | 0.9079 | 0.8911 | 0.8708 | Lost (−0.0238) |
| Issues | 0.878 | 0.7981 | 0.7858 | 0.7779 | 0.7656 | Lost (−0.0799) |
| Pull requests | 0.862 | 0.7924 | 0.7476 | 0.7170 | 0.6562 | Lost (−0.0696) |
| Commits | 0.910\* | 0.6861 | 0.5822 | 0.7508 | 0.6050 | Lost (−0.1592) |

\* Commit baseline is the supervisor replication value. Pipeline A underperforms the BiLSTM baseline on all four artifact types. The largest gap appears on commit identification.

### 7.6 Pipeline A results — categorization

| Artifact | Baseline | Qwen3 XGB | Qwen3 LR | E5 XGB | E5 LR | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Code comments | 0.882 | 0.9000 | 0.8556 | 0.9041 | 0.8554 | Won (+0.0221) |
| Issues | 0.899 | 0.9410 | 0.8679 | 0.9385 | 0.8851 | Won (+0.0420) |
| Pull requests | 0.876 | 0.9270 | 0.8634 | 0.9360 | 0.9264 | Won (+0.0600) |
| Commits | 0.9804\* | 0.9589 | 0.9117 | 0.9378 | 0.9127 | Lost (−0.0215) |

\* Commit baseline is the supervisor replication value. In categorization, frozen embeddings with XGBoost beat BERT on code comments, issues, and pull requests, but remain below the commit baseline.

**Interpretation.** The important contrast is not simply “good” versus “bad” performance. The same representation can categorize SATD types very well while struggling to separate SATD from Not-SATD under this rebuilt and T5-equalized distribution. That pattern points toward the data and augmentation protocol as a major factor.

XGBoost beats logistic regression in all 32 Pipeline A comparisons. The README reports Qwen3 + XGBoost at an overall average macro F1 of **0.8648** across the two tasks and four artifacts.

---

## 8. Pipeline B — paper pre-augmented data

Pipeline B isolates the representation and classifier from the broader 31-source reconstruction. It uses the four AugGPT CSVs and retains a paper-like Not-SATD majority rather than equalizing all five classes. This provides a more controlled comparison for determining whether the identification gap observed in Pipeline A is primarily associated with the embedding/classifier stack or with the changed data protocol.

### 8.1 Data and split policy

- Files: `data-augmentation-code_comments.csv`, `data-augmentation-issues.csv`, `data-augmentation-commit-messages.csv`, and `data-augmentation-pull-requests.csv`.
- Approximately 109,000 rows are loaded; **95,704** remain after dropping missing, very short, and unmapped rows.
- The four SATD subtypes are already equalized per artifact, while Not-SATD remains the majority at roughly 65–82%.
- The stratified 80/10/10 split occurs **before** any optional extra augmentation, reducing the synthetic-sibling leakage risk present in Pipeline A.
- Logistic regression uses `C=0.5`. The notebook contains optional Qwen2.5 rephrasing, but the reported Qwen/E5/BGE tables are described as using the paper’s pre-augmented dataset without Pipeline A’s five-class T5 equalization.

### 8.2 Qwen3 and E5 results — identification

| Artifact | Baseline | Qwen3 XGB | Qwen3 LR | E5 XGB | E5 LR |
|---|---:|---:|---:|---:|---:|
| Code comments | 0.939 | 0.9664 (+0.0274) | 0.9336 | 0.9587 (+0.0197) | 0.9139 |
| Issues | 0.878 | 0.8656 (−0.0124) | 0.8381 | 0.8632 (−0.0148) | 0.8426 |
| Pull requests | 0.862 | 0.8642 (+0.0022) | 0.8180 | 0.8703 (+0.0083) | 0.8312 |
| Commits | 0.910 | 0.8917 (−0.0183) | 0.8357 | 0.8817 (−0.0283) | 0.8374 |

**RQ1 on Pipeline B.** Identification becomes competitive. Qwen3 beats the baseline on code comments, and E5 beats it on pull requests. Issues and commits remain below baseline, but the gaps are small compared with the collapse seen in Pipeline A.

### 8.3 Qwen3 and E5 results — categorization

| Artifact | Baseline | Qwen3 XGB | Qwen3 LR | E5 XGB | E5 LR |
|---|---:|---:|---:|---:|---:|
| Code comments | 0.882 | 0.9470 (+0.0650) | 0.8694 | 0.9581 (+0.0761) | 0.8638 |
| Issues | 0.899 | 0.9442 (+0.0452) | 0.8748 | 0.9503 (+0.0513) | 0.8790 |
| Pull requests | 0.876 | 0.9548 (+0.0788) | 0.8578 | 0.9335 (+0.0575) | 0.9242 |
| Commits | 0.980 | 0.9639 (−0.0161) | 0.8996 | 0.9487 (−0.0313) | 0.9240 |

Categorization again beats BERT on code comments, issues, and pull requests, with larger margins than in Pipeline A. Commits remain the only artifact where the BERT baseline stays ahead.

Across artifacts and both tasks, the reported averages are E5 **0.8988** and Qwen3 **0.8953** when aggregating the embedding families, while the best model combination is Qwen3 + XGBoost at approximately **0.9247**. XGBoost outperforms the logistic-regression heads.

### 8.4 BGE-M3 extension — same Pipeline B protocol

BGE-M3 is not a third pipeline. The same four files, split policy, tasks, and classifier families are reused; only the encoder changes. The BGE report is dated 12 August 2026. The training notebook for this extension is not currently checked into the repository.

**BGE-M3 identification with XGBoost**

| Artifact | Baseline | Qwen3 | E5 | BGE-M3 |
|---|---:|---:|---:|---:|
| Code comments | 0.939 | 0.9664 | 0.9587 | 0.9594 |
| Issues | 0.878 | 0.8656 | 0.8632 | 0.8533 |
| Pull requests | 0.862 | 0.8642 | 0.8703 | 0.8580 |
| Commits | 0.910 | 0.8917 | 0.8817 | 0.9014 |

Identification averages with XGBoost are very close: Qwen3 0.8970, E5 0.8935, and BGE-M3 0.8930. BGE-M3 is the closest encoder to the commit BiLSTM baseline, at 0.9014 versus 0.910.

**BGE-M3 categorization with XGBoost**

| Artifact | Baseline | Qwen3 | E5 | BGE-M3 |
|---|---:|---:|---:|---:|
| Code comments | 0.882 | 0.9470 | 0.9581 | 0.9558 |
| Issues | 0.899 | 0.9442 | 0.9503 | 0.9562 |
| Pull requests | 0.876 | 0.9548 | 0.9335 | 0.9644 |
| Commits | 0.980 | 0.9639 | 0.9487 | 0.9603 |

**BGE result.** BGE-M3 has the strongest XGBoost categorization average (**0.9592**) and reaches **0.9644** on pull requests—a +0.0884 gap over the BERT baseline. No tested encoder reaches the 0.980 commit categorization baseline.

Across both tasks, the report gives BGE-M3 + XGBoost **0.9261**, Qwen3 + XGBoost about **0.9248**, and E5 + XGBoost **0.9206**. BGE-M3 with logistic regression is substantially weaker (combined about 0.8489), reinforcing the importance of the classifier head.

---

## 9. Head-to-head interpretation: what is robust and what changes

The two pipelines should not be collapsed into a single “our F1” number. They answer different questions. The strongest interpretation comes from separating findings that remain stable across both protocols from findings that flip when the data protocol changes.

| Claim | Pipeline A | Pipeline B |
|---|---|---|
| Categorization beats BERT on comments / issues / PRs | Yes | Yes, with larger margins |
| Categorization beats BERT on commits | No | No |
| Identification beats BiLSTM | No — loses all four | Yes on comments and pull requests; no on issues and commits |
| XGBoost vs logistic regression | XGBoost wins 32/32 | XGBoost wins across Qwen3, E5, and BGE |
| Embedding choice | Mixed; Qwen3 headline overall | Qwen3/E5 near-tie; BGE slightly best overall with XGBoost |

**Robust finding.** Frozen embeddings + XGBoost are a strong, computationally cheaper alternative for SATD categorization on code comments, issues, and pull requests. This result survives both data protocols.

**Data-dependent finding.** Identification performance is highly sensitive to the data protocol. On the paper’s AugGPT distribution, frozen embeddings are close to BiLSTM and sometimes better. On the 31-source T5-equalized rebuild, identification drops sharply. This is suggestive evidence that augmentation and distribution matter greatly, but Pipeline A also changes the source mixture, so the effect cannot be attributed to one factor alone.

**RQ4.** Classifier choice appears more influential than swapping among Qwen3, E5, and BGE-M3 on the same data. The largest identification shift comes from Pipeline A versus Pipeline B, not from changing the frozen encoder.

---

## 10. Compute limits and larger models

The experiments use frozen encoders in roughly the 0.6B / few-hundred-million-parameter range, T4-class GPUs for embedding or augmentation, and lightweight CPU classifiers. This was a practical resource constraint. It is not evidence that a 0.6B encoder is theoretically optimal for SATD.

**What the current results support**

- On the same Pipeline B data, three different frozen embeddings are nearly interchangeable for identification.
- For categorization, the tested frozen embeddings already exceed the fine-tuned BERT baseline on three artifact types.
- The largest identification change aligns with the data and augmentation protocol, not with the absence of a much larger encoder.

**What remains open**

- A larger embedding model or task-specific fine-tuning might improve identification, especially on small and terse artifacts such as commits and pull requests.
- That improvement is not guaranteed. If much of the baseline advantage comes from AugGPT-distribution effects, increasing model size alone may not recover the gap on Pipeline A.

---

## 11. Limitations and reproducibility considerations

1. Results from Pipeline A and Pipeline B must be reported separately because they evaluate materially different data protocols.
2. Pipeline A splits after paraphrasing, creating a potential leakage risk between synthetic siblings.
3. Pipeline A includes the paper’s own data-augmentation files inside the 31-source merge, so it is not a purely raw-corpus reconstruction.
4. Commit baselines mix paper values and supervisor replication values; tables should label these explicitly.
5. Reproducibility materials should be consolidated so that the final thesis package contains a complete, traceable implementation for every reported experiment.
6. Documentation should be standardized before final submission so that experimental settings, data provenance, and result-generation steps can be independently followed.
7. The BGE-M3 extension currently has reported results but requires its training implementation to be archived alongside the other experiments.
8. Large processed datasets, embedding arrays, and trained models are stored separately from the main codebase; their versions and paths should be fixed and documented for final reproducibility.
9. Committed result tables primarily report macro F1; detailed error analysis and confusion matrices are not part of the current repository.
10. The formal related-work review is currently limited and should be expanded beyond the primary comparison study before thesis submission.

---

## 12. Future research directions

Two directions are currently of particular interest for extending this work: incorporating SATD research into the context of Agile software development, and investigating whether SATD is associated with developer emotion and with the time required to resolve issues.

**Additional open experiment.** Larger or fine-tuned encoders may be tested later, but their effect should be reported as an empirical result for each pipeline—not assumed in advance.
