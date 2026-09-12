# Final Presentation Slide Guide — SATD Embedding Comparison

**Course:** SWE 4606 / CSE 4610 Design Project  
**Total time:** 10 minutes (slides ≈ **7 min** + live demo ≈ **3 min**) + max 5 min Q/A  
**Slide budget:** up to **20 slides** (this deck uses **20**)  
**Demo:** Gradio app in `Demo/` — keep slide text about the demo **light**; you will update the demo separately.

**All previous course guidelines still apply** (overview, research insights, implementation, solution, field contribution, future work, multi-angle coverage, ~9–10 papers, demo timing).  
**New rule:** the mandatory section list below **must** appear in the deck (section titles on slides should match closely).

---

## Mandatory presentation sections (must-have)

These sections are **required**. The 20-slide blueprint below covers each of them. Do not drop any.

| # | Required section | Covered by slide(s) | Section title to put on slides |
|---|---|---|---|
| 1 | Project title + Team members’ info (**Name, ID**) | **1** | Title / Team |
| 2 | Introduction | **2–4** | Introduction |
| 3 | Related Works | **5–6** | Related Works |
| 4 | Methodology | **7–9** | Methodology |
| 5 | Implementation | **10–12, 17** | Implementation |
| 6 | Results | **13–16** | Results |
| 7 | Challenges and Limitations | **18** | Challenges and Limitations |
| 8 | Conclusion and Future Direction | **19** | Conclusion and Future Direction |
| 9 | Github Link and Individual Contribution | **20** | GitHub & Individual Contribution |

**GitHub (put on Slide 20, optionally tiny footer on Slide 1):**  
https://github.com/A-Piyas-04/SATD-embedding-DP2

---

## Coverage audit — every research angle must appear

Your project is **not one experiment**. It is the **same SATD topic studied from multiple angles**. Every workstream below must be visible on slides and/or spoken.

| # | Research angle (what is distinct) | Where it lives in the repo | Must appear on | Status to say honestly |
|---|---|---|---|---|
| 1 | **Pipeline A — from-scratch rebuild** (31 sources + T5 5-class balance) | `notebooks/full_pipeline.ipynb`, `results/`, `docs/pipeline_a_from_scratch.md` | Slides **8, 10, 13–16** | Complete (phases 1–6) |
| 2 | **Pipeline B — paper-data protocol** | `navid-experiment/glm_pipeline_artifacts_separated.ipynb`, `satd_report.md/.pdf` | Slides **8, 11, 13–16** | Complete |
| 3 | **BGE-M3 encoder extension** | `navid-experiment/satd_report_bge.md/.pdf` | Slides **8, 11–12, 13–16** | Results complete; training notebook not in repo |
| 4 | **Dual-protocol synthesis** | `docs/SATD_Research_Progress.md`, root `README.md`, Navid reports | Slides **8, 15–16, 19** | Complete |
| 5 | **Applied large-scale inference** | `musaddiq_rafi/` notebooks + README | Slides **8, 17** | Notebooks ready; batch result upload may still be in progress — say so |
| 6 | **Optimized batch inference engineering** | `musaddiq_rafi/satd-classifications-optimized.ipynb`, `notebooks/satd-classifications-optimized V2.ipynb` | Slide **17** (one bullet) | Implementation ready |
| 7 | **Live system / demo** | `Demo/` | Slide **17** + **3 min live demo** | Presentation-ready if models preloaded |
| 8 | **Prepared keyword / Phase 7–9 track** | `data/keywords/`, planned phases in pipeline A docs | Slide **19** | Planned / prepared |
| 9 | **Literature exploration** (~9–10 papers) | report / your notes | Slides **5–6** | Done as study; list titles when ready |
| 10 | **Limitations & challenges** | progress report / README | Slide **18** (dedicated) + Q/A | Required |

**Do not present as separate “researches” on slides:** V2 vs Musaddiq optimized = same applied track. Optional Qwen2.5 rephrase cell in Pipeline B notebook is **not** the reported protocol.

**Faculty-facing story:** one topic → **multiple complementary angles** (rebuild vs paper protocol vs encoder ablation vs real-world application vs live tool vs next-step features).

---

## How this guide maps to older course guidelines + new must-haves

| Expectation | Where it lives |
|---|---|
| Project title + Name/ID | Slide 1 |
| Brief overview / Introduction | Slides 2–4 |
| Related works (~9–10 papers) | Slides 5–6 |
| Key research insights / Methodology | Slides 7–9, 15–16 |
| Implementation details | Slides 10–12, 17 |
| Results / final solution evidence | Slides 13–16 + Demo |
| Challenges and Limitations | Slide 18 |
| Contribution + future (CO5) | Slide 19 |
| GitHub + individual contribution (CO2) | Slide 20 |
| Ethical / societal (CO3) | Slide 4 (brief) + Q/A |
| Formal presentation + Q/A (CO4) | Full deck + demo + Q/A |

---

## Timing plan (strict)

| Block | Minutes | Slides | Required section |
|---|---:|---|---|
| Title + Introduction | ~1:00 | 1–4 | Title/Team + Introduction |
| Related Works | ~0:45 | 5–6 | Related Works |
| Methodology | ~1:15 | 7–9 | Methodology |
| Implementation | ~1:15 | 10–12 | Implementation |
| Results | ~1:30 | 13–16 | Results |
| Applied implementation + demo bridge | ~0:30 | 17 | Implementation (applied) |
| **Live demo** | **~3:00** | after 17 | Working system |
| Challenges → Conclusion → GitHub | ~0:45 | 18–20 | Challenges · Conclusion/Future · GitHub/Contributions |
| **Total talk** | **~10:00** | | |

**Practical tip:** If over time, shorten speech on slides 10–11 and 6 first. **Never cut** slides 1, 5, 8, 13–14, 17, **18**, **19**, **20**, or the demo.

---

## Global instructions for the whole deck

### Related work / papers (important)

You studied about **9–10 research / review / survey papers**. Slides must show a literature base, not only one baseline paper.

1. On **Slide 5**, show a compact literature map (clusters), not 10 full APA citations.
2. Say in speech: *“We reviewed roughly nine to ten research, review, and survey papers on SATD and related detection methods.”*
3. Keep **Sutoyo, Avgeriou & Capiluppi (2024)** (arXiv:2410.15804) visible as primary baseline.
4. Theme clusters: SATD taxonomies/datasets · classical/DL detection · augmentation · embeddings/transfer learning.
5. Optional backup Q/A slide: full short bibliography once titles are finalized.

### Multi-angle narrative rule (critical)

On Slide 8 and in speech:

> We did **not** run one single experiment. We attacked the same SATD problem from **several research angles**—a from-scratch rebuild, a paper-protocol study, an encoder extension, an applied large-scale classification track, and a live demo.

Folder cues:
- **Pipeline A** → `notebooks/`
- **Pipeline B + BGE** → `navid-experiment/`
- **Applied 458k issues** → `musaddiq_rafi/` (+ optimized V2)
- **Live demo** → `Demo/`
- **Next features** → `data/keywords/` + Phases 7–9

### Demo policy (for slides)

- High level only on Slide 17: issue text → Identification → Category.
- No UI screenshots, model filenames, latency tables, or setup steps.
- Run the **3-minute demo after Slide 17**, then finish Challenges → Conclusion → GitHub (18–20) if not already shown, or show 18–20 briefly before Q/A.

**Recommended flow:** Slides 1–17 → **Demo (3 min)** → Slides 18–20 → Q/A.

### Narrative spine

> Frozen modern embeddings plus a cheap classifier can **type** SATD better than fine-tuned BERT on most artifacts, while **detecting** SATD depends more on the data protocol than on which embedding brand we pick — validated across **multiple research angles**, then pushed toward **real issue text** and a **working demo**.

### Numbers rules

- Always say **macro F1**.
- Never average Pipeline A and Pipeline B into one “our score.”
- Prefer **XGBoost** numbers on slides.
- Commit baselines marked with `*` = supervisor replication where relevant.

### Suggested visual style

- Put the **mandatory section name** as a small header/eyebrow on each slide (e.g. `Results`, `Methodology`).
- Tables over paragraphs; portfolio boxes on Slide 8.
- One takeaway box on result slides.

### Team table template (Name + ID required)

Fill real student IDs before presenting. Roles are suggested; adjust to match actual work.

| Name | Student ID | Primary contribution (short) |
|---|---|---|
| Shahriar Pias | `[FILL ID]` | Pipeline A (`notebooks/`), docs, integration / demo glue |
| Navid Ibrahim | `[FILL ID]` | Pipeline B + BGE (`navid-experiment/`), comparison reports |
| Musaddiq Rafi | `[FILL ID]` | Applied large-scale issue classification (`musaddiq_rafi/`) |

Supervisor: `[FILL NAME]`

---

## Slide-by-slide blueprint

Format: **Section · Title → On-slide content → Speech → Design**

---

### Slide 1 — Title & Team (Name, ID)
**Section:** Project title · Team members’ info

**On the slide**
- Project title: **SATD Embedding Comparison**
- Subtitle: Identifying and Categorizing Self-Admitted Technical Debt with Frozen Embeddings
- One line: *Multi-angle design project — rebuild · protocol study · encoder ablation · applied inference · live demo*
- Course: SWE 4606 / CSE 4610 Design Project
- Supervisor: `[Name]`
- Team table:

| Name | ID |
|---|---|
| Shahriar Pias | `[ID]` |
| Navid Ibrahim | `[ID]` |
| Musaddiq Rafi | `[ID]` |

- Optional tiny footer: GitHub link (full contribution detail is Slide 20)

**Speech (~15s)**  
“Good [morning/afternoon]. We are [names]. Our project compares frozen embeddings for Self-Admitted Technical Debt detection and typing, studied from multiple research angles. We will cover introduction through results, then a live demo, limitations, conclusions, and contributions.”

**Design**  
Title + Name/ID must be readable on the projector. No results yet.

---

### Slide 2 — Introduction: What is SATD?
**Section:** Introduction

**On the slide**
- Definition: developers **explicitly admit** temporary compromises in text
- Short examples: “TODO: fix later”, “missing tests”, “docs outdated”
- Four artifacts: Code comments · Issues · Commits · Pull requests
- Why it matters: direct signal of unfinished work; unmanaged debt raises defect/maintenance risk

**Speech (~20s)**  
“Self-Admitted Technical Debt is when developers write that something is incomplete, temporary, or owed. It appears across comments, issues, commits, and pull requests—but large projects produce too much text to inspect by hand.”

---

### Slide 3 — Introduction: Problem
**Section:** Introduction

**On the slide**
- Manual review does not scale
- Two jobs:
  1. **Identification** — SATD or not?
  2. **Categorization** — C/D · REQ · TES · DOC
- Different debt types need different responses
- Artifact language differs (long issues vs short commits)

**Speech (~20s)**  
“The problem is scale and routing: first detect SATD, then categorize it so teams can respond correctly. One universal model is risky because artifact text styles differ.”

---

### Slide 4 — Introduction: Goals
**Section:** Introduction

**On the slide**
- Goal: research-backed, **lower-training-cost** alternative to heavy task-specific DL models
- Approach: **frozen embeddings + XGBoost / logistic regression**
- Baseline: Sutoyo et al. 2024 (BiLSTM / BERT + AugGPT)
- Responsible engineering (short):
  - Transparent macro-F1 evaluation, per-artifact models
  - Multi-angle design to avoid overclaiming
  - Support maintainability triage (human-in-the-loop)

**Speech (~20s)**  
“Our goal is a system and a research claim: can frozen embeddings plus cheap heads replace expensive fine-tuned models? We test that from several angles so one lucky setup cannot overstate the result.”

---

### Slide 5 — Related Works
**Section:** Related Works

**On the slide**
- Header: *Literature base: ~9–10 research / review / survey papers*
- Cluster map:
  1. SATD definitions, taxonomies, multi-artifact datasets
  2. Classical and deep-learning SATD detection
  3. Data augmentation for software NLP
  4. Embedding / transfer-learning classifiers
- Primary baseline:
  - **Sutoyo, Avgeriou & Capiluppi (2024)** — Deep Learning and Data Augmentation for Detecting SATD (arXiv:2410.15804)
- Dataset lineage cues: Maldonado-style comments, multi-artifact SATD corpora, AugGPT resources

**Speech (~25s)**  
“We reviewed about nine to ten research, review, and survey papers on SATD and related methods. Our main comparison target is Sutoyo et al., 2024—BiLSTM for identification and BERT for categorization with AugGPT-style augmentation.”

---

### Slide 6 — Related Works: Gap
**Section:** Related Works

**On the slide**
- Baseline is strong but **training-heavy**
- Unclear whether success is mainly **architecture** or **augmented data distribution**
- Opportunity: frozen embeddings + cheap heads
- Need: multi-artifact comparison · protocol contrast · path from lab scores to real issue text

**Speech (~15s)**  
“The gap is practical and scientific: fine-tuning is costly, and published scores may be tied to one AugGPT distribution. So we need a lighter recipe, a protocol contrast, and an applied path to real issue triage.”

---

### Slide 7 — Methodology: Research questions
**Section:** Methodology

**On the slide**
- **RQ1** Identification: frozen embeddings + XGB/LR vs BiLSTM+AugGPT?
- **RQ2** Categorization: same stack vs BERT+AugGPT for C/D, REQ, TES, DOC?
- **RQ3** Protocol: how much do conclusions change between Pipeline A and Pipeline B?
- **RQ4** Sensitivity: embedding brand vs classifier head?
- Footer preview: *Typing tends to be robust; detection is protocol-sensitive; classifier > brand.*

**Speech (~25s)**  
“Four research questions structure the methodology. RQ1–RQ2 compare to the baseline. RQ3 forces dual-protocol design. RQ4 asks whether encoder brand or classifier head matters more.”

---

### Slide 8 — Methodology: Multi-angle research portfolio
**Section:** Methodology

**On the slide** (portfolio boxes)
1. **Angle A — From-scratch rebuild** — `notebooks/full_pipeline.ipynb` · 31 sources · T5 balance · 32 models · `results/`
2. **Angle B — Paper-protocol study** — `navid-experiment/` · 4 AugGPT files · Qwen3 + E5 · reports
3. **Angle B+ — Encoder extension** — **BGE-M3** under same B protocol
4. **Angle C — Applied inference** — `musaddiq_rafi/` · ~458k issues · Title/Desc/Title+Desc · optimized batch notebooks
5. **Angle D — Working system** — `Demo/` · live identify → categorize
6. **Prepared next** — `data/keywords/` · Phases 7–9

Footer: *Same topic · different scientific / engineering questions · shared frozen-embedding recipe*

**Speech (~30s)**  
“Methodologically, this is a multi-angle design—not one notebook. Angle A rebuilds from scratch. Angle B, in the Navid experiment folder, holds the paper protocol fixed. We extend B with BGE-M3. Musaddiq’s applied track runs models on about four hundred fifty-eight thousand new issues. Then a live demo. Keyword phases are prepared for future work.”

---

### Slide 9 — Methodology: Shared method recipe
**Section:** Methodology

**On the slide**
- Shared recipe across angles:
  1. Artifact text
  2. Frozen encoder → 1024-d (mean-pool + L2-norm)
  3. Classifier head (XGBoost or LogReg)
  4. Two-step: Identify → (if SATD) Categorize
- Separate models **per artifact**
- Metric: **macro F1**
- No BERT/BiLSTM fine-tuning in our stack

**Speech (~20s)**  
“The shared methodological core is freeze–encode–classify in two steps per artifact. That shared recipe makes Angle A versus Angle B scientifically comparable.”

---

### Slide 10 — Implementation: Angle A (`notebooks/`)
**Section:** Implementation

**On the slide**
- Question this angle implements: *Does the frozen stack work after a broad multi-source rebuild and class equalization?*
- Data: 31 sources → clean ~386k → T5 balance all 5 labels → split after aug ≈ 59k / 7.4k / 7.4k
- Models: Qwen3 + E5 × XGB/LogReg × ID/Cat × 4 artifacts = **32 models**
- Preview: strong categorization; weak identification vs BiLSTM
- Artifacts: `notebooks/full_pipeline.ipynb` + `results/phase5–6`

**Speech (~20s)**  
“Implementation Angle A is the from-scratch pipeline in our notebooks folder: thirty-one sources, T5 balancing, thirty-two trained models.”

---

### Slide 11 — Implementation: Angle B + BGE (`navid-experiment/`)
**Section:** Implementation

**On the slide**
- Question this angle implements: *What happens if we keep the paper’s AugGPT distribution?*
- Data: 4 AugGPT CSVs · ~95.7k clean · Not-SATD majority · split before extra aug
- Models: Qwen3 + E5 (**32**), then **BGE-M3 (+16)**
- Different research control from A — not a duplicate experiment
- Reports: `satd_report` + `satd_report_bge`

**Speech (~20s)**  
“Implementation Angle B, in the Navid experiment folder, uses the paper AugGPT protocol so we can isolate architecture and classifier effects. We then implemented a BGE-M3 extension under the same protocol.”

---

### Slide 12 — Implementation: Models & stack
**Section:** Implementation

**On the slide**
- Embeddings: Qwen3-Embedding-0.6B · E5-Large-v2 · BGE-M3 (B only)
- Heads: XGBoost (primary) · Logistic Regression (ablation)
- Seed finding: XGB beat LogReg in every recorded pair
- Stack footer: Python, Transformers, XGBoost, scikit-learn, PyTorch, joblib · GPU for embeddings

**Speech (~15s)**  
“Across implementations we compared three embedding families and two heads. Headline results use XGBoost because it consistently won.”

---

### Slide 13 — Results: Identification (RQ1)
**Section:** Results

**On the slide**
Title: Identification macro F1 (SATD vs Not-SATD)

| Artifact | BiLSTM | A best | B Qwen3 | B E5 | B BGE |
|---|---:|---:|---:|---:|---:|
| Comments | 0.939 | 0.915 | **0.966** | 0.959 | 0.959 |
| Issues | 0.878 | 0.798 | 0.866 | 0.863 | 0.853 |
| PRs | 0.862 | 0.792 | 0.864 | **0.870** | 0.858 |
| Commits | 0.910* | 0.751 | 0.892 | 0.882 | 0.901 |

Takeaway: Angle A never beats BiLSTM; Angle B wins/close on comments & PRs → detection is **protocol-sensitive**

**Speech (~30s)**  
“Identification results show why multi-angle design matters. Angle A loses every artifact; Angle B largely recovers. RQ1 depends on the data protocol, not only the encoder name.”

---

### Slide 14 — Results: Categorization (RQ2)
**Section:** Results

**On the slide**
Title: Categorization macro F1 (C/D, REQ, TES, DOC)

| Artifact | BERT | A best | B Qwen3 | B E5 | B BGE |
|---|---:|---:|---:|---:|---:|
| Comments | 0.882 | **0.904** | 0.947 | 0.958 | 0.956 |
| Issues | 0.899 | **0.941** | 0.944 | 0.950 | **0.956** |
| PRs | 0.876 | **0.936** | 0.955 | 0.934 | **0.964** |
| Commits | 0.980* | 0.959 | 0.964 | 0.949 | 0.960 |

Takeaway: Beat BERT on comments/issues/PRs in **both** angles; commits hardest; e.g. BGE PRs **0.964** vs **0.876**

**Speech (~30s)**  
“Categorization is our strongest result: in both angles, frozen embeddings with XGBoost beat BERT on three of four artifacts. Commits remain the hard case.”

---

### Slide 15 — Results: Robust vs protocol-dependent (RQ3)
**Section:** Results

**On the slide**
**Robust across A & B**
- Categorization wins on comments / issues / PRs
- XGBoost > LogReg
- Commits hardest vs BERT typing baseline

**Changes with protocol**
- Identification collapses under Angle A equalization
- Identification recovers under Angle B Not-SATD majority
- Never merge A+B into one score

**Speech (~20s)**  
“RQ3: typing gains are robust; detection conclusions are protocol-sensitive. That is a core scientific finding of the project.”

**Design**  
Keep detailed limitation bullets for Slide 18 (do not dump the full challenges list here).

---

### Slide 16 — Results: Embedding vs classifier (RQ4)
**Section:** Results

**On the slide**
- Angle B identification averages nearly tied (~0.89x across Qwen/E5/BGE)
- Angle B categorization (XGB): BGE **0.959** > Qwen3 **0.953** > E5 **0.948**
- Combined Angle B XGB: BGE **0.926** ≈ Qwen3 **0.925** > E5 **0.921**
- BGE + LogReg much weaker (~0.85) vs BGE + XGB
- Conclusion: **classifier head > embedding brand**

**Speech (~20s)**  
“RQ4: swapping classifier heads moved scores more than swapping embedding brands. Practical recipe: frozen embedding plus a strong tree head.”

---

### Slide 17 — Implementation: Applied track + demo bridge
**Section:** Implementation

**On the slide** (three bands)

**Band 1 — Applied (`musaddiq_rafi/`)**
- Pipeline A issue models (Qwen3 + XGBoost)
- ~**458,232** new unlabeled issues
- Title · Description · Title+Description
- Optimized batch notebooks (Musaddiq optimized + notebooks V2)
- Status: *notebooks ready; state batch-output status honestly*

**Band 2 — Final reusable solution**
- Frozen embedding + XGBoost · Identify → Categorize · per-artifact models

**Band 3 — Live demo next**
- Watch: paste issue text → Identification → Category  
- (No UI/setup details)

**Speech (~25s, then demo)**  
“We also implemented an applied track on roughly four hundred fifty-eight thousand new issues, with optimized batch notebooks for real compute limits. Next is the live demo—watch identify-then-categorize.”

---

### Slide 18 — Challenges and Limitations
**Section:** Challenges and Limitations *(mandatory dedicated slide)*

**On the slide**
**Challenges faced**
- Multi-source cleaning / label harmonization across 31 files (Angle A)
- Compute limits for embedding generation & paraphrasing (GPU / Kaggle constraints)
- Balancing scientific controls: rebuild vs paper protocol without conflating scores
- Scaling applied inference to ~458k issues (memory, sharding, runtime)
- Short, noisy artifacts (especially commits) remain hard

**Limitations (honesty)**
- Angle A: split-after-paraphrase **leakage risk**
- Angle A includes paper AugGPT files among the 31-source merge
- Commit baselines mix paper vs supervisor replication (`*`)
- BGE-M3 training notebook not in the repository
- Reported metric focus is macro F1; limited confusion-matrix / error analysis in slides
- Applied 458k result distribution may still be finalizing (state current status)

**Speech (~25s)**  
“We want to be explicit about challenges and limits. Pipeline A has leakage risk from splitting after paraphrasing. Detection results are protocol-sensitive, so we refuse one blended score. BGE training code is missing from the repo, commits remain difficult, and large-scale applied inference stressed our compute pipeline.”

**Design**  
This slide must be clearly titled **Challenges and Limitations**. Do not hide it inside Results.

---

### Slide 19 — Conclusion and Future Direction
**Section:** Conclusion and Future Direction *(mandatory)*

**On the slide**
**Conclusions**
1. One SATD problem studied through **multiple research angles**
2. Frozen embeddings + XGBoost are strong for **categorization** across angles
3. **Identification** depends on data protocol / research angle
4. Classifier choice mattered more than embedding brand
5. Applied inference + live demo connect research to practice
6. Field value: cheaper typing recipe + protocol-aware evaluation

**Future direction**
- Near-term: Phase 7 code-aware encoders; Phases 8–9 KeyBERT keywords (`data/keywords/` ready) → 1033-d concat retrain
- Stronger error analysis / confusion matrices
- Finalize/publish 458k applied result distributions if needed
- Longer-term: SATD in Agile; SATD ↔ emotion ↔ issue resolution time; larger/adapted encoders under controlled protocols

**Speech (~25s)**  
“To conclude: multi-angle evidence supports a cheaper strong typing recipe, while detection is protocol-sensitive. Future work is already scaffolded with keyword lists and code-encoder phases, plus broader Agile and process-analytics directions.”

---

### Slide 20 — GitHub Link and Individual Contribution
**Section:** Github Link and Individual Contribution *(mandatory)*

**On the slide**
- **GitHub repository:**  
  https://github.com/A-Piyas-04/SATD-embedding-DP2
- Optional QR code to the repo
- Individual contribution table (**Name · ID · Contribution**):

| Name | ID | Individual contribution |
|---|---|---|
| Shahriar Pias | `[FILL ID]` | Pipeline A from-scratch experiments (`notebooks/`, `results/`), project documentation, integration / demo support |
| Navid Ibrahim | `[FILL ID]` | Pipeline B paper-protocol experiments (`navid-experiment/`), Qwen3/E5 comparison report, BGE-M3 extension report |
| Musaddiq Rafi | `[FILL ID]` | Applied inference on ~458k issues (`musaddiq_rafi/`), optimized batch classification notebooks |
| Shared / team | — | Related-work study (~9–10 papers), dual-protocol interpretation, final report & presentation |

- Thank you / Questions

**Speech (~15s)**  
“Our code and artifacts are on GitHub at this link. Individual contributions map to the multi-angle tracks: from-scratch pipeline, paper-protocol and BGE study, and large-scale applied inference. Thank you—we welcome questions.”

**Design**  
Make the GitHub URL large and clickable/QR-friendly. Name + ID must appear again here (required section).

---

## Demo segment script (~3 minutes)

Use **after Slide 17**, before Slides 18–20 (recommended).

### Pre-demo checklist
1. `Demo` app running; status **Ready**
2. GPU preferred
3. 2–3 prepared examples
4. Large browser zoom
5. One drives, one narrates

### Spoken demo outline

| Time | Action | Say |
|---|---|---|
| 0:00–0:20 | Show UI | “Live triage demo—the systems angle of the same research.” |
| 0:20–1:10 | Example A (SATD) | “Step 1: SATD vs not. Step 2: debt type.” |
| 1:10–2:00 | Example B | “Contrast so the two-step flow is clear.” |
| 2:00–2:40 | Optional third / honesty | “Short noisy text can still fail—commits were hard in experiments.” |
| 2:40–3:00 | Close | “Next: challenges, conclusion, and contributions.” |

### What not to explain in demo
- Joblib filenames / Drive setup / hyperparameters
- Full 458k logistics (if asked: point to `musaddiq_rafi` + notebooks V2)

---

## Suggested speech cut list (if over time)

Cut first:
1. Slide 12 tech-stack footer
2. Slide 4 ethics → one line
3. Slide 6 to ~10 seconds
4. Slide 19 longer-term bullets (keep near-term)

Never cut:
- Slide 1 (title + Name/ID)
- Slide 5 (related works / 9–10 papers)
- Slide 8 (portfolio map)
- Slides 13–14 (result tables)
- Slide 17 (applied + demo bridge)
- Slide **18** (Challenges and Limitations)
- Slide **19** (Conclusion and Future Direction)
- Slide **20** (GitHub + Individual Contribution)
- Demo narration

---

## Q/A preparation (max 5 minutes)

1. **Why multiple angles / two pipelines?** Different scientific controls on the same topic.  
2. **Is Navid’s work a duplicate of A?** No—paper-protocol control.  
3. **What did Musaddiq add?** Applied ~458k issue inference + optimized batching.  
4. **Why not fine-tune BERT?** Test a cheaper frozen-embedding alternative.  
5. **Why macro F1?** Fairer under imbalance / multi-class typing.  
6. **Why commits are hard?** Short context; BERT typing ≈ 0.98.  
7. **Challenges/limitations?** Point to Slide 18.  
8. **Keywords?** Prepared, not in reported models yet (Phases 8–9).  
9. **Papers?** ~9–10; Sutoyo 2024 primary; list on backup slide.  
10. **Who did what / GitHub?** Point to Slide 20.

---

## Build checklist before presentation day

- [ ] Slide 1 has **Name + Student ID** for every member
- [ ] Every mandatory section header appears on the matching slides
- [ ] Slide 8 portfolio includes A, B, BGE, Musaddiq, Demo, keywords/future
- [ ] Slide 17 covers applied + optimized notebooks + demo bridge
- [ ] Slide **18** is a dedicated Challenges and Limitations slide
- [ ] Slide **19** combines Conclusion + Future Direction
- [ ] Slide **20** has **GitHub URL** + **Individual Contribution** table with Name/ID
- [ ] Insert real ~9–10 paper titles into Slide 5 / backup bibliography
- [ ] Verify F1 numbers against root `README.md`
- [ ] Honest one-line status for 458k batch outputs
- [ ] Rehearse: 7 min slides + 3 min demo + short 18–20 close
- [ ] Preload Gradio until Ready

---

## Optional compression to ~16 slides (only if forced)

Merge only:
- 2+3 → Introduction overview
- 5+6 → Related works + gap
- 15+16 → Results synthesis

**Never delete Slides 1, 8, 13–14, 17, 18, 19, or 20.**

---

## Source alignment

- `README.md` (numbers + GitHub)
- `docs/SATD_Research_Progress.md`
- `docs/pipeline_a_from_scratch.md` / `docs/pipeline_b_paper_data.md`
- `docs/thesis_research_narrative.md`
- `notebooks/full_pipeline.ipynb` + optimized V2
- `navid-experiment/` notebook + reports
- `musaddiq_rafi/` README + notebooks
- `Demo/README.md` (high level)
- `data/keywords/` + phases 7–9
- `results/` Pipeline A CSVs / summary PDF
- Course guidelines + **new mandatory slide section list**

When numbers conflict, prefer root `README.md` tables for slides.
