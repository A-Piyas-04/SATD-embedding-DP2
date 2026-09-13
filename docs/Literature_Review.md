# Literature Review

**SATD Embedding Comparison Project**
**Team: Shahriar Pias, Navid Ibrahim, Musaddiq Rafi**
**Supervisor: [Supervisor Name]**
**Date: [Date]**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Review Methodology](#2-review-methodology)
3. [Paper 1: A Survey of Vibe Coding with Large Language Models](#3-paper-1)
4. [Paper 2: From Code Generation to Code Verification](#4-paper-2)
5. [Paper 3: GenAI Code Security Report](#5-paper-3)
6. [Paper 4: HoarePrompt](#6-paper-4)
7. [Paper 5: GenAI-Induced Self-Admitted Technical Debt](#7-paper-5)
8. [Paper 6: Speed at the Cost of Quality (Cursor)](#8-paper-6)
9. [Paper 7: From Technical Debt to Cognitive and Intent Debt](#9-paper-7)
10. [Paper 8: Slopsquatting](#10-paper-8)
11. [Paper 9: Deep Learning and Data Augmentation for Detecting SATD (Baseline)](#11-paper-9)
12. [Summary Comparison Table](#12-summary-comparison-table)
13. [Thematic Analysis](#13-thematic-analysis)
14. [Research Gaps Identified](#14-research-gaps)
15. [How These Papers Relate to Our Project](#15-relation-to-our-project)
16. [Conclusion](#16-conclusion)
17. [References](#17-references)

---

## 1. Introduction

### 1.1 Background

The rapid adoption of artificial intelligence in software development has changed how code is written, reviewed, and maintained. Tools such as GitHub Copilot, Cursor, and other large language model (LLM) based assistants now generate significant portions of production code. While these tools promise faster development cycles, they also introduce new challenges for software quality, security, and maintainability.

One critical challenge is Self-Admitted Technical Debt (SATD). SATD occurs when developers explicitly acknowledge in their text that some part of the code, documentation, or process is incomplete, temporary, or compromised. Common examples include comments like "TODO: fix later", "this is a hack", "tests are missing for this module", or "docs are outdated". The developer has already supplied a direct signal that work remains, but in large projects, there is too much text for any team to inspect manually.

The rise of AI coding assistants has made SATD detection even more important. Developers using AI tools may accept generated code without fully understanding it, leave notes about uncertain behavior, or introduce technical debt at a faster rate than before. At the same time, the volume of text artifacts (code comments, issues, commit messages, pull requests) continues to grow, making manual SATD review increasingly impractical.

### 1.2 Purpose of This Review

This literature review examines nine papers that collectively form the foundation for our research project. The papers span several themes:

- The new paradigm of AI-assisted coding and its infrastructure
- The paradox between AI-generated speed and verification burden
- Security vulnerabilities in AI-generated code
- Program correctness and verification approaches
- Self-Admitted Technical Debt in the context of AI-generated code
- Causal evidence of AI assistants increasing code complexity
- Theoretical frameworks for understanding software health beyond code debt
- Supply chain risks from AI hallucinations in dependencies

Together, these papers establish that the field is shifting from "how do we generate code faster" to "how do we verify and manage what AI generates". Our project sits at the intersection of this shift: we test whether cheap, frozen text embeddings can automatically find and type SATD in developer text, providing a lightweight verification and triage tool.

### 1.3 Scope

This review covers papers published between 2024 and 2026. The primary baseline for our experimental comparison is Sutoyo et al. (2024), which provides the reference scores for BiLSTM-based SATD identification and BERT-based SATD categorization. The other eight papers provide the intellectual context: they explain why AI-era SATD matters, how AI assistants change code quality, and what verification approaches exist.

Each paper is reviewed individually with a consistent structure: full citation, problem statement, methodology, dataset, key results, limitations, and relevance to our project. The review concludes with a summary comparison table, thematic analysis, identification of research gaps, and a discussion of how the literature informs our work.

---

## 2. Review Methodology

### 2.1 Search and Selection

Papers were selected based on their relevance to the following topics:

- Self-Admitted Technical Debt detection and categorization
- AI-assisted code generation and its impact on code quality
- Data augmentation for software engineering natural language tasks
- Embedding-based approaches for software text classification
- Program verification and correctness checking
- Supply chain security in AI-generated code

Papers were identified through a combination of instructor recommendations, reference chain following (checking citations of key papers), and targeted searches on arXiv, ACM Digital Library, and Google Scholar. The selection was not exhaustive but aimed to cover the breadth of issues surrounding AI-generated code and technical debt management.

### 2.2 Evaluation Criteria

Each paper was evaluated on the following dimensions:

1. **Problem clarity**: How well does the paper define what it is trying to solve?
2. **Methodological rigor**: Is the approach well-designed and the evaluation fair?
3. **Dataset and reproducibility**: Are the data and code available? Could the results be reproduced?
4. **Strength of evidence**: How convincing are the findings?
5. **Relevance to our project**: Does the paper inform our research questions, experimental design, or interpretation of results?

### 2.3 Organization

Papers are organized thematically rather than strictly chronologically. The review begins with broad surveys and frameworks, moves to specific empirical studies, and concludes with the primary baseline paper for our experimental comparison.

---

## 3. Paper 1: A Survey of Vibe Coding with Large Language Models

### 3.1 Full Citation

Ge, Mei, Duan, Li, Zheng, Wang, Wang, Yao, Liu, Cai, Bi, Guo, Guo, Liu, Cheng (2025). *A Survey of Vibe Coding with Large Language Models.* arXiv preprint (cs.AI). Available at: https://arxiv.org/abs/2510.12399 and https://github.com/A-Piyas-04/SATD-embedding-DP2

### 3.2 Paper Type

Survey paper. This is a systematic review of over 1,000 papers related to "vibe coding" — the practice of using AI agents to generate code based on high-level descriptions, where developers validate the output through outcomes rather than line-by-line inspection.

### 3.3 Problem Statement

The practice of vibe coding has spread rapidly since the release of powerful code LLMs, but the paradigm lacked a systematic definition, taxonomy, and review of its infrastructure, agents, and collaboration models. Developers and researchers needed a structured understanding of what vibe coding actually involves, what tools exist, and what the open challenges are.

### 3.4 Methodology

The authors conducted a systematic survey of over 1,000 papers. They formalized vibe coding as a Constrained Markov Decision Process (CMDP) over the triad of developer, project, and agent. This formalization captures the idea that the developer provides goals and constraints, the agent generates actions (code changes), and the project state evolves through these actions — all under constraints imposed by the development environment, testing infrastructure, and human review.

The survey synthesizes five development models:

1. **Unconstrained Automation**: the agent generates code with minimal human guidance. Fast but risky for complex projects.
2. **Iterative Conversational Collaboration**: the developer and agent engage in back-and-forth dialogue, refining the code through conversation.
3. **Planning-Driven**: the developer provides a detailed plan or specification, and the agent fills in implementation details.
4. **Test-Driven**: tests are written first, and the agent generates code to pass them. Provides built-in verification.
5. **Context-Enhanced**: additional context (codebase structure, documentation, project history) is provided to the agent to improve output quality.

The survey also reviews the ecosystem of code LLMs, agents (with planning, memory, tools, and reflection capabilities), development environments, and feedback loops.

### 3.5 Key Results and Findings

The survey's most important finding is that success in vibe coding depends less on raw agent skill and more on context engineering, execution environments, and the chosen collaboration model. In other words, the quality of the output is determined more by how the human sets up the task and provides context than by which LLM is used.

The survey also cites evidence from METR (Model Evaluation & Threat Research) that experienced developers can be approximately 19% slower when using Cursor with Claude compared to their normal workflow. This finding is significant because it challenges the popular assumption that AI tools always speed up development.

Open challenges identified include infrastructure (how to set up environments for reliable agent operation), security (how to prevent agents from introducing vulnerabilities), and human-centered design (how to make AI tools fit naturally into developer workflows rather than forcing developers to adapt to the tools).

### 3.6 Limitations

- This is a survey and synthesis, not a new causal experiment. The CMDP model is conceptual and has not been empirically fitted to real data.
- The field is moving extremely fast. Agent capabilities and security findings from 2025 may already be outdated by the time of publication.
- The survey does not provide primary evidence on long-term maintainability and team cognition under vibe coding workflows.
- The scope is very broad, which means individual topics receive relatively shallow coverage.

### 3.7 Relevance to Our Project

This paper is foundational for understanding the landscape in which our project operates. The five development models provide a taxonomy for categorizing different AI-assisted workflows. Our project addresses one specific aspect of vibe coding: when a developer writes a comment like "TODO: fix later" or "this is a hack" in AI-generated code, that is SATD, and our system can automatically detect and type it.

The finding that context engineering matters more than agent skill aligns with our own RQ4 result: the classifier head (XGBoost vs logistic regression) moved scores more than the embedding brand (Qwen3 vs E5 vs BGE-M3). Just as the development environment matters more than the LLM in vibe coding, the classifier choice matters more than the embedding model in SATD detection.

The METR finding about developers being 19% slower with AI tools connects directly to the productivity paradox described in Paper 2 (Singh, 2025) and provides motivation for why automated SATD detection is valuable: if AI tools slow developers down through increased review burden, automated debt triage can reduce that burden.

---

## 4. Paper 2: From Code Generation to Code Verification

### 4.1 Full Citation

Singh, J. (2025). *From Code Generation to Code Verification: Explaining and Mitigating the AI Productivity Paradox in Software Engineering.* Course/academic paper (Chitkara University, B.E. CSE). Local PDF: 2210990434_JapneetSingh_AIProductivityParadox.pdf

### 4.2 Paper Type

Review paper. This is a structured evidence synthesis that codes each source on task type, developer context, measurement level, and downstream validation impact. It introduces the Verification Budget model to explain why AI coding assistants can speed up drafting while slowing down or not improving overall delivery.

### 4.3 Problem Statement

Why do AI coding assistants sometimes speed up development (on narrow tasks) and sometimes slow it down (on realistic tasks)? The paper proposes that the answer lies in the total delivery time equation:

T_delivery = T_write + T_verify + T_rework

When AI reduces T_write but increases T_verify (more code to review, more complexity, more potential bugs) and T_rework (fixing issues found during verification), the net effect on T_delivery can be zero or negative. This is the AI Productivity Paradox.

### 4.4 Methodology

The author synthesized evidence from multiple sources, coding each on task type, developer context, and measurement level. Key sources include:

- **Peng et al. (2023)**: Narrow coding tasks with AI assistants showed +55.8% speed improvement.
- **Cui et al. (2025)**: Field experiment with 4,867 developers showed +26% tasks completed with AI.
- **METR (2025)**: Realistic open-source experiment with 16 developers on 246 issues showed -19% speed with Cursor+Claude.
- **Faros AI (2025)**: Data from 10,000+ developers across 1,255 teams showed +21% tasks completed but +91% review time and +154% PR size.
- **Apiiro (2025)**: Security analysis showed -76% syntax issues but +322% privilege escalation paths and +153% design flaws in AI-generated code.
- **Stack Overflow (2025)**: Survey of 49,009 responses showed 66% of AI code is "almost right", 45.2% found debugging harder, and distrust outweighed trust.
- **GitGuardian**: Repositories using Copilot had 40% more secret leaks (4.6% to 6.4%).

### 4.5 Key Results and Findings

The paper presents a clear picture of the productivity paradox:

**Narrow tasks benefit from AI**: When the task is small, self-contained, and does not require deep understanding of the surrounding codebase, AI assistants provide significant speed gains. The Peng et al. finding of +55.8% speed is the upper bound.

**Realistic tasks show mixed results**: When tasks require understanding context, navigating a codebase, and integrating changes, the benefits shrink or disappear. The METR finding of -19% is the lower bound.

**High AI adoption creates review burden**: The Faros data shows that while more tasks are completed, each task generates more code (PR size +154%) that requires more review time (+91%). The net effect on delivery speed is uncertain.

**Security gets worse**: The Apiiro and GitGuardian data show that AI-generated code has fewer syntax errors but more serious design flaws and security vulnerabilities. Developers may not be equipped to catch these issues during review.

**The Verification Budget model**: The paper proposes that teams have a fixed "verification budget" — the total amount of code review, testing, and quality assurance they can perform per unit time. AI generates code faster than humans can verify it, creating a verification deficit. Over time, this deficit accumulates as technical debt.

### 4.6 Limitations

- Many sources are industry reports, not peer-reviewed studies. Outcomes are measured differently (time vs throughput vs security), making direct comparison difficult.
- The paper is a student work, not peer-reviewed.
- No original primary experiment is conducted; the contribution is synthesis and framing.
- The Verification Budget model is conceptual, not quantified with specific numbers.

### 4.7 Relevance to Our Project

This paper provides the strongest framing for why our project matters. The Verification Budget model explains that the bottleneck in AI-assisted development has shifted from code generation to code verification. Our project addresses this bottleneck by providing an automated tool that can identify and categorize SATD in developer text, effectively reducing the verification burden.

The specific numbers from this paper (review time +91%, PR size +154%, debugging harder for 45.2%) provide context for why manual SATD review does not scale. When developers are already overwhelmed with reviewing AI-generated code, manually scanning for SATD in comments, issues, commits, and pull requests is simply not feasible.

The finding that AI-generated code has +322% privilege escalation paths and +153% design flaws connects directly to SATD: these are exactly the kinds of shortcuts and compromises that developers acknowledge in their text. Our system can detect these acknowledgments automatically.

The security data from Veracode (Paper 3) and the supply chain risks from slopsquatting (Paper 8) further reinforce the argument: AI generates code faster than humans can verify it, and the verification gap creates debt, security vulnerabilities, and supply chain risks. Our project contributes to closing this gap for one specific type of debt.

---

## 5. Paper 3: GenAI Code Security Report

### 5.1 Full Citation

Veracode Research (Milzarek, Guyer, Tahir, Simpson, Brombacher, Puvvala, et al.) (2025). *GenAI Code Security Report: Assessing the Security Performance of Newer LLMs (October 2025 Update).* Industry research report (Veracode). Available at: https://www.veracode.com

### 5.2 Paper Type

Industry research report. This is a benchmark study that tests whether LLMs produce secure code when given prompts that specify functionality but not security requirements.

### 5.3 Problem Statement

When developers ask an LLM to write code for a specific function, without mentioning security, does the LLM produce code that is secure by default? Or does it introduce common vulnerabilities? The report tests this question across multiple LLMs and programming languages.

### 5.4 Methodology

The study uses an 80-task benchmark: 4 programming languages (Java, JavaScript, C#, Python) × 4 Common Weakness Enumerations (CWEs) × 5 tasks per combination. The CWEs tested are:

- **CWE-89**: SQL Injection (SQLi)
- **CWE-80**: Cross-Site Scripting (XSS)
- **CWE-117**: Log Injection
- **CWE-327**: Weak Cryptography

Each task can be implemented securely or insecurely. Veracode's Static Application Security Testing (SAST) tool scores each generation on security pass rate and syntax pass rate. The same protocol was used in the original July 2025 report and re-run on October 2025 models.

Over 100 LLMs have been tested historically. The October 2025 update includes GPT-5 family, Claude 4.x, Gemini 2.5, Qwen3 Coder, Grok, and gpt-oss models.

### 5.5 Key Results and Findings

**Baseline**: Historically, about 45% of LLM code generations introduce the target vulnerability (security pass rate approximately 55%).

**October 2025 results**: Most new models cluster between 50-59% security pass rate. This means that even with significant improvements in model scale and training, the majority of LLMs still introduce security flaws about half the time.

**Top performers**: GPT-5 Mini achieved 72% and GPT-5 achieved 70%, setting new records. However, GPT-5-chat (the non-reasoning variant) only achieved 52%, showing that reasoning capabilities specifically improve security performance.

**Claude models**: Claude Sonnet 4.5 achieved 50% and Opus 4.1 achieved 49%, showing a slight dip compared to some earlier versions. This suggests that model improvements do not always translate to security improvements.

**Language effects**: Java consistently shows worse security performance than other languages. The authors hypothesize this is due to training data composition — Java security patterns may be underrepresented in training corpora.

**CWE effects**: XSS and log injection are harder for LLMs because they require understanding taint flow and context that LLMs typically lack. These vulnerabilities depend on how data flows through the application, not just on the specific code snippet being generated.

**Key insight**: Gains are concentrated in OpenAI's reasoning-tuned models. Scale alone does not automatically improve security. The reasoning capability — the ability to think through security implications step by step — is what makes the difference.

### 5.6 Limitations

- Function-level tasks without full application context. In real applications, vulnerabilities depend on how data flows through the entire system, not just on individual functions.
- Only four CWEs and four languages. The results may not generalize to other vulnerability types or programming languages.
- Vendor-authored report. While the methodology is sound, the report is produced by a security company with a commercial interest in demonstrating that AI code needs security scanning.
- Pass rate is not exploitability. A "security fail" does not mean the code is exploitable in production; it means the SAST tool detected a potential issue.
- Does not evaluate agentic multi-file coding or developer-in-the-loop repair.

### 5.7 Relevance to Our Project

This report provides quantitative evidence that AI-generated code is not automatically secure, even with the latest models. The finding that reasoning-tuned models perform better aligns with the broader theme in Papers 1 and 2: the value of AI tools depends on how they are used (context engineering, verification), not just on the raw capability of the model.

For our project, the Veracode data reinforces the argument that automated verification tools are needed. If 45% of LLM code introduces vulnerabilities, and developers are already overwhelmed with review (Paper 2: +91% review time), then automated tools that can detect debt and security issues in developer text are essential.

The finding that Java performs worst is interesting for our project because our dataset includes code comments from Java projects (Apache Ant, JMeter, ArgoUML, JFreeChart, JRuby). If Java code is more likely to have security issues, Java code comments may also carry more SATD signals, which could affect the distribution of debt types in our data.

---

## 6. Paper 4: HoarePrompt

### 6.1 Full Citation

Bouras, Dai, Wang, Xiong, Mechtaev (2025). *HoarePrompt: Structural Reasoning About Program Correctness in Natural Language.* arXiv preprint (cs.SE). Available at: https://arxiv.org/abs/2503.19599 and artifacts at https://figshare.com/s/5d39b388bd9ff2c7ffab

### 6.2 Paper Type

Research paper. This paper introduces a novel approach to program verification that uses natural language as the semantic medium, rather than formal specifications.

### 6.3 Problem Statement

Requirements are often written in natural language, but LLMs fail to detect even simple bugs when asked to judge whether code meets those requirements. The question is whether verification ideas from formal methods can improve natural language-level correctness checking without requiring full formalization.

### 6.4 Methodology

HoarePrompt works in three steps:

1. **Natural Hoare triples**: The LLM infers natural strongest postconditions stepwise. Instead of using formal logic, it uses natural language to describe what each line of code does to the program state. This is inspired by Hoare logic but uses natural language instead of formal assertions.

2. **k-induction for loops**: For loops, HoarePrompt uses few-shot-driven k-induction. It unrolls the loop k iterations, observes the pattern, and then induces the loop postcondition. This handles the common failure mode where LLMs lose track of loop invariants.

3. **Classification**: The annotated program (with natural language state descriptions) is then classified against the natural language requirements to determine if the code is correct or incorrect.

The approach was compared against Zero-shot Chain-of-Thought (CoT) and LLM test generation. It was also used as feedback for code generation, providing a novel use case for verification-based prompting.

### 6.5 Key Results and Findings

- **Matthews Correlation Coefficient (MCC)**: HoarePrompt achieved +61% MCC improvement over Zero-shot-CoT and +106% improvement over LLM-generated tests.
- **k-induction contribution**: Adding k-induction for loops contributed +26% MCC improvement.
- **State annotations**: Annotated programs raised bug detection rates significantly. For example, Llama3.1-70B went from 50% to 100% detection on the motivating example.
- **Generation feedback**: When used as feedback for code generation, HoarePrompt improved generation quality by +19.4% compared to direct generation and +12.7% compared to test-based feedback.
- **Token cost**: The approach is more expensive in terms of tokens because it requires step-by-step state reasoning.

### 6.6 Limitations

- Expensive token usage due to step-by-step reasoning.
- Evaluated mainly on contest-style Python programs, not large industrial codebases.
- LLM postconditions are not sound or complete like formal strongest postcondition calculus.
- Binary classification only (correct/incorrect), not full proof.
- Data leakage risk reduced but not eliminated for all models.
- The CoCoClaNeL benchmark is Python-focused and competition-style, which may not represent real-world code.

### 6.7 Relevance to Our Project

HoarePrompt is relevant to our project in two ways. First, it demonstrates that LLMs can reason about program correctness when given the right structure. This supports the broader argument that the bottleneck is not model capability but how we frame the verification task — similar to how our project shows that classifier choice matters more than embedding brand.

Second, HoarePrompt's approach to natural language verification connects to SATD detection. When a developer writes "TODO: fix later" or "this is a hack", they are implicitly providing a natural language assertion about the code's correctness status. Our system detects these assertions; HoarePrompt uses similar assertions for verification. The two approaches are complementary: HoarePrompt checks whether code meets requirements; our system checks whether developers have admitted that code does not meet requirements.

The finding that verification is more expensive than generation (higher token cost) also aligns with the productivity paradox in Paper 2: verification always costs more than generation, whether the generation is done by humans or AI.

---

## 7. Paper 5: GenAI-Induced Self-Admitted Technical Debt

### 7.1 Full Citation

Al Mujahid, A. & Imran, M. M. (2026). *"TODO: Fix the Mess Gemini Created": Towards Understanding GenAI-Induced Self-Admitted Technical Debt.* ACM conference preprint (placeholder venue); arXiv cs.SE. Available at: https://arxiv.org/abs/2601.07786

### 7.2 Paper Type

Research paper. This is the first empirical study that specifically examines SATD comments that reference AI tool usage.

### 7.3 Problem Statement

When developers use AI coding assistants, they sometimes leave comments acknowledging that the AI-generated code has issues. How does this AI-linked SATD look? What debt types appear? How do developers attribute AI's role in creating the debt? No prior study had examined this specific intersection of AI usage and SATD.

### 7.4 Methodology

The authors used GitHub Code Search with 196 queries to find LLM-referencing comments in public Python and JavaScript repositories (November 2022 to July 2025). From 6,540 unique LLM-referencing comments, they applied a keyword SATD filter (TODO, FIXME, HACK, XXX) to identify 96 candidates. Two annotators independently validated these, achieving a Cohen's kappa of 0.896 (strong agreement), resulting in 81 valid SATD comments.

The taxonomy used was the Maldonado taxonomy for SATD types (Code/Design, Requirement, Test, Defect, Documentation) extended with an open coding of AI roles:

- **Source**: the AI directly created the debt (e.g., "Gemini wrote this wrong")
- **Catalyst**: the AI accelerated or enabled the debt (e.g., "I rushed because Copilot made it easy")
- **Mitigator**: the AI helped reduce existing debt (e.g., "Copilot helped me fix this TODO")
- **Neutral**: AI is mentioned but not clearly attributed as causing or fixing debt

### 7.5 Key Results and Findings

**Debt type distribution**:
- Code/Design: 33/81 (40.7%)
- Requirement: 17/81 (21.0%)
- Test: 17/81 (21.0%)
- Defect: 11/81 (13.6%)
- Documentation: 3/81 (3.7%)

**Comparison with Maldonado's traditional SATD**: The AI-era distribution shows less design debt (40.7% vs 71.8% in Maldonado) and significantly more requirement debt (21.0% vs 14.2%) and test debt (21.0% vs 2.1%). This suggests that AI shifts SATD toward later-stage concerns — requirements that were not fully understood and tests that were not written — rather than pure code design issues.

**AI role distribution**:
- Catalyst: 34/81 (42.0%) — AI enabled or accelerated the debt
- Source: 22/81 (27.2%) — AI directly created the debt
- Mitigator: 19/81 (23.5%) — AI helped reduce debt
- Neutral: 6/81 (7.4%) — AI mentioned but not clearly causal

**GIST concept**: The authors introduce the term "Generative AI-Induced Technical Debt" (GIST) to describe the specific pattern of adopting AI-generated code while acknowledging uncertainty about its behavior or correctness. GIST captures the tension between speed and understanding that characterizes AI-assisted development.

### 7.6 Limitations

- Small sample size (n=81), which limits statistical generalizability.
- Keyword-based detection may miss SATD that does not use standard markers (TODO, FIXME, HACK, XXX).
- Python and JavaScript open-source repositories only; results may not generalize to other languages or proprietary code.
- SATD keywords miss unspoken debt — developers may know the code is problematic but not write it down.
- Not longitudinal — the study does not track whether GIST gets paid down over time.
- Placeholder ACM metadata suggests the paper is still in pre-publication stage.

### 7.7 Relevance to Our Project

This is the most directly relevant paper to our work. It provides the first empirical evidence that AI-generated code creates a specific type of SATD that differs from traditional SATD in its type distribution. The finding that AI-era debt shifts toward requirement and test types is particularly important because it means that SATD categorization matters more, not less, in the AI era.

Our project builds on this paper in several ways:

1. **Shared label scheme**: Both papers use the Maldonado taxonomy (C/D, REQ, TES, DOC) for SATD classification, making results directly comparable.
2. **Extended scope**: While the GIST paper examines 81 AI-linked comments, our project develops an automated system that can detect and categorize SATD at scale across 458,232 issues.
3. **Practical tool**: The GIST paper is diagnostic (understanding the problem); our project is therapeutic (building a tool to manage it).
4. **AI-era relevance**: The GIST finding that AI shifts debt toward requirement and test types supports our multi-artifact approach. Different artifacts (comments vs issues vs commits vs PRs) carry different types of debt, and a one-size-fits-all detector would miss this nuance.

The GIST concept also connects to the Triple Debt framework in Paper 7: AI-generated code may create not just technical debt but also cognitive debt (developers don't understand what the AI wrote) and intent debt (the original goals and constraints are lost).

---

## 8. Paper 6: Speed at the Cost of Quality (Cursor)

### 8.1 Full Citation

He, Miller, Agarwal, Kaestner, Vasilescu (2026). *Speed at the Cost of Quality: How Cursor AI Increases Short-Term Velocity and Long-Term Complexity in Open-Source Projects.* MSR 2026 (23rd International Conference on Mining Software Repositories), Rio de Janeiro. Available at: https://doi.org/10.1145/3793302.3793349 and arXiv:2511.04427

### 8.2 Paper Type

Research paper. This is the first large-scale quasi-experimental study of a modern agentic IDE (Cursor), using difference-in-differences (DiD) methodology to establish causal effects.

### 8.3 Problem Statement

Practitioners claim that AI coding assistants like Cursor provide large productivity gains. But what happens at the project level when teams adopt Cursor? Does the quality of the codebase change? And does any quality change later affect development velocity?

### 8.4 Methodology

**Adoption detection**: The authors used first commit touching `.cursorrules` as a proxy for Cursor adoption.

**Sample**: 806 Cursor-adopting GitHub repositories (mostly August 2024 to March 2025) matched against 1,380 propensity-score-matched control repositories.

**Method**: Staggered difference-in-differences with the Borusyak imputation estimator. This is a rigorous causal inference method that accounts for staggered adoption timing and heterogeneous treatment effects.

**Outcomes measured**:
- Velocity: commits, lines added
- Quality: SonarQube static analysis warnings, code complexity, duplicate-line density
- Feedback loop: Panel GMM to test whether accumulated quality debt predicts later velocity slowdown

**Data sources**: GitHub Archive / BigQuery for commit data, SonarQube Community metrics for code quality.

### 8.5 Key Results and Findings

**Short-term velocity boost**: In the first month after adoption, Cursor-adopting projects show approximately 3-5x more lines added compared to controls. This is the "speed" that practitioners report.

**Velocity fades**: The velocity gains fade after approximately 2 months. By month 3, the difference between adopting and control projects is much smaller.

**Quality degrades**: Static analysis warnings increase by approximately 30%. Code complexity increases by approximately 41% (persistent, not fading). One table specification also reports approximately 25% complexity increase.

**Debt ratchet**: The panel GMM analysis shows that accumulated warnings and complexity predict later slowdown. In other words, the quality debt created during the initial velocity burst later taxes development velocity. The authors call this a "debt ratchet" — once quality degrades, it is hard to reverse, and the costs compound over time.

**Cross-language consistency**: The effects are qualitatively consistent across JavaScript/TypeScript, Python, and Go, suggesting the pattern is not language-specific.

### 8.6 Limitations

- `.cursorrules` is an imperfect adoption proxy. Some developers may adopt Cursor without creating a `.cursorrules` file.
- Open-source projects may differ from enterprises with mandatory quality assurance processes.
- The complexity metrics used (cyclomatic complexity, duplication) were designed for human-written code and may not perfectly capture AI-generated code quality.
- The study covers a snapshot of 2024-2025 tools; results may change as tools improve.
- Matching on observables cannot rule out all selection effects (developers who adopt Cursor may differ from those who don't in unobserved ways).
- The study does not measure security vulnerabilities directly.

### 8.7 Relevance to Our Project

This paper provides the strongest causal evidence that AI assistants increase code complexity and that this complexity later slows development. For our project, this is important because:

1. **SATD as a symptom**: The complexity increase documented by He et al. is exactly the kind of code that generates SATD. When developers accept AI-generated code that increases complexity by 41%, they are likely to leave notes like "TODO: refactor this" or "this is too complex, simplify later".

2. **Debt ratchet and our tool**: If accumulated quality debt predicts later slowdown, then early detection and categorization of SATD can help teams prioritize repayment before the debt compounds. Our tool provides exactly this capability.

3. **Scale of the problem**: 806 Cursor-adopting projects showing persistent 41% complexity increase represents a large-scale, real-world validation that AI-generated code creates technical debt. This is not a hypothetical concern — it is a documented phenomenon.

4. **Connection to Paper 2**: The velocity fade (speed gains disappear after 2 months) directly explains the productivity paradox: the initial burst of speed is real, but the quality debt it creates later cancels out the gains. Our project addresses this by providing automated debt detection that can catch the debt before it compounds.

5. **Multi-artifact relevance**: The study measures complexity at the code level. Our project extends this to the text level: we detect when developers acknowledge that the code they wrote (possibly with AI assistance) is problematic.

---

## 9. Paper 7: From Technical Debt to Cognitive and Intent Debt

### 9.1 Full Citation

Storey, M.-A. (2026). *From Technical Debt to Cognitive and Intent Debt: Rethinking Software Health in the Age of AI.* Essay/position article (University of Victoria); cites FoSE Thoughtworks retreat February 2026. Local PDF: StoreyTripleDebt.pdf

### 9.2 Paper Type

Essay/position article. This is a conceptual paper that proposes a new framework for understanding software health in the AI era. It is not an empirical study but a theoretical contribution based on literature synthesis, teaching experience, and community discussion.

### 9.3 Problem Statement

Traditional technical debt (code-level compromises) does not fully capture the risks of AI-assisted development. AI tools may help pay down code debt while simultaneously creating new forms of debt: cognitive debt (eroded shared understanding within teams) and intent debt (lost goals, constraints, and rationale for both humans and agents). The paper argues that software health metrics need to expand beyond code-level debt to include these higher-level concerns.

### 9.4 Methodology

The paper is argumentative and synthesizes multiple sources:

- Naur's (1985) theory of software as a system of knowledge
- Cunningham's original technical debt metaphor
- Kosmyna's work on cognitive load in development
- Shaw and Nave's concept of cognitive surrender (automating understanding rather than building it)
- Teaching case from the author's classroom
- Literature on AI-assisted development (Peng 2023, Hou 2024, Miller 2026, Starr and Storey 2026)

The paper uses a classroom vignette to illustrate the triple-debt model: students using AI tools produced functional code but could not explain why it worked, what assumptions it made, or how to modify it. This is cognitive debt — the understanding that should exist in developers' minds was never built because the AI did the thinking.

### 9.5 Key Results and Findings

**Triple-debt model**:
1. **Intent debt** (artifacts): Missing goals, constraints, and rationale. What was the code supposed to do? What were the non-functional requirements? These answers are often not captured when AI generates code quickly.
2. **Technical debt** (code): The traditional code-level compromises — hacks, workarounds, missing tests, outdated documentation.
3. **Cognitive debt** (shared understanding): Eroded team understanding of how the system works. When AI does the thinking, developers lose the mental model they need to maintain, debug, and extend the code.

**Diagnostic signals**:
- Change resistance: the system resists modifications that should be simple
- Unexpected results: the code does something the team did not anticipate
- Slow onboarding: new team members take much longer to understand the system
- Lost transactive memory: the team cannot remember who knows what about which parts
- Low bus factor: only one person (or no one) understands critical components

**Intent debt signals**:
- Behavior drift: the system does things that were never intended
- Agents needing excessive clarification: AI tools keep asking for more context because the original intent was never captured
- Forgotten non-functional requirements: performance, security, accessibility requirements that were stated once but never tracked

**Practices proposed**:
- Architecture Decision Records (ADRs) to capture design rationale
- Behavior-Driven Development (BDD) as executable intent
- Context artifacts that capture goals and constraints
- Walkthroughs that build shared understanding rather than just reviewing code
- Do not fully automate understanding — keep humans in the loop

### 9.6 Limitations

- This is a conceptual paper, not an empirical study. The triple-debt constructs have not been operationalized or validated at scale.
- The debate section of the paper itself acknowledges disagreement: some argue that documenting intent adds overhead that slows development; others argue it is essential.
- Measurement of cognitive and intent debt remains an open research problem.
- The paper draws heavily on a single teaching case, which may not represent industry practice.

### 9.7 Relevance to Our Project

Storey's triple-debt framework provides the theoretical foundation for our project's broader significance. Our tool addresses technical debt (SATD detection and categorization), but the framework suggests that this is only one layer of the problem.

The connection to our work is multi-layered:

1. **SATD as a diagnostic signal**: Storey's diagnostic signals for cognitive debt (slow onboarding, unexpected results, change resistance) are exactly the kinds of problems that SATD comments predict. When a developer writes "TODO: this is a hack", they are signaling that the code may create cognitive debt for future maintainers.

2. **Categorization matters more in the AI era**: The triple-debt model suggests that different types of debt require different responses. Code debt needs refactoring, cognitive debt needs knowledge sharing, and intent debt needs documentation. Our categorization system (C/D, REQ, TES, DOC) provides the first step toward routing SATD to the appropriate response.

3. **GIST connection**: The GIST concept from Paper 5 (AI-generated code adopted with uncertainty) is a specific instance of Storey's cognitive debt. Developers who use AI to generate code without understanding it accumulate cognitive debt, and GIST comments are the explicit acknowledgment of that debt.

4. **Verification-first approach**: Storey's argument that understanding should not be fully automated aligns with our project's approach: we do not try to fully automate debt management, we provide a tool that helps developers identify and prioritize what needs human attention.

5. **Future direction**: The triple-debt framework suggests natural extensions for our work. In future phases, we could expand our categorization to include cognitive and intent debt types, providing a richer picture of AI-era software health.

---

## 10. Paper 8: Slopsquatting

### 10.1 Full Citation

Park, S. (2025). *Slopsquatting: Hallucination in Coding Agents and Vibe Coding.* Trend Research technical brief (Trend Micro). Available as techbrief-slopsquatting.pdf; also referenced on GitHub.

### 10.2 Paper Type

Industry technical brief. This paper evaluates whether AI coding agents hallucinate package names that attackers can register (squat), creating supply chain vulnerabilities.

### 10.3 Problem Statement

When AI coding agents generate code, they sometimes invent package names that do not exist. If an attacker registers these hallucinated names on package registries (like PyPI), they can distribute malware to developers who copy the AI-generated code. This attack vector is called "slopsquatting" — a play on "typosquatting" where the vulnerability comes from AI hallucination rather than human error.

The question is whether reasoning coding agents and MCP (Model Context Protocol) augmented workflows reduce or eliminate this hallucination risk.

### 10.4 Methodology

The study used a multi-stage agent pipeline:

1. **Prompt composer**: Generates 100 Python web development tasks, each requiring at least 4 recent libraries.
2. **Solver agents**: Multiple foundation models (GPT-4.1 family, GPT-4o, o4-mini, Gemini 2.x, Claude 3.5/3.7) generate code for each task.
3. **Module extractor**: Identifies which packages the generated code imports.
4. **Manual evaluation**: The 10 highest-hallucination tasks are tested on Claude Code CLI, Codex CLI, and Cursor Agent with MCP (Context7, Sequential Thinking, custom Tavily).
5. **Qualitative analysis**: Examines the mechanisms behind hallucination.

The dataset (HallucinationDB) is a SQLite database containing the 100 generated tasks and model outputs.

### 10.5 Key Results and Findings

**Foundation model hallucination**: Most models are mostly clean for common libraries but spike to 2-4 phantom (hallucinated) package names when tasks require novel or recent library combinations.

**Agent mitigation**: Coding agents cut hallucination roughly in half compared to raw model generation, but they do not eliminate it entirely.

**MCP best but not perfect**: Cursor with MCP (Context7, Sequential Thinking, Tavily) achieved the lowest hallucination rate, but still misses via:
- Cross-ecosystem name borrowing (suggesting a JavaScript package name for a Python project)
- Morpheme splicing (combining parts of real package names into fake ones)

**PyPI checks insufficient**: Even checking whether a package exists on PyPI is not enough if attackers pre-register the hallucinated names before the developer tries to install them.

**Hallucination mechanisms**:
- Context-gap filling: when the model does not know the right package, it invents one that sounds plausible
- Surface-form mimicry: the invented name follows the naming pattern of real packages
- Edge-domain pressure: novel or niche requirements increase hallucination because the model has less training data to draw from

### 10.6 Limitations

- The 100 synthetic tasks are designed to maximize hallucination (stress test), not to represent typical CRUD development. Results may overestimate hallucination rates in normal development.
- Agent comparison was done on only 10 tasks with manual evaluation, which is a very small sample.
- The Cursor Auto-select model was not fixed, introducing variability.
- This is a vendor brief (Trend Micro), not a peer-reviewed paper.
- Does not measure actual malware installs in the wild — the risk is theoretical until someone actually registers a hallucinated name.

### 10.7 Relevance to Our Project

Slopsquatting is relevant to our project because it represents another form of AI-generated debt: supply chain risk. When an AI invents a package name, the developer who copies that code is introducing a dependency on a non-existent or malicious package. If they later write a comment like "TODO: verify this dependency" or "hack: using unverified package", that is SATD.

The paper's finding that MCP validation reduces but does not eliminate hallucination aligns with the broader theme across all reviewed papers: AI tools reduce certain risks but create others. Our project addresses the text-level acknowledgment of these risks (SATD) while slopsquatting research addresses the technical mechanism.

For our project specifically, the slopsquatting paper provides context for why SATD in pull requests and issues may increase: developers using AI tools may generate code with hallucinated dependencies, then leave notes about the uncertainty. Our system can detect these notes.

---

## 11. Paper 9: Deep Learning and Data Augmentation for Detecting SATD (Baseline)

### 11.1 Full Citation

Sutoyo, E., Avgeriou, P., & Capiluppi, A. (2024). *Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt.* arXiv:2410.15804.

### 11.2 Paper Type

Research paper. This is the primary baseline paper for our experimental comparison. It applies deep learning models (BiLSTM and BERT) with GPT-style data augmentation to detect and categorize SATD across four software artifacts.

### 11.3 Problem Statement

SATD appears across multiple software artifacts (code comments, issues, commit messages, pull requests), and different artifact types have different language characteristics. Previous approaches used traditional machine learning or shallow deep learning. This paper asks whether modern deep learning models (BiLSTM for binary detection, BERT for multi-class categorization) combined with data augmentation can improve SATD detection and categorization across all four artifact types.

### 11.4 Methodology

**Two tasks**:
1. **Identification**: Binary classification — is this text SATD or Not-SATD?
2. **Categorization**: Four-class classification on SATD rows only — Code/Design (C/D), Requirement (REQ), Test (TES), Documentation (DOC).

**Models**:
- Identification: GloVe embeddings + BiLSTM
- Categorization: Fine-tuned BERT

**Data augmentation**: GPT-style augmentation (AugGPT) to address class imbalance. The augmentation generates synthetic SATD examples for underrepresented classes.

**Evaluation**: Macro F1 on held-out test set, per artifact type.

**Artifacts**: Code comments, issues, commit messages, pull requests — separate models for each.

### 11.5 Key Results and Findings

**Identification results (BiLSTM+AugGPT)**:
| Artifact | Macro F1 |
|---|---|
| Code comments | 0.939 |
| Issues | 0.878 |
| Pull requests | 0.862 |
| Commits | 0.910 (supervisor replication) |

**Categorization results (BERT+AugGPT)**:
| Artifact | Macro F1 |
|---|---|
| Code comments | 0.882 |
| Issues | 0.899 |
| Pull requests | 0.876 |
| Commits | 0.980 (supervisor replication) |

**Key observations**:
- BiLSTM performs well on identification, especially for code comments (0.939) and commits (0.910).
- BERT performs well on categorization, especially for commits (0.980).
- Different artifacts have different difficulty levels: commits are easiest for categorization but hard for identification; issues and pull requests are harder overall.
- Data augmentation is critical for handling class imbalance — without it, models would bias toward the majority class (Not-SATD).

### 11.6 Limitations

- The paper uses purpose-trained models (BiLSTM, BERT) that require significant compute and training data. This raises the question: can cheaper alternatives achieve comparable results?
- The data augmentation strategy (AugGPT) may create distribution-specific effects. If the augmentation is what makes the model work, the results may not generalize to unaugmented data.
- The paper does not systematically compare different data protocols. It uses one specific augmentation approach and reports results on that basis.
- Commit-message baselines in some replications use supervisor replication values (BiLSTM 0.91, BERT 0.9804), which may differ from the paper's original published values.

### 11.7 Relevance to Our Project

This is our primary comparison target. Every result in our project is measured against the scores reported (or replicated) by Sutoyo et al. The paper establishes the state of the art that we aim to match or exceed with cheaper methods.

Our project addresses the paper's limitations directly:

1. **Cost**: We test frozen embeddings (no fine-tuning) plus lightweight classifiers (XGBoost, logistic regression) instead of fine-tuned BiLSTM and BERT. If frozen embeddings match the baselines, they provide a much cheaper alternative.

2. **Data protocol**: We test two different data protocols (Pipeline A: 31-source rebuild with T5 equalization; Pipeline B: paper's AugGPT files with natural imbalance). This isolates whether the paper's results depend on the specific data distribution or on the model architecture.

3. **Scale**: We deploy the best models on 458,232 unseen issues, demonstrating that the approach works at scale, not just on curated test sets.

4. **Embedding variety**: We test three different frozen embeddings (Qwen3, E5, BGE-M3) to understand whether the embedding model matters or whether the classifier is more important (RQ4).

The paper's results serve as the benchmark for our RQ1 (identification) and RQ2 (categorization). Our finding that frozen embeddings beat BERT on categorization in 3 of 4 artifacts (while losing on identification in Pipeline A) provides a nuanced comparison that goes beyond simple "better or worse" claims.

---

## 12. Summary Comparison Table

| Paper | Type | Key Method | Key Finding | Direct Relevance |
|---|---|---|---|---|
| Ge et al. 2025 (Vibe Coding Survey) | Survey (1000+ papers) | CMDP formalization, five development models | Success depends on context engineering, not agent skill | Landscape context; connects to RQ4 (classifier > embedding) |
| Singh 2025 (Productivity Paradox) | Review | Verification Budget model | Narrow tasks +55.8%, realistic OSS -19%, review time +91% | Framing: verification is the bottleneck, not generation |
| Veracode 2025 (Security Report) | Industry benchmark | 80-task SAST benchmark across 4 CWEs | ~45% of LLM code introduces target flaw; GPT-5 Mini 72% best | AI code is not automatically secure; need for automated triage |
| Bouras et al. 2025 (HoarePrompt) | Research | Natural Hoare triples + k-induction | MCC +61% over CoT; step-by-step reasoning helps verification | Verification approach; classifier > raw model capability |
| Al Mujahid & Imran 2026 (GIST) | Research | GitHub search + annotation (n=81) | AI shifts SATD toward requirement/test debt; AI as catalyst 42% | Most directly relevant: AI-era SATD is different |
| He et al. 2026 (Cursor DiD) | Research (MSR) | DiD with propensity matching (n=2,186 repos) | Velocity +3-5x first month, complexity +41% persistent | Causal evidence: AI increases code debt that later slows velocity |
| Storey 2026 (Triple Debt) | Position paper | Conceptual framework | Cognitive and intent debt alongside technical debt | Theoretical frame: SATD is one layer of AI-era software health |
| Park 2025 (Slopsquatting) | Industry brief | Agent hallucination evaluation (n=100 tasks) | Agents halve but do not eliminate phantom package names | Supply chain risk as another form of AI-generated debt |
| Sutoyo et al. 2024 (Baseline) | Research | BiLSTM + BERT + AugGPT | BiLSTM 0.939/0.878/0.862/0.910 identification; BERT 0.882/0.899/0.876/0.980 categorization | Primary experimental baseline for all our comparisons |

---

## 13. Thematic Analysis

### 13.1 Theme 1: The Verification Bottleneck

Six of the nine papers converge on the same conclusion: the bottleneck in AI-assisted software development has shifted from code generation to code verification. Papers 1, 2, 3, 4, 6, and 8 all document different aspects of this shift:

- Paper 1: context engineering and verification matter more than agent skill
- Paper 2: T_verify and T_rework dominate T_write in the delivery equation
- Paper 3: 45% of LLM code introduces vulnerabilities that require verification
- Paper 4: verification requires step-by-step reasoning, not just output inspection
- Paper 6: velocity gains from AI fade after 2 months as quality debt accumulates
- Paper 8: agents reduce but do not eliminate hallucination risks

Our project addresses this bottleneck by providing automated SATD detection and categorization, which is one form of verification: checking whether developers have acknowledged that code needs further work.

### 13.2 Theme 2: AI Changes the Nature of Debt

Papers 5, 6, and 7 document that AI does not simply create more of the same debt — it changes the type and nature of debt:

- Paper 5: AI shifts SATD toward requirement and test debt (less pure design debt)
- Paper 6: AI creates persistent complexity increases (+41%) that do not fade
- Paper 7: AI creates cognitive and intent debt alongside technical debt

This means that SATD detection and categorization tools need to be sensitive to AI-era debt patterns. Our multi-class categorization (C/D, REQ, TES, DOC) is well-positioned for this because it can distinguish between different debt types.

### 13.3 Theme 3: Cheap Methods Can Compete

Papers 1, 2, 4, and the baseline paper collectively raise the question of whether expensive, purpose-trained models are necessary. Our project provides evidence that frozen embeddings plus XGBoost can match or exceed fine-tuned BERT for SATD categorization, supporting the broader trend toward simpler, cheaper methods:

- Paper 1: success depends on context engineering, not model size
- Paper 4: structured reasoning (HoarePrompt) outperforms raw model capability
- Our results: classifier choice matters more than embedding brand (RQ4)

### 13.4 Theme 4: Data Protocol Matters More Than Architecture

Our RQ3 finding — that identification results change dramatically between Pipeline A and Pipeline B — resonates with a broader observation across the literature:

- Paper 2: the same AI tool produces different results in narrow vs realistic tasks
- Paper 6: the same tool produces different results in the first month vs later months
- Paper 8: the same model produces different hallucination rates for common vs novel libraries

The lesson is consistent: performance claims must be qualified by the data protocol, task context, and evaluation setting. This is why our project insists on reporting Pipeline A and Pipeline B results separately.

### 13.5 Theme 5: Multi-Artifact Thinking

All papers agree that software artifacts are not homogeneous. Different artifacts carry different signals:

- Paper 5: comments, issues, commits, and PRs carry different debt types
- Paper 6: complexity increases at the code level have different effects than at the commit level
- Our project: separate models per artifact are necessary because language characteristics differ

---

## 14. Research Gaps Identified

### 14.1 Gap 1: No Frozen Embedding Comparison for SATD

While the baseline paper (Sutoyo et al. 2024) demonstrates strong results with fine-tuned BiLSTM and BERT, no prior study has tested whether frozen, general-purpose embeddings can achieve comparable performance at lower training cost. This is our RQ1 and RQ2.

### 14.2 Gap 2: No Protocol Contrast Study

No prior study systematically compares SATD detection results under two different data protocols using the same model architecture. The baseline paper uses one specific data distribution (AugGPT-augmented with Not-SATD majority). Our Pipeline A tests a different distribution (31-source rebuild with five-class equalization). This is our RQ3.

### 14.3 Gap 3: No Embedding vs Classifier Sensitivity Analysis

While many SATD studies test different model architectures, none systematically separates the contribution of the text representation (embedding) from the contribution of the classifier head. Our RQ4 addresses this directly.

### 14.4 Gap 4: No Large-Scale Deployment Study

The baseline paper evaluates on curated test sets. No study has deployed SATD detection models on hundreds of thousands of real, unseen issues to measure real-world applicability. Our 458,232-issue inference fills this gap.

### 14.5 Gap 5: Limited AI-Era SATD Research

The GIST paper (Al Mujahid & Imran 2026) provides the first empirical data on AI-linked SATD, but with only 81 samples. Our project develops an automated system that can study AI-era SATD at scale, enabling future research on how AI assistants change debt patterns across large corpora.

### 14.6 Gap 6: No Integration of SATD Detection with Verification Frameworks

Papers 2 and 4 propose verification frameworks (Verification Budget, HoarePrompt), but neither integrates with automated SATD detection. Our project demonstrates that SATD detection can serve as one component of a verification pipeline, catching developer-acknowledged debt before it compounds.

---

## 15. How These Papers Relate to Our Project

### 15.1 Intellectual Foundation

The nine papers collectively establish that:

1. AI coding assistants create real, measurable technical debt (Papers 1, 2, 5, 6)
2. This debt is different from traditional debt in type and nature (Papers 5, 7)
3. Verification is the bottleneck, not generation (Papers 1, 2, 3, 4)
4. Automated tools are needed to manage the verification burden (Papers 2, 3, 4)
5. SATD is a valuable signal because it comes from the developer themselves (Paper 5)

Our project builds on this foundation by developing an automated system that detects and categorizes SATD across four artifact types using cheap, frozen embeddings.

### 15.2 Experimental Design

The papers inform our experimental design in specific ways:

- **Per-artifact models**: Paper 5 shows that different artifacts carry different debt types, supporting our decision to train separate models per artifact.
- **Multi-class categorization**: Paper 5's finding that AI shifts debt toward requirement and test types supports our four-class categorization scheme.
- **Protocol contrast**: Papers 2 and 6 show that the same tool produces different results in different contexts, motivating our dual-pipeline design.
- **Classifier sensitivity**: Paper 1's finding that context engineering matters more than model capability motivates our RQ4 (embedding vs classifier).

### 15.3 Interpretation of Results

The papers help us interpret our results:

- **Categorization wins**: Our finding that frozen embeddings beat BERT on categorization aligns with Paper 1's conclusion that agent skill matters less than context engineering. The embedding provides the context; the classifier provides the engineering.
- **Identification protocol dependence**: Our finding that identification depends on the data protocol aligns with Paper 2's observation that performance depends on the task context (narrow vs realistic).
- **Commits as hardest artifact**: The difficulty of commit messages aligns with Paper 5's finding that short, terse text carries less signal than longer text.

### 15.4 Broader Significance

Our project contributes to the verification-first paradigm documented across the literature:

- We provide a cheap alternative to expensive fine-tuned models (addressing Paper 2's verification budget concern)
- We detect developer-acknowledged debt at scale (addressing Paper 5's GIST phenomenon)
- We categorize debt types to enable targeted response (addressing Paper 7's triple-debt framework)
- We deploy on real issues to demonstrate practical applicability (addressing Paper 6's call for QA-first agent design)

---

## 16. Detailed Cross-Paper Comparison by Methodology Type

### 16.1 Survey and Review Papers (Papers 1, 2)

Papers 1 (Ge et al. 2025) and 2 (Singh 2025) are both secondary studies that synthesize existing evidence. They differ in scope and rigor:

| Dimension | Paper 1 (Vibe Coding Survey) | Paper 2 (Productivity Paradox) |
|---|---|---|
| Scope | 1,000+ papers, full AI coding ecosystem | Focused on productivity measurement |
| Formalization | CMDP model (conceptual) | Verification Budget equation |
| Primary contribution | Taxonomy of development models | Evidence synthesis with coded sources |
| Data availability | No primary data | Coded evidence from 7+ sources |
| Reproducibility | Not applicable (survey) | Partially reproducible (source coding) |

Both papers arrive at the same conclusion from different angles: Paper 1 says success depends on context engineering; Paper 2 says the bottleneck is verification, not generation. Our project inherits both insights: the context (data protocol) matters more than the model (embedding), and verification (SATD detection) is more valuable than generation (code writing).

### 16.2 Industry Benchmark Reports (Papers 3, 8)

Papers 3 (Veracode 2025) and 8 (Park 2025) are industry-authored reports that test specific failure modes of AI coding tools:

| Dimension | Paper 3 (Security Report) | Paper 8 (Slopsquatting) |
|---|---|---|
| Sample size | 80 tasks (4 CWEs x 4 languages x 5 tasks) | 100 tasks (10 highest-hallucination for deep eval) |
| Methodology | Automated SAST scanning | Multi-agent pipeline + manual evaluation |
| Key metric | Security pass rate | Hallucination rate per model |
| Models tested | 100+ LLMs | 6+ foundation models, 3 agent tools |
| Reproducibility | High (standardized benchmark) | Medium (HallucinationDB available) |

Both reports demonstrate that AI tools have systematic failure modes: Paper 3 shows ~45% of LLM code introduces security vulnerabilities; Paper 8 shows agents hallucinate package names. These failure modes generate SATD: developers who encounter security issues or hallucinated dependencies leave notes acknowledging the problems. Our system can detect these notes at scale.

### 16.3 Empirical Research Papers (Papers 5, 6)

Papers 5 (Al Mujahid & Imran 2026) and 6 (He et al. 2026) provide the strongest empirical evidence for AI-era technical debt:

| Dimension | Paper 5 (GIST) | Paper 6 (Cursor DiD) |
|---|---|---|
| Sample size | 81 AI-linked SATD comments | 2,186 repositories (806 treatment + 1,380 control) |
| Methodology | GitHub search + manual annotation | Difference-in-differences with propensity matching |
| Causal claims | Descriptive (no causal inference) | Causal (DiD with staggered adoption) |
| Data source | Public GitHub repos | GitHub Archive + SonarQube |
| Key finding | AI shifts SATD toward REQ/TES | AI increases complexity by 41%, velocity fades after 2 months |
| Reproducibility | Medium (search queries provided) | High (public data, standard methods) |

Paper 5 tells us what AI-era SATD looks like (different type distribution); Paper 6 tells us why it matters (persistent complexity that slows velocity). Our project builds on both: we detect the SATD that Paper 5 describes, and we provide a tool that can catch the debt before it causes the velocity slowdown that Paper 6 documents.

### 16.4 Theoretical and Position Papers (Papers 4, 7)

Papers 4 (HoarePrompt) and 7 (Storey Triple Debt) are more theoretical in nature:

| Dimension | Paper 4 (HoarePrompt) | Paper 7 (Triple Debt) |
|---|---|---|
| Type | Research with experiments | Position/conceptual essay |
| Primary contribution | Natural Hoare triples for verification | Triple-debt framework |
| Empirical basis | Controlling for Python programs | Teaching case + literature synthesis |
| Key insight | Step-by-step reasoning improves verification | Cognitive and intent debt alongside technical debt |
| Practical output | Verification tool (HoarePrompt) | Practices (ADRs, BDD, walkthroughs) |

Paper 4 provides a verification methodology; Paper 7 provides a theoretical framework. Our project connects them: we detect SATD (a form of developer-acknowledged verification failure) and categorize it into types that map to the triple-debt framework (C/D = technical debt, REQ = intent debt, TES = technical debt, DOC = intent debt).

---

## 17. Methodology Critique Across Papers

### 17.1 Strengths

**Paper 1 (Ge et al. 2025)**: Comprehensive survey with formal CMDP model. The five development models provide a useful taxonomy. The inclusion of METR data (developers 19% slower with AI) adds empirical grounding.

**Paper 2 (Singh 2025)**: Strong evidence synthesis with coded sources. The Verification Budget model is intuitive and actionable. The paper honestly acknowledges limitations (student work, not peer-reviewed).

**Paper 3 (Veracode 2025)**: Large-scale, standardized benchmark. Over 100 LLMs tested. Clear methodology with reproducible protocol.

**Paper 4 (Bouras et al. 2025)**: Novel approach to verification. Strong experimental design with multiple baselines. Clear connection between formal methods and natural language reasoning.

**Paper 5 (Al Mujahid & Imran 2026)**: First empirical study of AI-linked SATD. Strong inter-annotator agreement (Cohen's kappa 0.896). Introduction of GIST concept fills a clear gap.

**Paper 6 (He et al. 2026)**: Rigorous causal inference (DiD with propensity matching). Large sample (2,186 repos). Longitudinal design captures fading effects.

**Paper 7 (Storey 2026)**: Important theoretical contribution. The triple-debt framework is well-motivated and clearly articulated.

**Paper 8 (Park 2025)**: Novel attack vector analysis. Multi-model comparison. Practical implications for supply chain security.

**Paper 9 (Sutoyo et al. 2024)**: Comprehensive baseline with multiple models and artifacts. Data augmentation strategy is well-motivated.

### 17.2 Weaknesses

**Paper 1 (Ge et al. 2025)**: CMDP model is conceptual, not empirically validated. Very broad scope means shallow coverage of individual topics.

**Paper 2 (Singh 2025)**: Student work, not peer-reviewed. Verification Budget model is not quantified. Many sources are industry reports with different measurement standards.

**Paper 3 (Veracode 2025)**: Vendor-authored report. Function-level tasks without application context. Only 4 CWEs and 4 languages.

**Paper 4 (Bouras et al. 2025)**: Expensive token usage. Evaluated mainly on contest-style programs. Binary classification only.

**Paper 5 (Al Mujahid & Imran 2026)**: Small sample (n=81). Keyword-based detection may miss non-standard SATD. Python/JavaScript only.

**Paper 6 (He et al. 2026)**: .cursorrules is an imperfect adoption proxy. Open-source projects may not generalize to enterprises. No direct security measurement.

**Paper 7 (Storey 2026)**: Conceptual, not empirical. Single teaching case. Triple-debt constructs not operationalized.

**Paper 8 (Park 2025)**: Synthetic stress-test tasks. Very small deep-evaluation sample (10 tasks). Vendor brief, not peer-reviewed.

**Paper 9 (Sutoyo et al. 2024)**: One data protocol. Purpose-trained models may not generalize. No systematic comparison of data vs model effects.

---

## 18. Dataset and Reproducibility Analysis

### 18.1 Dataset Availability

| Paper | Data Available | Format | Size |
|---|---|---|---|
| Paper 1 | No primary data | N/A | 1,000+ papers surveyed |
| Paper 2 | Coded sources (methodology described) | Text coding | 7+ sources coded |
| Paper 3 | Benchmark protocol described | SAST output | 80 tasks |
| Paper 4 | Artifacts on Figshare | Code + data | Contest-style programs |
| Paper 5 | Search queries provided | GitHub search results | 81 SATD comments |
| Paper 6 | Public GitHub data + SonarQube | BigQuery + metrics | 2,186 repos |
| Paper 7 | Teaching case (anecdotal) | Text | Single classroom example |
| Paper 8 | HallucinationDB (SQLite) | Database | 100 tasks |
| Paper 9 | AugGPT files (shared by supervisor) | CSV | ~109k rows |

### 18.2 Reproducibility Assessment

**Most reproducible**: Paper 6 (public data, standard DiD methods), Paper 3 (standardized benchmark), Paper 8 (HallucinationDB).

**Partially reproducible**: Paper 5 (search queries provided, but GitHub search results change over time), Paper 9 (AugGPT files available, but original training code not published).

**Least reproducible**: Paper 1 (survey, no primary data), Paper 2 (synthesis, not original experiment), Paper 7 (conceptual essay).

### 18.3 Our Project's Reproducibility

Our project improves reproducibility by:
1. Publishing all result CSVs in the repository.
2. Sharing embedding .npy files and trained .joblib models via Google Drive.
3. Providing complete notebook code for both pipelines.
4. Publishing the inference output (458k classified issues) as Parquet.
5. Including a reproduction guide with exact parameter settings.

---

## 19. Impact Assessment

### 19.1 Paper Impact Rankings (by Citation and Relevance)

| Rank | Paper | Estimated Impact | Why |
|---|---|---|---|
| 1 | Sutoyo et al. 2024 (Baseline) | Highest | Direct experimental baseline for all our comparisons |
| 2 | Al Mujahid & Imran 2026 (GIST) | Very High | First empirical evidence that AI-era SATD is different |
| 3 | He et al. 2026 (Cursor DiD) | High | Causal evidence that AI increases code debt |
| 4 | Singh 2025 (Productivity Paradox) | High | Strongest framing for why verification matters |
| 5 | Ge et al. 2025 (Vibe Coding Survey) | Medium-High | Comprehensive landscape overview |
| 6 | Veracode 2025 (Security Report) | Medium | Quantifies AI code security risks |
| 7 | Bouras et al. 2025 (HoarePrompt) | Medium | Verification methodology with practical tools |
| 8 | Storey 2026 (Triple Debt) | Medium | Theoretical framework for future work |
| 9 | Park 2025 (Slopsquatting) | Low-Medium | Novel attack vector, but narrow scope |

### 19.2 How Papers Cite Each Other

- Paper 1 cites Papers 2, 3, 6 (vibe coding ecosystem context)
- Paper 2 cites Papers 1, 6 (productivity measurement)
- Paper 5 cites Paper 9 (SATD baseline)
- Paper 6 cites Papers 1, 2 (AI adoption context)
- Paper 7 cites Papers 1, 5, 6 (triple-debt grounding)
- Our project cites all nine papers

### 19.3 Influence on Our Research Design

| Paper | Design Decision Influenced |
|---|---|
| Paper 1 | RQ4 (classifier vs embedding sensitivity) |
| Paper 2 | Dual-pipeline design (data protocol contrast) |
| Paper 5 | Four-class categorization scheme (C/D, REQ, TES, DOC) |
| Paper 6 | Large-scale inference on real issues |
| Paper 7 | Future direction: cognitive and intent debt |
| Paper 9 | Baseline scores for RQ1 and RQ2 |

---

## 20. Future Research Directions Identified

### 20.1 Short-Term Extensions

1. **Deploy E5 and BGE-M3 on 458k issues**: Currently only Qwen3+XGBoost has been deployed at scale. Running E5 and BGE-M3 on the same issues would provide deployment-side comparison.

2. **Reconcile binary vs categorical labels**: The current inference output shows 0/1 labels rather than per-type (C/D, REQ, TES, DOC). Re-running the categorizer head would provide richer output.

3. **Cross-artifact transfer**: Test whether models trained on one artifact (e.g., code comments) can detect SATD in another (e.g., issues). This would test generalizability.

### 20.2 Medium-Term Research

4. **AI-era SATD tracking**: Use our automated system to track how SATD patterns change over time in repositories that adopt AI coding assistants. This connects Paper 5 (GIST) with Paper 6 (Cursor DiD) in a longitudinal study.

5. **Triple-debt operationalization**: Extend the categorization scheme to include cognitive and intent debt types from Paper 7. This would require new annotation guidelines and potentially new label definitions.

6. **Keyword-feature integration**: Combine embedding-based classification with KeyBERT keyword signals (already collected in data/keywords/) to test whether keywords add value beyond embeddings.

### 20.3 Long-Term Vision

7. **Real-time SATD monitoring**: Integrate SATD detection into CI/CD pipelines so that new SATD is caught at commit time, not after deployment.

8. **Multi-modal SATD detection**: Combine text analysis with code complexity metrics (Paper 6's SonarQube data) to predict SATD from both code and text signals.

9. **Verification pipeline integration**: Combine SATD detection with verification frameworks (Paper 4's HoarePrompt, Paper 2's Verification Budget) to create a comprehensive AI-code quality assurance system.

---

## 21. Key Definitions and Terminology

### 21.1 Self-Admitted Technical Debt (SATD)

SATD occurs when developers explicitly acknowledge in text that some part of the code, documentation, or process is incomplete, temporary, or compromised. Common markers include TODO, FIXME, HACK, XXX, and similar annotations. The key property is that the developer has already identified the problem — the signal is explicit, not inferred.

### 21.2 GIST (Generative AI-Induced Technical Debt)

Introduced by Al Mujahid & Imran (2026), GIST describes the specific pattern of adopting AI-generated code while acknowledging uncertainty about its behavior or correctness. GIST captures the tension between speed and understanding that characterizes AI-assisted development.

### 21.3 Verification Budget

Introduced by Singh (2025), the Verification Budget model describes the total amount of code review, testing, and quality assurance a team can perform per unit time. When AI generates code faster than humans can verify it, a verification deficit accumulates as technical debt.

### 21.4 Triple-Debt Framework

Introduced by Storey (2026), the triple-debt framework identifies three forms of software debt: technical debt (code-level compromises), cognitive debt (eroded shared understanding), and intent debt (lost goals and rationale). The framework argues that AI tools may reduce technical debt while increasing cognitive and intent debt.

### 21.5 Slopsquatting

Introduced by Park (2025), slopsquatting describes the attack vector where AI coding agents hallucinate package names that attackers can register on package registries. The vulnerability comes from AI hallucination rather than human error.

### 21.6 Vibe Coding

Introduced by Ge et al. (2025), vibe coding describes the practice of using AI agents to generate code based on high-level descriptions, where developers validate the output through outcomes rather than line-by-line inspection.

### 21.7 Hoare Triads

Used by Bouras et al. (2025) in HoarePrompt, natural Hoare triples describe program correctness in natural language: {precondition} code {postcondition}. The approach adapts formal verification concepts to work with LLMs and natural language requirements.

---

## 22. Quantitative Summary of All Papers

### 22.1 Sample Sizes

| Paper | Sample Size | Unit |
|---|---|---|
| Paper 1 | 1,000+ | Papers surveyed |
| Paper 2 | 7+ | Sources coded |
| Paper 3 | 80 | Tasks (4 CWEs x 4 langs x 5 tasks) |
| Paper 4 | ~100 | Contest-style programs |
| Paper 5 | 81 | SATD comments |
| Paper 6 | 2,186 | Repositories (806 + 1,380) |
| Paper 7 | 1 | Teaching case |
| Paper 8 | 100 | Tasks (10 deep-evaluated) |
| Paper 9 | ~109k | SATD rows |

### 22.2 Key Metrics Reported

| Paper | Primary Metric | Value |
|---|---|---|
| Paper 1 | Developer speed change | -19% with Cursor (METR) |
| Paper 2 | Review time increase | +91% (Faros AI) |
| Paper 3 | Security pass rate | ~55% average (best: 72%) |
| Paper 4 | MCC improvement | +61% over CoT |
| Paper 5 | AI-role distribution | Catalyst 42%, Source 27% |
| Paper 6 | Complexity increase | +41% persistent |
| Paper 7 | N/A (conceptual) | N/A |
| Paper 8 | Hallucination rate | 2-4 phantom packages per task |
| Paper 9 | Identification F1 | 0.862-0.939 (BiLSTM) |
| Paper 9 | Categorization F1 | 0.876-0.980 (BERT) |

### 22.3 Our Results vs Baselines

| Task | Artifact | Baseline | Our Best (Pipeline B) | Delta |
|---|---|---|---|---|
| Identification | Code comments | 0.939 | 0.966 (Qwen3) | +0.027 |
| Identification | Issues | 0.878 | 0.866 (Qwen3) | -0.012 |
| Identification | Pull requests | 0.862 | 0.870 (E5) | +0.008 |
| Identification | Commits | 0.910 | 0.901 (BGE-M3) | -0.009 |
| Categorization | Code comments | 0.882 | 0.958 (E5) | +0.076 |
| Categorization | Issues | 0.899 | 0.956 (BGE-M3) | +0.057 |
| Categorization | Pull requests | 0.876 | 0.964 (BGE-M3) | +0.088 |
| Categorization | Commits | 0.980 | 0.964 (Qwen3) | -0.016 |

---

## 23. Conclusion

This literature review examined nine papers that collectively establish the intellectual foundation for our SATD Embedding Comparison project. The papers span surveys, empirical studies, benchmark reports, and position papers, covering the full spectrum from AI coding infrastructure to theoretical frameworks for software health.

The key findings from the literature are:

1. **The verification bottleneck is real and well-documented.** AI tools generate code faster than humans can verify it, creating a growing verification deficit that manifests as technical, cognitive, and intent debt.

2. **AI changes the nature of technical debt.** AI-era SATD shifts toward requirement and test types, cognitive debt erodes team understanding, and intent debt loses design rationale. These are qualitatively different from traditional code-level debt.

3. **Data context matters more than model capability.** Across multiple studies, the same model produces very different results depending on the data distribution, task context, and evaluation setting.

4. **Cheap methods can compete.** Frozen embeddings, structured reasoning, and context engineering can match or exceed expensive fine-tuned models when the task is well-defined and the data is appropriate.

5. **SATD is a valuable signal in the AI era.** Developer-acknowledged debt provides a direct, actionable signal that can be detected and categorized automatically, enabling prioritized repayment.

Our project addresses identified research gaps by testing frozen embeddings against fine-tuned baselines under two data protocols, deploying on 458k real issues, and providing a live demonstration. The results show that frozen embeddings plus XGBoost are a strong, cheap alternative for SATD categorization, while identification performance depends critically on the data protocol.

The literature establishes that AI-era software development requires new tools for managing technical debt. Our project contributes one such tool: an automated SATD detector that works with cheap frozen embeddings, operates across four artifact types, and has been validated on 458,232 real issues. The literature also suggests natural extensions: tracking AI-era SATD patterns over time, expanding categorization to include cognitive and intent debt, and integrating SATD detection into broader verification pipelines.

---

## 24. References

1. Ge, Mei, Duan, Li, Zheng, Wang, Wang, Yao, Liu, Cai, Bi, Guo, Guo, Liu, Cheng (2025). A Survey of Vibe Coding with Large Language Models. arXiv preprint (cs.AI). https://arxiv.org/abs/2510.12399

2. Singh, J. (2025). From Code Generation to Code Verification: Explaining and Mitigating the AI Productivity Paradox in Software Engineering. Chitkara University, B.E. CSE.

3. Veracode Research (2025). GenAI Code Security Report: Assessing the Security Performance of Newer LLMs (October 2025 Update). https://www.veracode.com

4. Bouras, Dai, Wang, Xiong, Mechtaev (2025). HoarePrompt: Structural Reasoning About Program Correctness in Natural Language. arXiv preprint (cs.SE). https://arxiv.org/abs/2503.19599

5. Al Mujahid, A. & Imran, M. M. (2026). "TODO: Fix the Mess Gemini Created": Towards Understanding GenAI-Induced Self-Admitted Technical Debt. arXiv cs.SE. https://arxiv.org/abs/2601.07786

6. He, Miller, Agarwal, Kaestner, Vasilescu (2026). Speed at the Cost of Quality: How Cursor AI Increases Short-Term Velocity and Long-Term Complexity in Open-Source Projects. MSR 2026. https://doi.org/10.1145/3793302.3793349

7. Storey, M.-A. (2026). From Technical Debt to Cognitive and Intent Debt: Rethinking Software Health in the Age of AI. University of Victoria.

8. Park, S. (2025). Slopsquatting: Hallucination in Coding Agents and Vibe Coding. Trend Research / Trend Micro.

9. Sutoyo, E., Avgeriou, P., & Capiluppi, A. (2024). Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt. arXiv:2410.15804.

---

*Document prepared for the SATD Embedding Comparison project.*
*Team: Shahriar Pias, Navid Ibrahim, Musaddiq Rafi*
*Supervisor: [Supervisor Name]*
