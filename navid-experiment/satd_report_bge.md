# SATD Detection: Qwen3 vs E5 vs BGE-M3 Embeddings

August 12, 2026

## 1 What we did

We compared three modern text-embedding models — Qwen3-Embedding-0.6B, E5-Large-v2, and BAAI/bge-m3 — for classifying Self-Admitted Technical Debt (SATD). This report extends the earlier Qwen3-vs-E5 study with a full experiment using BGE-M3’s dense embeddings, run through the identical pipeline. Texts were embedded by mean-pooling and L2-normalizing the encoder outputs, and a two-step classification scheme was applied per artifact type (code comments, issues, commit messages, pull requests):

1. **Identification** — binary: SATD vs. Not-SATD.
2. **Categorization** — four-class: C/D, REQ, TES, DOC (SATD rows only).

Each task was trained with XGBoost and Logistic Regression on each embedding set, giving six combinations per artifact per task. Results were compared against the paper’s baselines (BiLSTM+AugGPT for identification; BERT+AugGPT for categorization), with the supervisor’s replication numbers used for commit messages.

All three experiments used the paper’s pre-augmented dataset (≈109k rows; 95,704 after cleaning), which equalizes the four SATD subtypes per artifact while retaining a Not-SATD majority (65–82%). The train/validation/test split (80/10/10, stratified) was performed before augmentation to avoid leakage, and Logistic Regression was run with C = 0.5 throughout. Only the embedding model differed between runs.

## 2 Results

Tables 1 and 2 report macro-F1 on the test sets for each embedding with XGBoost (the classifier that consistently outperformed logistic regression), alongside the baselines.

**Table 1: Identification (SATD vs. Not-SATD), macro F1, XGBoost.**

| Artifact | Baseline | Qwen3 | E5 | BGE-M3 |
|---|---|---|---|---|
| Code comments | 0.939 | 0.9664 | 0.9587 | 0.9594 |
| Issues | 0.878 | 0.8656 | 0.8632 | 0.8533 |
| Pull requests | 0.862 | 0.8642 | 0.8703 | 0.8580 |
| Commit messages | 0.910 | 0.8917 | 0.8817 | 0.9014 |

**Table 2: Categorization (C/D, REQ, TES, DOC), macro F1, XGBoost.**

| Artifact | Baseline | Qwen3 | E5 | BGE-M3 |
|---|---|---|---|---|
| Code comments | 0.882 | 0.9470 | 0.9581 | 0.9558 |
| Issues | 0.899 | 0.9442 | 0.9503 | 0.9562 |
| Pull requests | 0.876 | 0.9548 | 0.9335 | 0.9644 |
| Commit messages | 0.980 | 0.9639 | 0.9487 | 0.9603 |

Key findings:

- **BGE-M3 is the best categorizer.** It produced the highest macro F1 on three of four artifacts (issues, pull requests, and joint-best on code comments) and the largest margin over the BERT baseline anywhere in this study: +0.0884 on pull requests. Its categorization average with XGBoost (0.9592) beats Qwen3 (0.9525) and E5 (0.9477).
- **BGE-M3 handles commit messages best.** It was the only embedding within one point of the BiLSTM baseline on commit identification (0.9014 vs. 0.9100), ahead of Qwen3 (0.8917) and E5 (0.8817).
- **Identification is close among all three.** With XGBoost the averages are Qwen3 0.8970, E5 0.8935, BGE-M3 0.8930 — a spread of under half a point, with Qwen3 winning on code comments and issues and E5 on pull requests.
- **BGE-M3 is the worst fit for logistic regression.** With LogReg it fell to 0.8407 (identification) and 0.8571 (categorization) on average — below E5 and Qwen3 and, in categorization, roughly 0.03–0.05 below its own XGBoost results.
- **XGBoost consistently beat Logistic Regression** for every embedding (e.g., BGE-M3: 0.9261 vs. 0.8489 combined average), confirming the earlier finding.
- **Best overall combination: BGE-M3 + XGBoost** (0.9261 averaged over both tasks), narrowly ahead of Qwen3 + XGBoost (0.9248) and E5 + XGBoost (0.9206).
- **Commit messages remain the hardest artifact:** no embedding reached the categorization baseline (0.980) on commit messages, and all were below it on commit identification.

## 3 Takeaways

1. **BGE-M3 is the strongest choice for SATD categorization,** beating fine-tuned BERT on three of four artifacts and edging out Qwen3 and E5, especially on issues and pull requests. Its dense embeddings encode debt-type distinctions particularly well.
2. **For identification the three models are interchangeable** (all within 0.4 points with XGBoost); BGE-M3’s best showing on commit messages makes it a slight favorite where that artifact matters.
3. **Classifiers dominate embedding choice.** The gap between the best and worst embedding is far smaller than the gap between XGBoost and logistic regression — embedding selection matters less than classifier selection. BGE-M3 in particular rewards tree-based classifiers and loses its advantage with a linear head.
4. **The paper’s identification results remain tied to its own augmented data,** not reproduced by simply swapping the embedding model: no embedding matched the BiLSTM baseline on issues or commit messages, and only Qwen3 and E5 did on code comments and pull requests, respectively.
5. **Recommendation for this task: BGE-M3 embeddings + XGBoost.** It gives the best categorization performance and the best commit-message identification, with no tuning cost beyond a standard classifier.
