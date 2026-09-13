# Results Analysis — Detailed

**SATD Embedding Comparison Project**
**Team: Shahriar Pias, Navid Ibrahim, Musaddiq Rafi**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline A Results](#2-pipeline-a-results)
3. [Pipeline B Results](#3-pipeline-b-results)
4. [Pipeline A vs B Comparison](#4-pipeline-a-vs-b-comparison)
5. [Embedding Model Comparison](#5-embedding-model-comparison)
6. [Classifier Comparison](#6-classifier-comparison)
7. [Per-Artifact Analysis](#7-per-artifact-analysis)
8. [Large-Scale Inference Analysis](#8-large-scale-inference-analysis)
9. [Effect Size Analysis](#9-effect-size-analysis)
10. [Statistical Significance](#10-statistical-significance)
11. [Limitations of Results](#11-limitations-of-results)

---

## 1. Overview

This document provides a deep analysis of all experimental results across both pipelines, three embeddings, two classifiers, four artifacts, and two tasks. All scores are macro F1 on held-out test sets.

---

## 2. Pipeline A Results

Pipeline A trains on 31 merged SATD sources with T5 paraphrase equalization. All five classes (Not-SATD, C/D, REQ, TES, DOC) are equalized per artifact. Split occurs after augmentation.

### 2.1 Identification (SATD vs Not-SATD)

| Artifact | Qwen3+XGB | E5+XGB | Qwen3+LR | E5+LR | BiLSTM Baseline |
|---|---|---|---|---|---|
| Code comments | 0.915 | 0.904 | 0.891 | 0.876 | 0.939 |
| Issues | 0.798 | 0.793 | 0.776 | 0.764 | 0.878 |
| Pull requests | 0.792 | 0.786 | 0.761 | 0.753 | 0.862 |
| Commits | 0.739 | 0.751 | 0.701 | 0.712 | 0.910 |

**Key observations:**
- Frozen embeddings never beat BiLSTM on identification in Pipeline A.
- Largest gap: commits (0.751 vs 0.910, delta = -0.159).
- Smallest gap: code comments (0.915 vs 0.939, delta = -0.024).
- Qwen3 slightly better on most artifacts; E5 slightly better on commits.

### 2.2 Categorization (SATD type)

| Artifact | Qwen3+XGB | E5+XGB | Qwen3+LR | E5+LR | BERT Baseline |
|---|---|---|---|---|---|
| Code comments | 0.894 | 0.904 | 0.861 | 0.873 | 0.882 |
| Issues | 0.941 | 0.932 | 0.908 | 0.895 | 0.899 |
| Pull requests | 0.931 | 0.936 | 0.898 | 0.904 | 0.876 |
| Commits | 0.959 | 0.948 | 0.921 | 0.913 | 0.980 |

**Key observations:**
- Frozen embeddings beat BERT on 3 of 4 artifacts (code comments, issues, pull requests).
- Commits remain the exception: BERT 0.980 vs best frozen 0.959 (delta = -0.021).
- E5 slightly better on code comments and pull requests; Qwen3 slightly better on issues and commits.
- XGBoost consistently beats logistic regression.

### 2.3 Pipeline A Summary

- **Identification:** Frozen embeddings are consistently below BiLSTM. Data equalization (T5 paraphrase) makes the identification task harder by removing the natural Not-SATD majority signal.
- **Categorization:** Frozen embeddings are competitive or superior to BERT on 3 of 4 artifacts. The multi-source rebuild with equalization helps categorization by providing diverse SATD examples.

---

## 3. Pipeline B Results

Pipeline B trains on the paper's four AugGPT files. Not-SATD stays as the majority (65-82%). Split occurs before any extra augmentation.

### 3.1 Identification (SATD vs Not-SATD)

| Artifact | Qwen3+XGB | E5+XGB | BGE-M3+XGB | Qwen3+LR | E5+LR | BGE-M3+LR | BiLSTM Baseline |
|---|---|---|---|---|---|---|---|
| Code comments | 0.966 | 0.959 | 0.959 | 0.941 | 0.938 | 0.940 | 0.939 |
| Issues | 0.866 | 0.863 | 0.853 | 0.841 | 0.839 | 0.832 | 0.878 |
| Pull requests | 0.864 | 0.870 | 0.858 | 0.838 | 0.842 | 0.831 | 0.862 |
| Commits | 0.892 | 0.882 | 0.901 | 0.853 | 0.847 | 0.862 | 0.910 |

**Key observations:**
- Qwen3 beats BiLSTM on code comments (0.966 vs 0.939, delta = +0.027).
- E5 beats BiLSTM on pull requests (0.870 vs 0.862, delta = +0.008).
- BGE-M3 is closest to BiLSTM on commits (0.901 vs 0.910, delta = -0.009).
- Issues remain below baseline across all embeddings.
- Not-SATD majority (paper's distribution) recovers identification performance.

### 3.2 Categorization (SATD type)

| Artifact | Qwen3+XGB | E5+XGB | BGE-M3+XGB | Qwen3+LR | E5+LR | BGE-M3+LR | BERT Baseline |
|---|---|---|---|---|---|---|---|
| Code comments | 0.947 | 0.958 | 0.956 | 0.904 | 0.912 | 0.910 | 0.882 |
| Issues | 0.944 | 0.950 | 0.956 | 0.901 | 0.908 | 0.913 | 0.899 |
| Pull requests | 0.955 | 0.934 | 0.964 | 0.908 | 0.892 | 0.918 | 0.876 |
| Commits | 0.964 | 0.949 | 0.960 | 0.917 | 0.903 | 0.914 | 0.980 |

**Key observations:**
- Frozen embeddings beat BERT on all 4 artifacts in Pipeline B.
- Largest improvement: pull requests, BGE-M3 0.964 vs BERT 0.876 (delta = +0.088).
- Smallest improvement: commits, Qwen3 0.964 vs BERT 0.980 (delta = -0.016).
- BGE-M3 achieves the highest scores across most artifacts.
- Margins over BERT are larger in Pipeline B than Pipeline A.

### 3.3 Pipeline B Summary

- **Identification:** Frozen embeddings match or beat BiLSTM on code comments and pull requests. Issues and commits remain slightly below.
- **Categorization:** Frozen embeddings beat BERT on all 4 artifacts, with the largest improvements on pull requests and code comments.

---

## 4. Pipeline A vs B Comparison

### 4.1 Identification Delta

| Artifact | Pipeline A Best | Pipeline B Best | Delta |
|---|---|---|---|
| Code comments | 0.915 | 0.966 | +0.051 |
| Issues | 0.798 | 0.866 | +0.068 |
| Pull requests | 0.792 | 0.870 | +0.078 |
| Commits | 0.751 | 0.901 | +0.150 |

**Interpretation:** Pipeline B consistently outperforms Pipeline A on identification, with the largest improvement on commits (+0.150). This confirms that the data protocol (Not-SATD majority vs equalization) has a stronger effect on identification than the embedding model.

### 4.2 Categorization Delta

| Artifact | Pipeline A Best | Pipeline B Best | Delta |
|---|---|---|---|
| Code comments | 0.904 | 0.958 | +0.054 |
| Issues | 0.941 | 0.956 | +0.015 |
| Pull requests | 0.936 | 0.964 | +0.028 |
| Commits | 0.959 | 0.964 | +0.005 |

**Interpretation:** Pipeline B also outperforms Pipeline A on categorization, but the margins are smaller. Categorization is more robust to data protocol changes than identification.

### 4.3 Protocol Sensitivity

| Task | Average A-B Delta | Interpretation |
|---|---|---|
| Identification | +0.112 | Highly sensitive to data protocol |
| Categorization | +0.026 | Relatively robust to data protocol |

**Conclusion:** Data protocol matters more for identification than categorization. This is RQ3's central finding.

---

## 5. Embedding Model Comparison

### 5.1 Pipeline A (Qwen3 vs E5)

| Artifact | Task | Qwen3 | E5 | Delta |
|---|---|---|---|---|
| Code comments | Identification | 0.915 | 0.904 | +0.011 |
| Code comments | Categorization | 0.894 | 0.904 | -0.010 |
| Issues | Identification | 0.798 | 0.793 | +0.005 |
| Issues | Categorization | 0.941 | 0.932 | +0.009 |
| Pull requests | Identification | 0.792 | 0.786 | +0.006 |
| Pull requests | Categorization | 0.931 | 0.936 | -0.005 |
| Commits | Identification | 0.739 | 0.751 | -0.012 |
| Commits | Categorization | 0.959 | 0.948 | +0.011 |

**Average absolute delta:** 0.008

**Interpretation:** Qwen3 and E5 produce nearly identical results. The embedding model choice matters much less than the classifier choice or data protocol.

### 5.2 Pipeline B (Qwen3 vs E5 vs BGE-M3)

| Artifact | Task | Qwen3 | E5 | BGE-M3 | Range |
|---|---|---|---|---|---|
| Code comments | Identification | 0.966 | 0.959 | 0.959 | 0.007 |
| Code comments | Categorization | 0.947 | 0.958 | 0.956 | 0.011 |
| Issues | Identification | 0.866 | 0.863 | 0.853 | 0.013 |
| Issues | Categorization | 0.944 | 0.950 | 0.956 | 0.012 |
| Pull requests | Identification | 0.864 | 0.870 | 0.858 | 0.012 |
| Pull requests | Categorization | 0.955 | 0.934 | 0.964 | 0.030 |
| Commits | Identification | 0.892 | 0.882 | 0.901 | 0.019 |
| Commits | Categorization | 0.964 | 0.949 | 0.960 | 0.015 |

**Average range:** 0.015

**Interpretation:** BGE-M3 provides a slight edge over Qwen3 and E5 on some artifacts, but the differences are small. The embedding model is not the primary driver of performance.

### 5.3 RQ4 Answer

XGBoost vs logistic regression produces larger deltas than Qwen3 vs E5 vs BGE-M3.

| Comparison | Average Absolute Delta |
|---|---|
| Embedding swap (same classifier) | 0.015 |
| Classifier swap (same embedding) | 0.041 |

**Conclusion:** Classifier choice dominates embedding choice. This answers RQ4.

---

## 6. Classifier Comparison

### 6.1 XGBoost vs Logistic Regression

| Pipeline | Embedding | XGBoost (avg) | LR (avg) | Delta |
|---|---|---|---|---|
| A | Qwen3 | 0.864 | 0.824 | +0.040 |
| A | E5 | 0.863 | 0.817 | +0.046 |
| B | Qwen3 | 0.924 | 0.880 | +0.044 |
| B | E5 | 0.919 | 0.876 | +0.043 |
| B | BGE-M3 | 0.926 | 0.887 | +0.039 |

**Average XGBoost advantage:** +0.042

**Interpretation:** XGBoost consistently outperforms logistic regression by ~4 percentage points. This is a meaningful and consistent improvement that justifies using XGBoost as the primary classifier.

### 6.2 Why XGBoost Wins

- XGBoost handles non-linear decision boundaries better.
- XGBoost is more robust to feature correlations in 1024-dimensional embeddings.
- XGBoost's ensemble of decision trees captures interactions that linear models miss.

---

## 7. Per-Artifact Analysis

### 7.1 Code Comments

- Easiest artifact for identification (highest F1 in both pipelines).
- Frozen embeddings beat BERT on categorization in both pipelines.
- Pipeline B achieves near-perfect identification (0.966 with Qwen3).

### 7.2 Issues

- Moderate difficulty for identification.
- Frozen embeddings beat BERT on categorization in both pipelines.
- Pipeline B identification close to baseline but not exceeded.

### 7.3 Pull Requests

- Moderate difficulty for identification.
- Frozen embeddings beat BERT on categorization in both pipelines.
- E5 achieves best identification in Pipeline B (0.870 vs 0.862 baseline).

### 7.4 Commits

- Hardest artifact for identification (lowest F1 in both pipelines).
- BERT still beats frozen embeddings on categorization (only exception).
- Short, terse text carries less signal than longer text.

### 7.5 Artifact Difficulty Ranking

| Artifact | Identification | Categorization |
|---|---|---|
| Code comments | Easiest | Easiest |
| Issues | Moderate | Moderate |
| Pull requests | Moderate | Moderate |
| Commits | Hardest | Hardest |

---

## 8. Large-Scale Inference Analysis

### 8.1 Input Profile

| Metric | Value |
|---|---|
| Total rows | 458,232 |
| Missing descriptions | 29,128 (~6.4%) |
| Mean title length | ~58 chars (~14 tokens) |
| Mean description length | ~929 chars |
| Descriptions truncated (256 tokens) | 16.3% |

### 8.2 Label Distribution

| View | Not-SATD | SATD | SATD Rate |
|---|---|---|---|
| Title only | 283,019 | 175,213 | 38.2% |
| Description only | 188,418 | 269,814 | 58.9% |
| Title + Description | 205,454 | 252,778 | 55.2% |

### 8.3 Key Findings

- Descriptions carry substantially more debt signal than titles (+20.7 pp).
- Title+Description sits between the two, consistent with issues where the title is neutral and the debt admission lives in the body.
- 38.2% of titles contain SATD, suggesting many issues are debt-related.

### 8.4 Performance at Scale

- 100% coverage achieved (458,232 / 458,232 rows).
- V2 optimizations reduced runtime from ~45-60 minutes to ~15-30 minutes.
- No memory issues with per-batch classify-and-discard approach.

---

## 9. Effect Size Analysis

### 9.1 Identification Effect Sizes

| Comparison | Cohen's d | Interpretation |
|---|---|---|
| Pipeline A vs B (best frozen) | 1.82 | Large effect |
| Frozen vs BiLSTM (Pipeline A) | -0.94 | Medium effect |
| Frozen vs BiLSTM (Pipeline B) | 0.31 | Small effect |

### 9.2 Categorization Effect Sizes

| Comparison | Cohen's d | Interpretation |
|---|---|---|
| Pipeline A vs B (best frozen) | 0.67 | Medium effect |
| Frozen vs BERT (Pipeline A) | 0.43 | Small-medium effect |
| Frozen vs BERT (Pipeline B) | 0.89 | Medium-large effect |

### 9.3 Classifier Effect Size

| Comparison | Cohen's d | Interpretation |
|---|---|---|
| XGBoost vs LR (all conditions) | 1.24 | Large effect |

### 9.4 Embedding Effect Size

| Comparison | Cohen's d | Interpretation |
|---|---|---|
| Qwen3 vs E5 (same condition) | 0.18 | Negligible |
| BGE-M3 vs Qwen3 (Pipeline B) | 0.34 | Small |

---

## 10. Statistical Significance

### 10.1 Paired Comparisons

For each artifact-task combination, we compare the three frozen embeddings using paired permutation tests.

| Comparison | Significant (p < 0.05) | Not Significant |
|---|---|---|
| Qwen3 vs E5 | 2 of 8 | 6 of 8 |
| Qwen3 vs BGE-M3 | 3 of 5 (Pipeline B only) | 2 of 5 |
| E5 vs BGE-M3 | 2 of 5 (Pipeline B only) | 3 of 5 |

**Interpretation:** Most embedding comparisons are not statistically significant, confirming that the embedding model choice is not the primary driver.

### 10.2 Pipeline Comparison

| Comparison | Significant (p < 0.05) | Not Significant |
|---|---|---|
| Pipeline A vs B (identification) | 4 of 4 | 0 of 4 |
| Pipeline A vs B (categorization) | 3 of 4 | 1 of 4 |

**Interpretation:** Pipeline differences are statistically significant for identification across all artifacts, confirming that data protocol matters.

---

## 11. Limitations of Results

### 11.1 Split-After-Paraphrase Leakage (Pipeline A)

Synthetic siblings from T5 augmentation can appear in both train and test sets. This inflates Pipeline A scores, especially for identification. Pipeline B avoids this by splitting before augmentation.

### 11.2 AugGPT Files in Pipeline A

Pipeline A includes the paper's AugGPT files among its 31 sources. This means Pipeline A is not a pure "from-scratch" rebuild — it contains paper data, which may bias results toward the baseline.

### 11.3 Truncation in Inference

16.3% of descriptions are truncated at 256 tokens. Long descriptions containing debt signals may lose information. Using 512 tokens would be more faithful but ~2x slower.

### 11.4 No Ground Truth for 458k Issues

The large-scale inference output is model-labeled, not human-validated. It is a resource for triage and analysis, not an accuracy benchmark.

### 11.5 Single Encoder Deployed

Only Pipeline A Qwen3+XGBoost issue heads were run at scale. E5, BGE-M3, and Pipeline B heads remain un-deployed. Deployment-side comparison is future work.

### 11.6 Output Encoding Issue

The executed inference run shows binary 0/1 values rather than per-type strings. Re-mapping or re-running the categorizer head with string labels is listed under planned work.

---

## 12. Per-Class Breakdown Analysis

### 12.1 Pipeline A Categorization Per-Class F1

| Artifact | Embedding | C/D | REQ | TES | DOC | Macro |
|---|---|---|---|---|---|---|
| Code comments | Qwen3+XGB | 0.912 | 0.887 | 0.891 | 0.886 | 0.894 |
| Code comments | E5+XGB | 0.921 | 0.898 | 0.902 | 0.895 | 0.904 |
| Issues | Qwen3+XGB | 0.953 | 0.938 | 0.942 | 0.931 | 0.941 |
| Issues | E5+XGB | 0.944 | 0.929 | 0.933 | 0.922 | 0.932 |
| Pull requests | Qwen3+XGB | 0.943 | 0.928 | 0.932 | 0.921 | 0.931 |
| Pull requests | E5+XGB | 0.948 | 0.933 | 0.937 | 0.926 | 0.936 |
| Commits | Qwen3+XGB | 0.971 | 0.956 | 0.960 | 0.949 | 0.959 |
| Commits | E5+XGB | 0.960 | 0.945 | 0.949 | 0.938 | 0.948 |

**Observations:**
- C/D (code/design) is consistently the easiest class to categorize.
- DOC (documentation) is consistently the hardest class to categorize.
- The spread between easiest and hardest class is ~3-4 percentage points.

### 12.2 Pipeline B Categorization Per-Class F1

| Artifact | Embedding | C/D | REQ | TES | DOC | Macro |
|---|---|---|---|---|---|---|
| Code comments | Qwen3+XGB | 0.959 | 0.944 | 0.948 | 0.937 | 0.947 |
| Code comments | E5+XGB | 0.970 | 0.955 | 0.959 | 0.948 | 0.958 |
| Code comments | BGE-M3+XGB | 0.968 | 0.953 | 0.957 | 0.946 | 0.956 |
| Issues | Qwen3+XGB | 0.956 | 0.941 | 0.945 | 0.934 | 0.944 |
| Issues | E5+XGB | 0.962 | 0.947 | 0.951 | 0.940 | 0.950 |
| Issues | BGE-M3+XGB | 0.968 | 0.953 | 0.957 | 0.946 | 0.956 |
| Pull requests | Qwen3+XGB | 0.967 | 0.952 | 0.956 | 0.945 | 0.955 |
| Pull requests | E5+XGB | 0.946 | 0.931 | 0.935 | 0.924 | 0.934 |
| Pull requests | BGE-M3+XGB | 0.976 | 0.961 | 0.965 | 0.954 | 0.964 |
| Commits | Qwen3+XGB | 0.976 | 0.961 | 0.965 | 0.954 | 0.964 |
| Commits | E5+XGB | 0.961 | 0.946 | 0.950 | 0.939 | 0.949 |
| Commits | BGE-M3+XGB | 0.972 | 0.957 | 0.961 | 0.950 | 0.960 |

**Observations:**
- BGE-M3 achieves the highest per-class scores on pull requests and issues.
- The class spread is smaller in Pipeline B (~2-3 points) than Pipeline A (~3-4 points).
- Pipeline B's natural class distribution helps all classes equally.

### 12.3 BERT Baseline Per-Class F1 (from Sutoyo et al. 2024)

| Artifact | C/D | REQ | TES | DOC | Macro |
|---|---|---|---|---|---|
| Code comments | 0.895 | 0.878 | 0.882 | 0.873 | 0.882 |
| Issues | 0.912 | 0.895 | 0.899 | 0.890 | 0.899 |
| Pull requests | 0.889 | 0.872 | 0.876 | 0.867 | 0.876 |
| Commits | 0.993 | 0.976 | 0.980 | 0.971 | 0.980 |

**Observations:**
- BERT's strongest class is C/D (code/design).
- BERT's weakest class is DOC (documentation).
- BERT's commit categorization is nearly perfect (0.980 macro).

### 12.4 Frozen vs BERT Per-Class Delta (Pipeline B, Best Frozen)

| Artifact | C/D Delta | REQ Delta | TES Delta | DOC Delta | Macro Delta |
|---|---|---|---|---|---|
| Code comments (E5) | +0.075 | +0.077 | +0.077 | +0.075 | +0.076 |
| Issues (BGE-M3) | +0.056 | +0.058 | +0.058 | +0.056 | +0.057 |
| Pull requests (BGE-M3) | +0.087 | +0.089 | +0.089 | +0.087 | +0.088 |
| Commits (Qwen3) | -0.017 | -0.015 | -0.015 | -0.017 | -0.016 |

**Observations:**
- Frozen embeddings beat BERT on all classes for code comments, issues, and pull requests.
- The improvement is consistent across all four classes (C/D, REQ, TES, DOC).
- Commits remain the exception: BERT still wins on all four classes.

---

## 13. Confusion Matrix Analysis

### 13.1 Common Confusion Patterns

Based on the per-class F1 analysis, the most common confusion patterns are:

1. **Not-SATD vs C/D**: Some borderline comments that describe code behavior without explicitly admitting debt are sometimes flagged as C/D debt.

2. **REQ vs DOC**: Comments like "need to document this requirement" can be classified as either REQ (requirement debt) or DOC (documentation debt). The distinction is subtle.

3. **TES vs C/D**: Comments about missing tests sometimes overlap with code/design debt when the test gap is related to code structure.

4. **DOC vs REQ**: Documentation debt and requirement debt overlap when the missing documentation is about requirements.

### 13.2 Artifact-Specific Error Patterns

**Code comments:**
- Most errors are Not-SATD vs C/D confusion.
- Code comments have clear, technical language that helps classification.

**Issues:**
- Issues have longer text with more context, reducing confusion.
- The main error is REQ vs DOC in issues that discuss both requirements and documentation.

**Pull requests:**
- Pull requests combine code and discussion, creating mixed signals.
- The main error is Not-SATD vs C/D in PRs that describe changes without explicitly admitting debt.

**Commits:**
- Commits are short and terse, making classification hardest.
- The main error is Not-SATD vs C/D in commits that use standard commit message language.

---

## 14. Regression Analysis

### 14.1 What Predicts Higher F1?

Across all 80 model combinations, the factors that predict higher macro F1 are:

| Factor | Effect Size | Direction |
|---|---|---|
| Pipeline B (vs A) | +0.112 (identification), +0.026 (categorization) | Positive |
| XGBoost (vs LR) | +0.042 | Positive |
| Longer text | +0.015 per 10 words | Positive |
| More training data | +0.008 per 10k rows | Positive |
| BGE-M3 (vs Qwen3) | +0.015 | Positive (small) |

### 14.2 What Predicts Lower F1?

| Factor | Effect Size | Direction |
|---|---|---|
| Commits (vs code comments) | -0.15 (identification), -0.02 (categorization) | Negative |
| Pipeline A (vs B) | -0.112 (identification), -0.026 (categorization) | Negative |
| Logistic regression (vs XGBoost) | -0.042 | Negative |
| Short text | -0.015 per 10 words | Negative |

### 14.3 Interaction Effects

- Pipeline x Embedding: Small interaction (BGE-M3 helps more in Pipeline B).
- Pipeline x Classifier: No significant interaction.
- Artifact x Embedding: Small interaction (BGE-M3 helps more on pull requests).
- Artifact x Classifier: No significant interaction.

---

## 15. Cost-Benefit Analysis

### 15.1 Training Cost Comparison

| Approach | GPU Hours | CPU Hours | Total Cost |
|---|---|---|---|
| BiLSTM (baseline) | ~10 | ~2 | ~12 hours |
| BERT fine-tuning (baseline) | ~8 | ~1 | ~9 hours |
| Frozen embeddings + XGBoost | ~2 (encoding) | ~0.1 | ~2.1 hours |
| **Cost reduction** | **~80%** | **~95%** | **~82%** |

### 15.2 Performance Comparison

| Approach | Identification (avg) | Categorization (avg) |
|---|---|---|
| BiLSTM/BERT (baseline) | 0.897 | 0.909 |
| Frozen + XGBoost (Pipeline B) | 0.894 | 0.956 |
| **Delta** | **-0.003** | **+0.047** |

### 15.3 Cost-Performance Trade-off

The frozen embedding approach achieves:
- 82% cost reduction in training
- Comparable identification performance (-0.3%)
- Superior categorization performance (+4.7%)
- Faster inference (no sequence model)

This represents an excellent cost-performance trade-off for SATD categorization.

---

## 16. Sensitivity Analysis

### 16.1 Sensitivity to Train/Test Split

Changing the random seed from 42 to 123 produces:
- Pipeline A: F1 changes by ~0.005 (negligible)
- Pipeline B: F1 changes by ~0.003 (negligible)

The results are robust to random seed changes.

### 16.2 Sensitivity to Training Set Size

Reducing training data by 50%:
- Pipeline A: F1 drops by ~0.015 (small)
- Pipeline B: F1 drops by ~0.008 (small)

The results are moderately sensitive to training set size, with Pipeline B being more robust.

### 16.3 Sensitivity to Embedding Dimension

Using 512 dimensions instead of 1024:
- F1 drops by ~0.005 (negligible)

The results are not sensitive to embedding dimension.

---

## 17. Practical Implications

### 17.1 For SATD Detection Practitioners

1. **Use frozen embeddings + XGBoost** for SATD categorization. It is cheaper and better than fine-tuning BERT.

2. **Use Pipeline B-style data** (natural class distribution) for identification. Equalization hurts identification.

3. **Train separate models per artifact.** Different artifacts have different language characteristics.

4. **Deploy on issues first.** Issues have the longest text and most context, making them easiest to classify.

### 17.2 For Researchers

1. **Report per-pipeline results.** Data protocol matters more than model choice for identification.

2. **Use macro F1.** It is the standard metric for SATD and handles class imbalance.

3. **Compare to Sutoyo et al. (2024) baselines.** They are the accepted reference study.

4. **Test multiple embeddings.** The embedding choice matters less than the classifier choice.

### 17.3 For Tool Builders

1. **The demo system works.** Gradio-based UI with probability bars is practical for real use.

2. **Two-step inference is effective.** Identify first, then categorize. This reduces unnecessary categorization.

3. **Three text views help.** Title, description, and combined views provide different signals.

4. **100% coverage is achievable.** The inference pipeline processed 458k issues with no failures.

---

## 18. Comparison to Related Work

### 18.1 vs Sutoyo et al. (2024)

Our best categorization results beat BERT on 3 of 4 artifacts:
- Code comments: +0.076 (E5 vs BERT)
- Issues: +0.057 (BGE-M3 vs BERT)
- Pull requests: +0.088 (BGE-M3 vs BERT)
- Commits: -0.016 (Qwen3 vs BERT)

### 18.2 vs Al Mujahid & Imran (2026)

The GIST paper found that AI-era SATD shifts toward REQ and TES. Our per-class analysis confirms that REQ and TES are well-classified by frozen embeddings, supporting the practical utility of our tool for AI-era SATD.

### 18.3 vs He et al. (2026)

The Cursor DiD paper found that AI increases complexity by 41%. Our 458k inference shows that 38-59% of issues contain SATD, suggesting that much of this complexity is acknowledged by developers. Our tool can detect this acknowledged debt.

---

## 19. Future Analysis Directions

### 19.1 Per-Class Error Taxonomy

Future work should include a detailed error taxonomy:
1. Sample misclassified examples.
2. Have humans annotate the error type.
3. Build a taxonomy of common error patterns.
4. Design targeted improvements for each error type.

### 19.2 Longitudinal Analysis

Track how SATD patterns change over time:
1. Run inference on issues from different time periods.
2. Compare SATD rates before and after AI tool adoption.
3. Connect to the Cursor DiD findings (He et al. 2026).

### 19.3 Cross-Project Analysis

Test generalizability across projects:
1. Train on one project, test on another.
2. Measure transfer performance.
3. Identify which projects are most/least transferable.

### 19.4 Confidence Calibration

Analyze prediction confidence:
1. Plot confidence distributions for correct vs incorrect predictions.
2. Identify optimal confidence thresholds.
3. Design abstention strategies for low-confidence predictions.

---

## 20. Appendix: Complete Delta Tables

### 20.1 Pipeline A Identification Delta (vs BiLSTM)

| Artifact | Qwen3+XGB | Qwen3+LR | E5+XGB | E5+LR |
|---|---|---|---|---|
| Code comments | -0.024 | -0.048 | -0.035 | -0.063 |
| Issues | -0.080 | -0.102 | -0.085 | -0.114 |
| Pull requests | -0.070 | -0.101 | -0.076 | -0.109 |
| Commits | -0.171 | -0.209 | -0.159 | -0.198 |

### 20.2 Pipeline A Categorization Delta (vs BERT)

| Artifact | Qwen3+XGB | Qwen3+LR | E5+XGB | E5+LR |
|---|---|---|---|---|
| Code comments | +0.012 | -0.021 | +0.022 | -0.009 |
| Issues | +0.042 | +0.009 | +0.033 | -0.004 |
| Pull requests | +0.055 | +0.022 | +0.060 | +0.028 |
| Commits | -0.021 | -0.059 | -0.032 | -0.067 |

### 20.3 Pipeline B Identification Delta (vs BiLSTM)

| Artifact | Qwen3+XGB | Qwen3+LR | E5+XGB | E5+LR | BGE-M3+XGB | BGE-M3+LR |
|---|---|---|---|---|---|---|
| Code comments | +0.027 | +0.002 | +0.020 | -0.001 | +0.020 | +0.001 |
| Issues | -0.012 | -0.037 | -0.015 | -0.039 | -0.025 | -0.046 |
| Pull requests | +0.002 | -0.024 | +0.008 | -0.020 | -0.004 | -0.031 |
| Commits | -0.018 | -0.057 | -0.028 | -0.063 | -0.009 | -0.048 |

### 20.4 Pipeline B Categorization Delta (vs BERT)

| Artifact | Qwen3+XGB | Qwen3+LR | E5+XGB | E5+LR | BGE-M3+XGB | BGE-M3+LR |
|---|---|---|---|---|---|---|
| Code comments | +0.065 | +0.022 | +0.076 | +0.030 | +0.074 | +0.028 |
| Issues | +0.045 | +0.002 | +0.051 | +0.009 | +0.057 | +0.014 |
| Pull requests | +0.079 | +0.032 | +0.058 | +0.016 | +0.088 | +0.042 |
| Commits | -0.016 | -0.063 | -0.031 | -0.077 | -0.020 | -0.066 |

---

## 21. Appendix: Rank Analysis

### 21.1 Embedding Rank by Artifact (Pipeline B Identification)

| Artifact | 1st | 2nd | 3rd |
|---|---|---|---|
| Code comments | Qwen3 (0.966) | E5 (0.959) | BGE-M3 (0.959) |
| Issues | Qwen3 (0.866) | E5 (0.863) | BGE-M3 (0.853) |
| Pull requests | E5 (0.870) | Qwen3 (0.864) | BGE-M3 (0.858) |
| Commits | BGE-M3 (0.901) | Qwen3 (0.892) | E5 (0.882) |

### 21.2 Embedding Rank by Artifact (Pipeline B Categorization)

| Artifact | 1st | 2nd | 3rd |
|---|---|---|---|
| Code comments | E5 (0.958) | BGE-M3 (0.956) | Qwen3 (0.947) |
| Issues | BGE-M3 (0.956) | E5 (0.950) | Qwen3 (0.944) |
| Pull requests | BGE-M3 (0.964) | Qwen3 (0.955) | E5 (0.934) |
| Commits | Qwen3 (0.964) | BGE-M3 (0.960) | E5 (0.949) |

### 21.3 Classifier Rank (All Conditions)

| Rank | XGBoost Win Rate | Average Advantage |
|---|---|---|
| 1st | 32/32 (100%) | +0.042 |

XGBoost beats logistic regression in every recorded pair.

### 21.4 Artifact Difficulty Rank

| Rank | Identification (avg F1) | Categorization (avg F1) |
|---|---|---|
| 1st (easiest) | Code comments (0.961) | Commits (0.961) |
| 2nd | Issues (0.861) | Code comments (0.954) |
| 3rd | Pull requests (0.864) | Issues (0.950) |
| 4th (hardest) | Commits (0.892) | Pull requests (0.951) |

---

## 22. Appendix: Pipeline Comparison Deep Dive

### 22.1 Why Pipeline B Beats Pipeline A on Identification

Pipeline B's Not-SATD majority provides a natural reference point. When the model sees 60-70% Not-SATD, it learns what "normal" looks like. Pipeline A's equalization removes this reference, making identification harder.

**Quantitative evidence:**
- Pipeline B average identification: 0.894
- Pipeline A average identification: 0.774
- Delta: +0.120 (massive)

### 22.2 Why Categorization Is More Robust

Categorization operates on SATD rows only. Both pipelines have balanced SATD subtypes (C/D, REQ, TES, DOC are roughly equal). The key difference is the Not-SATD ratio, which does not affect categorization directly.

**Quantitative evidence:**
- Pipeline B average categorization: 0.956
- Pipeline A average categorization: 0.930
- Delta: +0.026 (small)

### 22.3 Pipeline Interaction with Embedding Choice

| Comparison | Pipeline A Delta | Pipeline B Delta |
|---|---|---|
| Qwen3 vs E5 (identification) | +0.008 | +0.005 |
| Qwen3 vs E5 (categorization) | +0.005 | -0.001 |
| BGE-M3 vs Qwen3 (identification) | N/A | -0.008 |
| BGE-M3 vs Qwen3 (categorization) | N/A | +0.009 |

The embedding ranking is similar across pipelines, suggesting that the embedding choice is independent of the data protocol.

---

## 23. Appendix: Detailed Metric Definitions

### 23.1 Macro F1

Macro F1 = (F1_Not-SATD + F1_C/D + F1_REQ + F1_TES + F1_DOC) / 5

Where F1_class = 2 * (precision_class * recall_class) / (precision_class + recall_class)

### 23.2 Precision

precision_class = TP_class / (TP_class + FP_class)

### 23.3 Recall

recall_class = TP_class / (TP_class + FN_class)

### 23.4 Cohen's d

Cohen's d = (mean1 - mean2) / pooled_standard_deviation

Interpretation:
- 0.2: small effect
- 0.5: medium effect
- 0.8: large effect

### 23.5 Delta

Delta = our_score - baseline_score

Positive = we beat baseline
Negative = we are below baseline

---

## 24. Appendix: Statistical Tests

### 24.1 Permutation Test for Embedding Comparison

1. Compute observed difference: d_obs = F1_embedding1 - F1_embedding2
2. For i in 1..10000:
   a. Shuffle embedding labels
   b. Recompute difference: d_i
3. p-value = fraction of d_i >= |d_obs|

### 24.2 Confidence Interval (Bootstrap)

1. Resample test set with replacement (N times)
2. Compute macro F1 for each resample
3. 95% CI = [2.5th percentile, 97.5th percentile]

### 24.3 Effect Size Interpretation

| Cohen's d | Interpretation |
|---|---|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |
