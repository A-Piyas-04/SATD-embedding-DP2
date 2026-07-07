# =============================================================================
# SATD CLASSIFICATION PROJECT — FULL PIPELINE OVERVIEW
# =============================================================================
# Goal:
#   Test whether two modern text embedding models (Qwen3 and E5-Large) can
#   classify "Self-Admitted Technical Debt" (SATD) comments better than the
#   method used in the original replication study (GloVe+BiLSTM / BERT).
#
# We compare results across 4 artifact types:
#   - Code Comments
#   - Issues
#   - Commit Messages
#   - Pull Requests
#
# Label scheme (Option A — matches the replication study exactly):
#   - Not-SATD  : not technical debt
#   - C/D       : code or design debt (merged from CODE, DESIGN, CODE/DESIGN)
#   - REQ       : requirement debt
#   - TES       : test debt
#   - DOC       : documentation debt
#   (Defect, Architecture, Build debt are dropped — not used in replication)
# =============================================================================


# -----------------------------------------------------------------------------
# PHASE 1 — CLEAN THE DATA  ✓ DONE
# -----------------------------------------------------------------------------
# What we did:
#   1. Loaded all 31 original source files and tagged each row with its
#      artifact type (code_comment / issue / commit / pull_request) based
#      on the filename.
#   2. Standardised column names across files (different files used different
#      names for the text and label columns).
#   3. Dropped rows labeled as Defect, Architecture, or Build debt — these
#      were not used in the replication study we are comparing against.
#   4. Merged CODE, DESIGN, and CODE/DESIGN labels into one "C/D" bucket.
#   5. Dropped rows where the text is blank or too short (2 words or fewer).
#   6. Removed duplicate rows (same text + same artifact type).
#   7. Saved the cleaned file as the single starting point for all later steps.
#
# Final row counts after cleaning:
#   code_comment   353,009 rows  (C/D: 11,600 | REQ: 5,406 | TES: 1,786 | DOC: 1,410 | Not-SATD: 332,753)
#   issue           23,128 rows  (C/D:  2,189 | REQ:   814 | TES:   965 | DOC: 1,081 | Not-SATD:  18,079)
#   commit           5,616 rows  (C/D:    487 | REQ:   390 | TES:   372 | DOC:   392 | Not-SATD:   3,975)
#   pull_request     4,893 rows  (C/D:    502 | REQ:   155 | TES:   247 | DOC:   380 | Not-SATD:   3,609)
#
# Output file: satd_with_artifact_type.csv
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# PHASE 2 — BALANCE THE DATA
# -----------------------------------------------------------------------------
# Why we need this:
#   Some categories have far more examples than others. For example, Not-SATD
#   has hundreds of thousands of rows while some SATD subtypes have only ~155.
#   A model trained on imbalanced data just learns to always predict Not-SATD.
#
# How we fix it (per artifact type):
#   - Find the largest SATD class → use its size as the TARGET for all classes.
#   - Not-SATD      : randomly downsample to the target size.
#   - SATD classes below target : use T5-based paraphrase model
#     (humarin/chatgpt_paraphraser_on_T5_base) running on Kaggle GPU to
#     generate reworded versions of existing examples until each class
#     reaches the target size.
#   - Special case — pull_request REQ (only 155 examples): cap augmentation
#     at 3x the original count (~465) to avoid repetitive low-quality text.
#
# Note: Qwen3 is reserved for Phase 4 (embeddings) where it is more suitable.
#
# Output file: satd_balanced.csv
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# PHASE 3 — SPLIT THE DATA
# -----------------------------------------------------------------------------
# We divide each artifact type into three groups:
#   - 80% Training set   : used to teach the model
#   - 10% Validation set : used to tune the model during development
#   - 10% Test set       : used only at the end to measure real performance
#
# Important: we use the same split ratio as the original replication study
# so our results are directly comparable to their numbers.
# The exact split is saved so every model we test uses identical test data.
#
# Output files: satd_train.csv, satd_val.csv, satd_test.csv
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# PHASE 4 — CONVERT TEXT TO NUMBERS (EMBEDDINGS)
# -----------------------------------------------------------------------------
# Machine learning models cannot read raw text — they need numbers.
# We run all text through two different embedding models, both run
# entirely on Kaggle (GPU T4 x2, no local computation needed):
#
#   - Qwen3-Embedding   : a newer multilingual embedding model
#                         (e.g. Qwen/Qwen3-Embedding-0.6B from Hugging Face)
#   - E5-Large          : a strong general-purpose embedding model
#                         (intfloat/e5-large-v2 from Hugging Face)
#
# How it works technically:
#   1. Load satd_train.csv, satd_val.csv, satd_test.csv (from Phase 3).
#   2. Load each embedding model + tokenizer via `transformers`.
#   3. Process text in batches (batch_size ~32-64) to fit in GPU memory.
#   4. For E5 models: prefix text with "query: " or "passage: " as required
#      by the model's training format (E5 is prefix-sensitive).
#   5. Mean-pool or use the model's pooling layer to get one fixed-length
#      vector per text (commonly 768-1024 dimensions depending on model).
#   6. L2-normalize embeddings if required by the model card.
#   7. Stack all vectors into a single NumPy array per split, aligned with
#      the row order of the corresponding CSV (so labels stay matched).
#   8. Save each array as .npy for fast loading in Phase 5.
#
# This step is repeated independently for each artifact type, OR all
# artifact types can be embedded together and filtered later — we will
# embed all text in one pass and keep the artifact_type column for
# filtering during training (simpler, avoids redundant model loading).
#
# HF_TOKEN should be set as a Kaggle secret before this phase to avoid
# rate limits when downloading the larger embedding models.
#
# Output files (saved to /kaggle/working/):
#   qwen3_train.npy   qwen3_val.npy   qwen3_test.npy
#   e5_train.npy      e5_val.npy      e5_test.npy
#   (plus train_labels.csv / val_labels.csv / test_labels.csv carried over
#    from Phase 3 — embeddings and labels are matched by row order)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# PHASE 5 — TRAIN THE CLASSIFIERS
# -----------------------------------------------------------------------------
# We train two simple classifier models on top of each set of embeddings,
# all done on Kaggle (CPU is sufficient for this phase — these are
# lightweight classifiers compared to the embedding models):
#
#   - XGBoost             : a tree-based gradient boosting model, good with
#                            structured/numeric input like embeddings
#   - Logistic Regression : a simple, fast linear baseline classifier
#
# This gives us 4 combinations in total:
#   1. Qwen3    embeddings + XGBoost
#   2. Qwen3    embeddings + Logistic Regression
#   3. E5-Large embeddings + XGBoost
#   4. E5-Large embeddings + Logistic Regression
#
# How it works technically:
#   1. Load the .npy embedding arrays + matching label CSVs from Phase 4.
#   2. Filter by artifact_type (code_comment / issue / commit / pull_request)
#      so each model is trained separately per artifact type.
#   3. Fit XGBoost (multi-class objective, e.g. "multi:softprob") and
#      Logistic Regression (multi_class="multinomial" or one-vs-rest)
#      on the training embeddings + labels.
#   4. Use the validation set to tune key hyperparameters:
#        - XGBoost: max_depth, n_estimators, learning_rate
#        - LogisticRegression: C (regularization strength)
#   5. Save each trained model (e.g. via joblib/pickle) for reuse in Phase 6.
#
# Total models trained = 4 combinations × 4 artifact types = 16 models.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# PHASE 6 — COMPARE RESULTS
# -----------------------------------------------------------------------------
# We evaluate all 4 combinations on the test set and compare them against
# the original replication study's numbers (GloVe+BiLSTM and BERT).
#
# Metrics used (same as the replication study):
#   - Precision, Recall, F1-score per class
#   - Macro-averaged F1 overall
#
# Results are broken down by artifact type so we can see where each
# embedding model performs better or worse than the original method.
# -----------------------------------------------------------------------------