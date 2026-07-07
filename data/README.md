# Data

## Large files (Google Drive)

All processed data files, embeddings, and trained models are hosted on Google Drive
due to file size limits. Raw source files are also included there.

**[SATD Project — Data and Models](PASTE_YOUR_DRIVE_LINK_HERE)**

### Folder structure on Drive

```
SATD Project - Data & Models/
│
├── 01_raw_sources/
│   └── satd_sources.zip           31 original CSV/XLSX files (pre-cleaning)
│
├── 02_processed_data/
│   ├── satd_with_artifact_type.csv   Phase 1 output — cleaned, artifact-tagged
│   ├── satd_balanced.csv             Phase 2 output — class-balanced
│   ├── satd_train.csv                Phase 3 output — 80% training split
│   ├── satd_val.csv                  Phase 3 output — 10% validation split
│   ├── satd_test.csv                 Phase 3 output — 10% test split
│   ├── train_labels.csv              label file aligned to train split
│   ├── val_labels.csv                label file aligned to val split
│   └── test_labels.csv               label file aligned to test split
│
├── 03_embeddings/
│   ├── qwen3_train.npy               Qwen3-Embedding-0.6B vectors (59105 x 1024)
│   ├── qwen3_val.npy                 Qwen3-Embedding-0.6B vectors  (7390 x 1024)
│   ├── qwen3_test.npy                Qwen3-Embedding-0.6B vectors  (7395 x 1024)
│   ├── e5_train.npy                  E5-Large-v2 vectors           (59105 x 1024)
│   ├── e5_val.npy                    E5-Large-v2 vectors            (7390 x 1024)
│   └── e5_test.npy                   E5-Large-v2 vectors            (7395 x 1024)
│
└── 04_trained_models/
    └── models/                       32 .joblib classifier files
        (naming: {task}_{artifact}_{embedding}_{classifier}.joblib)
        e.g. identification_code_comment_qwen3_xgboost.joblib
```

---

## Keyword files (in this repo, `data/keywords/`)

These 8 files contain scored SATD keyword lists extracted using KeyBERT from the
 replication study. They are stored in the repo (small text files)\
 (seprated in 2 folders `SATD Keyowrds for different types of SATD/` and `SATD Keywords for different sources/`),
are ready to use for **future Phase 8**.

| File | Content |
|---|---|
| `Keywords_for_code_comments.txt` | Keywords for code comment artifact type |
| `Keywords_for_issues.txt` | Keywords for issue tracker artifact type |
| `Keywords_for_commit_messages.txt` | Keywords for commit message artifact type |
| `Keywords_for_pull_requests.txt` | Keywords for pull request artifact type |
| `Keywords_for_code_or_design_debt.txt` | Keywords for C/D debt category |
| `Keywords_for_documentation_debt.txt` | Keywords for DOC debt category |
| `Keywords_for_test_debt.txt` | Keywords for TES debt category |
| `Keywords_for_requirement_debt.txt` | Keywords for REQ debt category |

Each file lists keyword phrases with importance scores in descending order:
```
0.092166 -> ('hack',)
0.045971 -> ('workaround',)
0.026151 -> ('defer', 'argument', 'checking')
```

See `docs/pipeline_overview.md` (Future Phase 8 section) for the planned
feature extraction approach using these files.