# Data

## Source files
The 31 original raw CSV/XLSX source files are available on Google Drive:
[SATD Project — Data & Models](PASTE_YOUR_DRIVE_LINK_HERE)

## Processed files (also on Google Drive)
| File | Description |
|---|---|
| `satd_with_artifact_type.csv` | Phase 1 output — cleaned, labeled with artifact type |
| `satd_balanced.csv` | Phase 2 output — class-balanced |
| `satd_train.csv` | Phase 3 output — 80% training split |
| `satd_val.csv` | Phase 3 output — 10% validation split |
| `satd_test.csv` | Phase 3 output — 10% test split |
| `train_labels.csv`, `val_labels.csv`, `test_labels.csv` | Label files aligned to splits |
| `qwen3_train/val/test.npy` | Phase 4 output — Qwen3 embeddings |
| `e5_train/val/test.npy` | Phase 4 output — E5-Large embeddings |

## Keyword files
The `keywords/` folder contains 8 scored keyword lists\
(seprated in 2 folders `SATD Keyowrds for different types of SATD/` and `SATD Keywords for different sources/`),
- `Keywords_for_code_comments.txt`
- `Keywords_for_issues.txt`
- `Keywords_for_commit_messages.txt`
- `Keywords_for_pull_requests.txt`
- `Keywords_for_code_or_design_debt.txt`
- `Keywords_for_documentation_debt.txt`
- `Keywords_for_test_debt.txt`
- `Keywords_for_requirement_debt.txt`\
{used in Phase 8 to build SATD signal features}