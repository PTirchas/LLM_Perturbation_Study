# Data Files

| File | Rows | Role |
| --- | ---: | --- |
| `test_ready.csv` | 63 | Main pancreatic evaluation cohort. Labels are stored separately in `actual.csv`. |
| `actual.csv` | 63 | Ground-truth labels for `test_ready.csv`. |
| `context_ready.csv` | 146 | Labelled pancreatic retrieval/context bank. |
| `breast_cancer_data_full.csv` | 569 | Wisconsin breast cancer extension cohort for Tier 3. |

The experiment runner uses `test_ready.csv`, `actual.csv`,
`context_ready.csv`, and `breast_cancer_data_full.csv` directly.
