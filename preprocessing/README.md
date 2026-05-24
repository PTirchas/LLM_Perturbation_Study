# Preprocessing

The main experiment loads prepared CSV files from `data/` through
`utils.data.load_data`. That loader applies one runtime preprocessing step:
string columns that encode numeric values with period separators are converted
to floats.

`prepare_inputs.py` makes this preprocessing step explicit and checks that the
packaged files have the expected columns and row counts. By default it performs
checks only and does not overwrite data.

```bash
python preprocessing\prepare_inputs.py --check-only
```

To write normalized copies into a separate folder:

```bash
python preprocessing\prepare_inputs.py --write-cleaned --output-dir data_prepared
```

The pipeline itself uses `data/test_ready.csv`, `data/context_ready.csv`,
`data/actual.csv`, and `data/breast_cancer_ready.csv`.
