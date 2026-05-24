# Reproducibility Map

## Data

Primary experiment inputs live in `data/`:

- `test_ready.csv`: 63-patient pancreatic evaluation cohort.
- `actual.csv`: labels for the pancreatic evaluation cohort.
- `context_ready.csv`: labelled retrieval/context bank.
- `breast_cancer_ready.csv`: Wisconsin breast cancer dataset prepared for Tier 3.
- `data_train_ready.csv`, `data_predict_ready.csv`: retained legacy/intermediate CSVs for provenance.

## Preprocessing

- `preprocessing/prepare_inputs.py` validates required columns and row counts.
- `utils/data.py` applies the runtime numeric cleaning used whenever CSV files are loaded.

## Prompting

- `prompts/system_instruction.txt` contains the exact system instruction used by `utils/llm.py`.
- `prompts/prompt_templates.md` describes canonical, retrieval, perturbation, and stress-test prompt templates.
- `prompts/prompt_templates.json` gives the same information in machine-readable form.
- `paper/prompts.tex` is the paper-facing prompt appendix.

## Perturbations

- `utils/perturbations.py` contains the executable perturbation functions.
- `perturbations/perturbation_definitions.json` is the paper-facing taxonomy.
- `perturbations/README.md` summarizes the distinction between information-preserving perturbations and information-altering stress tests.

## Experiments

- `main.py` runs Tier 1, Tier 2, and Tier 3 experiments.
- `run.py` is a helper wrapper retained from the working project.
- `utils/experiment.py` orchestrates patients, families, prompt variants, and stochastic replicates.

## Analysis

- `scripts/comprehensive_analysis.py` generates main publication tables and figures.
- `scripts/tier2_visualization.py` explains Tier 2 patient selection.
- `new/retrieval_k_ablation.py` evaluates retrieval context depth.
- `new/measure_theorem_threshold.py` measures perturbation-distance threshold behavior.
- `new/measure_local_stability_bound.py` measures empirical local-stability/Lipschitz quantities.
- `new/one_patient_threshold_search.py` and `new/validate_threshold_sample.py` run controlled stress tests.
- `new/fill_placeholders.py` generates placeholder replacement tables for draft material.

## Saved Results

- `outputs/` contains the saved output JSON files used by `scripts/comprehensive_analysis.py`.
- `outputs_correct/` is retained because some auxiliary scripts reference that legacy layout directly.
