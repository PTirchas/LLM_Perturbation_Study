# Perturbation Stability LLM Medical Prediction

This repository contains the reproducibility package accompanying the paper:

> **Evaluating Prompt-Induced Diagnostic Instability of Large Language Models in Clinical Decision Support**
>
> Panagiotis Tirchas, Nicholas Christakis, Dimitris Drikakis
>
> Institute for Advanced Modelling and Simulation, University of Nicosia,
> Nicosia CY-2417, Cyprus
>
> Prepared for submission to *Artificial Intelligence in Medicine* (Elsevier)

---

## Overview

This study evaluates whether large language model (LLM) predictions in
clinical decision-support workflows remain stable when the same patient facts
are rendered in semantically equivalent prompt forms. The central requirement is
simple: if two prompts describe the same biomarker record, changes in feature
order, instruction wording, structural format, or numeric presentation should
not change the diagnosis, confidence, or explanation in a clinically meaningful
way.

The repository implements a perturbation-stability framework for black-box LLM
pipelines. Structured patient biomarker records are converted into canonical
prompts, optionally augmented with similar labelled patient cases through a
retrieval module, and then re-rendered through controlled perturbation
families. The package includes the prepared data, prompt templates,
perturbation definitions, experiment runner, saved LLM outputs, and analysis
scripts needed to reproduce the reported instability metrics.

**Important:** The LLM is not fine-tuned or retrained on the pancreatic cancer
dataset. The context bank is used only as a retrieval lookup table for
context-augmented prompts. Ground-truth labels for target evaluation patients
are withheld from the model and used only for scoring.

---

## Repository Structure

```text
.
|-- LICENSE                         # GNU GPL v3 license
|-- README.md                       # Project description and reproduction guide
|-- MANIFEST.md                     # Inclusion rationale for the package
|-- requirements.txt                # Python dependencies
|-- .env.example                    # LLM provider configuration template
|-- main.py                         # Main Tier 1, Tier 2, and Tier 3 runner
|-- run.py                          # Helper wrapper for repeated runs
|-- data/                           # Prepared cohorts and labels
|-- preprocessing/                  # Data validation and cleaning script
|-- prompts/                        # System instruction and prompt templates
|-- perturbations/                  # Perturbation taxonomy in JSON form
|-- utils/                          # Core pipeline, retrieval, LLM, and metrics code
|-- outputs/                        # Saved LLM JSON outputs used by analysis scripts
|-- scripts/                        # Publication table and figure generation
|-- analysis/                       # Auxiliary validation and theorem-support analyses
`-- docs/                           # Reproducibility mapping notes
```

---

## Dataset

The main experiments use the public pancreatic cancer biomarker dataset from
Debernardi et al. The model receives structured clinical variables and predicts
a binary disease outcome, where `0` denotes no disease and `1` denotes disease.

The pancreatic prompts use seven patient features:

| Feature | Description |
|---------|-------------|
| `age` | Patient age |
| `plasma_CA19_9` | Plasma CA19-9 biomarker |
| `creatinine` | Urinary normalisation variable |
| `LYVE1` | Urinary biomarker |
| `REG1B` | Urinary biomarker |
| `TFF1` | Urinary biomarker |
| `REG1A` | Urinary biomarker |

Packaged data files:

| File | Rows | Role |
|------|-----:|------|
| `data/test_ready.csv` | 63 | Pancreatic evaluation cohort; labels are withheld from prompts. |
| `data/actual.csv` | 63 | Ground-truth labels for the evaluation cohort. |
| `data/context_ready.csv` | 146 | Labelled retrieval/context bank for similar-patient prompts. |
| `data/breast_cancer_data_full.csv` | 569 | Wisconsin breast cancer extension cohort retained for cross-dataset work. |

The evaluation cohort and context bank are patient-disjoint. In
context-augmented prompts, neighbour labels are shown only for retrieved
context cases; the target patient's label is never shown to the model.

---

## Experimental Conditions

The saved outputs cover the main pancreatic study under two context conditions.

### Without Context

The LLM receives only:

1. The common system instruction.
2. The target patient's biomarker record.

No retrieved cases or neighbour outcomes are appended to the prompt.

### With Context

The retrieval module is active. Similar patients are selected from
`data/context_ready.csv` using nearest-neighbour search in normalised biomarker
feature space. The main perturbation experiments use `k = 3` retrieved cases,
chosen as the efficiency elbow in the context-depth ablation reported in the
paper.

The LLM receives:

1. The common system instruction.
2. Labelled similar-patient cases from the context bank.
3. The target patient's biomarker record.

---

## Experimental Tiers

| Tier | Cohort | Purpose |
|------|--------|---------|
| Tier 1 | 63 pancreatic evaluation patients | Broad perturbation-stability benchmark across all evaluation patients. |
| Tier 2 | 20 boundary-case pancreatic patients | Denser perturbation sweep on patients selected for high instability or boundary uncertainty. |
| Tier 3 | Cross-dataset extension | Optional extension code for a secondary breast cancer cohort. |

Saved JSON outputs are provided for Tier 1 and Tier 2 with and without
retrieval context:

| File | Role |
|------|------|
| `outputs/output_tier_1_without_context_0.json` | Tier 1 saved outputs without retrieval context. |
| `outputs/output_tier_1_with_context_0.json` | Tier 1 saved outputs with retrieval context. |
| `outputs/output_tier_2_without_context_0.json` | Tier 2 boundary-case outputs without retrieval context. |
| `outputs/output_tier_2_with_context_0.json` | Tier 2 boundary-case outputs with retrieval context. |

Each JSON file contains the run configuration, patient-level outputs,
canonical predictions, perturbation outputs, and summary metrics.

---

## Perturbation Families

The study separates information-preserving perturbations from
information-altering stress tests. The main paper-facing outputs focus on
families A through D.

| Family | Type | Description |
|--------|------|-------------|
| `A_order` | Information-preserving | Randomly permutes the order of feature-value pairs. |
| `B_instruction` | Information-preserving | Prepends semantically equivalent clinical instruction paraphrases. |
| `C_json` | Information-preserving | Renders the same patient facts as JSON. |
| `D_numeric_precision` | Near-equivalent stress perturbation | Changes numeric display precision or presentation. |
| `Composite` | Information-preserving stress | Combines order, delimiter, and instruction changes. |
| `F_retrieval` | Information-preserving context perturbation | Varies retrieved support cases by context depth and diversity. |
| `G_feature_mask` to `J_missingness` | Information-altering | Deliberately degrades patient evidence for stress testing. |

Executable perturbations live in `utils/perturbations.py`. The paper-facing
taxonomy is in `perturbations/perturbation_definitions.json`.

---

## Model Output Schema

All model calls use the system instruction in
`prompts/system_instruction.txt`. The expected response is a JSON object:

```json
{
  "prediction": 0,
  "confidence": 0.75,
  "explanation": "brief clinical rationale",
  "decision": "predict"
}
```

The parser converts confidence to a 0-100 percentage scale for analysis.
Outputs with confidence below 50 percent are treated as `defer`.

---

## Metrics

The analysis quantifies instability across multiple output channels:

| Metric | Definition |
|--------|------------|
| Label flip rate | Fraction of perturbed calls whose binary prediction differs from the matched canonical prediction. |
| Confidence instability | Mean absolute confidence difference relative to the matched canonical output. |
| Decision instability | Frequency with which the `predict`/`defer` decision changes. |
| Semantic drift | Embedding distance between canonical and perturbed explanation texts. |
| Brier score | Mean squared error of model-implied positive-class probabilities. |
| AUROC | Discrimination of positive versus negative patient labels. |
| Risk-coverage / AURC | Selective-deferral performance when unstable cases are deferred first. |

The manuscript also reports context-depth ablation, empirical local-stability
Lipschitz summaries, threshold stress tests, semantic-entropy analysis, and
variance attribution.

---

## Quick Start

Create an environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` with the relevant LLM provider credentials before making new API
calls. Do not commit `.env`.

Run a no-cost data validation check:

```bash
python preprocessing\prepare_inputs.py --check-only
```

Run the default safety-capped smoke test:

```bash
python main.py
```

Run the compact Tier 1 study after explicitly acknowledging API cost:

```bash
python main.py --tier 1 --compact --run-full
```

Run Tier 2 boundary analysis after Tier 1 outputs exist:

```bash
python main.py --tier 2 --run-full
```

Regenerate publication tables and figures from saved outputs:

```bash
python scripts\comprehensive_analysis.py
```

Generate Tier 2 selection visualisations:

```bash
python scripts\tier2_visualization.py
```

---

## Reproducing the Paper Results

The saved JSON files in `outputs/` allow the main analysis to be regenerated
without new LLM calls. This is the recommended path for checking tables and
figures without incurring API cost.

1. Validate packaged inputs:

   ```bash
   python preprocessing\prepare_inputs.py --check-only
   ```

2. Regenerate the comprehensive analysis report:

   ```bash
   python scripts\comprehensive_analysis.py
   ```

3. Run auxiliary analyses as needed:

   ```bash
   python analysis\retrieval_k_ablation.py
   python analysis\measure_theorem_threshold.py
   python analysis\measure_local_stability_bound.py
   python analysis\one_patient_threshold_search.py
   python analysis\validate_threshold_sample.py
   ```

Some auxiliary scripts support a dry-run/report mode by default and require
`--run` before making new LLM calls.

---

## Paper Snippet Mapping

The phrase "preprocessing scripts, prompt templates, perturbation definitions
and analysis code" maps to:

| Paper phrase | Repository paths |
|--------------|------------------|
| Preprocessing scripts | `preprocessing/prepare_inputs.py`, `utils/data.py` |
| Prompt templates | `prompts/system_instruction.txt`, `prompts/prompt_templates.md`, `prompts/prompt_templates.json` |
| Perturbation definitions | `perturbations/perturbation_definitions.json`, `utils/perturbations.py` |
| Experiment runner | `main.py`, `run.py`, `utils/experiment.py` |
| Retrieval module | `utils/retrieval.py` |
| Results and metrics | `utils/results.py`, `scripts/comprehensive_analysis.py` |
| Auxiliary validation analyses | `analysis/` |
| Input data | `data/` |
| Saved outputs | `outputs/` |

See `MANIFEST.md` and `docs/REPRODUCIBILITY_MAP.md` for additional mapping
notes.

---

## Citation

If you use these materials, please cite:

```text
Panagiotis Tirchas, Nicholas Christakis, Dimitris Drikakis.
Evaluating Prompt-Induced Diagnostic Instability of Large Language Models
in Clinical Decision Support.
Prepared for submission to Artificial Intelligence in Medicine.
Citation details to be updated upon publication.
```

---

## License

This project is released under the **GNU General Public License v3.0 or later**.
See `LICENSE` for the full license text.
