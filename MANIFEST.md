# Reproducibility Manifest

| Path | Purpose |
| --- | --- |
| `.env.example` | Credential/configuration template with placeholders only. |
| `requirements.txt` | Python dependencies for experiment and analysis scripts. |
| `main.py`, `run.py` | Experiment entry points. |
| `utils/` | Core configuration, data loading, LLM calling, retrieval, perturbation, experiment, results, and Tier 2 selection code. |
| `data/` | Prepared input cohorts, retrieval bank, breast-cancer extension data, and pancreatic labels. |
| `preprocessing/` | Explicit preprocessing/validation script and documentation. |
| `prompts/` | Visible prompt templates and output schema. |
| `perturbations/` | Visible perturbation taxonomy. |
| `scripts/` | Publication table/figure generation and Tier 2 visualization code. |
| `analysis/` | Auxiliary analyses for theorem threshold checks, sample validation, retrieval-k ablation, local-stability measurement, and placeholder table filling. |
| `outputs/` | Saved Tier 1 and Tier 2 JSON outputs, summaries, and comprehensive-analysis artifacts. |
| `paper/prompts.tex`, `paper/figures/` | Prompt appendix and generated manuscript figures. |