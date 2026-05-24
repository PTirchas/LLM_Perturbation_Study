# Saved Outputs

This folder contains saved LLM output JSON files so the analysis scripts can
regenerate tables and figures without making new API calls.

| File | Role |
| --- | --- |
| `output_tier_1_without_context_0.json` | Tier 1 compact run without retrieval context. |
| `output_tier_1_with_context_0.json` | Tier 1 compact run with retrieval context. |
| `output_tier_2_without_context_0.json` | Tier 2 boundary run without retrieval context. |
| `output_tier_2_with_context_0.json` | Tier 2 boundary run with retrieval context. |
| `comprehensive_analysis/` | Generated report, LaTeX tables, and figure files. |

`outputs_correct/` is also included at the package root for legacy auxiliary
scripts that reference that folder name directly.
