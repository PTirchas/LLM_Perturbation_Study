# Perturbation Definitions

`utils/perturbations.py` is the executable source of truth. The JSON file in
this folder exposes the same taxonomy in a paper-friendly form.

The study separates:

- Information-preserving perturbations: prompt surface form changes while
  patient facts remain fixed.
- Information-altering stress tests: patient evidence is deliberately degraded
  or modified to test robustness.

Saved Tier 1 and Tier 2 outputs cover the compact paper-facing subset:
`A_order`, `B_instruction`, `C_json`, `D_numeric_precision`, and Tier 2
composite runs in the legacy outputs.
