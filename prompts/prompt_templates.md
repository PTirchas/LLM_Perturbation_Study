# Prompt Templates

All model calls use the common system instruction in
`prompts/system_instruction.txt`. User prompts are generated from structured
patient features.

## Output Schema

```json
{
  "prediction": 0,
  "confidence": 0.75,
  "explanation": "brief clinical rationale",
  "decision": "predict"
}
```

During parsing, confidence is converted to a 0-100 percentage. Responses with
confidence below 50 percent are treated as `defer`.

## Canonical Prompt

```text
age: [age]; plasma_CA19_9: [plasma_CA19_9]; creatinine: [creatinine]; LYVE1: [LYVE1]; REG1B: [REG1B]; TFF1: [TFF1]; REG1A: [REG1A]
```

## Retrieval Context Prompt

```text
Here are similar patient cases for reference:

Case 1: [context feature values] -> Outcome: [label]
...
Case k: [context feature values] -> Outcome: [label]

Now predict for the following patient:

[target patient prompt]
```

## Family A - Feature Order

Same feature-value pairs as canonical, randomly permuted.

```text
[feature_j]: [value_j]; [feature_m]: [value_m]; ...; [feature_r]: [value_r]
```

## Family B - Instruction Template

One paraphrased clinical instruction is prepended to the unchanged patient
record. The implementation samples from a fixed list in `utils/perturbations.py`.

```text
[instruction paraphrase]
age: [age]; plasma_CA19_9: [plasma_CA19_9]; ...
```

## Family C - Structural Format

The same patient facts are rendered as JSON, CSV, markdown, or prose. The saved
main results use `C_json`.

```json
{
  "age": "[age]",
  "plasma_CA19_9": "[plasma_CA19_9]",
  "creatinine": "[creatinine]",
  "LYVE1": "[LYVE1]",
  "REG1B": "[REG1B]",
  "TFF1": "[TFF1]",
  "REG1A": "[REG1A]"
}
```

## Family D - Numeric Precision

All features are present, but numeric values are shown with reduced precision,
for example integer rounding or one decimal place.

## Family E - Delimiters

Feature-value pairs are separated with alternate delimiters such as commas,
pipes, newlines, dashes, or slashes.

## Family F - Retrieval

Target patient facts are fixed while the retrieved context set varies by `k`
and diversity penalty.

## Composite

Feature order, delimiter style, and instruction paraphrase are varied in the
same prompt.

## Information-Altering Stress Tests

Families G-J deliberately change or degrade the patient facts:

- `G_feature_mask`: drops one or two features.
- `H_controlled_rounding`: aggressively rounds all numeric values.
- `I_noise`: applies bounded numeric noise.
- `J_missingness`: replaces values with missing-value placeholders.

These are robustness stress tests, not invariance tests.
