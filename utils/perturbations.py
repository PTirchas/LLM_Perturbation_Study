# Copyright (C) 2026 Panagiotis Tirchas
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Perturbation functions — all the ways we rephrase the same patient data.

Information-preserving perturbations (A–F):
    These preserve the patient facts exactly but alter the surface form.

Information-altering perturbations (G–J):
    These intentionally degrade or modify the evidence as robustness tests.
"""

import json
import random
from typing import Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Family A – Feature-order permutations
# ---------------------------------------------------------------------------

def perturb_feature_order(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Shuffle the order of feature–value pairs."""
    if seed is not None:
        random.seed(seed)
    keys = list(features.keys())
    random.shuffle(keys)
    return "; ".join([f"{k}: {features[k]}" for k in keys])


# ---------------------------------------------------------------------------
# Family B – Instruction-template perturbations
# ---------------------------------------------------------------------------

def perturb_instruction(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Replace the high-level instruction with a semantically equivalent template."""
    if seed is not None:
        random.seed(seed)
    templates = [
        "You are an oncology assistant. Given the following patient data, predict the outcome.",
        "As a clinical decision support system, use these biomarkers to determine disease.",
        "Use only the provided features to assess whether the patient should be diagnosed.",
        "Please provide a diagnosis and brief justification.",
        "Analyse the following biomarker panel and predict whether the patient is positive.",
        "Based on clinical laboratory values below, classify the patient's disease status.",
    ]
    prefix = random.choice(templates)
    facts = "; ".join([f"{k}: {v}" for k, v in features.items()])
    return prefix + "\n" + facts


# ---------------------------------------------------------------------------
# Family C – Structural-format perturbations
# ---------------------------------------------------------------------------

def perturb_json_format(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Render patient data as a JSON object."""
    return json.dumps(features, indent=2)


def perturb_csv_format(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Render patient data as a CSV row with header."""
    header = ",".join(features.keys())
    values = ",".join([str(v) for v in features.values()])
    return header + "\n" + values


def perturb_markdown_table(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Render patient data as a markdown table."""
    header = "| Feature | Value |"
    sep = "|---------|-------|"
    rows = [f"| {k} | {v} |" for k, v in features.items()]
    return "\n".join([header, sep] + rows)


def perturb_prose(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Render patient data as free-text prose."""
    if seed is not None:
        random.seed(seed)

    intros = [
        "The patient is",
        "This individual is",
        "We have a patient who is",
    ]
    intro = random.choice(intros)

    parts = []
    items = list(features.items())
    for i, (k, v) in enumerate(items):
        # First item includes the intro
        if i == 0 and k.lower() == "age":
            parts.append(f"{intro} {int(v)} years old")
        elif k.lower() == "age":
            parts.append(f"aged {int(v)}")
        else:
            parts.append(f"a {k} level of {v}")

    return ", with ".join(parts) + "."


# ---------------------------------------------------------------------------
# Family D – Numeric-format perturbations
# ---------------------------------------------------------------------------

def perturb_numeric_precision(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Vary numeric presentation: randomly round to int or show 1 decimal."""
    if seed is not None:
        random.seed(seed)
    parts = []
    for k, v in features.items():
        if random.random() < 0.5:
            parts.append(f"{k}: {int(round(v))}")
        else:
            parts.append(f"{k}: {v:.1f}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Family E – Delimiter perturbations
# ---------------------------------------------------------------------------

def perturb_delimiters(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Change the delimiters between feature–value pairs."""
    if seed is not None:
        random.seed(seed)
    delimiters = [", ", "; ", " | ", "\n", " — ", " // "]
    delim = random.choice(delimiters)
    return delim.join([f"{k}: {features[k]}" for k in features])


# ---------------------------------------------------------------------------
# Family F – Retrieval perturbations (information-preserving)
# ---------------------------------------------------------------------------

def perturb_retrieval(features: Dict[str, float], seed: Optional[int] = None,
                      context_df=None, **kw) -> str:
    """Vary the retrieved support set: change K and diversity penalty.

    Holds patient facts fixed while varying what context examples
    are prepended to the prompt (number of neighbours, diversity).
    """
    if context_df is None:
        # Fallback: no context available, just return canonical format
        return "; ".join([f"{k}: {v}" for k, v in features.items()])

    from utils.retrieval import get_nearest_neighbours, format_neighbours_as_context

    if seed is not None:
        random.seed(seed)

    # Vary K (number of neighbours) and diversity
    k_options = [1, 2, 3, 5]
    diversity_options = [0.0, 0.1, 0.3, 0.5]
    k = random.choice(k_options)
    diversity = random.choice(diversity_options)

    neighbours = get_nearest_neighbours(
        features, context_df, k=k, seed=seed, diversity_penalty=diversity
    )
    context_str = format_neighbours_as_context(neighbours)
    patient_str = "; ".join([f"{k}: {v}" for k, v in features.items()])

    return context_str + patient_str


# ---------------------------------------------------------------------------
# Family G – Feature masking (information-altering)
# ---------------------------------------------------------------------------

def perturb_feature_mask(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Randomly drop 1–2 features from the patient record.

    This is an information-altering perturbation: it intentionally
    degrades the evidence to test robustness.
    """
    if seed is not None:
        random.seed(seed)

    keys = list(features.keys())
    num_to_drop = random.randint(1, min(2, len(keys) - 1))
    dropped = random.sample(keys, num_to_drop)

    remaining = {k: v for k, v in features.items() if k not in dropped}
    return "; ".join([f"{k}: {v}" for k, v in remaining.items()])


# ---------------------------------------------------------------------------
# Family H – Controlled rounding (information-altering)
# ---------------------------------------------------------------------------

def perturb_controlled_rounding(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Round all numeric values to 0 or 1 decimal places.

    Unlike Family D which randomly varies precision per-feature,
    this applies a uniform aggressive rounding to all values.
    """
    if seed is not None:
        random.seed(seed)

    decimals = random.choice([0, 1])
    parts = []
    for k, v in features.items():
        rounded = round(v, decimals)
        if decimals == 0:
            parts.append(f"{k}: {int(rounded)}")
        else:
            parts.append(f"{k}: {rounded}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Family I – Numeric noise (information-altering)
# ---------------------------------------------------------------------------

def perturb_numeric_noise(features: Dict[str, float], seed: Optional[int] = None,
                          noise: float = 0.05, **kw) -> str:
    """Add bounded random noise (±noise%) to all numeric values."""
    if seed is not None:
        np.random.seed(seed)
    noisy = {
        k: v * (1 + np.random.uniform(-noise, noise))
        for k, v in features.items()
    }
    return "; ".join([f"{k}: {v}" for k, v in noisy.items()])


# ---------------------------------------------------------------------------
# Family J – Missingness introduction (information-altering)
# ---------------------------------------------------------------------------

def perturb_missingness(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Replace 1–2 feature values with 'N/A' or 'missing'.

    Simulates incomplete patient records to test whether the model
    can handle missing data gracefully.
    """
    if seed is not None:
        random.seed(seed)

    keys = list(features.keys())
    num_missing = random.randint(1, min(2, len(keys) - 1))
    missing_keys = random.sample(keys, num_missing)
    placeholders = ["N/A", "missing", "unknown", "not available"]

    parts = []
    for k, v in features.items():
        if k in missing_keys:
            parts.append(f"{k}: {random.choice(placeholders)}")
        else:
            parts.append(f"{k}: {v}")
    return "; ".join(parts)


def perturb_composite(features: Dict[str, float], seed: Optional[int] = None, **kw) -> str:
    """Composite family: simultaneously applies order, delimiter, and template changes."""
    # To chain perturbations safely without re-parsing, we adapt the logic directly.
    # 1. Shuffle order
    if seed is not None:
        random.seed(seed)
    keys = list(features.keys())
    random.shuffle(keys)
    
    # 2. Change delimiters
    delimiters = [", ", "; ", " | ", "\\n", " — ", " // "]
    delim = random.choice(delimiters)
    facts_str = delim.join([f"{k}: {features[k]}" for k in keys])
    
    # 3. Change instruction template
    templates = [
        "You are an oncology assistant. Given the following patient data, predict the outcome.",
        "As a clinical decision support system, use these biomarkers to determine disease.",
        "Use only the provided features to assess whether the patient should be diagnosed.",
        "Please provide a diagnosis and brief justification.",
        "Analyse the following biomarker panel and predict whether the patient is positive.",
        "Based on clinical laboratory values below, classify the patient's disease status.",
    ]
    prefix = random.choice(templates)
    
    return prefix + "\n" + facts_str


# ---------------------------------------------------------------------------
# Registry — maps family names to functions
# ---------------------------------------------------------------------------

# Information-preserving perturbations
PERTURBATIONS_PRESERVING = {
    "A_order":            perturb_feature_order,
    "B_instruction":      perturb_instruction,
    "C_json":             perturb_json_format,
    "C_csv":              perturb_csv_format,
    "C_markdown":         perturb_markdown_table,
    "C_prose":            perturb_prose,
    "D_numeric_precision": perturb_numeric_precision,
    "E_delimiters":       perturb_delimiters,
    "F_retrieval":        perturb_retrieval,
    "Composite":          perturb_composite,
}

# Information-altering perturbations (robustness tests)
PERTURBATIONS_ALTERING = {
    "G_feature_mask":        perturb_feature_mask,
    "H_controlled_rounding": perturb_controlled_rounding,
    "I_noise":               lambda f, seed=None, **kw: perturb_numeric_noise(f, seed=seed, noise=0.05),
    "J_missingness":         perturb_missingness,
}

# Combined registry (used by the experiment loop)
PERTURBATIONS = {**PERTURBATIONS_PRESERVING, **PERTURBATIONS_ALTERING}


# ---------------------------------------------------------------------------
# Canonical (baseline) prompt
# ---------------------------------------------------------------------------

def canonical_prompt(features: Dict[str, float]) -> str:
    """Standard prompt format — the baseline all perturbations are compared against."""
    return "; ".join([f"{k}: {v}" for k, v in features.items()])
