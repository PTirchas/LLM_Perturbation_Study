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
Experiment runner — orchestrates the perturbation experiment for each patient.

For each patient:
  1. Get a baseline ("canonical") prediction from the LLM
  2. For each perturbation family, generate N variant prompts
  3. Ask the LLM for each variant and store the result
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.config import Config
from utils.llm import call_llm
from utils.perturbations import PERTURBATIONS, canonical_prompt
from utils.retrieval import get_nearest_neighbours, format_neighbours_as_context


@dataclass
class PerturbationResult:
    """Result from a single LLM call (canonical or perturbed)."""
    label: int              # Binary prediction (0 or 1)
    confidence: float       # Confidence score (0–100)
    explanation: str        # Natural-language reasoning
    decision: str           # "predict" or "defer"
    perturbation_name: str  # Which family produced this prompt
    prompt: str             # The actual prompt sent to the LLM
    instance_id: int        # Which perturbation variant (0 for canonical)
    replicate_id: int       # Which stochastic repeat (temperature sampling)


def run_experiment(
    client,
    config: Config,
    predict_df: pd.DataFrame,
    feature_names: List[str],
    context_df: Optional[pd.DataFrame] = None,
) -> Dict[int, Dict[str, List[PerturbationResult]]]:
    """Run perturbation experiments on all patients.

    Parameters
    ----------
    client : LLM client (Azure/Claude/Gemini)
    config : experiment configuration
    predict_df : patients to predict on
    feature_names : which columns are features
    context_df : optional training data for retrieval perturbations (Family F)
    """

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    results = {}

    for patient_idx, row in predict_df.iterrows():
        # Extract features for this patient
        features = {
            k: float(row[k])
            for k in feature_names
            if k in row
        }

        print(f"\nPatient {patient_idx}:")
        patient_results = run_patient_experiment(
            client, config, features, patient_idx, context_df
        )
        results[patient_idx] = patient_results

    return results


def run_patient_experiment(
    client,
    config: Config,
    features: Dict[str, float],
    patient_idx: int,
    context_df: Optional[pd.DataFrame] = None,
) -> Dict[str, List[PerturbationResult]]:
    """Run all perturbation families for a single patient."""

    patient_results = {}

    # ---- Base Context (for retrieval condition) ----
    base_context_str = ""
    if context_df is not None:
        neighbours = get_nearest_neighbours(features, context_df, k=3, seed=0, diversity_penalty=0.0)
        base_context_str = format_neighbours_as_context(neighbours) + "\n\n"

    # ---- Canonical (baseline) prediction ----
    canonical = base_context_str + canonical_prompt(features)
    canonical_results = []
    for r in range(config.num_stochastic_replicates):
        label, conf, expl, dec = call_llm(
            client,
            config.llm_provider,
            config.llm_model,
            canonical,
            temperature=config.temperature,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
        )
        canonical_results.append(
            PerturbationResult(label, conf, expl, dec, "canonical", canonical, 0, r)
        )
        if r == 0:
            print(f"  Canonical (rep 0): label={label}, confidence={conf:.1f}")
    patient_results["canonical"] = canonical_results

    # ---- Run each perturbation family ----
    for family_name, perturb_func in PERTURBATIONS.items():

        # Skip retrieval perturbations if no context data is available
        if family_name == "F_retrieval" and context_df is None:
            print(f"  {family_name}: SKIPPED (no context data)")
            # Store empty results so downstream code can handle it
            patient_results[family_name] = []
            continue

        family_results = []

        for i in range(config.num_perturbations):
            # Generate deterministic seed for reproducibility
            seed = hash((config.random_seed, patient_idx, family_name, i)) % (2**31)

            # Build perturbation keyword arguments
            perturb_kwargs = {"seed": seed}
            if family_name == "F_retrieval":
                perturb_kwargs["context_df"] = context_df

            prompt = perturb_func(features, **perturb_kwargs)
            
            # Prepend base context for all families EXCEPT F_retrieval (which generates its own)
            if context_df is not None and family_name != "F_retrieval":
                prompt = base_context_str + prompt
            
            for r in range(config.num_stochastic_replicates):
                label, conf, expl, dec = call_llm(
                    client,
                    config.llm_provider,
                    config.llm_model,
                    prompt,
                    temperature=config.temperature,
                    max_retries=config.max_retries,
                    retry_delay=config.retry_delay,
                )

                family_results.append(
                    PerturbationResult(label, conf, expl, dec, family_name, prompt, i, r)
                )
                if r == 0:
                    print(f"  {family_name} #{i+1} (rep 0): label={label}, confidence={conf:.1f}")

        patient_results[family_name] = family_results

    return patient_results
