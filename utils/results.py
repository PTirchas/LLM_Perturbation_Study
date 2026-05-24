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
Results — calculates all four instability metrics and saves outputs.

Metrics (per the paper's perturbation stability profile):
  1. Label-flip rate        — how often the prediction (0/1) changes
  2. Confidence instability — mean absolute difference in confidence scores
  3. Decision instability   — how often predict↔defer changes
  4. Semantic drift         — mean cosine distance between explanation embeddings
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.perturbations import PERTURBATIONS


# ---------------------------------------------------------------------------
# Lazy-loaded sentence transformer (only imported when semantic drift is used)
# ---------------------------------------------------------------------------

_embedder = None


def _get_embedder():
    """Load the sentence-transformer model on first use."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("WARNING: sentence-transformers not installed. "
                  "Semantic drift will be reported as 0.0. "
                  "Install with: pip install sentence-transformers")
            _embedder = None
    return _embedder


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors (1 - cosine_similarity)."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return 1.0 - dot / norm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_results(results: Dict[int, Dict[str, List]]) -> None:
    """Print the full 4-metric instability table to the console."""

    summary = compute_summary(results)

    print("\n" + "=" * 85)
    print("INSTABILITY SUMMARY (all patients)")
    print("=" * 85)
    print(
        f"{'Family':<25} | {'Flip Rate':<10} | {'Conf Diff':<10} | "
        f"{'Decision Diff':<13} | {'Sem. Drift':<10}"
    )
    print("-" * 85)
    for family_name, stats in summary.items():
        print(
            f"{family_name:<25} | "
            f"{stats['flip_rate']:<10.3f} | "
            f"{stats['mean_conf_diff']:<10.2f} | "
            f"{stats['decision_flip_rate']:<13.3f} | "
            f"{stats['semantic_drift']:<10.4f}"
        )
    print("=" * 85 + "\n")


def save_results(results: Dict[int, Dict[str, List]], config) -> str:
    """Save results to outputs folder with incrementing filenames.

    Returns: path to saved output file
    """

    # Create outputs folder if it doesn't exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    context_suffix = "with_context" if config.use_retrieval else "without_context"
    tier_prefix = f"output_tier_{config.tier}_{context_suffix}"
    
    # Find next available number
    existing_files = list(output_dir.glob(f"{tier_prefix}_*.json"))
    if existing_files:
        # Get numbers from filenames like output_tier_1_with_context_0.json
        numbers = []
        for f in existing_files:
            try:
                parts = f.stem.split("_")
                numbers.append(int(parts[-1]))
            except (ValueError, IndexError):
                continue
        next_num = max(numbers) + 1 if numbers else 0
    else:
        next_num = 0

    output_file = output_dir / f"{tier_prefix}_{next_num}.json"

    # Prepare data for JSON serialization
    summary = compute_summary(results)
    patient_profiles = compute_patient_profiles(results)

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_provider": config.llm_provider.value,
            "llm_model": config.llm_model,
            "num_perturbations": config.num_perturbations,
            "temperature": config.temperature,
            "random_seed": config.random_seed,
            "use_retrieval": config.use_retrieval,
        },
        "results_by_patient": {},
        "patient_profiles": patient_profiles,
        "summary": summary,
    }

    # Add detailed results per patient
    for patient_idx in sorted(results.keys()):
        patient_data = {}
        for family_name, family_results in results[patient_idx].items():
            # Group by instance
            instances = {}
            for r in family_results:
                inst_key = f"instance_{r.instance_id}"
                if inst_key not in instances:
                    instances[inst_key] = {
                        "prompt": r.prompt,
                        "replicates": []
                    }
                
                instances[inst_key]["replicates"].append({
                    "replicate_id": r.replicate_id,
                    "label": r.label,
                    "confidence": r.confidence,
                    "decision": r.decision,
                    "explanation": r.explanation
                })
            patient_data[family_name] = instances
            
        json_data["results_by_patient"][f"patient_{patient_idx}"] = patient_data

    # Save JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # Also save a nice text summary
    summary_file = output_dir / f"{tier_prefix}_{next_num}_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write("LLM PERTURBATION STABILITY EXPERIMENT RESULTS\n")
        f.write("=" * 85 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Provider: {config.llm_provider.value}\n")
        f.write(f"Model: {config.llm_model}\n")
        f.write(f"Patients tested: {len(results)}\n")
        f.write(f"Perturbations per family: {config.num_perturbations}\n")
        f.write(f"Retrieval enabled: {config.use_retrieval}\n\n")

        f.write("-" * 85 + "\n")
        f.write("INSTABILITY SUMMARY\n")
        f.write("-" * 85 + "\n")
        f.write(
            f"{'Family':<25} | {'Flip Rate':<10} | {'Conf Diff':<10} | "
            f"{'Decision Diff':<13} | {'Sem. Drift':<10}\n"
        )
        f.write("-" * 85 + "\n")

        for family_name, stats in summary.items():
            f.write(
                f"{family_name:<25} | "
                f"{stats['flip_rate']:<10.3f} | "
                f"{stats['mean_conf_diff']:<10.2f} | "
                f"{stats['decision_flip_rate']:<13.3f} | "
                f"{stats['semantic_drift']:<10.4f}\n"
            )

        f.write("\n" + "=" * 85 + "\n\n")

        # Per-patient profiles
        f.write("-" * 85 + "\n")
        f.write("PER-PATIENT STABILITY PROFILES\n")
        f.write("-" * 85 + "\n\n")
        for pid, profile in patient_profiles.items():
            f.write(f"  Patient {pid}:\n")
            f.write(f"    Overall flip rate:       {profile['overall_flip_rate']:.3f}\n")
            f.write(f"    Overall conf diff:       {profile['overall_conf_diff']:.2f}\n")
            f.write(f"    Overall decision flip:   {profile['overall_decision_flip']:.3f}\n")
            f.write(f"    Overall semantic drift:  {profile['overall_semantic_drift']:.4f}\n")
            f.write(f"    Most unstable family:    {profile['most_unstable_family']}\n\n")

        f.write("=" * 85 + "\n")

    print(f"\n>> Results saved to: {output_file}")
    print(f">> Summary saved to: {summary_file}\n")

    return str(output_file)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_summary(results: Dict[int, Dict[str, List]]) -> Dict:
    """Compute all 4 instability metrics for each perturbation family."""

    # Get the active families (those that actually have results)
    active_families = _get_active_families(results)

    flip_counts = {k: 0 for k in active_families}
    total_counts = {k: 0 for k in active_families}
    confidence_diffs = {k: [] for k in active_families}
    decision_flips = {k: 0 for k in active_families}
    explanations_canonical = {k: [] for k in active_families}
    explanations_perturbed = {k: [] for k in active_families}

    # Aggregate statistics across all patients
    for patient_results in results.values():
        canonical_label = patient_results["canonical"][0].label
        canonical_conf = patient_results["canonical"][0].confidence
        canonical_dec = patient_results["canonical"][0].decision
        canonical_expl = patient_results["canonical"][0].explanation

        for family_name in active_families:
            for result in patient_results.get(family_name, []):
                # Skip comparing canonical prep 0 against itself
                if family_name == "canonical" and getattr(result, "replicate_id", 0) == 0:
                    continue
                    
                total_counts[family_name] += 1


                # 1. Label flip
                if result.label != canonical_label:
                    flip_counts[family_name] += 1

                # 2. Confidence difference
                conf_diff = abs(result.confidence - canonical_conf)
                confidence_diffs[family_name].append(conf_diff)

                # 3. Decision instability
                if result.decision != canonical_dec:
                    decision_flips[family_name] += 1

                # 4. Collect explanations for semantic drift
                explanations_canonical[family_name].append(canonical_expl)
                explanations_perturbed[family_name].append(result.explanation)

    # Compute semantic drift per family
    semantic_drifts = _compute_semantic_drifts(
        explanations_canonical, explanations_perturbed, active_families
    )

    # Build summary dict
    summary = {}
    for family_name in active_families:
        total = max(1, total_counts[family_name])
        summary[family_name] = {
            "flip_rate": flip_counts[family_name] / total,
            "mean_conf_diff": (
                sum(confidence_diffs[family_name]) /
                max(1, len(confidence_diffs[family_name]))
            ),
            "decision_flip_rate": decision_flips[family_name] / total,
            "semantic_drift": semantic_drifts.get(family_name, 0.0),
            "total_perturbations": total_counts[family_name],
            "label_flips": flip_counts[family_name],
            "decision_flips": decision_flips[family_name],
        }

    return summary


def compute_patient_profiles(results: Dict[int, Dict[str, List]]) -> Dict:
    """Compute a per-patient stability profile.

    For each patient, aggregate metrics across all perturbation families
    to produce an overall instability score.
    """

    active_families = _get_active_families(results)
    profiles = {}

    for patient_idx, patient_results in results.items():
        canonical_label = patient_results["canonical"][0].label
        canonical_conf = patient_results["canonical"][0].confidence
        canonical_dec = patient_results["canonical"][0].decision
        canonical_expl = patient_results["canonical"][0].explanation

        family_flips = {}
        all_flip = 0
        all_conf = []
        all_dec_flip = 0
        all_total = 0

        can_expls = []
        pert_expls = []

        for family_name in active_families:
            fam_flip = 0
            fam_total = 0
            for result in patient_results.get(family_name, []):
                if family_name == "canonical" and getattr(result, "replicate_id", 0) == 0:
                    continue
                    
                fam_total += 1
                all_total += 1
                if result.label != canonical_label:
                    fam_flip += 1
                    all_flip += 1
                all_conf.append(abs(result.confidence - canonical_conf))
                if result.decision != canonical_dec:
                    all_dec_flip += 1
                can_expls.append(canonical_expl)
                pert_expls.append(result.explanation)

            family_flips[family_name] = fam_flip / max(1, fam_total)

        # Patient-level semantic drift
        patient_drift = _compute_pairwise_drift(can_expls, pert_expls)

        total = max(1, all_total)
        most_unstable = max(family_flips, key=family_flips.get) if family_flips else "N/A"

        profiles[patient_idx] = {
            "overall_flip_rate": all_flip / total,
            "overall_conf_diff": sum(all_conf) / max(1, len(all_conf)),
            "overall_decision_flip": all_dec_flip / total,
            "overall_semantic_drift": patient_drift,
            "most_unstable_family": most_unstable,
            "family_flip_rates": family_flips,
        }

    return profiles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_active_families(results: Dict[int, Dict[str, List]]) -> List[str]:
    """Return perturbation families that actually have results (non-empty)."""
    active = set()
    for patient_results in results.values():
        for family_name, family_results in patient_results.items():
            if len(family_results) > 0:
                active.add(family_name)
    
    # Sort order: canonical first, then active PERTURBATIONS order
    ordered = []
    if "canonical" in active:
        ordered.append("canonical")
    for k in PERTURBATIONS:
        if k in active:
            ordered.append(k)
    return ordered


def _compute_semantic_drifts(
    canonical_expls: Dict[str, List[str]],
    perturbed_expls: Dict[str, List[str]],
    families: List[str],
) -> Dict[str, float]:
    """Compute mean cosine distance of explanation embeddings per family."""

    embedder = _get_embedder()
    if embedder is None:
        return {f: 0.0 for f in families}

    drifts = {}
    for family in families:
        can_texts = canonical_expls[family]
        pert_texts = perturbed_expls[family]

        if not can_texts or not pert_texts:
            drifts[family] = 0.0
            continue

        # Embed all texts at once for efficiency
        all_texts = can_texts + pert_texts
        embeddings = embedder.encode(all_texts, show_progress_bar=False)
        n = len(can_texts)
        can_embs = embeddings[:n]
        pert_embs = embeddings[n:]

        # Pairwise cosine distances
        distances = [
            _cosine_distance(can_embs[i], pert_embs[i])
            for i in range(n)
        ]
        drifts[family] = float(np.mean(distances)) if distances else 0.0

    return drifts


def _compute_pairwise_drift(can_texts: List[str], pert_texts: List[str]) -> float:
    """Compute mean cosine distance for a single patient's explanations."""
    embedder = _get_embedder()
    if embedder is None or not can_texts:
        return 0.0

    all_texts = can_texts + pert_texts
    embeddings = embedder.encode(all_texts, show_progress_bar=False)
    n = len(can_texts)
    distances = [
        _cosine_distance(embeddings[i], embeddings[n + i])
        for i in range(n)
    ]
    return float(np.mean(distances)) if distances else 0.0
