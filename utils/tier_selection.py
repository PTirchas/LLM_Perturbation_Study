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

import json
from pathlib import Path
from typing import List, Tuple, Dict

from utils.perturbations import PERTURBATIONS


def load_tier1_results(results_file: str) -> Dict:
    """Load tier 1 results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def calculate_patient_metrics(results: Dict) -> Dict[int, Dict]:
    """Calculate uncertainty and instability metrics for each patient.
    
    Returns:
        Dict mapping patient_idx to {
            'confidence': avg canonical confidence,
            'label_flip_rate': proportion of perturbations that flip the label,
            'uncertainty_score': combined score for tier 2 selection
        }
    """
    
    metrics = {}
    
    for patient_key, patient_results in results["results_by_patient"].items():
        patient_idx = int(patient_key.split("_")[1])
        
        # Get canonical predictions (new format with instances and replicates)
        canonical_data = patient_results["canonical"]
        if isinstance(canonical_data, dict):
            # New format: canonical is dict with instance keys
            canonical_labels = []
            canonical_confidences = []
            for instance_key, instance_data in canonical_data.items():
                for replicate in instance_data.get("replicates", []):
                    canonical_labels.append(replicate["label"])
                    canonical_confidences.append(replicate["confidence"])
            canonical_label = max(set(canonical_labels), key=canonical_labels.count) if canonical_labels else 0
            canonical_confidence = sum(canonical_confidences) / len(canonical_confidences) if canonical_confidences else 50.0
        else:
            # Old format: canonical is list
            canonical = canonical_data[0] if isinstance(canonical_data, list) else canonical_data
            canonical_label = canonical["label"]
            canonical_confidence = canonical["confidence"]
        
        # Calculate label flip rate across all perturbations
        total_predictions = 0
        label_flips = 0
        
        for family_name in PERTURBATIONS:
            if family_name not in patient_results:
                continue
            family_data = patient_results[family_name]
            
            # Handle both dict (with instances) and list formats
            if isinstance(family_data, dict):
                for instance_key, instance_data in family_data.items():
                    for replicate in instance_data.get("replicates", []):
                        total_predictions += 1
                        if replicate["label"] != canonical_label:
                            label_flips += 1
            else:
                for result in family_data:
                    total_predictions += 1
                    if result["label"] != canonical_label:
                        label_flips += 1
        
        label_flip_rate = label_flips / total_predictions if total_predictions > 0 else 0
        
        # Distance from decision boundary (0.5) - lower is more uncertain
        distance_from_boundary = abs(canonical_confidence - 50.0) / 50.0
        uncertainty_from_boundary = 1.0 - distance_from_boundary
        
        # Combined uncertainty score
        uncertainty_score = (0.6 * label_flip_rate) + (0.4 * uncertainty_from_boundary)
        
        metrics[patient_idx] = {
            "confidence": canonical_confidence,
            "label_flip_rate": label_flip_rate,
            "uncertainty_from_boundary": uncertainty_from_boundary,
            "uncertainty_score": uncertainty_score,
        }
    
    return metrics


def select_tier2_patients(
    results_file: str, 
    num_patients: int = 15,
    verbose: bool = True
) -> List[int]:
    """Select top patients for tier 2 analysis.
    
    Prioritizes patients that are:
    - Near the decision boundary (uncertain predictions)
    - Unstable (high label flip rates from perturbations)
    
    Args:
        results_file: Path to tier 1 results JSON file
        num_patients: Number of patients to select for tier 2
        verbose: Print selection details
        
    Returns:
        List of patient indices selected for tier 2
    """
    
    # Load and analyze tier 1 results
    results = load_tier1_results(results_file)
    metrics = calculate_patient_metrics(results)
    
    # Sort by uncertainty score (descending)
    sorted_patients = sorted(
        metrics.items(),
        key=lambda x: x[1]["uncertainty_score"],
        reverse=True
    )
    
    # Select top N patients
    tier2_patients = [p[0] for p in sorted_patients[:num_patients]]
    
    if verbose:
        print("\n" + "=" * 80)
        print("TIER 2 PATIENT SELECTION")
        print("=" * 80)
        print(f"\nSelected {num_patients} patients with highest uncertainty scores:\n")
        print(f"{'Patient':<10} {'Confidence':<12} {'Flip Rate':<12} {'Boundary Dist':<15} {'Score':<10}")
        print("-" * 80)
        
        for patient_idx, metric in sorted_patients[:num_patients]:
            print(
                f"{patient_idx:<10} "
                f"{metric['confidence']:<12.1f} "
                f"{metric['label_flip_rate']:<12.3f} "
                f"{metric['uncertainty_from_boundary']:<15.3f} "
                f"{metric['uncertainty_score']:<10.3f}"
            )
        
        print("\n" + "=" * 80 + "\n")
    
    return tier2_patients


if __name__ == "__main__":
    # Example usage
    results_file = "outputs/output_tier_1_with_context_0.json"
    tier2_patients = select_tier2_patients(results_file, num_patients=15)
    print(f"Tier 2 patients: {tier2_patients}")
    results_file = "outputs/output_tier_1_without_context_0.json"
    tier2_patients = select_tier2_patients(results_file, num_patients=15)
    print(f"Tier 2 patients: {tier2_patients}")
