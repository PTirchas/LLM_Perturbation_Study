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
Comprehensive Analysis Script for LLM Perturbation Experiments

Generates all available tables and figures from Tier 1 & Tier 2 results:
  ✓ Table 1: Perturbation taxonomy
  ✓ Table 2: Experimental design matrix
  ✓ Table 3: Instability metrics by family and retrieval condition
  ✓ Table 4: Variance decomposition of instability signals
  ✓ Table 5: Deferral baselines benchmark
  ✓ Figure 1: System architecture
  ✓ Figure 2: Taxonomy (invariance vs robustness)
  ✓ Figure 3: Risk-coverage curves
  ✓ Figure 4: Patient-by-perturbation heatmap
  ✓ Figure 6: Variance attribution plot
  
  ⏳ Later (need additional data):
  ⊘ Figure 5: Semantic-drift embeddings (needs embeddings)
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

FAMILY_LABELS = {
    "A_order": "Family A",
    "B_instruction": "Family B",
    "C_json": "Family C",
    "D_numeric_precision": "Family D",
    "Composite": "Composite",
}

FAMILY_DESCRIPTIONS = {
    "A_order": "Feature order perturbations",
    "B_instruction": "Instruction-template perturbations",
    "C_json": "Structural-format perturbations",
    "D_numeric_precision": "Numeric-precision perturbations",
}

ANALYSIS_FAMILY_ORDER = ["A_order", "B_instruction", "C_json", "D_numeric_precision"]
PLOT_COLORS = {
    "Perturbation flip rate": "#c76b00",
    "Boundary uncertainty": "#1f7a5c",
    "Semantic drift": "#7a5195",
}
FAMILY_COLORS = {
    "Canonical": "#111111",
    "Family A": "#1f77b4",
    "Family B": "#e07a00",
    "Family C": "#2a9d8f",
    "Family D": "#c05621",
}

# Set style
sns.set_style("white")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False


def load_all_results() -> Tuple[Dict, Dict, Dict, Dict]:
    """Load all available tier 1 and tier 2 results."""
    results = {
        'tier1_with_context': None,
        'tier1_without_context': None,
        'tier2_with_context': None,
        'tier2_without_context': None,
    }
    
    patterns = {
        'tier1_with_context': 'outputs/output_tier_1_with_context*.json',
        'tier1_without_context': 'outputs/output_tier_1_without_context*.json',
        'tier2_with_context': 'outputs/output_tier_2_with_context*.json',
        'tier2_without_context': 'outputs/output_tier_2_without_context*.json',
    }
    
    for key, pattern in patterns.items():
        files = sorted(Path('outputs').glob(pattern.split('/')[-1]))
        if files:
            latest = files[-1]
            print(f"Loading {key}: {latest}")
            with open(latest) as f:
                results[key] = json.load(f)
    
    return results


def _flatten_result_entries(result_group) -> List[Dict]:
    """Normalize old list-based and new instance/replicate result formats."""
    if isinstance(result_group, list):
        return [entry for entry in result_group if isinstance(entry, dict)]

    if not isinstance(result_group, dict):
        return []

    if "replicates" in result_group:
        replicates = result_group.get("replicates", [])
        return [entry for entry in replicates if isinstance(entry, dict)]

    flattened = []
    for value in result_group.values():
        if isinstance(value, dict) and "replicates" in value:
            replicates = value.get("replicates", [])
            flattened.extend(entry for entry in replicates if isinstance(entry, dict))
        elif isinstance(value, list):
            flattened.extend(entry for entry in value if isinstance(entry, dict))

    if flattened:
        return flattened

    if "label" in result_group:
        return [result_group]

    return []


def _extract_canonical_label(patient_data: Dict, default: int = 0) -> int:
    """Return the majority canonical label across all available replicates."""
    canonical_results = _flatten_result_entries(patient_data.get("canonical"))
    labels = [entry.get("label") for entry in canonical_results if entry.get("label") is not None]
    if not labels:
        return default

    return Counter(labels).most_common(1)[0][0]


def _extract_canonical_stats(patient_data: Dict) -> Dict:
    """Return canonical majority label/decision and mean confidence."""
    canonical_results = _flatten_result_entries(patient_data.get("canonical"))
    if not canonical_results:
        return {"label": 0, "confidence": 50.0, "decision": "predict"}

    labels = [entry.get("label", 0) for entry in canonical_results]
    decisions = [entry.get("decision", "predict") for entry in canonical_results]
    confidences = [entry.get("confidence", 50.0) for entry in canonical_results]

    return {
        "label": Counter(labels).most_common(1)[0][0],
        "confidence": float(np.mean(confidences)) if confidences else 50.0,
        "decision": Counter(decisions).most_common(1)[0][0] if decisions else "predict",
    }


def _iter_result_entries_with_instance(result_group):
    """Yield `(instance_id, result_dict)` pairs across supported result formats."""
    if isinstance(result_group, list):
        for idx, entry in enumerate(result_group):
            if isinstance(entry, dict):
                yield idx, entry
        return

    if not isinstance(result_group, dict):
        return

    if "label" in result_group:
        yield 0, result_group
        return

    if "replicates" in result_group:
        for entry in result_group.get("replicates", []):
            if isinstance(entry, dict):
                yield 0, entry
        return

    for idx, (instance_key, instance_data) in enumerate(result_group.items()):
        instance_id = idx
        if isinstance(instance_key, str) and instance_key.startswith("instance_"):
            try:
                instance_id = int(instance_key.split("_", 1)[1])
            except ValueError:
                instance_id = idx

        if isinstance(instance_data, dict) and "replicates" in instance_data:
            for entry in instance_data.get("replicates", []):
                if isinstance(entry, dict):
                    yield instance_id, entry
        elif isinstance(instance_data, list):
            for entry in instance_data:
                if isinstance(entry, dict):
                    yield instance_id, entry
        elif isinstance(instance_data, dict) and "label" in instance_data:
            yield instance_id, instance_data


def _get_result_metadata(result_key: str) -> Tuple[str, str]:
    """Map stored result keys to display labels."""
    tier = "Tier 1" if "tier1" in result_key else "Tier 2"
    context = "With Context" if "with_context" in result_key else "Without Context"
    return tier, context


def _get_family_label(family_name: str) -> str:
    """Map JSON perturbation keys to paper-facing family labels."""
    return FAMILY_LABELS.get(family_name, family_name)


def _include_family_for_tier(family_name: str, tier: str) -> bool:
    """Exclude Tier 2 composite family from paper analyses."""
    if family_name == "canonical":
        return True
    if tier == "Tier 2" and family_name == "Composite":
        return False
    return family_name in ANALYSIS_FAMILY_ORDER or family_name == "Composite"


def _style_publication_axes(ax, *, heatmap: bool = False) -> None:
    """Use clean publication-style axes."""
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False, labelsize=10)
    if heatmap:
        ax.tick_params(length=0)


def _add_panel_label(ax, label: str) -> None:
    """Add a bold panel label in the top-left corner."""
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )


def _save_figure_bundle(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save each figure as both PNG and PDF for manuscript use."""
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[OK] Figure saved to: {png_path}")
    print(f"[OK] Figure saved to: {pdf_path}")


def _load_ground_truth_labels() -> Dict[int, int]:
    """Load pancreatic dataset labels aligned to result patient indices."""
    actual_path = Path("data/actual.csv")
    if not actual_path.exists():
        return {}

    df = pd.read_csv(actual_path)
    if "patient_diagnosis" not in df.columns:
        return {}

    if "Patient" in df.columns:
        df = df.sort_values("Patient").reset_index(drop=True)

    return {idx: int(label) for idx, label in enumerate(df["patient_diagnosis"].tolist())}


def _lookup_patient_profile(patient_profiles: Dict, patient_idx: int) -> Dict:
    """Handle both string and integer patient-profile keys."""
    return patient_profiles.get(str(patient_idx), patient_profiles.get(patient_idx, {}))


def _infer_design_row(result_data: Dict, tier_label: str, purpose: str, goal: str, context_count: int) -> Dict:
    """Infer design-matrix values from a saved results file."""
    results_by_patient = result_data.get("results_by_patient", {})
    patient_count = len(results_by_patient)

    if not results_by_patient:
        return {
            "Tier": tier_label,
            "Purpose": purpose,
            "Context": "N/A",
            "Patients": 0,
            "Families": 0,
            "Instances/Family": 0,
            "Replicates": 0,
            "Perturb Calls": 0,
            "Goal": goal,
        }

    sample_patient = next(iter(results_by_patient.values()))
    perturbation_families = [
        family for family in sample_patient.keys()
        if family != "canonical" and _include_family_for_tier(family, tier_label)
    ]

    instance_counts = []
    replicate_counts = []
    for family_name in perturbation_families:
        family_data = sample_patient.get(family_name, {})
        instance_ids = set()
        family_replicates = []
        for instance_id, entry in _iter_result_entries_with_instance(family_data):
            instance_ids.add(instance_id)
            family_replicates.append(int(entry.get("replicate_id", 0)))
        if instance_ids:
            instance_counts.append(len(instance_ids))
        if family_replicates:
            replicate_counts.append(max(family_replicates) + 1)

    context_label = "Both" if context_count >= 2 else "Single"
    instances_per_family = int(round(np.mean(instance_counts))) if instance_counts else 0
    replicates = int(round(np.mean(replicate_counts))) if replicate_counts else 0
    perturb_calls = patient_count * len(perturbation_families) * instances_per_family * replicates * context_count

    return {
        "Tier": tier_label,
        "Purpose": purpose,
        "Context": context_label,
        "Patients": patient_count,
        "Families": len(perturbation_families),
        "Instances/Family": instances_per_family,
        "Replicates": replicates,
        "Perturb Calls": perturb_calls,
        "Goal": goal,
    }


def _build_prediction_dataframe(results: Dict) -> pd.DataFrame:
    """Build a prediction-level dataframe across all saved results."""
    rows = []
    ground_truth = _load_ground_truth_labels()

    for result_key, result_data in results.items():
        if not result_data:
            continue

        tier, context = _get_result_metadata(result_key)
        results_by_patient = result_data.get("results_by_patient", {})

        for patient_key, patient_data in results_by_patient.items():
            patient_idx = int(patient_key.split("_")[1])
            canonical = _extract_canonical_stats(patient_data)
            actual_label = ground_truth.get(patient_idx)

            for family_name, family_data in patient_data.items():
                if not _include_family_for_tier(family_name, tier):
                    continue
                for instance_id, entry in _iter_result_entries_with_instance(family_data):
                    replicate_id = int(entry.get("replicate_id", 0))
                    label = entry.get("label")
                    confidence = float(entry.get("confidence", np.nan))
                    decision = entry.get("decision", "predict")
                    is_baseline_call = family_name == "canonical" and instance_id == 0 and replicate_id == 0

                    rows.append({
                        "tier": tier,
                        "context": context,
                        "patient_idx": patient_idx,
                        "family": family_name,
                        "instance_id": instance_id,
                        "family_instance": f"{family_name}:{instance_id}",
                        "replicate_id": replicate_id,
                        "label": label,
                        "confidence": confidence,
                        "decision": decision,
                        "label_flip": int(label != canonical["label"]) if label is not None else np.nan,
                        "conf_delta": abs(confidence - canonical["confidence"]) if not np.isnan(confidence) else np.nan,
                        "decision_flip": int(decision != canonical["decision"]),
                        "correct": int(label == actual_label) if actual_label is not None and label is not None else np.nan,
                        "error": int(label != actual_label) if actual_label is not None and label is not None else np.nan,
                        "is_baseline_call": is_baseline_call,
                    })

    return pd.DataFrame(rows)


def _build_patient_metric_dataframe(results: Dict) -> pd.DataFrame:
    """Aggregate per-patient deferral scores across tiers and retrieval conditions."""
    rows = []
    ground_truth = _load_ground_truth_labels()

    for result_key, result_data in results.items():
        if not result_data:
            continue

        tier, context = _get_result_metadata(result_key)
        patient_profiles = result_data.get("patient_profiles", {})

        for patient_key, patient_data in result_data.get("results_by_patient", {}).items():
            patient_idx = int(patient_key.split("_")[1])
            canonical = _extract_canonical_stats(patient_data)
            actual_label = ground_truth.get(patient_idx)

            perturbation_entries = []
            for family_name, family_data in patient_data.items():
                if family_name == "canonical" or not _include_family_for_tier(family_name, tier):
                    continue
                perturbation_entries.extend(entry for _, entry in _iter_result_entries_with_instance(family_data))

            flip_rate = float(np.mean([
                entry.get("label") != canonical["label"] for entry in perturbation_entries
            ])) if perturbation_entries else 0.0

            confidence_instability = float(np.mean([
                abs(float(entry.get("confidence", canonical["confidence"])) - canonical["confidence"])
                for entry in perturbation_entries
            ])) if perturbation_entries else 0.0

            decision_instability = float(np.mean([
                entry.get("decision", canonical["decision"]) != canonical["decision"]
                for entry in perturbation_entries
            ])) if perturbation_entries else 0.0

            boundary_uncertainty = 1.0 - abs(canonical["confidence"] - 50.0) / 50.0
            profile = _lookup_patient_profile(patient_profiles, patient_idx)
            semantic_drift = float(profile.get("overall_semantic_drift", 0.0))
            composite_uncertainty = (0.6 * flip_rate) + (0.4 * boundary_uncertainty)

            rows.append({
                "tier": tier,
                "context": context,
                "patient_idx": patient_idx,
                "canonical_label": canonical["label"],
                "canonical_confidence": canonical["confidence"],
                "actual_label": actual_label,
                "correct": int(canonical["label"] == actual_label) if actual_label is not None else np.nan,
                "error": int(canonical["label"] != actual_label) if actual_label is not None else np.nan,
                "boundary_uncertainty": boundary_uncertainty,
                "flip_rate": flip_rate,
                "confidence_instability": confidence_instability,
                "decision_instability": decision_instability,
                "semantic_drift": semantic_drift,
                "composite_uncertainty": composite_uncertainty,
            })

    return pd.DataFrame(rows)


def _hierarchical_variance_decomposition(
    df: pd.DataFrame,
    target_col: str,
    factor_cols: List[str],
) -> Dict[str, float]:
    """Sequential grouped-residual variance decomposition."""
    numeric = pd.to_numeric(df[target_col], errors="coerce")
    valid_mask = numeric.notna()
    if not valid_mask.any():
        return {factor: 0.0 for factor in factor_cols} | {"Residual": 0.0}

    sub_df = df.loc[valid_mask].copy()
    centered = numeric.loc[valid_mask] - numeric.loc[valid_mask].mean()
    total_var = float(np.var(centered.to_numpy()))

    if total_var == 0:
        return {factor: 0.0 for factor in factor_cols} | {"Residual": 0.0}

    residual = centered.copy()
    components = {}

    for factor in factor_cols:
        grouped_component = residual.groupby(sub_df[factor]).transform("mean")
        components[factor] = float(np.var(grouped_component.to_numpy())) / total_var
        residual = residual - grouped_component

    components["Residual"] = float(np.var(residual.to_numpy())) / total_var
    return components


def _pairwise_auc(scores: pd.Series, labels: pd.Series) -> float:
    """Binary AUC computed from pairwise ranking probabilities."""
    score_values = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    label_values = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=float)

    valid_mask = ~np.isnan(score_values) & ~np.isnan(label_values)
    score_values = score_values[valid_mask]
    label_values = label_values[valid_mask].astype(int)

    pos = score_values[label_values == 1]
    neg = score_values[label_values == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    wins = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(wins + 0.5 * ties)


def _compute_aurc(scores: pd.Series, errors: pd.Series) -> float:
    """Area under the risk-coverage curve when higher scores are deferred first."""
    ranked = pd.DataFrame({
        "score": pd.to_numeric(scores, errors="coerce"),
        "error": pd.to_numeric(errors, errors="coerce"),
    }).dropna()

    if ranked.empty:
        return float("nan")

    ranked = ranked.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    error_values = ranked["error"].to_numpy(dtype=float)
    risks = []

    for deferred_count in range(len(error_values) + 1):
        retained = error_values[deferred_count:]
        risks.append(float(retained.mean()) if len(retained) else 0.0)

    return float(np.mean(risks))


def _evaluate_deferral_baseline(sub_df: pd.DataFrame, score_col: str, defer_fraction: float = 0.2) -> Dict:
    """Evaluate how well a score prioritizes incorrect canonical predictions for deferral."""
    ranked = sub_df.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    if ranked.empty:
        return {
            "AUROC": np.nan,
            "AURC": np.nan,
            "Error Recall@20%": np.nan,
            "Retained Acc@80%": np.nan,
        }

    defer_n = max(1, int(np.ceil(len(ranked) * defer_fraction)))
    deferred = ranked.iloc[:defer_n]
    retained = ranked.iloc[defer_n:]
    total_errors = max(1, int(ranked["error"].sum()))

    return {
        "AUROC": _pairwise_auc(ranked[score_col], ranked["error"]),
        "AURC": _compute_aurc(ranked[score_col], ranked["error"]),
        "Error Recall@20%": float(deferred["error"].sum()) / total_errors,
        "Retained Acc@80%": float(retained["correct"].mean()) if len(retained) else np.nan,
    }


def _compute_risk_coverage_curve(sub_df: pd.DataFrame, score_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return retained-fraction vs risk curve when deferring highest-score patients first."""
    ranked = sub_df[["patient_idx", score_col, "error"]].copy()
    ranked[score_col] = pd.to_numeric(ranked[score_col], errors="coerce")
    ranked["error"] = pd.to_numeric(ranked["error"], errors="coerce")
    ranked = ranked.dropna().sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)

    if ranked.empty:
        return np.array([]), np.array([])

    retained_fractions = [1.0]
    risks = [float(ranked["error"].mean())]

    for deferred_count in range(1, len(ranked) + 1):
        retained = ranked.iloc[deferred_count:]
        retained_fraction = len(retained) / len(ranked)
        risk = float(retained["error"].mean()) if len(retained) else 0.0
        retained_fractions.append(retained_fraction)
        risks.append(risk)

    coverage_array = np.array(retained_fractions[::-1])
    risk_array = np.array(risks[::-1])
    return coverage_array, risk_array


def _risk_at_retention(sub_df: pd.DataFrame, score_col: str, retained_fraction: float) -> Dict:
    """Return the risk at a target retained fraction after deferring highest-score patients."""
    ranked = sub_df[["patient_idx", score_col, "error"]].copy()
    ranked[score_col] = pd.to_numeric(ranked[score_col], errors="coerce")
    ranked["error"] = pd.to_numeric(ranked["error"], errors="coerce")
    ranked = ranked.dropna().sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)

    if ranked.empty:
        return {"retained_n": 0, "retention": np.nan, "risk": np.nan}

    retained_n = int(np.ceil(len(ranked) * retained_fraction))
    retained_n = min(len(ranked), max(1, retained_n))
    retained = ranked.iloc[len(ranked) - retained_n:]

    return {
        "retained_n": retained_n,
        "retention": retained_n / len(ranked),
        "risk": float(retained["error"].mean()) if len(retained) else np.nan,
    }


def _sanitize_latex_text(text: str) -> str:
    """Clean mojibake and normalize text for LaTeX output."""
    replacements = {
        "Â±": "+/-",
        "â€¢": "-",
        "â†’": "->",
        "âœ“": "OK",
        "â³": "Later",
        "âŠ˜": "Pending",
        "–": "-",
        "—": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _latex_escape(value) -> str:
    """Escape values for LaTeX table cells."""
    if pd.isna(value):
        text = ""
    else:
        text = str(value)

    text = _sanitize_latex_text(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile("|".join(re.escape(key) for key in replacements))
    return pattern.sub(lambda match: replacements[match.group(0)], text)


def _dataframe_to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    *,
    size_command: str = r"\small",
    use_resizebox: bool = False,
) -> str:
    """Render a dataframe as a LaTeX table environment."""
    safe_df = df.copy()
    for column in safe_df.columns:
        safe_df[column] = safe_df[column].map(_latex_escape)

    column_format = "l" * len(safe_df.columns)
    tabular = safe_df.to_latex(index=False, escape=False, column_format=column_format)

    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        size_command,
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
    ]
    if use_resizebox:
        latex.append(r"\resizebox{\textwidth}{!}{%")
        latex.append(tabular.rstrip())
        latex.append(r"}")
    else:
        latex.append(tabular.rstrip())
    latex.append(r"\end{table}")

    return "\n".join(latex) + "\n"


def _table1_dataframe() -> pd.DataFrame:
    """Return Table 1 as a dataframe."""
    return pd.DataFrame([
        {
            'Family': 'Family A',
            'Type': 'Invariance',
            'Description': 'Feature order perturbations',
            'Equivalence': 'Fully equivalent',
            'Expected Effect': 'Should be invariant'
        },
        {
            'Family': 'Family B',
            'Type': 'Robustness',
            'Description': 'Instruction-template perturbations',
            'Equivalence': 'Semantically equivalent',
            'Expected Effect': 'Should be robust'
        },
        {
            'Family': 'Family C',
            'Type': 'Invariance',
            'Description': 'Structural-format perturbations',
            'Equivalence': 'Fully equivalent',
            'Expected Effect': 'Should be invariant'
        },
        {
            'Family': 'Family D',
            'Type': 'Robustness',
            'Description': 'Numeric-precision perturbations',
            'Equivalence': 'Near-equivalent',
            'Expected Effect': 'Should be robust'
        },
    ])


def _table2_dataframe(results: Dict) -> pd.DataFrame:
    """Return Table 2 as a dataframe."""
    tier_specs = [
        (
            "Tier 1",
            [results.get("tier1_without_context"), results.get("tier1_with_context")],
            "Broad instability screening",
            "Identify unstable patients for deeper review",
        ),
        (
            "Tier 2",
            [results.get("tier2_without_context"), results.get("tier2_with_context")],
            "Boundary-focused deep analysis",
            "Stress-test high-uncertainty cases",
        ),
    ]

    rows = []
    for tier_label, variants, purpose, goal in tier_specs:
        available_variants = [variant for variant in variants if variant]
        if not available_variants:
            continue
        rows.append(
            _infer_design_row(
                available_variants[0],
                tier_label=tier_label,
                purpose=purpose,
                goal=goal,
                context_count=len(available_variants),
            )
        )

    return pd.DataFrame(rows)


def _table3_dataframe(results: Dict) -> pd.DataFrame:
    """Return Table 3 as a dataframe."""
    metrics_data = []
    for context_label, result_key in [
        ('Without Context', 'tier1_without_context'),
        ('With Context', 'tier1_with_context'),
    ]:
        if results[result_key] is None:
            continue

        data = results[result_key]['summary'] if 'summary' in results[result_key] else {}
        for family_name in ANALYSIS_FAMILY_ORDER:
            if family_name in data:
                metrics = data[family_name]
                metrics_data.append({
                    'Condition': context_label,
                    'Perturbation Family': _get_family_label(family_name),
                    'Label Flip Rate': f"{metrics.get('flip_rate', 0):.3f}",
                    'Mean Conf Diff': f"{metrics.get('mean_conf_diff', 0):.2f}",
                    'Total Perturbs': metrics.get('total_perturbations', 0),
                    'Label Flips': metrics.get('label_flips', 0),
                })
    return pd.DataFrame(metrics_data)


def _table4_dataframe(results: Dict) -> pd.DataFrame:
    """Return Table 4 as a dataframe."""
    prediction_df = _build_prediction_dataframe(results)
    if prediction_df.empty:
        return pd.DataFrame()

    analysis_df = prediction_df[~prediction_df["is_baseline_call"]].copy()
    factor_order = ["patient_idx", "context", "family", "family_instance", "replicate_id"]
    factor_labels = {
        "patient_idx": "Patient identity",
        "context": "Retrieval context",
        "family": "Perturbation family",
        "family_instance": "Instance within family",
        "replicate_id": "Stochastic replicate",
        "Residual": "Residual",
    }

    decompositions = {}
    for tier_label in ["Tier 1", "Tier 2"]:
        tier_df = analysis_df[analysis_df["tier"] == tier_label].copy()
        if tier_df.empty:
            continue
        decompositions[tier_label] = {
            "label_flip": _hierarchical_variance_decomposition(tier_df, "label_flip", factor_order),
            "conf_delta": _hierarchical_variance_decomposition(tier_df, "conf_delta", factor_order),
        }

    rows = []
    for factor_name in factor_order + ["Residual"]:
        row = {"Source": factor_labels[factor_name]}
        for tier_label in ["Tier 1", "Tier 2"]:
            if tier_label not in decompositions:
                row[f"{tier_label} Flip %"] = "N/A"
                row[f"{tier_label} ConfDelta %"] = "N/A"
            else:
                row[f"{tier_label} Flip %"] = f"{100 * decompositions[tier_label]['label_flip'][factor_name]:.1f}%"
                row[f"{tier_label} ConfDelta %"] = f"{100 * decompositions[tier_label]['conf_delta'][factor_name]:.1f}%"
        rows.append(row)

    return pd.DataFrame(rows)


def _table5_dataframe(results: Dict) -> pd.DataFrame:
    """Return Table 5 as a dataframe."""
    patient_df = _build_patient_metric_dataframe(results)
    if patient_df.empty:
        return pd.DataFrame()

    patient_df = patient_df.dropna(subset=["error", "correct"]).copy()
    if patient_df.empty:
        return pd.DataFrame()

    baselines = [
        ("Perturbation flip rate", "flip_rate"),
        ("Boundary uncertainty", "boundary_uncertainty"),
        ("Confidence instability", "confidence_instability"),
        ("Semantic drift", "semantic_drift"),
    ]

    rows = []
    for tier_label in ["Tier 1", "Tier 2"]:
        for context_label in ["Without Context", "With Context"]:
            split_df = patient_df[
                (patient_df["tier"] == tier_label) &
                (patient_df["context"] == context_label)
            ].copy()

            if split_df.empty:
                continue

            for baseline_name, score_col in baselines:
                metrics = _evaluate_deferral_baseline(split_df, score_col)
                rows.append({
                    "Tier": tier_label,
                    "Context": context_label,
                    "Baseline": baseline_name,
                    "N": len(split_df),
                    "Errors": int(split_df["error"].sum()),
                    "AUROC(error)": f"{metrics['AUROC']:.3f}" if not np.isnan(metrics["AUROC"]) else "N/A",
                    "AURC": f"{metrics['AURC']:.3f}" if not np.isnan(metrics["AURC"]) else "N/A",
                    "Error Recall@20%": f"{metrics['Error Recall@20%']:.1%}" if not np.isnan(metrics["Error Recall@20%"]) else "N/A",
                    "Retained Acc@80%": f"{metrics['Retained Acc@80%']:.1%}" if not np.isnan(metrics["Retained Acc@80%"]) else "N/A",
                })

    return pd.DataFrame(rows)


def _table6_dataframe(results: Dict) -> pd.DataFrame:
    """Return Figure 3 operating-point risks for 100%, 90%, and 80% retention."""
    patient_df = _build_patient_metric_dataframe(results)
    patient_df = patient_df[
        (patient_df["tier"] == "Tier 1") &
        patient_df["error"].notna()
    ].copy()

    if patient_df.empty:
        return pd.DataFrame()

    baselines = [
        ("Perturbation flip rate", "flip_rate"),
        ("Boundary uncertainty", "boundary_uncertainty"),
    ]
    retention_targets = [1.0, 0.9, 0.8]

    rows = []
    for context_label in ["Without Context", "With Context"]:
        split_df = patient_df[patient_df["context"] == context_label].copy()
        if split_df.empty:
            continue

        for baseline_name, score_col in baselines:
            row = {
                "Context": context_label,
                "Baseline": baseline_name,
                "N": len(split_df),
            }
            for target in retention_targets:
                metrics = _risk_at_retention(split_df, score_col, target)
                target_label = f"{int(round(target * 100))}%"
                row[f"Risk@{target_label}"] = f"{metrics['risk']:.1%}" if not np.isnan(metrics["risk"]) else "N/A"
                row[f"n@{target_label}"] = int(metrics["retained_n"])
            rows.append(row)

    return pd.DataFrame(rows)


def generate_latex_tables(results: Dict) -> str:
    """Generate all analysis tables as a LaTeX snippet file."""
    sections = [
        "% Requires: \\usepackage{booktabs}",
        "% Optional for wide tables: \\usepackage{graphicx}",
        "",
        _dataframe_to_latex_table(
            _table1_dataframe(),
            "Perturbation taxonomy used in the analysis.",
            "tab:perturbation-taxonomy",
        ),
        _dataframe_to_latex_table(
            _table2_dataframe(results),
            "Experimental design inferred from the saved Tier 1 and Tier 2 outputs.",
            "tab:experimental-design-matrix",
            use_resizebox=True,
        ),
        _dataframe_to_latex_table(
            _table3_dataframe(results),
            "Instability metrics by perturbation family and retrieval condition for Tier 1.",
            "tab:instability-metrics-by-family",
            use_resizebox=True,
        ),
        _dataframe_to_latex_table(
            _table4_dataframe(results),
            "Variance decomposition of instability signals across patient, retrieval, family, instance, and replicate factors.",
            "tab:variance-decomposition",
            use_resizebox=True,
        ),
        _dataframe_to_latex_table(
            _table5_dataframe(results),
            "Deferral-baseline comparison against canonical prediction error on Tier 1 and Tier 2 cohorts.",
            "tab:deferral-baseline-comparison",
            size_command=r"\scriptsize",
            use_resizebox=True,
        ),
        _dataframe_to_latex_table(
            _table6_dataframe(results),
            "Figure 3 operating points for Tier 1 retained-risk curves at 100%, 90%, and 80% patient retention. The reported patient counts are the nearest achievable discrete operating points.",
            "tab:figure3-operating-points",
            size_command=r"\scriptsize",
            use_resizebox=True,
        ),
    ]
    return "\n".join(sections).strip() + "\n"


def generate_latex_figures() -> str:
    """Generate LaTeX figure snippets with explanatory captions."""
    figures = [
        (
            "figure_3_risk_coverage.png",
            "Empirical risk-coverage curves for Tier 1 deferral scores. Panel A shows the without-context setting and Panel B the with-context setting. The orange curve ranks patients by perturbation flip rate and the green curve ranks them by boundary uncertainty. The x-axis reports the percentage of retained patients after deferring the highest-score cases first, and the y-axis reports the retained-patient error rate. Lower curves indicate better selective prediction.",
            "fig:tier1-risk-coverage",
        ),
        (
            "figure_4_heatmap_without_context.png",
            "Tier 1 without-context instability heatmap for the most unstable patients. Rows are sorted by patient index, columns correspond to the four paper families, and each cell reports the patient-specific label flip rate. Darker cells indicate stronger brittleness, while blank annotations denote zero observed flips.",
            "fig:tier1-instability-heatmap-without-context",
        ),
        (
            "figure_4_heatmap_with_context.png",
            "Tier 1 with-context instability heatmap for the most unstable patients. Rows are sorted by patient index, columns correspond to the four paper families, and each cell reports the patient-specific label flip rate. Darker cells indicate stronger brittleness, while blank annotations denote zero observed flips.",
            "fig:tier1-instability-heatmap-with-context",
        ),
        (
            "figure_5_semantic_drift_embeddings.png",
            "Semantic-drift map for the most unstable Tier 2 patients. Panel A shows the without-context setting and Panel B the with-context setting. Each black marker is the patient-level canonical explanation centroid, each colored marker is the centroid for one perturbation family, and colored line segments connect family centroids back to the canonical anchor for the same patient. Greater displacement indicates stronger semantic drift in the explanation space.",
            "fig:tier2-semantic-drift",
        ),
        (
            "figure_6_variance_attribution.png",
            "Family-level instability attribution for Tier 1 shown as a ranked lollipop chart. The horizontal axis reports label flip rate, so farther-right points indicate perturbation families that more often change the final prediction.",
            "fig:tier1-family-attribution",
        ),
    ]

    parts = ["% Requires: \\usepackage{graphicx}", ""]
    for filename, caption, label in figures:
        parts.extend([
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=\textwidth]{{outputs/comprehensive_analysis/{filename}}}",
            rf"\caption{{{_latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
            "",
        ])
    return "\n".join(parts).strip() + "\n"


def generate_table1_taxonomy() -> str:
    """Generate Table 1: Perturbation Taxonomy"""
    
    taxonomy = {
        'A_order': {
            'family': 'Feature Order',
            'type': 'Invariance',
            'description': 'Shuffle key-value pair order',
            'semantic_equivalence': 'Fully equivalent',
            'test_type': 'Should be invariant'
        },
        'B_instruction': {
            'family': 'Instruction Template',
            'type': 'Robustness',
            'description': 'Vary instruction phrasing',
            'semantic_equivalence': 'Semantically equivalent',
            'test_type': 'Should be robust'
        },
        'C_json': {
            'family': 'JSON Format',
            'type': 'Invariance',
            'description': 'Render features as JSON',
            'semantic_equivalence': 'Fully equivalent',
            'test_type': 'Should be invariant'
        },
        'C_csv': {
            'family': 'CSV Format',
            'type': 'Invariance',
            'description': 'Render features as comma-separated values',
            'semantic_equivalence': 'Fully equivalent',
            'test_type': 'Should be invariant'
        },
        'D_numeric_precision': {
            'family': 'Numeric Precision',
            'type': 'Robustness',
            'description': 'Vary decimal places (int vs float)',
            'semantic_equivalence': 'Near-equivalent',
            'test_type': 'Should be robust'
        },
        'E_delimiters': {
            'family': 'Delimiters',
            'type': 'Invariance',
            'description': 'Use different separators (semicolon, pipe, newline)',
            'semantic_equivalence': 'Fully equivalent',
            'test_type': 'Should be invariant'
        },
        'I_noise': {
            'family': 'Numeric Noise',
            'type': 'Robustness',
            'description': 'Add ±5% random noise to values',
            'semantic_equivalence': 'Semantically equivalent',
            'test_type': 'Should be robust'
        },
    }
    
    table_text = "\n" + "="*120 + "\n"
    table_text += "TABLE 1: PERTURBATION TAXONOMY\n"
    table_text += "="*120 + "\n\n"
    taxonomy_rows = _table1_dataframe().rename(columns={
        "Family": "family",
        "Type": "type",
        "Description": "description",
        "Equivalence": "semantic_equivalence",
        "Expected Effect": "test_type",
    }).to_dict("records")
    
    table_text += f"{'Family':<25} {'Type':<15} {'Description':<40} {'Equivalence':<20} {'Expected Effect':<15}\n"
    table_text += "-"*120 + "\n"
    
    for info in taxonomy_rows:
        table_text += (
            f"{info['family']:<25} "
            f"{info['type']:<15} "
            f"{info['description']:<40} "
            f"{info['semantic_equivalence']:<20} "
            f"{info['test_type']}\n"
        )
    
    table_text += "\n"
    return table_text


def generate_table2_design_matrix(results: Dict) -> str:
    """Generate Table 2: Experimental Design Matrix"""
    
    table_text = "\n" + "="*120 + "\n"
    table_text += "TABLE 2: EXPERIMENTAL DESIGN MATRIX\n"
    table_text += "="*120 + "\n\n"
    
    tier_specs = [
        (
            "Tier 1",
            [results.get("tier1_without_context"), results.get("tier1_with_context")],
            "Broad instability screening",
            "Identify unstable patients for deeper review",
        ),
        (
            "Tier 2",
            [results.get("tier2_without_context"), results.get("tier2_with_context")],
            "Boundary-focused deep analysis",
            "Stress-test high-uncertainty cases",
        ),
    ]

    design_rows = []
    for tier_label, variants, purpose, goal in tier_specs:
        available_variants = [variant for variant in variants if variant]
        if not available_variants:
            continue
        design_rows.append(
            _infer_design_row(
                available_variants[0],
                tier_label=tier_label,
                purpose=purpose,
                goal=goal,
                context_count=len(available_variants),
            )
        )

    if design_rows:
        df = pd.DataFrame(design_rows)
        table_text += df.to_string(index=False)
    else:
        table_text += "No results available yet.\n"
    table_text += "\n\n"
    
    # Add context definition
    table_text += "CONTEXT CONDITIONS:\n"
    table_text += "  - With Context: Patient features augmented with retrieved similar cases\n"
    table_text += "  - Without Context: Patient features only (baseline)\n\n"
    
    return table_text


def generate_table3_instability_metrics(results: Dict) -> str:
    """Generate Table 3: Instability metrics by perturbation family and retrieval"""
    
    table_text = "\n" + "="*120 + "\n"
    table_text += "TABLE 3: INSTABILITY METRICS BY PERTURBATION FAMILY AND RETRIEVAL CONDITION\n"
    table_text += "="*120 + "\n\n"
    
    df = _table3_dataframe(results)
    
    if not df.empty:
        table_text += df.to_string(index=False)
    else:
        table_text += "No results available yet.\n"
    
    table_text += "\n\n"
    return table_text


def generate_table4_variance_decomposition(results: Dict) -> str:
    """Generate Table 4: Variance decomposition across experimental factors."""
    table_text = "\n" + "="*120 + "\n"
    table_text += "TABLE 4: MIXED-EFFECTS-STYLE VARIANCE DECOMPOSITION OF INSTABILITY SIGNALS\n"
    table_text += "="*120 + "\n\n"

    prediction_df = _build_prediction_dataframe(results)
    if prediction_df.empty:
        return table_text + "No prediction-level results available.\n\n"

    analysis_df = prediction_df[~prediction_df["is_baseline_call"]].copy()
    factor_order = ["patient_idx", "context", "family", "family_instance", "replicate_id"]
    factor_labels = {
        "patient_idx": "Patient identity",
        "context": "Retrieval context",
        "family": "Perturbation family",
        "family_instance": "Instance within family",
        "replicate_id": "Stochastic replicate",
        "Residual": "Residual",
    }

    decompositions = {}
    observation_counts = {}
    for tier_label in ["Tier 1", "Tier 2"]:
        tier_df = analysis_df[analysis_df["tier"] == tier_label].copy()
        if tier_df.empty:
            continue
        observation_counts[tier_label] = len(tier_df)
        decompositions[tier_label] = {
            "label_flip": _hierarchical_variance_decomposition(tier_df, "label_flip", factor_order),
            "conf_delta": _hierarchical_variance_decomposition(tier_df, "conf_delta", factor_order),
        }

    if not decompositions:
        return table_text + "No variance-decomposition data available.\n\n"

    rows = []
    ordered_factors = factor_order + ["Residual"]
    for factor_name in ordered_factors:
        row = {"Source": factor_labels[factor_name]}
        for tier_label in ["Tier 1", "Tier 2"]:
            if tier_label not in decompositions:
                row[f"{tier_label} Flip %"] = "N/A"
                row[f"{tier_label} ConfDelta %"] = "N/A"
                continue
            row[f"{tier_label} Flip %"] = f"{100 * decompositions[tier_label]['label_flip'][factor_name]:.1f}%"
            row[f"{tier_label} ConfDelta %"] = f"{100 * decompositions[tier_label]['conf_delta'][factor_name]:.1f}%"
        rows.append(row)

    table_text += pd.DataFrame(rows).to_string(index=False)
    table_text += "\n\n"
    table_text += "METHOD NOTE:\n"
    table_text += "  Sequential grouped-residual decomposition over patient -> context -> family -> instance -> replicate.\n"
    table_text += "  Percentages sum to ~100% within each target column and quantify where instability signal is concentrated.\n"
    for tier_label in ["Tier 1", "Tier 2"]:
        if tier_label in observation_counts:
            table_text += f"  {tier_label} observations analysed: {observation_counts[tier_label]:,}\n"
    table_text += "\n"

    return table_text


def generate_table5_deferral_results(results: Dict) -> str:
    """Generate Table 5: Deferral baselines benchmarked against ground truth."""
    table_text = "\n" + "="*120 + "\n"
    table_text += "TABLE 5: DEFERRAL RESULTS ACROSS AVAILABLE BASELINES\n"
    table_text += "="*120 + "\n\n"

    patient_df = _build_patient_metric_dataframe(results)
    if patient_df.empty:
        return table_text + "No patient-level results available.\n\n"

    patient_df = patient_df.dropna(subset=["error", "correct"]).copy()
    if patient_df.empty:
        return table_text + "Ground-truth labels were not available, so deferral metrics could not be computed.\n\n"

    baselines = [
        ("Perturbation flip rate", "flip_rate"),
        ("Boundary uncertainty", "boundary_uncertainty"),
        ("Confidence instability", "confidence_instability"),
        ("Semantic drift", "semantic_drift"),
    ]

    rows = []
    for tier_label in ["Tier 1", "Tier 2"]:
        for context_label in ["Without Context", "With Context"]:
            split_df = patient_df[
                (patient_df["tier"] == tier_label) &
                (patient_df["context"] == context_label)
            ].copy()

            if split_df.empty:
                continue

            for baseline_name, score_col in baselines:
                metrics = _evaluate_deferral_baseline(split_df, score_col)
                rows.append({
                    "Tier": tier_label,
                    "Context": context_label,
                    "Baseline": baseline_name,
                    "N": len(split_df),
                    "Errors": int(split_df["error"].sum()),
                    "AUROC(error)": f"{metrics['AUROC']:.3f}" if not np.isnan(metrics["AUROC"]) else "N/A",
                    "AURC": f"{metrics['AURC']:.3f}" if not np.isnan(metrics["AURC"]) else "N/A",
                    "Error Recall@20%": f"{metrics['Error Recall@20%']:.1%}" if not np.isnan(metrics["Error Recall@20%"]) else "N/A",
                    "Retained Acc@80%": f"{metrics['Retained Acc@80%']:.1%}" if not np.isnan(metrics["Retained Acc@80%"]) else "N/A",
                })

    if rows:
        table_text += pd.DataFrame(rows).to_string(index=False)
    else:
        table_text += "No evaluable deferral rows available.\n"

    table_text += "\n\n"
    table_text += "EVALUATION NOTE:\n"
    table_text += "  Higher AUROC/error-recall and lower AURC are better. Scores are benchmarked against canonical prediction errors\n"
    table_text += "  using `data/actual.csv`. Tier 2 is a hard-case subset selected for uncertainty, so Tier 1 remains the unbiased cohort.\n\n"

    return table_text


def generate_table6_figure3_operating_points(results: Dict) -> str:
    """Generate Table 6: Figure 3 operating points."""
    table_text = "\n" + "="*120 + "\n"
    table_text += "TABLE 6: FIGURE 3 RISK AT 100%, 90%, AND 80% RETENTION\n"
    table_text += "="*120 + "\n\n"

    df = _table6_dataframe(results)
    if df.empty:
        return table_text + "No tier 1 ground-truth-aligned data available.\n\n"

    table_text += df.to_string(index=False)
    table_text += "\n\n"
    table_text += "NOTE:\n"
    table_text += "  Risk is the error rate among retained patients after deferring the highest-score cases first.\n"
    table_text += "  Patient counts are the nearest achievable discrete operating points for each retention target.\n\n"
    return table_text


def generate_figure5_semantic_drift(results: Dict) -> Tuple[str, plt.Figure]:
    """Generate Figure 5: 2D embedding projection of unstable explanation clusters."""
    figure_text = "Generating Figure 5: Semantic-Drift Embedding Visualization...\n"

    rows = []
    for result_key in ["tier2_without_context", "tier2_with_context"]:
        result_data = results.get(result_key)
        if not result_data:
            continue

        tier, context = _get_result_metadata(result_key)
        patient_profiles = result_data.get("patient_profiles", {})
        ranked_patients = sorted(
            result_data.get("results_by_patient", {}).items(),
            key=lambda item: _lookup_patient_profile(patient_profiles, int(item[0].split("_")[1])).get("overall_semantic_drift", 0.0),
            reverse=True,
        )[:5]

        for patient_key, patient_data in ranked_patients:
            patient_idx = int(patient_key.split("_")[1])
            canonical_entries = _flatten_result_entries(patient_data.get("canonical"))
            for entry in canonical_entries:
                explanation = entry.get("explanation", "").strip()
                if explanation:
                    rows.append({
                        "context": context,
                        "patient_idx": patient_idx,
                        "family_label": "Canonical",
                        "explanation": explanation,
                    })

            for family_name in ANALYSIS_FAMILY_ORDER:
                if family_name not in patient_data:
                    continue
                for entry in _flatten_result_entries(patient_data[family_name]):
                    explanation = entry.get("explanation", "").strip()
                    if explanation:
                        rows.append({
                            "context": context,
                            "patient_idx": patient_idx,
                            "family_label": _get_family_label(family_name),
                            "explanation": explanation,
                        })

    explanation_df = pd.DataFrame(rows).drop_duplicates(
        subset=["context", "patient_idx", "family_label", "explanation"]
    ).reset_index(drop=True)
    if explanation_df.empty:
        return figure_text + "No explanation text available.\n", None

    embeddings = None
    embedding_method = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        embeddings = model.encode(explanation_df["explanation"].tolist(), show_progress_bar=False)
        embedding_method = "sentence-transformer"
    except Exception:
        vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
        embeddings = vectorizer.fit_transform(explanation_df["explanation"].tolist()).toarray()
        embedding_method = "tf-idf"

    embedding_matrix = np.asarray(embeddings)
    centroid_rows = []
    for (context, patient_idx, family_label), group in explanation_df.groupby(["context", "patient_idx", "family_label"]):
        centroid = embedding_matrix[group.index].mean(axis=0)
        centroid_rows.append({
            "context": context,
            "patient_idx": patient_idx,
            "family_label": family_label,
            "embedding": centroid,
        })

    centroid_df = pd.DataFrame(centroid_rows)
    pca = PCA(n_components=2, random_state=0)
    projection = pca.fit_transform(np.vstack(centroid_df["embedding"].to_numpy()))
    centroid_df["x"] = projection[:, 0]
    centroid_df["y"] = projection[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), sharex=True, sharey=True)
    for panel_label, ax, context_label in zip(["A", "B"], axes, ["Without Context", "With Context"]):
        context_df = centroid_df[centroid_df["context"] == context_label].copy()
        if context_df.empty:
            ax.text(0.5, 0.5, f"No {context_label} data", ha="center", va="center", transform=ax.transAxes)
            _style_publication_axes(ax)
            continue

        canonical_df = context_df[context_df["family_label"] == "Canonical"].set_index("patient_idx")
        for family_label in ["Family A", "Family B", "Family C", "Family D"]:
            fam_df = context_df[context_df["family_label"] == family_label]
            if fam_df.empty:
                continue
            for _, row in fam_df.iterrows():
                if row["patient_idx"] in canonical_df.index:
                    anchor = canonical_df.loc[row["patient_idx"]]
                    ax.plot(
                        [anchor["x"], row["x"]],
                        [anchor["y"], row["y"]],
                        color=FAMILY_COLORS[family_label],
                        linewidth=1.2,
                        alpha=0.45,
                        zorder=1,
                    )
            ax.scatter(
                fam_df["x"],
                fam_df["y"],
                s=78,
                marker="o",
                color=FAMILY_COLORS[family_label],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.9,
                label=family_label,
                zorder=2,
            )

        canonical_plot_df = context_df[context_df["family_label"] == "Canonical"]
        ax.scatter(
            canonical_plot_df["x"],
            canonical_plot_df["y"],
            s=115,
            marker="X",
            color=FAMILY_COLORS["Canonical"],
            edgecolors="white",
            linewidths=1.0,
            label="Canonical",
            zorder=3,
        )
        for _, row in canonical_plot_df.iterrows():
            ax.text(
                row["x"] + 0.01,
                row["y"] + 0.01,
                f"P{int(row['patient_idx'])}",
                fontsize=8.5,
                fontweight="bold",
                color="#111111",
            )

        _add_panel_label(ax, panel_label)
        ax.set_title(context_label, fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", fontsize=11, fontweight="bold")
        _style_publication_axes(ax)

    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", fontsize=11, fontweight="bold")
    legend_handles = [
        Line2D([0], [0], marker="X" if label == "Canonical" else "o", color="w",
               markerfacecolor=FAMILY_COLORS[label], markeredgecolor="white",
               markersize=9 if label == "Canonical" else 7, linewidth=0, label=label)
        for label in ["Canonical", "Family A", "Family B", "Family C", "Family D"]
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=10,
    )
    fig.subplots_adjust(top=0.80, wspace=0.08)

    figure_text += (
        f"Patient-level family centroids are projected to two dimensions with PCA using {embedding_method} features.\n"
        "Panel A shows the without-context setting and Panel B the with-context setting.\n"
        "Black X markers are canonical explanation centroids; colored circles are family-specific centroids, and line segments show the drift away from the canonical explanation for the same patient.\n"
    )
    return figure_text, fig


def generate_figure1_architecture() -> str:
    """Return the original report text for Figure 1 without writing image files."""
    # figure_text = "\n" + "="*120 + "\n"
    # figure_text += "FIGURE 1: SYSTEM ARCHITECTURE - PERTURBATION PIPELINE\n"
    # figure_text += "="*120 + "\n\n"
    # figure_text += (
    #     "Patient features -> family-level prompt perturbations (Family A to Family D) -> optional retrieval context -> "
    #     "LLM inference -> stability analysis.\n\n"
    #     "Family A: feature order perturbations.\n"
    #     "Family B: instruction-template perturbations.\n"
    #     "Family C: structural-format perturbations.\n"
    #     "Family D: numeric-precision perturbations.\n\n"
    #     "The architecture figure is conceptual: it shows the experiment flow rather than a measured quantity.\n\n"
    # )
    # return figure_text
    
    figure_text = "\n" + "="*120 + "\n"
    figure_text += "FIGURE 1: SYSTEM ARCHITECTURE - PERTURBATION PIPELINE\n"
    figure_text += "="*120 + "\n\n"
    
    architecture = """
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         PATIENT DATA PIPELINE                               │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    ┌──────────────────┐
    │  Patient Data    │
    │  (Raw Features)  │
    └────────┬─────────┘
             │
             v
    ┌──────────────────────────────────────┐
    │    PERTURBATION APPLICATION          │
    ├──────────────────────────────────────┤
    │ • Shuffle feature order (A_order)    │
    │ • Vary instructions (B_instruction)  │
    │ • Format as JSON/CSV (C_json/csv)    │
    │ • Change precision (D_precision)     │
    │ • Use different delimiters (E_delim) │
    │ • Add numeric noise (I_noise)        │
    └────────┬─────────────────────────────┘
             │
             v
    ┌──────────────────────────────────────┐
    │   RETRIEVAL CONTEXT (Optional)       │
    │   "With Context" vs "Without Context"│
    └────────┬─────────────────────────────┘
             │
             v
    ┌──────────────────────────────────────┐
    │    LLM INFERENCE PIPELINE            │
    │  (Azure OpenAI / Claude / Gemini)    │
    ├──────────────────────────────────────┤
    │ Input:  Perturbed prompt with context│
    │ Output: Prediction + Confidence      │
    └────────┬─────────────────────────────┘
             │
             v
    ┌──────────────────────────────────────┐
    │   RESULT AGGREGATION & ANALYSIS      │
    ├──────────────────────────────────────┤
    │ • Label consistency across variants  │
    │ • Confidence variance                │
    │ • Instability metrics                │
    │ • Patient stratification             │
    └──────────────────────────────────────┘
    
    """
    
    figure_text += architecture
    figure_text += "\n"
    return figure_text


def generate_figure2_taxonomy() -> str:
    """Return the original report text for Figure 2 without writing image files."""
    # figure_text = "\n" + "="*120 + "\n"
    # figure_text += "FIGURE 2: PERTURBATION TAXONOMY - INVARIANCE vs ROBUSTNESS TESTS\n"
    # figure_text += "="*120 + "\n\n"
    # figure_text += (
    #     "Invariance families preserve the patient facts exactly and should not change the prediction.\n"
    #     "  Family A: feature order.\n"
    #     "  Family C: structural format.\n\n"
    #     "Robustness families make small presentation changes and should not materially change the prediction.\n"
    #     "  Family B: instruction template.\n"
    #     "  Family D: numeric precision.\n\n"
    #     "If invariance fails, the model is format-sensitive. If robustness fails, the model is brittle to small prompt changes.\n\n"
    # )
    # return figure_text
    
    figure_text = "\n" + "="*120 + "\n"
    figure_text += "FIGURE 2: PERTURBATION TAXONOMY - INVARIANCE vs ROBUSTNESS TESTS\n"
    figure_text += "="*120 + "\n\n"
    
    taxonomy_visual = """
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      PERTURBATION TAXONOMY                              │
    └─────────────────────────────────────────────────────────────────────────┘
    
    
                         INVARIANCE TESTS                   ROBUSTNESS TESTS
          (Should NOT change predictions)         (Should be somewhat robust)
    
    ┌─────────────────────────────────────┐   ┌────────────────────────────────┐
    │  Information-Preserving Changes     │   │  Semantic-Tolerant Changes     │
    ├─────────────────────────────────────┤   ├────────────────────────────────┤
    │                                     │   │                                │
    │ • A_order                          │   │ • B_instruction                │
    │   Shuffle feature order            │   │   Vary instruction phrasing    │
    │   ✓ Still has same information     │   │   ~ Slightly different meaning │
    │   → Should be INVARIANT            │   │   → Should be ROBUST           │
    │                                     │   │                                │
    │ • C_json / C_csv / E_delimiters    │   │ • D_numeric_precision          │
    │   Change representation format     │   │   Round values differently     │
    │   ✓ Still has same information     │   │   ~ Approximate equivalence    │
    │   → Should be INVARIANT            │   │   → Should be ROBUST           │
    │                                     │   │                                │
    │                                     │   │ • I_noise                      │
    │                                     │   │   Add ±5% random variation     │
    │                                     │   │   ~ Realistic measurement err  │
    │                                     │   │   → Should be ROBUST           │
    │                                     │   │                                │
    └─────────────────────────────────────┘   └────────────────────────────────┘
    
                        WHAT WE'RE TESTING:
    
    INVARIANCE: "Does the LLM treat semantically identical inputs identically?"
                If FAIL → System lacks true understanding (format-dependent)
    
    ROBUSTNESS: "Does the LLM handle reasonable variations gracefully?"
                If FAIL → System is brittle (overly sensitive to wording/noise)
    
    """
    
    figure_text += taxonomy_visual
    figure_text += "\n"
    return figure_text


def _build_figure4_dataframe(result_data: Dict) -> pd.DataFrame:
    """Build the patient-family flip-rate matrix used by Figure 4."""
    flip_data = defaultdict(lambda: defaultdict(list))

    results_by_patient = result_data.get("results_by_patient", {})
    for patient_key, patient_data in results_by_patient.items():
        patient_idx = int(patient_key.split("_")[1])
        canonical_label = _extract_canonical_label(patient_data)

        for family_name in ANALYSIS_FAMILY_ORDER:
            if family_name in patient_data:
                perturb_results = _flatten_result_entries(patient_data[family_name])
                flips = sum(1 for r in perturb_results if r.get("label") != canonical_label)
                flip_rate = flips / len(perturb_results) if perturb_results else 0
                flip_data[patient_idx][_get_family_label(family_name)] = flip_rate

    df_flip = pd.DataFrame(flip_data).T.fillna(0)
    if df_flip.empty:
        return df_flip

    ordered_columns = [_get_family_label(family_name) for family_name in ANALYSIS_FAMILY_ORDER]
    df_flip = df_flip[[col for col in ordered_columns if col in df_flip.columns]]
    df_flip["Mean Instability"] = df_flip.mean(axis=1)
    df_flip = df_flip.sort_values("Mean Instability", ascending=False).head(15).drop(columns=["Mean Instability"])
    df_flip = df_flip.sort_index()
    df_flip.index = [f"P{int(idx)}" for idx in df_flip.index]
    return df_flip


def _make_figure4_heatmap(df_flip: pd.DataFrame, context_label: str) -> plt.Figure:
    """Render a single Figure 4 heatmap panel for one context setting."""
    annotation = df_flip.copy()
    for column in annotation.columns:
        annotation[column] = annotation[column].map(lambda value: f"{value:.0%}" if value > 0 else "")

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    heatmap = sns.heatmap(
        df_flip,
        cmap=sns.color_palette(["#f8fbff", "#d5e6f7", "#6f9fca", "#123b6d"], as_cmap=True),
        cbar_kws={"label": "Label flip rate", "shrink": 0.92},
        ax=ax,
        vmin=0,
        vmax=max(0.5, float(df_flip.to_numpy().max())),
        linewidths=0.7,
        linecolor="#ffffff",
        annot=annotation,
        fmt="",
        annot_kws={"fontsize": 8.5, "fontweight": "bold"},
    )

    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.outline.set_visible(False)
    ax.set_xlabel("Family", fontsize=11, fontweight="bold")
    ax.set_ylabel("Patient", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    _style_publication_axes(ax, heatmap=True)

    for text in ax.texts:
        try:
            value = float(text.get_text().strip("%")) / 100 if text.get_text() else 0.0
        except ValueError:
            value = 0.0
        text.set_color("white" if value >= 0.45 else "#14324d")

    plt.tight_layout()
    return fig


def generate_figure4_heatmap(results: Dict) -> Tuple[str, Dict[str, plt.Figure]]:
    """Generate separate Figure 4 heatmaps for without-context and with-context Tier 1 results."""
    figure_text = "Generating Figure 4: Patient-by-Perturbation Heatmaps...\n"
    figures = {}

    context_specs = [
        ("Without Context", "tier1_without_context", "figure_4_heatmap_without_context"),
        ("With Context", "tier1_with_context", "figure_4_heatmap_with_context"),
    ]

    for context_label, result_key, figure_stem in context_specs:
        result_data = results.get(result_key)
        if not result_data:
            continue

        df_flip = _build_figure4_dataframe(result_data)
        if df_flip.empty:
            continue

        figures[figure_stem] = _make_figure4_heatmap(df_flip, context_label)

    if not figures:
        return figure_text + "No tier 1 data available.\n", {}

    figure_text += (
        "Two separate heatmaps are produced for Tier 1: one without retrieval context and one with retrieval context.\n"
        "Each heatmap shows label flip rate for the 15 most unstable patients in that setting, with rows sorted by patient index.\n"
        "Darker cells indicate more frequent label changes across Families A to D.\n"
    )
    return figure_text, figures


def generate_figure6_variance_attribution(results: Dict) -> Tuple[str, plt.Figure]:
    """Generate Figure 6: Variance Attribution by Perturbation Family"""
    
    figure_text = "Generating Figure 6: Variance Attribution...\n"
    
    # Use tier 1 without context
    result_data = results.get('tier1_without_context')
    if not result_data:
        return figure_text + "No tier 1 data available.\n", None
    
    # Calculate variance contribution per family
    summary = result_data.get('summary', {})
    
    families = []
    variances = []
    
    for family_name in ANALYSIS_FAMILY_ORDER:
        if family_name in summary:
            metrics = summary[family_name]
            # Use flip rate as proxy for variance contribution
            variance_contrib = metrics.get('flip_rate', 0)
            families.append(_get_family_label(family_name))
            variances.append(variance_contrib)
    
    if not families:
        return figure_text + "No summary data available.\n", None
    
    plot_df = pd.DataFrame({"Family": families, "Instability": variances}).sort_values("Instability", ascending=True)
    y_positions = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.hlines(y_positions, 0, plot_df["Instability"], color="#cfd8e3", linewidth=2.2, zorder=1)
    ax.scatter(
        plot_df["Instability"],
        y_positions,
        s=140,
        color=[FAMILY_COLORS[family] for family in plot_df["Family"]],
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["Family"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Label flip rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlim(0, max(variances) * 1.28 if variances else 0.5)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    for value, y_pos in zip(plot_df["Instability"], y_positions):
        ax.text(value + 0.004, y_pos, f"{value:.1%}", va="center", ha="left", fontsize=10, fontweight="bold", color="#1a1a1a")

    _style_publication_axes(ax)
    plt.tight_layout()

    figure_text += (
        "This ranked lollipop chart summarizes Tier 1 family-level label flip rate.\n"
        "Points farther to the right indicate perturbation families that more often change the final prediction.\n"
    )
    return figure_text, fig


def generate_figure3_risk_coverage(results: Dict) -> Tuple[str, plt.Figure]:
    """Generate Figure 3: Empirical risk-coverage curves from deferral scores."""
    
    figure_text = "Generating Figure 3: Risk-Coverage Curves...\n"

    patient_df = _build_patient_metric_dataframe(results)
    patient_df = patient_df[
        (patient_df["tier"] == "Tier 1") &
        patient_df["error"].notna()
    ].copy()

    if patient_df.empty:
        return figure_text + "No tier 1 ground-truth-aligned data available.\n", None

    baselines = [
        ("Perturbation flip rate", "flip_rate"),
        ("Boundary uncertainty", "boundary_uncertainty"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    max_risk = max(float(patient_df["error"].mean()), 0.0)

    for panel_label, ax, context_label in zip(["A", "B"], axes, ["Without Context", "With Context"]):
        split_df = patient_df[patient_df["context"] == context_label].copy()
        if split_df.empty:
            ax.text(0.5, 0.5, f"No {context_label} data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(context_label, fontsize=12, fontweight="bold")
            _style_publication_axes(ax)
            continue

        for baseline_name, score_col in baselines:
            coverage, risk = _compute_risk_coverage_curve(split_df, score_col)
            if len(coverage) == 0:
                continue
            ax.step(
                coverage,
                risk,
                where="post",
                linewidth=2.4,
                color=PLOT_COLORS[baseline_name],
                label=baseline_name,
            )
            if len(risk):
                max_risk = max(max_risk, float(np.max(risk)))

        _add_panel_label(ax, panel_label)
        ax.set_title(context_label, fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xlabel("Retained patients (%)", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        _style_publication_axes(ax)

    axes[0].set_ylabel("Retained-set error rate", fontsize=11, fontweight="bold")
    y_max = max_risk * 1.08 if max_risk > 0 else 0.05
    for ax in axes:
        ax.set_ylim(0, y_max)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=10)

    figure_text += (
        "Panel A shows the without-context setting and Panel B the with-context setting.\n"
        "The orange curve ranks patients by perturbation flip rate and the green curve ranks them by boundary uncertainty.\n"
        "The x-axis is the percentage of retained patients after deferring the highest-score cases first, and the y-axis is the retained-set error rate.\n"
        "Flat segments occur when several deferred patients share the same score or when removing them does not change the retained-set error rate.\n"
    )
    fig.subplots_adjust(top=0.80, wspace=0.08)
    return figure_text, fig


def main():
    """Generate all available tables and figures."""
    
    print("\n" + "="*120)
    print("COMPREHENSIVE ANALYSIS: LLM PERTURBATION STABILITY EXPERIMENTS")
    print("="*120 + "\n")
    
    # Create output directory
    output_dir = Path('outputs/comprehensive_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    print("Loading experimental results...")
    results = load_all_results()
    
    # Generate report text
    report_text = f"\n{'='*120}\n"
    report_text += f"COMPREHENSIVE ANALYSIS REPORT\n"
    report_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_text += f"{'='*120}\n"
    
    # Generate all tables
    print("\nGenerating tables...")
    report_text += generate_table1_taxonomy()
    report_text += generate_table2_design_matrix(results)
    report_text += generate_table3_instability_metrics(results)
    report_text += generate_table4_variance_decomposition(results)
    report_text += generate_table5_deferral_results(results)
    report_text += generate_table6_figure3_operating_points(results)
    
    # Generate all figures (text descriptions + visualizations)
    print("Generating figures...")
    report_text += generate_figure1_architecture()
    report_text += generate_figure2_taxonomy()
    
    # Figures 3, 4, 5, 6 with visualizations
    fig3_text, fig3 = generate_figure3_risk_coverage(results)
    report_text += "\n" + "="*120 + "\n"
    report_text += "FIGURE 3: RISK-COVERAGE CURVES\n"
    report_text += "="*120 + "\n"
    report_text += fig3_text
    if fig3:
        report_text += "Saved to: comprehensive_analysis/figure_3_risk_coverage.png\n"
    
    fig4_text, fig4_figures = generate_figure4_heatmap(results)
    report_text += "\n" + "="*120 + "\n"
    report_text += "FIGURE 4: PATIENT-BY-PERTURBATION INSTABILITY HEATMAP\n"
    report_text += "="*120 + "\n"
    report_text += fig4_text
    if fig4_figures:
        report_text += "Saved to: comprehensive_analysis/figure_4_heatmap_without_context.png\n"
        report_text += "Saved to: comprehensive_analysis/figure_4_heatmap_with_context.png\n"

    fig5_text, fig5 = generate_figure5_semantic_drift(results)
    report_text += "\n" + "="*120 + "\n"
    report_text += "FIGURE 5: SEMANTIC-DRIFT EMBEDDING VISUALIZATION\n"
    report_text += "="*120 + "\n"
    report_text += fig5_text
    if fig5:
        report_text += "Saved to: comprehensive_analysis/figure_5_semantic_drift_embeddings.png\n"
    
    fig6_text, fig6 = generate_figure6_variance_attribution(results)
    report_text += "\n" + "="*120 + "\n"
    report_text += "FIGURE 6: VARIANCE ATTRIBUTION BY PERTURBATION FAMILY\n"
    report_text += "="*120 + "\n"
    report_text += fig6_text
    if fig6:
        report_text += "Saved to: comprehensive_analysis/figure_6_variance_attribution.png\n"
    
    # Save report
    report_path = output_dir / 'FULL_ANALYSIS_REPORT.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    latex_tables_path = output_dir / 'TABLES.tex'
    with open(latex_tables_path, 'w', encoding='utf-8') as f:
        f.write(generate_latex_tables(results))

    latex_figures_path = output_dir / 'FIGURES.tex'
    with open(latex_figures_path, 'w', encoding='utf-8') as f:
        f.write(generate_latex_figures())

    print(f"\n[OK] Report saved to: {report_path}")
    print(f"[OK] LaTeX tables saved to: {latex_tables_path}")
    print(f"[OK] LaTeX figures saved to: {latex_figures_path}")
    
    # Save figures
    if fig3:
        _save_figure_bundle(fig3, output_dir, "figure_3_risk_coverage")
        plt.close(fig3)
    
    for figure_stem, fig4 in fig4_figures.items():
        _save_figure_bundle(fig4, output_dir, figure_stem)
        plt.close(fig4)

    if fig5:
        _save_figure_bundle(fig5, output_dir, "figure_5_semantic_drift_embeddings")
        plt.close(fig5)
    
    if fig6:
        _save_figure_bundle(fig6, output_dir, "figure_6_variance_attribution")
        plt.close(fig6)
    
    print(f"\n{'='*120}")
    print(f"Analysis complete! All results saved to: {output_dir}")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
