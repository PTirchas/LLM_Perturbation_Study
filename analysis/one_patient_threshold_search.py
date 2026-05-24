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
One-patient stress test for the perturbation-threshold claim.

This is a small, cost-controlled script for testing the claim:

    "After perturbation magnitude exceeds T, the probability assigned to the
    true outcome approaches 0."

In the PDF this is the threshold-collapse claim from Theorem 3, not the
invariance claim from Theorem 1.

By default this script is a dry run and makes no LLM calls. To actually run:

    .venv\\Scripts\\python.exe new\\one_patient_threshold_search.py --run

Recommended first real run:

    .venv\\Scripts\\python.exe new\\one_patient_threshold_search.py --run --patient-index 0 --family counterfactual --variants 3 --replicates 2

Outputs:
    new/one_patient_threshold_calls.csv
    new/one_patient_threshold_summary.csv
    new/one_patient_threshold_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_config, print_config  # noqa: E402
from utils.data import load_data  # noqa: E402
from utils.llm import call_llm, initialize_client  # noqa: E402
from utils.perturbations import canonical_prompt  # noqa: E402


DEFAULT_NOISE_LEVELS = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00]
DEFAULT_MASK_LEVELS = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
DEFAULT_COUNTERFACTUAL_LEVELS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]


def _safe_float(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(clean) / len(clean) if clean else math.nan


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or math.isnan(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _percent(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "N/A"
    return f"{100.0 * float(value):.1f}%"


def _console_safe(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def _load_actual_labels(repo_root: Path) -> Dict[int, int]:
    actual_path = repo_root / "data" / "actual.csv"
    with actual_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    labels = {}
    for fallback_idx, row in enumerate(rows):
        patient_idx = int(row["Patient"]) - 1 if row.get("Patient") else fallback_idx
        labels[patient_idx] = int(row["patient_diagnosis"])
    return labels


def _load_patient(
    repo_root: Path,
    dataset_path: str,
    patient_index: int,
) -> Tuple[Dict[str, float], int, Dict[str, float]]:
    df = load_data(str(repo_root / dataset_path))
    if patient_index < 0 or patient_index >= len(df):
        raise IndexError(f"patient_index {patient_index} is outside dataset size {len(df)}")

    actual_labels = _load_actual_labels(repo_root)
    actual_label = actual_labels[patient_index]

    row = df.iloc[patient_index]
    exclude_cols = {"patient_diagnosis", "Patient", "sample_id"}
    feature_names = [col for col in df.columns if col not in exclude_cols]
    features = {name: float(row[name]) for name in feature_names}

    return features, actual_label, row.to_dict()


def _format_features(features: Dict[str, float]) -> str:
    return "; ".join(f"{key}: {value}" for key, value in features.items())


def _numeric_noise_prompt(features: Dict[str, float], severity: float, seed: int) -> str:
    rng = np.random.default_rng(seed)
    perturbed = {}
    for key, value in features.items():
        multiplier = 1.0 + rng.uniform(-severity, severity)
        new_value = value * multiplier
        if key.lower() == "age":
            new_value = _clamp(new_value, 0.0, 120.0)
        else:
            new_value = max(0.0, new_value)
        perturbed[key] = round(float(new_value), 6)

    return (
        f"Numeric-noise stress level: +/-{severity * 100:.0f}%.\n"
        f"{_format_features(perturbed)}"
    )


def _feature_mask_prompt(features: Dict[str, float], severity: float, seed: int) -> str:
    rng = random.Random(seed)
    keys = list(features.keys())
    drop_count = max(1, min(len(keys) - 1, int(round(len(keys) * severity))))
    dropped = set(rng.sample(keys, drop_count))
    remaining = {key: value for key, value in features.items() if key not in dropped}
    return (
        f"Feature-mask stress level: dropped {drop_count} of {len(keys)} features "
        f"({', '.join(sorted(dropped))}).\n"
        f"{_format_features(remaining)}"
    )


def _missingness_prompt(features: Dict[str, float], severity: float, seed: int) -> str:
    rng = random.Random(seed)
    keys = list(features.keys())
    missing_count = max(1, min(len(keys), int(round(len(keys) * severity))))
    missing = set(rng.sample(keys, missing_count))
    parts = []
    for key, value in features.items():
        if key in missing:
            parts.append(f"{key}: missing")
        else:
            parts.append(f"{key}: {value}")
    return (
        f"Missingness stress level: masked {missing_count} of {len(keys)} features "
        f"({', '.join(sorted(missing))}).\n"
        + "; ".join(parts)
    )


def _counterfactual_prompt(
    features: Dict[str, float],
    actual_label: int,
    severity: float,
    seed: int,
) -> str:
    """Move the record toward an opposite-class-looking biomarker profile.

    This deliberately changes patient facts. It is useful for finding a collapse
    threshold, but it should be described as an information-altering stress test,
    not as an invariance perturbation.
    """
    del seed
    high_profile = {
        "plasma_CA19_9": 10000.0,
        "LYVE1": 5.0,
        "REG1B": 12.0,
        "TFF1": 300.0,
        "REG1A": 60.0,
    }
    low_profile = {
        "plasma_CA19_9": 5.0,
        "LYVE1": 0.001,
        "REG1B": 0.5,
        "TFF1": 2.0,
        "REG1A": 0.0,
    }

    target = low_profile if actual_label == 1 else high_profile
    alpha = max(0.0, severity)
    perturbed = dict(features)

    for key, target_value in target.items():
        if key not in perturbed:
            continue
        current = float(perturbed[key])
        moved = current + alpha * (target_value - current)
        perturbed[key] = round(max(0.0, moved), 6)

    direction = "negative-looking" if actual_label == 1 else "positive-looking"
    return (
        f"Counterfactual stress level: {severity:.2f}; moved biomarkers toward a "
        f"{direction} profile.\n"
        f"{_format_features(perturbed)}"
    )


def _build_prompt(
    family: str,
    features: Dict[str, float],
    actual_label: int,
    severity: float,
    seed: int,
) -> str:
    if family == "numeric_noise":
        return _numeric_noise_prompt(features, severity, seed)
    if family == "feature_mask":
        return _feature_mask_prompt(features, severity, seed)
    if family == "missingness":
        return _missingness_prompt(features, severity, seed)
    if family == "counterfactual":
        return _counterfactual_prompt(features, actual_label, severity, seed)
    raise ValueError(f"Unknown family: {family}")


def _levels_for_family(family: str) -> List[float]:
    if family == "numeric_noise":
        return DEFAULT_NOISE_LEVELS
    if family in {"feature_mask", "missingness"}:
        return DEFAULT_MASK_LEVELS
    if family == "counterfactual":
        return DEFAULT_COUNTERFACTUAL_LEVELS
    raise ValueError(f"Unknown family: {family}")


def _family_list(family: str) -> List[str]:
    if family == "all":
        return ["numeric_noise", "feature_mask", "missingness", "counterfactual"]
    return [family]


def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _cosine_distances(canonical: str, prompts: List[str], model_name: str) -> List[float]:
    embedder = _load_embedder(model_name)
    embeddings = embedder.encode(
        [canonical] + prompts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    canonical_embedding = embeddings[0]
    return [float(1.0 - np.dot(canonical_embedding, embeddings[i + 1])) for i in range(len(prompts))]


def _criterion_met(summary: Dict, criterion: str, target_prob: float) -> bool:
    if criterion == "model_p_true":
        return summary["mean_p_true_label"] <= target_prob
    if criterion == "empirical_accuracy":
        return summary["empirical_accuracy"] <= target_prob
    if criterion == "both":
        return (
            summary["mean_p_true_label"] <= target_prob
            and summary["empirical_accuracy"] <= target_prob
        )
    raise ValueError(f"Unknown criterion: {criterion}")


def _write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_report(
    patient_index: int,
    actual_label: int,
    summary_rows: List[Dict],
    collapse_row: Optional[Dict],
    criterion: str,
    target_prob: float,
) -> str:
    table = _markdown_table(
        [
            "Family",
            "Severity",
            "Mean distance",
            "Calls",
            "Empirical accuracy",
            "Mean p(true label)",
            "Mean confidence",
            "Collapse?",
        ],
        [
            [
                row["family"],
                _fmt(row["severity"], 2),
                _fmt(row["mean_distance"]),
                str(row["calls"]),
                _percent(row["empirical_accuracy"]),
                _fmt(row["mean_p_true_label"]),
                _fmt(row["mean_confidence"]),
                "yes" if row["collapse_detected"] else "no",
            ]
            for row in summary_rows
        ],
    )

    if collapse_row:
        conclusion = (
            f"First detected collapse: `{collapse_row['family']}` at severity "
            f"`{collapse_row['severity']:.2f}`, with mean perturbation distance "
            f"`{collapse_row['mean_distance']:.3f}` and mean p(true label) "
            f"`{collapse_row['mean_p_true_label']:.3f}`."
        )
    else:
        conclusion = (
            "No tested severity met the collapse criterion. Increase severity, variants, "
            "or replicates, or test a stronger information-altering family."
        )

    return (
        "# One-Patient Threshold Search\n\n"
        f"- Patient index: `{patient_index}`\n"
        f"- Actual label: `{actual_label}`\n"
        f"- Collapse criterion: `{criterion} <= {target_prob}`\n\n"
        f"{table}\n\n"
        "## Conclusion\n\n"
        f"{conclusion}\n\n"
        "This is a stress test, not an invariance test. The stronger families deliberately "
        "alter or degrade patient facts, so they should be used as evidence for the "
        "threshold-collapse theorem, not for Theorem 1's information-preserving invariance claim.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-patient theorem threshold stress test.")
    parser.add_argument("--run", action="store_true", help="Actually call the configured LLM.")
    parser.add_argument("--patient-index", type=int, default=0, help="Zero-based patient index.")
    parser.add_argument("--dataset", default="data/test_ready.csv", help="Dataset CSV path.")
    parser.add_argument(
        "--family",
        choices=["numeric_noise", "feature_mask", "missingness", "counterfactual", "all"],
        default="counterfactual",
        help="Stress family to scan.",
    )
    parser.add_argument("--variants", type=int, default=3, help="Prompt variants per severity.")
    parser.add_argument("--replicates", type=int, default=2, help="LLM calls per prompt variant.")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature.")
    parser.add_argument("--target-prob", type=float, default=0.10, help="Collapse threshold.")
    parser.add_argument(
        "--criterion",
        choices=["model_p_true", "empirical_accuracy", "both"],
        default="model_p_true",
        help="Quantity that must fall below target-prob.",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    features, actual_label, raw_row = _load_patient(repo_root, args.dataset, args.patient_index)
    canonical = canonical_prompt(features)

    families = _family_list(args.family)
    planned_calls = sum(len(_levels_for_family(f)) for f in families) * args.variants * args.replicates

    print(f"Patient index: {args.patient_index}")
    print(f"Actual label: {actual_label}")
    print(f"Families: {', '.join(families)}")
    print(f"Planned LLM calls: {planned_calls}")
    print(f"Canonical prompt: {canonical}")

    if not args.run:
        print("\nDRY RUN ONLY: no LLM calls were made.")
        print("Add --run to execute the stress test.")
        for family in families:
            level = _levels_for_family(family)[0]
            sample = _build_prompt(family, features, actual_label, level, args.seed)
            print(f"\nSample {family} prompt at severity {level}:")
            print(sample)
        return

    config = load_config()
    if args.temperature is not None:
        config.temperature = args.temperature
    print_config(config)

    client = initialize_client(config.llm_provider, config.azure_api_version)

    call_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    collapse_row: Optional[Dict] = None

    for family in families:
        for severity in _levels_for_family(family):
            prompts = [
                _build_prompt(
                    family,
                    features,
                    actual_label,
                    severity,
                    args.seed + (1000 * variant_idx) + int(severity * 1000),
                )
                for variant_idx in range(args.variants)
            ]
            distances = _cosine_distances(canonical, prompts, args.model)

            severity_rows: List[Dict] = []
            for variant_idx, (prompt, distance) in enumerate(zip(prompts, distances)):
                for replicate_id in range(args.replicates):
                    label, confidence, explanation, decision = call_llm(
                        client,
                        config.llm_provider,
                        config.llm_model,
                        prompt,
                        temperature=config.temperature,
                        max_retries=config.max_retries,
                        retry_delay=config.retry_delay,
                    )
                    conf_prob = _clamp(float(confidence) / 100.0, 0.0, 1.0)
                    correct = int(int(label) == actual_label)
                    p_true_label = conf_prob if correct else 1.0 - conf_prob

                    row = {
                        "patient_index": args.patient_index,
                        "actual_label": actual_label,
                        "family": family,
                        "severity": severity,
                        "variant_id": variant_idx,
                        "replicate_id": replicate_id,
                        "perturbation_distance": distance,
                        "predicted_label": int(label),
                        "confidence": conf_prob,
                        "correct": correct,
                        "p_true_label": p_true_label,
                        "decision": decision,
                        "prompt": prompt,
                        "explanation": explanation,
                    }
                    call_rows.append(row)
                    severity_rows.append(row)

            summary = {
                "patient_index": args.patient_index,
                "actual_label": actual_label,
                "family": family,
                "severity": severity,
                "calls": len(severity_rows),
                "mean_distance": _mean(row["perturbation_distance"] for row in severity_rows),
                "empirical_accuracy": _mean(row["correct"] for row in severity_rows),
                "mean_p_true_label": _mean(row["p_true_label"] for row in severity_rows),
                "mean_confidence": _mean(row["confidence"] for row in severity_rows),
                "label_counts": json.dumps(Counter(row["predicted_label"] for row in severity_rows), sort_keys=True),
            }
            summary["collapse_detected"] = _criterion_met(summary, args.criterion, args.target_prob)
            summary_rows.append(summary)

            print(
                f"{family} severity={severity:.2f}: "
                f"distance={summary['mean_distance']:.3f}, "
                f"accuracy={_percent(summary['empirical_accuracy'])}, "
                f"p_true={summary['mean_p_true_label']:.3f}, "
                f"collapse={summary['collapse_detected']}"
            )

            if summary["collapse_detected"] and collapse_row is None:
                collapse_row = summary
                print("Stopping after first detected collapse.")
                break

        if collapse_row is not None:
            break

    _write_csv(
        out_dir / "one_patient_threshold_calls.csv",
        call_rows,
        [
            "patient_index",
            "actual_label",
            "family",
            "severity",
            "variant_id",
            "replicate_id",
            "perturbation_distance",
            "predicted_label",
            "confidence",
            "correct",
            "p_true_label",
            "decision",
            "prompt",
            "explanation",
        ],
    )
    _write_csv(
        out_dir / "one_patient_threshold_summary.csv",
        summary_rows,
        [
            "patient_index",
            "actual_label",
            "family",
            "severity",
            "calls",
            "mean_distance",
            "empirical_accuracy",
            "mean_p_true_label",
            "mean_confidence",
            "label_counts",
            "collapse_detected",
        ],
    )
    (out_dir / "one_patient_threshold_report.md").write_text(
        _render_report(
            args.patient_index,
            actual_label,
            summary_rows,
            collapse_row,
            args.criterion,
            args.target_prob,
        ),
        encoding="utf-8",
    )

    print(f"Wrote outputs to {_console_safe(out_dir)}")


if __name__ == "__main__":
    main()
