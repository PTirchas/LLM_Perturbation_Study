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
Validate the perturbation-threshold claim on a patient sample.

This is the cohort-level version of `one_patient_threshold_search.py`.
It samples a fraction of patients, runs an ordered stress ladder for each
patient, and reports whether the sample supports:

  1. monotonic degradation:
       larger perturbations -> lower p(true label)

  2. collapse:
       some threshold makes p(true label) <= target_prob

By default this is a dry run and makes no LLM calls.

Dry run:
    .venv\\Scripts\\python.exe new\\validate_threshold_sample.py

Recommended first real run over ~10% of patients:
    .venv\\Scripts\\python.exe new\\validate_threshold_sample.py --run --sample-frac 0.10 --family counterfactual --variants 3 --replicates 2

Outputs:
    new/sample_threshold_calls.csv
    new/sample_threshold_patient_summary.csv
    new/sample_threshold_level_summary.csv
    new/sample_threshold_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from one_patient_threshold_search import (  # noqa: E402
    _build_prompt,
    _criterion_met,
    _family_list,
    _levels_for_family,
    _load_actual_labels,
    _load_embedder,
)
from utils.config import load_config, print_config  # noqa: E402
from utils.data import load_data  # noqa: E402
from utils.llm import call_llm, initialize_client  # noqa: E402
from utils.perturbations import canonical_prompt  # noqa: E402


def _safe_float(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _clamp_probability(value: float) -> float:
    if math.isnan(value):
        return value
    return max(0.0, min(1.0, value))


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


def _load_dataset(repo_root: Path, dataset_path: str) -> Tuple[List[Dict], Dict[int, int]]:
    df = load_data(str(repo_root / dataset_path))
    labels = _load_actual_labels(repo_root)
    exclude_cols = {"patient_diagnosis", "Patient", "sample_id"}
    feature_names = [col for col in df.columns if col not in exclude_cols]

    patients = []
    for patient_idx, row in df.iterrows():
        patients.append({
            "patient_index": int(patient_idx),
            "actual_label": labels[int(patient_idx)],
            "features": {name: float(row[name]) for name in feature_names},
        })
    return patients, labels


def _parse_patient_indices(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    indices = []
    for item in value.split(","):
        item = item.strip()
        if item:
            indices.append(int(item))
    return indices


def _sample_patients(
    patients: List[Dict],
    sample_frac: float,
    seed: int,
    max_patients: Optional[int],
    explicit_indices: Optional[List[int]],
) -> List[Dict]:
    if explicit_indices is not None:
        lookup = {patient["patient_index"]: patient for patient in patients}
        missing = [idx for idx in explicit_indices if idx not in lookup]
        if missing:
            raise ValueError(f"Unknown patient indices: {missing}")
        selected = [lookup[idx] for idx in explicit_indices]
    else:
        sample_count = max(1, int(math.ceil(len(patients) * sample_frac)))
        if max_patients is not None:
            sample_count = min(sample_count, max_patients)
        rng = random.Random(seed)
        selected = rng.sample(patients, sample_count)

    return sorted(selected, key=lambda row: row["patient_index"])


def _embed_distances(embedder, canonical: str, prompts: List[str]) -> List[float]:
    embeddings = embedder.encode(
        [canonical] + prompts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    base = embeddings[0]
    return [float(1.0 - np.dot(base, embeddings[idx + 1])) for idx in range(len(prompts))]


def _slope(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return math.nan
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.allclose(x, x[0]):
        return math.nan
    return float(np.polyfit(x, y, 1)[0])


def _adjacent_drop_fraction(values: List[float]) -> float:
    if len(values) < 2:
        return math.nan
    drops = 0
    comparisons = 0
    for left, right in zip(values[:-1], values[1:]):
        if math.isnan(left) or math.isnan(right):
            continue
        comparisons += 1
        if right < left:
            drops += 1
    return drops / comparisons if comparisons else math.nan


def _summarize_patient(
    patient_index: int,
    actual_label: int,
    level_rows: List[Dict],
    target_prob: float,
    min_drop: float,
    min_adjacent_drop_fraction: float,
) -> Dict:
    p_values = [row["mean_p_true_label"] for row in level_rows]
    severities = [row["severity_order"] for row in level_rows]
    distances = [row["mean_distance"] for row in level_rows]

    first_p = p_values[0] if p_values else math.nan
    last_p = p_values[-1] if p_values else math.nan
    p_drop = first_p - last_p if not math.isnan(first_p) and not math.isnan(last_p) else math.nan
    severity_slope = _slope(severities, p_values)
    distance_slope = _slope(distances, p_values)
    adjacent_drop_fraction = _adjacent_drop_fraction(p_values)
    collapse_rows = [row for row in level_rows if row["collapse_detected"]]
    first_collapse = collapse_rows[0] if collapse_rows else None

    supports_degradation = (
        not math.isnan(p_drop)
        and p_drop >= min_drop
        and not math.isnan(severity_slope)
        and severity_slope < 0
        and (
            math.isnan(adjacent_drop_fraction)
            or adjacent_drop_fraction >= min_adjacent_drop_fraction
        )
    )

    return {
        "patient_index": patient_index,
        "actual_label": actual_label,
        "levels_tested": len(level_rows),
        "first_mean_p_true_label": first_p,
        "last_mean_p_true_label": last_p,
        "p_true_drop": p_drop,
        "severity_slope": severity_slope,
        "distance_slope": distance_slope,
        "adjacent_drop_fraction": adjacent_drop_fraction,
        "supports_degradation": supports_degradation,
        "collapse_detected": first_collapse is not None,
        "collapse_family": first_collapse["family"] if first_collapse else "",
        "collapse_severity": first_collapse["severity"] if first_collapse else math.nan,
        "collapse_distance": first_collapse["mean_distance"] if first_collapse else math.nan,
        "collapse_p_true_label": first_collapse["mean_p_true_label"] if first_collapse else math.nan,
        "target_prob": target_prob,
    }


def _validation_passed(
    patient_rows: List[Dict],
    mode: str,
    min_support_rate: float,
) -> Tuple[bool, float]:
    if not patient_rows:
        return False, math.nan

    if mode == "degradation":
        support = _mean(row["supports_degradation"] for row in patient_rows)
    elif mode == "collapse":
        support = _mean(row["collapse_detected"] for row in patient_rows)
    elif mode == "either":
        support = _mean(row["supports_degradation"] or row["collapse_detected"] for row in patient_rows)
    elif mode == "both":
        support = _mean(row["supports_degradation"] and row["collapse_detected"] for row in patient_rows)
    else:
        raise ValueError(f"Unknown validation mode: {mode}")

    return support >= min_support_rate, support


def _render_report(
    selected_patients: List[Dict],
    patient_rows: List[Dict],
    level_rows: List[Dict],
    validation_mode: str,
    validation_passed: bool,
    support_rate: float,
    min_support_rate: float,
    dry_run: bool,
    planned_calls: int,
) -> str:
    selected_indices = ", ".join(str(row["patient_index"]) for row in selected_patients)

    if dry_run:
        conclusion = (
            "Dry run only. No LLM calls were made, so validation was not evaluated."
        )
    elif validation_passed:
        conclusion = (
            f"Validation passed under mode `{validation_mode}`: support rate "
            f"{_percent(support_rate)} met the required {_percent(min_support_rate)}."
        )
    else:
        conclusion = (
            f"Validation did not pass under mode `{validation_mode}`: support rate "
            f"{_percent(support_rate)} was below the required {_percent(min_support_rate)}."
        )

    patient_table = _markdown_table(
        [
            "Patient",
            "Actual",
            "p(true) first",
            "p(true) last",
            "Drop",
            "Severity slope",
            "Adj. drops",
            "Degradation?",
            "Collapse?",
        ],
        [
            [
                str(row["patient_index"]),
                str(row["actual_label"]),
                _fmt(row["first_mean_p_true_label"]),
                _fmt(row["last_mean_p_true_label"]),
                _fmt(row["p_true_drop"]),
                _fmt(row["severity_slope"]),
                _percent(row["adjacent_drop_fraction"]),
                "yes" if row["supports_degradation"] else "no",
                "yes" if row["collapse_detected"] else "no",
            ]
            for row in patient_rows
        ],
    ) if patient_rows else "_No patient-level results yet._"

    level_table_rows = []
    for row in level_rows[:30]:
        level_table_rows.append([
            str(row["patient_index"]),
            row["family"],
            _fmt(row["severity"], 2),
            _fmt(row["mean_distance"]),
            _percent(row["empirical_accuracy"]),
            _fmt(row["mean_p_true_label"]),
            "yes" if row["collapse_detected"] else "no",
        ])
    level_table = _markdown_table(
        [
            "Patient",
            "Family",
            "Severity",
            "Mean distance",
            "Accuracy",
            "p(true)",
            "Collapse?",
        ],
        level_table_rows,
    ) if level_table_rows else "_No level results yet._"

    return (
        "# Sample Threshold Validation\n\n"
        f"- Selected patients: `{selected_indices}`\n"
        f"- Number of patients: `{len(selected_patients)}`\n"
        f"- Planned LLM calls: `{planned_calls}`\n"
        f"- Validation mode: `{validation_mode}`\n"
        f"- Minimum support rate: `{_percent(min_support_rate)}`\n\n"
        "## Conclusion\n\n"
        f"{conclusion}\n\n"
        "## Patient-Level Summary\n\n"
        f"{patient_table}\n\n"
        "## Level Summary Preview\n\n"
        f"{level_table}\n\n"
        "A degradation result supports the weaker theorem claim: larger perturbation "
        "magnitude reduces reliability. A collapse result supports the stronger claim: "
        "some threshold makes p(true label) approach zero.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate threshold behavior on a patient sample.")
    parser.add_argument("--run", action="store_true", help="Actually call the configured LLM.")
    parser.add_argument("--dataset", default="data/test_ready.csv", help="Dataset CSV path.")
    parser.add_argument("--sample-frac", type=float, default=0.10, help="Fraction of patients to sample.")
    parser.add_argument("--max-patients", type=int, default=None, help="Optional cap on sampled patients.")
    parser.add_argument("--patient-indices", default=None, help="Comma-separated explicit zero-based indices.")
    parser.add_argument(
        "--family",
        choices=["numeric_noise", "feature_mask", "missingness", "counterfactual", "all"],
        default="counterfactual",
        help="Stress family to scan.",
    )
    parser.add_argument("--variants", type=int, default=3, help="Prompt variants per severity.")
    parser.add_argument("--replicates", type=int, default=2, help="LLM calls per prompt variant.")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature.")
    parser.add_argument("--target-prob", type=float, default=0.10, help="Collapse p(true label) threshold.")
    parser.add_argument(
        "--criterion",
        choices=["model_p_true", "empirical_accuracy", "both"],
        default="model_p_true",
        help="Per-level collapse criterion.",
    )
    parser.add_argument("--min-drop", type=float, default=0.05, help="Minimum first-to-last p(true) drop.")
    parser.add_argument(
        "--min-adjacent-drop-fraction",
        type=float,
        default=0.50,
        help="Minimum fraction of adjacent severity steps with lower p(true).",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["degradation", "collapse", "either", "both"],
        default="degradation",
        help="Sample-level validation criterion.",
    )
    parser.add_argument(
        "--min-support-rate",
        type=float,
        default=0.60,
        help="Fraction of sampled patients required for validation.",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    patients, _ = _load_dataset(repo_root, args.dataset)
    explicit_indices = _parse_patient_indices(args.patient_indices)
    selected_patients = _sample_patients(
        patients,
        sample_frac=args.sample_frac,
        seed=args.seed,
        max_patients=args.max_patients,
        explicit_indices=explicit_indices,
    )
    families = _family_list(args.family)
    levels_by_family = {family: _levels_for_family(family) for family in families}
    planned_calls = (
        len(selected_patients)
        * sum(len(levels) for levels in levels_by_family.values())
        * args.variants
        * args.replicates
    )

    print(f"Dataset patients: {len(patients)}")
    print(f"Selected patients: {[row['patient_index'] for row in selected_patients]}")
    print(f"Families: {', '.join(families)}")
    print(f"Planned LLM calls: {planned_calls}")

    if not args.run:
        print("\nDRY RUN ONLY: no LLM calls were made.")
        print("Add --run to execute the sample validation.")
        sample_patient = selected_patients[0]
        sample_family = families[0]
        sample_level = levels_by_family[sample_family][0]
        sample_prompt = _build_prompt(
            sample_family,
            sample_patient["features"],
            sample_patient["actual_label"],
            sample_level,
            args.seed,
        )
        print(f"\nSample prompt for patient {sample_patient['patient_index']}:")
        print(sample_prompt)
        report = _render_report(
            selected_patients,
            patient_rows=[],
            level_rows=[],
            validation_mode=args.validation_mode,
            validation_passed=False,
            support_rate=math.nan,
            min_support_rate=args.min_support_rate,
            dry_run=True,
            planned_calls=planned_calls,
        )
        (out_dir / "sample_threshold_report.md").write_text(report, encoding="utf-8")
        return

    config = load_config()
    if args.temperature is not None:
        config.temperature = args.temperature
    print_config(config)
    client = initialize_client(config.llm_provider, config.azure_api_version)
    embedder = _load_embedder(args.model)

    call_rows: List[Dict] = []
    level_rows: List[Dict] = []
    patient_rows: List[Dict] = []

    for patient in selected_patients:
        patient_index = patient["patient_index"]
        actual_label = patient["actual_label"]
        features = patient["features"]
        canonical = canonical_prompt(features)
        patient_level_rows: List[Dict] = []
        severity_order = 0

        print(f"\nPatient {patient_index} actual={actual_label}")

        for family in families:
            for severity in levels_by_family[family]:
                prompts = [
                    _build_prompt(
                        family,
                        features,
                        actual_label,
                        severity,
                        args.seed + patient_index * 100000 + severity_order * 1000 + variant_idx,
                    )
                    for variant_idx in range(args.variants)
                ]
                distances = _embed_distances(embedder, canonical, prompts)

                current_call_rows: List[Dict] = []
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
                        confidence_prob = _clamp_probability(_safe_float(confidence) / 100.0)
                        predicted_label = int(label)
                        correct = int(predicted_label == actual_label)
                        p_true_label = confidence_prob if correct else 1.0 - confidence_prob

                        row = {
                            "patient_index": patient_index,
                            "actual_label": actual_label,
                            "family": family,
                            "severity_order": severity_order,
                            "severity": severity,
                            "variant_id": variant_idx,
                            "replicate_id": replicate_id,
                            "perturbation_distance": distance,
                            "predicted_label": predicted_label,
                            "confidence": confidence_prob,
                            "correct": correct,
                            "p_true_label": p_true_label,
                            "decision": decision,
                            "prompt": prompt,
                            "explanation": explanation,
                        }
                        current_call_rows.append(row)
                        call_rows.append(row)

                summary = {
                    "patient_index": patient_index,
                    "actual_label": actual_label,
                    "family": family,
                    "severity_order": severity_order,
                    "severity": severity,
                    "calls": len(current_call_rows),
                    "mean_distance": _mean(row["perturbation_distance"] for row in current_call_rows),
                    "empirical_accuracy": _mean(row["correct"] for row in current_call_rows),
                    "mean_p_true_label": _mean(row["p_true_label"] for row in current_call_rows),
                    "mean_confidence": _mean(row["confidence"] for row in current_call_rows),
                }
                summary["collapse_detected"] = _criterion_met(summary, args.criterion, args.target_prob)
                level_rows.append(summary)
                patient_level_rows.append(summary)

                print(
                    f"  {family} severity={severity:.2f}: "
                    f"p_true={summary['mean_p_true_label']:.3f}, "
                    f"acc={_percent(summary['empirical_accuracy'])}, "
                    f"collapse={summary['collapse_detected']}"
                )
                severity_order += 1

        patient_summary = _summarize_patient(
            patient_index,
            actual_label,
            patient_level_rows,
            target_prob=args.target_prob,
            min_drop=args.min_drop,
            min_adjacent_drop_fraction=args.min_adjacent_drop_fraction,
        )
        patient_rows.append(patient_summary)

    passed, support_rate = _validation_passed(
        patient_rows,
        mode=args.validation_mode,
        min_support_rate=args.min_support_rate,
    )

    _write_csv(
        out_dir / "sample_threshold_calls.csv",
        call_rows,
        [
            "patient_index",
            "actual_label",
            "family",
            "severity_order",
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
        out_dir / "sample_threshold_level_summary.csv",
        level_rows,
        [
            "patient_index",
            "actual_label",
            "family",
            "severity_order",
            "severity",
            "calls",
            "mean_distance",
            "empirical_accuracy",
            "mean_p_true_label",
            "mean_confidence",
            "collapse_detected",
        ],
    )
    _write_csv(
        out_dir / "sample_threshold_patient_summary.csv",
        patient_rows,
        [
            "patient_index",
            "actual_label",
            "levels_tested",
            "first_mean_p_true_label",
            "last_mean_p_true_label",
            "p_true_drop",
            "severity_slope",
            "distance_slope",
            "adjacent_drop_fraction",
            "supports_degradation",
            "collapse_detected",
            "collapse_family",
            "collapse_severity",
            "collapse_distance",
            "collapse_p_true_label",
            "target_prob",
        ],
    )
    (out_dir / "sample_threshold_report.md").write_text(
        _render_report(
            selected_patients,
            patient_rows,
            level_rows,
            validation_mode=args.validation_mode,
            validation_passed=passed,
            support_rate=support_rate,
            min_support_rate=args.min_support_rate,
            dry_run=False,
            planned_calls=planned_calls,
        ),
        encoding="utf-8",
    )

    print(f"\nSupport rate: {_percent(support_rate)}")
    print(f"Validation passed: {passed}")
    print(f"Wrote outputs to {_console_safe(out_dir)}")


if __name__ == "__main__":
    main()
