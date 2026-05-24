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
Measure the empirical "perturbation threshold -> correctness collapse" claim.

Plain-language claim being tested:
    If perturbation magnitude is greater than some threshold T, then the
    probability that the model's predicted outcome is true approaches 0.

Important interpretation:
    The PDF labels this as Theorem 3, not Theorem 1. Theorem 1 is the
    normative invariance statement. This script measures the threshold-collapse
    claim using saved experiment outputs.

What this script can measure from current results:
    - Perturbation magnitude: embedding cosine distance between the canonical
      prompt and each perturbed prompt.
    - Empirical correctness: whether the perturbed prediction equals
      data/actual.csv.
    - Model-implied probability of the true label: if the predicted label is
      correct, confidence / 100; otherwise, 1 - confidence / 100.

Outputs are written to the new/ folder by default:
    - theorem_threshold_call_metrics.csv
    - theorem_threshold_summary.csv
    - theorem_threshold_report.md

Run from the repository root:
    .venv\\Scripts\\python.exe new\\measure_theorem_threshold.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


SETTING_SPECS = (
    ("Tier 1", "without", "outputs", "output_tier_1_without_context_0.json"),
    ("Tier 1", "with", "outputs", "output_tier_1_with_context_0.json"),
    ("Tier 2", "without", "outputs_correct", "output_tier_2_without_context_0.json"),
    ("Tier 2", "with", "outputs_correct", "output_tier_2_with_context_0.json"),
)


QUANTILES = (0.0, 0.25, 0.50, 2.0 / 3.0, 0.75, 0.80, 0.90, 0.95, 0.99)


def _safe_float(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp_probability(value: float) -> float:
    if math.isnan(value):
        return value
    return max(0.0, min(1.0, value))


def _mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return sum(clean) / len(clean) if clean else math.nan


def _percent(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "N/A"
    return f"{100.0 * float(value):.1f}%"


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or math.isnan(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _console_safe(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def _load_ground_truth(repo_root: Path) -> Dict[int, int]:
    actual_path = repo_root / "data" / "actual.csv"
    if not actual_path.exists():
        raise FileNotFoundError(f"Missing ground truth file: {actual_path}")

    with actual_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    labels: Dict[int, int] = {}
    for fallback_idx, row in enumerate(rows):
        if "Patient" in row and row["Patient"]:
            patient_idx = int(row["Patient"]) - 1
        else:
            patient_idx = fallback_idx
        labels[patient_idx] = int(row["patient_diagnosis"])
    return labels


def _patient_idx_from_key(patient_key: str) -> int:
    if patient_key.startswith("patient_"):
        return int(patient_key.split("_", 1)[1])
    return int(patient_key)


def _iter_entries_with_instance(result_group) -> Iterable[Tuple[int, Dict]]:
    def with_parent_fields(entry: Dict, parent: Dict) -> Dict:
        enriched = dict(entry)
        if "prompt" not in enriched and isinstance(parent, dict):
            enriched["prompt"] = parent.get("prompt", "")
        return enriched

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
                yield 0, with_parent_fields(entry, result_group)
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
                    yield instance_id, with_parent_fields(entry, instance_data)
        elif isinstance(instance_data, list):
            for entry in instance_data:
                if isinstance(entry, dict):
                    yield instance_id, entry
        elif isinstance(instance_data, dict) and "label" in instance_data:
            yield instance_id, instance_data


def _canonical_entries(patient_data: Dict) -> List[Dict]:
    entries = [entry for _, entry in _iter_entries_with_instance(patient_data.get("canonical"))]
    return entries


def _canonical_by_replicate(patient_data: Dict) -> Tuple[List[Dict], Dict[int, Dict]]:
    entries = _canonical_entries(patient_data)
    by_replicate = {}
    for entry in entries:
        replicate_id = _safe_int(entry.get("replicate_id"), 0)
        by_replicate[replicate_id] = entry
    return entries, by_replicate


def _fallback_canonical(canonical_entries: List[Dict], replicate_id: int) -> Optional[Dict]:
    if not canonical_entries:
        return None
    if replicate_id < len(canonical_entries):
        return canonical_entries[replicate_id]
    return canonical_entries[0]


def _load_call_rows(repo_root: Path) -> List[Dict]:
    ground_truth = _load_ground_truth(repo_root)
    rows: List[Dict] = []

    for tier, context, folder_name, file_name in SETTING_SPECS:
        result_path = repo_root / folder_name / file_name
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result file: {result_path}")

        with result_path.open(encoding="utf-8") as handle:
            result_data = json.load(handle)

        for patient_key, patient_data in result_data.get("results_by_patient", {}).items():
            patient_idx = _patient_idx_from_key(patient_key)
            actual_label = ground_truth.get(patient_idx)
            if actual_label is None:
                continue

            canonical_entries, canonical_lookup = _canonical_by_replicate(patient_data)
            if not canonical_entries:
                continue

            for family_name, family_data in patient_data.items():
                if family_name == "canonical":
                    continue

                for instance_id, entry in _iter_entries_with_instance(family_data):
                    replicate_id = _safe_int(entry.get("replicate_id"), 0) or 0
                    canonical = canonical_lookup.get(
                        replicate_id,
                        _fallback_canonical(canonical_entries, replicate_id),
                    )
                    if canonical is None:
                        continue

                    predicted_label = _safe_int(entry.get("label"))
                    confidence = _clamp_probability(_safe_float(entry.get("confidence")) / 100.0)
                    if predicted_label is None or math.isnan(confidence):
                        continue

                    correct = int(predicted_label == actual_label)
                    p_true_label = confidence if correct else 1.0 - confidence

                    rows.append({
                        "tier": tier,
                        "context": context,
                        "source_file": str(result_path.relative_to(repo_root)),
                        "patient_idx": patient_idx,
                        "patient_number": patient_idx + 1,
                        "family": family_name,
                        "instance_id": instance_id,
                        "replicate_id": replicate_id,
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "correct": correct,
                        "p_true_label": p_true_label,
                        "canonical_prompt": canonical.get("prompt", ""),
                        "perturbed_prompt": entry.get("prompt", ""),
                        "canonical_explanation": canonical.get("explanation", ""),
                        "perturbed_explanation": entry.get("explanation", ""),
                    })

    return rows


def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embed_unique_texts(texts: Iterable[str], model_name: str, batch_size: int) -> Dict[str, np.ndarray]:
    unique_texts = sorted({text or "" for text in texts})
    embedder = _load_embedder(model_name)
    embeddings = embedder.encode(
        unique_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return {text: embeddings[idx] for idx, text in enumerate(unique_texts)}


def _cosine_distance_from_normalized(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def _add_distances(rows: List[Dict], model_name: str, batch_size: int, source: str) -> None:
    if source == "prompt":
        left_key = "canonical_prompt"
        right_key = "perturbed_prompt"
        out_key = "perturbation_distance"
    elif source == "explanation":
        left_key = "canonical_explanation"
        right_key = "perturbed_explanation"
        out_key = "explanation_distance"
    else:
        raise ValueError(f"Unknown distance source: {source}")

    embeddings = _embed_unique_texts(
        [row[left_key] for row in rows] + [row[right_key] for row in rows],
        model_name=model_name,
        batch_size=batch_size,
    )

    for row in rows:
        row[out_key] = _cosine_distance_from_normalized(
            embeddings[row[left_key] or ""],
            embeddings[row[right_key] or ""],
        )


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.quantile(np.array(values, dtype=float), q))


def _summarize_thresholds(
    rows: List[Dict],
    distance_key: str,
    target_prob: float,
    min_n: int,
) -> Tuple[List[Dict], Optional[Dict]]:
    distances = sorted(row[distance_key] for row in rows if not math.isnan(row[distance_key]))
    thresholds = sorted({_quantile(distances, q) for q in QUANTILES})

    summary_rows: List[Dict] = []
    best_collapse_row: Optional[Dict] = None
    for threshold in thresholds:
        subset = [row for row in rows if row[distance_key] > threshold]
        if not subset:
            continue

        summary = {
            "threshold": threshold,
            "n_above": len(subset),
            "fraction_above": len(subset) / len(rows),
            "mean_distance_above": _mean(row[distance_key] for row in subset),
            "empirical_accuracy_above": _mean(row["correct"] for row in subset),
            "error_rate_above": 1.0 - _mean(row["correct"] for row in subset),
            "mean_p_true_label_above": _mean(row["p_true_label"] for row in subset),
            "median_p_true_label_above": float(np.median([row["p_true_label"] for row in subset])),
            "min_p_true_label_above": min(row["p_true_label"] for row in subset),
        }
        summary_rows.append(summary)

        if (
            summary["n_above"] >= min_n
            and summary["mean_p_true_label_above"] <= target_prob
            and best_collapse_row is None
        ):
            best_collapse_row = summary

    return summary_rows, best_collapse_row


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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
    rows: List[Dict],
    summary_rows: List[Dict],
    collapse_row: Optional[Dict],
    distance_key: str,
    target_prob: float,
    min_n: int,
) -> str:
    all_accuracy = _mean(row["correct"] for row in rows)
    all_p_true = _mean(row["p_true_label"] for row in rows)
    max_distance = max(row[distance_key] for row in rows)
    high_tail = summary_rows[-1] if summary_rows else None

    threshold_table = _markdown_table(
        [
            "Threshold T",
            "N with distance > T",
            "Mean distance",
            "Empirical accuracy",
            "Mean p(true label)",
            "Min p(true label)",
        ],
        [
            [
                _fmt(row["threshold"]),
                str(row["n_above"]),
                _fmt(row["mean_distance_above"]),
                _percent(row["empirical_accuracy_above"]),
                _fmt(row["mean_p_true_label_above"]),
                _fmt(row["min_p_true_label_above"]),
            ]
            for row in summary_rows
        ],
    )

    if collapse_row is None:
        conclusion = (
            f"No observed threshold with at least `{min_n}` calls made the mean model-implied "
            f"probability of the true label fall to `{target_prob}` or lower. "
            "So the current saved results do not empirically support the strong "
            "\"probability approaches zero\" version of the theorem."
        )
    else:
        conclusion = (
            f"The first observed threshold satisfying the target is `T > "
            f"{collapse_row['threshold']:.3f}` with `N={collapse_row['n_above']}` calls "
            f"and mean p(true label) `{collapse_row['mean_p_true_label_above']:.3f}`."
        )

    tail_sentence = ""
    if high_tail is not None:
        tail_sentence = (
            f"In the highest-distance tail measured here (`T > {high_tail['threshold']:.3f}`), "
            f"there are `{high_tail['n_above']}` calls, empirical accuracy is "
            f"{_percent(high_tail['empirical_accuracy_above'])}, and mean p(true label) is "
            f"{_fmt(high_tail['mean_p_true_label_above'])}."
        )

    return (
        "# Theorem Threshold Measurement\n\n"
        "This measures the threshold-collapse claim stated in the PDF as Theorem 3. "
        "Theorem 1 is the invariance theorem; it is not the theorem that predicts "
        "probability collapse after large perturbations.\n\n"
        "## Definitions\n\n"
        f"- Perturbation magnitude: `{distance_key}`, the cosine distance between the "
        "canonical prompt embedding and the perturbed prompt embedding.\n"
        "- Empirical accuracy: fraction of perturbed calls where the predicted label equals "
        "`data/actual.csv`.\n"
        "- p(true label): model-implied probability assigned to the true label. If the "
        "model predicted the true label, this is `confidence / 100`; otherwise it is "
        "`1 - confidence / 100`.\n\n"
        "## Overall Results\n\n"
        f"- Perturbed calls measured: `{len(rows)}`\n"
        f"- Overall empirical accuracy: `{_percent(all_accuracy)}`\n"
        f"- Overall mean p(true label): `{_fmt(all_p_true)}`\n"
        f"- Maximum observed perturbation distance: `{_fmt(max_distance)}`\n\n"
        "## Threshold Scan\n\n"
        f"{threshold_table}\n\n"
        "## Interpretation\n\n"
        f"{conclusion}\n\n"
        f"{tail_sentence}\n\n"
        "This means the current data can support a weaker statement: larger perturbation "
        "distance is associated with degraded reliability only if the threshold table shows "
        "falling accuracy or falling p(true label). It should not be written as true-label "
        "probability approaching zero unless a future run produces high-distance examples "
        "with near-zero empirical accuracy or near-zero p(true label).\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure empirical perturbation thresholds for theorem evidence.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory.",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model for prompt embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--target-prob",
        type=float,
        default=0.10,
        help="Collapse target for mean p(true label).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=30,
        help="Minimum number of calls above threshold for accepting a collapse threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_call_rows(repo_root)
    if not rows:
        raise RuntimeError("No perturbed call rows found.")

    print(f"Loaded {len(rows)} perturbed calls.")
    print("Embedding canonical and perturbed prompts...")
    _add_distances(rows, args.model, args.batch_size, source="prompt")

    summary_rows, collapse_row = _summarize_thresholds(
        rows,
        distance_key="perturbation_distance",
        target_prob=args.target_prob,
        min_n=args.min_n,
    )

    call_fields = [
        "tier",
        "context",
        "source_file",
        "patient_number",
        "patient_idx",
        "family",
        "instance_id",
        "replicate_id",
        "actual_label",
        "predicted_label",
        "confidence",
        "correct",
        "p_true_label",
        "perturbation_distance",
    ]
    _write_csv(out_dir / "theorem_threshold_call_metrics.csv", rows, call_fields)
    _write_csv(
        out_dir / "theorem_threshold_summary.csv",
        summary_rows,
        [
            "threshold",
            "n_above",
            "fraction_above",
            "mean_distance_above",
            "empirical_accuracy_above",
            "error_rate_above",
            "mean_p_true_label_above",
            "median_p_true_label_above",
            "min_p_true_label_above",
        ],
    )
    (out_dir / "theorem_threshold_report.md").write_text(
        _render_report(
            rows,
            summary_rows,
            collapse_row,
            distance_key="perturbation_distance",
            target_prob=args.target_prob,
            min_n=args.min_n,
        ),
        encoding="utf-8",
    )

    print(f"Wrote outputs to {_console_safe(out_dir)}")
    if collapse_row is None:
        print(
            "No threshold met the collapse target "
            f"mean p(true label) <= {args.target_prob} with min_n={args.min_n}."
        )
    else:
        print(
            "Collapse threshold found: "
            f"T > {collapse_row['threshold']:.3f}, "
            f"mean p(true label)={collapse_row['mean_p_true_label_above']:.3f}, "
            f"N={collapse_row['n_above']}."
        )


if __name__ == "__main__":
    main()
