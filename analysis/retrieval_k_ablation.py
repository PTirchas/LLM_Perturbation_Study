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
Retrieval-context k ablation for k in {1, 3, 5, 10, 20, 50}.

Purpose:
    Validate the manuscript statement that k=3 was chosen by preliminary
    cross-validation over candidate context depths, balancing context
    richness and prompt length.

Default mode is dry-run: it estimates prompt/token cost without LLM calls.
The default experiment is canonical-only with 3 stochastic replicates per
patient/k, intended as a lightweight cross-check of which k is best.

Dry run:
    .venv\\Scripts\\python.exe new\\retrieval_k_ablation.py

Full run over all patients:
    .venv\\Scripts\\python.exe new\\retrieval_k_ablation.py --run

The script is resumable: if retrieval_k_ablation_calls.csv already exists
in the output directory, completed calls whose keys match the requested
plan are reused and only missing calls are sent to the LLM.

Outputs:
    new/retrieval_k_ablation_calls.csv
    new/retrieval_k_ablation_patient_summary.csv
    new/retrieval_k_ablation_k_summary.csv
    new/retrieval_k_ablation_report.md
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
from utils.perturbations import PERTURBATIONS, canonical_prompt  # noqa: E402
from utils.retrieval import get_nearest_neighbours, format_neighbours_as_context  # noqa: E402


DEFAULT_FAMILIES: List[str] = []


def _safe_float(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_probability(value: float) -> float:
    if math.isnan(value):
        return value
    return max(0.0, min(1.0, value))


def _mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(clean) / len(clean) if clean else math.nan


def _std(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.std(clean, ddof=0)) if clean else math.nan


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


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> List[str]:
    if value.lower().strip() in {"", "none", "canonical"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _call_key(row: Dict) -> Tuple[int, int, str, int, int]:
    return (
        int(row["k"]),
        int(row["patient_index"]),
        str(row["family"]),
        int(row["variant_id"]),
        int(row["replicate_id"]),
    )


def _maybe_int(value) -> Optional[int]:
    if value in {None, ""}:
        return None
    return int(float(value))


def _load_existing_calls(path: Path, allowed_keys: set[Tuple[int, int, str, int, int]]) -> Dict[Tuple[int, int, str, int, int], Dict]:
    if not path.exists():
        return {}

    int_fields = {
        "k",
        "patient_index",
        "actual_label",
        "variant_id",
        "replicate_id",
        "prompt_chars",
        "prompt_tokens",
        "predicted_label",
        "correct",
    }
    float_fields = {"confidence", "p_true_label"}
    existing: Dict[Tuple[int, int, str, int, int], Dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                for field in int_fields:
                    row[field] = _maybe_int(row.get(field))
                for field in float_fields:
                    row[field] = _safe_float(row.get(field))
                key = _call_key(row)
            except (TypeError, ValueError, KeyError):
                continue

            if key in allowed_keys and row.get("predicted_label") is not None:
                existing[key] = row
    return existing


def _load_actual_labels(repo_root: Path) -> Dict[int, int]:
    actual_path = repo_root / "data" / "actual.csv"
    with actual_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    labels = {}
    for fallback_idx, row in enumerate(rows):
        patient_idx = int(row["Patient"]) - 1 if row.get("Patient") else fallback_idx
        labels[patient_idx] = int(row["patient_diagnosis"])
    return labels


def _load_patients(repo_root: Path, dataset_path: str) -> List[Dict]:
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
    return patients


def _select_patients(
    patients: List[Dict],
    patient_indices: Optional[str],
    max_patients: Optional[int],
) -> List[Dict]:
    if patient_indices:
        wanted = set(_parse_int_list(patient_indices))
        selected = [patient for patient in patients if patient["patient_index"] in wanted]
        missing = sorted(wanted - {patient["patient_index"] for patient in selected})
        if missing:
            raise ValueError(f"Unknown patient indices: {missing}")
    else:
        selected = list(patients)

    if max_patients is not None:
        selected = selected[:max_patients]
    return selected


def _make_context(features: Dict[str, float], context_df, k: int, seed: int) -> str:
    neighbours = get_nearest_neighbours(
        features,
        context_df,
        k=k,
        seed=seed,
        diversity_penalty=0.0,
    )
    # format_neighbours_as_context mutates entries by popping "label"; pass copies.
    return format_neighbours_as_context([dict(neighbour) for neighbour in neighbours]) + "\n"


def _get_token_counter(encoding_name: str):
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        return lambda text: len(encoding.encode(text))
    except Exception:
        return lambda text: int(math.ceil(len(text) / 4.0))


def _build_prompt_plan(
    patients: List[Dict],
    context_df,
    k_values: List[int],
    families: List[str],
    variants: int,
    canonical_replicates: int,
    perturb_replicates: int,
    seed: int,
    token_count,
) -> List[Dict]:
    rows: List[Dict] = []
    for k in k_values:
        for patient in patients:
            patient_idx = patient["patient_index"]
            features = patient["features"]
            context = _make_context(features, context_df, k, seed + patient_idx)

            canonical = context + canonical_prompt(features)
            for replicate_id in range(canonical_replicates):
                rows.append({
                    "k": k,
                    "patient_index": patient_idx,
                    "actual_label": patient["actual_label"],
                    "family": "canonical",
                    "variant_id": 0,
                    "replicate_id": replicate_id,
                    "prompt": canonical,
                    "prompt_chars": len(canonical),
                    "prompt_tokens": token_count(canonical),
                })

            for family in families:
                if family not in PERTURBATIONS:
                    raise ValueError(f"Unknown perturbation family: {family}")
                perturb_func = PERTURBATIONS[family]
                for variant_id in range(variants):
                    perturb_seed = hash((seed, patient_idx, k, family, variant_id)) % (2**31)
                    prompt = context + perturb_func(features, seed=perturb_seed)
                    for replicate_id in range(perturb_replicates):
                        rows.append({
                            "k": k,
                            "patient_index": patient_idx,
                            "actual_label": patient["actual_label"],
                            "family": family,
                            "variant_id": variant_id,
                            "replicate_id": replicate_id,
                            "prompt": prompt,
                            "prompt_chars": len(prompt),
                            "prompt_tokens": token_count(prompt),
                        })
    return rows


def _majority(values: List[int]) -> Optional[int]:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return Counter(clean).most_common(1)[0][0]


def _canonical_stats(rows: List[Dict]) -> Dict:
    labels = [row["predicted_label"] for row in rows if row.get("predicted_label") is not None]
    confidences = [row["confidence"] for row in rows if row.get("confidence") is not None]
    label = _majority(labels)
    confidence = _mean(confidences)
    return {"label": label, "confidence": confidence}


def _summarize_patient_k(call_rows: List[Dict]) -> List[Dict]:
    output: List[Dict] = []
    grouped: Dict[Tuple[int, int], List[Dict]] = {}
    for row in call_rows:
        grouped.setdefault((row["k"], row["patient_index"]), []).append(row)

    for (k, patient_idx), rows in sorted(grouped.items()):
        actual_label = rows[0]["actual_label"]
        canonical_rows = [row for row in rows if row["family"] == "canonical"]
        perturb_rows = [row for row in rows if row["family"] != "canonical"]
        canonical = _canonical_stats(canonical_rows)

        if canonical["label"] is None:
            canonical_correct = math.nan
            canonical_p_true = math.nan
            brier_component = math.nan
        else:
            canonical_correct = int(canonical["label"] == actual_label)
            canonical_p_positive = (
                canonical["confidence"]
                if canonical["label"] == 1
                else 1.0 - canonical["confidence"]
            )
            canonical_p_true = (
                canonical_p_positive
                if actual_label == 1
                else 1.0 - canonical_p_positive
            )
            brier_component = (canonical_p_positive - actual_label) ** 2

        label_flips = [
            int(row["predicted_label"] != canonical["label"])
            for row in perturb_rows
            if canonical["label"] is not None and row.get("predicted_label") is not None
        ]
        conf_deltas = [
            abs(row["confidence"] - canonical["confidence"])
            for row in perturb_rows
            if canonical["confidence"] is not None and row.get("confidence") is not None
        ]

        output.append({
            "k": k,
            "patient_index": patient_idx,
            "actual_label": actual_label,
            "canonical_label": canonical["label"],
            "canonical_confidence": canonical["confidence"],
            "canonical_correct": canonical_correct,
            "canonical_p_true": canonical_p_true,
            "brier_component": brier_component,
            "perturb_calls": len(perturb_rows),
            "label_flip_rate": _mean(label_flips),
            "confidence_instability": _mean(conf_deltas),
            "prompt_tokens_mean": _mean(row["prompt_tokens"] for row in rows),
            "prompt_tokens_total": sum(row["prompt_tokens"] for row in rows),
            "prompt_chars_mean": _mean(row["prompt_chars"] for row in rows),
            "prompt_chars_total": sum(row["prompt_chars"] for row in rows),
        })
    return output


def _pairwise_auc(scores: List[float], labels: List[int]) -> float:
    clean = [
        (float(score), int(label))
        for score, label in zip(scores, labels)
        if score is not None and label is not None and not math.isnan(float(score))
    ]
    positives = [score for score, label in clean if label == 1]
    negatives = [score for score, label in clean if label == 0]
    if not positives or not negatives:
        return math.nan
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _summarize_k(patient_rows: List[Dict], call_rows: List[Dict]) -> List[Dict]:
    output: List[Dict] = []
    for k in sorted({row["k"] for row in patient_rows}):
        patients = [row for row in patient_rows if row["k"] == k]
        calls = [row for row in call_rows if row["k"] == k]
        canonical_calls = [row for row in calls if row["family"] == "canonical"]
        perturb_calls = [row for row in calls if row["family"] != "canonical"]

        p_positive = [
            row["canonical_confidence"] if row["canonical_label"] == 1 else 1.0 - row["canonical_confidence"]
            for row in patients
            if row["canonical_label"] is not None and not math.isnan(row["canonical_confidence"])
        ]
        labels = [
            row["actual_label"]
            for row in patients
            if row["canonical_label"] is not None and not math.isnan(row["canonical_confidence"])
        ]

        output.append({
            "k": k,
            "patients": len(patients),
            "canonical_calls": len(canonical_calls),
            "perturb_calls": len(perturb_calls),
            "total_calls": len(calls),
            "accuracy": _mean(row["canonical_correct"] for row in patients),
            "brier_score": _mean(row["brier_component"] for row in patients),
            "auroc": _pairwise_auc(p_positive, labels),
            "mean_p_true": _mean(row["canonical_p_true"] for row in patients),
            "label_flip_rate": _mean(row["label_flip_rate"] for row in patients),
            "confidence_instability": _mean(row["confidence_instability"] for row in patients),
            "mean_prompt_tokens": _mean(row["prompt_tokens"] for row in calls),
            "total_prompt_tokens": sum(row["prompt_tokens"] for row in calls),
            "mean_prompt_chars": _mean(row["prompt_chars"] for row in calls),
            "total_prompt_chars": sum(row["prompt_chars"] for row in calls),
        })
    return output


def _add_token_savings(k_rows: List[Dict]) -> None:
    by_k = {row["k"]: row for row in k_rows}
    k5_tokens = by_k.get(5, {}).get("total_prompt_tokens", math.nan)
    k3_tokens = by_k.get(3, {}).get("total_prompt_tokens", math.nan)
    k3_accuracy = by_k.get(3, {}).get("accuracy", math.nan)
    for row in k_rows:
        total_tokens = row["total_prompt_tokens"]
        row["token_savings_vs_k5"] = (
            k5_tokens - total_tokens if not math.isnan(k5_tokens) else math.nan
        )
        row["token_savings_pct_vs_k5"] = (
            (k5_tokens - total_tokens) / k5_tokens if k5_tokens else math.nan
        )
        row["token_delta_vs_k3"] = (
            total_tokens - k3_tokens if not math.isnan(k3_tokens) else math.nan
        )
        row["token_delta_pct_vs_k3"] = (
            (total_tokens - k3_tokens) / k3_tokens if k3_tokens else math.nan
        )
        row["accuracy_pp_per_1k_tokens"] = (
            100.0 * row["accuracy"] / (total_tokens / 1000.0)
            if total_tokens and not math.isnan(float(row["accuracy"]))
            else math.nan
        )
        row["accuracy_delta_vs_k3"] = (
            row["accuracy"] - k3_accuracy
            if not math.isnan(float(row["accuracy"])) and not math.isnan(float(k3_accuracy))
            else math.nan
        )
        row["accuracy_delta_per_1k_tokens_vs_k3"] = (
            1000.0 * row["accuracy_delta_vs_k3"] / row["token_delta_vs_k3"]
            if row["token_delta_vs_k3"] not in {0, None}
            and not math.isnan(float(row["token_delta_vs_k3"]))
            and not math.isnan(float(row["accuracy_delta_vs_k3"]))
            else math.nan
        )

    previous = None
    for row in sorted(k_rows, key=lambda item: item["k"]):
        if previous is None:
            row["accuracy_delta_vs_previous"] = math.nan
            row["token_delta_vs_previous"] = math.nan
            row["accuracy_delta_per_1k_tokens_vs_previous"] = math.nan
        else:
            row["accuracy_delta_vs_previous"] = (
                row["accuracy"] - previous["accuracy"]
                if not math.isnan(float(row["accuracy"])) and not math.isnan(float(previous["accuracy"]))
                else math.nan
            )
            row["token_delta_vs_previous"] = row["total_prompt_tokens"] - previous["total_prompt_tokens"]
            row["accuracy_delta_per_1k_tokens_vs_previous"] = (
                1000.0 * row["accuracy_delta_vs_previous"] / row["token_delta_vs_previous"]
                if row["token_delta_vs_previous"]
                and not math.isnan(float(row["accuracy_delta_vs_previous"]))
                else math.nan
            )
        previous = row


def _pairwise_rate_of_learning(k_rows: List[Dict]) -> List[Dict]:
    """Return pairwise accuracy gain per additional 1,000 prompt tokens."""
    rows = sorted(k_rows, key=lambda item: item["k"])
    output: List[Dict] = []
    for start in rows:
        rate_row = {"from_k": start["k"]}
        for end in rows:
            key = f"to_{end['k']}"
            if end["k"] <= start["k"]:
                rate_row[key] = math.nan
                continue
            accuracy_delta_pp = 100.0 * (end["accuracy"] - start["accuracy"])
            token_delta_1k = (end["total_prompt_tokens"] - start["total_prompt_tokens"]) / 1000.0
            rate_row[key] = accuracy_delta_pp / token_delta_1k if token_delta_1k else math.nan
        output.append(rate_row)
    return output


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
    k_rows: List[Dict],
    dry_run: bool,
    families: List[str],
    variants: int,
    canonical_replicates: int,
    perturb_replicates: int,
) -> str:
    k_table = _markdown_table(
        [
            "k",
            "Accuracy",
            "Mean tokens",
            "Total tokens",
        ],
        [
            [
                str(row["k"]),
                _percent(row["accuracy"]),
                _fmt(row["mean_prompt_tokens"], 1),
                str(int(row["total_prompt_tokens"])),
            ]
            for row in k_rows
        ],
    )

    sorted_rows = sorted(k_rows, key=lambda item: item["k"])
    rate_rows = _pairwise_rate_of_learning(sorted_rows) if not dry_run else []
    rate_headers = ["from/to"] + [str(row["k"]) for row in sorted_rows]
    rate_table = _markdown_table(
        rate_headers,
        [
            [str(row["from_k"])]
            + [
                "--" if math.isnan(float(row[f"to_{target['k']}"])) else _fmt(row[f"to_{target['k']}"], 3)
                for target in sorted_rows
            ]
            for row in rate_rows
        ],
    ) if rate_rows else ""

    status = "Dry run only. No LLM calls were made." if dry_run else "LLM run completed."
    family_text = ", ".join(families) if families else "canonical only"
    variant_line = (
        f"- Variants per family/patient/k: `{variants}`\n"
        f"- Perturbation replicates per variant: `{perturb_replicates}`\n"
        if families
        else ""
    )

    interpretation = (
        "Rate of Learning is defined as (accuracy_j - accuracy_i) / "
        "((tokens_j - tokens_i) / 1000), where accuracy is measured in percentage points. "
        "Use it to identify when extra context stops buying much additional accuracy."
    )

    if not dry_run and {1, 3, 5}.issubset({row["k"] for row in k_rows}):
        by_k = {row["k"]: row for row in k_rows}
        acc_gap_3_5 = by_k[3]["accuracy"] - by_k[5]["accuracy"]
        flip_gap_3_5 = by_k[3]["label_flip_rate"] - by_k[5]["label_flip_rate"]
        savings_3_5 = by_k[3]["token_savings_pct_vs_k5"]
        flip_text = (
            f", label-flip gap `{flip_gap_3_5:+.3f}`"
            if not math.isnan(float(flip_gap_3_5))
            else ""
        )
        interpretation += (
            f"\n\nObserved k=3 vs k=5: accuracy gap `{100.0 * acc_gap_3_5:+.1f}` percentage points"
            f"{flip_text}, token savings `{_percent(savings_3_5)}`. "
            f"The adjacent Rate of Learning is `{_fmt(by_k[3]['accuracy_delta_per_1k_tokens_vs_previous'] * 100, 3)}` "
            f"for k=1 to k=3 and `{_fmt(by_k[5]['accuracy_delta_per_1k_tokens_vs_previous'] * 100, 3)}` "
            f"for k=3 to k=5."
        )

    if not dry_run and 3 in {row["k"] for row in k_rows}:
        by_k = {row["k"]: row for row in k_rows}
        best_accuracy = max(row["accuracy"] for row in k_rows if not math.isnan(float(row["accuracy"])))
        best_ks = [row["k"] for row in k_rows if row["accuracy"] == best_accuracy]
        k3 = by_k[3]
        high_k_text = []
        for high_k in [20, 50]:
            if high_k in by_k:
                token_multiple = by_k[high_k]["total_prompt_tokens"] / k3["total_prompt_tokens"]
                acc_delta = by_k[high_k]["accuracy"] - k3["accuracy"]
                high_k_text.append(
                    f"k={high_k} uses `{token_multiple:.2f}x` the k=3 tokens for "
                    f"`{100.0 * acc_delta:.1f}` percentage points higher accuracy"
                )
        if high_k_text:
            interpretation += (
                f"\n\nHighest raw accuracy was `{_percent(best_accuracy)}` at "
                f"`k={', '.join(str(k) for k in best_ks)}`. "
                + "; ".join(high_k_text)
                + ". This supports reporting k=3 as the efficient operating point, not as the raw maximum."
            )

    return (
        "# Retrieval k Ablation\n\n"
        f"{status}\n\n"
        f"- k values: `{', '.join(str(row['k']) for row in k_rows)}`\n"
        f"- Perturbation families: `{family_text}`\n"
        f"- Canonical replicates per patient/k: `{canonical_replicates}`\n"
        f"{variant_line}\n"
        "## Summary By k\n\n"
        f"{k_table}\n\n"
        "## Pairwise Rate of Learning\n\n"
        f"{rate_table}\n\n"
        "## Interpretation\n\n"
        f"{interpretation}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-context k ablation.")
    parser.add_argument("--run", action="store_true", help="Actually call the configured LLM.")
    parser.add_argument("--dataset", default="data/test_ready.csv", help="Prediction dataset.")
    parser.add_argument("--context", default="data/context_ready.csv", help="Retrieval context dataset.")
    parser.add_argument("--k-values", default="1,3,5,10,20,50", help="Comma-separated k values.")
    parser.add_argument("--families", default="none", help="Comma-separated perturbation families, or none.")
    parser.add_argument("--variants", type=int, default=3, help="Perturbation variants per family.")
    parser.add_argument("--canonical-replicates", type=int, default=3, help="Canonical replicates per patient/k.")
    parser.add_argument("--perturb-replicates", type=int, default=1, help="Perturbation replicates per prompt.")
    parser.add_argument("--patient-indices", default=None, help="Optional comma-separated zero-based patient indices.")
    parser.add_argument("--max-patients", type=int, default=None, help="Optional first-N patient cap.")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding if available.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = _parse_int_list(args.k_values)
    families = _parse_str_list(args.families)
    token_count = _get_token_counter(args.encoding)

    patients = _select_patients(
        _load_patients(repo_root, args.dataset),
        patient_indices=args.patient_indices,
        max_patients=args.max_patients,
    )
    context_df = load_data(str(repo_root / args.context))

    plan_rows = _build_prompt_plan(
        patients=patients,
        context_df=context_df,
        k_values=k_values,
        families=families,
        variants=args.variants,
        canonical_replicates=args.canonical_replicates,
        perturb_replicates=args.perturb_replicates,
        seed=args.seed,
        token_count=token_count,
    )

    dry_k_rows = []
    for k in k_values:
        subset = [row for row in plan_rows if row["k"] == k]
        dry_k_rows.append({
            "k": k,
            "patients": len(patients),
            "canonical_calls": sum(row["family"] == "canonical" for row in subset),
            "perturb_calls": sum(row["family"] != "canonical" for row in subset),
            "total_calls": len(subset),
            "accuracy": math.nan,
            "brier_score": math.nan,
            "auroc": math.nan,
            "mean_p_true": math.nan,
            "label_flip_rate": math.nan,
            "confidence_instability": math.nan,
            "mean_prompt_tokens": _mean(row["prompt_tokens"] for row in subset),
            "total_prompt_tokens": sum(row["prompt_tokens"] for row in subset),
            "mean_prompt_chars": _mean(row["prompt_chars"] for row in subset),
            "total_prompt_chars": sum(row["prompt_chars"] for row in subset),
        })
    _add_token_savings(dry_k_rows)

    print(f"Patients: {len(patients)}")
    print(f"k values: {k_values}")
    print(f"Families: {families if families else ['canonical only']}")
    print(f"Planned LLM calls: {len(plan_rows)}")
    for row in dry_k_rows:
        print(
            f"  k={row['k']}: calls={row['total_calls']}, "
            f"mean_tokens={row['mean_prompt_tokens']:.1f}, "
            f"total_tokens={int(row['total_prompt_tokens'])}"
        )

    if not args.run:
        print("\nDRY RUN ONLY: no LLM calls were made. Add --run to execute.")
        (out_dir / "retrieval_k_ablation_report.md").write_text(
            _render_report(
                dry_k_rows,
                dry_run=True,
                families=families,
                variants=args.variants,
                canonical_replicates=args.canonical_replicates,
                perturb_replicates=args.perturb_replicates,
            ),
            encoding="utf-8",
        )
        _write_csv(
            out_dir / "retrieval_k_ablation_plan.csv",
            plan_rows,
            ["k", "patient_index", "actual_label", "family", "variant_id", "replicate_id", "prompt_chars", "prompt_tokens", "prompt"],
        )
        return

    config = load_config()
    if args.temperature is not None:
        config.temperature = args.temperature
    print_config(config)
    client = initialize_client(config.llm_provider, config.azure_api_version)

    plan_keys = {_call_key(plan) for plan in plan_rows}
    existing_calls = _load_existing_calls(out_dir / "retrieval_k_ablation_calls.csv", plan_keys)
    if existing_calls:
        print(f"Resuming from {len(existing_calls)} existing completed calls.")

    call_rows_by_key: Dict[Tuple[int, int, str, int, int], Dict] = dict(existing_calls)
    pending_rows = [plan for plan in plan_rows if _call_key(plan) not in call_rows_by_key]
    print(f"Pending LLM calls: {len(pending_rows)}")

    for idx, plan in enumerate(plan_rows, 1):
        key = _call_key(plan)
        if key in call_rows_by_key:
            continue

        label, confidence, explanation, decision = call_llm(
            client,
            config.llm_provider,
            config.llm_model,
            plan["prompt"],
            temperature=config.temperature,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
        )
        confidence_prob = _clamp_probability(_safe_float(confidence) / 100.0)
        actual_label = int(plan["actual_label"])
        predicted_label = int(label)
        correct = int(predicted_label == actual_label)
        p_true = confidence_prob if correct else 1.0 - confidence_prob

        row = dict(plan)
        row.update({
            "predicted_label": predicted_label,
            "confidence": confidence_prob,
            "correct": correct,
            "p_true_label": p_true,
            "decision": decision,
            "explanation": explanation,
        })
        call_rows_by_key[key] = row

        if idx % 50 == 0 or idx == len(plan_rows):
            print(f"Completed/available {idx}/{len(plan_rows)} planned calls")

    call_rows = [call_rows_by_key[_call_key(plan)] for plan in plan_rows if _call_key(plan) in call_rows_by_key]

    patient_rows = _summarize_patient_k(call_rows)
    k_rows = _summarize_k(patient_rows, call_rows)
    _add_token_savings(k_rows)

    _write_csv(
        out_dir / "retrieval_k_ablation_calls.csv",
        call_rows,
        [
            "k",
            "patient_index",
            "actual_label",
            "family",
            "variant_id",
            "replicate_id",
            "prompt_chars",
            "prompt_tokens",
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
        out_dir / "retrieval_k_ablation_patient_summary.csv",
        patient_rows,
        [
            "k",
            "patient_index",
            "actual_label",
            "canonical_label",
            "canonical_confidence",
            "canonical_correct",
            "canonical_p_true",
            "brier_component",
            "perturb_calls",
            "label_flip_rate",
            "confidence_instability",
            "prompt_tokens_mean",
            "prompt_tokens_total",
            "prompt_chars_mean",
            "prompt_chars_total",
        ],
    )
    _write_csv(
        out_dir / "retrieval_k_ablation_k_summary.csv",
        k_rows,
        [
            "k",
            "patients",
            "canonical_calls",
            "perturb_calls",
            "total_calls",
            "accuracy",
            "brier_score",
            "auroc",
            "mean_p_true",
            "label_flip_rate",
            "confidence_instability",
            "mean_prompt_tokens",
            "total_prompt_tokens",
            "mean_prompt_chars",
            "total_prompt_chars",
            "token_savings_vs_k5",
            "token_savings_pct_vs_k5",
            "token_delta_vs_k3",
            "token_delta_pct_vs_k3",
            "accuracy_pp_per_1k_tokens",
            "accuracy_delta_vs_k3",
            "accuracy_delta_per_1k_tokens_vs_k3",
            "accuracy_delta_vs_previous",
            "token_delta_vs_previous",
            "accuracy_delta_per_1k_tokens_vs_previous",
        ],
    )
    rate_rows = _pairwise_rate_of_learning(k_rows)
    rate_fields = ["from_k"] + [f"to_{row['k']}" for row in sorted(k_rows, key=lambda item: item["k"])]
    _write_csv(
        out_dir / "retrieval_k_ablation_rate_of_learning.csv",
        rate_rows,
        rate_fields,
    )
    (out_dir / "retrieval_k_ablation_report.md").write_text(
        _render_report(
            k_rows,
            dry_run=False,
            families=families,
            variants=args.variants,
            canonical_replicates=args.canonical_replicates,
            perturb_replicates=args.perturb_replicates,
        ),
        encoding="utf-8",
    )

    print(f"Wrote outputs to {_console_safe(out_dir)}")


if __name__ == "__main__":
    main()
