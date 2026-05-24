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
Measure empirical evidence for Theorem 2: the local stability/Lipschitz bound.

This script does not make any LLM calls. It reuses saved experiment outputs and
the prompt-distance values already computed for the threshold analysis.

Plain-language question:
    Do larger prompt perturbation distances correspond to bounded output changes?

Outputs:
    - new/local_stability_call_metrics.csv
    - new/local_stability_summary.csv
    - new/local_stability_report.md
    - new/local_stability_table.tex
    - paper/figures/figure_local_stability_tier1_without.pdf if matplotlib is available
    - paper/figures/figure_local_stability_tier1_with.pdf if matplotlib is available
    - paper/figures/figure_local_stability_tier2_without.pdf if matplotlib is available
    - paper/figures/figure_local_stability_tier2_with.pdf if matplotlib is available

Run from the repository root:
    .venv\\Scripts\\python.exe new\\measure_local_stability_bound.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


SETTING_SPECS = (
    ("Tier 1", "without", "outputs", "output_tier_1_without_context_0.json"),
    ("Tier 1", "with", "outputs", "output_tier_1_with_context_0.json"),
    ("Tier 2", "without", "outputs_correct", "output_tier_2_without_context_0.json"),
    ("Tier 2", "with", "outputs_correct", "output_tier_2_with_context_0.json"),
)

MIN_RATIO_DISTANCE = 0.005


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


def _median(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(np.median(clean)) if clean else math.nan


def _quantile(values: Iterable[float], q: float) -> float:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(np.quantile(clean, q)) if clean else math.nan


def _percent(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "N/A"
    return f"{100.0 * float(value):.1f}%"


def _percent_latex(value: float) -> str:
    return _percent(value).replace("%", "\\%")


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or math.isnan(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _console_safe(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


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
    return [entry for _, entry in _iter_entries_with_instance(patient_data.get("canonical"))]


def _canonical_by_replicate(patient_data: Dict) -> Tuple[List[Dict], Dict[int, Dict]]:
    entries = _canonical_entries(patient_data)
    by_replicate = {}
    for entry in entries:
        replicate_id = _safe_int(entry.get("replicate_id"), 0) or 0
        by_replicate[replicate_id] = entry
    return entries, by_replicate


def _fallback_canonical(canonical_entries: List[Dict], replicate_id: int) -> Optional[Dict]:
    if not canonical_entries:
        return None
    if 0 <= replicate_id < len(canonical_entries):
        return canonical_entries[replicate_id]
    return canonical_entries[0]


def _positive_probability(label: int, confidence: float) -> float:
    return confidence if label == 1 else 1.0 - confidence


def _decision_change(a: Dict, b: Dict) -> int:
    return int(str(a.get("decision", "")).strip().lower() != str(b.get("decision", "")).strip().lower())


def _distance_key(row: Dict) -> Tuple[str, str, int, str, int, int]:
    return (
        row["tier"],
        row["context"],
        int(row["patient_idx"]),
        row["family"],
        int(row["instance_id"]),
        int(row["replicate_id"]),
    )


def _load_precomputed_prompt_distances(repo_root: Path) -> Dict[Tuple[str, str, int, str, int, int], float]:
    metrics_path = repo_root / "new" / "theorem_threshold_call_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}. Run new\\measure_theorem_threshold.py first."
        )

    distances: Dict[Tuple[str, str, int, str, int, int], float] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            distances[_distance_key(row)] = _safe_float(row.get("perturbation_distance"))
    return distances


def _load_call_rows(repo_root: Path) -> List[Dict]:
    prompt_distances = _load_precomputed_prompt_distances(repo_root)
    rows: List[Dict] = []

    for tier, context, folder_name, file_name in SETTING_SPECS:
        result_path = repo_root / folder_name / file_name
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result file: {result_path}")

        with result_path.open(encoding="utf-8") as handle:
            result_data = json.load(handle)

        for patient_key, patient_data in result_data.get("results_by_patient", {}).items():
            patient_idx = _patient_idx_from_key(patient_key)
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
                    canonical_label = _safe_int(canonical.get("label"))
                    confidence = _clamp_probability(_safe_float(entry.get("confidence")) / 100.0)
                    canonical_confidence = _clamp_probability(
                        _safe_float(canonical.get("confidence")) / 100.0
                    )
                    if (
                        predicted_label is None
                        or canonical_label is None
                        or math.isnan(confidence)
                        or math.isnan(canonical_confidence)
                    ):
                        continue

                    key = (tier, context, patient_idx, family_name, instance_id, replicate_id)
                    prompt_distance = prompt_distances.get(key, math.nan)
                    if math.isnan(prompt_distance):
                        continue

                    label_change = int(predicted_label != canonical_label)
                    confidence_delta = abs(confidence - canonical_confidence)
                    p_positive = _positive_probability(predicted_label, confidence)
                    canonical_p_positive = _positive_probability(canonical_label, canonical_confidence)
                    p_positive_delta = abs(p_positive - canonical_p_positive)
                    decision_change = _decision_change(canonical, entry)

                    output_distance = math.sqrt(
                        label_change ** 2
                        + confidence_delta ** 2
                        + p_positive_delta ** 2
                        + decision_change ** 2
                    )
                    ratio = (
                        output_distance / prompt_distance
                        if prompt_distance > 1e-12
                        else math.nan
                    )

                    rows.append({
                        "tier": tier,
                        "context": context,
                        "source_file": str(result_path.relative_to(repo_root)),
                        "patient_idx": patient_idx,
                        "patient_number": patient_idx + 1,
                        "family": family_name,
                        "instance_id": instance_id,
                        "replicate_id": replicate_id,
                        "prompt_distance": prompt_distance,
                        "canonical_label": canonical_label,
                        "predicted_label": predicted_label,
                        "label_change": label_change,
                        "canonical_confidence": canonical_confidence,
                        "confidence": confidence,
                        "confidence_delta": confidence_delta,
                        "canonical_p_positive": canonical_p_positive,
                        "p_positive": p_positive,
                        "p_positive_delta": p_positive_delta,
                        "decision_change": decision_change,
                        "output_distance": output_distance,
                        "lipschitz_ratio": ratio,
                    })

    return rows


def _pearson(x: List[float], y: List[float]) -> float:
    clean = [(float(a), float(b)) for a, b in zip(x, y) if not math.isnan(a) and not math.isnan(b)]
    if len(clean) < 2:
        return math.nan
    xs = np.array([a for a, _ in clean], dtype=float)
    ys = np.array([b for _, b in clean], dtype=float)
    if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return math.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def _summarize_group(rows: List[Dict], group_name: str) -> Dict:
    ratio_rows = [row for row in rows if row["prompt_distance"] >= MIN_RATIO_DISTANCE]
    near_zero_rows = [
        row for row in rows
        if not math.isnan(row["prompt_distance"]) and row["prompt_distance"] < MIN_RATIO_DISTANCE
    ]
    ratios = [row["lipschitz_ratio"] for row in ratio_rows]
    prompt_distances = [row["prompt_distance"] for row in rows]
    output_distances = [row["output_distance"] for row in rows]
    l95 = _quantile(ratios, 0.95)
    violations_l95 = [
        row for row in ratio_rows
        if not math.isnan(l95) and row["output_distance"] > l95 * row["prompt_distance"] + 1e-12
    ]

    return {
        "group": group_name,
        "n": len(rows),
        "n_ratio": len(ratio_rows),
        "near_zero_fraction": len(near_zero_rows) / len(rows) if rows else math.nan,
        "near_zero_label_change": _mean(row["label_change"] for row in near_zero_rows),
        "near_zero_output_distance": _mean(row["output_distance"] for row in near_zero_rows),
        "mean_prompt_distance": _mean(prompt_distances),
        "mean_output_distance": _mean(output_distances),
        "mean_label_change": _mean(row["label_change"] for row in rows),
        "mean_confidence_delta": _mean(row["confidence_delta"] for row in rows),
        "mean_p_positive_delta": _mean(row["p_positive_delta"] for row in rows),
        "median_lipschitz_ratio": _median(ratios),
        "lipschitz_q95": l95,
        "empirical_lipschitz_max": max(ratios) if ratios else math.nan,
        "coverage_at_l95": 1.0 - (len(violations_l95) / len(ratio_rows) if ratio_rows else math.nan),
        "pearson_distance_output": _pearson(prompt_distances, output_distances),
    }


def _build_summary(rows: List[Dict]) -> List[Dict]:
    summary: List[Dict] = []
    summary.append(_summarize_group(rows, "All"))

    by_setting: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    by_family: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        by_setting[(row["tier"], row["context"])].append(row)
        by_family[(row["tier"], row["context"], row["family"])].append(row)

    for (tier, context), group_rows in sorted(by_setting.items()):
        summary.append(_summarize_group(group_rows, f"{tier} {context}"))

    for (tier, context, family), group_rows in sorted(by_family.items()):
        summary.append(_summarize_group(group_rows, f"{tier} {context} {family}"))

    return summary


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


def _setting_rows(summary_rows: List[Dict]) -> List[Dict]:
    return [
        row for row in summary_rows
        if row["group"] in {
            "Tier 1 without",
            "Tier 1 with",
            "Tier 2 without",
            "Tier 2 with",
        }
    ]


def _render_report(rows: List[Dict], summary_rows: List[Dict]) -> str:
    all_row = summary_rows[0]
    setting_table = _markdown_table(
        [
            "Setting",
            "N",
            f"N d>={MIN_RATIO_DISTANCE:g}",
            "Mean input dist.",
            "Mean output dist.",
            "Label change",
            "Near-zero label",
            "Conf. delta",
            "L95",
            "Lmax",
            "Coverage @L95",
            "Pearson r",
        ],
        [
            [
                row["group"],
                str(row["n"]),
                str(row["n_ratio"]),
                _fmt(row["mean_prompt_distance"]),
                _fmt(row["mean_output_distance"]),
                _percent(row["mean_label_change"]),
                _percent(row["near_zero_label_change"]),
                _fmt(row["mean_confidence_delta"]),
                _fmt(row["lipschitz_q95"], 2),
                _fmt(row["empirical_lipschitz_max"], 2),
                _percent(row["coverage_at_l95"]),
                _fmt(row["pearson_distance_output"], 3),
            ]
            for row in _setting_rows(summary_rows)
        ],
    )

    family_candidates = [
        row for row in summary_rows
        if row["group"].count(" ") >= 3
    ]
    top_families = sorted(
        family_candidates,
        key=lambda row: row["lipschitz_q95"],
        reverse=True,
    )[:8]
    family_table = _markdown_table(
        [
            "Group",
            "N",
            "Mean input dist.",
            "Mean output dist.",
            "L95",
            "Lmax",
        ],
        [
            [
                row["group"],
                str(row["n"]),
                _fmt(row["mean_prompt_distance"]),
                _fmt(row["mean_output_distance"]),
                _fmt(row["lipschitz_q95"], 2),
                _fmt(row["empirical_lipschitz_max"], 2),
            ]
            for row in top_families
        ],
    )

    return (
        "# Local Stability Bound Measurement\n\n"
        "This measures empirical evidence for Theorem 2 using saved perturbation outputs. "
        "No LLM calls are made.\n\n"
        "## Definition\n\n"
        "- Input distance: cosine distance between the canonical prompt embedding and the perturbed prompt embedding.\n"
        "- Output distance: Euclidean distance over four normalized output channels: label change, confidence change, positive-class probability change, and defer/predict decision change.\n"
        "- Empirical Lipschitz ratio: `output distance / input distance`.\n"
        f"- `L95`: the 95th percentile empirical Lipschitz ratio after excluding near-zero input distances (`d_in < {MIN_RATIO_DISTANCE:g}`). `Lmax`: the corresponding maximum observed ratio.\n\n"
        "## Overall Result\n\n"
        f"- Calls measured: `{len(rows)}`\n"
        f"- Mean input distance: `{_fmt(all_row['mean_prompt_distance'])}`\n"
        f"- Mean output distance: `{_fmt(all_row['mean_output_distance'])}`\n"
        f"- Near-zero input distance fraction: `{_percent(all_row['near_zero_fraction'])}`\n"
        f"- Label-change rate among near-zero input distances: `{_percent(all_row['near_zero_label_change'])}`\n"
        f"- Median Lipschitz ratio: `{_fmt(all_row['median_lipschitz_ratio'], 2)}`\n"
        f"- 95th percentile Lipschitz ratio: `{_fmt(all_row['lipschitz_q95'], 2)}`\n"
        f"- Maximum observed Lipschitz ratio: `{_fmt(all_row['empirical_lipschitz_max'], 2)}`\n"
        f"- Pearson correlation between input and output distance: `{_fmt(all_row['pearson_distance_output'])}`\n\n"
        "## Setting Summary\n\n"
        f"{setting_table}\n\n"
        "## Highest-Ratio Family Groups\n\n"
        f"{family_table}\n\n"
        "## Interpretation\n\n"
        "These results provide a direct empirical application of Theorem 2. The theorem is a "
        "conditional bound, so an empirical maximum ratio always defines a finite observed "
        "bound on the tested domain. The scientifically useful quantity is therefore not "
        "whether a bound exists, but how large and stable the required bound is. Smaller "
        "L95/Lmax values indicate smoother local behavior; very large ratios indicate brittle "
        "behavior where small prompt movements can cause disproportionately large output "
        "changes. Near-zero input distances are reported separately because they are the most "
        "important stress case for local smoothness: output changes at near-zero embedding "
        "distance imply a very large local constant.\n"
    )


def _render_latex_table(summary_rows: List[Dict]) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Empirical local stability bound.} Input distance is the prompt-embedding cosine distance; output distance combines label, confidence, positive-class probability and decision changes. Smaller Lipschitz ratios indicate smoother local behaviour.}",
        "\\label{tab:local_stability_bound}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        f"Setting & $n$ & Mean $d_{{in}}$ & Mean $d_{{out}}$ & Label change & $L_{{95}}$ & $L_{{max}}$ \\\\",
        "\\midrule",
    ]
    for row in _setting_rows(summary_rows):
        lines.append(
            f"{row['group']} & {row['n']} & {_fmt(row['mean_prompt_distance'])} & "
            f"{_fmt(row['mean_output_distance'])} & {_percent_latex(row['mean_label_change'])} & "
            f"{_fmt(row['lipschitz_q95'], 2)} & {_fmt(row['empirical_lipschitz_max'], 2)} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def _write_plot(rows: List[Dict], output_path: Path) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        return f"matplotlib unavailable: {exc}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure_specs = [
        ("Tier 1", "without", "figure_local_stability_tier1_without.pdf"),
        ("Tier 1", "with", "figure_local_stability_tier1_with.pdf"),
        ("Tier 2", "without", "figure_local_stability_tier2_without.pdf"),
        ("Tier 2", "with", "figure_local_stability_tier2_with.pdf"),
    ]

    for tier, context, file_name in figure_specs:
        subset = [row for row in rows if row["tier"] == tier and row["context"] == context]
        if not subset:
            continue

        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        x = np.array([row["prompt_distance"] for row in subset], dtype=float)
        y = np.array([row["output_distance"] for row in subset], dtype=float)
        ax.scatter(x, y, s=5, alpha=0.20, color="#1f77b4", linewidths=0)

        l95 = _quantile(
            [
                row["lipschitz_ratio"] for row in subset
                if row["prompt_distance"] >= MIN_RATIO_DISTANCE
            ],
            0.95,
        )
        if not math.isnan(l95):
            raw_x_max = float(np.nanmax(x)) if len(x) else 1.0
            x_limit = max(0.01, math.ceil(raw_x_max * 100.0) / 100.0)
            y_limit = l95 * x_limit
            xs = np.linspace(0, x_limit, 200)
            ax.plot(
                xs,
                l95 * xs,
                linewidth=1.1,
                color="black",
                linestyle="--",
            )
        else:
            x_limit = max(0.01, math.ceil(float(np.nanmax(x)) * 100.0) / 100.0)
            y_limit = max(1.0, float(np.nanmax(y)) * 1.05)

        ax.set_xlabel("Input distance")
        ax.set_ylabel("Output distance")
        ax.set_xlim(0, x_limit)
        ax.set_ylim(0, y_limit)
        ax.set_box_aspect(1)
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=5)
        if not math.isnan(l95):
            ax.text(
                0.04,
                0.96,
                rf"$L_{{95}}$: 95th pct. $d_{{out}}/d_{{in}}$ = {l95:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", direction="out", top=False, right=False)
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(output_path.parent / file_name, bbox_inches="tight")
        plt.close(fig)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure empirical local stability bounds.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("new"))
    parser.add_argument("--figure-path", type=Path, default=Path("paper/figures/figure_local_stability_lipschitz.pdf"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_call_rows(repo_root)
    summary_rows = _build_summary(rows)

    call_fields = [
        "tier",
        "context",
        "source_file",
        "patient_number",
        "patient_idx",
        "family",
        "instance_id",
        "replicate_id",
        "prompt_distance",
        "canonical_label",
        "predicted_label",
        "label_change",
        "canonical_confidence",
        "confidence",
        "confidence_delta",
        "canonical_p_positive",
        "p_positive",
        "p_positive_delta",
        "decision_change",
        "output_distance",
        "lipschitz_ratio",
    ]
    summary_fields = [
        "group",
        "n",
        "n_ratio",
        "near_zero_fraction",
        "near_zero_label_change",
        "near_zero_output_distance",
        "mean_prompt_distance",
        "mean_output_distance",
        "mean_label_change",
        "mean_confidence_delta",
        "mean_p_positive_delta",
        "median_lipschitz_ratio",
        "lipschitz_q95",
        "empirical_lipschitz_max",
        "coverage_at_l95",
        "pearson_distance_output",
    ]

    _write_csv(output_dir / "local_stability_call_metrics.csv", rows, call_fields)
    _write_csv(output_dir / "local_stability_summary.csv", summary_rows, summary_fields)
    (output_dir / "local_stability_report.md").write_text(
        _render_report(rows, summary_rows),
        encoding="utf-8",
    )
    (output_dir / "local_stability_table.tex").write_text(
        _render_latex_table(summary_rows),
        encoding="utf-8",
    )

    plot_error = _write_plot(rows, repo_root / args.figure_path)
    if plot_error:
        print(f"Plot skipped: {plot_error}")

    print(f"Measured {len(rows)} perturbed calls.")
    print(f"Wrote {_console_safe(output_dir / 'local_stability_report.md')}")


if __name__ == "__main__":
    main()
