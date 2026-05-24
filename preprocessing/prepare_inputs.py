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
Validate and optionally write cleaned copies of the packaged CSV inputs.

This script mirrors the numeric cleaning used by utils.data.load_data and adds
explicit dataset checks for reproducibility. It is non-destructive by default.

Examples:
    python preprocessing/prepare_inputs.py --check-only
    python preprocessing/prepare_inputs.py --write-cleaned --output-dir data_prepared
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

PANCREATIC_FEATURES = [
    "age",
    "plasma_CA19_9",
    "creatinine",
    "LYVE1",
    "REG1B",
    "TFF1",
    "REG1A",
]

DATASET_SPECS = {
    "test_ready.csv": {
        "required": ["Patient", *PANCREATIC_FEATURES],
        "rows": 63,
        "label": "Pancreatic evaluation cohort",
    },
    "context_ready.csv": {
        "required": [*PANCREATIC_FEATURES, "patient_diagnosis"],
        "rows": 146,
        "label": "Pancreatic retrieval/context cohort",
    },
    "actual.csv": {
        "required": ["Patient", "patient_diagnosis"],
        "rows": 63,
        "label": "Pancreatic evaluation labels",
    },
    "breast_cancer_ready.csv": {
        "required": ["Patient", "patient_diagnosis"],
        "rows": 569,
        "label": "Wisconsin breast cancer extension cohort",
    },
}


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking string columns using the runtime repo rule."""
    cleaned = df.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            normalized = (
                cleaned[column]
                .astype(str)
                .str.replace(".", "", regex=False)
                .replace({"nan": np.nan})
            )
            try:
                cleaned[column] = normalized.astype(float)
            except ValueError:
                cleaned[column] = cleaned[column]
    return cleaned


def validate_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> List[str]:
    missing = [column for column in required if column not in df.columns]
    if missing:
        return [f"{path.name}: missing required columns {missing}"]
    return []


def validate_row_count(df: pd.DataFrame, expected_rows: int, path: Path) -> List[str]:
    if len(df) != expected_rows:
        return [f"{path.name}: expected {expected_rows} rows, found {len(df)}"]
    return []


def validate_dataset(path: Path, spec: Dict) -> List[str]:
    if not path.exists():
        return [f"{path.name}: file is missing"]

    df = pd.read_csv(path)
    errors = []
    errors.extend(validate_columns(df, spec["required"], path))
    errors.extend(validate_row_count(df, spec["rows"], path))

    cleaned = clean_numeric(df)
    numeric_columns = [
        column
        for column in cleaned.columns
        if column not in {"Patient", "sample_id"} and column in spec["required"]
    ]
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            errors.append(f"{path.name}: column {column} could not be parsed as numeric")

    return errors


def write_cleaned_file(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = clean_numeric(pd.read_csv(input_path))
    cleaned.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and clean packaged CSV inputs.")
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data_prepared")
    parser.add_argument("--check-only", action="store_true", help="Validate only; do not write cleaned copies.")
    parser.add_argument("--write-cleaned", action="store_true", help="Write cleaned copies to --output-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    write_cleaned = args.write_cleaned and not args.check_only

    all_errors: List[str] = []
    for filename, spec in DATASET_SPECS.items():
        path = input_dir / filename
        errors = validate_dataset(path, spec)
        all_errors.extend(errors)
        status = "OK" if not errors else "FAIL"
        print(f"{status}: {filename} - {spec['label']}")
        if write_cleaned and path.exists():
            write_cleaned_file(path, output_dir / filename)

    if write_cleaned:
        print(f"Wrote cleaned copies to {output_dir}")

    if all_errors:
        print("\nValidation errors:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("All required packaged inputs validated.")


if __name__ == "__main__":
    main()
