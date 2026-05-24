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
Retrieval module — provides nearest-neighbour context examples.

Used by Family F (retrieval perturbations) to append similar patient
records from the training set to the prompt, simulating a RAG pipeline.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.data import load_data


# Cache so we only load + normalise once
_context_cache: Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]] = None


def load_context_data(path: str = "data/context_ready.csv") -> pd.DataFrame:
    """Load the context (training) dataset."""
    return load_data(path)


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns (exclude label and ID columns)."""
    exclude = {"patient_diagnosis", "Patient", "sample_id"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in (np.float64, np.int64, float, int)]


def _prepare_context(context_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Normalise features for distance calculation. Returns (df, feature_matrix, mean, std)."""
    global _context_cache
    if _context_cache is not None:
        return _context_cache

    feat_cols = _get_feature_columns(context_df)
    matrix = context_df[feat_cols].values.astype(float)

    # Z-score normalisation (avoid division by zero)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0

    _context_cache = (context_df, matrix, mean, std)
    return _context_cache


def get_nearest_neighbours(
    patient_features: Dict[str, float],
    context_df: pd.DataFrame,
    k: int = 3,
    seed: Optional[int] = None,
    diversity_penalty: float = 0.0,
) -> List[Dict]:
    """Find the K nearest neighbours from the context dataset.

    Parameters
    ----------
    patient_features : dict
        Feature values for the query patient.
    context_df : pd.DataFrame
        The full context/training dataset.
    k : int
        Number of neighbours to return.
    seed : int, optional
        Random seed (used when diversity_penalty > 0 to break ties).
    diversity_penalty : float
        0.0 = pure nearest-neighbour.  Higher values penalise selecting
        neighbours that are close to already-selected neighbours, producing
        a more diverse support set.

    Returns
    -------
    list of dict
        Each dict has feature names + "label" key.
    """
    df, matrix, mean, std = _prepare_context(context_df)
    feat_cols = _get_feature_columns(df)

    # Build query vector in same feature order
    query = np.array([patient_features.get(c, 0.0) for c in feat_cols])
    query_norm = (query - mean) / std

    # Euclidean distances
    normed = (matrix - mean) / std
    dists = np.linalg.norm(normed - query_norm, axis=1)

    if diversity_penalty <= 0.0:
        # Simple top-K
        indices = np.argsort(dists)[:k]
    else:
        # Greedy diverse selection
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(0)

        remaining = set(range(len(dists)))
        selected: List[int] = []

        for _ in range(min(k, len(dists))):
            best_idx = None
            best_score = np.inf
            for idx in remaining:
                score = dists[idx]
                # Add diversity penalty: discourage picking neighbours
                # that are close to already-selected ones
                for sel in selected:
                    neighbour_dist = np.linalg.norm(normed[idx] - normed[sel])
                    score -= diversity_penalty * neighbour_dist
                if score < best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None:
                selected.append(best_idx)
                remaining.discard(best_idx)

        indices = selected

    # Build result dicts
    neighbours = []
    for idx in indices:
        row = df.iloc[idx]
        entry = {c: float(row[c]) for c in feat_cols}
        if "patient_diagnosis" in df.columns:
            entry["label"] = int(row["patient_diagnosis"])
        neighbours.append(entry)

    return neighbours


def format_neighbours_as_context(neighbours: List[Dict]) -> str:
    """Format neighbour records as a context string to prepend to prompts."""
    lines = ["Here are similar patient cases for reference:\n"]
    for i, nbr in enumerate(neighbours, 1):
        label = nbr.pop("label", None)
        parts = [f"{k}: {v}" for k, v in nbr.items()]
        outcome = f" -> Outcome: {label}" if label is not None else ""
        lines.append(f"Case {i}: {'; '.join(parts)}{outcome}")
    lines.append("\nNow predict for the following patient:\n")
    return "\n".join(lines)
