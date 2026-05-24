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

import os

import numpy as np
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV file and clean numeric strings.
    
    Handles periods as thousands separators (e.g. 3.529.648 -> 3529648.0)
    """
    df = pd.read_csv(file_path)
    df = clean_numeric(df)
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric strings with period separators to floats."""
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .replace({"nan": np.nan})
            )
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass
    
    return df
