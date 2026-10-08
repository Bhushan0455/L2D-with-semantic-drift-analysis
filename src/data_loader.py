"""
Data loading and preprocessing utilities for L2D-SDDA
"""

from typing import List, Tuple
import pandas as pd
from src.config import DATASET_PATH, LEAK_COLS, TARGET_COL


def load_dataset(filepath=None) -> pd.DataFrame:
    """Loads the processed dataset from CSV."""
    path = filepath or DATASET_PATH
    if not path or not pd.io.common.file_exists(str(path)):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Returns the list of predictive features excluding leaks and target."""
    return [c for c in df.columns if c not in LEAK_COLS + [TARGET_COL]]


def get_window_data(df: pd.DataFrame, window_id: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Extracts features (X) and target (y) for a specified deployment window."""
    features = get_feature_columns(df)
    subset = df[df["window_id"] == window_id]
    X = subset[features]
    y = subset[TARGET_COL]
    return X, y
