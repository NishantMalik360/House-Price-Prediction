"""
=========================================================
AI House Price Prediction System

File: data_loader.py

Purpose:
Load datasets from the configured project paths.
=========================================================
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_FILE


def load_data(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load dataset from CSV.

    Parameters
    ----------
    path : Path
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    return df