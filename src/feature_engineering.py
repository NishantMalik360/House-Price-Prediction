"""
=========================================================
AI House Price Prediction System

File: feature_engineering.py

Purpose:
Prepare cleaned data for Machine Learning.
=========================================================
"""

import pandas as pd


def prepare_features(df: pd.DataFrame):
    """
    Prepare features and target.

    Returns
    -------
    X : Features
    y : Target
    """

    # Drop columns not required
    df = df.drop(columns=["availability", "size"], errors="ignore")

    # One-Hot Encoding
    df = pd.get_dummies(
        df,
        columns=["area_type", "location"],
        drop_first=True,
        dtype=int
    )

    # Features & Target
    X = df.drop(columns=["price"])
    y = df["price"]

    return X, y