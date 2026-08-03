"""
=========================================================
AI House Price Prediction System

File: feature_engineering.py

Purpose:
Prepare cleaned data for Machine Learning.
=========================================================
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not required for model training.
    """

    columns_to_drop = [
        "availability",
        "size",
        "price_per_sqft",   # Prevent data leakage
    ]

    return df.drop(columns=columns_to_drop, errors="ignore")


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot Encode categorical features.
    """

    categorical_columns = [
        "area_type",
        "location",
    ]

    return pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int,
    )


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features (X) and target (y).
    """

    X = df.drop(columns=["price"])

    y = df["price"]

    return X, y


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete feature engineering pipeline.
    """

    df = drop_unused_columns(df)

    df = encode_features(df)

    X, y = split_features_target(df)

    return X, y