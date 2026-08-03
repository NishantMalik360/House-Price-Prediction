"""
=========================================================
AI House Price Prediction System

File: preprocessing.py

Purpose:
Complete preprocessing pipeline for the Bengaluru House
Price Prediction project.
=========================================================
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

# --------------------------------------------------------
# Logging
# --------------------------------------------------------

logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Basic Cleaning
# --------------------------------------------------------


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    logger.info("Removed %s duplicate rows.", removed)

    return df


def drop_society(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop society column if present.
    """

    if "society" in df.columns:
        df = df.drop(columns=["society"])
        logger.info("Dropped society column.")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values.

    Strategy
    --------
    location -> drop rows
    size     -> drop rows
    bath     -> drop rows
    balcony  -> fill median
    """

    before = len(df)

    df = df.dropna(
        subset=[
            "location",
            "size",
            "bath",
        ]
    )

    removed = before - len(df)

    logger.info(
        "Removed %s rows containing missing values.",
        removed,
    )

    median_balcony = df["balcony"].median()

    df["balcony"] = df["balcony"].fillna(
        median_balcony
    )

    logger.info(
        "Filled balcony missing values using median (%s).",
        median_balcony,
    )

    return df


# --------------------------------------------------------
# Size Cleaning
# --------------------------------------------------------


def clean_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert size column into numerical BHK.
    """

    df["bhk"] = (
        df["size"]
        .str.split()
        .str[0]
        .astype(int)
    )

    logger.info("Created BHK feature.")

    return df

# --------------------------------------------------------
# Total Sqft Cleaning
# --------------------------------------------------------

# Conversion factors to Square Feet
UNIT_CONVERSION = {
    "Sq. Meter": 10.7639,
    "Sq. Yards": 9.0,
    "Sq. Yard": 9.0,
    "Perch": 272.25,
    "Guntha": 1089.0,
    "Grounds": 2400.0,
    "Acres": 43560.0,
    "Cents": 435.6,
}


def convert_total_sqft(value: str) -> Optional[float]:
    """
    Convert total_sqft values into square feet.

    Supported formats
    -----------------
    1200
    2100-2850
    34.46 Sq. Meter
    100 Sq. Yard
    2 Acres

    Returns
    -------
    float | None
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    # -----------------------------
    # Range
    # -----------------------------

    if "-" in value:

        try:

            start, end = value.split("-")

            return (float(start) + float(end)) / 2

        except ValueError:

            return None

    # -----------------------------
    # Plain Number
    # -----------------------------

    try:

        return float(value)

    except ValueError:

        pass

    # -----------------------------
    # Unit Conversion
    # -----------------------------

    for unit, factor in UNIT_CONVERSION.items():

        if unit in value:

            number = value.replace(unit, "").strip()

            try:

                return float(number) * factor

            except ValueError:

                return None

    return None


def clean_total_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean total_sqft column.
    """

    df["total_sqft"] = df["total_sqft"].apply(
        convert_total_sqft
    )

    before = len(df)

    df = df.dropna(
        subset=["total_sqft"]
    )

    removed = before - len(df)

    logger.info(
        "Removed %s invalid total_sqft rows.",
        removed,
    )

    return df


# --------------------------------------------------------
# Location Cleaning
# --------------------------------------------------------

def clean_location(df: pd.DataFrame, min_frequency: int = 10) -> pd.DataFrame:
    """
    Clean location names and group rare locations into 'Other'.

    Parameters
    ----------
    df : pd.DataFrame
    min_frequency : int
        Minimum number of occurrences required to keep a location.
    """

    df["location"] = df["location"].str.strip()

    location_counts = df["location"].value_counts()

    rare_locations = location_counts[
        location_counts <= min_frequency
    ].index

    df["location"] = df["location"].replace(
        rare_locations,
        "Other",
    )

    logger.info(
        "Location cleaning completed. Remaining unique locations: %s",
        df["location"].nunique(),
    )

    return df


# --------------------------------------------------------
# Price Per Sqft
# --------------------------------------------------------

def create_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price_per_sqft feature.
    """

    df["price_per_sqft"] = (
        df["price"] * 100000
    ) / df["total_sqft"]

    logger.info("Created price_per_sqft feature.")

    return df


# --------------------------------------------------------
# Outlier Removal
# --------------------------------------------------------

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove basic outliers.
    """

    before = len(df)

    # Minimum 300 sqft per BHK
    df = df[
        (df["total_sqft"] / df["bhk"]) >= 300
    ]

    removed = before - len(df)

    logger.info(
        "Removed %s BHK outliers.",
        removed,
    )

    return df


# --------------------------------------------------------
# Complete Pipeline
# --------------------------------------------------------

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute complete preprocessing pipeline.
    """

    logger.info("Starting preprocessing pipeline...")

    df = remove_duplicates(df)

    df = drop_society(df)

    df = handle_missing_values(df)

    df = clean_size(df)

    df = clean_total_sqft(df)

    df = clean_location(df)

    df = create_price_per_sqft(df)

    df = remove_outliers(df)

    logger.info("Preprocessing completed.")

    return df