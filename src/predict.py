"""
=========================================================
AI House Price Prediction System

File: predict.py

Purpose:
Load trained model and make predictions.
=========================================================
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


MODEL_DIR = Path("models/trained")

MODEL_PATH = MODEL_DIR / "house_price_model.pkl"
FEATURE_PATH = MODEL_DIR / "feature_columns.pkl"


def load_model():
    """
    Load trained ML model.
    """
    return joblib.load(MODEL_PATH)


def load_feature_columns():
    """
    Load saved feature columns.
    """
    return joblib.load(FEATURE_PATH)


def prepare_input(input_data: dict) -> pd.DataFrame:
    """
    Convert user input into model input.
    """

    columns = load_feature_columns()

    df = pd.DataFrame(
        [[0] * len(columns)],
        columns=columns,
    )

    # Numerical Features
    df.at[0, "total_sqft"] = input_data["total_sqft"]
    df.at[0, "bath"] = input_data["bath"]
    df.at[0, "balcony"] = input_data["balcony"]
    df.at[0, "bhk"] = input_data["bhk"]

    # Area Type
    area_column = f"area_type_{input_data['area_type']}"

    if area_column in df.columns:
        df.at[0, area_column] = 1

    # Location
    location_column = f"location_{input_data['location']}"

    if location_column in df.columns:
        df.at[0, location_column] = 1

    return df


def predict_house(input_data: dict):
    """
    Predict house price.
    """

    model = load_model()

    X = prepare_input(input_data)

    prediction = model.predict(X)

    return float(prediction[0])


def get_locations():
    """
    Return all available locations.
    """

    columns = load_feature_columns()

    locations = []

    for column in columns:

        if column.startswith("location_"):

            locations.append(
                column.replace("location_", "")
            )

    locations.sort()

    return locations