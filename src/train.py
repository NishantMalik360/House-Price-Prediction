"""
=========================================================
AI House Price Prediction System

File: train.py

Purpose:
Train and compare multiple machine learning models.
=========================================================
"""

from __future__ import annotations

from math import sqrt
from typing import Dict

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from src.config import RANDOM_STATE, TEST_SIZE


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Split data into train and test sets.
    """

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def get_models() -> Dict[str, object]:
    """
    Return all ML models.
    """

    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_estimators=200,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
    }


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model.
    """

    predictions = model.predict(X_test)

    rmse = sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    return {
        "R2": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": rmse,
    }


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Train all models and compare performance.
    """

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    results = []

    for name, model in get_models().items():

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        results.append(
            {
                "Model": name,
                **metrics,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            by="R2",
            ascending=False,
        )
        .reset_index(drop=True)
    )