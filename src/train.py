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

from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

import joblib
from pathlib import Path

from src.config import RANDOM_STATE, TEST_SIZE

import joblib
from pathlib import Path

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
            n_estimators=500,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
        ),
        "XGBoost": XGBRegressor(
            random_state=RANDOM_STATE,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
        )
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



def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Perform 5-Fold Cross Validation.
    """

    results = []

    for name, model in get_models().items():

        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )

        results.append(
            {
                "Model": name,
                "Mean R2": scores.mean(),
                "Std": scores.std(),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            by="Mean R2",
            ascending=False,
        )
        .reset_index(drop=True)
    )

def save_model(model, model_name: str) -> None:
    """
    Save trained model to disk.
    """

    model_dir = Path("models/trained")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}.pkl"

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")


def save_feature_columns(X: pd.DataFrame) -> None:
    """
    Save feature column names for prediction.
    """

    model_dir = Path("models/trained")
    model_dir.mkdir(parents=True, exist_ok=True)

    columns_path = model_dir / "feature_columns.pkl"

    joblib.dump(list(X.columns), columns_path)

    print(f"Feature columns saved to: {columns_path}")


def train_best_model(X, y):
    """
    Train the best model and save required artifacts.
    """

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
    )

    model.fit(X_train, y_train)

    save_model(model, "house_price_model")

    save_feature_columns(X)

    return model