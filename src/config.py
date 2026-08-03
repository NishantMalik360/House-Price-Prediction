"""
=========================================================
AI House Price Prediction System

File: config.py

Purpose:
Store all project constants and file paths.
=========================================================
"""

from pathlib import Path

# --------------------------------------------------------
# Project Root
# --------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------
# Data Paths
# --------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "Bengaluru_House_Data.csv"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "bengaluru_house_cleaned.csv"

# --------------------------------------------------------
# Models
# --------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"

TRAINED_MODEL_DIR = MODELS_DIR / "trained"
METRICS_DIR = MODELS_DIR / "metrics"
PLOTS_DIR = MODELS_DIR / "plots"

MODEL_FILE = TRAINED_MODEL_DIR / "house_price_model.joblib"

# --------------------------------------------------------
# Reports
# --------------------------------------------------------

REPORTS_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
LOGS_DIR = REPORTS_DIR / "logs"

# --------------------------------------------------------
# Random State
# --------------------------------------------------------

RANDOM_STATE = 42

# --------------------------------------------------------
# Train/Test Split
# --------------------------------------------------------

TEST_SIZE = 0.20

# --------------------------------------------------------
# Target Column
# --------------------------------------------------------

TARGET_COLUMN = "price"