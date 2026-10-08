"""
Centralized Configuration and Path Management for L2D-SDDA
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Primary Dataset Path
if (PROCESSED_DATA_DIR / "churnData_Fixed.csv").exists():
    DATASET_PATH = PROCESSED_DATA_DIR / "churnData_Fixed.csv"
else:
    DATASET_PATH = PROJECT_ROOT / "churnData_Fixed.csv"

# Model Serialization Paths
CALIBRATED_MODEL_PATH = MODELS_DIR / "l2d_calibrated_model.pkl"
PIPELINE_BUNDLE_PATH = MODELS_DIR / "l2d_full_pipeline_bundle.pkl"

# Model Hyperparameters & Features
LEAK_COLS = ["signup_date", "year_month", "window_id"]
TARGET_COL = "Churn"

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "eval_metric": "logloss",
    "random_state": 42
}

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5

# Policy Constants
DEFAULT_DEFERRAL_BUDGET = 0.15
DEFAULT_LAMBDA = 0.15
DEFAULT_THRESHOLD = 0.1298
