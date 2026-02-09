"""
Project 1 — Wine Quality: configuration.
Reproducibility: random seeds, paths, metrics, and constants.
"""

import os

# ----- Reproducibility -----
RANDOM_SEED = 42

# ----- Paths -----
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "wine.csv")

# ----- Task -----
TARGET_COLUMN = "quality"  # Discrete rating/class
TASK_TYPE = "multiclass_classification"

# ----- Evaluation metrics (required: Macro-F1, Accuracy; confusion matrix and per-class discussion) -----
METRICS_NAMES = ["accuracy", "f1_macro"]  # Macro-F1 and Accuracy for multiclass

# ----- Train/test split -----
TEST_SIZE = 0.2  # Single held-out test split; tuning via CV on training only

# ----- Optional: where to save figures/tables -----
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
