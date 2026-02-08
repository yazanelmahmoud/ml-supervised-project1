"""
Project 1 — Adult Income (Census): configuration.
Reproducibility: random seeds, paths, metrics, and constants.
"""

import os

# ----- Reproducibility -----
RANDOM_SEED = 42

# ----- Paths -----
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "adult.csv")

# ----- Task -----
TARGET_COLUMN = "class"  # Binary: <=50K vs >50K
TASK_TYPE = "binary_classification"

# ----- Evaluation metrics (required: F1, Accuracy; PR-AUC for threshold-based) -----
METRICS_NAMES = ["accuracy", "f1", "average_precision"]  # average_precision = PR-AUC

# ----- Train/test split -----
TEST_SIZE = 0.2  # Single held-out test split; tuning via CV on training only

# ----- Optional: where to save figures/tables -----
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
