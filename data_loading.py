"""
Load Adult (Census Income) dataset.
Declare target, task type; basic loading and column info.
"""

import pandas as pd

from config import DATA_PATH, TARGET_COLUMN, TASK_TYPE, RANDOM_SEED


def load_adult(path=None, remove_duplicates=True):
    """
    Load adult.csv. Returns full DataFrame.
    
    Args:
        path: Path to CSV file (defaults to config.DATA_PATH)
        remove_duplicates: If True, remove exact duplicate rows (default: True)
    
    Returns:
        DataFrame with duplicates removed if remove_duplicates=True
    """
    path = path or DATA_PATH
    df = pd.read_csv(path, na_values="?")
    
    if remove_duplicates:
        initial_rows = len(df)
        df = df.drop_duplicates(keep='first')
        n_removed = initial_rows - len(df)
        if n_removed > 0:
            print(f"Removed {n_removed} duplicate row(s). Dataset: {initial_rows} -> {len(df)} rows.")
    
    return df


def get_target_and_features(df):
    """Split into features and target. Target: income (<=50K vs >50K)."""
    y = df[TARGET_COLUMN].copy()
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


# For reference: columns in adult.csv
# age, workclass, fnlwgt, education, education-num, marital-status, occupation,
# relationship, race, sex, capital-gain, capital-loss, hours-per-week,
# native-country, class
