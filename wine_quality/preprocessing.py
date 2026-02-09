"""
Preprocessing for Wine Quality dataset.
Based on EDA: drop 'class' column (perfect correlation with target, data leakage),
standardize all numeric features.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import TARGET_COLUMN, TEST_SIZE, RANDOM_SEED

# Drop 'class' column: perfect correlation (1.0) with target 'quality' = data leakage
DROP_COLUMNS = ["class"]


def prepare_X_y(df, scaler=None, fit=True):
    """
    Prepare feature matrix X and target y from DataFrame.
    Steps: (1) Drop 'class' column (data leakage), (2) Extract target, 
    (3) Standardize all numeric features.
    
    Args:
        df: DataFrame with target column
        scaler: Pre-fitted StandardScaler (for test set)
        fit: If True, fit scaler; if False, use provided scaler
    
    Returns:
        X (float32), y, and scaler for reuse
    """
    df = df.copy()
    
    # Drop leakage columns (class has perfect correlation with quality)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    
    y = df[TARGET_COLUMN].copy()
    X_df = df.drop(columns=[TARGET_COLUMN])
    
    # All features are numeric, standardize them
    X_num = X_df.select_dtypes(include=[np.number]).astype(np.float32)
    
    if fit:
        scaler = StandardScaler()
        scaler.fit(X_num)
    else:
        if scaler is None:
            raise ValueError("For test set pass scaler from train.")
    
    X_scaled = scaler.transform(X_num)
    X = X_scaled.astype(np.float32)
    
    return X, y, scaler


def get_preprocessed_train_test(X_train_raw, y_train_raw, X_test_raw, y_test_raw):
    """
    Apply preprocessing to train and test. Fit on train only; transform both.
    Returns (X_train, y_train, X_test, y_test).
    """
    train_df = X_train_raw.copy()
    train_df[TARGET_COLUMN] = y_train_raw
    test_df = X_test_raw.copy()
    test_df[TARGET_COLUMN] = y_test_raw
    
    X_train, y_train, scaler = prepare_X_y(train_df, fit=True)
    X_test, y_test, _ = prepare_X_y(test_df, scaler=scaler, fit=False)
    return X_train, y_train, X_test, y_test


def get_dataset(path=None, test_size=None, random_state=None):
    """
    Single entry point: load data, stratified train/test split, preprocess.
    Use this everywhere (notebook, model scripts) for consistent data.

    Returns:
        X_train, y_train, X_test, y_test (ready for fit/predict).
    """
    from data_loading import load_wine, get_target_and_features
    
    test_size = test_size if test_size is not None else TEST_SIZE
    random_state = random_state if random_state is not None else RANDOM_SEED
    
    df = load_wine(path=path)
    X, y = get_target_and_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return get_preprocessed_train_test(X_train, y_train, X_test, y_test)
