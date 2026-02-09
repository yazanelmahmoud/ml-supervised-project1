"""
Preprocessing for Adult Income dataset.
Applies EDA rules: drop education & fnlwgt, one-hot + target encoding, StandardScaler.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import TARGET_COLUMN, TEST_SIZE, RANDOM_SEED

# Columns to drop (EDA: redundancy + near-zero correlation)
DROP_COLUMNS = ["education", "fnlwgt"]

# Numeric and categorical feature names after drops
NUMERIC_FEATURES = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_ONEHOT = ["workclass", "marital-status", "occupation", "relationship", "race", "sex"]
CATEGORICAL_TARGET_ENC = ["native-country"]


def encode_target(y):
    """Encode target: <=50K -> 0, >50K -> 1. Returns int32."""
    return (y.str.strip().str.lower() == ">50k").astype(np.int32)


def fit_target_encoding(X_train, y_train, col):
    """Fit: map each category to mean target on training set. Return dict + global mean."""
    enc = pd.Series(y_train).groupby(X_train[col].astype(str)).mean().to_dict()
    global_mean = float(y_train.mean())
    return enc, global_mean


def transform_target_encoding(X, encoding, global_mean, col):
    """Transform: replace category with encoded value; unseen -> global mean."""
    return X[col].astype(str).map(encoding).fillna(global_mean).values.reshape(-1, 1)


def prepare_X_y(df, target_encodings=None, scaler=None, ohe=None, fit=True):
    """
    Prepare feature matrix X and target y from DataFrame.
    - Drops DROP_COLUMNS, encodes target to 0/1.
    - If fit=True: fits OneHotEncoder, StandardScaler, target encoding (train).
    - If fit=False: uses provided ohe, scaler, target_encodings (test).
    Returns X (float32), y (int32), and (scaler, ohe, target_encodings) for reuse.
    """
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    y = encode_target(df[TARGET_COLUMN])
    X_df = df.drop(columns=[TARGET_COLUMN])

    X_num = X_df[NUMERIC_FEATURES].astype(np.float32).copy()
    # Log1p transform for right-skewed, zero-inflated capital-gain/capital-loss
    X_num["capital-gain"] = np.log1p(X_num["capital-gain"])
    X_num["capital-loss"] = np.log1p(X_num["capital-loss"])
    X_ohe = X_df[CATEGORICAL_ONEHOT].astype(str)

    if fit:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        ohe.fit(X_ohe)
        scaler = StandardScaler()
        scaler.fit(X_num)
        target_encodings = {}
        for col in CATEGORICAL_TARGET_ENC:
            target_encodings[col] = fit_target_encoding(X_df, y, col)
    else:
        if ohe is None or scaler is None or target_encodings is None:
            raise ValueError("For test set pass ohe, scaler, and target_encodings from train.")

    X_ohe_arr = ohe.transform(X_ohe)
    X_num_scaled = scaler.transform(X_num)
    te_list = [
        transform_target_encoding(
            X_df, target_encodings[c][0], target_encodings[c][1], c
        )
        for c in CATEGORICAL_TARGET_ENC
    ]
    X_te = np.hstack(te_list)
    X = np.hstack([X_num_scaled, X_ohe_arr, X_te]).astype(np.float32)

    return X, y, (scaler, ohe, target_encodings)


def get_preprocessed_train_test(X_train_raw, y_train_raw, X_test_raw, y_test_raw):
    """
    Apply preprocessing to train and test. Fit on train only; transform both.
    Returns (X_train, y_train, X_test, y_test).
    """
    train_df = X_train_raw.copy()
    train_df[TARGET_COLUMN] = y_train_raw
    test_df = X_test_raw.copy()
    test_df[TARGET_COLUMN] = y_test_raw

    X_train, y_train, (scaler, ohe, te) = prepare_X_y(train_df, fit=True)
    X_test, y_test, _ = prepare_X_y(test_df, target_encodings=te, scaler=scaler, ohe=ohe, fit=False)
    return X_train, y_train, X_test, y_test


def get_dataset(path=None, test_size=None, random_state=None):
    """
    Single entry point: load data, stratified train/test split, preprocess.
    Use this everywhere (notebook, model scripts) for consistent data.

    Returns:
        X_train, y_train, X_test, y_test (ready for fit/predict).
    """
    from data_loading import load_adult, get_target_and_features

    test_size = test_size if test_size is not None else TEST_SIZE
    random_state = random_state if random_state is not None else RANDOM_SEED

    df = load_adult(path=path)
    X, y = get_target_and_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return get_preprocessed_train_test(X_train, y_train, X_test, y_test)
