"""
k-Nearest Neighbors for Wine Quality.
TODO: Compare meaningfully different k (small/medium/large); justify choice.
Scale features (required); justify distance metric and weighting.
Required: learning curves, model-complexity curve (k), runtime.
"""

import os
import platform
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix

from config import RANDOM_SEED, TASK_TYPE
from evaluation import score_multiclass

KNN_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KNN_results.txt")

# TODO: Define k values and options based on dataset size
K_VALUES = [3, 5, 10, 15, 20, 25, 30, 40, 50]
WEIGHTS_OPTIONS = ["uniform", "distance"]
METRIC_OPTIONS = ["euclidean", "manhattan"]

K_REF_TABLE = 20


def run_knn_step2(X_train, y_train, X_test, y_test, cv=5):
    """
    TODO: kNN model-complexity: all (weights × metric) combos, CV F1 vs k.
    Plot two panels (uniform | distance), each with euclidean + manhattan curves.
    Print table at k=K_REF_TABLE and best (k, weights, metric). Save to KNN_results.txt.
    """
    # TODO: Implement
    pass


def run_knn_learning_curves(X_train, y_train, X_test, y_test, best_config=None, cv=5):
    """
    TODO: Learning curves — train/val F1 vs training size.
    (a) Baseline: small k. (b) Tuned: best_config from Step 1.
    Plot two panels. Append to KNN_results.txt.
    """
    # TODO: Implement
    pass


def run_knn_test_eval(X_train, y_train, X_test, y_test, best_config=None):
    """
    TODO: Refit best kNN on full train; evaluate once on test.
    best_config from Step 1. Append to KNN_results.txt.
    """
    # TODO: Implement
    pass
