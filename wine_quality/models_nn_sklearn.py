"""
Neural Networks (sklearn MLPClassifier) for Wine Quality.
TODO: Train MLPClassifier with SGD only (no momentum, no adaptive optimizers).
Compare capacity scaling (depth vs width) while keeping SGD constant.
Required: epoch-based learning curves, early stopping, regularization.
"""

import os
import platform
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix

from config import RANDOM_SEED, TASK_TYPE
from evaluation import score_multiclass

NN_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN_sklearn_results.txt")

# TODO: Define architecture and hyperparameter ranges
HIDDEN_LAYER_SIZES_OPTIONS = [
    (50,),      # Shallow-wide
    (100,),     # Shallow-wide
    (50, 50),   # Deeper
    (25, 25, 25),  # Deeper-narrow
]
ALPHA_VALUES = [0.0001, 0.001, 0.01, 0.1]  # L2 regularization
LEARNING_RATE_INIT = 0.001
BATCH_SIZE = 32
MAX_ITER = 500


def run_nn_model_complexity(X_train, y_train, X_test, y_test, cv=5):
    """
    TODO: NN model-complexity: depth vs width, regularization.
    Plot validation metric vs architecture/regularization.
    Save to NN_sklearn_results.txt.
    """
    # TODO: Implement
    pass


def run_nn_learning_curves(X_train, y_train, X_test, y_test, best_config=None, cv=5):
    """
    TODO: Learning curves — train/val metric vs training size and epochs.
    Plot epoch-based curves showing early stopping.
    Append to NN_sklearn_results.txt.
    """
    # TODO: Implement
    pass


def run_nn_test_eval(X_train, y_train, X_test, y_test, best_config=None):
    """
    TODO: Refit best NN on full train; evaluate once on test.
    Report metrics, runtime, confusion matrix. Append to NN_sklearn_results.txt.
    """
    # TODO: Implement
    pass
