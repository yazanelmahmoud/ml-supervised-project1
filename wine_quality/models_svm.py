"""
Support Vector Machines for Wine Quality.
TODO: Evaluate ≥2 kernels (linear vs RBF), tune C and γ.
Required: learning curves, model-complexity curves, runtime.
"""

import os
import platform
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix

from config import RANDOM_SEED, TASK_TYPE
from evaluation import score_multiclass

SVM_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SVM_results.txt")

# TODO: Define hyperparameter ranges
C_VALUES = [0.1, 1, 10, 100, 1000]
GAMMA_VALUES = [0.001, 0.01, 0.1, 1, 10]
KERNELS = ["linear", "rbf"]


def run_svm_model_complexity(X_train, y_train, X_test, y_test, cv=5):
    """
    TODO: SVM model-complexity: kernel × C (and γ for RBF).
    Plot CV metric vs C (linear) and vs C/gamma (RBF).
    Identify best kernel and hyperparams. Save to SVM_results.txt.
    """
    # TODO: Implement
    pass


def run_svm_learning_curves(X_train, y_train, X_test, y_test, best_config=None, cv=5):
    """
    TODO: Learning curves — train/val metric vs training size.
    Use (a) linear and (b) RBF with chosen hyperparams.
    Plot two panels. Append to SVM_results.txt.
    """
    # TODO: Implement
    pass


def run_svm_test_eval(X_train, y_train, X_test, y_test, best_config=None):
    """
    TODO: Refit best SVM on full train; evaluate once on test.
    Report metrics, runtime, confusion matrix. Append to SVM_results.txt.
    """
    # TODO: Implement
    pass
