"""
Evaluation utilities for Project 1.
Metrics: F1, Accuracy (minimum); PR-AUC (average_precision) for threshold-based.
"""

from sklearn.metrics import accuracy_score, f1_score, average_precision_score, confusion_matrix


def score_binary(y_true, y_pred, y_proba=None):
    """Return dict with accuracy, f1, average_precision (if y_proba)."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if y_proba is not None:
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    return out
