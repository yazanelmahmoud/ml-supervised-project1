"""
Evaluation utilities for Wine Quality project (Multiclass Classification).
Metrics: Macro-F1, Accuracy (minimum); confusion matrix and per-class discussion.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def score_binary(y_true, y_pred, y_proba=None):
    """Return dict with accuracy, f1, average_precision (if y_proba). For binary classification."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if y_proba is not None:
        from sklearn.metrics import average_precision_score
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    return out


def score_multiclass(y_true, y_pred, y_proba=None, average='macro'):
    """
    Return dict with accuracy, macro-F1 (default), and per-class metrics.
    For Wine Quality: Macro-F1 and Accuracy are required.
    Includes per-class accuracy and F1 scores.
    """
    from sklearn.metrics import precision_score, recall_score
    
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro')),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted')),
        "f1_micro": float(f1_score(y_true, y_pred, average='micro')),
    }
    
    # Get unique classes
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
    out["f1_per_class"] = {f"class_{label}": float(score) for label, score in zip(unique_classes, f1_per_class)}
    
    # Per-class accuracy (per-class accuracy = recall for that class)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
    out["accuracy_per_class"] = {f"class_{label}": float(score) for label, score in zip(unique_classes, recall_per_class)}
    
    # Per-class precision
    precision_per_class = precision_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
    out["precision_per_class"] = {f"class_{label}": float(score) for label, score in zip(unique_classes, precision_per_class)}
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    out["confusion_matrix"] = cm.tolist()
    
    # Classification report (includes precision, recall, f1 per class)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=unique_classes)
    out["classification_report"] = report
    
    return out
