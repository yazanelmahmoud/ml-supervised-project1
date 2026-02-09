"""
Support Vector Machines for Wine Quality (Multiclass Classification).
Evaluate at least 2 kernels (linear vs RBF); tune C and gamma (where relevant).
Scale features (required — preprocessing uses StandardScaler).
Required: learning curves, model-complexity curves (C, gamma), runtime.
Uses Macro-F1 for multiclass evaluation and class_weight='balanced' for imbalance.
"""

import os
import platform
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix

from config import RANDOM_SEED, OUTPUT_DIR, TASK_TYPE
from evaluation import score_multiclass

SVM_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SVM_results.txt")

# Hyperparameter ranges
C_VALUES = [0.1, 1, 10, 100, 1000]
C_REF = 1  # reference C for table
GAMMA_VALUES = [0.01, "scale", 0.1]  # 3 values around scale


def _get_hardware_note():
    try:
        cpu = platform.processor() or platform.machine() or "unknown"
        return f"{platform.system()} {platform.release()}, CPU: {cpu}"
    except Exception:
        return "unknown"


def run_svm_model_complexity(X_train, y_train, X_test, y_test, cv=5):
    """
    SVM model-complexity: linear and RBF kernels, CV Macro-F1 vs C (and gamma for RBF).
    Two panels: Linear (C vs Macro-F1) | RBF (C vs Macro-F1 for different gammas).
    Best params from linear + RBF. Save plots to outputs/.
    """
    results = {"linear": {}, "rbf": {}, "best_config": None, "best_cv_f1_macro": -1}
    np.random.seed(RANDOM_SEED)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    # Linear: CV Macro-F1 vs C
    linear_cv_f1_macro = []
    for c in C_VALUES:
        clf = SVC(kernel="linear", C=c, random_state=RANDOM_SEED, max_iter=3000, class_weight='balanced', probability=True)
        scores = cross_val_score(clf, X_train, y_train, cv=cv_splitter, scoring="f1_macro", n_jobs=-1)
        linear_cv_f1_macro.append(float(scores.mean()))
    results["linear"] = {"C_values": list(C_VALUES), "cv_f1_macro": linear_cv_f1_macro}
    best_linear_idx = int(np.argmax(linear_cv_f1_macro))
    best_linear_f1_macro = linear_cv_f1_macro[best_linear_idx]
    if best_linear_f1_macro > results["best_cv_f1_macro"]:
        results["best_cv_f1_macro"] = best_linear_f1_macro
        results["best_config"] = {"kernel": "linear", "C": C_VALUES[best_linear_idx], "gamma": None}

    # RBF: CV Macro-F1 vs C for different gammas
    rbf_curves = {}
    for gamma in GAMMA_VALUES:
        key = f"gamma={gamma}"
        cv_f1_macro_list = []
        for c in C_VALUES:
            clf = SVC(kernel="rbf", C=c, gamma=gamma, random_state=RANDOM_SEED, max_iter=3000, class_weight='balanced', probability=True)
            scores = cross_val_score(clf, X_train, y_train, cv=cv_splitter, scoring="f1_macro", n_jobs=-1)
            cv_f1_macro_list.append(float(scores.mean()))
        rbf_curves[key] = cv_f1_macro_list
        best_idx = int(np.argmax(cv_f1_macro_list))
        if cv_f1_macro_list[best_idx] > results["best_cv_f1_macro"]:
            results["best_cv_f1_macro"] = cv_f1_macro_list[best_idx]
            results["best_config"] = {"kernel": "rbf", "C": C_VALUES[best_idx], "gamma": gamma}
    results["rbf"] = {"C_values": list(C_VALUES), "curves": rbf_curves}

    _plot_model_complexity(results)
    _print_model_complexity(results)
    _save_model_complexity(results)
    return results


def _plot_model_complexity(results):
    """Two panels: Linear (C vs Macro-F1) | RBF (C vs Macro-F1 for different gammas)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Panel 1: Linear
    ax = axes[0]
    ax.plot(results["linear"]["C_values"], results["linear"]["cv_f1_macro"], "o-", label="linear", linewidth=2, markersize=8)
    ax.set_xlabel("C", fontsize=12)
    ax.set_ylabel("Cross-Val Macro-F1", fontsize=12)
    ax.set_title("SVM — Linear kernel (CV Macro-F1 vs C)", fontsize=14, fontweight='bold')
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Panel 2: RBF
    ax = axes[1]
    for key, vals in results["rbf"]["curves"].items():
        ax.plot(results["rbf"]["C_values"], vals, "o-", label=key, linewidth=2, markersize=6)
    ax.set_xlabel("C", fontsize=12)
    ax.set_ylabel("Cross-Val Macro-F1", fontsize=12)
    ax.set_title("SVM — RBF kernel (CV Macro-F1 vs C)", fontsize=14, fontweight='bold')
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "svm_model_complexity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _print_model_complexity(results):
    c_ref_idx = C_VALUES.index(C_REF) if C_REF in C_VALUES else 0
    print("--- SVM Step 1: Model-complexity (linear vs RBF, CV Macro-F1 vs C) ---")
    print("Linear — CV Macro-F1 at C=", C_VALUES[c_ref_idx], ":", results["linear"]["cv_f1_macro"][c_ref_idx])
    for key in results["rbf"]["curves"]:
        print("RBF", key, "— CV Macro-F1:", [round(x, 4) for x in results["rbf"]["curves"][key]])
    print("Best config:", results["best_config"], "| best_cv_f1_macro:", round(results["best_cv_f1_macro"], 4))
    print("Results saved to:", SVM_RESULTS_PATH)


def _save_model_complexity(results):
    lines = [
        "=" * 60,
        "SUPPORT VECTOR MACHINES — RESULTS (Wine Quality)",
        "=" * 60,
        "",
        "--- Step 1: Model-complexity (linear vs RBF, CV Macro-F1 vs C) ---",
        "C_values: " + str(C_VALUES),
        "Linear cv_f1_macro: " + str([round(x, 4) for x in results["linear"]["cv_f1_macro"]]),
        "",
    ]
    for key in results["rbf"]["curves"]:
        lines.append(f"RBF {key} cv_f1_macro: " + str([round(x, 4) for x in results["rbf"]["curves"][key]]))
    lines.extend([
        "",
        f"Best config: {results['best_config']}",
        f"best_cv_f1_macro: {results['best_cv_f1_macro']:.4f}",
        "",
        "=" * 60,
    ])
    with open(SVM_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_svm_learning_curves(X_train, y_train, X_test, y_test, best_config=None, cv=5):
    """
    Step 2: Learning curves — train/val Macro-F1 vs training size.
    Use (a) linear and (b) RBF with chosen hyperparams (or baseline vs best). Two panels.
    """
    if best_config is None:
        best_config = {"kernel": "rbf", "C": 100, "gamma": "scale"}
    
    np.random.seed(RANDOM_SEED)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    train_sizes = np.linspace(0.1, 1.0, 10)

    # Linear baseline
    linear_clf = SVC(kernel="linear", C=1, random_state=RANDOM_SEED, max_iter=3000, class_weight='balanced', probability=True)
    lc_sizes_lin, lc_train_lin, lc_val_lin = learning_curve(
        linear_clf, X_train, y_train, train_sizes=train_sizes, cv=cv_splitter, scoring="f1_macro", random_state=RANDOM_SEED
    )
    
    # Best config (RBF or linear)
    if best_config["kernel"] == "linear":
        best_clf = SVC(kernel="linear", C=best_config["C"], random_state=RANDOM_SEED, max_iter=3000, class_weight='balanced', probability=True)
    else:
        best_clf = SVC(kernel="rbf", C=best_config["C"], gamma=best_config["gamma"], random_state=RANDOM_SEED, max_iter=3000, class_weight='balanced', probability=True)
    
    _, lc_train_best, lc_val_best = learning_curve(
        best_clf, X_train, y_train, train_sizes=train_sizes, cv=cv_splitter, scoring="f1_macro", random_state=RANDOM_SEED
    )

    results = {
        "train_sizes": np.asarray(lc_sizes_lin).flatten().tolist(),
        "linear": {
            "train_f1_macro_mean": lc_train_lin.mean(axis=1).tolist(),
            "train_f1_macro_std": lc_train_lin.std(axis=1).tolist(),
            "val_f1_macro_mean": lc_val_lin.mean(axis=1).tolist(),
            "val_f1_macro_std": lc_val_lin.std(axis=1).tolist(),
        },
        "best": {
            "train_f1_macro_mean": lc_train_best.mean(axis=1).tolist(),
            "train_f1_macro_std": lc_train_best.std(axis=1).tolist(),
            "val_f1_macro_mean": lc_val_best.mean(axis=1).tolist(),
            "val_f1_macro_std": lc_val_best.std(axis=1).tolist(),
        },
        "best_config": best_config,
    }
    _plot_learning_curves(results)
    _print_learning_curves(results)
    _append_learning_curves(results)
    return results


def _plot_learning_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sizes = results["train_sizes"]
    for ax, label, data in zip(
        axes,
        ["Linear (baseline)", f"Best ({results['best_config']['kernel']})"],
        [results["linear"], results["best"]],
    ):
        ax.plot(sizes, data["train_f1_macro_mean"], "o-", label="Train Macro-F1")
        ax.fill_between(sizes, np.array(data["train_f1_macro_mean"]) - np.array(data["train_f1_macro_std"]), 
                       np.array(data["train_f1_macro_mean"]) + np.array(data["train_f1_macro_std"]), alpha=0.2)
        ax.plot(sizes, data["val_f1_macro_mean"], "s-", label="Val Macro-F1")
        ax.fill_between(sizes, np.array(data["val_f1_macro_mean"]) - np.array(data["val_f1_macro_std"]), 
                       np.array(data["val_f1_macro_mean"]) + np.array(data["val_f1_macro_std"]), alpha=0.2)
        ax.set_xlabel("Training size")
        ax.set_ylabel("Macro-F1")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "svm_learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()


def _print_learning_curves(results):
    print("--- SVM Step 2: Learning curves ---")
    print("train_sizes:", results["train_sizes"])
    print("Linear: val_f1_macro_mean:", [round(x, 4) for x in results["linear"]["val_f1_macro_mean"]])
    print("Best:", results["best_config"], "val_f1_macro_mean:", [round(x, 4) for x in results["best"]["val_f1_macro_mean"]])
    print("Appended to:", SVM_RESULTS_PATH)


def _append_learning_curves(results):
    lines = [
        "",
        "--- Learning curves ---",
        "train_sizes: " + str(results["train_sizes"]),
        "linear (baseline):",
        "  train_f1_macro_mean: " + str([round(x, 4) for x in results["linear"]["train_f1_macro_mean"]]),
        "  val_f1_macro_mean: " + str([round(x, 4) for x in results["linear"]["val_f1_macro_mean"]]),
        f"best ({results['best_config']['kernel']}): " + str(results["best_config"]),
        "  train_f1_macro_mean: " + str([round(x, 4) for x in results["best"]["train_f1_macro_mean"]]),
        "  val_f1_macro_mean: " + str([round(x, 4) for x in results["best"]["val_f1_macro_mean"]]),
        "",
    ]
    with open(SVM_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_svm_test_eval(X_train, y_train, X_test, y_test, best_config=None):
    """
    Step 3: Refit best SVM on full train; evaluate once on test.
    Report metrics, runtime, confusion matrix. Append to SVM_results.txt.
    """
    if best_config is None:
        best_config = {"kernel": "rbf", "C": 100, "gamma": "scale"}
    
    np.random.seed(RANDOM_SEED)
    if best_config["kernel"] == "linear":
        clf = SVC(kernel="linear", C=best_config["C"], random_state=RANDOM_SEED, max_iter=3000, class_weight='balanced', probability=True)
    else:
        clf = SVC(kernel="rbf", C=best_config["C"], gamma=best_config["gamma"], random_state=RANDOM_SEED, max_iter=3000, class_weight='balanced', probability=True)
    
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    predict_time = time.perf_counter() - t0
    metrics = score_multiclass(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "best_config": best_config,
        "test_metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "runtime": {"fit_sec": fit_time, "predict_sec": predict_time},
    }
    _print_test_eval(results)
    _append_test_eval(results)
    _plot_confusion_matrix(cm)
    return results


def _plot_confusion_matrix(cm):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Quality')
    ax.set_ylabel('True Quality')
    ax.set_title('SVM Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "svm_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()


def _print_test_eval(results):
    print("--- SVM Step 3: Test evaluation ---")
    print("Best config:", results["best_config"])
    print("Test metrics:", results["test_metrics"])
    print("Runtime - fit (s):", round(results["runtime"]["fit_sec"], 4), "| predict (s):", round(results["runtime"]["predict_sec"], 4))
    print("Confusion matrix:", results["confusion_matrix"])
    print("Appended to:", SVM_RESULTS_PATH)


def _append_test_eval(results):
    m = results["test_metrics"]
    cm = results["confusion_matrix"]
    rt = results["runtime"]
    unique_classes = sorted([int(k.split('_')[1]) for k in m.get("f1_per_class", {}).keys()])
    f1_per_class = m.get("f1_per_class", {})
    accuracy_per_class = m.get("accuracy_per_class", {})
    
    lines = [
        "--- Test evaluation ---",
        "best_config (from Step 1): " + str(results["best_config"]),
        f"Accuracy: {m.get('accuracy', 0):.4f}",
        f"Macro-F1: {m.get('f1_macro', 0):.4f}",
        f"Weighted-F1: {m.get('f1_weighted', 0):.4f}",
        "",
        "--- Per-class performance (F1 scores and Accuracy) ---",
    ]
    
    for class_label in unique_classes:
        class_key = f"class_{class_label}"
        f1_score_val = f1_per_class.get(class_key, 0.0)
        accuracy_val = accuracy_per_class.get(class_key, 0.0)
        lines.append(f"  Quality {class_label}: F1 = {f1_score_val:.4f}, Accuracy = {accuracy_val:.4f}")
    
    lines.extend([
        "",
        f"Fit (sec): {rt['fit_sec']:.4f}",
        f"Predict (sec): {rt['predict_sec']:.4f}",
        "Hardware: " + _get_hardware_note(),
        "Confusion matrix: " + str(cm),
        "",
        "=" * 60,
    ])
    with open(SVM_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
