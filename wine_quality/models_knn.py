"""
k-Nearest Neighbors for Wine Quality (Multiclass Classification).
Compare meaningfully different k (small/medium/large); justify choice.
Scale features (required); justify distance metric and weighting.
Required: learning curves, model-complexity curve (k), runtime.
Best config from Step 1 (model-complexity); no separate grid search.
Uses Macro-F1 for multiclass evaluation.
"""

import os
import platform
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix

from config import RANDOM_SEED, TASK_TYPE
from evaluation import score_multiclass

KNN_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KNN_results.txt")

K_VALUES = [3, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100, 200]
WEIGHTS_OPTIONS = ["uniform", "distance"]
METRIC_OPTIONS = ["euclidean", "manhattan"]  # L2, L1

# Reference k for "CV Macro-F1 at k" table
K_REF_TABLE = 20


def _get_hardware_note():
    """Return a brief hardware description for reproducibility."""
    try:
        cpu = platform.processor() or platform.machine() or "unknown"
        return f"{platform.system()} {platform.release()}, CPU: {cpu}"
    except Exception:
        return "unknown"


def run_knn_step2(X_train, y_train, X_test, y_test, cv=5):
    """
    kNN model-complexity: all 4 (weights × metric) combos, CV Macro-F1 vs k.
    Plot two panels (uniform | distance), each with euclidean + manhattan curves.
    Print table at k=K_REF_TABLE and best (k, weights, metric). Save to KNN_results.txt.
    """
    results = {"weights_metric_curves": {}, "cv_f1_macro_at_k": {}}
    np.random.seed(RANDOM_SEED)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    for w in WEIGHTS_OPTIONS:
        for m in METRIC_OPTIONS:
            key = (w, m)
            cv_f1_macro_list = []
            for k in K_VALUES:
                clf = KNeighborsClassifier(n_neighbors=k, weights=w, metric=m)
                scores = cross_val_score(clf, X_train, y_train, cv=cv_splitter, scoring="f1_macro")
                cv_f1_macro_list.append(float(scores.mean()))
            results["weights_metric_curves"][key] = cv_f1_macro_list

    k_idx = K_VALUES.index(K_REF_TABLE) if K_REF_TABLE in K_VALUES else min(
        range(len(K_VALUES)), key=lambda i: abs(K_VALUES[i] - K_REF_TABLE)
    )
    for key in results["weights_metric_curves"]:
        results["cv_f1_macro_at_k"][key] = results["weights_metric_curves"][key][k_idx]

    best_score = -1
    best_config = None
    for k in K_VALUES:
        for w in WEIGHTS_OPTIONS:
            for m in METRIC_OPTIONS:
                idx = K_VALUES.index(k)
                s = results["weights_metric_curves"][(w, m)][idx]
                if s > best_score:
                    best_score = s
                    best_config = {"k": k, "weights": w, "metric": m}
    results["best_k_weights_metric"] = best_config
    results["best_cv_f1_macro"] = best_score
    results["k_values"] = list(K_VALUES)

    _plot(results)
    _print_results(results)
    _save(results)
    return results


def _plot(results):
    """Two panels: uniform (k vs Macro-F1 for euclidean, manhattan); distance (same)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    k_vals = K_VALUES
    for ax, weights in zip(axes, WEIGHTS_OPTIONS):
        for m in METRIC_OPTIONS:
            ax.plot(k_vals, results["weights_metric_curves"][(weights, m)], "o-", label=f"metric={m}")
        ax.set_xlabel("n_neighbors (k)")
        ax.set_ylabel("Cross-Val Macro-F1")
        ax.set_title(f"kNN — weights={weights}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(KNN_RESULTS_PATH), "outputs", "knn_model_complexity.png"), dpi=150, bbox_inches="tight")
    plt.show()


def _print_results(results):
    """Print table at k=K_REF_TABLE and best config."""
    print("--- kNN: Model-complexity (weights × metric, CV Macro-F1 vs k) ---")
    print("k_values:", results["k_values"])
    print(f"CV Macro-F1 at k={K_REF_TABLE}:")
    for w in WEIGHTS_OPTIONS:
        for m in METRIC_OPTIONS:
            print(f"  weights={w}, metric={m}: {results['cv_f1_macro_at_k'][(w, m)]:.4f}")
    print("Best (k, weights, metric):", results["best_k_weights_metric"], "| best_cv_f1_macro:", round(results["best_cv_f1_macro"], 4))
    print("Results saved to:", KNN_RESULTS_PATH)


def _save(results):
    """Write full content to KNN_results.txt."""
    k_ref = K_REF_TABLE
    lines = [
        "=" * 60,
        "k-NEAREST NEIGHBORS — RESULTS (Wine Quality)",
        "=" * 60,
        "",
        "--- Model-complexity: weights × metric (CV Macro-F1 vs k) ---",
        "k_values: " + str(results["k_values"]),
        "",
    ]
    for w in WEIGHTS_OPTIONS:
        for m in METRIC_OPTIONS:
            lines.append(f"  weights={w}, metric={m}: " + str([round(x, 4) for x in results["weights_metric_curves"][(w, m)]]))
    lines.extend([
        "",
        f"CV Macro-F1 at k={k_ref}:",
    ])
    for w in WEIGHTS_OPTIONS:
        for m in METRIC_OPTIONS:
            lines.append(f"  weights={w}, metric={m}: {results['cv_f1_macro_at_k'][(w, m)]:.4f}")
    lines.extend([
        f"Best (k, weights, metric): {results['best_k_weights_metric']}",
        f"best_cv_f1_macro: {results['best_cv_f1_macro']:.4f}",
        "",
        "=" * 60,
    ])
    with open(KNN_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_knn_learning_curves(X_train, y_train, X_test, y_test, best_config=None, cv=5):
    """
    Step 2: Learning curves — train/val Macro-F1 vs training size.
    (a) Baseline: small k (5). (b) Tuned: best_config from Step 1.
    Plot two panels. Append to KNN_results.txt.
    """
    if best_config is None:
        best_config = {"k": 20, "weights": "uniform", "metric": "euclidean"}
    
    np.random.seed(RANDOM_SEED)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    train_sizes = np.linspace(0.1, 1.0, 10)

    # Baseline: k=5, uniform, euclidean
    clf_baseline = KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean")
    lc_sizes, lc_train_b, lc_val_b = learning_curve(
        clf_baseline, X_train, y_train, train_sizes=train_sizes, cv=cv_splitter, scoring="f1_macro"
    )
    # Tuned: best config
    clf_tuned = KNeighborsClassifier(
        n_neighbors=best_config["k"], weights=best_config["weights"], metric=best_config["metric"]
    )
    _, lc_train_t, lc_val_t = learning_curve(
        clf_tuned, X_train, y_train, train_sizes=train_sizes, cv=cv_splitter, scoring="f1_macro"
    )

    results = {
        "train_sizes": np.asarray(lc_sizes).flatten().tolist(),
        "baseline": {
            "train_f1_macro_mean": lc_train_b.mean(axis=1).tolist(),
            "train_f1_macro_std": lc_train_b.std(axis=1).tolist(),
            "val_f1_macro_mean": lc_val_b.mean(axis=1).tolist(),
            "val_f1_macro_std": lc_val_b.std(axis=1).tolist(),
        },
        "tuned": {
            "train_f1_macro_mean": lc_train_t.mean(axis=1).tolist(),
            "train_f1_macro_std": lc_train_t.std(axis=1).tolist(),
            "val_f1_macro_mean": lc_val_t.mean(axis=1).tolist(),
            "val_f1_macro_std": lc_val_t.std(axis=1).tolist(),
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
        ["Baseline (k=5, uniform, euclidean)", "Tuned (best from Step 1)"],
        [results["baseline"], results["tuned"]],
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
    plt.savefig(os.path.join(os.path.dirname(KNN_RESULTS_PATH), "outputs", "knn_learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()


def _print_learning_curves(results):
    print("--- kNN Step 2: Learning curves ---")
    print("train_sizes:", results["train_sizes"])
    print("Baseline (k=5): val_f1_macro_mean:", [round(x, 4) for x in results["baseline"]["val_f1_macro_mean"]])
    print("Tuned:", results["best_config"], "val_f1_macro_mean:", [round(x, 4) for x in results["tuned"]["val_f1_macro_mean"]])
    print("Appended to:", KNN_RESULTS_PATH)


def _append_learning_curves(results):
    lines = [
        "",
        "--- Learning curves ---",
        "train_sizes: " + str(results["train_sizes"]),
        "baseline (k=5, uniform, euclidean):",
        "  train_f1_macro_mean: " + str([round(x, 4) for x in results["baseline"]["train_f1_macro_mean"]]),
        "  val_f1_macro_mean: " + str([round(x, 4) for x in results["baseline"]["val_f1_macro_mean"]]),
        "tuned (best_config from Step 1): " + str(results["best_config"]),
        "  train_f1_macro_mean: " + str([round(x, 4) for x in results["tuned"]["train_f1_macro_mean"]]),
        "  val_f1_macro_mean: " + str([round(x, 4) for x in results["tuned"]["val_f1_macro_mean"]]),
        "",
    ]
    with open(KNN_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_knn_test_eval(X_train, y_train, X_test, y_test, best_config=None):
    """
    Step 3: Refit best kNN on full train; evaluate once on test.
    best_config from Step 1 (no separate grid search). Append to KNN_results.txt.
    """
    if best_config is None:
        best_config = {"k": 20, "weights": "uniform", "metric": "euclidean"}
    
    np.random.seed(RANDOM_SEED)
    clf = KNeighborsClassifier(
        n_neighbors=best_config["k"], weights=best_config["weights"], metric=best_config["metric"]
    )
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
    ax.set_title('kNN Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(KNN_RESULTS_PATH), "outputs", "knn_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()


def _print_test_eval(results):
    print("--- kNN Step 3: Test evaluation ---")
    print("Best config:", results["best_config"])
    print("Test metrics:", results["test_metrics"])
    print("Runtime - fit (s):", round(results["runtime"]["fit_sec"], 4), "| predict (s):", round(results["runtime"]["predict_sec"], 4))
    print("Confusion matrix:", results["confusion_matrix"])
    print("Appended to:", KNN_RESULTS_PATH)


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
    with open(KNN_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
