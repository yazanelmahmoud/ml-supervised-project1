"""
Decision Tree for Wine Quality (Multiclass Classification).
Split criterion: Gini (faster, often equivalent to entropy for this task).
Pruning/regularization: ccp_alpha, max_depth, min_samples_leaf.
Learning curves, model-complexity curve, runtime, confusion matrix.
Uses class_weight='balanced' to handle severe class imbalance.
"""

import os
import platform
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix

from config import RANDOM_SEED, TASK_TYPE
from evaluation import score_multiclass

# Gini: faster to compute, often equivalent to entropy for splits; entropy slightly
# prefers balanced splits but Gini is preferred here for speed with similar performance.
CRITERION = "gini"
DT_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DT_results.txt")

# Hyperparameter grid for tuning (ccp_alpha, max_depth, min_samples_leaf)
CCP_ALPHAS = np.logspace(-4, -1, 12)
# Extended ccp_alphas for MC plot: include 0 and 1e-5 to visualize extremes
CCP_ALPHAS_EXTENDED = np.unique(np.concatenate([[0.0, 1e-5], CCP_ALPHAS]))
MAX_DEPTHS = [5, 10, 15, 20, 25, 30, None]
MIN_SAMPLES_LEAF = [1, 2, 5, 10, 20, 30, 50]
# Ranges for model-complexity curves (max_depth, min_samples_leaf)
MAX_DEPTHS_MC = [3, 5, 8, 10, 12, 15, 20, 25, 30, None]
MIN_SAMPLES_LEAF_MC = [1, 2, 5, 10, 20, 30, 50, 100]


def _get_hardware_note():
    """Return a brief hardware description for reproducibility."""
    try:
        cpu = platform.processor() or platform.machine() or "unknown"
        return f"{platform.system()} {platform.release()}, CPU: {cpu}"
    except Exception:
        return "unknown"


def _train_and_eval(clf, X_tr, y_tr, X_te, y_te):
    """Train and evaluate classifier, return metrics, predictions, and runtime."""
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te) if hasattr(clf, "predict_proba") else None
    predict_time = time.perf_counter() - t0
    metrics = score_multiclass(y_te, y_pred, y_proba)
    return metrics, y_pred, y_proba, fit_time, predict_time


def run_dt(X_train, y_train, X_test, y_test, cv=5):
    """
    Train and tune DT (ccp_alpha, max_depth, min_samples_leaf) via CV on training only.
    Uses class_weight='balanced' to handle severe class imbalance.
    Compute learning/model-complexity curves, runtime.
    Returns dict with all results and saves to DT_results.txt.
    """
    results = {}
    np.random.seed(RANDOM_SEED)

    # 1) Grid search: tune ccp_alpha, max_depth, min_samples_leaf via CV on training only
    # Use f1_macro scoring for multiclass
    param_grid = {
        "ccp_alpha": CCP_ALPHAS.tolist(),
        "max_depth": MAX_DEPTHS,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
    }
    base_clf = DecisionTreeClassifier(
        criterion=CRITERION, 
        random_state=RANDOM_SEED,
        class_weight='balanced'  # Handle severe class imbalance
    )
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(base_clf, param_grid, cv=cv_splitter, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train)
    best_ccp = gs.best_params_["ccp_alpha"]
    best_max_depth = gs.best_params_["max_depth"]
    best_min_samples = gs.best_params_["min_samples_leaf"]
    results["grid_search_best_cv_f1_macro"] = float(gs.best_score_)
    results["grid_search_best_params"] = dict(gs.best_params_)

    # 2) Model-complexity curve: Macro-F1 vs ccp_alpha (standard fixed params: max_depth=None, min_samples_leaf=1)
    mc_train_f1, mc_cross_val_f1 = [], []
    alphas_list = CCP_ALPHAS_EXTENDED.tolist()
    for alpha in CCP_ALPHAS_EXTENDED:
        clf = DecisionTreeClassifier(
            criterion=CRITERION,
            ccp_alpha=float(alpha),
            max_depth=None,
            min_samples_leaf=1,
            random_state=RANDOM_SEED,
            class_weight='balanced'
        )
        scores = cross_val_score(clf, X_train, y_train, cv=cv_splitter, scoring="f1_macro")
        mc_cross_val_f1.append(scores.mean())
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        mc_train_f1.append(score_multiclass(y_train, train_pred)["f1_macro"])
    cv_f1_arr = np.array(mc_cross_val_f1)
    best_idx_ccp = int(np.argmax(cv_f1_arr))
    results["model_complexity"] = {
        "ccp_alpha": alphas_list,
        "train_f1_macro": mc_train_f1,
        "cross_val_f1_macro": mc_cross_val_f1,
        "best_ccp_alpha_from_curve": alphas_list[best_idx_ccp],
        "best_cv_f1_macro_from_curve": float(cv_f1_arr[best_idx_ccp]),
    }

    # 2b) Model-complexity curve: Macro-F1 vs max_depth (standard: ccp_alpha=0, min_samples_leaf=1)
    mc_md_train_f1, mc_md_cross_val_f1 = [], []
    for depth in MAX_DEPTHS_MC:
        clf = DecisionTreeClassifier(
            criterion=CRITERION,
            ccp_alpha=0.0,
            max_depth=depth,
            min_samples_leaf=1,
            random_state=RANDOM_SEED,
            class_weight='balanced'
        )
        scores = cross_val_score(clf, X_train, y_train, cv=cv_splitter, scoring="f1_macro")
        mc_md_cross_val_f1.append(scores.mean())
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        mc_md_train_f1.append(score_multiclass(y_train, train_pred)["f1_macro"])
    cv_f1_md = np.array(mc_md_cross_val_f1)
    best_idx_md = int(np.argmax(cv_f1_md))
    results["model_complexity_max_depth"] = {
        "max_depths": list(MAX_DEPTHS_MC),
        "train_f1_macro": mc_md_train_f1,
        "cross_val_f1_macro": mc_md_cross_val_f1,
        "best_max_depth_from_curve": MAX_DEPTHS_MC[best_idx_md],
        "best_cv_f1_macro_from_curve": float(cv_f1_md[best_idx_md]),
    }

    # 2c) Model-complexity curve: Macro-F1 vs min_samples_leaf (standard: ccp_alpha=0, max_depth=None)
    mc_msl_train_f1, mc_msl_cross_val_f1 = [], []
    for msl in MIN_SAMPLES_LEAF_MC:
        clf = DecisionTreeClassifier(
            criterion=CRITERION,
            ccp_alpha=0.0,
            max_depth=None,
            min_samples_leaf=msl,
            random_state=RANDOM_SEED,
            class_weight='balanced'
        )
        scores = cross_val_score(clf, X_train, y_train, cv=cv_splitter, scoring="f1_macro")
        mc_msl_cross_val_f1.append(scores.mean())
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        mc_msl_train_f1.append(score_multiclass(y_train, train_pred)["f1_macro"])
    cv_f1_msl = np.array(mc_msl_cross_val_f1)
    best_idx_msl = int(np.argmax(cv_f1_msl))
    results["model_complexity_min_samples_leaf"] = {
        "min_samples_leaf": list(MIN_SAMPLES_LEAF_MC),
        "train_f1_macro": mc_msl_train_f1,
        "cross_val_f1_macro": mc_msl_cross_val_f1,
        "best_min_samples_leaf_from_curve": MIN_SAMPLES_LEAF_MC[best_idx_msl],
        "best_cv_f1_macro_from_curve": float(cv_f1_msl[best_idx_msl]),
    }

    # 3) Learning curves (train size vs train/val metric)
    train_sizes = np.linspace(0.1, 1.0, 10)
    best_clf_base = DecisionTreeClassifier(
        criterion=CRITERION,
        ccp_alpha=best_ccp,
        max_depth=best_max_depth,
        min_samples_leaf=best_min_samples,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )
    lc_sizes, lc_train, lc_cross_val = learning_curve(
        best_clf_base,
        X_train, y_train, train_sizes=train_sizes, cv=cv_splitter, scoring="f1_macro", random_state=RANDOM_SEED
    )
    sizes = np.asarray(lc_sizes).flatten()
    results["learning_curve"] = {
        "train_sizes": sizes.tolist(),
        "train_f1_macro_mean": lc_train.mean(axis=1).tolist(),
        "train_f1_macro_std": lc_train.std(axis=1).tolist(),
        "cross_val_f1_macro_mean": lc_cross_val.mean(axis=1).tolist(),
        "cross_val_f1_macro_std": lc_cross_val.std(axis=1).tolist(),
    }

    # 4) Final model on full train with best params
    clf_final = DecisionTreeClassifier(
        criterion=CRITERION,
        ccp_alpha=best_ccp,
        max_depth=best_max_depth,
        min_samples_leaf=best_min_samples,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )
    metrics, y_pred, y_proba, fit_time, predict_time = _train_and_eval(
        clf_final, X_train, y_train, X_test, y_test
    )
    cm = confusion_matrix(y_test, y_pred)

    results["best_model"] = clf_final
    results["best_params"] = {
        "criterion": CRITERION,
        "ccp_alpha": best_ccp,
        "max_depth": best_max_depth,
        "min_samples_leaf": best_min_samples,
        "class_weight": "balanced"
    }
    results["depth"] = int(clf_final.get_depth())
    results["n_leaves"] = int(clf_final.get_n_leaves())
    results["test_metrics"] = metrics
    results["confusion_matrix"] = cm.tolist()
    results["runtime"] = {"fit_sec": fit_time, "predict_sec": predict_time}

    save_results(results, X_train, y_train, X_test, y_test, y_pred, y_proba)
    _plot_and_print(results)
    return results


def _plot_and_print(results):
    """Plot learning curve and three model-complexity curves (ccp_alpha, max_depth, min_samples_leaf)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Learning curve
    ax = axes[0, 0]
    lc = results["learning_curve"]
    ax.plot(lc["train_sizes"], lc["train_f1_macro_mean"], "o-", label="Train Macro-F1")
    ax.fill_between(lc["train_sizes"],
        np.array(lc["train_f1_macro_mean"]) - np.array(lc["train_f1_macro_std"]),
        np.array(lc["train_f1_macro_mean"]) + np.array(lc["train_f1_macro_std"]), alpha=0.2)
    ax.plot(lc["train_sizes"], lc["cross_val_f1_macro_mean"], "s-", label="Cross-Val Macro-F1")
    ax.fill_between(lc["train_sizes"],
        np.array(lc["cross_val_f1_macro_mean"]) - np.array(lc["cross_val_f1_macro_std"]),
        np.array(lc["cross_val_f1_macro_mean"]) + np.array(lc["cross_val_f1_macro_std"]), alpha=0.2)
    ax.set_xlabel("Training size")
    ax.set_ylabel("Macro-F1")
    ax.set_title("DT Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Model-complexity vs ccp_alpha
    ax = axes[0, 1]
    mc = results["model_complexity"]
    alphas = np.array(mc["ccp_alpha"])
    ax.plot(alphas, mc["train_f1_macro"], "o-", label="Train Macro-F1")
    ax.plot(alphas, mc["cross_val_f1_macro"], "s-", label="Cross-Val Macro-F1")
    ax.axvline(mc["best_ccp_alpha_from_curve"], color="gray", ls="--", label="best")
    ax.set_xlabel("ccp_alpha")
    ax.set_ylabel("Macro-F1")
    ax.set_title("DT Model-Complexity (ccp_alpha)")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlim(left=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Model-complexity vs max_depth
    ax = axes[1, 0]
    mc_md = results["model_complexity_max_depth"]
    depths = mc_md["max_depths"]
    x_md = [d if d is not None else 40 for d in depths]
    ax.plot(x_md, mc_md["train_f1_macro"], "o-", label="Train Macro-F1")
    ax.plot(x_md, mc_md["cross_val_f1_macro"], "s-", label="Cross-Val Macro-F1")
    best_md = mc_md["best_max_depth_from_curve"]
    ax.axvline(best_md if best_md is not None else 40, color="gray", ls="--", label="best")
    ax.set_xticks(x_md)
    ax.set_xticklabels([str(d) if d is not None else "None" for d in depths], rotation=45)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Macro-F1")
    ax.set_title("DT Model-Complexity (max_depth)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Model-complexity vs min_samples_leaf
    ax = axes[1, 1]
    mc_msl = results["model_complexity_min_samples_leaf"]
    ax.plot(mc_msl["min_samples_leaf"], mc_msl["train_f1_macro"], "o-", label="Train Macro-F1")
    ax.plot(mc_msl["min_samples_leaf"], mc_msl["cross_val_f1_macro"], "s-", label="Cross-Val Macro-F1")
    ax.axvline(mc_msl["best_min_samples_leaf_from_curve"], color="gray", ls="--", label="best")
    ax.set_xlabel("min_samples_leaf")
    ax.set_ylabel("Macro-F1")
    ax.set_title("DT Model-Complexity (min_samples_leaf)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(DT_RESULTS_PATH), "outputs", "dt_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Confusion matrix heatmap
    cm = np.array(results["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Quality')
    ax.set_ylabel('True Quality')
    ax.set_title('Decision Tree Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(DT_RESULTS_PATH), "outputs", "dt_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Print summary
    mc = results["model_complexity"]
    mc_md = results["model_complexity_max_depth"]
    mc_msl = results["model_complexity_min_samples_leaf"]
    gs_f1 = results["grid_search_best_cv_f1_macro"]
    gs_bp = results["grid_search_best_params"]
    print("--- Best from each model-complexity curve (other params at standard) ---")
    print(f"  ccp_alpha curve (max_depth=None, min_samples_leaf=1): best ccp_alpha={mc['best_ccp_alpha_from_curve']:.6f}, CV Macro-F1={mc['best_cv_f1_macro_from_curve']:.4f}")
    print(f"  max_depth curve (ccp_alpha=0, min_samples_leaf=1):     best max_depth={mc_md['best_max_depth_from_curve']}, CV Macro-F1={mc_md['best_cv_f1_macro_from_curve']:.4f}")
    print(f"  min_samples_leaf curve (ccp_alpha=0, max_depth=None):  best min_samples_leaf={mc_msl['best_min_samples_leaf_from_curve']}, CV Macro-F1={mc_msl['best_cv_f1_macro_from_curve']:.4f}")
    print("--- Grid-search best (joint tuning) ---")
    print(f"  CV Macro-F1={gs_f1:.4f}, params={gs_bp}")
    print()
    bp = results["best_params"]
    print("Best params (used for final model):", bp)
    print("Test metrics:", results["test_metrics"])
    print("Depth:", results["depth"], "| Leaves:", results["n_leaves"])
    print("Runtime - fit:", round(results["runtime"]["fit_sec"], 4), "s | predict:", round(results["runtime"]["predict_sec"], 4), "s")
    print("\nConfusion matrix:")
    print(np.array(results["confusion_matrix"]))
    print("\nResults saved to:", DT_RESULTS_PATH)


def save_results(results, X_train, y_train, X_test, y_test, y_pred, y_proba):
    """Write DT results to DT_results.txt including DATA & METHODOLOGY."""
    m = results["test_metrics"]
    cm = np.array(results["confusion_matrix"])
    rt = results["runtime"]
    bp = results["best_params"]

    # Class distribution (train set)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique_classes, class_counts))
    total_train = len(y_train)
    imbalance_ratio = max(class_counts) / min(class_counts)

    # Per-class performance
    f1_per_class = m.get("f1_per_class", {})
    class_report = m.get("classification_report", {})

    lines = [
        "=" * 60,
        "DECISION TREE — RESULTS (Wine Quality)",
        "=" * 60,
        "",
        "--- DATA & METHODOLOGY ---",
        "Target: quality (discrete rating/class, 1-8); task: multiclass classification.",
        "Metrics: Macro-F1 and Accuracy (minimum). Macro-F1 treats all classes equally,",
        "         critical for severe class imbalance (ratio 108.29).",
        f"Class distribution (train): {dict(sorted(class_dist.items()))}.",
        f"Imbalance ratio (majority/minority): {imbalance_ratio:.2f}.",
        "Severe imbalance: minority classes (1, 8) represent <0.4% each;",
        "                   majority class (4) represents 34.6%.",
        "Leakage controls: 'class' column dropped (perfect correlation with 'quality').",
        "Single held-out test split; tuning via 5-fold CV on training only.",
        "Class weighting: 'balanced' used to handle severe imbalance.",
        "",
        "--- Split criterion and justification ---",
        f"Criterion: {CRITERION}. Gini is faster than entropy and yields similar splits;",
        "entropy slightly prefers balanced splits; for this dataset Gini is chosen for speed.",
        "",
        "--- Best hyperparameters (CV on training) ---",
        f"ccp_alpha: {bp['ccp_alpha']:.6f}",
        f"max_depth: {bp['max_depth']}",
        f"min_samples_leaf: {bp['min_samples_leaf']}",
        f"class_weight: {bp['class_weight']}",
        f"Final depth: {results['depth']}",
        f"Number of leaves: {results['n_leaves']}",
        "",
        "--- Best from each model-complexity curve (other params at standard) ---",
        "  ccp_alpha curve (max_depth=None, min_samples_leaf=1):",
        f"    best ccp_alpha={results['model_complexity']['best_ccp_alpha_from_curve']:.6f}, CV Macro-F1={results['model_complexity']['best_cv_f1_macro_from_curve']:.4f}",
        "  max_depth curve (ccp_alpha=0, min_samples_leaf=1):",
        f"    best max_depth={results['model_complexity_max_depth']['best_max_depth_from_curve']}, CV Macro-F1={results['model_complexity_max_depth']['best_cv_f1_macro_from_curve']:.4f}",
        "  min_samples_leaf curve (ccp_alpha=0, max_depth=None):",
        f"    best min_samples_leaf={results['model_complexity_min_samples_leaf']['best_min_samples_leaf_from_curve']}, CV Macro-F1={results['model_complexity_min_samples_leaf']['best_cv_f1_macro_from_curve']:.4f}",
        "",
        "--- Grid-search best (joint tuning, val Macro-F1) ---",
        f"CV Macro-F1: {results['grid_search_best_cv_f1_macro']:.4f}",
        f"Params: {results['grid_search_best_params']}",
        "",
        "--- Test metrics ---",
        f"Accuracy:     {m.get('accuracy', 0):.4f}",
        f"Macro-F1:     {m.get('f1_macro', 0):.4f}",
        f"Weighted-F1:  {m.get('f1_weighted', 0):.4f}",
        "",
        "--- Per-class performance (F1 scores) ---",
    ]
    
    # Add per-class F1 scores
    for class_label in sorted(unique_classes):
        class_key = f"class_{class_label}"
        f1_score = f1_per_class.get(class_key, 0.0)
        lines.append(f"  Quality {class_label}: F1 = {f1_score:.4f}")
    
    lines.extend([
        "",
        "--- Confusion matrix (rows=true, cols=predicted) ---",
    ])
    
    # Add confusion matrix with labels
    cm_str = "    " + " ".join([f"Q{i}" for i in sorted(unique_classes)])
    lines.append(cm_str)
    for i, class_label in enumerate(sorted(unique_classes)):
        row_str = f"Q{class_label} " + " ".join([f"{cm[i, j]:4d}" for j in range(len(unique_classes))])
        lines.append(row_str)
    
    lines.extend([
        "",
        "--- Runtime ---",
        f"Fit (sec):    {rt['fit_sec']:.4f}",
        f"Predict (sec): {rt['predict_sec']:.4f}",
        f"Hardware: {_get_hardware_note()}",
        "",
        "--- Note ---",
        "Revisit hypothesis from hypothesis.txt in DT conclusions.",
        "",
        "=" * 60,
    ])
    
    path = DT_RESULTS_PATH
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# Wrapper functions for consistency with notebook structure
def run_dt_model_complexity(X_train, y_train, X_test, y_test, cv=5):
    """
    Wrapper: Run full DT analysis including model-complexity curves.
    Returns results dict.
    """
    return run_dt(X_train, y_train, X_test, y_test, cv=cv)


def run_dt_learning_curves(X_train, y_train, X_test, y_test, best_config=None, cv=5):
    """
    Wrapper: Learning curves are included in run_dt().
    This function is kept for API consistency but calls run_dt().
    """
    print("Note: Learning curves are generated as part of run_dt().")
    return run_dt(X_train, y_train, X_test, y_test, cv=cv)


def run_dt_test_eval(X_train, y_train, X_test, y_test, best_config=None):
    """
    Wrapper: Test evaluation is included in run_dt().
    This function is kept for API consistency but calls run_dt().
    """
    print("Note: Test evaluation is generated as part of run_dt().")
    return run_dt(X_train, y_train, X_test, y_test)
