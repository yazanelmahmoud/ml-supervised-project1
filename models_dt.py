"""
Decision Tree for Adult Income.
Split criterion: Gini (faster, often equivalent to entropy for this task).
Pruning/regularization: ccp_alpha, max_depth, min_samples_leaf.
Learning curves, model-complexity curve, runtime, confusion matrix.
"""

import os
import platform
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix

from config import RANDOM_SEED, OUTPUT_DIR
from evaluation import score_binary


# Gini: faster to compute, often equivalent to entropy for splits; entropy slightly
# prefers balanced splits but Gini is preferred here for speed with similar performance.
CRITERION = "gini"
DT_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DT_results.txt")

# Hyperparameter grid for tuning (ccp_alpha, max_depth, min_samples_leaf)
CCP_ALPHAS = np.logspace(-4, -1, 12)
# Extended ccp_alphas for MC plot: include 0 and 1e-5 to visualize extremes
CCP_ALPHAS_EXTENDED = np.unique(np.concatenate([[0.0, 1e-5], CCP_ALPHAS]))
MAX_DEPTHS = [8, 12, 16, 20, 30, 50]
MIN_SAMPLES_LEAF = [1,4, 10, 30, 50, 70, 100]
# Ranges for model-complexity curves (max_depth, min_samples_leaf)
MAX_DEPTHS_MC = [2, 4, 6, 8, 10, 12, 16, 20, 30, 50]  # None = no limit
MIN_SAMPLES_LEAF_MC = [1,4, 10, 30, 50, 70, 100]


def _get_hardware_note():
    """Return a brief hardware description for reproducibility."""
    try:
        cpu = platform.processor() or platform.machine() or "unknown"
        return f"{platform.system()} {platform.release()}, CPU: {cpu}"
    except Exception:
        return "unknown"


def _train_and_eval(clf, X_tr, y_tr, X_te, y_te):
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None
    predict_time = time.perf_counter() - t0
    metrics = score_binary(y_te, y_pred, y_proba)
    return metrics, y_pred, y_proba, fit_time, predict_time


def run_dt(X_train, y_train, X_test, y_test, cv=5, class_weight=None):
    """
    Train and tune DT (ccp_alpha, max_depth, min_samples_leaf) via CV on training only.
    Compute learning/model-complexity curves, runtime.
    Returns dict with all results and saves to DT_results.txt (or DT_results_class_weight.txt if class_weight='balanced').
    class_weight: None (default) or 'balanced' for imbalance handling.
    """
    results = {}
    np.random.seed(RANDOM_SEED)
    cw_kw = {} if class_weight is None else {"class_weight": class_weight}

    # 1) Grid search: tune ccp_alpha, max_depth, min_samples_leaf via CV on training only
    param_grid = {
        "ccp_alpha": CCP_ALPHAS.tolist(),
        "max_depth": MAX_DEPTHS,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
    }
    base_clf = DecisionTreeClassifier(criterion=CRITERION, random_state=RANDOM_SEED, **cw_kw)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(base_clf, param_grid, cv=cv_splitter, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    best_ccp = gs.best_params_["ccp_alpha"]
    best_max_depth = gs.best_params_["max_depth"]
    best_min_samples = gs.best_params_["min_samples_leaf"]
    results["grid_search_best_cv_f1"] = float(gs.best_score_)
    results["grid_search_best_params"] = dict(gs.best_params_)

    # 2) Model-complexity curve: F1 vs ccp_alpha (standard fixed params: max_depth=None, min_samples_leaf=1)
    #    Include 0 and 1e-5 to see behavior at no pruning and very light pruning
    mc_train_f1, mc_cross_val_f1 = [], []
    alphas_list = CCP_ALPHAS_EXTENDED.tolist()
    for alpha in CCP_ALPHAS_EXTENDED:
        clf = DecisionTreeClassifier(
            criterion=CRITERION,
            ccp_alpha=float(alpha),
            max_depth=None,
            min_samples_leaf=1,
            random_state=RANDOM_SEED,
            **cw_kw,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
        mc_cross_val_f1.append(scores.mean())
        clf.fit(X_train, y_train)
        mc_train_f1.append(score_binary(y_train, clf.predict(X_train))["f1"])
    cv_f1_arr = np.array(mc_cross_val_f1)
    best_idx_ccp = int(np.argmax(cv_f1_arr))
    results["model_complexity"] = {
        "ccp_alpha": alphas_list,
        "train_f1": mc_train_f1,
        "cross_val_f1": mc_cross_val_f1,
        "best_ccp_alpha_from_curve": alphas_list[best_idx_ccp],
        "best_cv_f1_from_curve": float(cv_f1_arr[best_idx_ccp]),
    }

    # 2b) Model-complexity curve: F1 vs max_depth (standard: ccp_alpha=0, min_samples_leaf=1)
    mc_md_train_f1, mc_md_cross_val_f1 = [], []
    for depth in MAX_DEPTHS_MC:
        clf = DecisionTreeClassifier(
            criterion=CRITERION,
            ccp_alpha=0.0,
            max_depth=depth,
            min_samples_leaf=1,
            random_state=RANDOM_SEED,
            **cw_kw,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
        mc_md_cross_val_f1.append(scores.mean())
        clf.fit(X_train, y_train)
        mc_md_train_f1.append(score_binary(y_train, clf.predict(X_train))["f1"])
    cv_f1_md = np.array(mc_md_cross_val_f1)
    best_idx_md = int(np.argmax(cv_f1_md))
    results["model_complexity_max_depth"] = {
        "max_depths": list(MAX_DEPTHS_MC),
        "train_f1": mc_md_train_f1,
        "cross_val_f1": mc_md_cross_val_f1,
        "best_max_depth_from_curve": MAX_DEPTHS_MC[best_idx_md],
        "best_cv_f1_from_curve": float(cv_f1_md[best_idx_md]),
    }

    # 2c) Model-complexity curve: F1 vs min_samples_leaf (standard: ccp_alpha=0, max_depth=None)
    mc_msl_train_f1, mc_msl_cross_val_f1 = [], []
    for msl in MIN_SAMPLES_LEAF_MC:
        clf = DecisionTreeClassifier(
            criterion=CRITERION,
            ccp_alpha=0.0,
            max_depth=None,
            min_samples_leaf=msl,
            random_state=RANDOM_SEED,
            **cw_kw,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
        mc_msl_cross_val_f1.append(scores.mean())
        clf.fit(X_train, y_train)
        mc_msl_train_f1.append(score_binary(y_train, clf.predict(X_train))["f1"])
    cv_f1_msl = np.array(mc_msl_cross_val_f1)
    best_idx_msl = int(np.argmax(cv_f1_msl))
    results["model_complexity_min_samples_leaf"] = {
        "min_samples_leaf": list(MIN_SAMPLES_LEAF_MC),
        "train_f1": mc_msl_train_f1,
        "cross_val_f1": mc_msl_cross_val_f1,
        "best_min_samples_leaf_from_curve": MIN_SAMPLES_LEAF_MC[best_idx_msl],
        "best_cv_f1_from_curve": float(cv_f1_msl[best_idx_msl]),
    }

    # 3) Learning curves (train size vs train/val metric)
    train_sizes = np.linspace(0.1, 1.0, 10)
    best_clf_base = DecisionTreeClassifier(
        criterion=CRITERION,
        ccp_alpha=best_ccp,
        max_depth=best_max_depth,
        min_samples_leaf=best_min_samples,
        random_state=RANDOM_SEED,
        **cw_kw,
    )
    lc_sizes, lc_train, lc_cross_val = learning_curve(
        best_clf_base,
        X_train, y_train, train_sizes=train_sizes, cv=cv, scoring="f1", random_state=RANDOM_SEED
    )
    sizes = np.asarray(lc_sizes).flatten()
    results["learning_curve"] = {
        "train_sizes": sizes.tolist(),
        "train_f1_mean": lc_train.mean(axis=1).tolist(),
        "train_f1_std": lc_train.std(axis=1).tolist(),
        "cross_val_f1_mean": lc_cross_val.mean(axis=1).tolist(),
        "cross_val_f1_std": lc_cross_val.std(axis=1).tolist(),
    }

    # 4) Final model on full train with best params
    clf_final = DecisionTreeClassifier(
        criterion=CRITERION,
        ccp_alpha=best_ccp,
        max_depth=best_max_depth,
        min_samples_leaf=best_min_samples,
        random_state=RANDOM_SEED,
        **cw_kw,
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
    }
    results["class_weight"] = class_weight
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
    cw_suffix = " (class_weight=balanced)" if results.get("class_weight") == "balanced" else ""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Learning curve
    ax = axes[0, 0]
    lc = results["learning_curve"]
    ax.plot(lc["train_sizes"], lc["train_f1_mean"], "o-", label="Train F1")
    ax.fill_between(lc["train_sizes"],
        np.array(lc["train_f1_mean"]) - np.array(lc["train_f1_std"]),
        np.array(lc["train_f1_mean"]) + np.array(lc["train_f1_std"]), alpha=0.2)
    ax.plot(lc["train_sizes"], lc["cross_val_f1_mean"], "s-", label="Cross-Val F1")
    ax.fill_between(lc["train_sizes"],
        np.array(lc["cross_val_f1_mean"]) - np.array(lc["cross_val_f1_std"]),
        np.array(lc["cross_val_f1_mean"]) + np.array(lc["cross_val_f1_std"]), alpha=0.2)
    ax.set_xlabel("Training size")
    ax.set_ylabel("F1")
    ax.set_title("DT Learning Curve" + cw_suffix)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Model-complexity vs ccp_alpha (best = best on this curve)
    ax = axes[0, 1]
    mc = results["model_complexity"]
    alphas = np.array(mc["ccp_alpha"])
    ax.plot(alphas, mc["train_f1"], "o-", label="Train F1")
    ax.plot(alphas, mc["cross_val_f1"], "s-", label="Cross-Val F1")
    ax.axvline(mc["best_ccp_alpha_from_curve"], color="gray", ls="--", label="best")
    ax.set_xlabel("ccp_alpha")
    ax.set_ylabel("F1")
    ax.set_title("DT Model-Complexity (ccp_alpha)" + cw_suffix)
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlim(left=0)  # start at 0, no negative part on x-axis
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Model-complexity vs max_depth (best = best on this curve)
    ax = axes[1, 0]
    mc_md = results["model_complexity_max_depth"]
    depths = mc_md["max_depths"]
    x_md = [d if d is not None else 40 for d in depths]  # None -> 40 for display
    ax.plot(x_md, mc_md["train_f1"], "o-", label="Train F1")
    ax.plot(x_md, mc_md["cross_val_f1"], "s-", label="Cross-Val F1")
    best_md = mc_md["best_max_depth_from_curve"]
    ax.axvline(best_md if best_md is not None else 40, color="gray", ls="--", label="best")
    ax.set_xticks(x_md)
    ax.set_xticklabels([str(d) if d is not None else "None" for d in depths])
    ax.set_xlabel("max_depth")
    ax.set_ylabel("F1")
    ax.set_title("DT Model-Complexity (max_depth)" + cw_suffix)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Model-complexity vs min_samples_leaf (best = best on this curve)
    ax = axes[1, 1]
    mc_msl = results["model_complexity_min_samples_leaf"]
    ax.plot(mc_msl["min_samples_leaf"], mc_msl["train_f1"], "o-", label="Train F1")
    ax.plot(mc_msl["min_samples_leaf"], mc_msl["cross_val_f1"], "s-", label="Cross-Val F1")
    ax.axvline(mc_msl["best_min_samples_leaf_from_curve"], color="gray", ls="--", label="best")
    ax.set_xlabel("min_samples_leaf")
    ax.set_ylabel("F1")
    ax.set_title("DT Model-Complexity (min_samples_leaf)" + cw_suffix)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Best from each model-complexity curve (val F1) and grid-search best
    mc = results["model_complexity"]
    mc_md = results["model_complexity_max_depth"]
    mc_msl = results["model_complexity_min_samples_leaf"]
    gs_f1 = results["grid_search_best_cv_f1"]
    gs_bp = results["grid_search_best_params"]
    print("--- Best from each model-complexity curve (other params at standard) ---")
    print(f"  ccp_alpha curve (max_depth=None, min_samples_leaf=1): best ccp_alpha={mc['best_ccp_alpha_from_curve']:.6f}, CV F1={mc['best_cv_f1_from_curve']:.4f}")
    print(f"  max_depth curve (ccp_alpha=0, min_samples_leaf=1):     best max_depth={mc_md['best_max_depth_from_curve']}, CV F1={mc_md['best_cv_f1_from_curve']:.4f}")
    print(f"  min_samples_leaf curve (ccp_alpha=0, max_depth=None):  best min_samples_leaf={mc_msl['best_min_samples_leaf_from_curve']}, CV F1={mc_msl['best_cv_f1_from_curve']:.4f}")
    print("--- Grid-search best (joint tuning) ---")
    print(f"  CV F1={gs_f1:.4f}, params={gs_bp}")
    print()
    bp = results["best_params"]
    print("Best params (used for final model):", bp)
    print("Test metrics:", results["test_metrics"])
    print("Depth:", results["depth"], "| Leaves:", results["n_leaves"])
    print("Runtime - fit:", round(results["runtime"]["fit_sec"], 4), "s | predict:", round(results["runtime"]["predict_sec"], 4), "s")
    print("\nConfusion matrix (0=<=50K, 1=>50K):")
    for row in results["confusion_matrix"]:
        print(row)
    _plot_confusion_matrix(results)
    _results_path = DT_RESULTS_PATH if results.get("class_weight") != "balanced" else DT_RESULTS_PATH.replace(".txt", "_class_weight.txt")
    print("\nResults saved to:", _results_path)


def _plot_confusion_matrix(results):
    """Plot confusion matrix for best DT (binary: <=50K, >50K). Save to outputs/."""
    cm = np.array(results["confusion_matrix"])
    suffix = "_class_weight" if results.get("class_weight") == "balanced" else ""
    title_suffix = " (class_weight=balanced)" if results.get("class_weight") == "balanced" else ""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["<=50K", ">50K"])
    ax.set_yticklabels(["<=50K", ">50K"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white")
    plt.colorbar(im, ax=ax, label="Count")
    ax.set_title("DT Confusion Matrix (threshold 0.5)" + title_suffix)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"dt_confusion_matrix{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def save_results(results, X_train, y_train, X_test, y_test, y_pred, y_proba):
    """Write DT results to DT_results.txt (or _class_weight.txt when class_weight='balanced')."""
    m = results["test_metrics"]
    cm = np.array(results["confusion_matrix"])
    rt = results["runtime"]
    bp = results["best_params"]

    # Class distribution (train set)
    n_train = len(y_train)
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    imbalance_ratio = n_neg / max(n_pos, 1)

    cw_note = "Class weight: balanced (sklearn class_weight='balanced')." if results.get("class_weight") == "balanced" else ""
    lines = [
        "=" * 60,
        "DECISION TREE — RESULTS (Adult Income)" + (" [class_weight=balanced]" if results.get("class_weight") == "balanced" else ""),
        "=" * 60,
        "",
        "--- DATA & METHODOLOGY ---",
        "Target: class (<=50K vs >50K); task: binary classification.",
        cw_note if cw_note else "",
        "Metrics: F1, Accuracy, PR-AUC — imbalance makes accuracy insufficient;",
        "         F1 and PR-AUC better reflect minority-class performance.",
        f"Class distribution (train): {n_neg} <=50K, {n_pos} >50K (~{imbalance_ratio:.2f}:1).",
        "Imbalance: minority class (>50K) under-represented; F1/PR-AUC preferred.",
        "Leakage controls: fnlwgt and education dropped (EDA: redundancy, near-zero correlation).",
        "Single held-out test split; tuning via 5-fold CV on training only.",
        "",
        "--- Split criterion and justification ---",
        f"Criterion: {CRITERION}. Gini is faster than entropy and yields similar splits;",
        "entropy slightly prefers balanced splits; for this dataset Gini is chosen for speed.",
        "",
        "--- Best hyperparameters (CV on training) ---",
        f"ccp_alpha: {bp['ccp_alpha']:.6f}",
        f"max_depth: {bp['max_depth']}",
        f"min_samples_leaf: {bp['min_samples_leaf']}",
        f"Final depth: {results['depth']}",
        f"Number of leaves: {results['n_leaves']}",
        "",
        "--- Best from each model-complexity curve (other params at standard) ---",
        "  ccp_alpha curve (max_depth=None, min_samples_leaf=1):",
        f"    best ccp_alpha={results['model_complexity']['best_ccp_alpha_from_curve']:.6f}, CV F1={results['model_complexity']['best_cv_f1_from_curve']:.4f}",
        "  max_depth curve (ccp_alpha=0, min_samples_leaf=1):",
        f"    best max_depth={results['model_complexity_max_depth']['best_max_depth_from_curve']}, CV F1={results['model_complexity_max_depth']['best_cv_f1_from_curve']:.4f}",
        "  min_samples_leaf curve (ccp_alpha=0, max_depth=None):",
        f"    best min_samples_leaf={results['model_complexity_min_samples_leaf']['best_min_samples_leaf_from_curve']}, CV F1={results['model_complexity_min_samples_leaf']['best_cv_f1_from_curve']:.4f}",
        "",
        "--- Grid-search best (joint tuning, val F1) ---",
        f"CV F1: {results['grid_search_best_cv_f1']:.4f}",
        f"Params: {results['grid_search_best_params']}",
        "",
        "--- Test metrics ---",
        f"Accuracy:  {m.get('accuracy', 0):.4f}",
        f"F1:        {m.get('f1', 0):.4f}",
        f"PR-AUC:    {m.get('pr_auc', 0):.4f}",
        "",
        "--- Confusion matrix (0=<=50K, 1=>50K, threshold 0.5) ---",
        f"TN={cm[0,0]}  FP={cm[0,1]}",
        f"FN={cm[1,0]}  TP={cm[1,1]}",
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
    ]
    path = DT_RESULTS_PATH if results.get("class_weight") != "balanced" else DT_RESULTS_PATH.replace(".txt", "_class_weight.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
