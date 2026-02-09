"""
Support Vector Machines for Adult Income.
Evaluate at least 2 kernels (linear vs RBF); tune C and gamma (where relevant).
Scale features (required — preprocessing uses StandardScaler).
Required: learning curves, model-complexity curves (C, gamma), runtime.
"""

import os
import platform
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix

from config import RANDOM_SEED, OUTPUT_DIR
from evaluation import score_binary

SVM_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SVM_results.txt")


def _svm_results_path(class_weight):
    """Path for results file; use _class_weight.txt when class_weight='balanced'."""
    return SVM_RESULTS_PATH.replace(".txt", "_class_weight.txt") if class_weight == "balanced" else SVM_RESULTS_PATH


def _svm_plot_suffix(class_weight):
    """Filename suffix for plots when using class_weight='balanced'."""
    return "_class_weight" if class_weight == "balanced" else ""


# Hyperparameter ranges
C_VALUES = [1e-3, 0.01, 0.1, 1, 10, 100]  # 10^(-3) through 100
C_REF = 1  # reference C for table


def _rbf_gammas_around_scale(X_train):
    """RBF gammas: 'scale' plus values around it. Sklearn uses scale = 1/(n_features * X.var())."""
    n_features = X_train.shape[1]
    scale_val = 1.0 / (n_features * np.var(X_train))
    # Return 'scale' (sklearn keyword) and two numeric gammas around that value
    return ["scale", round(scale_val * 0.5, 6), round(scale_val * 2.0, 6)]


def _get_hardware_note():
    try:
        cpu = platform.processor() or platform.machine() or "unknown"
        return f"{platform.system()} {platform.release()}, CPU: {cpu}"
    except Exception:
        return "unknown"


def run_svm_step1(X_train, y_train, X_test, y_test, cv=5, class_weight=None):
    """
    SVM model-complexity: linear and RBF kernels, CV F1 vs C (and gamma for RBF).
    Two panels: Linear (C vs F1) | RBF (C vs F1 for gamma=scale and gammas around scale).
    class_weight: None (default) or 'balanced'.
    """
    results = {"linear": {}, "rbf": {}, "best_config": None, "best_cv_f1": -1, "class_weight": class_weight}
    np.random.seed(RANDOM_SEED)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    cw_kw = {} if class_weight is None else {"class_weight": class_weight}

    # Linear: CV F1 vs C
    linear_cv_f1 = []
    for c in tqdm(C_VALUES, desc="SVM linear (C)"):
        clf = SVC(kernel="linear", C=c, random_state=RANDOM_SEED, max_iter=3000, **cw_kw)
        scores = cross_val_score(clf, X_train, y_train, cv=cv_splitter, scoring="f1", n_jobs=-1)
        linear_cv_f1.append(float(scores.mean()))
    results["linear"] = {"C_values": list(C_VALUES), "cv_f1": linear_cv_f1}
    best_linear_idx = int(np.argmax(linear_cv_f1))
    best_linear_f1 = linear_cv_f1[best_linear_idx]
    if best_linear_f1 > results["best_cv_f1"]:
        results["best_cv_f1"] = best_linear_f1
        results["best_config"] = {"kernel": "linear", "C": C_VALUES[best_linear_idx], "gamma": None}

    # RBF: CV F1 vs C for gamma='scale' and gammas around scale (derived from X_train)
    rbf_gammas = _rbf_gammas_around_scale(X_train)
    rbf_curves = {}
    for gamma in tqdm(rbf_gammas, desc="SVM RBF (gamma)"):
        key = f"gamma={gamma}"
        cv_f1_list = []
        for c in C_VALUES:
            clf = SVC(kernel="rbf", C=c, gamma=gamma, random_state=RANDOM_SEED, max_iter=3000, **cw_kw)
            scores = cross_val_score(clf, X_train, y_train, cv=cv_splitter, scoring="f1", n_jobs=-1)
            cv_f1_list.append(float(scores.mean()))
        rbf_curves[key] = cv_f1_list
        best_idx = int(np.argmax(cv_f1_list))
        if cv_f1_list[best_idx] > results["best_cv_f1"]:
            results["best_cv_f1"] = cv_f1_list[best_idx]
            results["best_config"] = {"kernel": "rbf", "C": C_VALUES[best_idx], "gamma": gamma}
    results["rbf"] = {"C_values": list(C_VALUES), "curves": rbf_curves}

    _plot_step1(results)
    _print_step1(results)
    _save_step1(results)
    return results


def _plot_step1(results):
    """Two panels: Linear (C vs F1) | RBF (C vs F1 for gamma=scale and gammas around scale)."""
    cw = results.get("class_weight")
    suffix = _svm_plot_suffix(cw)
    title_suffix = " (class_weight=balanced)" if cw == "balanced" else ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Panel 1: Linear
    ax = axes[0]
    ax.plot(results["linear"]["C_values"], results["linear"]["cv_f1"], "o-", label="linear")
    ax.set_xlabel("C")
    ax.set_ylabel("Cross-Val F1")
    ax.set_title("SVM — Linear kernel (CV F1 vs C)" + title_suffix)
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Panel 2: RBF
    ax = axes[1]
    for key, vals in results["rbf"]["curves"].items():
        ax.plot(results["rbf"]["C_values"], vals, "o-", label=key)
    ax.set_xlabel("C")
    ax.set_ylabel("Cross-Val F1")
    ax.set_title("SVM — RBF kernel (CV F1 vs C)" + title_suffix)
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"svm_model_complexity{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _print_step1(results):
    c_ref_idx = C_VALUES.index(C_REF) if C_REF in C_VALUES else 0
    print("--- SVM Step 1: Model-complexity (linear vs RBF, CV F1 vs C) ---")
    print("Linear — CV F1 at C=", C_VALUES[c_ref_idx], ":", results["linear"]["cv_f1"][c_ref_idx])
    for key in results["rbf"]["curves"]:
        print("RBF", key, "— CV F1:", [round(x, 4) for x in results["rbf"]["curves"][key]])
    print("Best config:", results["best_config"], "| best_cv_f1:", round(results["best_cv_f1"], 4))
    print("Results saved to:", _svm_results_path(results.get("class_weight")))


def _save_step1(results):
    cw = results.get("class_weight")
    header = "SUPPORT VECTOR MACHINES — RESULTS (Adult Income)" + (" [class_weight=balanced]" if cw == "balanced" else "")
    lines = [
        "=" * 60,
        header,
        "=" * 60,
        "",
        "--- Step 1: Model-complexity (linear vs RBF, CV F1 vs C) ---",
        "C_values: " + str(C_VALUES),
        "Linear cv_f1: " + str([round(x, 4) for x in results["linear"]["cv_f1"]]),
        "",
    ]
    for key in results["rbf"]["curves"]:
        lines.append(f"RBF {key} cv_f1: " + str([round(x, 4) for x in results["rbf"]["curves"][key]]))
    lines.extend([
        "",
        f"Best config: {results['best_config']}",
        f"best_cv_f1: {results['best_cv_f1']:.4f}",
        "",
        "=" * 60,
    ])
    with open(_svm_results_path(cw), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


DEFAULT_BEST_CONFIG = {"kernel": "linear", "C": 1, "gamma": None}


def run_svm_learning_curves(X_train, y_train, X_test, y_test, best_config=None, cv=5, class_weight=None):
    """
    Step 2: Learning curves — train/val F1 vs training size.
    class_weight: None (default) or 'balanced'.
    """
    best_config = best_config or DEFAULT_BEST_CONFIG
    np.random.seed(RANDOM_SEED)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    train_sizes = np.linspace(0.1, 1.0, 10)
    cw_kw = {} if class_weight is None else {"class_weight": class_weight}

    lc_sizes = lc_train_b = lc_val_b = lc_train_t = lc_val_t = None
    if best_config["kernel"] == "linear":
        clf_tuned = SVC(kernel="linear", C=best_config["C"], random_state=RANDOM_SEED, max_iter=3000, **cw_kw)
    else:
        clf_tuned = SVC(kernel="rbf", C=best_config["C"], gamma=best_config["gamma"],
                        random_state=RANDOM_SEED, max_iter=3000, **cw_kw)
    configs = [
        ("baseline", SVC(kernel="linear", C=0.1, random_state=RANDOM_SEED, max_iter=3000, **cw_kw)),
        ("tuned", clf_tuned),
    ]
    for label, clf in tqdm(configs, desc="SVM learning curves"):
        sizes, train_f1, val_f1 = learning_curve(
            clf, X_train, y_train, train_sizes=train_sizes, cv=cv_splitter, scoring="f1", n_jobs=-1
        )
        if label == "baseline":
            lc_sizes, lc_train_b, lc_val_b = sizes, train_f1, val_f1
        else:
            lc_train_t, lc_val_t = train_f1, val_f1

    results = {
        "train_sizes": np.asarray(lc_sizes).flatten().tolist(),
        "baseline": {
            "train_f1_mean": lc_train_b.mean(axis=1).tolist(),
            "train_f1_std": lc_train_b.std(axis=1).tolist(),
            "val_f1_mean": lc_val_b.mean(axis=1).tolist(),
            "val_f1_std": lc_val_b.std(axis=1).tolist(),
        },
        "tuned": {
            "train_f1_mean": lc_train_t.mean(axis=1).tolist(),
            "train_f1_std": lc_train_t.std(axis=1).tolist(),
            "val_f1_mean": lc_val_t.mean(axis=1).tolist(),
            "val_f1_std": lc_val_t.std(axis=1).tolist(),
        },
        "best_config": best_config,
        "class_weight": class_weight,
    }
    _plot_learning_curves(results)
    _print_learning_curves(results)
    _append_learning_curves(results)
    return results


def _plot_learning_curves(results):
    suffix = _svm_plot_suffix(results.get("class_weight"))
    title_suffix = " (class_weight=balanced)" if results.get("class_weight") == "balanced" else ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sizes = results["train_sizes"]
    for ax, label, data in zip(
        axes,
        ["Baseline (linear, C=0.1)", "Tuned (best from Step 1)"],
        [results["baseline"], results["tuned"]],
    ):
        ax.plot(sizes, data["train_f1_mean"], "o-", label="Train F1")
        ax.fill_between(
            sizes,
            np.array(data["train_f1_mean"]) - np.array(data["train_f1_std"]),
            np.array(data["train_f1_mean"]) + np.array(data["train_f1_std"]),
            alpha=0.2,
        )
        ax.plot(sizes, data["val_f1_mean"], "s-", label="Val F1")
        ax.fill_between(
            sizes,
            np.array(data["val_f1_mean"]) - np.array(data["val_f1_std"]),
            np.array(data["val_f1_mean"]) + np.array(data["val_f1_std"]),
            alpha=0.2,
        )
        ax.set_xlabel("Training size")
        ax.set_ylabel("F1")
        ax.set_title(label + title_suffix)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"svm_learning_curves{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _print_learning_curves(results):
    print("--- SVM Step 2: Learning curves ---")
    print("train_sizes:", results["train_sizes"])
    print("Baseline (linear C=0.1): val_f1_mean:", [round(x, 4) for x in results["baseline"]["val_f1_mean"]])
    print("Tuned:", results["best_config"], "val_f1_mean:", [round(x, 4) for x in results["tuned"]["val_f1_mean"]])
    print("Appended to:", _svm_results_path(results.get("class_weight")))


def _append_learning_curves(results):
    lines = [
        "",
        "--- Learning curves ---",
        "train_sizes: " + str(results["train_sizes"]),
        "baseline (linear C=0.1):",
        "  train_f1_mean: " + str([round(x, 4) for x in results["baseline"]["train_f1_mean"]]),
        "  val_f1_mean: " + str([round(x, 4) for x in results["baseline"]["val_f1_mean"]]),
        "tuned (best_config from Step 1): " + str(results["best_config"]),
        "  train_f1_mean: " + str([round(x, 4) for x in results["tuned"]["train_f1_mean"]]),
        "  val_f1_mean: " + str([round(x, 4) for x in results["tuned"]["val_f1_mean"]]),
        "",
    ]
    with open(_svm_results_path(results.get("class_weight")), "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_svm_test_eval(X_train, y_train, X_test, y_test, best_config=None, class_weight=None):
    """
    Step 3: Refit best SVM on full train; evaluate once on test.
    class_weight: None (default) or 'balanced'.
    """
    best_config = best_config or DEFAULT_BEST_CONFIG
    np.random.seed(RANDOM_SEED)
    cw_kw = {} if class_weight is None else {"class_weight": class_weight}
    if best_config["kernel"] == "linear":
        clf = SVC(kernel="linear", C=best_config["C"], random_state=RANDOM_SEED, max_iter=3000, probability=True, **cw_kw)
    else:
        clf = SVC(
            kernel="rbf", C=best_config["C"], gamma=best_config["gamma"],
            random_state=RANDOM_SEED, max_iter=3000, probability=True, **cw_kw
        )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    predict_time = time.perf_counter() - t0
    metrics = score_binary(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "best_config": best_config,
        "test_metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "runtime": {"fit_sec": fit_time, "predict_sec": predict_time},
        "class_weight": class_weight,
    }
    _plot_confusion_matrix(results)
    _print_test_eval(results)
    _append_test_eval(results)
    return results


def _plot_confusion_matrix(results):
    """results is dict with 'confusion_matrix' and optionally 'class_weight'."""
    cm = np.asarray(results["confusion_matrix"])
    suffix = _svm_plot_suffix(results.get("class_weight"))
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
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < np.array(cm).max() / 2 else "white")
    plt.colorbar(im, ax=ax, label="Count")
    ax.set_title("SVM Confusion Matrix (threshold 0.5)" + title_suffix)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"svm_confusion_matrix{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _print_test_eval(results):
    print("--- SVM Step 3: Test evaluation ---")
    print("Best config:", results["best_config"])
    print("Test metrics:", results["test_metrics"])
    print("Runtime - fit (s):", round(results["runtime"]["fit_sec"], 4), "| predict (s):", round(results["runtime"]["predict_sec"], 4))
    print("Confusion matrix (0=<=50K, 1=>50K):", results["confusion_matrix"])
    print("Appended to:", _svm_results_path(results.get("class_weight")))


def _append_test_eval(results):
    m = results["test_metrics"]
    cm = results["confusion_matrix"]
    rt = results["runtime"]
    lines = [
        "--- Test evaluation ---",
        "best_config (from Step 1): " + str(results["best_config"]),
        f"Accuracy: {m.get('accuracy', 0):.4f}",
        f"F1: {m.get('f1', 0):.4f}",
        f"PR-AUC: {m.get('pr_auc', 0):.4f}",
        f"Fit (sec): {rt['fit_sec']:.4f}",
        f"Predict (sec): {rt['predict_sec']:.4f}",
        "Hardware: " + _get_hardware_note(),
        "Confusion matrix (0=<=50K, 1=>50K): " + str(cm),
        "",
        "=" * 60,
    ]
    with open(_svm_results_path(results.get("class_weight")), "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def train_and_tune_svm(X_train, y_train, X_val, y_val):
    """Legacy; use run_svm_step1, run_svm_learning_curves, run_svm_test_eval."""
    raise NotImplementedError("Use run_svm_step1, run_svm_learning_curves, run_svm_test_eval.")
