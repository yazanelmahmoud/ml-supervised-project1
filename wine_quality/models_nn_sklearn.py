"""
Neural Network — scikit-learn MLPClassifier for Wine Quality (Multiclass Classification).
SGD only (no momentum, Nesterov, Adam/Adagrad/RMSprop).
Report: learning rate, batch size, epochs/early stopping, regularization.
Required: epoch-based learning curves (train vs val loss/metric), early stopping.
Compare capacity scaling (depth vs width) with ~constant parameter count.
Uses Macro-F1 for multiclass evaluation.
Note: MLPClassifier doesn't support class_weight parameter directly.
"""

import os
import platform
import time
from collections import defaultdict
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import RANDOM_SEED, OUTPUT_DIR, TASK_TYPE
from evaluation import score_multiclass

# Results file (sklearn pipeline)
NN_SKLEARN_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN_sklearn_results.txt")

# Step 1 — Width search
WIDTH_VALUES = [8, 16, 32, 64, 128, 200, 400, 700]
# Step 2 — Depth vs width: [64], [32,32], [16,16,16,16]
STEP2_ARCHITECTURES = [(64,), (32, 32), (16, 16, 16, 16)]
# Step 3 — LR sweep
STEP3_LR_VALUES = [0.1, 0.01, 0.001, 0.0003]

NN_LR = 0.01
NN_L2 = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
CV_SPLITS = 5
N_JOBS = -1  # Use all CPU cores for parallel CV


def _get_hardware_note():
    """Return a brief hardware description for reproducibility."""
    try:
        cpu = platform.processor() or platform.machine() or "unknown"
        return f"{platform.system()} {platform.release()}, CPU: {cpu}"
    except Exception:
        return "unknown"


def _fit_one_fold_width(w, max_iter, train_idx, val_idx, X_train, y_train):
    """Fit one (width, fold); return (train_f1_macro, val_f1_macro). Picklable for joblib."""
    from sklearn.metrics import f1_score
    X_tr = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    clf = _make_mlp((w,), max_iter=max_iter)
    clf.fit(X_tr, y_tr)
    return (f1_score(y_tr, clf.predict(X_tr), average='macro', zero_division=0), 
            f1_score(y_val, clf.predict(X_val), average='macro', zero_division=0))


def _fit_one_fold_arch(arch, max_iter, train_idx, val_idx, X_train, y_train):
    """Fit one (arch, fold); return (train_f1_macro, val_f1_macro). Picklable for joblib."""
    from sklearn.metrics import f1_score
    X_tr = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    clf = _make_mlp(arch, max_iter=max_iter)
    clf.fit(X_tr, y_tr)
    return (f1_score(y_tr, clf.predict(X_tr), average='macro', zero_division=0), 
            f1_score(y_val, clf.predict(X_val), average='macro', zero_division=0))


def _fit_one_fold_lr(arch, lr, max_iter, train_idx, val_idx, X_train, y_train):
    """Fit one (lr, fold); return (train_f1_macro, val_f1_macro). Picklable for joblib."""
    from sklearn.metrics import f1_score
    X_tr = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    clf = _make_mlp(arch, learning_rate_init=lr, max_iter=max_iter)
    clf.fit(X_tr, y_tr)
    return (f1_score(y_tr, clf.predict(X_tr), average='macro', zero_division=0), 
            f1_score(y_val, clf.predict(X_val), average='macro', zero_division=0))


def _append_nn_results(text):
    """Append a line or block to NN_sklearn_results.txt."""
    with open(NN_SKLEARN_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def _get_max_iter(n_samples):
    """Max iterations for ~100 epochs with batch_size=64."""
    return max(1000, 100 * ((n_samples + BATCH_SIZE - 1) // BATCH_SIZE))


def _make_mlp(hidden_layer_sizes, learning_rate_init=NN_LR, alpha=NN_L2, max_iter=10000, random_state=RANDOM_SEED):
    """Build MLPClassifier with SGD, no momentum, early stopping.
    Note: MLPClassifier doesn't support class_weight parameter directly.
    For class imbalance, consider using sample_weight in fit() if needed.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="sgd",
        alpha=alpha,
        batch_size=BATCH_SIZE,
        learning_rate="constant",
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=EARLY_STOPPING_PATIENCE,
        validation_fraction=0.1,
        random_state=random_state,
        momentum=0.0
    )


def run_nn_step1(X_train, y_train, X_test, y_test, cv=CV_SPLITS):
    """
    Step 1 — Width search (1 hidden layer).
    Architecture: Input → Dense(W) → ReLU → Output.
    Uses stratified K-fold CV on training set.
    For each width: mean train Macro-F1 and mean CV Macro-F1; plot Model Complexity Curve.
    """
    np.random.seed(RANDOM_SEED)
    # Start fresh results file
    with open(NN_SKLEARN_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("NN (sklearn) pipeline results — Wine Quality\n")
        f.write("=" * 60 + "\n\n")
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    n_samples = X_train.shape[0]
    max_iter = _get_max_iter(n_samples)

    folds_list = list(cv_splitter.split(X_train, y_train))
    tasks = [
        (w, max_iter, train_idx, val_idx)
        for w in WIDTH_VALUES
        for train_idx, val_idx in folds_list
    ]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_one_fold_width)(w, max_iter, train_idx, val_idx, X_train, y_train)
        for w, max_iter, train_idx, val_idx in tqdm(tasks, desc="NN width sweep", total=len(tasks))
    )
    # Aggregate by width
    by_width = defaultdict(lambda: {"train": [], "val": []})
    idx = 0
    for w in WIDTH_VALUES:
        for _ in folds_list:
            train_f1_macro, val_f1_macro = results[idx]
            by_width[w]["train"].append(train_f1_macro)
            by_width[w]["val"].append(val_f1_macro)
            idx += 1
    train_f1_macro_list = [float(np.mean(by_width[w]["train"])) for w in WIDTH_VALUES]
    cv_f1_macro_list = [float(np.mean(by_width[w]["val"])) for w in WIDTH_VALUES]

    best_width = WIDTH_VALUES[int(np.argmax(cv_f1_macro_list))]
    results = {
        "widths": list(WIDTH_VALUES),
        "train_f1_macro": train_f1_macro_list,
        "cv_f1_macro": cv_f1_macro_list,
        "best_width": best_width,
    }
    _plot_nn_step1(results)
    _write_step1_results(results)
    return results


def _plot_nn_step1(results):
    """Model Complexity Curve: X=width, Y=train + CV Macro-F1."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(results["widths"], results["train_f1_macro"], "o-", label="Train Macro-F1 (mean CV)")
    ax.plot(results["widths"], results["cv_f1_macro"], "s-", label="Cross-Val Macro-F1")
    ax.axvline(results["best_width"], color="gray", ls="--", alpha=0.7, label=f"best width={results['best_width']}")
    ax.set_xlabel("Hidden layer width")
    ax.set_ylabel("Macro-F1")
    ax.set_title("NN Step 1 — Model Complexity (width, 1 hidden layer)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "nn_width_model_complexity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step1_results(results):
    """Write Step 1 summary to NN_sklearn_results.txt."""
    lines = [
        "",
        "========== NN (sklearn) Step 1 — Width search ==========",
        f"Widths: {results['widths']}",
        f"Mean train Macro-F1: {[round(x, 4) for x in results['train_f1_macro']]}",
        f"Mean CV Macro-F1:    {[round(x, 4) for x in results['cv_f1_macro']]}",
        f"Best width: {results['best_width']}",
        "",
    ]
    _append_nn_results("\n".join(lines))
    print(f"Step 1 best width: {results['best_width']}")
    print(f"Appended to: {NN_SKLEARN_RESULTS_PATH}")


def run_nn_step2(X_train, y_train, X_test, y_test, nn_step1, cv=CV_SPLITS):
    """
    Step 2 — Depth vs width: [64], [32,32], [16,16,16,16].
    Stratified K-fold CV; plot Model Complexity (number of layers vs Macro-F1).
    """
    np.random.seed(RANDOM_SEED)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    n_samples = X_train.shape[0]
    max_iter = _get_max_iter(n_samples)

    n_layers_list = [len(arch) for arch in STEP2_ARCHITECTURES]
    folds_list = list(cv_splitter.split(X_train, y_train))
    tasks = [
        (arch, max_iter, train_idx, val_idx)
        for arch in STEP2_ARCHITECTURES
        for train_idx, val_idx in folds_list
    ]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_one_fold_arch)(arch, max_iter, train_idx, val_idx, X_train, y_train)
        for arch, max_iter, train_idx, val_idx in tqdm(tasks, desc="NN step2 depth sweep", total=len(tasks))
    )
    by_arch = defaultdict(lambda: {"train": [], "val": []})
    idx = 0
    for arch in STEP2_ARCHITECTURES:
        for _ in folds_list:
            train_f1_macro, val_f1_macro = results[idx]
            by_arch[arch]["train"].append(train_f1_macro)
            by_arch[arch]["val"].append(val_f1_macro)
            idx += 1
    train_f1_macro_list = [float(np.mean(by_arch[arch]["train"])) for arch in STEP2_ARCHITECTURES]
    cv_f1_macro_list = [float(np.mean(by_arch[arch]["val"])) for arch in STEP2_ARCHITECTURES]

    best_idx = int(np.argmax(cv_f1_macro_list))
    best_architecture = STEP2_ARCHITECTURES[best_idx]
    results = {
        "architectures": list(STEP2_ARCHITECTURES),
        "n_layers": n_layers_list,
        "train_f1_macro": train_f1_macro_list,
        "cv_f1_macro": cv_f1_macro_list,
        "best_architecture": best_architecture,
        "step1": nn_step1,
    }
    _plot_nn_step2(results)
    _write_step2_results(results)
    return results


def _plot_nn_step2(results):
    """Model Complexity: X=number of layers, Y=train + CV Macro-F1."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(results["n_layers"], results["train_f1_macro"], "o-", label="Train Macro-F1 (mean CV)")
    ax.plot(results["n_layers"], results["cv_f1_macro"], "s-", label="Cross-Val Macro-F1")
    best_n = len(results["best_architecture"])
    ax.axvline(best_n, color="gray", ls="--", alpha=0.7, label=f"best n_layers={best_n}")
    ax.set_xlabel("Number of hidden layers")
    ax.set_ylabel("Macro-F1")
    ax.set_title("NN Step 2 — Model Complexity (depth)")
    ax.set_xticks(results["n_layers"])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "nn_depth_model_complexity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step2_results(results):
    """Write Step 2 summary to NN_sklearn_results.txt."""
    arch_str = [list(a) for a in results["architectures"]]
    lines = [
        "========== NN (sklearn) Step 2 — Depth vs width ==========",
        f"Architectures: {arch_str}",
        f"Mean train Macro-F1: {[round(x, 4) for x in results['train_f1_macro']]}",
        f"Mean CV Macro-F1:    {[round(x, 4) for x in results['cv_f1_macro']]}",
        f"Best architecture: {list(results['best_architecture'])}",
        "",
    ]
    _append_nn_results("\n".join(lines))
    print(f"Step 2 best architecture: {list(results['best_architecture'])}")
    print(f"Appended to: {NN_SKLEARN_RESULTS_PATH}")


def run_nn_step3(X_train, y_train, X_test, y_test, nn_step2, cv=CV_SPLITS):
    """
    Step 3 — Learning rate sweep with best architecture from Step 2.
    Test LR: [0.1, 0.01, 0.001, 0.0003]. Plot LR vs Macro-F1 and epoch (training loss) curve.
    """
    np.random.seed(RANDOM_SEED)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    best_arch = nn_step2["best_architecture"]
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    n_samples = X_train.shape[0]
    max_iter = _get_max_iter(n_samples)

    folds_list = list(cv_splitter.split(X_train, y_train))
    tasks = [
        (best_arch, lr, max_iter, train_idx, val_idx)
        for lr in STEP3_LR_VALUES
        for train_idx, val_idx in folds_list
    ]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_one_fold_lr)(best_arch, lr, max_iter, train_idx, val_idx, X_train, y_train)
        for best_arch, lr, max_iter, train_idx, val_idx in tqdm(tasks, desc="NN step3 LR sweep", total=len(tasks))
    )
    by_lr = defaultdict(lambda: {"train": [], "val": []})
    idx = 0
    for lr in STEP3_LR_VALUES:
        for _ in folds_list:
            train_f1_macro, val_f1_macro = results[idx]
            by_lr[lr]["train"].append(train_f1_macro)
            by_lr[lr]["val"].append(val_f1_macro)
            idx += 1
    train_f1_macro_list = [float(np.mean(by_lr[lr]["train"])) for lr in STEP3_LR_VALUES]
    cv_f1_macro_list = [float(np.mean(by_lr[lr]["val"])) for lr in STEP3_LR_VALUES]

    best_idx = int(np.argmax(cv_f1_macro_list))
    best_lr = STEP3_LR_VALUES[best_idx]
    # Epoch curve: fit once with best_lr on full train to get loss_curve_
    clf_best = _make_mlp(best_arch, learning_rate_init=best_lr, max_iter=max_iter)
    clf_best.fit(X_train, y_train)
    loss_curve_for_plot = clf_best.loss_curve_

    results = {
        "lr_values": list(STEP3_LR_VALUES),
        "train_f1_macro": train_f1_macro_list,
        "cv_f1_macro": cv_f1_macro_list,
        "best_lr": best_lr,
        "best_architecture": best_arch,
        "loss_curve": loss_curve_for_plot,
        "step2": nn_step2,
    }
    _plot_nn_step3(results)
    _write_step3_results(results)
    _write_best_model_summary(results)
    return results


def _plot_nn_step3(results):
    """LR vs Macro-F1 (model complexity) and training loss vs iteration (epoch curve)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(results["lr_values"], results["train_f1_macro"], "o-", label="Train Macro-F1 (mean CV)")
    ax.plot(results["lr_values"], results["cv_f1_macro"], "s-", label="Cross-Val Macro-F1")
    ax.axvline(results["best_lr"], color="gray", ls="--", alpha=0.7, label=f"best LR={results['best_lr']}")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Macro-F1")
    ax.set_xscale("log")
    ax.set_title("NN Step 3 — Model Complexity (LR)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(results["loss_curve"], color="C0", label="Train loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("NN Step 3 — Epoch curve (train loss, best LR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "nn_lr_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step3_results(results):
    """Write Step 3 summary to NN_sklearn_results.txt."""
    lines = [
        "========== NN (sklearn) Step 3 — Learning rate sweep ==========",
        f"LR values: {results['lr_values']}",
        f"Mean train Macro-F1: {[round(x, 4) for x in results['train_f1_macro']]}",
        f"Mean CV Macro-F1:    {[round(x, 4) for x in results['cv_f1_macro']]}",
        f"Best LR: {results['best_lr']}",
        "",
    ]
    _append_nn_results("\n".join(lines))
    print(f"Step 3 best LR: {results['best_lr']}")
    print(f"Appended to: {NN_SKLEARN_RESULTS_PATH}")


def _write_best_model_summary(results):
    """Write best model summary to NN_sklearn_results.txt."""
    lines = [
        "========== NN (sklearn) Best model ==========",
        f"Best architecture (from Step 2): {list(results['best_architecture'])}",
        f"Best learning rate (from Step 3): {results['best_lr']}",
        f"Fixed: L2={NN_L2}, batch_size={BATCH_SIZE}, early_stopping_patience={EARLY_STOPPING_PATIENCE}",
        f"Note: MLPClassifier doesn't support class_weight parameter",
        "",
    ]
    _append_nn_results("\n".join(lines))
    print("Best model — architecture:", list(results["best_architecture"]), "| LR:", results["best_lr"])


# Step 4 — Learning curve sizes [10%, 25%, 50%, 75%, 100%]
LEARNING_CURVE_TRAIN_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]


def run_nn_step4(X_train, y_train, X_test, y_test, nn_step3, cv=CV_SPLITS):
    """
    Step 4 — Final model: retrain with best arch + best LR.
    Generate: learning curve, confusion matrix, runtime table.
    Save test set metrics and all outputs to NN_sklearn_results.txt.
    """
    np.random.seed(RANDOM_SEED)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).ravel()
    best_arch = nn_step3["best_architecture"]
    best_lr = nn_step3["best_lr"]
    n_samples = X_train.shape[0]
    max_iter = _get_max_iter(n_samples)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    # 1) Learning curve: train sizes [10%, 25%, 50%, 75%, 100%]
    clf_lc = _make_mlp(best_arch, learning_rate_init=best_lr, max_iter=max_iter)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        clf_lc,
        X_train,
        y_train,
        train_sizes=LEARNING_CURVE_TRAIN_SIZES,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    _plot_learning_curve(train_sizes_abs, train_scores_mean, val_scores_mean)

    # 2) Retrain final model on full train; measure fit time
    clf_final = _make_mlp(best_arch, learning_rate_init=best_lr, max_iter=max_iter)
    t0 = time.perf_counter()
    clf_final.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    # Predict on test; measure predict time
    t0 = time.perf_counter()
    y_pred = clf_final.predict(X_test)
    predict_time = time.perf_counter() - t0

    # Get probabilities
    y_proba = clf_final.predict_proba(X_test) if hasattr(clf_final, "predict_proba") else None

    # 3) Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm, y_test, y_pred)

    # 4) Test set metrics (multiclass)
    metrics = score_multiclass(y_test, y_pred, y_proba)

    # 5) Write everything to NN_sklearn_results.txt
    _write_step4_results(
        cm=cm,
        metrics=metrics,
        fit_time=fit_time,
        predict_time=predict_time,
        learning_curve_sizes=train_sizes_abs.tolist(),
        learning_curve_train=train_scores_mean.tolist(),
        learning_curve_val=val_scores_mean.tolist(),
    )
    print("Step 4 done. Test Macro-F1:", round(metrics['f1_macro'], 4), "| Fit time:", round(fit_time, 3), "s")
    return {
        "clf_final": clf_final,
        "test_metrics": metrics,
        "confusion_matrix": cm,
        "fit_time": fit_time,
        "predict_time": predict_time,
    }


def _plot_learning_curve(train_sizes_abs, train_scores_mean, val_scores_mean):
    """Plot learning curve: X=training size, Y=train + validation Macro-F1."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(train_sizes_abs, train_scores_mean, "o-", label="Train Macro-F1")
    ax.plot(train_sizes_abs, val_scores_mean, "s-", label="Validation Macro-F1")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Macro-F1")
    ax.set_title("NN Step 4 — Learning curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "nn_learning_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _plot_confusion_matrix(cm, y_test, y_pred):
    """Plot confusion matrix and save to outputs/."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Quality')
    ax.set_ylabel('True Quality')
    ax.set_title('NN Step 4 — Confusion Matrix (test set)')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "nn_confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step4_results(
    cm,
    metrics,
    fit_time,
    predict_time,
    learning_curve_sizes,
    learning_curve_train,
    learning_curve_val,
):
    """Append Step 4 results to NN_sklearn_results.txt."""
    unique_classes = sorted([int(k.split('_')[1]) for k in metrics.get("f1_per_class", {}).keys()])
    f1_per_class = metrics.get("f1_per_class", {})
    accuracy_per_class = metrics.get("accuracy_per_class", {})
    
    lines = [
        "",
        "========== NN (sklearn) Step 4 — Final model (test set) ==========",
        "",
        "Learning curve (train sizes 10%, 25%, 50%, 75%, 100%):",
        f"  Train sizes: {[int(x) for x in learning_curve_sizes]}",
        f"  Train Macro-F1 (mean CV): {[round(x, 4) for x in learning_curve_train]}",
        f"  Validation Macro-F1 (mean CV): {[round(x, 4) for x in learning_curve_val]}",
        "",
        "Test set metrics:",
        f"  Accuracy: {metrics.get('accuracy', 0):.4f}",
        f"  Macro-F1: {metrics.get('f1_macro', 0):.4f}",
        f"  Weighted-F1: {metrics.get('f1_weighted', 0):.4f}",
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
        "Confusion matrix (test set):",
        f"  {cm.tolist()}",
        "",
        "Runtime:",
        f"  Fit time: {fit_time:.4f} s",
        f"  Predict time: {predict_time:.4f} s",
        f"  Hardware: {_get_hardware_note()}",
        "",
    ])
    _append_nn_results("\n".join(lines))


# Wrapper functions for consistency
def run_nn_model_complexity(X_train, y_train, X_test, y_test, cv=5):
    """Wrapper: Run Step 1 (width search)."""
    return run_nn_step1(X_train, y_train, X_test, y_test, cv=cv)


def run_nn_learning_curves(X_train, y_train, X_test, y_test, best_config=None, cv=5):
    """Wrapper: Learning curves are in Step 4."""
    print("Note: Learning curves are generated in Step 4.")
    return None


def run_nn_test_eval(X_train, y_train, X_test, y_test, best_config=None):
    """Wrapper: Test evaluation is in Step 4."""
    print("Note: Test evaluation is generated in Step 4.")
    return None
