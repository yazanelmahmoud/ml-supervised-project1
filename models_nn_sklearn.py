"""
Neural Network — scikit-learn MLPClassifier.
SGD only (no momentum, Nesterov, Adam/Adagrad/RMSprop).
Report: learning rate, batch size, epochs/early stopping, regularization.
Required: epoch-based learning curves (train vs val loss/metric), early stopping.
Compare capacity scaling (depth vs width) with ~constant parameter count.
"""

import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import RANDOM_SEED, OUTPUT_DIR

# Results file (sklearn pipeline)
NN_SKLEARN_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN_sklearn_results.txt")

# Step 1 — Width search (NN_strategy.txt)
WIDTH_VALUES = [8, 16, 32, 64, 128, 256, 512]
# Step 2 — Depth vs width: [256], [128,128], [64,64,64]
STEP2_ARCHITECTURES = [(256,), (128, 128), (64, 64, 64)]
# Step 3 — LR sweep
STEP3_LR_VALUES = [0.1, 0.01, 0.001, 0.0003]

NN_LR = 0.01
NN_L2 = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
CV_SPLITS = 5


def _append_nn_results(text):
    """Append a line or block to NN_sklearn_results.txt."""
    with open(NN_SKLEARN_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def _get_max_iter(n_samples):
    """Max iterations for ~100 epochs with batch_size=64."""
    return max(1000, 100 * ((n_samples + BATCH_SIZE - 1) // BATCH_SIZE))


def _make_mlp(hidden_layer_sizes, learning_rate_init=NN_LR, alpha=NN_L2, max_iter=10000, random_state=RANDOM_SEED):
    """Build MLPClassifier with SGD, no momentum, early stopping."""
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
        momentum=0.0,
    )


def run_nn_step1(X_train, y_train, X_test, y_test, cv=CV_SPLITS):
    """
    Step 1 — Width search (1 hidden layer).
    Architecture: Input → Dense(W) → ReLU → Output.
    Uses stratified K-fold CV on training set (like SVM/kNN/DT).
    For each width: mean train F1 and mean CV F1; plot Model Complexity Curve.
    Returns dict with widths, train_f1, cv_f1, best_width.
    """
    np.random.seed(RANDOM_SEED)
    # Start fresh results file for this pipeline run
    with open(NN_SKLEARN_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("NN (sklearn) pipeline results\n")
        f.write("============================\n\n")
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    n_samples = X_train.shape[0]
    max_iter = _get_max_iter(n_samples)

    train_f1_list = []
    cv_f1_list = []

    for w in tqdm(WIDTH_VALUES, desc="NN width sweep"):
        fold_train_f1 = []
        fold_val_f1 = []
        for train_idx, val_idx in cv_splitter.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            clf = _make_mlp((w,), max_iter=max_iter)
            clf.fit(X_tr, y_tr)
            fold_train_f1.append(f1_score(y_tr, clf.predict(X_tr), zero_division=0))
            fold_val_f1.append(f1_score(y_val, clf.predict(X_val), zero_division=0))
        train_f1_list.append(float(np.mean(fold_train_f1)))
        cv_f1_list.append(float(np.mean(fold_val_f1)))

    best_width = WIDTH_VALUES[int(np.argmax(cv_f1_list))]
    results = {
        "widths": list(WIDTH_VALUES),
        "train_f1": train_f1_list,
        "cv_f1": cv_f1_list,
        "best_width": best_width,
    }
    _plot_nn_step1(results)
    _write_step1_results(results)
    return results


def _plot_nn_step1(results):
    """Model Complexity Curve: X=width, Y=train + CV F1 (mean over folds)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(results["widths"], results["train_f1"], "o-", label="Train F1 (mean CV)")
    ax.plot(results["widths"], results["cv_f1"], "s-", label="Cross-Val F1")
    ax.axvline(results["best_width"], color="gray", ls="--", alpha=0.7, label=f"best width={results['best_width']}")
    ax.set_xlabel("Hidden layer width")
    ax.set_ylabel("F1")
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
        f"Mean train F1: {[round(x, 4) for x in results['train_f1']]}",
        f"Mean CV F1:    {[round(x, 4) for x in results['cv_f1']]}",
        f"Best width: {results['best_width']}",
        "",
    ]
    _append_nn_results("\n".join(lines))
    print(f"Step 1 best width: {results['best_width']}")
    print(f"Appended to: {NN_SKLEARN_RESULTS_PATH}")


def run_nn_step2(X_train, y_train, X_test, y_test, nn_step1, cv=CV_SPLITS):
    """
    Step 2 — Depth vs width: [256], [128,128], [64,64,64].
    Stratified K-fold CV; plot Model Complexity (number of layers vs F1).
    Returns dict with best_architecture, step1, and curve data.
    """
    np.random.seed(RANDOM_SEED)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    n_samples = X_train.shape[0]
    max_iter = _get_max_iter(n_samples)

    n_layers_list = [len(arch) for arch in STEP2_ARCHITECTURES]
    train_f1_list = []
    cv_f1_list = []

    for arch in tqdm(STEP2_ARCHITECTURES, desc="NN step2 depth sweep"):
        fold_train_f1 = []
        fold_val_f1 = []
        for train_idx, val_idx in cv_splitter.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            clf = _make_mlp(arch, max_iter=max_iter)
            clf.fit(X_tr, y_tr)
            fold_train_f1.append(f1_score(y_tr, clf.predict(X_tr), zero_division=0))
            fold_val_f1.append(f1_score(y_val, clf.predict(X_val), zero_division=0))
        train_f1_list.append(float(np.mean(fold_train_f1)))
        cv_f1_list.append(float(np.mean(fold_val_f1)))

    best_idx = int(np.argmax(cv_f1_list))
    best_architecture = STEP2_ARCHITECTURES[best_idx]
    results = {
        "architectures": list(STEP2_ARCHITECTURES),
        "n_layers": n_layers_list,
        "train_f1": train_f1_list,
        "cv_f1": cv_f1_list,
        "best_architecture": best_architecture,
        "step1": nn_step1,
    }
    _plot_nn_step2(results)
    _write_step2_results(results)
    return results


def _plot_nn_step2(results):
    """Model Complexity: X=number of layers, Y=train + CV F1."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(results["n_layers"], results["train_f1"], "o-", label="Train F1 (mean CV)")
    ax.plot(results["n_layers"], results["cv_f1"], "s-", label="Cross-Val F1")
    best_n = len(results["best_architecture"])
    ax.axvline(best_n, color="gray", ls="--", alpha=0.7, label=f"best n_layers={best_n}")
    ax.set_xlabel("Number of hidden layers")
    ax.set_ylabel("F1")
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
        f"Mean train F1: {[round(x, 4) for x in results['train_f1']]}",
        f"Mean CV F1:    {[round(x, 4) for x in results['cv_f1']]}",
        f"Best architecture: {list(results['best_architecture'])}",
        "",
    ]
    _append_nn_results("\n".join(lines))
    print(f"Step 2 best architecture: {list(results['best_architecture'])}")
    print(f"Appended to: {NN_SKLEARN_RESULTS_PATH}")


def run_nn_step3(X_train, y_train, X_test, y_test, nn_step2, cv=CV_SPLITS):
    """
    Step 3 — Learning rate sweep with best architecture from Step 2.
    Test LR: [0.1, 0.01, 0.001, 0.0003]. Plot LR vs F1 and epoch (training loss) curve.
    Returns dict with best_lr, best_architecture, and curve data.
    """
    np.random.seed(RANDOM_SEED)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    best_arch = nn_step2["best_architecture"]
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    n_samples = X_train.shape[0]
    max_iter = _get_max_iter(n_samples)

    train_f1_list = []
    cv_f1_list = []

    for lr in tqdm(STEP3_LR_VALUES, desc="NN step3 LR sweep"):
        fold_train_f1 = []
        fold_val_f1 = []
        for train_idx, val_idx in cv_splitter.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            clf = _make_mlp(best_arch, learning_rate_init=lr, max_iter=max_iter)
            clf.fit(X_tr, y_tr)
            fold_train_f1.append(f1_score(y_tr, clf.predict(X_tr), zero_division=0))
            fold_val_f1.append(f1_score(y_val, clf.predict(X_val), zero_division=0))
        train_f1_list.append(float(np.mean(fold_train_f1)))
        cv_f1_list.append(float(np.mean(fold_val_f1)))

    best_idx = int(np.argmax(cv_f1_list))
    best_lr = STEP3_LR_VALUES[best_idx]
    # Epoch curve: fit once with best_lr on full train to get loss_curve_
    clf_best = _make_mlp(best_arch, learning_rate_init=best_lr, max_iter=max_iter)
    clf_best.fit(X_train, y_train)
    loss_curve_for_plot = clf_best.loss_curve_

    results = {
        "lr_values": list(STEP3_LR_VALUES),
        "train_f1": train_f1_list,
        "cv_f1": cv_f1_list,
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
    """LR vs F1 (model complexity) and training loss vs iteration (epoch curve)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(results["lr_values"], results["train_f1"], "o-", label="Train F1 (mean CV)")
    ax.plot(results["lr_values"], results["cv_f1"], "s-", label="Cross-Val F1")
    ax.axvline(results["best_lr"], color="gray", ls="--", alpha=0.7, label=f"best LR={results['best_lr']}")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("F1")
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
        f"Mean train F1: {[round(x, 4) for x in results['train_f1']]}",
        f"Mean CV F1:    {[round(x, 4) for x in results['cv_f1']]}",
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
        "",
    ]
    _append_nn_results("\n".join(lines))
    print("Best model — architecture:", list(results["best_architecture"]), "| LR:", results["best_lr"])
