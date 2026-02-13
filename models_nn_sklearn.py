"""
Neural Network — scikit-learn MLPClassifier.
SGD only (no momentum, Nesterov, Adam/Adagrad/RMSprop).
Report: learning rate, batch size, epochs/early stopping, regularization.
Required: epoch-based learning curves (train vs val loss/metric), early stopping.
Compare capacity scaling (depth vs width) with ~constant parameter count.
"""

import os
import time
from collections import defaultdict
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import RANDOM_SEED, OUTPUT_DIR

# Results file (sklearn pipeline)
NN_SKLEARN_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN_sklearn_results.txt")


def _nn_results_path(use_class_weight):
    """Path for results file when use_class_weight=True."""
    return NN_SKLEARN_RESULTS_PATH.replace(".txt", "_class_weight.txt") if use_class_weight else NN_SKLEARN_RESULTS_PATH


def _nn_plot_suffix(use_class_weight):
    """Filename suffix for plots when use_class_weight=True."""
    return "_class_weight" if use_class_weight else ""


# Step 1 — Width search (NN_strategy.txt)
WIDTH_VALUES = [8, 16, 32, 64, 128, 200, 400, 700]
# Step 2 — Depth: 1-layer 64, 2-layer 32→14, 3-layer 28→14→8
STEP2_ARCHITECTURES = [(64,), (32, 14), (28, 14, 8)]
# Step 3 — LR sweep
STEP3_LR_VALUES = [0.1, 0.01, 0.001, 0.0003]

NN_LR = 0.01
NN_L2 = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
CV_SPLITS = 5
N_JOBS = -1  # Use all CPU cores for parallel CV


def _fit_one_fold_width(w, max_iter, train_idx, val_idx, X_train, y_train, use_class_weight=False):
    """Fit one (width, fold); return (train_f1, val_f1). Picklable for joblib."""
    X_tr = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    clf = _make_mlp((w,), max_iter=max_iter)
    if use_class_weight:
        sw = compute_sample_weight("balanced", y_tr)
        clf.fit(X_tr, y_tr, sample_weight=sw)
    else:
        clf.fit(X_tr, y_tr)
    return (f1_score(y_tr, clf.predict(X_tr), zero_division=0), f1_score(y_val, clf.predict(X_val), zero_division=0))


def _fit_one_fold_arch(arch, max_iter, train_idx, val_idx, X_train, y_train, use_class_weight=False):
    """Fit one (arch, fold); return (train_f1, val_f1). Picklable for joblib."""
    X_tr = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    clf = _make_mlp(arch, max_iter=max_iter)
    if use_class_weight:
        sw = compute_sample_weight("balanced", y_tr)
        clf.fit(X_tr, y_tr, sample_weight=sw)
    else:
        clf.fit(X_tr, y_tr)
    return (f1_score(y_tr, clf.predict(X_tr), zero_division=0), f1_score(y_val, clf.predict(X_val), zero_division=0))


def _fit_one_fold_lr(arch, lr, max_iter, train_idx, val_idx, X_train, y_train, use_class_weight=False):
    """Fit one (lr, fold); return (train_f1, val_f1). Picklable for joblib."""
    X_tr = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    clf = _make_mlp(arch, learning_rate_init=lr, max_iter=max_iter)
    if use_class_weight:
        sw = compute_sample_weight("balanced", y_tr)
        clf.fit(X_tr, y_tr, sample_weight=sw)
    else:
        clf.fit(X_tr, y_tr)
    return (f1_score(y_tr, clf.predict(X_tr), zero_division=0), f1_score(y_val, clf.predict(X_val), zero_division=0))


def _append_nn_results(text, path=None):
    """Append a line or block to NN_sklearn_results.txt (or path when given)."""
    if path is None:
        path = NN_SKLEARN_RESULTS_PATH
    with open(path, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def _get_max_iter(n_samples):
    """Max iterations (epochs) for SGD solver. 
    Note: For SGD solver, max_iter counts epochs, not batch iterations.
    For consistency with PyTorch which uses min(500, max(100, ...)), 
    we use a similar maximum epoch limit."""
    # PyTorch uses: max_epochs = min(500, max(100, n_samples // BATCH_SIZE * 100))
    # But that formula seems to calculate batches*100, not epochs directly
    # For sklearn, max_iter should be epochs directly, so we use a reasonable limit
    # Use 500 epochs max to match PyTorch's cap, with minimum of 100
    return 500  # Match PyTorch's maximum epoch limit


def _iterations_per_epoch(n_samples):
    """Iterations per epoch: sklearn uses validation_fraction=0.1, so train on 90%."""
    n_train = max(1, int(0.9 * n_samples))
    return max(1, (n_train + BATCH_SIZE - 1) // BATCH_SIZE)


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


def run_nn_step1(X_train, y_train, X_test, y_test, cv=CV_SPLITS, use_class_weight=False):
    """
    Step 1 — Width search (1 hidden layer).
    use_class_weight: if True, use sample_weight='balanced' in fit (for class imbalance).
    """
    np.random.seed(RANDOM_SEED)
    results_path = _nn_results_path(use_class_weight)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("NN (sklearn) pipeline results" + (" [class_weight=balanced]" if use_class_weight else "") + "\n")
        f.write("============================\n\n")
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
        delayed(_fit_one_fold_width)(w, max_iter, train_idx, val_idx, X_train, y_train, use_class_weight)
        for w, max_iter, train_idx, val_idx in tqdm(tasks, desc="NN width sweep", total=len(tasks))
    )
    # Aggregate by width (same order as WIDTH_VALUES)
    by_width = defaultdict(lambda: {"train": [], "val": []})
    idx = 0
    for w in WIDTH_VALUES:
        for _ in folds_list:
            train_f1, val_f1 = results[idx]
            by_width[w]["train"].append(train_f1)
            by_width[w]["val"].append(val_f1)
            idx += 1
    train_f1_list = [float(np.mean(by_width[w]["train"])) for w in WIDTH_VALUES]
    cv_f1_list = [float(np.mean(by_width[w]["val"])) for w in WIDTH_VALUES]

    best_width = WIDTH_VALUES[int(np.argmax(cv_f1_list))]
    results = {
        "widths": list(WIDTH_VALUES),
        "train_f1": train_f1_list,
        "cv_f1": cv_f1_list,
        "best_width": best_width,
        "use_class_weight": use_class_weight,
    }
    _plot_nn_step1(results)
    _write_step1_results(results)
    return results


def _plot_nn_step1(results):
    """Model Complexity Curve: X=width, Y=train + CV F1 (mean over folds)."""
    suffix = _nn_plot_suffix(results.get("use_class_weight", False))
    title_suffix = " (class_weight=balanced)" if results.get("use_class_weight") else ""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(results["widths"], results["train_f1"], "o-", label="Train F1 (mean CV)")
    ax.plot(results["widths"], results["cv_f1"], "s-", label="Cross-Val F1")
    ax.axvline(results["best_width"], color="gray", ls="--", alpha=0.7, label=f"best width={results['best_width']}")
    ax.set_xlabel("Hidden layer width")
    ax.set_ylabel("F1")
    ax.set_title("NN Step 1 — Model Complexity (width, 1 hidden layer)" + title_suffix)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"nn_width_model_complexity{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step1_results(results):
    """Write Step 1 summary to NN_sklearn_results.txt."""
    path = _nn_results_path(results.get("use_class_weight", False))
    lines = [
        "",
        "========== NN (sklearn) Step 1 — Width search ==========",
        f"Widths: {results['widths']}",
        f"Mean train F1: {[round(x, 4) for x in results['train_f1']]}",
        f"Mean CV F1:    {[round(x, 4) for x in results['cv_f1']]}",
        f"Best width: {results['best_width']}",
        "",
    ]
    _append_nn_results("\n".join(lines), path=path)
    print(f"Step 1 best width: {results['best_width']}")
    print(f"Appended to: {path}")


def run_nn_step2(X_train, y_train, X_test, y_test, nn_step1, cv=CV_SPLITS):
    """
    Step 2 — Depth vs width: [256], [128,128], [64,64,64].
    use_class_weight is taken from nn_step1 when present.
    """
    use_class_weight = nn_step1.get("use_class_weight", False)
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
        delayed(_fit_one_fold_arch)(arch, max_iter, train_idx, val_idx, X_train, y_train, use_class_weight)
        for arch, max_iter, train_idx, val_idx in tqdm(tasks, desc="NN step2 depth sweep", total=len(tasks))
    )
    by_arch = defaultdict(lambda: {"train": [], "val": []})
    idx = 0
    for arch in STEP2_ARCHITECTURES:
        for _ in folds_list:
            train_f1, val_f1 = results[idx]
            by_arch[arch]["train"].append(train_f1)
            by_arch[arch]["val"].append(val_f1)
            idx += 1
    train_f1_list = [float(np.mean(by_arch[arch]["train"])) for arch in STEP2_ARCHITECTURES]
    cv_f1_list = [float(np.mean(by_arch[arch]["val"])) for arch in STEP2_ARCHITECTURES]

    best_idx = int(np.argmax(cv_f1_list))
    best_architecture = STEP2_ARCHITECTURES[best_idx]
    results = {
        "architectures": list(STEP2_ARCHITECTURES),
        "n_layers": n_layers_list,
        "train_f1": train_f1_list,
        "cv_f1": cv_f1_list,
        "best_architecture": best_architecture,
        "step1": nn_step1,
        "use_class_weight": use_class_weight,
    }
    _plot_nn_step2(results)
    _write_step2_results(results)
    return results


def _plot_nn_step2(results):
    """Model Complexity: X=number of layers, Y=train + CV F1."""
    suffix = _nn_plot_suffix(results.get("use_class_weight", False))
    title_suffix = " (class_weight=balanced)" if results.get("use_class_weight") else ""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(results["n_layers"], results["train_f1"], "o-", label="Train F1 (mean CV)")
    ax.plot(results["n_layers"], results["cv_f1"], "s-", label="Cross-Val F1")
    best_n = len(results["best_architecture"])
    ax.axvline(best_n, color="gray", ls="--", alpha=0.7, label=f"best n_layers={best_n}")
    ax.set_xlabel("Number of hidden layers")
    ax.set_ylabel("F1")
    ax.set_title("NN Step 2 — Model Complexity (depth)" + title_suffix)
    ax.set_xticks(results["n_layers"])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"nn_depth_model_complexity{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step2_results(results):
    """Write Step 2 summary to NN_sklearn_results.txt."""
    path = _nn_results_path(results.get("use_class_weight", False))
    arch_str = [list(a) for a in results["architectures"]]
    lines = [
        "========== NN (sklearn) Step 2 — Depth vs width ==========",
        f"Architectures: {arch_str}",
        f"Mean train F1: {[round(x, 4) for x in results['train_f1']]}",
        f"Mean CV F1:    {[round(x, 4) for x in results['cv_f1']]}",
        f"Best architecture: {list(results['best_architecture'])}",
        "",
    ]
    _append_nn_results("\n".join(lines), path=path)
    print(f"Step 2 best architecture: {list(results['best_architecture'])}")
    print(f"Appended to: {path}")


def run_nn_step3(X_train, y_train, X_test, y_test, nn_step2, cv=CV_SPLITS):
    """
    Step 3 — Learning rate sweep with best architecture from Step 2.
    use_class_weight is taken from nn_step2 when present.
    """
    use_class_weight = nn_step2.get("use_class_weight", False)
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
        delayed(_fit_one_fold_lr)(best_arch, lr, max_iter, train_idx, val_idx, X_train, y_train, use_class_weight)
        for best_arch, lr, max_iter, train_idx, val_idx in tqdm(tasks, desc="NN step3 LR sweep", total=len(tasks))
    )
    by_lr = defaultdict(lambda: {"train": [], "val": []})
    idx = 0
    for lr in STEP3_LR_VALUES:
        for _ in folds_list:
            train_f1, val_f1 = results[idx]
            by_lr[lr]["train"].append(train_f1)
            by_lr[lr]["val"].append(val_f1)
            idx += 1
    train_f1_list = [float(np.mean(by_lr[lr]["train"])) for lr in STEP3_LR_VALUES]
    cv_f1_list = [float(np.mean(by_lr[lr]["val"])) for lr in STEP3_LR_VALUES]

    best_idx = int(np.argmax(cv_f1_list))
    best_lr = STEP3_LR_VALUES[best_idx]
    # Epoch curve: fit once with best_lr on full train to get loss_curve_
    clf_best = _make_mlp(best_arch, learning_rate_init=best_lr, max_iter=max_iter)
    if use_class_weight:
        sw = compute_sample_weight("balanced", y_train)
        clf_best.fit(X_train, y_train, sample_weight=sw)
    else:
        clf_best.fit(X_train, y_train)
    loss_curve_for_plot = clf_best.loss_curve_

    results = {
        "lr_values": list(STEP3_LR_VALUES),
        "train_f1": train_f1_list,
        "cv_f1": cv_f1_list,
        "best_lr": best_lr,
        "best_architecture": best_arch,
        "loss_curve": loss_curve_for_plot,
        "n_samples": X_train.shape[0],
        "step2": nn_step2,
        "use_class_weight": use_class_weight,
    }
    _plot_nn_step3(results)
    _write_step3_results(results)
    _write_best_model_summary(results)
    return results


def _plot_nn_step3(results):
    """LR vs F1 (model complexity) and training loss vs epoch (epoch curve)."""
    suffix = _nn_plot_suffix(results.get("use_class_weight", False))
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
    # Note: sklearn's loss_curve_ contains one value per epoch (n_iter_ epochs)
    # For SGD solver, n_iter_ counts epochs, so loss_curve_ is already per epoch
    # Use indices directly as epochs (0, 1, 2, ...) instead of dividing by iters_per_epoch
    epochs_x = np.arange(len(results["loss_curve"]))
    ax.plot(epochs_x, results["loss_curve"], color="C0", label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("NN Step 3 — Epoch curve (train loss, best LR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"nn_lr_curves{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step3_results(results):
    """Write Step 3 summary to NN_sklearn_results.txt."""
    path = _nn_results_path(results.get("use_class_weight", False))
    lines = [
        "========== NN (sklearn) Step 3 — Learning rate sweep ==========",
        f"LR values: {results['lr_values']}",
        f"Mean train F1: {[round(x, 4) for x in results['train_f1']]}",
        f"Mean CV F1:    {[round(x, 4) for x in results['cv_f1']]}",
        f"Best LR: {results['best_lr']}",
        "",
    ]
    _append_nn_results("\n".join(lines), path=path)
    print(f"Step 3 best LR: {results['best_lr']}")
    print(f"Appended to: {path}")


def _write_best_model_summary(results):
    """Write best model summary to NN_sklearn_results.txt."""
    path = _nn_results_path(results.get("use_class_weight", False))
    lines = [
        "========== NN (sklearn) Best model ==========",
        f"Best architecture (from Step 2): {list(results['best_architecture'])}",
        f"Best learning rate (from Step 3): {results['best_lr']}",
        f"Fixed: L2={NN_L2}, batch_size={BATCH_SIZE}, early_stopping_patience={EARLY_STOPPING_PATIENCE}",
        "",
    ]
    _append_nn_results("\n".join(lines), path=path)
    print("Best model — architecture:", list(results["best_architecture"]), "| LR:", results["best_lr"])


# Step 4 — Learning curve sizes [10%, 25%, 50%, 75%, 100%]
LEARNING_CURVE_TRAIN_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]


def run_nn_step4(X_train, y_train, X_test, y_test, nn_step3, cv=CV_SPLITS):
    """
    Step 4 — Final model: retrain with best arch + best LR.
    use_class_weight is taken from nn_step3 when present.
    """
    use_class_weight = nn_step3.get("use_class_weight", False)
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
        scoring="f1",
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    _plot_learning_curve(train_sizes_abs, train_scores_mean, val_scores_mean, use_class_weight=use_class_weight)

    # 2) Retrain final model on full train; measure fit time
    clf_final = _make_mlp(best_arch, learning_rate_init=best_lr, max_iter=max_iter)
    t0 = time.perf_counter()
    if use_class_weight:
        sw = compute_sample_weight("balanced", y_train)
        clf_final.fit(X_train, y_train, sample_weight=sw)
    else:
        clf_final.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    # Predict on test; measure predict time
    t0 = time.perf_counter()
    y_pred = clf_final.predict(X_test)
    predict_time = time.perf_counter() - t0

    # Get probabilities for PR-AUC
    y_proba = clf_final.predict_proba(X_test)[:, 1] if hasattr(clf_final, "predict_proba") else None

    # 3) Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm, y_test, y_pred, use_class_weight=use_class_weight)

    # 4) Test set metrics
    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    test_pr_auc = float(average_precision_score(y_test, y_proba)) if y_proba is not None else None

    # 5) Write everything to NN_sklearn_results.txt
    # Note: sklearn's n_iter_ counts epochs (not batch iterations) for SGD solver
    # According to sklearn docs: "For stochastic solvers ('sgd', 'adam'), 
    # note that this determines the number of epochs (how many times each data point will be used)"
    n_iter_used = clf_final.n_iter_ if hasattr(clf_final, 'n_iter_') else None
    # n_iter_ is already in epochs for SGD solver, so use it directly
    n_epochs_used = n_iter_used if n_iter_used is not None else None
    max_epochs = max_iter  # max_iter is also in epochs for SGD solver
    _write_step4_results(
        cm=cm,
        test_accuracy=test_accuracy,
        test_f1=test_f1,
        test_pr_auc=test_pr_auc,
        fit_time=fit_time,
        predict_time=predict_time,
        learning_curve_sizes=train_sizes_abs.tolist(),
        learning_curve_train=train_scores_mean.tolist(),
        learning_curve_val=val_scores_mean.tolist(),
        use_class_weight=use_class_weight,
        n_epochs_used=n_epochs_used,
        max_epochs=max_epochs,
    )
    print("Step 4 done. Test F1:", round(test_f1, 4), "| Fit time:", round(fit_time, 3), "s")
    return {
        "clf_final": clf_final,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_pr_auc": test_pr_auc,
        "confusion_matrix": cm,
        "fit_time": fit_time,
        "predict_time": predict_time,
    }


def _plot_learning_curve(train_sizes_abs, train_scores_mean, val_scores_mean, use_class_weight=False):
    """Plot learning curve: X=training size, Y=train + validation F1."""
    suffix = _nn_plot_suffix(use_class_weight)
    title_suffix = " (class_weight=balanced)" if use_class_weight else ""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(train_sizes_abs, train_scores_mean, "o-", label="Train F1")
    ax.plot(train_sizes_abs, val_scores_mean, "s-", label="Validation F1")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("F1")
    ax.set_title("NN Step 4 — Learning curve" + title_suffix)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"nn_learning_curve{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _plot_confusion_matrix(cm, y_test, y_pred, use_class_weight=False):
    """Plot confusion matrix and save to outputs/."""
    suffix = _nn_plot_suffix(use_class_weight)
    title_suffix = " (class_weight=balanced)" if use_class_weight else ""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    labels = sorted(set(y_test) | set(y_pred))
    disp = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("NN Step 4 — Confusion matrix (test set)" + title_suffix)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"nn_confusion_matrix{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step4_results(
    cm,
    test_accuracy,
    test_f1,
    test_pr_auc,
    fit_time,
    predict_time,
    learning_curve_sizes,
    learning_curve_train,
    learning_curve_val,
    use_class_weight=False,
    n_epochs_used=None,
    max_epochs=None,
):
    """Append Step 4 results to NN_sklearn_results.txt."""
    path = _nn_results_path(use_class_weight)
    lines = [
        "",
        "========== NN (sklearn) Step 4 — Final model (test set) ==========",
        "",
        "Learning curve (train sizes 10%, 25%, 50%, 75%, 100%):",
        f"  Train sizes: {[int(x) for x in learning_curve_sizes]}",
        f"  Train F1 (mean CV): {[round(x, 4) for x in learning_curve_train]}",
        f"  Validation F1 (mean CV): {[round(x, 4) for x in learning_curve_val]}",
        "",
        "Test set metrics:",
        f"  Accuracy: {test_accuracy:.4f}",
        f"  F1: {test_f1:.4f}",
        f"  PR-AUC (average_precision): {(test_pr_auc if test_pr_auc is not None else float('nan')):.4f}",
        "",
        "Confusion matrix (test set):",
        f"  {cm.tolist()}",
        "",
        "Runtime:",
        f"  Fit time: {fit_time:.4f} s",
        f"  Predict time: {predict_time:.4f} s",
    ]
    if n_epochs_used is not None and max_epochs is not None:
        lines.extend([
            "",
            "Training details:",
            f"  Epochs used: {n_epochs_used} / {max_epochs}",
            f"  Early stopping: {'Yes' if n_epochs_used < max_epochs else 'No'}",
        ])
    lines.append("")
    _append_nn_results("\n".join(lines), path=path)
