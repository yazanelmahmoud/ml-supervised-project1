"""
Neural Network — PyTorch MLP for Adult Income (Binary Classification).
SGD only (no momentum). Same 4-step pipeline as sklearn: width, depth, LR sweep, final model.
Uses class_weight balanced. Same plots as sklearn NN.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from joblib import Parallel, delayed
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import RANDOM_SEED, OUTPUT_DIR

# Results file
NN_PYTORCH_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN_pytorch_results.txt")
PLOT_SUFFIX = "_pytorch_class_weight"

# Same hyperparameters as sklearn
WIDTH_VALUES = [8, 16, 32, 64, 128, 200, 400, 700]
STEP2_ARCHITECTURES = [(64,), (32, 14), (28, 14, 8)]
STEP3_LR_VALUES = [0.1, 0.01, 0.001, 0.0003]
NN_L2 = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
CV_SPLITS = 5
N_JOBS = -1  # Use all CPU cores for parallel CV

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _get_max_epochs(n_samples):
    """~100 epochs with batch_size=64."""
    return max(10, 100 * ((n_samples + BATCH_SIZE - 1) // BATCH_SIZE) // n_samples)


class MLP(nn.Module):
    """MLP for binary classification. SGD, ReLU, L2 via optimizer weight_decay."""

    def __init__(self, input_size, hidden_sizes, output_size=2):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(prev, output_size)

    def forward(self, x):
        return self.out(self.net(x))


def _fit_one_fold(X_tr, y_tr, X_val, y_val, arch, lr, max_epochs, class_weight_tensor, device):
    """Train on one fold; return (train_f1, val_f1), loss_curve."""
    n_features = X_tr.shape[1]
    model = MLP(n_features, arch, output_size=2).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=NN_L2)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    ds = TensorDataset(
        torch.FloatTensor(X_tr).to(device),
        torch.LongTensor(y_tr).to(device)
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_f1 = -1
    patience_counter = 0
    loss_curve = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        loss_curve.append(epoch_loss / len(loader))

        model.eval()
        with torch.no_grad():
            logits_val = model(torch.FloatTensor(X_val).to(device))
            pred_val = logits_val.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, pred_val, zero_division=0)
            pred_tr = model(torch.FloatTensor(X_tr).to(device)).argmax(dim=1).cpu().numpy()
            train_f1 = f1_score(y_tr, pred_tr, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

    return train_f1, best_val_f1, loss_curve


def _fit_one_fold_width(w, max_epochs, train_idx, val_idx, X_train, y_train, class_weight):
    """Fit one (width, fold); return (train_f1, val_f1). Picklable for joblib."""
    # Set device per worker (CPU for parallelization)
    device = torch.device("cpu")
    # Recreate class_weight_tensor on worker device
    class_weight_tensor = torch.FloatTensor(class_weight).to(device)
    # Set seed per worker for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    X_tr = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    
    tr_f1, v_f1, _ = _fit_one_fold(
        X_tr, y_tr, X_val, y_val,
        (w,), 0.01, max_epochs, class_weight_tensor, device
    )
    return (tr_f1, v_f1)


def _fit_one_fold_arch(arch, max_epochs, train_idx, val_idx, X_train, y_train, class_weight):
    """Fit one (arch, fold); return (train_f1, val_f1). Picklable for joblib."""
    device = torch.device("cpu")
    class_weight_tensor = torch.FloatTensor(class_weight).to(device)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    X_tr = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    
    tr_f1, v_f1, _ = _fit_one_fold(
        X_tr, y_tr, X_val, y_val,
        arch, 0.01, max_epochs, class_weight_tensor, device
    )
    return (tr_f1, v_f1)


def _fit_one_fold_lr(arch, lr, max_epochs, train_idx, val_idx, X_train, y_train, class_weight):
    """Fit one (lr, fold); return (train_f1, val_f1). Picklable for joblib."""
    device = torch.device("cpu")
    class_weight_tensor = torch.FloatTensor(class_weight).to(device)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    X_tr = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    
    tr_f1, v_f1, _ = _fit_one_fold(
        X_tr, y_tr, X_val, y_val,
        arch, lr, max_epochs, class_weight_tensor, device
    )
    return (tr_f1, v_f1)


def _fit_one_fold_lc(arch, lr, max_epochs, train_idx, val_idx, X_train, y_train, class_weight):
    """Fit one fold for learning curve; return (train_f1, val_f1). Picklable for joblib."""
    device = torch.device("cpu")
    class_weight_tensor = torch.FloatTensor(class_weight).to(device)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    X_tr = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    
    tr_f1, v_f1, _ = _fit_one_fold(
        X_tr, y_tr, X_val, y_val,
        arch, lr, max_epochs, class_weight_tensor, device
    )
    return (tr_f1, v_f1)


def _append_results(text, path=NN_PYTORCH_RESULTS_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def run_nn_step1(X_train, y_train, X_test, y_test, cv=CV_SPLITS):
    """Step 1 — Width search (1 hidden layer)."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train).ravel().astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weight = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_tensor = torch.FloatTensor(class_weight).to(device)

    with open(NN_PYTORCH_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("NN (PyTorch) pipeline results [class_weight=balanced]\n")
        f.write("============================\n\n")

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    n_samples = X_train.shape[0]
    max_epochs = min(500, max(100, n_samples // BATCH_SIZE * 100))

    # Parallelize CV folds across widths
    folds_list = list(cv_splitter.split(X_train, y_train))
    tasks = [
        (w, max_epochs, train_idx, val_idx)
        for w in WIDTH_VALUES
        for train_idx, val_idx in folds_list
    ]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_one_fold_width)(w, max_epochs, train_idx, val_idx, X_train, y_train, class_weight)
        for w, max_epochs, train_idx, val_idx in tqdm(tasks, desc="NN Step1 width", total=len(tasks))
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
    results = {"widths": WIDTH_VALUES, "train_f1": train_f1_list, "cv_f1": cv_f1_list, "best_width": best_width}
    _plot_nn_step1(results)
    _write_step1_results(results)
    return results


def _plot_nn_step1(results):
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
    out_path = os.path.join(OUTPUT_DIR, f"nn_width_model_complexity{PLOT_SUFFIX}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step1_results(results):
    lines = [
        "", "========== NN (PyTorch) Step 1 — Width search ==========",
        f"Widths: {results['widths']}",
        f"Mean train F1: {[round(x, 4) for x in results['train_f1']]}",
        f"Mean CV F1:    {[round(x, 4) for x in results['cv_f1']]}",
        f"Best width: {results['best_width']}", "",
    ]
    _append_results("\n".join(lines))
    print(f"Step 1 best width: {results['best_width']}")


def run_nn_step2(X_train, y_train, X_test, y_test, nn_step1, cv=CV_SPLITS):
    """Step 2 — Depth vs width."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train).ravel().astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weight = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_tensor = torch.FloatTensor(class_weight).to(device)

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    n_samples = X_train.shape[0]
    max_epochs = min(500, max(100, n_samples // BATCH_SIZE * 100))

    # Parallelize CV folds across architectures
    folds_list = list(cv_splitter.split(X_train, y_train))
    tasks = [
        (arch, max_epochs, train_idx, val_idx)
        for arch in STEP2_ARCHITECTURES
        for train_idx, val_idx in folds_list
    ]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_one_fold_arch)(arch, max_epochs, train_idx, val_idx, X_train, y_train, class_weight)
        for arch, max_epochs, train_idx, val_idx in tqdm(tasks, desc="NN Step2 depth", total=len(tasks))
    )
    # Aggregate by architecture
    by_arch = defaultdict(lambda: {"train": [], "val": []})
    idx = 0
    for arch in STEP2_ARCHITECTURES:
        for _ in folds_list:
            train_f1, val_f1 = results[idx]
            by_arch[arch]["train"].append(train_f1)
            by_arch[arch]["val"].append(val_f1)
            idx += 1
    
    n_layers_list = [len(a) for a in STEP2_ARCHITECTURES]
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
    }
    _plot_nn_step2(results)
    _write_step2_results(results)
    return results


def _plot_nn_step2(results):
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
    out_path = os.path.join(OUTPUT_DIR, f"nn_depth_model_complexity{PLOT_SUFFIX}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step2_results(results):
    lines = [
        "========== NN (PyTorch) Step 2 — Depth vs width ==========",
        f"Architectures: {[list(a) for a in results['architectures']]}",
        f"Mean train F1: {[round(x, 4) for x in results['train_f1']]}",
        f"Mean CV F1:    {[round(x, 4) for x in results['cv_f1']]}",
        f"Best architecture: {list(results['best_architecture'])}", "",
    ]
    _append_results("\n".join(lines))


def run_nn_step3(X_train, y_train, X_test, y_test, nn_step2, cv=CV_SPLITS):
    """Step 3 — Learning rate sweep + epoch curve."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train).ravel().astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weight = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_tensor = torch.FloatTensor(class_weight).to(device)
    best_arch = nn_step2["best_architecture"]

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    n_samples = X_train.shape[0]
    max_epochs = min(500, max(100, n_samples // BATCH_SIZE * 100))

    # Parallelize CV folds across learning rates
    folds_list = list(cv_splitter.split(X_train, y_train))
    tasks = [
        (best_arch, lr, max_epochs, train_idx, val_idx)
        for lr in STEP3_LR_VALUES
        for train_idx, val_idx in folds_list
    ]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_one_fold_lr)(best_arch, lr, max_epochs, train_idx, val_idx, X_train, y_train, class_weight)
        for best_arch, lr, max_epochs, train_idx, val_idx in tqdm(tasks, desc="NN Step3 LR", total=len(tasks))
    )
    # Aggregate by learning rate
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

    # Epoch curve: fit on full train once
    _, _, loss_curve = _fit_one_fold(
        X_train, y_train, X_train, y_train,
        best_arch, best_lr, max_epochs, class_weight_tensor, device
    )

    results = {
        "lr_values": list(STEP3_LR_VALUES),
        "train_f1": train_f1_list,
        "cv_f1": cv_f1_list,
        "best_lr": best_lr,
        "best_architecture": best_arch,
        "loss_curve": loss_curve,
    }
    _plot_nn_step3(results)
    _write_step3_results(results)
    return results


def _plot_nn_step3(results):
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
    out_path = os.path.join(OUTPUT_DIR, f"nn_lr_curves{PLOT_SUFFIX}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step3_results(results):
    lines = [
        "========== NN (PyTorch) Step 3 — Learning rate sweep ==========",
        f"LR values: {results['lr_values']}",
        f"Mean train F1: {[round(x, 4) for x in results['train_f1']]}",
        f"Mean CV F1:    {[round(x, 4) for x in results['cv_f1']]}",
        f"Best LR: {results['best_lr']}",
        "========== NN (PyTorch) Best model ==========",
        f"Best architecture (from Step 2): {list(results['best_architecture'])}",
        f"Best learning rate (from Step 3): {results['best_lr']}",
        f"Fixed: L2={NN_L2}, batch_size={BATCH_SIZE}, early_stopping_patience={EARLY_STOPPING_PATIENCE}",
        "",
    ]
    _append_results("\n".join(lines))


LEARNING_CURVE_TRAIN_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]


def run_nn_step4(X_train, y_train, X_test, y_test, nn_step3, cv=CV_SPLITS):
    """Step 4 — Learning curve, final model, confusion matrix."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train).ravel().astype(np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test).ravel().astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weight = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_tensor = torch.FloatTensor(class_weight).to(device)

    best_arch = nn_step3["best_architecture"]
    best_lr = nn_step3["best_lr"]
    n_samples = X_train.shape[0]
    max_epochs = min(500, max(100, n_samples // BATCH_SIZE * 100))
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    # Learning curve
    train_sizes_abs, train_scores, val_scores = [], [], []
    for frac in LEARNING_CURVE_TRAIN_SIZES:
        n_use = max(1, int(n_samples * frac))
        idx = np.random.RandomState(RANDOM_SEED).permutation(n_samples)[:n_use]
        X_sub, y_sub = X_train[idx], y_train[idx]
        
        # Parallelize CV folds for this training size
        folds_list = list(cv_splitter.split(X_sub, y_sub))
        tasks = [
            (train_idx, val_idx)
            for train_idx, val_idx in folds_list
        ]
        results = Parallel(n_jobs=N_JOBS)(
            delayed(_fit_one_fold_lc)(best_arch, best_lr, max_epochs, train_idx, val_idx, X_sub, y_sub, class_weight)
            for train_idx, val_idx in tasks
        )
        tr_f1s = [r[0] for r in results]
        v_f1s = [r[1] for r in results]
        
        train_sizes_abs.append(n_use)
        train_scores.append(np.mean(tr_f1s))
        val_scores.append(np.mean(v_f1s))
    _plot_learning_curve(np.array(train_sizes_abs), np.array(train_scores), np.array(val_scores))

    # Final model - with early stopping
    n_features = X_train.shape[1]
    model = MLP(n_features, best_arch, output_size=2).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=best_lr, momentum=0.0, weight_decay=NN_L2)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    ds = TensorDataset(torch.FloatTensor(X_train).to(device), torch.LongTensor(y_train).to(device))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # Early stopping for final model (similar to sklearn's tol-based stopping)
    best_loss = float('inf')
    patience_counter = 0
    tol = 1e-4  # Same tolerance as sklearn default
    n_epochs_used = 0
    
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        n_epochs_used = epoch + 1
        
        # Early stopping: stop if loss doesn't improve by tol for patience epochs
        if avg_loss < best_loss - tol:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
    fit_time = time.perf_counter() - t0

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test).to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    predict_time = time.perf_counter() - t0

    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm, y_test, y_pred)

    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    test_pr_auc = float(average_precision_score(y_test, y_proba))

    _write_step4_results(
        cm, test_accuracy, test_f1, test_pr_auc, fit_time, predict_time,
        train_sizes_abs, train_scores, val_scores,
        n_epochs_used, max_epochs,
    )
    print("Step 4 done. Test F1:", round(test_f1, 4), "| Fit time:", round(fit_time, 3), "s")
    return {
        "clf_final": model,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_pr_auc": test_pr_auc,
        "confusion_matrix": cm,
        "fit_time": fit_time,
        "predict_time": predict_time,
    }


def _plot_learning_curve(train_sizes_abs, train_scores_mean, val_scores_mean):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(train_sizes_abs, train_scores_mean, "o-", label="Train F1")
    ax.plot(train_sizes_abs, val_scores_mean, "s-", label="Validation F1")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("F1")
    ax.set_title("NN Step 4 — Learning curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"nn_learning_curve{PLOT_SUFFIX}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _plot_confusion_matrix(cm, y_test, y_pred):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    labels = sorted(set(y_test) | set(y_pred))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("NN Step 4 — Confusion matrix (test set)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"nn_confusion_matrix{PLOT_SUFFIX}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def _write_step4_results(cm, test_accuracy, test_f1, test_pr_auc, fit_time, predict_time,
                         learning_curve_sizes, learning_curve_train, learning_curve_val,
                         n_epochs_used=None, max_epochs=None):
    lines = [
        "", "========== NN (PyTorch) Step 4 — Final model (test set) ==========",
        "",
        "Learning curve (train sizes 10%, 25%, 50%, 75%, 100%):",
        f"  Train sizes: {[int(x) for x in learning_curve_sizes]}",
        f"  Train F1 (mean CV): {[round(x, 4) for x in learning_curve_train]}",
        f"  Validation F1 (mean CV): {[round(x, 4) for x in learning_curve_val]}",
        "",
        "Test set metrics:",
        f"  Accuracy: {test_accuracy:.4f}",
        f"  F1: {test_f1:.4f}",
        f"  PR-AUC (average_precision): {test_pr_auc:.4f}",
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
    _append_results("\n".join(lines))
