"""
Baseline models for Adult Income (binary classification).
Minimal config: DummyClassifier, DecisionTree, SVM (linear + RBF), NN-sklearn, NN-PyTorch (1 layer, 10 neurons).
Writes hyperparameters and metrics to baseline_results.txt.
"""

import os
import sys
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

from config import RANDOM_SEED
from utils import set_seed

CV_FOLDS = 5


def _nn_pytorch_binary(X_train, y_train, X_test, y_test, seed=42):
    """One hidden layer, 10 neurons; binary classification."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    set_seed(seed)
    device = torch.device("cpu")
    n_features = X_train.shape[1]
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train.values if hasattr(y_train, "values") else y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(n_features, 10)
            self.fc2 = nn.Linear(10, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = Net().to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)
    loader = DataLoader(TensorDataset(X_t, y_t.float().unsqueeze(1)), batch_size=64, shuffle=True)

    model.train()
    for _ in range(100):
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_te).squeeze()
    pred = (torch.sigmoid(logits).numpy() >= 0.5).astype(np.int32)
    return pred


def _nn_pytorch_binary_cv(X_train, y_train, n_splits=5, seed=42):
    """Run stratified K-fold CV on train; return mean CV accuracy and F1."""
    from sklearn.model_selection import StratifiedKFold as SKF
    cv = SKF(n_splits=n_splits, shuffle=True, random_state=seed)
    y_tr = y_train.values if hasattr(y_train, "values") else y_train
    accs, f1s = [], []
    for train_idx, val_idx in cv.split(X_train, y_tr):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_tr[train_idx], y_tr[val_idx]
        pred = _nn_pytorch_binary(X_t, y_t, X_v, y_v, seed=seed)
        accs.append(accuracy_score(y_v, pred))
        f1s.append(f1_score(y_v, pred, zero_division=0))
    return np.mean(accs), np.mean(f1s)


def run_baselines(output_path=None):
    from preprocessing import get_dataset

    set_seed()
    output_path = output_path or os.path.join(os.path.dirname(__file__), "baseline_results.txt")
    X_train, y_train, X_test, y_test = get_dataset()
    y_test_np = y_test.values if hasattr(y_test, "values") else y_test

    lines = []
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    lines.append("=" * 60)
    lines.append("BASELINE MODELS — Adult Income (binary classification)")
    lines.append("CV = %d-fold stratified on train; Test = held-out test set." % CV_FOLDS)
    lines.append("=" * 60)

    # DummyClassifier (stratified)
    m = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    scoring = {"accuracy": "accuracy", "f1": "f1", "average_precision": "average_precision"}
    res = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc_t, f1_t = accuracy_score(y_test_np, pred), f1_score(y_test_np, pred, zero_division=0)
    pr_auc_t = average_precision_score(y_test_np, m.predict_proba(X_test)[:, 1])
    lines.append("\n--- DummyClassifier ---")
    lines.append("Hyperparameters: strategy='stratified', random_state=%s" % RANDOM_SEED)
    lines.append("CV (train): Accuracy=%.4f, F1=%.4f, PR-AUC=%.4f" % (res["test_accuracy"].mean(), res["test_f1"].mean(), res["test_average_precision"].mean()))
    lines.append("Test:       Accuracy=%.4f, F1=%.4f, PR-AUC=%.4f" % (acc_t, f1_t, pr_auc_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    # DecisionTree (default/minimal)
    m = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=10)
    res = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc_t, f1_t = accuracy_score(y_test_np, pred), f1_score(y_test_np, pred, zero_division=0)
    pr_auc_t = average_precision_score(y_test_np, m.predict_proba(X_test)[:, 1])
    lines.append("\n--- DecisionTree ---")
    lines.append("Hyperparameters: max_depth=10, random_state=%s" % RANDOM_SEED)
    lines.append("CV (train): Accuracy=%.4f, F1=%.4f, PR-AUC=%.4f" % (res["test_accuracy"].mean(), res["test_f1"].mean(), res["test_average_precision"].mean()))
    lines.append("Test:       Accuracy=%.4f, F1=%.4f, PR-AUC=%.4f" % (acc_t, f1_t, pr_auc_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    # SVM linear (LinearSVC — no predict_proba, so no PR-AUC in CV)
    m = LinearSVC(max_iter=2000, random_state=RANDOM_SEED)
    scoring_linear = {"accuracy": "accuracy", "f1": "f1"}
    res = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring_linear, n_jobs=1)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc_t, f1_t = accuracy_score(y_test_np, pred), f1_score(y_test_np, pred, zero_division=0)
    lines.append("\n--- SVM (linear) ---")
    lines.append("Hyperparameters: LinearSVC, max_iter=2000, random_state=%s" % RANDOM_SEED)
    lines.append("CV (train): Accuracy=%.4f, F1=%.4f, PR-AUC=N/A" % (res["test_accuracy"].mean(), res["test_f1"].mean()))
    lines.append("Test:       Accuracy=%.4f, F1=%.4f, PR-AUC=N/A" % (acc_t, f1_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    # SVM RBF (subsample train for speed; cap max_iter)
    SVM_MAX_TRAIN = 5000
    if len(y_train) > SVM_MAX_TRAIN:
        X_s, _, y_s, _ = train_test_split(X_train, y_train, train_size=SVM_MAX_TRAIN, stratify=y_train, random_state=RANDOM_SEED)
    else:
        X_s, y_s = X_train, y_train
    m = SVC(kernel="rbf", max_iter=2000, random_state=RANDOM_SEED, probability=True)
    res = cross_validate(m, X_s, y_s, cv=cv, scoring=scoring, n_jobs=1)
    m.fit(X_s, y_s)
    pred = m.predict(X_test)
    acc_t, f1_t = accuracy_score(y_test_np, pred), f1_score(y_test_np, pred, zero_division=0)
    pr_auc_t = average_precision_score(y_test_np, m.predict_proba(X_test)[:, 1])
    lines.append("\n--- SVM (RBF) ---")
    lines.append("Hyperparameters: kernel='rbf', max_iter=2000, train_subsample=%s, random_state=%s" % (len(y_s), RANDOM_SEED))
    lines.append("CV (train): Accuracy=%.4f, F1=%.4f, PR-AUC=%.4f" % (res["test_accuracy"].mean(), res["test_f1"].mean(), res["test_average_precision"].mean()))
    lines.append("Test:       Accuracy=%.4f, F1=%.4f, PR-AUC=%.4f" % (acc_t, f1_t, pr_auc_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    # NN sklearn (1 layer, 10 neurons) — SGD only
    m = MLPClassifier(hidden_layer_sizes=(10,), solver="sgd", momentum=0, max_iter=500, random_state=RANDOM_SEED)
    res = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc_t, f1_t = accuracy_score(y_test_np, pred), f1_score(y_test_np, pred, zero_division=0)
    pr_auc_t = average_precision_score(y_test_np, m.predict_proba(X_test)[:, 1])
    lines.append("\n--- NN (sklearn) ---")
    lines.append("Hyperparameters: hidden_layer_sizes=(10,), solver='sgd', momentum=0, max_iter=500, random_state=%s" % RANDOM_SEED)
    lines.append("CV (train): Accuracy=%.4f, F1=%.4f, PR-AUC=%.4f" % (res["test_accuracy"].mean(), res["test_f1"].mean(), res["test_average_precision"].mean()))
    lines.append("Test:       Accuracy=%.4f, F1=%.4f, PR-AUC=%.4f" % (acc_t, f1_t, pr_auc_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    # NN PyTorch (1 layer, 10 neurons) — SGD only; CV done manually
    cv_acc, cv_f1 = _nn_pytorch_binary_cv(X_train, y_train, n_splits=CV_FOLDS, seed=RANDOM_SEED)
    pred = _nn_pytorch_binary(X_train, y_train, X_test, y_test, seed=RANDOM_SEED)
    acc_t = accuracy_score(y_test_np, pred)
    f1_t = f1_score(y_test_np, pred, zero_division=0)
    lines.append("\n--- NN (PyTorch) ---")
    lines.append("Hyperparameters: 1 hidden layer, 10 neurons, SGD lr=0.01 momentum=0, 100 epochs, batch_size=64")
    lines.append("CV (train): Accuracy=%.4f, F1=%.4f, PR-AUC=N/A" % (cv_acc, cv_f1))
    lines.append("Test:       Accuracy=%.4f, F1=%.4f, PR-AUC=N/A" % (acc_t, f1_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    lines.append("\n" + "=" * 60)
    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print("\nResults written to:", output_path)
    return text
