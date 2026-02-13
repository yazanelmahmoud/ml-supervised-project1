"""
Baseline models for Wine Quality (multiclass classification).
Minimal config: DummyClassifier, DecisionTree, SVM (linear + RBF), NN-sklearn, NN-PyTorch (1 layer, 10 neurons).
Writes hyperparameters and metrics to baseline_results.txt.
"""

import os
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from config import RANDOM_SEED
from utils import set_seed

CV_FOLDS = 5


def _nn_pytorch_multiclass(X_train, y_train, X_test, y_test, n_classes, seed=42):
    """One hidden layer, 10 neurons; multiclass."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    set_seed(seed)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.values if hasattr(y_train, "values") else y_train)
    device = torch.device("cpu")
    n_features = X_train.shape[1]
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train_enc, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(n_features, 10)
            self.fc2 = nn.Linear(10, n_classes)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

    model.train()
    for _ in range(100):
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_te)
    pred_enc = logits.argmax(dim=1).numpy()
    pred = le.inverse_transform(pred_enc)
    return pred


def _nn_pytorch_multiclass_cv(X_train, y_train, n_classes, n_splits=5, seed=42):
    """Run stratified K-fold CV on train; return mean CV accuracy and Macro-F1."""
    from sklearn.model_selection import StratifiedKFold as SKF
    cv = SKF(n_splits=n_splits, shuffle=True, random_state=seed)
    y_tr = y_train.values if hasattr(y_train, "values") else y_train
    accs, f1s = [], []
    for train_idx, val_idx in cv.split(X_train, y_tr):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_tr[train_idx], y_tr[val_idx]
        # need to pass y as pandas-like for inverse_transform; use raw arrays and same le
        le = LabelEncoder()
        y_t_enc = le.fit_transform(y_t)
        pred_enc = _nn_pytorch_multiclass_impl(X_t, y_t_enc, X_v, n_classes, seed=seed)
        pred = le.inverse_transform(pred_enc)
        accs.append(accuracy_score(y_v, pred))
        f1s.append(f1_score(y_v, pred, average="macro", zero_division=0))
    return np.mean(accs), np.mean(f1s)


def _nn_pytorch_multiclass_impl(X_train, y_train_enc, X_test, n_classes, seed=42):
    """Train on (X_train, y_train_enc), return predicted class indices for X_test."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    set_seed(seed)
    device = torch.device("cpu")
    n_features = X_train.shape[1]
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train_enc, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(n_features, 10)
            self.fc2 = nn.Linear(10, n_classes)
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
    model.train()
    for _ in range(100):
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(X_te)
    return logits.argmax(dim=1).numpy()


def run_baselines(output_path=None):
    from preprocessing import get_dataset

    set_seed()
    output_path = output_path or os.path.join(os.path.dirname(__file__), "baseline_results.txt")
    X_train, y_train, X_test, y_test = get_dataset()
    y_test_np = y_test.values if hasattr(y_test, "values") else y_test

    lines = []
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    lines.append("=" * 60)
    lines.append("BASELINE MODELS — Wine Quality (multiclass classification)")
    lines.append("CV = %d-fold stratified on train; Test = held-out test set." % CV_FOLDS)
    lines.append("=" * 60)

    # DummyClassifier (stratified)
    m = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    res = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc_t = accuracy_score(y_test_np, pred)
    f1_t = f1_score(y_test_np, pred, average="macro", zero_division=0)
    lines.append("\n--- DummyClassifier ---")
    lines.append("Hyperparameters: strategy='stratified', random_state=%s" % RANDOM_SEED)
    lines.append("CV (train): Accuracy=%.4f, Macro-F1=%.4f" % (res["test_accuracy"].mean(), res["test_f1_macro"].mean()))
    lines.append("Test:       Accuracy=%.4f, Macro-F1=%.4f" % (acc_t, f1_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    # DecisionTree
    m = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=10)
    res = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc_t = accuracy_score(y_test_np, pred)
    f1_t = f1_score(y_test_np, pred, average="macro", zero_division=0)
    lines.append("\n--- DecisionTree ---")
    lines.append("Hyperparameters: max_depth=10, random_state=%s" % RANDOM_SEED)
    lines.append("CV (train): Accuracy=%.4f, Macro-F1=%.4f" % (res["test_accuracy"].mean(), res["test_f1_macro"].mean()))
    lines.append("Test:       Accuracy=%.4f, Macro-F1=%.4f" % (acc_t, f1_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    # SVM linear
    m = LinearSVC(max_iter=2000, random_state=RANDOM_SEED)
    res = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc_t = accuracy_score(y_test_np, pred)
    f1_t = f1_score(y_test_np, pred, average="macro", zero_division=0)
    lines.append("\n--- SVM (linear) ---")
    lines.append("Hyperparameters: LinearSVC, max_iter=2000, random_state=%s" % RANDOM_SEED)
    lines.append("CV (train): Accuracy=%.4f, Macro-F1=%.4f" % (res["test_accuracy"].mean(), res["test_f1_macro"].mean()))
    lines.append("Test:       Accuracy=%.4f, Macro-F1=%.4f" % (acc_t, f1_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    # SVM RBF
    m = SVC(kernel="rbf", max_iter=2000, random_state=RANDOM_SEED)
    res = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc_t = accuracy_score(y_test_np, pred)
    f1_t = f1_score(y_test_np, pred, average="macro", zero_division=0)
    lines.append("\n--- SVM (RBF) ---")
    lines.append("Hyperparameters: kernel='rbf', max_iter=2000, random_state=%s" % RANDOM_SEED)
    lines.append("CV (train): Accuracy=%.4f, Macro-F1=%.4f" % (res["test_accuracy"].mean(), res["test_f1_macro"].mean()))
    lines.append("Test:       Accuracy=%.4f, Macro-F1=%.4f" % (acc_t, f1_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    # NN sklearn (1 layer, 10 neurons) — SGD only
    m = MLPClassifier(hidden_layer_sizes=(10,), solver="sgd", momentum=0, max_iter=500, random_state=RANDOM_SEED)
    res = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc_t = accuracy_score(y_test_np, pred)
    f1_t = f1_score(y_test_np, pred, average="macro", zero_division=0)
    lines.append("\n--- NN (sklearn) ---")
    lines.append("Hyperparameters: hidden_layer_sizes=(10,), solver='sgd', momentum=0, max_iter=500, random_state=%s" % RANDOM_SEED)
    lines.append("CV (train): Accuracy=%.4f, Macro-F1=%.4f" % (res["test_accuracy"].mean(), res["test_f1_macro"].mean()))
    lines.append("Test:       Accuracy=%.4f, Macro-F1=%.4f" % (acc_t, f1_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    # NN PyTorch — CV done manually
    n_classes = len(np.unique(y_train))
    cv_acc, cv_f1 = _nn_pytorch_multiclass_cv(X_train, y_train, n_classes, n_splits=CV_FOLDS, seed=RANDOM_SEED)
    pred = _nn_pytorch_multiclass(X_train, y_train, X_test, y_test, n_classes, seed=RANDOM_SEED)
    acc_t = accuracy_score(y_test_np, pred)
    f1_t = f1_score(y_test_np, pred, average="macro", zero_division=0)
    lines.append("\n--- NN (PyTorch) ---")
    lines.append("Hyperparameters: 1 hidden layer, 10 neurons, SGD lr=0.01 momentum=0, 100 epochs, batch_size=64")
    lines.append("CV (train): Accuracy=%.4f, Macro-F1=%.4f" % (cv_acc, cv_f1))
    lines.append("Test:       Accuracy=%.4f, Macro-F1=%.4f" % (acc_t, f1_t))
    print(lines[-4] + "\n" + lines[-3] + "\n" + lines[-2] + "\n" + lines[-1])

    lines.append("\n" + "=" * 60)
    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print("\nResults written to:", output_path)
    return text
