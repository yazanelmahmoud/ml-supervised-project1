"""
Neural Networks (PyTorch) for Wine Quality.
TODO: Train compact MLP with SGD only (no momentum, no adaptive optimizers).
Compare capacity scaling (depth vs width) while keeping SGD constant.
Hold total parameter count approximately constant when comparing architectures.
Required: epoch-based learning curves, early stopping, regularization.
"""

import os
import platform
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from config import RANDOM_SEED, TASK_TYPE
from evaluation import score_multiclass
from utils import set_seed

NN_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN_pytorch_results.txt")

# TODO: Define architecture and hyperparameter ranges
# Keep total parameter count approximately constant
HIDDEN_LAYER_SIZES_OPTIONS = [
    (50,),      # Shallow-wide
    (100,),     # Shallow-wide
    (50, 50),   # Deeper
    (25, 25, 25),  # Deeper-narrow
]
ALPHA_VALUES = [0.0001, 0.001, 0.01, 0.1]  # L2 regularization
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 20


class MLP(nn.Module):
    """Multi-layer perceptron for Wine Quality."""
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def run_nn_model_complexity(X_train, y_train, X_test, y_test, cv=5):
    """
    TODO: NN model-complexity: depth vs width, regularization.
    Plot validation metric vs architecture/regularization.
    Save to NN_pytorch_results.txt.
    """
    # TODO: Implement
    pass


def run_nn_learning_curves(X_train, y_train, X_test, y_test, best_config=None, cv=5):
    """
    TODO: Learning curves — train/val metric vs training size and epochs.
    Plot epoch-based curves showing early stopping.
    Append to NN_pytorch_results.txt.
    """
    # TODO: Implement
    pass


def run_nn_test_eval(X_train, y_train, X_test, y_test, best_config=None):
    """
    TODO: Refit best NN on full train; evaluate once on test.
    Report metrics, runtime, confusion matrix. Append to NN_pytorch_results.txt.
    """
    # TODO: Implement
    pass
