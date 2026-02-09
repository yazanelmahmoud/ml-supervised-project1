"""
Shared utilities: set seeds, plotting style, small helpers.
"""

import numpy as np
import random

from config import RANDOM_SEED


def set_seed(seed=None):
    """Fix random seeds for reproducibility (NumPy, random, and if available torch)."""
    seed = seed or RANDOM_SEED
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
