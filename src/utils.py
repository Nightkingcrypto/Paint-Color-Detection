import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import LOGS_DIR, LABEL_COUNTS_PATH


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_logs_dir():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def save_label_counts(label_counts):
    LABEL_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_COUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump(label_counts, f, indent=2)
