"""src/utils/seed.py — reproducibility helper."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set every RNG this project touches. Called once per repeated run
    (see training/repeated_runs.py) with a DIFFERENT seed each time —
    that's what gives the N runs of "the same architecture" different
    initial weights, per the assignment's mean±SD requirement.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
