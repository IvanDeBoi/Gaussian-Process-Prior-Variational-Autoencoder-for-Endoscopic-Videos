"""Random seeds, and capturing/restoring every random-number-generator state (for checkpoint resume).

Nothing is seeded at import time.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed numpy's global RNG and torch (CPU and all CUDA devices), exactly as the original driver did.

    Python's `random` module is not seeded: the original code never seeded or used it, and leaving it alone
    keeps runs identical to the original.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_rng_state() -> dict:
    """Capture all RNG states (python, numpy, torch CPU, torch CUDA). Storable with ``torch.save``."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict) -> None:
    """Restore states captured by :func:`get_rng_state`.

    CUDA states are restored when they were captured and CUDA is available. A CUDA state captured with a
    different number of devices raises instead of silently continuing with a different random stream.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        if len(cuda) != torch.cuda.device_count():
            raise RuntimeError(
                f"RNG state holds {len(cuda)} CUDA device states but {torch.cuda.device_count()} devices are "
                "visible; cannot restore the CUDA random streams."
            )
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda])
