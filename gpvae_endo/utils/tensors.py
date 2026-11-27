"""Small tensor helpers."""
from __future__ import annotations

import numpy as np
import torch


def cpu_copy(obj):
    """A real, independent copy of a (nested) state dict on the CPU: tensors are copied (never shared with
    the live weights), numpy arrays copied, dicts rebuilt, everything else kept as is."""
    if isinstance(obj, dict):
        return {k: cpu_copy(v) for k, v in obj.items()}
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    return obj
