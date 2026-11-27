"""Choosing the device (GPU or CPU) for training.

The original code silently fell back to the CPU when no CUDA GPU was found. That can
turn a few hours of GPU work into days of CPU work without anyone noticing. Now a missing GPU stops
the run with a clear message, unless the environment variable ``GPVAE_ALLOW_CPU=1`` says the CPU is
wanted (for example for a small test on a laptop). With a GPU nothing changes.
"""
from __future__ import annotations

import os

import torch

ALLOW_CPU_ENV = "GPVAE_ALLOW_CPU"


class NoGPUError(RuntimeError):
    """No CUDA GPU is available and the CPU was not allowed explicitly."""


def cpu_allowed() -> bool:
    return os.environ.get(ALLOW_CPU_ENV, "").strip().lower() in ("1", "true", "yes")


_warned = False


def require_gpu_or_allowed_cpu() -> None:
    """Raise :class:`NoGPUError` when there is no CUDA GPU and ``GPVAE_ALLOW_CPU`` is not set."""
    global _warned
    if torch.cuda.is_available():
        return
    if not cpu_allowed():
        raise NoGPUError(
            "No CUDA GPU is available (torch.cuda.is_available() is False). Training on the CPU is very "
            f"slow, so it is not done silently. Set {ALLOW_CPU_ENV}=1 to run on the CPU anyway, or check the "
            "GPU driver / the job's GPU request."
        )
    if not _warned:
        print(f"WARNING: no CUDA GPU found; running on the CPU because {ALLOW_CPU_ENV}=1 (very slow).")
        _warned = True


def resolve_device(device: str) -> torch.device:
    """The models' ``device`` argument: ``"gpu"`` means CUDA; anything else means the CPU (as originally)."""
    if device != "gpu":
        return torch.device("cpu")
    require_gpu_or_allowed_cpu()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
