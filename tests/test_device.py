"""No silent CPU fallback: without a GPU the run stops unless GPVAE_ALLOW_CPU=1."""
from __future__ import annotations

import pytest
import torch

from gpvae_endo import cli
from gpvae_endo.utils import device as dev


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv(dev.ALLOW_CPU_ENV, raising=False)


def test_gpu_request_without_gpu_stops(no_gpu):
    with pytest.raises(dev.NoGPUError, match="GPVAE_ALLOW_CPU"):
        dev.resolve_device("gpu")


def test_cpu_can_be_allowed(no_gpu, monkeypatch):
    monkeypatch.setenv(dev.ALLOW_CPU_ENV, "1")
    assert dev.resolve_device("gpu") == torch.device("cpu")


def test_explicit_cpu_is_always_fine(no_gpu):
    assert dev.resolve_device("cpu") == torch.device("cpu")


def test_cli_reports_missing_gpu(no_gpu, monkeypatch, capsys):
    import gpvae_endo.pipeline as pipeline

    def fake_run(config_path, options):
        dev.require_gpu_or_allowed_cpu()

    monkeypatch.setattr(pipeline, "run", fake_run)
    assert cli.main(["run", "cfg.py"]) == 1
    assert "GPVAE_ALLOW_CPU" in capsys.readouterr().out
