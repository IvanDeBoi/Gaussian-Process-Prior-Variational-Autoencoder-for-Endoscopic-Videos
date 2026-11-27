"""Record every epoch's weights during a tiny training run, to check which epoch the model ends with.

The recorder hooks ``ProgressTracker.check_stop_criterion``, which both training loops call once per epoch,
right after the best-epoch snapshot and before the learning-rate step: the weights at that moment are the
ones a snapshot of that epoch holds. Recording does not touch the random generators or the model.
"""
from __future__ import annotations

import torch

from gpvae_endo.training.loop import kl_beta
from gpvae_endo.training.progress import ProgressTracker
from gpvae_endo.utils.tensors import cpu_copy

# Fingerprint parts that depend only on the training run itself, not on which epoch's weights the model
# keeps at the end. Everything else (weights, latents, encode/decode outputs, GP values) does.
TRAJECTORY_KEYS = ("losses", "epoch_lines", "lr_trace", "lrs_final", "best_state_restores", "saved_keys",
                   "gp_init", "rng_after")


def weights(model) -> dict:
    out = {"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict()}
    if getattr(model, "gp_prior", None) is not None:
        out["gp_prior"] = model.gp_prior.state_dict()
    if hasattr(model, "_quantizer_state"):
        out["quantizer"] = model._quantizer_state()
        out["lat_proj"] = model.lat_proj.state_dict()
        out["lat_unproj"] = model.lat_unproj.state_dict()
    return cpu_copy(out)


def same_weights(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        if isinstance(a[k], dict):
            if not same_weights(a[k], b[k]):
                return False
        elif not torch.equal(a[k], b[k]):
            return False
    return True


class EpochRecorder:
    def __init__(self, monkeypatch):
        self.model = None
        self.epoch_weights = []
        original = ProgressTracker.check_stop_criterion
        recorder = self

        def check_stop_criterion(tracker, loss, current_iteration):
            recorder.epoch_weights.append(weights(recorder.model))
            return original(tracker, loss, current_iteration)

        monkeypatch.setattr(ProgressTracker, "check_stop_criterion", check_stop_criterion)

    def build_with(self, build):
        def wrapped(kind, args, settings):
            self.model = build(kind, args, settings)
            self.kind = kind
            return self.model
        return wrapped

    @property
    def epochs_done(self) -> int:
        return len(self.epoch_weights)

    def expected_epoch(self) -> int:
        """The epoch whose weights the model should end with: the lowest training loss among the eligible
        epochs (full KL weight; every epoch for MSVQ, which has no KL), else the last epoch."""
        losses = self.model.loss_total_list
        n = len(losses)
        if self.kind == "MSVQ_VAE":
            eligible = list(range(n))
        else:
            eligible = [e for e in range(n) if kl_beta(self.model, e) >= self.model.max_beta]
        if not eligible:
            return n - 1
        return min(eligible, key=lambda e: losses[e])  # first of equal minima, like the loop's strict "<"
