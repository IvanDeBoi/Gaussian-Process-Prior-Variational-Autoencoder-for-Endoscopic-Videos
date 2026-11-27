"""Checkpoint and resume for training (VAE, GPVAE_Casale, GPVAE_HPA, GPVAE_SPA and MSVQ_VAE).

Why: a cluster job has a wall-time limit (24 h on CalcUA). Without checkpoints a run that does not finish
in time is lost. With them, training saves its full state at epoch boundaries and the next job continues
exactly where the previous one stopped.

How it is used (see :func:`gpvae_endo.pipeline._train_and_evaluate`)::

    model.train(checkpoint=CheckpointConfig(path="Saved_Models/<run>.ckpt.pt"))

* If the file exists, training resumes from it automatically.
* A checkpoint is written only at an epoch boundary, after the learning-rate update:
  - every ``interval_s`` seconds (default 15 min);
  - when the next epoch would end after the deadline (``SLURM_JOB_END_TIME`` minus ``deadline_margin_s``);
    then :class:`RequeueRequested` is raised and the program exits with code 85 ("saved, start me again");
  - after ``stop_after_epochs`` epochs in this process (for tests); then also :class:`RequeueRequested`;
  - once training has finished, marked ``finished``: resuming a finished checkpoint restores the final
    state and does not train again.
* Without a checkpoint config nothing is saved or loaded and training is exactly the original.

A resumed run is identical to a straight run (tested in ``tests/test_checkpoint.py``). That needs, in
this order: the model constructed as usual (construction consumes random numbers), weights loaded *into the
existing modules* (the optimiser holds references to them), optimiser state, both LR schedulers, progress
tracker, best-epoch bookkeeping and loss histories, and finally all RNG states.

The best-epoch snapshot (``best_state``) is stored with the checkpoint. Since the B2 fix it is a real copy,
so its values are simply saved. :func:`pack_snapshot` / :func:`unpack_snapshot` still recognise a tensor
that *is* the live tensor and re-link it on resume (the original's aliasing, finding B2), so a snapshot of
either kind survives a resume unchanged.

Checkpoints are loaded with ``torch.load(weights_only=False)``: they hold numpy RNG states and arrays.
Only load checkpoints this code wrote.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch

from gpvae_endo.utils.io import atomic_save
from gpvae_endo.utils.seeding import get_rng_state, set_rng_state

EXIT_REQUEUE = 85
FORMAT = 1


class RequeueRequested(Exception):
    """Training stopped after saving a checkpoint because the job's time is nearly up (or the per-process
    epoch budget is used). The program should exit with :data:`EXIT_REQUEUE` (85)."""

    exit_code = EXIT_REQUEUE

    def __init__(self, reason: str, next_epoch: int, path, saved: bool):
        self.reason, self.next_epoch, self.path, self.saved = reason, next_epoch, Path(path), saved
        state = "saved" if saved else "NOT saved (non-finite weights; the previous checkpoint was kept)"
        super().__init__(f"requeue ({reason}) before epoch {next_epoch}; checkpoint {state}: {path}")


class CheckpointMismatch(RuntimeError):
    """The checkpoint was written with different settings than the model being resumed."""


@dataclass
class CheckpointConfig:
    path: str | os.PathLike
    interval_s: float = 900.0
    deadline: Optional[float] = None  # absolute unix time to stop by; overrides SLURM_JOB_END_TIME
    deadline_margin_s: float = 600.0  # subtracted from SLURM_JOB_END_TIME
    stop_after_epochs: Optional[int] = None  # save and requeue after this many epochs in this process
    extra_config: dict = field(default_factory=dict)  # also part of the settings fingerprint (seed, video, ...)
    clock: Callable[[], float] = time.time

    def resolved_deadline(self, environ=None) -> Optional[float]:
        if self.deadline is not None:
            return float(self.deadline)
        end = (os.environ if environ is None else environ).get("SLURM_JOB_END_TIME")
        if end:
            return float(end) - float(self.deadline_margin_s)
        return None


# --------------------------------------------------------------------------------------------------------
# Settings fingerprint: refuse to resume a checkpoint written with different settings
# --------------------------------------------------------------------------------------------------------
_FINGERPRINT_ATTRS = (
    "underlying_method", "latent_dim", "img_H", "img_W", "img_C", "epochs", "lr", "max_beta", "batch_size",
    "warmup_epochs", "ramp_epochs", "lpips_target_size", "loss_weights", "kernel_gp_prior", "n_neighbors",
    "use_decoder_gan", "encoder_head_lr", "decoder_lr", "lora_lr", "beta_commit", "code_dim",
    "early_stopping_patience", "early_stopping_start_fraction",
)


def settings_fingerprint(model, extra: dict) -> dict:
    cfg = {name: getattr(model, name) for name in _FINGERPRINT_ATTRS if hasattr(model, name)}
    cfg["encoder_class"] = type(model.encoder).__name__
    cfg["decoder_class"] = type(model.decoder).__name__
    cfg["n_frames"] = int(model.Ytensor.shape[0])
    cfg["run"] = dict(extra)
    return json.loads(json.dumps(cfg, default=str))


def _hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _diff(old: dict, new: dict) -> list[str]:
    keys = sorted(set(old) | set(new))
    return [f"{k}: checkpoint={old.get(k, '<missing>')!r} now={new.get(k, '<missing>')!r}"
            for k in keys if old.get(k, "<missing>") != new.get(k, "<missing>")]


# --------------------------------------------------------------------------------------------------------
# The best-epoch snapshot, with its aliasing preserved
# --------------------------------------------------------------------------------------------------------
_ALIAS = "__alias_of_live_tensor__"


def _same_tensor(a, b) -> bool:
    return (torch.is_tensor(a) and torch.is_tensor(b) and a.device == b.device and a.shape == b.shape
            and a.dtype == b.dtype and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
            and a.storage_offset() == b.storage_offset() and a.stride() == b.stride())


def pack_snapshot(snapshot, live):
    """Replace every tensor of `snapshot` that is still the live tensor at the same place in `live` (the
    model's current ``state_dict_full()``) by a marker; everything else is kept by value."""
    if isinstance(snapshot, dict):
        live = live if isinstance(live, dict) else {}
        return {k: pack_snapshot(v, live.get(k)) for k, v in snapshot.items()}
    if torch.is_tensor(snapshot) and _same_tensor(snapshot, live):
        return _ALIAS
    return snapshot


def unpack_snapshot(packed, live):
    """Inverse of :func:`pack_snapshot`: markers become the live tensors again."""
    if isinstance(packed, dict):
        live = live if isinstance(live, dict) else {}
        return {k: unpack_snapshot(v, live.get(k)) for k, v in packed.items()}
    if isinstance(packed, str) and packed == _ALIAS:
        return live
    return packed


def _without_lists(state: dict) -> dict:
    return {k: v for k, v in state.items() if "list" not in k}


# --------------------------------------------------------------------------------------------------------
# The checkpointer used inside the training loops
# --------------------------------------------------------------------------------------------------------
class TrainingCheckpointer:
    """Saves and restores everything a training loop needs to continue identically.

    `history` maps loss-list names to the list objects the loop appends to (restored in place).
    `model.checkpoint_state()` / `model.load_checkpoint_state()` provide the model-specific part
    (weights loaded into the existing modules, never new objects).
    """

    def __init__(self, model, cfg: CheckpointConfig, history: dict):
        self.model = model
        self.cfg = cfg
        self.history = history
        self.path = Path(cfg.path)
        self.clock = cfg.clock
        self.settings = settings_fingerprint(model, cfg.extra_config)
        self.settings_hash = _hash(self.settings)
        self.deadline = cfg.resolved_deadline()
        self.last_save = self.clock()
        self._last_mark = self.last_save
        self.max_epoch_s = 0.0
        self.epochs_this_process = 0

    # -- resume ---------------------------------------------------------------------------------------------
    def resume(self, tracker) -> Optional[dict]:
        if not self.path.exists():
            return None
        m = self.model
        ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
        if ckpt.get("format") != FORMAT:
            raise CheckpointMismatch(f"{self.path}: unknown checkpoint format {ckpt.get('format')!r}")
        if ckpt["settings_hash"] != self.settings_hash:
            diff = _diff(ckpt.get("settings", {}), self.settings)
            raise CheckpointMismatch(
                f"refusing to resume {self.path}: it was written with different settings "
                f"({len(diff)} differences):\n  " + "\n  ".join(diff[:30])
                + "\nDelete the checkpoint file to start this run from scratch."
            )

        m.load_checkpoint_state(ckpt["model"])                   # weights into the existing modules
        m.optimizer.load_state_dict(ckpt["optimizer"])           # then the optimiser(s)
        if getattr(m, "disc_optimizer", None) is not None and ckpt.get("disc_optimizer") is not None:
            m.disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        m.warmup_scheduler.load_state_dict(ckpt["warmup_scheduler"])  # then the schedulers
        m.main_scheduler.load_state_dict(ckpt["main_scheduler"])
        tracker.__dict__.update(ckpt["tracker"])
        for name, lst in self.history.items():
            lst[:] = ckpt["history"][name]

        best_state = ckpt["best_state"]
        if best_state is not None and ckpt.get("best_state_packed", False):
            best_state = unpack_snapshot(best_state, _without_lists(m.state_dict_full()))

        if ckpt["finished"]:
            for name, lst in self.history.items():  # what train() leaves on the model when it ends
                setattr(m, name, lst)
        set_rng_state(ckpt["rng"])                                # RNG last: construction consumed RNG
        self._last_mark = self.clock()
        what = "finished run; not training again" if ckpt["finished"] else f"resuming at epoch {ckpt['next_epoch']}"
        print(f"[checkpoint] loaded {self.path}: {what}")
        return {
            "finished": bool(ckpt["finished"]),
            "next_epoch": int(ckpt["next_epoch"]),
            "best_loss": ckpt["best_loss"],
            "best_state": best_state,
            "extra": ckpt.get("extra"),
        }

    # -- per epoch ------------------------------------------------------------------------------------------
    def end_of_epoch(self, epoch: int, tracker, best_loss, best_state, extra=None) -> None:
        """Called after the LR update of `epoch`. May save, and may raise RequeueRequested after saving."""
        now = self.clock()
        self.max_epoch_s = max(self.max_epoch_s, now - self._last_mark)
        self._last_mark = now
        self.epochs_this_process += 1
        if epoch >= self.model.epochs - 1:
            return  # training finishes next and writes the finished checkpoint

        reason = None
        if self.cfg.stop_after_epochs is not None and self.epochs_this_process >= self.cfg.stop_after_epochs:
            reason = "stop_after"
        elif self.deadline is not None and now + self.max_epoch_s >= self.deadline:
            reason = "deadline"
        elif now - self.last_save >= self.cfg.interval_s:
            reason = "interval"
        if reason is None:
            return
        saved = self.save(epoch + 1, tracker, best_loss, best_state, finished=False, extra=extra)
        if reason != "interval":
            raise RequeueRequested(reason, epoch + 1, self.path, saved)

    def save_finished(self, tracker, best_loss, best_state, extra=None) -> bool:
        return self.save(len(self.history["loss_total_list"]), tracker, best_loss, best_state,
                         finished=True, extra=extra)

    # -- save -----------------------------------------------------------------------------------------------
    def save(self, next_epoch: int, tracker, best_loss, best_state, finished: bool, extra=None) -> bool:
        m = self.model
        weights = m.checkpoint_state()
        if not _weights_finite(weights):
            print(f"[checkpoint] NOT saving before epoch {next_epoch}: non-finite weights; "
                  f"keeping the previous checkpoint {self.path}")
            return False
        packed_best = None
        if best_state is not None:
            packed_best = pack_snapshot(best_state, _without_lists(m.state_dict_full()))
        state = {
            "format": FORMAT,
            "finished": bool(finished),
            "next_epoch": int(next_epoch),
            "settings_hash": self.settings_hash,
            "settings": self.settings,
            "model": weights,
            "optimizer": m.optimizer.state_dict(),
            "disc_optimizer": m.disc_optimizer.state_dict() if getattr(m, "disc_optimizer", None) is not None else None,
            "warmup_scheduler": m.warmup_scheduler.state_dict(),
            "main_scheduler": m.main_scheduler.state_dict(),
            "best_loss": best_loss,
            "best_state": packed_best,
            "best_state_packed": True,
            "tracker": dict(tracker.__dict__),
            "history": {k: list(v) for k, v in self.history.items()},
            "extra": extra,
            "rng": get_rng_state(),
            "saved_at": time.time(),
        }
        t0 = time.time()
        atomic_save(self.path, lambda f: torch.save(state, f))
        self.last_save = self.clock()
        print(f"[checkpoint] saved {'finished ' if finished else ''}checkpoint before epoch {next_epoch} "
              f"to {self.path} ({time.time() - t0:.1f} s)")
        return True


def _weights_finite(state) -> bool:
    """True when every floating-point tensor in the (nested) model state is finite."""
    if isinstance(state, dict):
        return all(_weights_finite(v) for v in state.values())
    if torch.is_tensor(state) and state.is_floating_point():
        return bool(torch.isfinite(state).all())
    return True
