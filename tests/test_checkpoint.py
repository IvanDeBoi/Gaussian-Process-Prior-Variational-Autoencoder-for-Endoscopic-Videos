"""Checkpoint and resume: a run that is stopped and resumed ends exactly like a run that never stopped.

The resumed half runs the way the next cluster job would: the model is built again from scratch with the
same seed (construction draws random numbers), then training finds the checkpoint file and continues.
Everything is compared on the CPU with one thread, bit for bit.
"""
from __future__ import annotations

import time

import pytest
import torch

import harness
import refactored
from gpvae_endo import cli
from gpvae_endo.training.checkpoint import (
    EXIT_REQUEUE,
    CheckpointConfig,
    CheckpointMismatch,
    RequeueRequested,
    pack_snapshot,
    unpack_snapshot,
    _weights_finite,
)
from gpvae_endo.utils.io import atomic_save

# Cases that cross warm-up, the KL ramp and the cosine schedule (Casale only in warm-up: after it, its KL
# term raises a TypeError).
RESUME_CASES = ["VAE_EndoVAE", "HPA_EndoVAE", "SPA_EndoVAE", "Casale_EndoVAE", "MSVQ"]

# Parts of the fingerprint that only describe the process that printed them: the resumed process prints
# and LR-traces only its own epochs.
PER_PROCESS = ("epoch_lines", "lr_trace")


def _without_per_process(fp: dict) -> dict:
    return {k: v for k, v in fp.items() if k not in PER_PROCESS}


def _train_with(checkpoint):
    def train(model, settings):
        refactored.train(model, settings, checkpoint=checkpoint)
    return train


@pytest.mark.parametrize("case", RESUME_CASES)
def test_resumed_run_equals_straight_run(case, tmp_path):
    straight = harness.run_case(refactored.build, refactored.train, case)

    path = tmp_path / "run.ckpt.pt"
    with pytest.raises(RequeueRequested) as stop:
        harness.run_case(refactored.build, _train_with(CheckpointConfig(path=path, stop_after_epochs=2)), case)
    assert stop.value.saved and stop.value.next_epoch == 2 and path.exists()

    resumed = harness.run_case(refactored.build, _train_with(CheckpointConfig(path=path)), case)

    problems = harness.compare(_without_per_process(straight), _without_per_process(resumed))
    assert not problems, "\n".join(problems[:25])
    # the per-process parts line up too: the resumed process printed exactly the remaining epochs
    assert resumed["epoch_lines"] == straight["epoch_lines"][2:]
    if straight["lr_trace"]:
        assert resumed["lr_trace"] == straight["lr_trace"][2:]


def test_checkpointing_without_stopping_changes_nothing(tmp_path):
    """Saving every epoch (interval 0) must not disturb training (no RNG use, no state change)."""
    straight = harness.run_case(refactored.build, refactored.train, "HPA_EndoVAE")
    saving = harness.run_case(refactored.build,
                              _train_with(CheckpointConfig(path=tmp_path / "c.pt", interval_s=0.0)), "HPA_EndoVAE")
    assert not harness.compare(straight, saving)


def test_finished_checkpoint_is_not_trained_again(tmp_path):
    path = tmp_path / "done.ckpt.pt"
    first = harness.run_case(refactored.build, _train_with(CheckpointConfig(path=path)), "VAE_EndoVAE")
    again = harness.run_case(refactored.build, _train_with(CheckpointConfig(path=path)), "VAE_EndoVAE")
    assert again["epoch_lines"] == []  # no epoch was trained
    assert again["best_state_restores"] == 0  # ... and the end-of-training restore did not run again
    first, again = _without_per_process(first), _without_per_process(again)
    first.pop("best_state_restores"), again.pop("best_state_restores")
    assert not harness.compare(first, again)


def test_deadline_saves_and_requests_requeue(tmp_path):
    path = tmp_path / "deadline.ckpt.pt"
    cfg = CheckpointConfig(path=path, deadline=time.time() - 1.0)  # already past
    with pytest.raises(RequeueRequested) as stop:
        harness.run_case(refactored.build, _train_with(cfg), "VAE_EndoVAE")
    assert stop.value.reason == "deadline" and stop.value.next_epoch == 1 and path.exists()
    assert stop.value.exit_code == EXIT_REQUEUE == 85


def test_slurm_job_end_time_sets_the_deadline():
    cfg = CheckpointConfig(path="x", deadline_margin_s=600)
    assert cfg.resolved_deadline({"SLURM_JOB_END_TIME": "1000000"}) == 1000000 - 600
    assert cfg.resolved_deadline({}) is None
    assert CheckpointConfig(path="x", deadline=5.0).resolved_deadline({"SLURM_JOB_END_TIME": "9"}) == 5.0


def test_refuses_checkpoint_from_other_settings(tmp_path):
    path = tmp_path / "other.ckpt.pt"
    with pytest.raises(RequeueRequested):
        harness.run_case(refactored.build, _train_with(CheckpointConfig(path=path, stop_after_epochs=1)),
                         "VAE_EndoVAE")
    with pytest.raises(CheckpointMismatch, match="different settings"):
        harness.run_case(refactored.build,
                         _train_with(CheckpointConfig(path=path, extra_config={"seed": 1})), "VAE_EndoVAE")


def test_cli_turns_requeue_into_exit_85(monkeypatch):
    import gpvae_endo.pipeline as pipeline

    def fake_run(config_path, options):
        assert options.enabled and options.stop_after_epochs == 3
        raise RequeueRequested("stop_after", 3, "x.ckpt.pt", saved=True)

    monkeypatch.setattr(pipeline, "run", fake_run)
    assert cli.main(["run", "cfg.py", "--checkpoint", "--stop-after-epochs", "3"]) == 85


def test_atomic_save_keeps_old_file_when_writing_fails(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"old")

    def broken(f):
        f.write(b"half")
        raise RuntimeError("killed")

    with pytest.raises(RuntimeError):
        atomic_save(path, broken)
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["f.bin"]  # no temporary file left behind
    atomic_save(path, lambda f: f.write(b"new"))
    assert path.read_bytes() == b"new"


def test_snapshot_aliasing_survives_a_checkpoint():
    """The original's aliasing is still handled: a best-state entry that still *is* the live tensor is
    re-linked on resume."""
    live = {"encoder": {"w": torch.ones(2), "b": torch.zeros(2)}}
    snapshot = {"encoder": {"w": live["encoder"]["w"], "b": live["encoder"]["b"].clone()}}
    packed = pack_snapshot(snapshot, live)
    assert isinstance(packed["encoder"]["w"], str) and torch.is_tensor(packed["encoder"]["b"])
    new_live = {"encoder": {"w": torch.full((2,), 3.0), "b": torch.zeros(2)}}
    restored = unpack_snapshot(packed, new_live)
    assert restored["encoder"]["w"] is new_live["encoder"]["w"]
    assert restored["encoder"]["b"] is packed["encoder"]["b"]


def test_non_finite_weights_are_not_saved():
    assert _weights_finite({"a": {"w": torch.ones(3)}, "n": 3})
    assert not _weights_finite({"a": {"w": torch.tensor([1.0, float("nan")])}})
