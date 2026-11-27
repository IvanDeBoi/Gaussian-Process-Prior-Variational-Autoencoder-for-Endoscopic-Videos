"""The refactored models train exactly like the original code (commit aaf8b6d).

Each case trains a tiny model on synthetic frames on the CPU (1 thread, seed 0) and compares a fingerprint
(losses, log lines, learning rates, every weight's norm, final latents, saved keys, encode/decode outputs,
random state afterwards) with the one the original code produced (``equivalence/legacy_reference.json``).

One intended difference: the best-epoch restore is fixed, so a model now ends with the weights of its best
epoch, where the original always kept the last epoch's. Therefore:

* the training itself (losses, log lines, learning rates, random state) must always equal the original;
* the final weights must be exactly those of the best eligible epoch (recorded during the run);
* when that best epoch is the last one, the *whole* fingerprint must equal the original.

Exact equality for every model except GPVAE_HPA: its KL is computed with a faster formula that agrees with
the original to float rounding, which then propagates through training at the 1e-6 level.
"""
from __future__ import annotations

import pytest
import torch

import harness
import refactored
from best_epoch import TRAJECTORY_KEYS, EpochRecorder, same_weights, weights

REFERENCE = harness.load_reference()
TOLERANCE = {"HPA_EndoVAE": 1e-4, "HPA_GastroEnc": 1e-4}


def _cases():
    for case in harness.CASES:
        marks = [pytest.mark.slow] if case in harness.SLOW_CASES else []
        yield pytest.param(case, marks=marks, id=case)


@pytest.mark.parametrize("case", _cases())
def test_matches_original(case, monkeypatch):
    if case not in REFERENCE:
        pytest.skip(f"{case} not recorded")
    expected = REFERENCE[case]
    recorder = EpochRecorder(monkeypatch)
    got = harness.run_case(recorder.build_with(refactored.build), refactored.train, case)

    best, last = recorder.expected_epoch(), recorder.epochs_done - 1
    assert same_weights(weights(recorder.model), recorder.epoch_weights[best]), \
        f"final weights are not those of the best epoch {best}"

    if best != last:  # best-epoch fix: only the training trajectory can be compared with the original
        expected = {k: v for k, v in expected.items() if k in TRAJECTORY_KEYS}
        got = {k: v for k, v in got.items() if k in TRAJECTORY_KEYS}
    rtol = TOLERANCE.get(case, 0.0)
    if rtol:
        # log lines print the rounded losses; compare them as numbers via the loss lists instead
        expected = {k: v for k, v in expected.items() if k != "epoch_lines"}
        got = {k: v for k, v in got.items() if k != "epoch_lines"}
    problems = harness.compare(expected, got, rtol=rtol)
    assert not problems, f"{len(problems)} differences, first ones:\n" + "\n".join(problems[:25])


@pytest.mark.parametrize("case", ["VAE_EndoVAE", "HPA_EndoVAE", "SPA_EndoVAE", "MSVQ"])
def test_best_epoch_is_restored(case, monkeypatch):
    """Best-epoch fix: make the last epoch's loss look worse (x10) so it cannot be the best; the model must
    then end with an earlier epoch's weights, not the last ones."""
    recorder = EpochRecorder(monkeypatch)
    epochs = harness.CASES[case][3]

    def build(kind, args, settings):
        model = refactored.build(kind, args, settings)
        original = model.compute_total_loss

        def compute_total_loss(*a, **kw):
            total, parts = original(*a, **kw)
            if recorder.epochs_done == epochs - 1:  # during the last epoch
                total = total * 10
                if "total" in parts:
                    parts = {**parts, "total": parts["total"] * 10}
            return total, parts

        model.compute_total_loss = compute_total_loss
        return model

    harness.run_case(recorder.build_with(build), refactored.train, case)
    best, last = recorder.expected_epoch(), recorder.epochs_done - 1
    assert best < last
    final = weights(recorder.model)
    assert same_weights(final, recorder.epoch_weights[best])
    assert not same_weights(final, recorder.epoch_weights[last])


def test_casale_still_fails_when_kl_switches_on():
    """Kept from the original: GPVAE_Casale's KL term does not accept the loop's arguments."""
    torch.set_num_threads(harness.THREADS)
    harness.seed_all(0)
    args = harness.model_args("GPVAE_Casale", "EndoVAEEnc", "EndoVAEDec", epochs=3)
    model = refactored.build("GPVAE_Casale", args, {**harness.TINY_SETTINGS, "WARMUP_EPOCHS": 1})
    with pytest.raises(TypeError, match="Xb_norm"):
        model.train()
