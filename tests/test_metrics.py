"""Denormalisation (fixed per colour channel) and the image metrics built on it."""
from __future__ import annotations

import numpy as np

from gpvae_endo.evaluation.metrics import Denormaliser

MEANS = (0.4, 0.3, 0.2)
STDS = (0.25, 0.2, 0.15)


def _expected(x):
    return x * np.asarray(STDS, dtype=x.dtype) + np.asarray(MEANS, dtype=x.dtype)


def test_batch_of_frames_is_denormalised_per_colour_channel():
    """The original indexed ``[:, :, c]``, which on a batch (N, H, W, C) is the *width* axis of frame
    rows, so only a 3-pixel-wide strip was denormalised and the rest stayed zero. Every metric used it."""
    x = np.random.default_rng(0).standard_normal((5, 12, 12, 3)).astype(np.float32)
    out = Denormaliser(MEANS, STDS)(x)
    np.testing.assert_allclose(out, _expected(x), rtol=1e-6)
    assert np.count_nonzero(out == 0) == 0


def test_single_frame_unchanged_from_original():
    """For one frame (H, W, C) the original was right; the fix gives exactly the same numbers there."""
    x = np.random.default_rng(1).standard_normal((12, 12, 3)).astype(np.float32)
    original = np.zeros_like(x)
    for c in range(3):
        original[:, :, c] = x[:, :, c] * STDS[c]
        original[:, :, c] += MEANS[c]
    np.testing.assert_array_equal(Denormaliser(MEANS, STDS)(x), original)


def test_identity_without_statistics():
    x = np.random.default_rng(2).random((2, 4, 4, 3)).astype(np.float32)
    np.testing.assert_array_equal(Denormaliser()(x), x)
