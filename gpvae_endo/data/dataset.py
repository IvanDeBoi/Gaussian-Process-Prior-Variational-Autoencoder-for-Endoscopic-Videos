"""Loading one preprocessed video (an NPZ file) and splitting its frames into training and test frames.

An NPZ file made by ``Prepare_Dataset.py`` holds, for one video:

* ``x``      frame numbers, shape (N, 1);
* ``Y``      the frames, normalised and flattened, shape (N, H*W*3);
* ``M``      valid-pixel masks (1 = use, 0 = specular highlight / ignore), shape (N, H*W);
* ``means``, ``stds`` (optional) the per-channel normalisation, used to turn frames back into RGB.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from gpvae_endo.evaluation.metrics import Denormaliser

if TYPE_CHECKING:
    from gpvae_endo.config import ExperimentConfig

IMG_H, IMG_W, IMG_C = 336, 336, 3  # the frame size the driver assumes (hard-coded in the original)


@dataclass
class VideoData:
    """All frames of one video as loaded from its NPZ file."""

    x: np.ndarray
    Y: np.ndarray
    M: Optional[np.ndarray]
    denormalise: Denormaliser


@dataclass
class Split:
    """Training and test frames, each sorted by frame number."""

    idx_train: np.ndarray
    idx_test: np.ndarray
    x_train: np.ndarray
    Y_train: np.ndarray
    M_train: Optional[np.ndarray]
    x_test: np.ndarray
    Y_test: np.ndarray
    M_test: Optional[np.ndarray]
    has_test: bool  # the original's `test_frac > 0 or MUST_USE_GAP_FOR_TEST`
    test_frac: float = field(default=0.0)


def npz_path(cfg: "ExperimentConfig") -> str:
    """``<FOLDER_DATA>/<video> <resolution> X Y<normalisation>M.npz``."""
    return os.path.join(cfg.folder_data, f"{cfg.substring_filter} {cfg.resolution} X Y{cfg.substring_normalisation}M.npz")


def load_video(cfg: "ExperimentConfig") -> VideoData:
    """Load the NPZ file for ``cfg.substring_filter`` and print the same summary lines as the original."""
    str_with_params = f"{cfg.substring_filter} {cfg.resolution}"
    path = npz_path(cfg)
    print(f"Loading {str_with_params} X Y{cfg.substring_normalisation} X Y{cfg.substring_normalisation}.npz")
    print("Loading from:", os.path.abspath(path))
    data = np.load(path)
    x = data["x"]
    print(f"x shape: {x.shape}, min {x.min()}, max {x.max()}")
    Y = data["Y"]
    print(f"Y shape: {Y.shape}, min {Y.min()}, max {Y.max()}")
    if cfg.must_use_mask:
        M = data["M"]
        print(f"M shape: {M.shape}, min {M.min()}, max {M.max()}")
    else:
        M = None
        print(f"M set to None")
    means, stds = [0, 0, 0], [1, 1, 1]
    if "means" in data.files and "stds" in data.files:
        means, stds = data["means"], data["stds"]
    print(f"Will denormalise by multiplying by {stds} and adding {means}")

    N = Y.shape[0]
    Y_reshaped = Y.reshape(N, IMG_H, IMG_W, IMG_C)
    print(Y_reshaped.shape)
    Y_R = Y_reshaped[:, :, :, 0].reshape(-1)
    Y_G = Y_reshaped[:, :, :, 1].reshape(-1)
    Y_B = Y_reshaped[:, :, :, 2].reshape(-1)
    print("Current Y means per channel:", Y_R.mean(), Y_G.mean(), Y_B.mean())
    print("Current Y stds per channel:", Y_R.std(), Y_G.std(), Y_B.std())
    return VideoData(x=x, Y=Y, M=M, denormalise=Denormaliser(means, stds))


def split_frames(video: VideoData, cfg: "ExperimentConfig") -> Split:
    """Choose the test frames, then sort both sets by frame number.

    * ``MUST_USE_GAP_FOR_TEST``: a block of ``GAP_SIZE`` consecutive frames in the middle of the video;
    * otherwise ``PERCENTAGE_FOR_TEST`` % random frames (fixed ``random_state=42``, so the same test frames
      for every seed, as in the original);
    * plus the last ``PERCENTAGE_FOR_TAIL`` % of the video.
    """
    x, Y, M = video.x, video.Y, video.M
    N = Y.shape[0]
    test_frac = cfg.percentage_for_test / 100.0
    n_tail = int(cfg.percentage_for_tail * N / 100.0)

    all_idx = np.arange(N, dtype=int)
    idx_test = np.array([], dtype=int)

    if cfg.must_use_gap_for_test:
        print("Picking frames for gap in the middle of the dataset")
        mid = N // 2
        half = cfg.gap_size // 2
        start = max(0, mid - half)
        end = min(N, mid + half)
        idx_test = all_idx[start:end]
    elif test_frac > 0:
        print("Picking random frames for test")
        _, idx_test = train_test_split(all_idx, test_size=test_frac, random_state=42, shuffle=True)
        idx_test = np.asarray(idx_test, dtype=int)

    has_test = bool(test_frac > 0 or cfg.must_use_gap_for_test)
    if has_test:
        idx_test_tail = np.arange(N - n_tail, N, dtype=int)
        idx_test = np.unique(np.concatenate([idx_test, idx_test_tail])).astype(int)
        idx_train = np.setdiff1d(all_idx, idx_test)
        print(f"Test indices: {idx_test}")
    else:
        idx_train = all_idx
    print(f"Train indices: {idx_train}")

    x_train, Y_train = x[idx_train], Y[idx_train]
    M_train = M[idx_train] if M is not None else None
    x_test, Y_test = x[idx_test], Y[idx_test]
    M_test = M[idx_test] if M is not None else None

    order = np.argsort(x_train, axis=0).flatten()
    x_train, Y_train = x_train[order], Y_train[order]
    if M_train is not None:
        M_train = M_train[order]

    order = np.argsort(x_test, axis=0).flatten()
    x_test, Y_test = x_test[order], Y_test[order]
    if M_test is not None:
        M_test = M_test[order]

    print(f"{cfg.resolution}x{cfg.resolution}={cfg.resolution * cfg.resolution}")
    print(f"Training Frames (x_train) shape: {x_train.shape}")
    print(f"Training Images (Y_train) shape: {Y_train.shape}")
    if M_train is not None:
        print(f"Training Masks (M_train) shape: {M_train.shape}")
    print(f"Test Frames (x_test) shape: {x_test.shape}")
    print(f"Test Images (Y_test) shape: {Y_test.shape}")
    if M_test is not None:
        print(f"Test Masks (M_test) shape: {M_test.shape}")

    return Split(idx_train=idx_train, idx_test=idx_test, x_train=x_train, Y_train=Y_train, M_train=M_train,
                 x_test=x_test, Y_test=Y_test, M_test=M_test, has_test=has_test, test_frac=test_frac)


class IdentityScaler:
    """Drop-in for sklearn's scalers that leaves x unchanged (raw frame numbers)."""

    def fit(self, x):
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    def fit_transform(self, x):
        return self.transform(x)

    def inverse_transform(self, x):
        return self.transform(x)


def scale_frame_numbers(split: Split, use_x_scaling: bool):
    """Frame numbers as the models see them: raw (default) or min-max scaled to [0, 1].

    Returns ``(x_scaler, x_train_scaled, x_test_scaled)``; ``x_test_scaled`` is None without test frames.
    """
    if use_x_scaling:
        x_scaler = MinMaxScaler()
        print("Scaling x with MinMaxScaler to [0, 1]")
    else:
        x_scaler = IdentityScaler()
        print("X scaling disabled; using raw frame numbers")
    x_train_scaled = x_scaler.fit_transform(split.x_train.reshape(-1, 1))
    x_test_scaled = x_scaler.transform(split.x_test.reshape(-1, 1)) if split.has_test else None
    print(f"x_train_scaled shape: {x_train_scaled.shape}, min {x_train_scaled.min()}, max {x_train_scaled.max()}")
    return x_scaler, x_train_scaled, x_test_scaled
