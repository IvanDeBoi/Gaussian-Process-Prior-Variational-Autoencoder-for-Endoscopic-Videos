"""Image-quality metrics used after training: RMSE, PSNR, SSIM (masked or not) and FID.

All functions are pure: they take arrays and return numbers. Images come in one of two layouts:

* flat:  ``(N, H*W*C)`` rows, the layout of ``Y`` in the NPZ files;
* image: ``(N, H, W, C)`` (or a single ``(H, W, C)`` image).

Masks are ``(N, H*W)`` with 1 = valid pixel, 0 = ignored (specular highlight / border).

Everything here is moved verbatim from the original driver (``Latent_Space_Discovery.py``), with one
intended fix: ``Denormaliser`` indexes the colour channel with ``[..., c]``. The original wrote
``arr[:, :, c]``, which for an ``(N, H, W, C)`` batch selects *pixel columns* 0-2 instead of the colour
channels, so every post-training metric was computed on a 3-pixel-wide strip (bug B1 in
``docs/findings.md``). For a single ``(H, W, C)`` image both forms are identical, so plots and videos are
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim


# --------------------------------------------------------------------------------------------------------
# Undoing the dataset normalisation
# --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Denormaliser:
    """Maps normalised pixels back to RGB in [0, 1]: ``rgb = x * std[c] + mean[c]`` per colour channel.

    ``means``/``stds`` come from the NPZ file (``data["means"]``, ``data["stds"]``); without them the data
    are already in [0, 1] and the identity (mean 0, std 1) is used.
    """

    means: Sequence[float] = (0, 0, 0)
    stds: Sequence[float] = (1, 1, 1)

    def __call__(self, normalised_image: np.ndarray) -> np.ndarray:
        out = np.zeros_like(normalised_image)
        for c in range(3):
            # B1 fix: `[..., c]` is the colour channel for one image (H, W, C) *and* for a batch (N, H, W, C).
            # Same two-step float arithmetic as the original (multiply, then add in place).
            out[..., c] = normalised_image[..., c] * self.stds[c]
            out[..., c] += self.means[c]
        return out


# --------------------------------------------------------------------------------------------------------
# RMSE / MSE / PSNR
# --------------------------------------------------------------------------------------------------------
def _expand_mask(M: np.ndarray, img_shape) -> np.ndarray:
    """(N, H*W) pixel mask -> (N, H*W*C) per-channel mask, with the original shape checks."""
    H, W, C = img_shape
    if M.shape[1] != H * W:
        raise ValueError(f"Expected mask shape (N,{H * W}), got {M.shape}")
    M_full = np.repeat(M, C, axis=1)  # (N, H*W*C)
    if M_full.shape[1] != H * W * C:
        raise ValueError(f"Expanded mask has wrong size {M_full.shape[1]}, expected {H * W * C}")
    return M_full


def rmse(Y_original, Y_predicted, M=None, img_shape=(336, 336, 3)):
    """RMSE between flat arrays; over the valid pixels only when a mask is given (was ``RMSE_resulting_Y``)."""
    diff2 = (Y_original - Y_predicted) ** 2  # (N, D)
    if M is None:
        return np.sqrt(diff2.mean())
    M_full = _expand_mask(M, img_shape)
    diff2_masked = diff2 * M_full
    valid = M_full.sum()
    if valid <= 0:
        return 0.0
    return np.sqrt(diff2_masked.sum() / valid)


def rmse_denormalised(Y_original, Y_predicted, denormalise: Denormaliser, M=None, img_shape=(336, 336, 3)):
    """RMSE after undoing the dataset normalisation (was ``RMSE_resulting_Y_denormalised``)."""
    Y_original = np.asarray(Y_original)
    Y_predicted = np.asarray(Y_predicted)
    Y_original_denorm = denormalise(Y_original.reshape(-1, *img_shape)).reshape(Y_original.shape[0], -1)
    Y_predicted_denorm = denormalise(Y_predicted.reshape(-1, *img_shape)).reshape(Y_predicted.shape[0], -1)
    return rmse(Y_original_denorm, Y_predicted_denorm, M=M, img_shape=img_shape)


def masked_mse(Y_original, Y_predicted, M, img_shape):
    """MSE over the valid pixels of flat arrays."""
    M_full = _expand_mask(M, img_shape)
    diff2 = (Y_original - Y_predicted) ** 2  # (N, D)
    diff2_masked = diff2 * M_full
    valid = M_full.sum()
    if valid <= 0:
        return 0.0
    return float(diff2_masked.sum() / valid)


def masked_psnr_from_flat(Y_original, Y_predicted, M, img_shape, data_range=1.0):
    """PSNR from the masked MSE. Note: the frames are *not* clipped to [0, 1] first (finding P6)."""
    mse = masked_mse(Y_original, Y_predicted, M, img_shape)
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def to_chw_tensor(arr: np.ndarray, device) -> torch.Tensor:
    """(N, H, W, 3) array -> clipped float32 (N, 3, H, W) tensor on `device` (was ``arr_to_chw_float01``)."""
    return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32)).permute(0, 3, 1, 2).to(device)


def psnr_from_denorm(Y_real_denorm, Y_pred_denorm, Y_real_flat_shape_ref, M, img_shape, device,
                     data_range=1.0):
    """PSNR of denormalised frames: masked MSE based with a mask, torchmetrics PSNR without."""
    if M is not None:
        return masked_psnr_from_flat(
            Y_real_denorm.reshape(Y_real_flat_shape_ref.shape[0], -1),
            Y_pred_denorm.reshape(Y_real_flat_shape_ref.shape[0], -1),
            M,
            img_shape,
            data_range=data_range,
        )
    return float(
        psnr(to_chw_tensor(Y_pred_denorm, device), to_chw_tensor(Y_real_denorm, device),
             data_range=data_range).mean()
    )


# --------------------------------------------------------------------------------------------------------
# SSIM
# --------------------------------------------------------------------------------------------------------
def masked_ssim_numpy(img1, img2, mask, data_range=1.0):
    """SSIM map of two (H, W, 3) images averaged over channels, then over the valid pixels of `mask`."""
    _, ssim_map = structural_similarity(img1, img2, data_range=data_range, channel_axis=-1, full=True)
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=-1)
    masked_ssim_map = ssim_map * mask
    denom = mask.sum()
    if denom == 0:
        return 0.0
    return float(masked_ssim_map.sum() / denom)


def ssim_from_denorm(Y_real_denorm, Y_pred_denorm, M, img_shape, device, data_range=1.0):
    """Mean SSIM of denormalised frames, clipped to [0, 1]: per-frame masked SSIM with a mask, torchmetrics
    SSIM over the batch without."""
    Y_real_denorm = np.clip(Y_real_denorm, 0.0, 1.0).astype(np.float32)
    Y_pred_denorm = np.clip(Y_pred_denorm, 0.0, 1.0).astype(np.float32)
    N = Y_real_denorm.shape[0]
    H, W, C = img_shape

    if M is not None:
        ssim_vals = [
            masked_ssim_numpy(Y_real_denorm[i], Y_pred_denorm[i], M[i].reshape(H, W), data_range=data_range)
            for i in range(N)
        ]
        return float(np.mean(ssim_vals))

    real_chw = to_chw_tensor(Y_real_denorm, device)
    pred_chw = to_chw_tensor(Y_pred_denorm, device)
    return float(ssim(pred_chw, real_chw, data_range=data_range).mean().item())


# --------------------------------------------------------------------------------------------------------
# FID
# --------------------------------------------------------------------------------------------------------
def to_uint8_chw(arr: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) in [0, 1] -> uint8 (N, 3, H, W). Truncates rather than rounds, as the original (B20)."""
    arr_u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return torch.from_numpy(arr_u8).permute(0, 3, 1, 2)


def fid(real_denorm: np.ndarray, fake_denorm: np.ndarray, device, batch_size: int = 32,
        verbose_batches: bool = False, label: str = "") -> torch.Tensor:
    """Frechet Inception Distance (2048-d Inception features) between two sets of denormalised frames.

    Returns the torchmetrics result tensor (the driver converts it with ``float``).
    `verbose_batches` reproduces the per-batch progress lines the original printed for the time-encoded
    test FID only.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048).to(device)
    real = to_uint8_chw(real_denorm)
    fake = to_uint8_chw(fake_denorm)
    n_real, n_fake = real.size(0), fake.size(0)
    for i in range(0, n_real, batch_size):
        metric.update(real[i:i + batch_size].to(device), real=True)
        if verbose_batches:
            print(f"  [real {label}] processed {min(i + batch_size, n_real)}/{n_real}")
    for i in range(0, n_fake, batch_size):
        metric.update(fake[i:i + batch_size].to(device), real=False)
        if verbose_batches:
            print(f"  [fake {label}] processed {min(i + batch_size, n_fake)}/{n_fake}")
    return metric.compute()


def has_frames(*arrays: Optional[np.ndarray]) -> bool:
    """True when every array is a non-empty numpy array (the driver's many `has_test_frames` checks)."""
    return all(
        a is not None and isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[0] > 0 for a in arrays
    )
