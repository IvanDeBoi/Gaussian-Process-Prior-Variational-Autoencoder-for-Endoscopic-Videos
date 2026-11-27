"""Everything that happens to a trained model: reconstructions, metrics and plots.

Two ways to reconstruct a frame are evaluated:

* **image-encoded**: frame -> encoder -> latent -> decoder -> frame (how well the model compresses);
* **time-encoded** (GP models only): frame *number* -> GP regression over the training latents -> latent ->
  decoder -> frame. For test frames this is the interesting case: the model never saw them.

For both, on training and on test frames: RMSE, and optionally FID, SSIM and PSNR, all on denormalised RGB
frames in [0, 1] (masked to the valid pixels when ``MUST_USE_MASK``).

The steps run in exactly the order of the original driver. That matters: some steps draw random numbers
(the PCA in the latent plots for more than 500 frames; building an FID network), and later steps (the next
model's initial weights) depend on what is left.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass, fields
from typing import Optional

import cv2
import numpy as np
import torch

from gpvae_endo.config import ExperimentConfig
from gpvae_endo.data.dataset import Split
from gpvae_endo.evaluation import metrics
from gpvae_endo.evaluation.metrics import Denormaliser, has_frames
from gpvae_endo.paths import OutputPaths
from gpvae_endo.viz.gp import plot_gp_posterior_per_latent_combined, plot_prior_covariance
from gpvae_endo.viz.images import compare_images, decoded_image_to_uint8
from gpvae_endo.viz.latents import plot_latent, plot_latent_with_orig_dec_2d_train_test, plot_pca
from gpvae_endo.viz.reconstruction import reconstruct_and_save_all_frames
from gpvae_endo.viz.training import plot_losses, plot_rmse_vs_frame


@dataclass
class EvalResult:
    """The metrics of one trained model (None = not computed / not available)."""

    rmse_train_encoded_image: Optional[float] = None
    rmse_train_encoded_time: Optional[float] = None
    rmse_test_encoded_image: Optional[float] = None
    rmse_test_encoded_time: Optional[float] = None
    fid_train_image_encoded: Optional[float] = None
    fid_test_image_encoded: Optional[float] = None
    fid_train_time_encoded: Optional[float] = None
    fid_test_time_encoded: Optional[float] = None
    ssim_train_image_encoded: Optional[float] = None
    ssim_test_image_encoded: Optional[float] = None
    ssim_train_time_encoded: Optional[float] = None
    ssim_test_time_encoded: Optional[float] = None
    psnr_train_image_encoded: Optional[float] = None
    psnr_test_image_encoded: Optional[float] = None
    psnr_train_time_encoded: Optional[float] = None
    psnr_test_time_encoded: Optional[float] = None

    def as_dict(self) -> dict:
        return {f.name: (None if getattr(self, f.name) is None else float(getattr(self, f.name)))
                for f in fields(self)}


def _free_memory():
    torch.cuda.empty_cache()
    gc.collect()


def _fmt_or_inf(v) -> str:
    return f"{v:.4f}" if np.isfinite(v) else "inf"


def evaluate_model(model, kind: str, encoder: str, decoder: str, split: Split, x_scaler, x_train_scaled,
                   x_test_scaled, cfg: ExperimentConfig, paths: OutputPaths, denormalise: Denormaliser,
                   str_with_params: str) -> EvalResult:
    """Run every post-training step of the original driver for one trained model."""
    r = EvalResult()
    res = cfg.resolution
    C = 3 if cfg.must_preserve_color else 1
    img_shape = (res, res, C) if C == 3 else (res, res)
    use_mask = cfg.must_use_mask
    has_test = split.has_test  # the original's `test_frac > 0 or MUST_USE_GAP_FOR_TEST`
    Y_train, Y_test = split.Y_train, split.Y_test
    debug_folder = paths.debug_dataset

    def compare(Y_orig, Y_pred, M_orig, nrs, prefix, show_masked, **titles):
        compare_images(Y_original=Y_orig, Y_predicted=Y_pred, M_original=M_orig, nrs=nrs, img_shape=img_shape,
                       folder=debug_folder, prefix=prefix, show_masked=show_masked, denormalise=denormalise,
                       **titles)

    def rmse(Y_orig, Y_pred, M):
        return metrics.rmse_denormalised(Y_orig, Y_pred, denormalise, M=M, img_shape=img_shape)

    plot_losses(model=model, str_with_params=str_with_params, paths=paths)
    if cfg.must_plot_covariance:
        plot_prior_covariance(model, str_with_params, subsample=200, save_folder=paths.output)

    # ---------------------------------------------------------------- image-encoded reconstructions
    n = 1  # every n-th frame (kept from the original's debugging switch)
    print(f"Debugging on {Y_train[::n].shape}")
    print("Encoding train images")
    Y_train_sub = Y_train[::n]
    M_train_sub = split.M_train[::n] if use_mask else None
    Z_img_train_mean, Z_img_train_std = model.image_encode(Y_train_sub.reshape(Y_train_sub.shape[0], res, res, 3))
    _free_memory()

    Y_test_sub = Y_test[::n]
    M_test_sub = split.M_test[::n] if use_mask else None
    if has_test:
        print("Encoding test images")
        Z_img_test_mean, Z_img_test_std = model.image_encode(Y_test_sub.reshape(Y_test_sub.shape[0], res, res, 3))
    else:
        print("No test images to encode")
        Z_img_test_mean = Z_img_test_std = None
    _free_memory()

    if model.Z is not None:
        plot_latent(Z=model.Z[::n], reconstructed_Z_train=Z_img_train_mean, reconstructed_Z_test=Z_img_test_mean,
                    std_train=Z_img_train_std, std_test=Z_img_test_std,
                    str_with_params=f"{str_with_params} image encoded", paths=paths)

    print("Decoding from image encoded latent to image")
    Y_train_img_dec = model.decode(Z_img_train_mean)
    Y_test_img_dec = model.decode(Z_img_test_mean) if has_test else None
    _free_memory()

    if cfg.must_plot_2d_latent:
        plot_latent_with_orig_dec_2d_train_test(
            Z_model=model.Z[::n], Z_train=Z_img_train_mean, Z_test=Z_img_test_mean,
            imgs_train_orig=Y_train[::n], imgs_test_orig=Y_test[::n],
            imgs_train_dec=Y_train_img_dec, imgs_test_dec=Y_test_img_dec,
            str_with_params=f"{str_with_params} image encoded", img_shape=img_shape, zoom=0.3, figsize=(8, 8),
            save_path_latent=paths.output + f"2D latent with orig image enc dec {str_with_params}.png",
            denormalise=denormalise,
        )

    rand_indices = [i for i in [5, 10, 20, 40, 80] if i < Y_train_sub.shape[0]]
    compare(Y_train_sub[rand_indices], Y_train_img_dec[rand_indices],
            M_train_sub[rand_indices] if use_mask else None, rand_indices, f"Tr_i {str_with_params}", True)
    print("Post-training image-space metrics are computed after denormalisation")
    rmse_train_per_image = np.array([
        rmse(Y_train_sub[i:i + 1], Y_train_img_dec[i:i + 1], M_train_sub[i:i + 1] if use_mask else None)
        for i in range(Y_train_sub.shape[0])
    ])
    rmse_test_per_image = None
    if has_test:
        # note: `rand_indices` now refers to test frames, and is reused below for the training-set comparison
        rand_indices = [i for i in [5, 10, 20, 40, 80] if i < Y_test_sub.shape[0]]
        compare(Y_test_sub[rand_indices], Y_test_img_dec[rand_indices],
                M_test_sub[rand_indices] if use_mask else None, rand_indices, f"Te_i{str_with_params}", True)
        rmse_test_per_image = np.array([
            rmse(Y_test_sub[i:i + 1], Y_test_img_dec[i:i + 1], M_test_sub[i:i + 1] if use_mask else None)
            for i in range(Y_test_sub.shape[0])
        ])

    r.rmse_train_encoded_image = rmse(Y_train_sub, Y_train_img_dec, M_train_sub if use_mask else None)
    print(f"RMSE reconstructed Y_train from encoding Y_train (images): {r.rmse_train_encoded_image:.6f}")
    if has_test:
        r.rmse_test_encoded_image = rmse(Y_test_sub, Y_test_img_dec, M_test_sub if use_mask else None)
        print(f"RMSE reconstructed Y_test from encoding Y_test (images): {r.rmse_test_encoded_image:.6f}")

    # ---------------------------------------------------------------- time-encoded reconstructions (GP models)
    Y_train_time_dec = None
    Y_test_time_dec = None
    if hasattr(model, "time_encode"):
        print("Time encoding training points and, if available, test points (latents from time)")
        has_test_x = (x_test_scaled is not None and isinstance(x_test_scaled, np.ndarray)
                      and x_test_scaled.ndim >= 1 and x_test_scaled.shape[0] > 0)
        if has_test_x:
            x_all_scaled = np.concatenate([x_train_scaled[::n], x_test_scaled[::n]], axis=0)
            model.time_encode(x_all_scaled, use_likelihood=True, return_full_cov=True)  # result unused (as before)
            plot_gp_posterior_per_latent_combined(
                model=model, x_train=x_train_scaled[::n], x_test=x_test_scaled[::n],
                z_train_ref=Z_img_train_mean, z_test_ref=Z_img_test_mean,
                str_with_params=f"{str_with_params} time-encoded full-cov", n_samples=10, n_grid=1000, seed=42,
                use_likelihood=True, save_folder=paths.output,
            )
        else:
            print("No test points available: skipping combined train+test full-cov time plot")

        Z_time_train_mean, Z_time_train_std = model.time_encode(x_train_scaled[::n], use_likelihood=True,
                                                                return_full_cov=False)
        if has_test_x:
            Z_time_test_mean, Z_time_test_std = model.time_encode(x_test_scaled[::n], use_likelihood=True,
                                                                  return_full_cov=False)
        else:
            Z_time_test_mean = Z_time_test_std = None

        plot_latent(Z=model.Z[::n], reconstructed_Z_train=Z_time_train_mean, reconstructed_Z_test=Z_time_test_mean,
                    std_train=Z_time_train_std, std_test=Z_time_test_std,
                    str_with_params=f"{str_with_params} time encoded", paths=paths)

        print("Decoding from time encoded latent to image")
        Y_train_time_dec = model.decode(Z_time_train_mean)
        Y_test_time_dec = model.decode(Z_time_test_mean) if has_test_x and Z_time_test_mean is not None else None

        if cfg.must_plot_2d_latent:
            plot_latent_with_orig_dec_2d_train_test(
                Z_model=model.Z[::n], Z_train=Z_time_train_mean, Z_test=Z_time_test_mean,
                imgs_train_orig=Y_train[::n], imgs_test_orig=Y_test[::n] if has_test_x else None,
                imgs_train_dec=Y_train_time_dec, imgs_test_dec=Y_test_time_dec,
                str_with_params=f"{str_with_params} time encoded", img_shape=img_shape, zoom=0.3, figsize=(8, 8),
                save_path_latent=paths.output + f"2D latent with orig time enc dec {str_with_params}.png",
                denormalise=denormalise,
            )

        titles = dict(left_title="Image encoded", right_title="Time encoded")
        print("Comparing image and time encoded reconstructions from train set")
        compare(Y_train_img_dec[rand_indices], Y_train_time_dec[rand_indices],
                M_train_sub[rand_indices] if use_mask else None, rand_indices, f"Tr_i {str_with_params}", False,
                **titles)
        if has_test_x and Y_test_time_dec is not None and Y_test_img_dec is not None:
            print("Comparing image and time encoded reconstructions from test set")
            compare(Y_test_img_dec[rand_indices], Y_test_time_dec[rand_indices],
                    M_test_sub[rand_indices] if use_mask else None, rand_indices, f"Te_i {str_with_params}", False,
                    **titles)
        else:
            print("Skipping test image-vs-time comparison: no test frames available")

        print("Plotting RMSEs vs frame number")
        plot_rmse_vs_frame(
            x_train_scaled=x_train_scaled[::n],
            x_test_scaled=x_test_scaled[::n] if has_test_x else np.empty((0, 1)),
            rmse_train_per_image=rmse_train_per_image,
            rmse_test_per_image=rmse_test_per_image if has_test_x else np.array([]),
            std_train=Z_time_train_std,
            std_test=Z_time_test_std if has_test_x else np.empty((0, Z_time_train_std.shape[1])),
            str_with_params=str_with_params, paths=paths,
        )

        r.rmse_train_encoded_time = rmse(Y_train_sub, Y_train_time_dec, M_train_sub if use_mask else None)
        print(f"RMSE reconstructed Y_train from encoding x_train (time): {r.rmse_train_encoded_time:.6f}")
        if has_test_x and Y_test_time_dec is not None:
            r.rmse_test_encoded_time = rmse(Y_test_sub, Y_test_time_dec, M_test_sub if use_mask else None)
            print(f"RMSE reconstructed Y_test from encoding x_test (time): {r.rmse_test_encoded_time:.6f}")
        else:
            print("RMSE reconstructed Y_test from encoding x_test (time): n/a (no test frames)")
    else:
        print("Skipping time_encode: model has no GP prior (pure VAE)")

    # ---------------------------------------------------------------- denormalised frames for FID/SSIM/PSNR
    Y_train_denorm = denormalise(Y_train_sub.reshape(-1, *img_shape))
    Y_train_img_denorm = denormalise(Y_train_img_dec.reshape(-1, *img_shape))
    Y_test_denorm = denormalise(Y_test_sub.reshape(-1, *img_shape))
    Y_test_img_denorm = denormalise(Y_test_img_dec.reshape(-1, *img_shape)) if has_test else None
    Y_train_time_denorm = (denormalise(Y_train_time_dec.reshape(-1, *img_shape))
                           if Y_train_time_dec is not None else None)
    test_time_available = (Y_test_denorm is not None and Y_test_denorm.shape[0] > 0
                           and Y_test_time_dec is not None and Y_test_time_dec.shape[0] > 0)
    Y_test_time_denorm = denormalise(Y_test_time_dec.reshape(-1, *img_shape)) if test_time_available else None

    if cfg.must_calculate_fid:
        _fid_section(r, cfg, kind, Y_train_denorm, Y_train_img_denorm, Y_test_denorm, Y_test_img_denorm,
                     Y_train_time_dec, Y_train_time_denorm, Y_test_time_dec, Y_test_time_denorm, test_time_available)
        _free_memory()

    M_train_metric = M_train_sub if use_mask else None
    M_test_metric = M_test_sub if use_mask else None
    test_img_available = has_frames(Y_test_denorm, Y_test_img_denorm)

    if cfg.must_calculate_ssim:
        print("Calculate SSIM")

        def ssim_of(real, pred, M):
            return metrics.ssim_from_denorm(real, pred, M, img_shape, cfg.device, data_range=1.0)

        r.ssim_train_image_encoded = ssim_of(Y_train_denorm, Y_train_img_denorm, M_train_metric)
        if test_img_available:
            r.ssim_test_image_encoded = ssim_of(Y_test_denorm, Y_test_img_denorm, M_test_metric)
        print(f"SSIM image decoded train real vs train recon: {r.ssim_train_image_encoded:.4f}")
        if r.ssim_test_image_encoded is not None:
            print(f"SSIM image decoded test real vs test recon: {r.ssim_test_image_encoded:.4f}")
        else:
            print("SSIM image decoded test real vs test recon: not available (no test frames)")

        if hasattr(model, "time_encode") and Y_train_time_dec is not None:
            r.ssim_train_time_encoded = ssim_of(Y_train_denorm, Y_train_time_denorm, M_train_metric)
            if test_time_available:
                r.ssim_test_time_encoded = ssim_of(Y_test_denorm, Y_test_time_denorm, M_test_metric)
            print(f"SSIM time-encoded train real vs train recon: {r.ssim_train_time_encoded:.4f}")
            if r.ssim_test_time_encoded is not None:
                print(f"SSIM time-encoded test real vs test recon: {r.ssim_test_time_encoded:.4f}")
            else:
                print("SSIM time-encoded test: not available (no test frames)")

        print(f"SSIM SUMMARY Substring: {cfg.substring_filter} | RESOLUTION: {res} | {kind}")
        print(f" Image-encoded SSIM (train real vs train recon): {r.ssim_train_image_encoded:.4f}")
        print(f" Image-encoded SSIM (test real vs test recon): {r.ssim_test_image_encoded:.4f}"
              if r.ssim_test_image_encoded is not None else " Image-encoded SSIM (test): not available (no test frames)")
        print(f" Time-encoded SSIM (train real vs train recon): {r.ssim_train_time_encoded:.4f}"
              if r.ssim_train_time_encoded is not None else " Time-encoded SSIM (train): not available (no time_encode)")
        print(f" Time-encoded SSIM (test real vs test recon): {r.ssim_test_time_encoded:.4f}"
              if r.ssim_test_time_encoded is not None else " Time-encoded SSIM (test): not available (no time_encode)")

    if cfg.must_calculate_psnr:
        print("Calculate PSNR")

        def psnr_of(real, pred, ref, M):
            return metrics.psnr_from_denorm(real, pred, ref, M, img_shape, cfg.device, data_range=1.0)

        r.psnr_train_image_encoded = psnr_of(Y_train_denorm, Y_train_img_denorm, Y_train_sub, M_train_metric)
        if test_img_available:
            r.psnr_test_image_encoded = psnr_of(Y_test_denorm, Y_test_img_denorm, Y_test_sub, M_test_metric)
        print("PSNR image decoded train real vs train recon:", _fmt_or_inf(r.psnr_train_image_encoded))
        if r.psnr_test_image_encoded is not None:
            print("PSNR image decoded test real vs test recon:", _fmt_or_inf(r.psnr_test_image_encoded))
        else:
            print("PSNR image decoded test real vs test recon: not available (no test frames)")

        if hasattr(model, "time_encode") and Y_train_time_dec is not None:
            r.psnr_train_time_encoded = psnr_of(Y_train_denorm, Y_train_time_denorm, Y_train_sub, M_train_metric)
            if test_time_available:
                r.psnr_test_time_encoded = psnr_of(Y_test_denorm, Y_test_time_denorm, Y_test_sub, M_test_metric)
            print("PSNR time-encoded train real vs train recon:", _fmt_or_inf(r.psnr_train_time_encoded))
            if r.psnr_test_time_encoded is not None:
                print("PSNR time-encoded test real vs test recon:", _fmt_or_inf(r.psnr_test_time_encoded))
            else:
                print("PSNR time-encoded test: not available (no test frames)")
        else:
            print("No time-encoded decoded arrays available for PSNR (pure VAE or no time_encode).")

    del Y_train_denorm, Y_test_denorm, Y_train_img_denorm, Y_test_img_denorm, Y_train_time_denorm, Y_test_time_denorm
    _free_memory()

    # ---------------------------------------------------------------- optional extras
    if cfg.must_reconstruct_all:
        print(f"\n\n{'=' * 42} Reconstruct all frames {'=' * 103}")
        reconstruct_and_save_all_frames(
            model=model, x_train=split.x_train, x_test=split.x_test, Y_train=Y_train, Y_test=Y_test,
            M_train=split.M_train if use_mask else None, M_test=split.M_test if use_mask else None,
            x_scaler=x_scaler, RESOLUTION=res, SUBSTRING_FILTER=cfg.substring_filter, underlying_model=kind,
            must_preserve_color=cfg.must_preserve_color, encoder_name=encoder, decoder_name=decoder,
            latent_dim=cfg.latent_dim, kernel=cfg.kernel, lpips=cfg.must_use_lpips, mask=use_mask,
            gan=cfg.must_use_decoder_gan, str_with_params=str_with_params, paths=paths, denormalise=denormalise,
        )

    if cfg.must_plot_pca:
        print("Plotting PCA on Y_train")
        plot_pca(Y_train, str_with_params=str_with_params + " Y_train", paths=paths)
        print("Plotting PCA on Z")
        plot_pca(model.Z, str_with_params=str_with_params + " Z", paths=paths)

    if cfg.must_time_encode_new_images and hasattr(model, "time_encode"):
        _save_synthetic_time_frames(model, kind, x_train_scaled, cfg, paths, img_shape, C, denormalise)

    return r


def _fid_section(r, cfg, kind, Y_train_denorm, Y_train_img_denorm, Y_test_denorm, Y_test_img_denorm,
                 Y_train_time_dec, Y_train_time_denorm, Y_test_time_dec, Y_test_time_denorm, test_time_available):
    """FID of image-encoded (and time-encoded) reconstructions vs the real frames."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = cfg.device
    print("Denormalise all decoded images and calculate FID")
    # The original computed the train FID twice (finding B8) and printed it twice. Building the FID
    # network draws torch random numbers, so one network is still built and discarded here to keep the
    # random state (and so the next model's initial weights) the same; computing the duplicate value is
    # skipped (it does not touch the random state and its result was never used).
    FrechetInceptionDistance(feature=2048)
    n_real, n_fake = Y_train_denorm.shape[0], Y_train_img_denorm.shape[0]
    print(f"Computing FID on {n_real} real and {n_fake} fake train images with batch_size=32")
    r.fid_train_image_encoded = metrics.fid(Y_train_denorm, Y_train_img_denorm, device)
    print("FID image decoded train real vs train recon:", float(r.fid_train_image_encoded))

    if has_frames(Y_test_denorm, Y_test_img_denorm):
        print(f"Computing FID on {Y_test_denorm.shape[0]} real and {Y_test_img_denorm.shape[0]} fake test images "
              f"with batch_size=32")
        r.fid_test_image_encoded = metrics.fid(Y_test_denorm, Y_test_img_denorm, device)
        print("FID image decoded test real vs test recon:", float(r.fid_test_image_encoded))
    else:
        print("Skipping test FID: no test frames available.")

    if Y_train_time_dec is not None:
        print(f"{Y_train_time_dec.shape} -> {Y_train_time_denorm.shape}")
        print(f"Computing TIME FID on {Y_train_denorm.shape[0]} real and {Y_train_time_denorm.shape[0]} fake "
              f"train images with batch_size=32")
        r.fid_train_time_encoded = metrics.fid(Y_train_denorm, Y_train_time_denorm, device)
        print("FID time-encoded train real vs train recon:", float(r.fid_train_time_encoded))
        if test_time_available and isinstance(Y_test_time_dec, np.ndarray):
            print(f"{Y_test_time_dec.shape} -> {Y_test_time_denorm.shape}")
            print(f"Computing TIME FID on {Y_test_denorm.shape[0]} real and {Y_test_time_denorm.shape[0]} fake "
                  f"test images with batch_size=32")
            r.fid_test_time_encoded = metrics.fid(Y_test_denorm, Y_test_time_denorm, device, verbose_batches=True,
                                                  label="test time")
            print("FID time-encoded test real vs test recon:", float(r.fid_test_time_encoded))
        else:
            print("Skipping time-encoded test FID: no test frames available.")
    else:
        print("No time-encoded decoded arrays available (pure VAE or no time_encode).")

    print(f"FID SUMMARY Substring: {cfg.substring_filter} | RESOLUTION: {cfg.resolution} | {kind}")
    print(f"  Image-encoded FID (train real vs train recon): {float(r.fid_train_image_encoded):.4f}")
    print(f"  Image-encoded FID (test  real vs test  recon): {float(r.fid_test_image_encoded):.4f}"
          if r.fid_test_image_encoded is not None else "  Image-encoded FID (test): not available (no test frames)")
    print(f"  Time-encoded  FID (train real vs train recon): {float(r.fid_train_time_encoded):.4f}"
          if r.fid_train_time_encoded is not None else "  Time-encoded  FID (train): not available (no time_encode)")
    print(f"  Time-encoded  FID (test  real vs test  recon): {float(r.fid_test_time_encoded):.4f}"
          if r.fid_test_time_encoded is not None else "  Time-encoded  FID (test):  not available (no time_encode)")


def _save_synthetic_time_frames(model, kind, x_train_scaled, cfg, paths, img_shape, C, denormalise):
    """Decode 1000 evenly spaced frame numbers across the training range and save them as 1350x1080 PNGs."""
    print("TITLE --- time-encoded recon (synthetic over t in [0,1]) ---")
    t_scaled = np.linspace(x_train_scaled.min(), x_train_scaled.max(), 1000, dtype=np.float32).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        print("Time encoding synthetic points (latents from time)")
        Z_mean, _ = model.time_encode(t_scaled, use_likelihood=True)
        print("Decoding from synthetic time-encoded latent to image")
        Y_synth = model.decode(Z_mean)
        if isinstance(Y_synth, torch.Tensor):
            Y_synth = Y_synth.detach().cpu().numpy()

    name = f"{cfg.substring_filter}_{cfg.resolution}_{kind}_time_encoded_new"
    folder = os.path.join(paths.reconstructed_root, name)
    os.makedirs(folder, exist_ok=True)
    print(f"Saving synthetic time-encoded images to {folder}")
    saved = 0
    for i in range(Y_synth.shape[0]):
        img_uint8 = decoded_image_to_uint8(Y_synth[i], img_shape, cfg.must_preserve_color, denormalise=denormalise)
        img_resized = cv2.resize(img_uint8, (1350, 1080), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(folder, f"{name}_{i:06d}.png")
        if C == 1:
            cv2.imwrite(out_path, img_resized[:, :, 0])
        else:
            cv2.imwrite(out_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        saved += 1
    print(f"Saved {saved} synthetic time-encoded images.")
