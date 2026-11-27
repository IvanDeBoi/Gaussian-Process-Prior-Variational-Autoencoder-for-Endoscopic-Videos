"""Multi-scale vector-quantised VAE (MSVQ-VAE): a baseline without a GP prior.

Image -> multi-scale CNN encoder -> feature map -> flattened -> linear layer -> latent vector (``latent_dim``)
-> linear layer -> feature map -> nearest codebook vectors (EMA-updated codebook) -> decoder -> image.

The latent vector is what ``image_encode`` returns and ``decode`` takes, like the other models. Training
(:mod:`gpvae_endo.training.msvq_loop`) minimises 5 * masked MSE + 2 * LPIPS + beta_commit * commitment loss.

Kept from the original on purpose:

* its loss weights are fixed in this file, not read from the config (mse 5, lpips 2, gan 0.01, kl 1);
* ``train()`` is ``nn.Module.train`` extended, so it also puts the LPIPS network in training mode (its dropout
  layers then draw random numbers, unlike in the VAE models);
* the final latents ``Z`` are not computed after training (``Z`` stays ``None``).
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as sched

from gpvae_endo.config import ModelSettings
from gpvae_endo.networks.gan import (
    PatchDiscriminatorWithFeatures,
    discriminator_bce_loss,
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_bce_loss,
    generator_hinge_loss,
)
from gpvae_endo.networks.msvq import EMAVectorQuantizer, MSVQDecoder, MSVQEncoder
from gpvae_endo.training.msvq_loop import MSVQ_HISTORY_NAMES, train_msvq
from gpvae_endo.utils.device import resolve_device

warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")

MSVQ_LOSS_WEIGHTS = {"mse": 5.0, "lpips": 2.0, "gan": 0.01, "kl": 1.0}


class MSVQVAEModel(nn.Module):
    """Multiscale vector-quantised VAE."""

    def __init__(
        self,
        x: np.ndarray,
        Y: np.ndarray,
        M: Optional[np.ndarray] = None,
        device: str = "gpu",
        img_H: int = 224,
        img_W: Optional[int] = None,
        img_C: int = 3,
        underlying_method: str = "MSVQ_VAE",
        codebook_size: int = 256,
        code_dim: int = 128,
        beta_commit: float = 0.1,
        ema_decay: float = 0.99,
        epochs: int = 2000,
        lr: float = 1e-3,
        max_beta: float = 1e-2,
        batch_size: int = 16,
        use_checkpoint: bool = True,
        *,
        settings: ModelSettings,
        **kwargs,
    ):
        super().__init__()
        print(
            f"Initializing {underlying_method}"
            f"\nY: {Y.shape} -> Z: ({Y.shape[0]}, {code_dim}) -> Y: {Y.shape}"
        )
        print(
            f"max_beta={max_beta}, epochs={epochs}, batch_size={batch_size}, lr={lr}, "
            f"codebook_size={codebook_size}, code_dim={code_dim}, beta_commit: {beta_commit}, "
            f"img_H={img_H}, img_W={img_W}, img_C={img_C}"
        )
        self.settings = settings
        self.underlying_method = underlying_method
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = resolve_device(device)  # "gpu" without a GPU stops unless GPVAE_ALLOW_CPU=1 (B10)

        self.img_H = img_H
        self.img_W = img_W or img_H
        self.img_C = img_C
        self.x = x
        self.Y = Y
        self.M = M

        N, D = Y.shape
        assert D == self.img_H * self.img_W * img_C, (
            f"Y.shape[1]={D} != img_C*img_H*img_W={self.img_C * self.img_H * self.img_W}"
        )
        # frames and masks stay on the CPU; batches are moved to the device in the training loop
        self.Ytensor = torch.from_numpy(Y.reshape(N, self.img_H, self.img_W, img_C)).float()
        print(self.Ytensor.shape)
        if M is not None:
            if M.shape[1] == self.img_H * self.img_W:
                M_imgs = M.reshape(N, self.img_H, self.img_W, 1)
            elif M.shape[1] == self.img_H * self.img_W * img_C:
                M_imgs = M.reshape(N, self.img_H, self.img_W, img_C)
            else:
                raise ValueError(f"Unexpected M shape {M.shape}, expected (N,H*W) or (N,H*W*C)")
            self.Mtensor = torch.from_numpy(M_imgs).float()
        else:
            self.Mtensor = None

        self.Xtensor = torch.from_numpy(x).float().to(self.device)
        if self.Xtensor.dim() == 1:
            self.Xtensor = self.Xtensor.unsqueeze(-1)
        self.X_mean = torch.zeros_like(self.Xtensor[0:1])
        self.X_std = torch.ones_like(self.Xtensor[0:1])
        self.Xtensor_norm = (self.Xtensor - self.X_mean) / (self.X_std + 1e-6)

        # --- networks (construction order = order of the random initial weights) ---
        self.encoder = MSVQEncoder(in_channels=img_C, base_channels=8, num_scales=3,
                                   use_checkpoint=use_checkpoint).to(self.device)
        enc_out_channels = self.encoder.out_channels
        self.code_dim = enc_out_channels   # the quantiser works on the encoder's channels
        self.latent_dim = code_dim         # the requested size is the latent vector's size
        self.enc_proj = nn.Identity()
        self.dec_in_proj = nn.Identity()

        # 3 stride-2 downsamples: the feature map is H/8 x W/8
        self._H_lat = self.img_H // (2 ** 3)
        self._W_lat = self.img_W // (2 ** 3)
        spatial_dim = self.code_dim * self._H_lat * self._W_lat
        self.lat_proj = nn.Linear(spatial_dim, self.latent_dim).to(self.device)
        self.lat_unproj = nn.Linear(self.latent_dim, spatial_dim).to(self.device)
        self.quantizer = EMAVectorQuantizer(num_embeddings=codebook_size, embedding_dim=self.code_dim,
                                            decay=ema_decay).to(self.device)
        self.decoder = MSVQDecoder(bottleneck_channels=enc_out_channels, out_channels=img_C, num_scales=3,
                                   use_checkpoint=use_checkpoint).to(self.device)

        self.beta_commit = beta_commit
        self.max_beta = max_beta

        params = (
            list(self.encoder.parameters()) + list(self.enc_proj.parameters()) + list(self.lat_proj.parameters())
            + list(self.lat_unproj.parameters()) + list(self.quantizer.parameters())
            + list(self.dec_in_proj.parameters()) + list(self.decoder.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        self.warmup_epochs = settings.warmup_epochs
        self.ramp_epochs = settings.ramp_epochs
        self.total_epochs = self.epochs
        self.early_stopping_patience = settings.early_stopping_patience
        self.early_stopping_start_fraction = settings.early_stopping_start_fraction
        self.warmup_scheduler = sched.LinearLR(self.optimizer, start_factor=0.1,
                                               total_iters=max(self.warmup_epochs, 1))
        self.main_scheduler = sched.CosineAnnealingLR(self.optimizer, T_max=max(self.total_epochs - self.warmup_epochs, 1),
                                                      eta_min=1e-5)

        self.lpips_net = lpips.LPIPS(net="vgg").to(self.device)
        self.lpips_net.eval()
        for p in self.lpips_net.parameters():
            p.requires_grad = False
        self.lpips_target_size = settings.lpips_res

        self.loss_weights = dict(MSVQ_LOSS_WEIGHTS)
        print(f"LPIPS target size: {self.lpips_target_size} (not full res for performance reasons)")
        print("Loss weights:")
        for k, v in self.loss_weights.items():
            print(f"  {k:>5}: {v}")
        self.loss_weights_commit = self.beta_commit
        print(f"  {'commit':>5}: {self.loss_weights_commit}")

        self._init_gan(settings)

        self.Z = None
        self.mu_all = None
        for name in MSVQ_HISTORY_NAMES + ("loss_weighted_codebook_list",):
            setattr(self, name, [])
        print(f"{underlying_method} initialized successfully")

    def _init_gan(self, s: ModelSettings):
        self.use_decoder_gan = s.must_use_decoder_gan
        self.gan_type = s.gan_type
        self.gan_loss_type = s.gan_loss
        self.gan_weight = MSVQ_LOSS_WEIGHTS.get("gan", 0.0)
        self.gan_disc_weight = s.gan_disc_weight
        self.gan_use_mask = s.gan_use_mask
        self.gan_detach_recon_for_d = s.gan_detach_recon_for_d
        self.gan_label_smoothing = s.gan_label_smoothing
        self.gan_disc_updates = s.gan_disc_updates
        self.gan_warmup_epochs = s.gan_warmup_epochs
        self.gan_feature_matching_weight = s.gan_feature_matching_weight
        self.discriminator = None
        self.disc_optimizer = None
        if self.use_decoder_gan:
            if self.gan_type.lower() != "patchgan":
                raise ValueError(f"Unsupported GAN_TYPE: {self.gan_type}")
            self.discriminator = PatchDiscriminatorWithFeatures(
                in_channels=self.img_C, base_channels=s.gan_base_channels, num_layers=s.gan_num_layers,
                norm_type=s.gan_norm,
            ).to(self.device)
            self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=s.gan_lr, betas=s.gan_betas)
            print(
                f"Initialized decoder GAN: type={self.gan_type}, "
                f"loss={self.gan_loss_type}, weight={self.gan_weight}, D_lr={s.gan_lr}"
            )
        else:
            print("No decoder GAN")

    # ------------------------------------------------------------------------------------------------------
    # encode / quantise / decode
    # ------------------------------------------------------------------------------------------------------
    def encode_continuous(self, x_bchw: torch.Tensor) -> torch.Tensor:
        """Image (B, C, H, W) -> encoder feature map (B, C', H/8, W/8)."""
        return self.enc_proj(self.encoder(x_bchw))

    def encode_to_latent(self, x_bchw: torch.Tensor) -> torch.Tensor:
        """Image -> latent vector (B, latent_dim): feature map, flattened, linear layer."""
        z_e = self.encode_continuous(x_bchw)
        return self.lat_proj(z_e.view(z_e.size(0), -1))

    def latent_to_spatial(self, z_vec: torch.Tensor) -> torch.Tensor:
        """Latent vector -> feature map (B, C', H/8, W/8) for the quantiser and decoder."""
        z_flat = self.lat_unproj(z_vec)
        return z_flat.view(z_vec.size(0), self.encoder.out_channels, self._H_lat, self._W_lat)

    def quantize(self, z_e: torch.Tensor):
        return self.quantizer(z_e)

    def decode_from_quantized(self, z_q_st: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.dec_in_proj(z_q_st))

    def forward(self, x_bchw: torch.Tensor):
        z_e = self.latent_to_spatial(self.encode_to_latent(x_bchw))
        z_q_st, z_q, indices = self.quantize(z_e)
        x_hat = self.decode_from_quantized(z_q_st)
        recon_loss = F.mse_loss(x_hat, x_bchw)
        commit_loss = F.mse_loss(z_e, z_q.detach())
        total_loss = recon_loss + self.beta_commit * commit_loss
        loss_dict = {"total": total_loss.detach(), "recon": recon_loss.detach(), "commit": commit_loss.detach()}
        return total_loss, loss_dict, x_hat, indices

    # ------------------------------------------------------------------------------------------------------
    # optional GAN (off in all configs; slightly different from the VAE's: the fake is always detached for D)
    # ------------------------------------------------------------------------------------------------------
    def _hwc_to_chw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Expected 4D HWC tensor, got shape {x.shape}")

    def _apply_gan_mask(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None or not self.gan_use_mask:
            return x
        m = mask
        while m.dim() < x.dim():
            m = m.unsqueeze(-1)
        return x * m.to(x.device, dtype=x.dtype)

    def _get_discriminator_losses(self, real_logits, fake_logits):
        if self.gan_loss_type.lower() == "hinge":
            return discriminator_hinge_loss(real_logits, fake_logits), generator_hinge_loss(fake_logits)
        if self.gan_loss_type.lower() == "bce":
            return (discriminator_bce_loss(real_logits, fake_logits, smoothing=self.gan_label_smoothing),
                    generator_bce_loss(fake_logits))
        raise ValueError(f"Unsupported GAN_LOSS: {self.gan_loss_type}")

    def _train_discriminator_step(self, real_hwc, fake_hwc, mask=None):
        real_chw = self._hwc_to_chw(self._apply_gan_mask(real_hwc, mask))
        fake_chw = self._hwc_to_chw(self._apply_gan_mask(fake_hwc, mask)).detach()
        real_logits = self.discriminator(real_chw)
        fake_logits = self.discriminator(fake_chw)
        d_loss, _ = self._get_discriminator_losses(real_logits, fake_logits)
        d_loss = self.gan_disc_weight * d_loss
        self.disc_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        del real_logits, fake_logits, d_loss
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.disc_optimizer.step()
        return torch.tensor(0.0, device=self.device)  # placeholder: only logged

    def _generator_adversarial_loss(self, real_hwc, fake_hwc, mask=None):
        real_hwc = self._apply_gan_mask(real_hwc, mask)
        fake_hwc = self._apply_gan_mask(fake_hwc, mask)
        real_chw = self._hwc_to_chw(real_hwc)
        fake_logits, fake_feats = self.discriminator(self._hwc_to_chw(fake_hwc), return_features=True)
        g_adv = generator_hinge_loss(fake_logits) if self.gan_loss_type.lower() == "hinge" else generator_bce_loss(fake_logits)
        if self.gan_feature_matching_weight > 0:
            with torch.no_grad():
                _, real_feats = self.discriminator(real_chw, return_features=True)
            fm = feature_matching_loss(real_feats, fake_feats)
        else:
            fm = torch.tensor(0.0, device=fake_hwc.device, dtype=fake_hwc.dtype)
        return g_adv, fm

    # ------------------------------------------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------------------------------------------
    def compute_total_loss(self, x_batch_hwc, x_batch_chw, x_hat_chw, z_e, z_q, mask=None, kl_beta: float = 1.0,
                           use_gan: bool = False):
        """Weighted masked MSE + LPIPS + commitment (+ optional GAN). No KL term (always 0 here)."""
        zero = lambda: torch.tensor(0.0, device=x_batch_chw.device, dtype=x_batch_chw.dtype)  # noqa: E731
        if self.loss_weights["mse"] > 0:
            if mask is not None:
                m_chw = mask.permute(0, 3, 1, 2).contiguous() if mask.dim() == 4 else mask.unsqueeze(1)
                m_chw = m_chw.to(x_batch_chw.device, dtype=x_batch_chw.dtype)
                se = (x_hat_chw - x_batch_chw) ** 2
                mse = (se * m_chw).sum() / torch.clamp_min(m_chw.sum(), 1.0)
            else:
                mse = F.mse_loss(x_hat_chw, x_batch_chw)
        else:
            mse = zero()

        if self.loss_weights["lpips"] > 0:
            x_ds = F.interpolate(x_batch_chw, size=self.lpips_target_size, mode="bilinear", align_corners=False)
            recon_ds = F.interpolate(x_hat_chw, size=self.lpips_target_size, mode="bilinear", align_corners=False)
            lpips_val = self.lpips_net(torch.tanh(recon_ds), torch.tanh(x_ds)).mean()
        else:
            lpips_val = zero()

        kl = zero()
        effective_kl_weight = 0.0

        commit = F.smooth_l1_loss(z_e, z_q.detach(), beta=1.0)
        codebk = F.mse_loss(z_e.detach(), z_q)

        use_disc_based_generator_loss = (
            use_gan and self.use_decoder_gan and self.discriminator is not None
            and (self.loss_weights["gan"] > 0 or self.gan_feature_matching_weight > 0)
        )
        if use_disc_based_generator_loss:
            gan, fm = self._generator_adversarial_loss(real_hwc=x_batch_hwc, fake_hwc=x_hat_chw.permute(0, 2, 3, 1),
                                                       mask=mask)
        else:
            gan, fm = zero(), zero()

        weighted_mse = self.loss_weights["mse"] * mse
        weighted_lpips = self.loss_weights["lpips"] * lpips_val
        weighted_kl = effective_kl_weight * kl
        weighted_commit = self.loss_weights_commit * commit
        weighted_gan = self.loss_weights["gan"] * gan
        weighted_fm = self.gan_feature_matching_weight * fm
        total = weighted_mse + weighted_lpips + weighted_kl + weighted_commit + weighted_gan + weighted_fm

        if not torch.isfinite(total):
            print(
                "[compute_total_loss] NON-FINITE loss! "
                f"wmse={weighted_mse.item():.6f}, "
                f"wlpips={weighted_lpips.item():.6f}, "
                f"wkl={weighted_kl.item():.6f}, "
                f"wcommit={weighted_commit.item():.6f}, "
                f"wgan={weighted_gan.item():.6f}, "
                f"wfm={weighted_fm.item():.6f}"
            )
        loss_dict = {
            "total": total.detach(), "mse": mse.detach(), "lpips": lpips_val.detach(), "kl": kl.detach(),
            "commit": commit.detach(), "codebook": codebk.detach(), "gan": gan.detach(), "fm": fm.detach(),
            "weighted_mse": weighted_mse.detach(), "weighted_lpips": weighted_lpips.detach(),
            "weighted_kl": weighted_kl.detach(), "weighted_commit": weighted_commit.detach(),
            "weighted_gan": weighted_gan.detach(), "weighted_fm": weighted_fm.detach(),
            "kl_beta": torch.tensor(kl_beta, device=x_batch_chw.device).detach(),
        }
        return total, loss_dict

    # ------------------------------------------------------------------------------------------------------
    # training and use
    # ------------------------------------------------------------------------------------------------------
    def train(self, mode: bool = True, checkpoint=None):
        """``nn.Module.train(mode)``; with ``mode=True`` it then trains the model
        (:func:`gpvae_endo.training.msvq_loop.train_msvq`). `checkpoint`: optional CheckpointConfig."""
        super().train(mode)
        if mode is False:
            return self
        return train_msvq(self, checkpoint)

    def decode(self, Z: np.ndarray, x_scaled: Optional[np.ndarray] = None) -> np.ndarray:
        """Latent vectors (N, latent_dim) -> frames, flattened (N, H*W*C), through the quantiser."""
        self.eval()
        with torch.no_grad():
            z_np = torch.from_numpy(Z).float().to(self.device)
            imgs_list = []
            for i in range(0, z_np.size(0), 16):
                z_batch = z_np[i:i + 16]
                z_q_st, _, _ = self.quantize(self.latent_to_spatial(z_batch))
                x_hat = self.decode_from_quantized(z_q_st)
                imgs_list.append(x_hat.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                                 .reshape(z_batch.size(0), self.img_H * self.img_W * self.img_C))
            return np.concatenate(imgs_list, axis=0)

    def image_encode(self, img: np.ndarray, batch_size=32):
        """Frames (N, H, W, C) or (H, W, C) -> (latent vectors, zeros): the encoder is deterministic."""
        self.encoder.eval()
        self.enc_proj.eval()
        self.lat_proj.eval()
        x = torch.from_numpy(img).float().to(self.device)
        if x.dim() == 3:
            if x.shape[-1] != self.img_C:
                raise ValueError(f"Expected last dim == img_C={self.img_C}, got {x.shape}")
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            if x.shape[-1] != self.img_C:
                raise ValueError(f"Expected last dim == img_C={self.img_C}, got {x.shape}")
        else:
            raise ValueError(f"Expected 3D or 4D img, got {x.shape}")
        if x.shape[1] != self.img_H or x.shape[2] != self.img_W:
            raise ValueError(f"Expected spatial ({self.img_H}, {self.img_W}), got ({x.shape[1]}, {x.shape[2]})")

        mus, stds = [], []
        with torch.no_grad():
            N = x.shape[0]
            print("Encoding batch ", end="")
            for i in range(0, N, batch_size):
                print(f" {i}", end="")
                z_vec = self.encode_to_latent(x[i:min(i + batch_size, N)].permute(0, 3, 1, 2).contiguous())
                mus.append(z_vec.cpu())
                stds.append(torch.zeros_like(z_vec).cpu())
            print("")
        return torch.cat(mus, dim=0).numpy(), torch.cat(stds, dim=0).numpy()

    # ------------------------------------------------------------------------------------------------------
    # saving and loading (the original .pth format)
    # ------------------------------------------------------------------------------------------------------
    def _quantizer_state(self) -> dict:
        return {
            "embedding": self.quantizer.embedding.detach().cpu(),
            "cluster_size": self.quantizer.cluster_size.detach().cpu(),
            "embed_avg": self.quantizer.embed_avg.detach().cpu(),
        }

    def state_dict_full(self) -> Dict[str, Any]:
        state = {
            "encoder": self.encoder.state_dict(),
            "enc_proj": self.enc_proj.state_dict(),
            "lat_proj": self.lat_proj.state_dict(),
            "lat_unproj": self.lat_unproj.state_dict(),
            "quantizer": self._quantizer_state(),
            "decoder": self.decoder.state_dict(),
            "dec_in_proj": self.dec_in_proj.state_dict(),
            "discriminator": self.discriminator.state_dict() if self.discriminator is not None else None,
            "latent_dim": self.latent_dim,
            "img_H": self.img_H,
            "img_W": self.img_W,
            "img_C": self.img_C,
            "underlying_method": self.underlying_method,
            "Z": self.Z,
            "X_mean": self.X_mean.detach().cpu(),
            "X_std": self.X_std.detach().cpu(),
            "epochs": self.epochs,
            "lr": self.lr,
            "beta_commit": self.beta_commit,
            "max_beta": self.max_beta,
        }
        for name in ("loss_total_list", "loss_mse_list", "loss_lpips_list", "loss_kl_list", "loss_gan_list",
                     "loss_fm_list", "loss_commit_list", "loss_codebook_list", "loss_adv_d_list",
                     "loss_weighted_mse_list", "loss_weighted_lpips_list", "loss_weighted_kl_list",
                     "loss_weighted_gan_list", "loss_weighted_fm_list", "loss_weighted_commit_list",
                     "loss_weighted_codebook_list"):
            state[name] = getattr(self, name)
        return state

    def _load_quantizer(self, q: dict) -> None:
        self.quantizer.embedding.copy_(q["embedding"].to(self.device))
        self.quantizer.cluster_size.copy_(q["cluster_size"].to(self.device))
        self.quantizer.embed_avg.copy_(q["embed_avg"].to(self.device))

    def load_state_dict_full(self, state: Dict[str, Any]):
        self.latent_dim = state.get("latent_dim", self.latent_dim)
        self.img_H = state.get("img_H", self.img_H)
        self.img_W = state.get("img_W", self.img_W)
        self.img_C = state.get("img_C", self.img_C)
        self.underlying_method = state.get("underlying_method", self.underlying_method)
        self.epochs = state.get("epochs", self.epochs)
        self.lr = state.get("lr", self.lr)
        self.beta_commit = state.get("beta_commit", self.beta_commit)
        self.max_beta = state.get("max_beta", self.max_beta)

        self.X_mean = state["X_mean"].to(self.device)
        self.X_std = state["X_std"].to(self.device)
        self.Xtensor_norm = (self.Xtensor - self.X_mean) / (self.X_std + 1e-6)

        self.encoder.load_state_dict(state["encoder"])
        self.enc_proj.load_state_dict(state["enc_proj"])
        self.lat_proj.load_state_dict(state["lat_proj"])
        self.lat_unproj.load_state_dict(state["lat_unproj"])
        self.decoder.load_state_dict(state["decoder"])
        self.dec_in_proj.load_state_dict(state["dec_in_proj"])
        self._load_quantizer(state["quantizer"])
        if self.use_decoder_gan and self.discriminator is not None and state.get("discriminator") is not None:
            self.discriminator.load_state_dict(state["discriminator"])
        self.Z = state.get("Z", None)
        for name in MSVQ_HISTORY_NAMES + ("loss_weighted_codebook_list",):
            setattr(self, name, state.get(name, []))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict_full(), path)
        print(f"Saved {self.underlying_method} to {path}")

    # -- checkpoint support -------------------------------------------------------------------------------
    def checkpoint_state(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "lat_proj": self.lat_proj.state_dict(),
            "lat_unproj": self.lat_unproj.state_dict(),
            "quantizer": self._quantizer_state(),
            "decoder": self.decoder.state_dict(),
            "discriminator": self.discriminator.state_dict() if self.discriminator is not None else None,
            "X_mean": self.X_mean.detach().cpu(),
            "X_std": self.X_std.detach().cpu(),
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.lat_proj.load_state_dict(state["lat_proj"])
        self.lat_unproj.load_state_dict(state["lat_unproj"])
        with torch.no_grad():
            self._load_quantizer(state["quantizer"])
        self.decoder.load_state_dict(state["decoder"])
        if state.get("discriminator") is not None:
            self.discriminator.load_state_dict(state["discriminator"])
        self.X_mean = state["X_mean"].to(self.device)
        self.X_std = state["X_std"].to(self.device)
        self.Xtensor_norm = (self.Xtensor - self.X_mean) / (self.X_std + 1e-6)
