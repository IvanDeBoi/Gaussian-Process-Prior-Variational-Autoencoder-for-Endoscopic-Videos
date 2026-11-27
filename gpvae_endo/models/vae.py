"""VAE with a standard Gaussian prior p(z) = N(0, I) -- the base class of every GP-VAE model.

The model turns each video frame into a latent vector (encoder), and a latent vector back into a frame
(decoder). Training minimises

    loss = w_mse * masked MSE + w_lpips * LPIPS + beta * w_kl * KL (+ optional GAN terms)

where beta is 0 during the LR warm-up and then ramps up to ``max_beta``. The GP models
(:mod:`gpvae_endo.models.gpvae_casale`, ``..._hpa``, ``..._spa``) only replace the KL term.

The training loop itself is in :mod:`gpvae_endo.training.loop`; ``model.train()`` runs it.
Everything here is the original ``Models/VAE_model.py`` with the settings passed in explicitly.
"""
from __future__ import annotations

import os
import warnings
from contextlib import nullcontext
from typing import Optional

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as sched

from gpvae_endo.config import ModelSettings
from gpvae_endo.models.adversarial import AdversarialMixin
from gpvae_endo.networks import (
    CNNDecoder,
    CNNDecoderResidual,
    CNNEncoder,
    EndoVAEDecoder,
    EndoVAEEncoder,
    GastronetViTEncoder,
    GastronetViTEncoderLoRABackbone,
)
from gpvae_endo.networks.gan import PatchDiscriminatorWithFeatures
from gpvae_endo.training.loop import LOSS_HISTORY_NAMES, train_vae
from gpvae_endo.utils.device import resolve_device

warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")  # torchvision, from lpips


def build_encoder(name: str, latent_dim: int, img_C: int, img_H: int, settings: ModelSettings) -> nn.Module:
    """Encoder by name (same construction as the original, so the same random initial weights)."""
    if name == "CNNEnc":
        return CNNEncoder(latent_dim=latent_dim, in_channels=img_C, in_size=img_H,
                          min_spatial=settings.min_spatial)
    if name == "GastroEnc":  # frozen DINOv2 ViT-B/14 at 224 px + trainable MLP head
        return GastronetViTEncoder(latent_dim=latent_dim, in_channels=3, in_size=img_H, train_backbone=False)
    if name == "GaLoRaEnc":  # DINOv2 ViT at 336 px with trainable rank-r LoRA adapters + MLP head
        return GastronetViTEncoderLoRABackbone(
            latent_dim=latent_dim, in_channels=img_C, in_size=img_H, train_backbone=False,
            use_lora_backbone=True, lora_rank=settings.lora_rank, lora_alpha=settings.lora_alpha,
            lora_target_keywords=settings.lora_target_keywords,
        )
    if name == "EndoVAEEnc":  # convolutional encoder
        return EndoVAEEncoder(latent_dim=latent_dim, in_channels=img_C, in_size=img_H)
    raise ValueError(f"Unknown encoder: {name}")


def build_decoder(name: str, latent_dim: int, img_C: int, img_H: int, settings: ModelSettings) -> nn.Module:
    """Decoder by name."""
    if name == "CNNDec":
        return CNNDecoder(latent_dim=latent_dim, out_channels=img_C, out_size=img_H,
                          min_spatial=settings.min_spatial)
    if name == "CNNDecRes":
        return CNNDecoderResidual(latent_dim=latent_dim, out_channels=img_C, out_size=img_H,
                                  min_spatial=settings.min_spatial)
    if name == "EndoVAEDec":
        return EndoVAEDecoder(latent_dim=latent_dim, out_channels=img_C, out_size=img_H)
    raise ValueError(f"Unknown decoder: {name}")


class VAEModel(AdversarialMixin, nn.Module):
    """Plain VAE with prior N(0, I).

    Args:
        x: frame numbers of the training frames, (N, 1).
        Y: training frames, flattened, (N, H*W*C).
        M: valid-pixel masks, (N, H*W), or None.
        settings: the training details (:class:`gpvae_endo.config.ModelSettings`).
        device: "gpu" uses CUDA when available, else the CPU (as in the original).
    """

    def __init__(
        self,
        x: np.ndarray,
        Y: np.ndarray,
        M: np.ndarray,
        device: str = "gpu",
        latent_dim: int = 6,
        epochs: int = 10000,
        lr: float = 1e-3,
        max_beta: float = 1e-2,
        img_H: int = 32,
        img_W: Optional[int] = None,
        img_C: int = 3,
        underlying_method: str = "VAE",
        batch_size: int = 16,
        encoder: str = "not set yet",
        decoder: str = "not set yet",
        *,
        settings: ModelSettings,
        **kwargs,
    ):
        super().__init__()
        print(
            f"Initializing {underlying_method}"
            f"\nY: {Y.shape} -> Z: ({Y.shape[0]}, {latent_dim}) -> Y: {Y.shape}"
        )
        print(
            f"max_beta={max_beta}, epochs={epochs}, batch_size={batch_size}, lr={lr}, "
            f"encoder={encoder}, decoder={decoder}, device: {device}, "
            f"img_H={img_H}, img_W={img_W}, img_C={img_C}"
        )
        self.settings = settings
        self.x = x
        self.Y = Y
        self.M = M
        self.latent_dim = latent_dim
        self.underlying_method = underlying_method
        self.batch_size = batch_size
        self.device = resolve_device(device)  # "gpu" without a GPU stops unless GPVAE_ALLOW_CPU=1

        self.img_H = img_H
        self.img_W = img_W or img_H
        self.img_C = img_C

        # --- data as tensors (frames stay on the CPU; batches are moved to the device) ---
        N, D = Y.shape
        assert D == self.img_H * self.img_W * img_C, (
            f"Y.shape[1]={D} != img_C*self.img_H*self.img_W={self.img_H * self.img_W * img_C}"
        )
        self.Ytensor = torch.from_numpy(Y.reshape(N, self.img_H, self.img_W, img_C)).float()
        self.model_dtype = self.Ytensor.dtype
        print(self.Ytensor.shape)
        self.Mtensor = self._mask_tensor(M, N, img_C)

        self.Xtensor = torch.from_numpy(x).float().to(self.device)
        if self.Xtensor.dim() == 1:
            self.Xtensor = self.Xtensor.unsqueeze(-1)
        self.X_mean = torch.zeros_like(self.Xtensor[0:1])
        self.X_std = torch.ones_like(self.Xtensor[0:1])
        self.Xtensor_norm = (self.Xtensor - self.X_mean) / (self.X_std + 1e-6)

        # --- optimisation settings ---
        self.epochs = epochs
        self.lr = lr
        self.max_beta = max_beta
        self.encoder_head_lr = kwargs.get("encoder_head_lr", self.lr)
        self.decoder_lr = kwargs.get("decoder_lr", self.lr)
        self.lora_lr = kwargs.get("lora_lr", settings.lora_lr if encoder == "GaLoRaEnc" else self.lr)

        # --- networks (construction order = order of the random initial weights) ---
        self.encoder = build_encoder(encoder, latent_dim, img_C, self.img_H, settings).to(self.device)
        self.encoder = self.encoder.to(device=self.device, dtype=self.model_dtype)
        self.decoder = build_decoder(decoder, latent_dim, img_C, self.img_H, settings).to(self.device)
        self.decoder = self.decoder.to(device=self.device, dtype=self.model_dtype)

        self.optimizer = torch.optim.Adam(
            self.get_trainable_encoder_params(enc_lr=self.encoder_head_lr, lora_lr=self.lora_lr)
            + [{"params": self.decoder.parameters(), "lr": self.decoder_lr}],
            lr=self.lr,
        )
        self.warmup_epochs = settings.warmup_epochs
        self.ramp_epochs = settings.ramp_epochs
        self.total_epochs = self.epochs
        self.early_stopping_patience = settings.early_stopping_patience
        self.early_stopping_start_fraction = settings.early_stopping_start_fraction
        self._build_schedulers()

        self.Z = None
        self.mu_all = None

        # --- frozen ViT: run the backbone once over all frames and cache its features ---
        if isinstance(self.encoder, GastronetViTEncoderLoRABackbone):
            train_backbone = getattr(self.encoder, "train_backbone", False)
            use_lora_backbone = getattr(self.encoder, "use_lora_backbone", False)
            self.has_frozen_vit_encoder = (not train_backbone) and (not use_lora_backbone)
        elif isinstance(self.encoder, GastronetViTEncoder):
            self.has_frozen_vit_encoder = not getattr(self.encoder, "train_backbone", False)
        else:
            self.has_frozen_vit_encoder = False
        self.backbone_feats = self._precompute_backbone_features() if self.has_frozen_vit_encoder else None

        # --- perceptual loss ---
        self.lpips_net = lpips.LPIPS(net="vgg").to(self.device)
        self.lpips_net.eval()
        for p in self.lpips_net.parameters():
            p.requires_grad = False
        self.loss_weights = settings.loss_weights
        self.lpips_target_size = settings.lpips_res
        self.lpips_batch_size = kwargs.get(
            "lpips_batch_size",
            min(self.batch_size, 8 if self.lpips_target_size >= 256 else self.batch_size),
        )
        print("Loss weights:")
        for k, v in self.loss_weights.items():
            print(f"  {k:>5}: {v}")
        print(f"LPIPS target size: {self.lpips_target_size} (not full res for performance reasons)")

        self._init_gan(settings)

        for name in LOSS_HISTORY_NAMES:
            setattr(self, name, [])
        print(f"{underlying_method} initialized successfully")

    # ------------------------------------------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------------------------------------------
    def _mask_tensor(self, M, N, img_C):
        if M is None:
            return None
        if M.shape[1] == self.img_H * self.img_W:
            M_imgs = M.reshape(N, self.img_H, self.img_W, 1)
        elif M.shape[1] == self.img_H * self.img_W * img_C:
            M_imgs = M.reshape(N, self.img_H, self.img_W, img_C)
        else:
            raise ValueError(f"Unexpected M shape {M.shape}, expected (N,H*W) or (N,H*W*C)")
        return torch.from_numpy(M_imgs).float()

    def _build_schedulers(self):
        """Linear LR warm-up (x0.1 -> x1) for `warmup_epochs`, then cosine decay to 1e-5."""
        self.warmup_scheduler = sched.LinearLR(self.optimizer, start_factor=0.1, total_iters=self.warmup_epochs)
        self.main_scheduler = sched.CosineAnnealingLR(
            self.optimizer, T_max=self.total_epochs - self.warmup_epochs, eta_min=1e-5
        )

    def _precompute_backbone_features(self):
        print("Precalculating backbone features...")
        encoder_mod = self.encoder
        encoder_mod.eval()
        Yfull = self.Ytensor
        N = Yfull.size(0)
        bs = 128
        all_feats = []
        print("Processing ", end="")
        for i in range(0, N, bs):
            print(f" {i}", end="")
            with torch.no_grad():
                y_batch = self._batch_to_device(Yfull[i:i + bs])
                _, h, w, c = y_batch.shape
                x_hwc = y_batch.view(-1, h, w, c)
                x_chw = x_hwc.permute(0, 3, 1, 2).contiguous().to(self.device)
                feats = encoder_mod.extract_backbone_features(x_chw)
            all_feats.append(feats.cpu())
            del y_batch, x_hwc, x_chw, feats
        backbone_feats = torch.cat(all_feats, dim=0).to(self.device)
        print(" done")
        assert backbone_feats.shape[0] == N
        return backbone_feats

    def _init_gan(self, s: ModelSettings):
        self.use_decoder_gan = s.must_use_decoder_gan
        self.gan_type = s.gan_type
        self.gan_loss_type = s.gan_loss
        self.gan_weight = self.loss_weights.get("gan", 0.0)
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
                in_channels=self.img_C, base_channels=s.gan_base_channels,
                num_layers=s.gan_num_layers, norm_type=s.gan_norm,
            ).to(device=self.device, dtype=self.model_dtype)
            self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=s.gan_lr, betas=s.gan_betas)
            print(
                f"Initialized decoder GAN: type={self.gan_type}, "
                f"loss={self.gan_loss_type}, weight={self.gan_weight}, D_lr={s.gan_lr}"
            )
        else:
            print("No decoder GAN")

    def get_trainable_encoder_params(self, enc_lr: float = None, lora_lr: float = None):
        """Adam parameter groups for the encoder. ViT encoders: only the head (and LoRA adapters) train."""
        if hasattr(self.encoder, "head"):
            if getattr(self.encoder, "train_backbone", False):
                return [{"params": self.encoder.parameters(), "lr": enc_lr or self.lr}]
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.encoder.head.parameters():
                p.requires_grad = True
            param_groups = [{"params": list(self.encoder.head.parameters()), "lr": enc_lr or self.lr}]
            if getattr(self.encoder, "use_lora_backbone", False):
                lora_params = []
                seen = set()
                for name, p in self.encoder.named_parameters():
                    if "lora_" not in name:
                        continue
                    p.requires_grad = True
                    if id(p) not in seen:
                        lora_params.append(p)
                        seen.add(id(p))
                if lora_params:
                    param_groups.append({"params": lora_params, "lr": lora_lr or self.lr})
            return param_groups
        return [{"params": self.encoder.parameters(), "lr": enc_lr or self.lr}]

    # ------------------------------------------------------------------------------------------------------
    # hooks the training loop calls (overridden by the GP models)
    # ------------------------------------------------------------------------------------------------------
    def refresh_epoch_state(self):
        """Called at the start of every epoch. Plain VAE: nothing (HPA/SPA re-encode all frames)."""
        return

    def update_lr(self, epoch):
        if epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def standard_normal_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(mu, sigma^2) || N(0, I)), summed over latent dims, averaged over the batch."""
        logvar_clamped = torch.clamp(logvar, -10.0, 10.0)
        var = torch.exp(logvar_clamped)
        kl = 0.5 * torch.sum(mu.pow(2) + var - 1.0 - logvar_clamped, dim=1)
        return kl.mean()

    def compute_kl_term(self, mu, logvar, x_batch=None, mask=None, Xb_norm=None, batch_idx=None, **kwargs):
        return self.standard_normal_kl(mu, logvar)

    def get_generator_parameters_for_clipping(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_extra_log_message(self) -> str:
        return ""

    def uses_vit_cache(self) -> bool:
        return (
            self.backbone_feats is not None
            and self.has_frozen_vit_encoder
            and hasattr(self.encoder, "encode_from_feats_batched")
        )

    # ------------------------------------------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------------------------------------------
    def _batch_to_device(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.device == self.device:
            return tensor
        return tensor.to(self.device, non_blocking=self.device.type == "cuda")

    def _encode_ytensor_in_batches(self, batch_size=None, use_no_grad=False, move_to_cpu=False):
        """Encode every training frame; returns (mu, logvar) for all frames."""
        batch_size = batch_size or self.batch_size
        use_vit_cache = self.uses_vit_cache()
        mu_parts, logvar_parts = [], []
        grad_context = torch.no_grad() if use_no_grad else nullcontext()
        with grad_context:
            n_samples = self.Ytensor.size(0)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                if use_vit_cache:
                    feats_batch = self.backbone_feats[start:end]
                    mu_b, logvar_b = self.encoder.encode_from_feats_batched(feats_batch, b=feats_batch.size(0), t=1)
                else:
                    y_batch = self._batch_to_device(self.Ytensor[start:end])
                    mu_b, logvar_b = self.encoder(y_batch)
                    del y_batch
                if move_to_cpu:
                    mu_parts.append(mu_b.detach().cpu())
                    logvar_parts.append(logvar_b.detach().cpu())
                else:
                    mu_parts.append(mu_b.detach() if use_no_grad else mu_b)
                    logvar_parts.append(logvar_b.detach() if use_no_grad else logvar_b)
                del mu_b, logvar_b
        return torch.cat(mu_parts, dim=0), torch.cat(logvar_parts, dim=0)

    def _compute_lpips_chunked(self, recon_ds: torch.Tensor, x_ds: torch.Tensor, chunk_size: int) -> torch.Tensor:
        total = torch.zeros((), device=recon_ds.device, dtype=recon_ds.dtype)
        total_count = 0
        for start in range(0, recon_ds.size(0), chunk_size):
            end = min(start + chunk_size, recon_ds.size(0))
            # tanh of standardised pixels, as in the original (LPIPS expects RGB in [-1, 1]).
            recon_lp = torch.tanh(recon_ds[start:end])
            with torch.no_grad():
                x_lp = torch.tanh(x_ds[start:end])
            lpips_chunk = self.lpips_net(recon_lp, x_lp).mean()
            chunk_count = end - start
            total = total + lpips_chunk * chunk_count
            total_count += chunk_count
        return total / max(total_count, 1)

    def _compute_lpips_value(self, recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """LPIPS on frames resized to `lpips_target_size`, in chunks; halves the chunk on CUDA out-of-memory."""
        x_ds = F.interpolate(x.permute(0, 3, 1, 2), size=self.lpips_target_size, mode="bilinear", align_corners=False)
        recon_ds = F.interpolate(recon.permute(0, 3, 1, 2), size=self.lpips_target_size, mode="bilinear",
                                 align_corners=False)
        chunk_size = max(1, min(self.lpips_batch_size, recon_ds.size(0)))
        while True:
            try:
                return self._compute_lpips_chunked(recon_ds, x_ds, chunk_size)
            except torch.OutOfMemoryError:
                if self.device.type != "cuda" or chunk_size == 1:
                    raise
                new_chunk_size = max(1, chunk_size // 2)
                if new_chunk_size == chunk_size:
                    raise
                print(f"[LPIPS] CUDA OOM with chunk size {chunk_size}; retrying with {new_chunk_size}")
                torch.cuda.empty_cache()
                chunk_size = new_chunk_size

    def compute_total_loss(self, recon, x, mu, logvar, mask=None, kl_beta: float = 1.0, use_gan: bool = True,
                           **kwargs):
        """Weighted sum of masked MSE, LPIPS, KL (times `kl_beta`) and the optional GAN terms."""
        zero = lambda: torch.tensor(0.0, device=x.device, dtype=x.dtype)  # noqa: E731
        if self.loss_weights["mse"] > 0:
            if mask is not None:
                m = mask
                while m.dim() < recon.dim():
                    m = m.unsqueeze(-1)
                m = m.to(recon.device, dtype=recon.dtype)
                se = (recon - x) ** 2
                se = se * m
                valid = m.sum()
                mse = se.sum() / torch.clamp_min(valid, 1.0)
            else:
                mse = F.mse_loss(recon, x)
        else:
            mse = zero()

        lpips_val = self._compute_lpips_value(recon, x) if self.loss_weights["lpips"] > 0 else zero()

        effective_kl_weight = self.loss_weights["kl"] * kl_beta
        if effective_kl_weight > 0:
            kl = self.compute_kl_term(mu=mu, logvar=logvar, x_batch=x, mask=mask, **kwargs)
        else:
            kl = zero()

        use_disc_based_generator_loss = (
            use_gan and self.use_decoder_gan and self.discriminator is not None
            and (self.loss_weights["gan"] > 0 or self.gan_feature_matching_weight > 0)
        )
        if use_disc_based_generator_loss:
            gan, fm = self._generator_adversarial_loss(real_hwc=x, fake_hwc=recon, mask=mask)
        else:
            gan, fm = zero(), zero()

        weighted_mse = self.loss_weights["mse"] * mse
        weighted_lpips = self.loss_weights["lpips"] * lpips_val
        weighted_gan = self.loss_weights["gan"] * gan
        weighted_kl = effective_kl_weight * kl
        weighted_fm = self.gan_feature_matching_weight * fm
        total_loss = weighted_mse + weighted_lpips + weighted_gan + weighted_kl + weighted_fm

        if not torch.isfinite(total_loss):
            print(
                "[compute_total_loss] NON-FINITE loss! "
                f"weighted_mse={weighted_mse.item():.6f}, "
                f"weighted_lpips={weighted_lpips.item():.6f}, "
                f"weighted_gan={weighted_gan.item():.6f}, "
                f"weighted_kl={weighted_kl.item():.6f}, "
                f"weighted_fm={weighted_fm.item():.6f}"
            )
        loss_dict = {
            "total": total_loss.detach(), "mse": mse.detach(), "lpips": lpips_val.detach(), "kl": kl.detach(),
            "gan": gan.detach(), "fm": fm.detach(), "weighted_mse": weighted_mse.detach(),
            "weighted_lpips": weighted_lpips.detach(), "weighted_kl": weighted_kl.detach(),
            "weighted_gan": weighted_gan.detach(), "weighted_fm": weighted_fm.detach(),
            "kl_beta": torch.tensor(kl_beta, device=x.device).detach(),
        }
        return total_loss, loss_dict

    # ------------------------------------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------------------------------------
    def train(self, checkpoint=None):
        """Train the model (runs :func:`gpvae_endo.training.loop.train_vae`).

        Note: this replaces ``nn.Module.train(mode)``, as in the original; the LPIPS network must stay in
        eval mode. `checkpoint`: optional :class:`gpvae_endo.training.checkpoint.CheckpointConfig`.
        """
        return train_vae(self, checkpoint)

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    # ------------------------------------------------------------------------------------------------------
    # using a trained model
    # ------------------------------------------------------------------------------------------------------
    def decode(self, Z: np.ndarray, xscaled: Optional[np.ndarray] = None) -> np.ndarray:
        """Latents (N, Q) -> frames, flattened (N, H*W*C)."""
        self.decoder.eval()
        with torch.no_grad():
            ztensor = torch.from_numpy(Z).float().to(self.device)
            N = ztensor.size(0)
            batch_size = 16
            decoded = []
            for i in range(0, N, batch_size):
                logits = self.decoder(ztensor[i:i + batch_size])
                decoded.append(logits.view(-1, self.img_H * self.img_W * self.img_C).cpu().numpy())
            imgs_np = np.concatenate(decoded, axis=0)
        return imgs_np

    def image_encode(self, img: np.ndarray, batch_size=32):
        """Frames (N, H, W, C) or one frame (H, W, C) -> latent mean and std, each (N, Q)."""
        self.encoder.eval()
        x = torch.from_numpy(img).float()
        if x.dim() == 3:
            if x.shape[-1] != self.img_C:
                raise ValueError(f"Expected last dim == imgC={self.img_C}, got {x.shape}")
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            if x.shape[-1] != self.img_C:
                raise ValueError(f"Expected last dim == imgC={self.img_C}, got {x.shape}")
        else:
            raise ValueError(f"Expected 3D or 4D img, got {x.shape}")
        if x.shape[1] != self.img_H or x.shape[2] != self.img_W:
            raise ValueError(f"Expected spatial ({self.img_H}, {self.img_W}), got ({x.shape[1]}, {x.shape[2]})")

        device = next(self.encoder.parameters()).device
        mus, stds = [], []
        with torch.no_grad():
            N = x.shape[0]
            print("Encoding batch ", end="")
            for i in range(0, N, batch_size):
                print(f" {i}", end="")
                end = min(i + batch_size, N)
                x_batch = x[i:end].to(device, non_blocking=device.type == "cuda")
                mu_b, logvar_b = self.encoder(x_batch)
                std_b = torch.exp(0.5 * logvar_b)
                mus.append(mu_b.cpu())
                stds.append(std_b.cpu())
                del x_batch, mu_b, logvar_b, std_b
            print("")
        return torch.cat(mus, dim=0).numpy(), torch.cat(stds, dim=0).numpy()

    def get_middle_covariance(self, x_eval=None, lengthscale: float = 1.0, scale: float = 1.0, subsample: int = 200):
        """A stand-in 'prior covariance' for the VAE plot: RBF in time times the latents' average variance."""
        self.eval()
        with torch.no_grad():
            if x_eval is None:
                N = self.x.shape[0]
                k = min(subsample, N)
                rand_idx = np.random.choice(N, size=k, replace=False)
                rand_idx.sort()
                x_eval = self.x[rand_idx]
            x_t = torch.tensor(x_eval).float().to(self.device)
            if x_t.dim() == 1:
                x_t.unsqueeze_(-1)
            x_norm = (x_t - self.X_mean) / (self.X_std + 1e-6)
            if self.Z is None:
                mu_all, _ = self._encode_ytensor_in_batches(use_no_grad=True)
                self.Z = mu_all.detach().cpu().numpy()
            Z_torch = torch.from_numpy(self.Z).float().to(self.device)
            d2 = (x_norm[:, 0:1] - self.Xtensor_norm.T) ** 2
            Z_eval = Z_torch[torch.argmin(d2, dim=1)]
            Z_centered = Z_eval - Z_eval.mean(dim=0, keepdim=True)
            Sigma_latent = (Z_centered.T @ Z_centered) / (Z_eval.size(0) - 1 + 1e-6)
            diff = x_norm - x_norm.T
            K_time = scale * torch.exp(-0.5 * (diff ** 2) / (lengthscale ** 2))
            cov_mat = (torch.trace(Sigma_latent) / Z_eval.size(1)) * K_time
            return cov_mat.cpu().numpy()

    # ------------------------------------------------------------------------------------------------------
    # saving and loading (the .pth format of the original, so old files load)
    # ------------------------------------------------------------------------------------------------------
    def state_dict_full(self):
        state = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "gp_prior": None,
            "latent_dim": self.latent_dim,
            "img_H": self.img_H,
            "img_W": self.img_W,
            "img_C": self.img_C,
            "kernel_gp_prior": None,
            "underlying_method": self.underlying_method,
            "Z": self.Z if self.Z is not None else None,
            "X_mean": self.X_mean.detach().cpu(),
            "X_std": self.X_std.detach().cpu(),
            "epochs": self.epochs,
            "lr": self.lr,
            "max_beta": self.max_beta,
            "use_decoder_gan": self.use_decoder_gan,
            "discriminator": self.discriminator.state_dict() if self.discriminator is not None else None,
            "loss_adv_d_list": self.loss_adv_d_list,
        }
        for name in LOSS_HISTORY_NAMES[:-1]:
            state[name] = getattr(self, name)
        return state

    def load_state_dict_full(self, state):
        self.latent_dim = state.get("latent_dim", self.latent_dim)
        self.img_H = state.get("img_H", self.img_H)
        self.img_W = state.get("img_W", self.img_W)
        self.img_C = state.get("img_C", self.img_C)
        self.underlying_method = state.get("underlying_method", self.underlying_method)
        self.epochs = state.get("epochs", self.epochs)
        self.lr = state.get("lr", self.lr)
        self.max_beta = state.get("max_beta", self.max_beta)

        self.X_mean = state["X_mean"].to(self.device)
        self.X_std = state["X_std"].to(self.device)
        self.Xtensor_norm = (self.Xtensor - self.X_mean) / (self.X_std + 1e-6)

        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.Z = state.get("Z", None)
        for name in LOSS_HISTORY_NAMES:
            setattr(self, name, state.get(name, []))
        if self.use_decoder_gan and self.discriminator is not None and state.get("discriminator") is not None:
            self.discriminator.load_state_dict(state["discriminator"])

    _save_message = "Saved {method} {path}"

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict_full(), path)
        print(self._save_message.format(method=self.underlying_method, path=path))

    # -- checkpoint support (weights loaded into the existing modules, never new objects) ------------------
    def checkpoint_state(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "discriminator": self.discriminator.state_dict() if self.discriminator is not None else None,
            "Z": self.Z,
            "mu_all": _with_device(self.mu_all),
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        if state.get("discriminator") is not None:
            self.discriminator.load_state_dict(state["discriminator"])
        self.Z = state.get("Z")
        self.mu_all = _from_device(state.get("mu_all"))


def _with_device(t):
    """A cached tensor plus the device it lived on (the CPU or the GPU, depending on how it was made)."""
    return {"tensor": t, "device": str(t.device)} if torch.is_tensor(t) else {"tensor": t, "device": None}


def _from_device(packed):
    t = packed["tensor"]
    return t.to(packed["device"]) if torch.is_tensor(t) else t
