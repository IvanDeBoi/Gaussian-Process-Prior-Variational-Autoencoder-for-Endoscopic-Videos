"""Optional decoder GAN (a PatchGAN discriminator on the reconstructions) for the VAE-family models.

Switched off in every shipped config (``MUST_USE_DECODER_GAN = False``); kept so the option still works.
"""
from __future__ import annotations

from typing import Optional

import torch

from gpvae_endo.networks.gan import (
    discriminator_bce_loss,
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_bce_loss,
    generator_hinge_loss,
)


class AdversarialMixin:
    """Discriminator step and generator-side adversarial loss. Expects the attributes set in
    ``VAEModel.__init__`` (``discriminator``, ``disc_optimizer``, ``gan_*``)."""

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
        m = m.to(x.device, dtype=x.dtype)
        return x * m

    def _get_discriminator_losses(self, real_logits, fake_logits):
        if self.gan_loss_type.lower() == "hinge":
            d_loss = discriminator_hinge_loss(real_logits, fake_logits)
            g_adv = generator_hinge_loss(fake_logits)
        elif self.gan_loss_type.lower() == "bce":
            d_loss = discriminator_bce_loss(real_logits, fake_logits, smoothing=self.gan_label_smoothing)
            g_adv = generator_bce_loss(fake_logits)
        else:
            raise ValueError(f"Unsupported GAN_LOSS: {self.gan_loss_type}")
        return d_loss, g_adv

    def _train_discriminator_step(self, real_hwc, fake_hwc, mask=None):
        real_hwc = self._apply_gan_mask(real_hwc, mask)
        fake_hwc = self._apply_gan_mask(fake_hwc, mask)
        real_chw = self._hwc_to_chw(real_hwc)
        fake_chw = self._hwc_to_chw(fake_hwc)
        if self.gan_detach_recon_for_d:
            fake_chw = fake_chw.detach()

        real_logits = self.discriminator(real_chw)
        fake_logits = self.discriminator(fake_chw)
        d_loss, _ = self._get_discriminator_losses(real_logits, fake_logits)
        d_loss = self.gan_disc_weight * d_loss

        self.disc_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.disc_optimizer.step()
        return d_loss.detach()

    def _generator_adversarial_loss(self, real_hwc, fake_hwc, mask=None):
        real_hwc = self._apply_gan_mask(real_hwc, mask)
        fake_hwc = self._apply_gan_mask(fake_hwc, mask)
        real_chw = self._hwc_to_chw(real_hwc)
        fake_chw = self._hwc_to_chw(fake_hwc)

        fake_logits, fake_feats = self.discriminator(fake_chw, return_features=True)
        g_adv = (
            generator_hinge_loss(fake_logits)
            if self.gan_loss_type.lower() == "hinge"
            else generator_bce_loss(fake_logits)
        )
        if self.gan_feature_matching_weight > 0:
            with torch.no_grad():
                _, real_feats = self.discriminator(real_chw, return_features=True)
            fm = feature_matching_loss(real_feats, fake_feats)
        else:
            fm = torch.tensor(0.0, device=fake_hwc.device, dtype=fake_hwc.dtype)
        return g_adv, fm
