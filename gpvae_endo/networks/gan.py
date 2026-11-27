"""Patch discriminators and GAN losses for the decoder GAN.

``PatchDiscriminator`` / ``PatchDiscriminatorWithFeatures``: image (B, C, H, W) -> patch logits (the second variant also
returns intermediate features for feature matching). Plus hinge / BCE losses and the feature-matching loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#Endoscopy papers and reviews show GANs are useful when realism of texture and lesion appearance matters, including colonoscopic synthesis and augmentation,
# but they also warn that GANs can hallucinate structures if adversarial pressure is too strong. That means your adversarial weight should start small,
# and your discriminator should be regularized and warmed up carefully rather than dominating the VAE objective.
# For your current codebase, this is especially sensible because your decoder already outputs images directly and your reconstruction loss already includes LPIPS.
# The GAN should therefore be treated as a texture realism corrector, not as the primary reconstruction driver.



def _get_norm(norm_type: str, num_channels: int):
    norm_type = (norm_type or "instance").lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == "group":
        groups = min(32, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for endoscopic RGB images.
    Input:  (B, C, H, W)
    Output: (B, 1, H_patch, W_patch)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        norm_type: str = "instance",
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_channels
        for n in range(1, num_layers):
            out_ch = min(base_channels * (2 ** n), 512)
            stride = 1 if n == num_layers - 1 else 2
            layers += [
                nn.Conv2d(ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
                _get_norm(norm_type, out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = out_ch

        layers += [
            nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminatorWithFeatures(nn.Module):
    """
    Same as PatchDiscriminator, but returns intermediate features too.
    Useful if you later want feature matching.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        norm_type: str = "instance",
    ):
        super().__init__()

        blocks = []
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        ch = base_channels
        for n in range(1, num_layers):
            out_ch = min(base_channels * (2 ** n), 512)
            stride = 1 if n == num_layers - 1 else 2
            blocks.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
                _get_norm(norm_type, out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x, return_features=False):
        feats = []
        for block in self.blocks:
            x = block(x)
            if return_features:
                feats.append(x)
        logits = self.final(x)
        if return_features:
            return logits, feats
        return logits


def discriminator_hinge_loss(real_logits, fake_logits):
    loss_real = torch.mean(F.relu(1.0 - real_logits))
    loss_fake = torch.mean(F.relu(1.0 + fake_logits))
    return loss_real + loss_fake


def generator_hinge_loss(fake_logits):
    return -fake_logits.mean()


def discriminator_bce_loss(real_logits, fake_logits, smoothing=0.0):
    real_target = torch.ones_like(real_logits) * (1.0 - smoothing)
    fake_target = torch.zeros_like(fake_logits)
    loss_real = F.binary_cross_entropy_with_logits(real_logits, real_target)
    loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_target)
    return loss_real + loss_fake


def generator_bce_loss(fake_logits):
    real_target = torch.ones_like(fake_logits)
    return F.binary_cross_entropy_with_logits(fake_logits, real_target)


def feature_matching_loss(real_feats, fake_feats):
    loss = 0.0
    for fr, ff in zip(real_feats, fake_feats):
        loss = loss + F.l1_loss(ff, fr.detach())
    return loss