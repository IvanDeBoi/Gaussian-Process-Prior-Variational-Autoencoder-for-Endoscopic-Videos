"""EndoVAE-style CNN encoder and decoder for square images.

``EndoVAEEncoder``: images (B, H, W, C) or (B, T, H, W, C) -> latent mean and log-variance, (B, latent_dim) or
(B, T, latent_dim). ``EndoVAEDecoder`` mirrors it and returns images in the same layout as the encoder input.
"""
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class EndoVAEEncoder(nn.Module):
    """
    Simple EndoVAE-style CNN encoder for square images.

    Inputs
    -------
    - vid_batch: (B, H, W, C) or (B, T, H, W, C)

    Outputs
    -------
    - q_means: (B, latent_dim) or (B, T, latent_dim)
    - q_logvars: same shape
    """
    def __init__(
        self,
        latent_dim: int = 32,
        in_channels: int = 3,
        in_size: int = 336,
        base_channels: int = 32,
        max_channels: int = 256,
    ):
        super().__init__()
        print(
            f"[EndoVAEEncoder] Initializing (latent_dim={latent_dim}, "
            f"in_channels={in_channels}, in_size={in_size})"
        )

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_size = in_size

        # Build conv stack (downsample by 2 each time)
        ch = base_channels
        layers = [
            nn.Conv2d(in_channels, ch, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]

        h = in_size
        w = in_size
        print(f"[EndoVAEEncoder] Input spatial: H={h}, W={w}, C={in_channels}")

        h //= 2
        w //= 2
        print(
            f"[EndoVAEEncoder] Stem conv: {in_channels} -> {ch}, "
            f"spatial {in_size}x{in_size} -> {h}x{w}"
        )

        block_idx = 0
        print(
            f"[EndoVAEEncoder] Entering downsampling loop with "
            f"H={h}, W={w}, C={ch}, max_channels={max_channels}"
        )

        while min(h, w) > 8 and ch < max_channels:
            in_ch = ch
            next_h = h // 2
            next_w = w // 2
            ch = min(ch * 2, max_channels)

            print(
                f"[EndoVAEEncoder] Block {block_idx}: "
                f"{in_ch} -> {ch}, spatial {h}x{w} -> {next_h}x{next_w}"
            )

            layers.extend(
                [
                    nn.Conv2d(in_ch, ch, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )

            h = next_h
            w = next_w
            block_idx += 1

        print(
            f"[EndoVAEEncoder] Downsampling finished after {block_idx} loop blocks. "
            f"Final conv feature map: C={ch}, H={h}, W={w}"
        )

        if min(h, w) <= 8:
            print(
                f"[EndoVAEEncoder] Loop stop reason: spatial size reached {h}x{w}"
            )
        if ch >= max_channels:
            print(
                f"[EndoVAEEncoder] Loop stop reason: channel cap {max_channels} reached"
            )

        self.conv = nn.Sequential(*layers)

        # Dynamically compute flat_dim with a dummy forward – no hardcoding
        with torch.no_grad():
            dummy = torch.zeros(1, in_size, in_size, in_channels)
            x = dummy.permute(0, 3, 1, 2).contiguous()
            x = self.conv(x)
            flat = x.view(1, -1)
            flat_dim = flat.shape[1]

        print(
            f"[EndoVAEEncoder] Conv output shape: C={ch}, H={h}, W={w}, "
            f"flat_dim={flat_dim}"
        )

        self.fc = nn.Linear(flat_dim, 2 * latent_dim)
        print(f"[EndoVAEEncoder] Linear layer: {flat_dim} -> {2 * latent_dim}")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, vid_batch: Tensor) -> Tuple[Tensor, Tensor]:
        if vid_batch.dim() == 5:
            b, t, h, w, c = vid_batch.shape
            x = vid_batch.view(-1, h, w, c)
        else:
            b, h, w, c = vid_batch.shape
            t = 1
            x = vid_batch

        x = x.permute(0, 3, 1, 2).contiguous()  # BT, C, H, W
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # BT, flat_dim
        x = self.fc(x)  # BT, 2*latent_dim
        x = x.view(b, t, -1)

        q_means = x[..., :self.latent_dim]
        q_logvars = x[..., self.latent_dim:]

        if t == 1:
            q_means = q_means.squeeze(1)
            q_logvars = q_logvars.squeeze(1)

        return q_means, q_logvars


class EndoVAEDecoder(nn.Module):
    """
    Simple EndoVAE-style CNN decoder, mirroring EndoVAEEncoder.

    Inputs
    -------
    - z: (B, latent_dim) or (B, T, latent_dim)

    Outputs
    -------
    - x logits: (B, H, W, C) or (B, T, H, W, C)
    """
    def __init__(
        self,
        latent_dim: int = 32,
        out_channels: int = 3,
        out_size: int = 336,
        base_channels: int = 32,
        max_channels: int = 256,
    ):
        super().__init__()
        print(
            f"[EndoVAEDecoder] Initializing (latent_dim={latent_dim}, "
            f"out_channels={out_channels}, out_size={out_size})"
        )

        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.out_size = out_size

        # Reconstruct bottleneck shape in the same way as encoder
        ch = base_channels
        h = out_size
        w = out_size

        print(f"[EndoVAEDecoder] Output target spatial: H={h}, W={w}, C={out_channels}")

        h //= 2
        w //= 2
        print(
            f"[EndoVAEDecoder] Mirror stem: {out_channels} -> {ch}, "
            f"spatial {out_size}x{out_size} -> {h}x{w}"
        )

        block_idx = 0
        print(
            f"[EndoVAEDecoder] Reconstructing bottleneck with "
            f"H={h}, W={w}, C={ch}, max_channels={max_channels}"
        )

        while min(h, w) > 8 and ch < max_channels:
            in_ch = ch
            next_h = h // 2
            next_w = w // 2
            ch = min(ch * 2, max_channels)

            print(
                f"[EndoVAEDecoder] Mirror Block {block_idx}: "
                f"{in_ch} -> {ch}, spatial {h}x{w} -> {next_h}x{next_w}"
            )

            h = next_h
            w = next_w
            block_idx += 1

        self.init_channels = ch
        self.init_h = h
        self.init_w = w

        print(
            f"[EndoVAEDecoder] Bottleneck reconstructed after {block_idx} mirror blocks: "
            f"C={ch}, H={h}, W={w}"
        )

        flat_dim = ch * h * w
        self.fc = nn.Linear(latent_dim, flat_dim)
        print(f"[EndoVAEDecoder] Linear layer: {latent_dim} -> {ch}x{h}x{w}")

        # Up-sampling stack: mirror encoder but just double until >= out_size
        layers = []
        cur_ch = ch
        cur_h = h
        cur_w = w
        up_idx = 0

        print(
            f"[EndoVAEDecoder] Entering upsampling loop with "
            f"H={cur_h}, W={cur_w}, C={cur_ch}"
        )

        while max(cur_h, cur_w) < out_size:
            out_ch = max(cur_ch // 2, base_channels)

            print(
                f"[EndoVAEDecoder] Up Block {up_idx}: "
                f"{cur_ch} -> {out_ch}, spatial {cur_h}x{cur_w} -> "
                f"{cur_h * 2}x{cur_w * 2}"
            )

            layers.append(
                nn.ConvTranspose2d(
                    cur_ch, out_ch, kernel_size=4, stride=2, padding=1
                )
            )
            layers.append(nn.ReLU(inplace=True))

            cur_ch = out_ch
            cur_h *= 2
            cur_w *= 2
            up_idx += 1

        print(
            f"[EndoVAEDecoder] Upsampling finished after {up_idx} blocks. "
            f"Feature map before final conv: C={cur_ch}, H={cur_h}, W={cur_w}"
        )

        self.deconv = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(cur_ch, out_channels, kernel_size=3, padding=1)

        print(
            f"[EndoVAEDecoder] Final conv: {cur_ch} -> {out_channels}, "
            f"kernel_size=3"
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        if z.dim() == 3:
            b, t, d = z.shape
            z_flat = z.view(-1, d)
        else:
            b, d = z.shape
            t = 1
            z_flat = z

        x = self.fc(z_flat)  # BT, flat_dim
        x = x.view(-1, self.init_channels, self.init_h, self.init_w)
        x = self.deconv(x)

        h, w = x.shape[-2], x.shape[-1]
        if h != self.out_size or w != self.out_size:
            x = F.interpolate(
                x,
                size=(self.out_size, self.out_size),
                mode="bilinear",
                align_corners=False,
            )

        x = self.final_conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # BT, H, W, C

        if t > 1:
            x = x.view(b, t, x.shape[1], x.shape[2], x.shape[3])
        else:
            x = x.view(b, x.shape[1], x.shape[2], x.shape[3])

        return x
