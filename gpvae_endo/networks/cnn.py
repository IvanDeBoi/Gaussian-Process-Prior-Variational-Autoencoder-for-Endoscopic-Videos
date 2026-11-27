"""Plain CNN encoder / decoders for square images.

``CNNEncoder``: images (B, H, W, C) or (B, T, H, W, C) -> latent mean and log-variance, (B, latent_dim) or
(B, T, latent_dim). ``CNNDecoder`` and ``CNNDecoderResidual`` map latents back to images (B, H, W, C) / (B, T, H, W, C)
(logits, no activation). The number of down/upsampling steps is derived from the image size and ``min_spatial``.
"""
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from gpvae_endo.networks.blocks import ResidualBlock


class CNNEncoder(nn.Module):
    """
    CNN encoder for arbitrary square images (e.g. 32, 64, 128, 256, 320, 336, 512).

    Improvements over SimpleCNNEncoder:
      - Shared, explicitly stored spatial/channel schedule (for symmetry with decoder)
      - Slightly higher channel capacity (cap at 256)
      - Optional BatchNorm2d for better training stability
      - Extra conv refinement head at the smallest spatial scale

    Supports inputs:
      - (B, H, W, C)
      - (B, T, H, W, C) -> treated as B*T images.

    Outputs:
      - q_means:   (B, T, latent_dim) or (B, latent_dim)
      - q_logvars: same shape
    """

    def __init__(self, latent_dim: int = 6, in_channels: int = 3, in_size: int = 64,
                 min_spatial: int = 8):
        super().__init__()
        print(
            f"[CNNEncoder] Initializing (latent_dim={latent_dim}, "
            f"in_channels={in_channels}, in_size={in_size}, min spatial: {min_spatial})"
        )

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_size = in_size

        layers = []
        ch = in_channels
        h = in_size
        w = in_size
        block_idx = 0

        # Store spatial/channel schedule for decoder symmetry
        self.spatial_schedule = []

        # Downsample with stride-2 convs until spatial dims are small enough
        # (<= min_spatial) and at least 2 blocks are used.
        while (h > min_spatial or w > min_spatial) or block_idx < 2:
            # slightly higher cap than SimpleCNNEncoder
            out_ch = min(256, ch * 2)
            print(
                f"[CNNEncoder] Block {block_idx}: "
                f"in_ch={ch}, out_ch={out_ch}, input_spatial=({h}, {w})"
            )

            layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

            # record schedule: (in_ch, out_ch, h, w)
            self.spatial_schedule.append((ch, out_ch, h, w))

            ch = out_ch
            # Conv2d output size formula for kernel=3, stride=2, padding=1
            h = (h + 2 * 1 - 3) // 2 + 1
            w = (w + 2 * 1 - 3) // 2 + 1
            block_idx += 1

        self.conv_layers = nn.ModuleList(layers)

        # Small refinement head at smallest spatial scale
        self.final_conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Dynamically compute flattened conv feature size
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_size, self.in_size, in_channels)  # HWC
            x = dummy.permute(0, 3, 1, 2).contiguous()  # (1, C, H, W)
            for block in self.conv_layers:
                x = block(x)
            x = self.final_conv(x)
            flat = x.reshape(1, -1)
            self.flat_dim = flat.shape[1]
        print(f"[CNNEncoder] Conv output flattened dim = {self.flat_dim}")

        # Linear layer to produce mean and logvar parameters
        self.fnn = nn.Linear(self.flat_dim, self.latent_dim * 2)
        print(
            f"[CNNEncoder] Linear layer: "
            f"{self.flat_dim} -> {self.latent_dim * 2}"
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, vid_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        vid_batch:
          - (B, H, W, C) or
          - (B, T, H, W, C)

        Returns:
          q_means, q_logvars:
            if input dim=5: (B, T, latent_dim)
            else:           (B, latent_dim)
        """
        if vid_batch.dim() == 5:
            # (B, T, H, W, C) -> (B*T, H, W, C)
            b, t, h, w, c = vid_batch.shape
            x = vid_batch.reshape(-1, h, w, c)
        else:
            # (B, H, W, C)
            b, h, w, c = vid_batch.shape
            t = 1
            x = vid_batch

        # HWC -> CHW
        x = x.permute(0, 3, 1, 2).contiguous()  # (B*T, C, H, W)
        for block in self.conv_layers:
            x = block(x)
        x = self.final_conv(x)
        x = x.reshape(x.size(0), -1)           # (B*T, flat_dim)
        x = self.fnn(x)                        # (B*T, 2*latent_dim)
        x = x.view(b, t, -1)

        q_means = x[..., :self.latent_dim]
        q_logvars = x[..., self.latent_dim:]

        if t == 1:
            q_means = q_means.squeeze(1)
            q_logvars = q_logvars.squeeze(1)

        return q_means, q_logvars


class CNNDecoder(nn.Module):
    """
    CNN decoder mirroring CNNEncoder for arbitrary square out_size.

      - Uses a symmetric spatial schedule (same number of upsampling steps)
      - Slightly higher channel capacity (cap at 256, mirroring encoder)
      - Optional BatchNorm2d in upsampling blocks
      - Extra full-resolution refinement head

    Supports inputs:
      - z: (B, latent_dim) or (B, T, latent_dim)

    Returns:
      - x: (B, H, W, C) or (B, T, H, W, C)  (logits, no activation)
    """

    def __init__(
        self,
        latent_dim: int = 6,
        out_channels: int = 3,
        out_size: int = 64,
        min_spatial: int = 8,
    ):
        super().__init__()
        print(
            f"[CNNDecoder] Initializing (latent_dim={latent_dim}, "
            f"out_channels={out_channels}, out_size={out_size})"
        )

        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.out_size = out_size

        # Reconstruct symmetric schedule from out_size using the same logic as encoder.
        # We do NOT rely on MIN_SPATIAL alone; we explicitly build a schedule and then
        # derive start_h from it, ensuring symmetry in the number of 2x steps.
        h = out_size
        w = out_size
        ch = out_channels
        block_idx = 0
        spatial_schedule = []

        while (h > min_spatial or w > min_spatial) or block_idx < 2:
            out_ch = min(256, ch * 2)
            spatial_schedule.append((ch, out_ch, h, w))
            print(
                f"[CNNDecoder] Enc-sim Block {block_idx}: "
                f"in_ch={ch}, out_ch={out_ch}, input_spatial=({h}, {w})"
            )

            ch = out_ch
            # mirror encoder conv formula (kernel=3, stride=2, padding=1)
            h = (h + 2 * 1 - 3) // 2 + 1
            w = (w + 2 * 1 - 3) // 2 + 1
            block_idx += 1

        # This is the spatial shape the encoder would stop at for this out_size.
        self.enc_final_channels = ch
        self.enc_final_h = h
        self.enc_final_w = w
        self.spatial_schedule = spatial_schedule

        self.start_h = self.enc_final_h
        self.start_w = self.enc_final_w

        print(
            f"[CNNDecoder] Start spatial size inferred from schedule: "
            f"({self.start_h}, {self.start_w}) with {self.enc_final_channels} channels"
        )

        # FC to initial conv feature map (internal CHW)
        conv_init_dim = self.enc_final_channels * self.start_h * self.start_w
        self.fnn = nn.Linear(latent_dim, conv_init_dim)
        print(
            f"[CNNDecoder] Linear layer: {latent_dim} -> {conv_init_dim}"
        )

        # Build deconv stack: Upsample + Conv2d blocks reversing the schedule.
        layers = []
        in_c = self.enc_final_channels
        # reverse schedule, but we only care about channels, not exact intermediate H/W
        for i, (enc_in_c, enc_out_c, h_enc, w_enc) in enumerate(reversed(self.spatial_schedule)):
            out_c = enc_in_c if i < len(self.spatial_schedule) - 1 else out_channels
            print(
                f"[CNNDecoder] Up Block {i}: in_ch={in_c}, out_ch={out_c}"
            )
            is_last = (i == len(self.spatial_schedule) - 1)

            block = [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            ]
            if not is_last:
                block.append(nn.BatchNorm2d(out_c))
                block.append(nn.LeakyReLU(0.2, inplace=True))

            layers.append(nn.Sequential(*block))
            in_c = out_c

        self.deconv_layers = nn.Sequential(*layers)
        print(
            f"[CNNDecoder] Deconv stack created with "
            f"{len(self.spatial_schedule)} upsample blocks."
        )

        # Full-resolution refinement head
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        """
        z:
          - (B, latent_dim) or
          - (B, T, latent_dim)

        Returns:
          x:
          - (B, H, W, C) or
          - (B, T, H, W, C), logits
        """
        if z.dim() == 3:
            # (B, T, D)
            b, t, d = z.shape
            z_flat = z.reshape(-1, d)  # (B*T, D)
        else:
            # (B, D)
            b, d = z.shape
            t = 1
            z_flat = z

        x = self.fnn(z_flat)  # (B*T, C_init*H_init*W_init)
        x = x.view(-1, self.enc_final_channels, self.start_h, self.start_w)  # (B*T, C, H, W)
        x = self.deconv_layers(x)  # (B*T, C, H_dec, W_dec)

        # After final upsampling / conv, before the assert
        H, W = x.shape[-2], x.shape[-1]
        if (H, W) != (self.out_size, self.out_size):
            # Central crop down to (out_size, out_size)
            top = (H - self.out_size) // 2
            left = (W - self.out_size) // 2
            x = x[..., top:top + self.out_size, left:left + self.out_size]

        # Force exact spatial size (assert instead of silently interpolating)
        assert x.shape[-2] == self.out_size and x.shape[-1] == self.out_size, (
            f"[CNNDecoder] Decoder produced spatial size "
            f"{x.shape[-2:]} but expected ({self.out_size}, {self.out_size})"
        )

        x = self.refine(x)

        # CHW -> HWC
        x = x.permute(0, 2, 3, 1).contiguous()  # (B*T, H, W, C)
        bT, h, w, c = x.shape
        x = x.view(b, t, h, w, c)
        if t == 1:
            x = x.squeeze(1)
        return x


class CNNDecoderResidual(nn.Module):
    """
    Stronger CNN decoder for arbitrary square out_size.

    - Residual blocks at each scale
    - GroupNorm (batch-size agnostic) instead of plain BN / no norm
    - Full‑resolution refinement head
    - z: (B, D) or (B, T, D)
    - x: (B, H, W, C) or (B, T, H, W, C), logits
    """
    def __init__(
        self,
        latent_dim: int = 6,
        out_channels: int = 3,
        out_size: int = 320,
        base_channels: int = 128,
        max_channels: int = 512,
        num_groups: int = 8,  # NEW: configurable GroupNorm groups
        min_spatial: int = 8,
    ):
        super().__init__()
        print(
            f"[CNNDecoderResidual] Initializing (latent_dim={latent_dim}, "
            f"out_channels={out_channels}, out_size={out_size}, "
            f"base_channels={base_channels}, max_channels={max_channels}, "
            f"MIN_SPATIAL={min_spatial})"
        )

        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.out_size = out_size
        self.base_channels = base_channels
        self.max_channels = max_channels
        self.num_groups = num_groups

        # Derive start_h / num_upsamples like GastroDecoderCNN.
        num_upsamples = 0
        start_h = out_size
        while (start_h > min_spatial) and (start_h % 2 == 0) and num_upsamples < 6:
            start_h //= 2
            num_upsamples += 1

        self.start_h = self.start_w = start_h
        assert out_size == self.start_h * (2 ** num_upsamples), (
            f"[CNNDecoderResidual] out_size {out_size} not compatible with "
            f"start_h {self.start_h} and {num_upsamples} upsamplings"
        )

        print(
            f"[CNNDecoderResidual] Start spatial size: "
            f"({self.start_h}, {self.start_w}) with {base_channels} channels"
        )

        # FC to initial feature map
        self.fnn = nn.Linear(latent_dim, base_channels * self.start_h * self.start_w)
        print(
            f"[CNNDecoderResidual] Linear layer: {latent_dim} -> "
            f"{base_channels * self.start_h * self.start_w}"
        )

        # Upsampling stack with residual blocks
        layers = []
        in_c = base_channels
        for i in range(num_upsamples):
            # richer schedule than GastroDecoderCNN but compatible.
            if i == num_upsamples - 1:
                out_c = out_channels
            else:
                # gradually decrease channels but keep reasonably high capacity
                out_c = max(base_channels // (2 ** (i + 1)), 32)
                out_c = min(out_c, max_channels)

            print(
                f"[CNNDecoderResidual] Up Block {i}: in_ch={in_c}, out_ch={out_c}"
            )

            is_last = (i == num_upsamples - 1)

            block = [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            ]

            if not is_last:
                # Choose a valid number of groups for GroupNorm
                g = self.num_groups
                if out_c % g != 0:
                    g = 1
                block.extend(
                    [
                        nn.GroupNorm(g, out_c),
                        nn.LeakyReLU(0.2, inplace=True),
                        ResidualBlock(out_c, num_groups=self.num_groups),
                    ]
                )
            else:
                # Last block maps to RGB (3 channels); avoid invalid GroupNorm(8,3)
                block.append(ResidualBlock(out_c, num_groups=self.num_groups))

            layers.append(nn.Sequential(*block))
            in_c = out_c

        self.deconv_layers = nn.Sequential(*layers)
        print(
            f"[CNNDecoderResidual] Deconv stack created with {num_upsamples} "
            f"upsample blocks."
        )

        # Full‑resolution refinement head (only acts at target resolution)
        refine_groups = self.num_groups
        if self.out_channels % refine_groups != 0:
            refine_groups = 1

        self.refine = nn.Sequential(
            ResidualBlock(out_channels, num_groups=self.num_groups),
            nn.GroupNorm(refine_groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        """
        z:
        - (B, latent_dim) or
        - (B, T, latent_dim)

        Returns:
        x:
        - (B, H, W, C) or
        - (B, T, H, W, C), logits
        """
        if z.dim() == 3:
            b, t, d = z.shape
            z_flat = z.reshape(-1, d)
        else:
            b, d = z.shape
            t = 1
            z_flat = z

        x = self.fnn(z_flat)  # (B*T, C*H*W)
        x = x.view(-1, self.base_channels, self.start_h, self.start_w)

        x = self.deconv_layers(x)

        # Ensure exact spatial size (interpolate if off by 1 due to odd sizes).
        if x.shape[-2] != self.out_size or x.shape[-1] != self.out_size:
            x = F.interpolate(
                x,
                size=(self.out_size, self.out_size),
                mode="bilinear",
                align_corners=False,
            )

        x = self.refine(x)

        # CHW -> HWC
        x = x.permute(0, 2, 3, 1).contiguous()  # (B*T, H, W, C)
        bT, h, w, c = x.shape
        x = x.view(b, t, h, w, c)

        if t == 1:
            x = x.squeeze(1)

        return x
