"""Small building blocks shared by the networks.

``ResidualBlock``: GroupNorm -> LeakyReLU -> 3x3 conv, twice, with a skip connection (input and output have the same
shape, (B, C, H, W)). Used by ``CNNDecoderResidual``.

The original file defined this class twice (identical code, only comments differed); this is the second definition,
which is the one that was effective at run time.
"""
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    """Simple residual block for decoder refinement."""
    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()

        # Ensure num_groups divides channels; fall back to 1 (InstanceNorm-like)
        if channels % num_groups != 0:
            num_groups = 1

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)
