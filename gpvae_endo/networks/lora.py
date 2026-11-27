"""LoRA (low-rank adaptation) linear layer.

``LoRALinear`` wraps a frozen ``nn.Linear`` and adds a trainable low-rank update ``scaling * (x A^T) B^T``. Used to
adapt the attention / MLP layers of the DINOv2 backbone in ``GastronetViTEncoderLoRABackbone``.

The original class body defined ``forward`` twice (a verbose one with NaN/device/dtype checks, then a short one using
``F.linear``); the second one was effective at run time and is the only one kept here.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        bias: bool = True,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)

        self.base = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.base.weight)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise RuntimeError(
                f"LoRALinear: expected last dim {self.in_features}, "
                f"got {x.shape[-1]} for module {self}"
            )

        out = self.base(x)
        if self.rank <= 0:
            return out

        lora = F.linear(x, self.lora_A)
        lora = F.linear(lora, self.lora_B)

        if lora.device != out.device or lora.dtype != out.dtype:
            lora = lora.to(device=out.device, dtype=out.dtype)

        return out + self.scaling * lora
