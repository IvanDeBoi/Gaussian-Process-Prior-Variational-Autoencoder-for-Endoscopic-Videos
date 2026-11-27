"""Multi-scale vector-quantised VAE networks (MSVQ-VAE).

``MSB``: multi-scale conv block (3x3, 5x5, 7x7 branches). ``MSVQEncoder``: image (B, C, H, W) -> feature map.
``EMAVectorQuantizer``: codebook updated by exponential moving average; returns the straight-through quantised map,
the plain quantised map and the code indices. ``MSVQDecoder``: quantised map -> image (B, C, H, W).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class MSB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.branch5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=1)
        self.branch7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, stride=1)

        self.pointwise = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)
        self.bottleneck = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.res_pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        b3 = self.act(self.branch3(x))
        b5 = self.act(self.branch5(x))
        b7 = self.act(self.branch7(x))

        ms = torch.cat([b3, b5, b7], dim=1)
        ms = self.act(self.pointwise(ms))
        ms = self.act(self.bottleneck(ms))

        res = self.res_conv(x)
        out = self.act(ms + res)
        out = self.act(self.res_pointwise(out))
        return out


# ---------- Encoder / Decoder ----------

class MSVQEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=16, num_scales=3, use_checkpoint=True):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, stride=1)
        self.act = nn.ReLU(inplace=True)
        # FIX: gradient checkpointing trades recompute for memory — large MSB blocks
        # keep b3/b5/b7 alive simultaneously during backward, so checkpointing them
        # cuts peak activation memory roughly in half per encoder scale.
        self.use_checkpoint = use_checkpoint

        blocks = []
        c_in = base_channels
        for _ in range(num_scales):
            c_out = c_in * 2
            blocks.append(
                nn.Sequential(
                    MSB(c_in, c_out),
                    nn.Conv2d(c_out, c_out, kernel_size=3, stride=2, padding=1),
                )
            )
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = c_in

    def forward(self, x):
        x = self.act(self.stem(x))
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


class MSVQDecoder(nn.Module):
    def __init__(self, bottleneck_channels: int, out_channels: int = 3, num_scales: int = 3,
                 use_checkpoint: bool = True):
        super().__init__()
        # FIX: same checkpointing rationale as encoder — MSB in decoder also holds
        # 3 parallel feature maps alive during backward.
        self.use_checkpoint = use_checkpoint
        blocks = []
        c_in = bottleneck_channels
        for _ in range(num_scales):
            c_out = c_in // 2
            blocks.append(
                nn.ModuleDict({
                    "msb": MSB(c_in, c_in),
                    "upsample": nn.ConvTranspose2d(
                        c_in, c_out, kernel_size=4, stride=2, padding=1
                    ),
                })
            )
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.final_msb = MSB(c_in, c_in)
        self.final_conv = nn.Conv2d(c_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z_q):
        x = z_q
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(blk["msb"], x, use_reentrant=False)
            else:
                x = blk["msb"](x)
            x = blk["upsample"](x)
        if self.use_checkpoint and self.training:
            x = grad_checkpoint(self.final_msb, x, use_reentrant=False)
        else:
            x = self.final_msb(x)
        x = self.final_conv(x)
        return x


# ---------- EMA Vector Quantizer ----------

class EMAVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 128,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        self.register_buffer("embedding", torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", torch.zeros(num_embeddings, embedding_dim))

    def forward(self, z_e: torch.Tensor):
        B, C, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)

        with torch.no_grad():
            dist = (
                    flat.pow(2).sum(dim=1, keepdim=True)
                    - 2 * flat @ self.embedding.t()
                    + self.embedding.pow(2).sum(dim=1)
            )

            _, indices = dist.min(dim=1)

            if self.training:
                one_hot = F.one_hot(indices, self.num_embeddings).type_as(flat)
                cluster_size = one_hot.sum(0)
                embed_sum = one_hot.t() @ flat

                self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

                n = self.cluster_size.sum()
                cluster_size_smoothed = (
                                                (self.cluster_size + self.eps)
                                                / (n + self.num_embeddings * self.eps)
                                        ) * n

                embed_normalized = self.embed_avg / cluster_size_smoothed.unsqueeze(1)
                self.embedding.data.copy_(embed_normalized)

                dead = (cluster_size == 0)
                if dead.any():
                    n_dead = dead.sum().item()
                    perm = torch.randperm(flat.size(0), device=flat.device)[:n_dead]
                    self.embedding.data[dead] = flat[perm].detach()
                    self.embed_avg.data[dead] = flat[perm].detach()
                    self.cluster_size.data[dead] = 1.0

            indices_spatial = indices.view(B, H, W)

        z_q = self.embedding[indices].view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, z_q, indices_spatial
