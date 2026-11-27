"""DINOv2 ViT encoders (GastroNet / GaLoRa).

``GastronetViTEncoder``: images (B, H, W, C) or (B, T, H, W, C) -> latent mean and log-variance, (B, latent_dim) or
(B, T, latent_dim), via a pretrained DINOv2 ViT backbone and an MLP head (``GastroMLPHead``).
``GastronetViTEncoderLoRABackbone`` additionally injects LoRA layers into the backbone. The pretrained weights are read
from ``gpvae_endo.paths.weights_dir()`` and downloaded from HuggingFace when missing.
"""
import argparse
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch import nn, Tensor
from torch.serialization import add_safe_globals

from gpvae_endo.networks.dinov2.layers import MemEffAttention, NestedTensorBlock as Block
from gpvae_endo.networks.dinov2.models.vision_transformer import DinoVisionTransformer as VisionTransformer
from gpvae_endo.networks.lora import LoRALinear
from gpvae_endo.paths import weights_dir as _weights_dir


class GastronetViTEncoder(nn.Module):
    """
    Gastronet ViT encoder, same API as EndoViTEncoder.

    Inputs:
        - B, H, W, C
        - B, T, H, W, C

    Outputs:
        - q_means:    B, latent_dim   or   B, T, latent_dim
        - q_logvars:  same shape
    """

    def __init__(
        self,
        latent_dim: int = 6,
        in_channels: int = 3,
        in_size: int = 320,
        repo_id: str = "tgwboers/GastroNet-5M_Pretrained_Weights",  # your repo
        img_size_vit: int = 224,
        # weights_name: str = "VITS_GastroNet-5M_DINOv1.pth",
        # weights_name: str = "dinov2.pth",
        weights_name: str = "dinov2_base.pth",
        # weights_name: str = "RN50_GastroNet-5M_MOCOv2.pth",
        # weights_name: str = "RN50_GastroNet-200K_DINOv1.pth",
        # weights_name: str = "RN50_GastroNet-1M_DINOv1.pth",
        # weights_name: str = "RN50_Billion-Scale-SWSL%2BGastroNet-5M_DINOv1.pth",
        # weights_name: str = "RN50_GastroNet-5M_SIMCLRv2.pth",
        # weights_name: str = "RN50_GastroNet-5M_DINOv1.pth",
        # weights_name: str = "EndoViT_mae_backbone.pth",
        train_backbone: bool = False,  # NEW FLAG
    ):
        super().__init__()

        print(f"[GastronetViTEncoder] latent_dim={latent_dim}, in_size={in_size}, "
              f"train_backbone={train_backbone}")
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_size = in_size
        self.img_size_vit = img_size_vit
        self.repo_id = repo_id
        self.weights_name = weights_name
        self.train_backbone = train_backbone  # store flag

        # ------------------------------------------------------------------
        # 1. Build ViT backbone (same architecture you use for EndoViT).
        # ------------------------------------------------------------------
        # self.backbone = VisionTransformer(
        #     img_size=self.img_size_vit,
        #     patch_size=14,
        #     in_chans=in_channels,
        #     embed_dim=768,
        #     depth=12,
        #     num_heads=12,
        #     mlp_ratio=4,
        #     qkv_bias=True,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # )

        self.backbone = VisionTransformer(
                    img_size=336,
                    patch_size=14,
                    in_chans=3,
                    embed_dim=768,
                    depth=12,
                    num_heads=12,
                    mlp_ratio=4,
                    block_fn=partial(Block, attn_class=MemEffAttention),
                    num_register_tokens=0,
                    block_chunks=0,
                    init_values=1e-5,
                )

        # ------------------------------------------------------------------
        # 2. Load weights: local cache in the weights folder (see gpvae_endo.paths.weights_dir).
        # ------------------------------------------------------------------

        weights_dir = _weights_dir()

        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / self.weights_name

        if weights_path.is_file():
            print(f"[GastronetViTEncoder] Found local weights at {weights_path}")
            state = torch.load(
                str(weights_path),
                map_location="cpu",
                weights_only=False,
            )
        else:
            print(
                f"[GastronetViTEncoder] Local weights not found at {weights_path}, "
                f"downloading from HuggingFace repo: {self.repo_id} ..."
            )

            local_repo_path = snapshot_download(
                repo_id=self.repo_id,
                allow_patterns=["*.bin", "*.pth", "*.pt"],
            )
            local_repo_path = Path(local_repo_path)
            candidate_files = (
                list(local_repo_path.glob("*.pth"))
                + list(local_repo_path.glob("*.bin"))
                + list(local_repo_path.glob("*.pt"))
            )
            if not candidate_files:
                raise FileNotFoundError(
                    f"[GastronetViTEncoder] No .pth/.bin/.pt files found in downloaded repo "
                    f"{local_repo_path}."
                )

            ckpt_file = candidate_files[0]
            print(f"[GastronetViTEncoder] Using checkpoint file: {ckpt_file}")

            add_safe_globals([argparse.Namespace])
            raw_state = torch.load(
                str(ckpt_file),
                map_location="cpu",
                weights_only=False,
            )

            if "model" in raw_state and isinstance(raw_state["model"], dict):
                state = raw_state["model"]
                print("[GastronetViTEncoder] Extracted 'model' sub-dict from checkpoint.")
            else:
                state = raw_state
                print("[GastronetViTEncoder] Using full checkpoint as state_dict.")

            torch.save(state, str(weights_path))
            print(f"[GastronetViTEncoder] Saved backbone weights to {weights_path}")

        # model has the weights as ['teacher'].backbone.*
        msg = self.backbone.load_state_dict(state, strict=True)
        print("[GastronetViTEncoder] Load state dict:", msg)

        # ------------------------------------------------------------------
        # 3. Freeze backbone and define latent head
        #    (NOW controlled by train_backbone flag)
        # ------------------------------------------------------------------
        if self.train_backbone:
            # allow finetuning of backbone
            for p in self.backbone.parameters():
                p.requires_grad = True
            # do NOT force eval; module.train()/eval() will control it
            print("[GastronetViTEncoder] Backbone set to trainable.")
        else:
            # freeze backbone as before
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            print("[GastronetViTEncoder] Backbone frozen and set to eval().")

        self.feat_dim = 768
        head_dim_1 = 1024
        head_dim_2 = 512
        head_dim = head_dim_2

        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, head_dim_1),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(head_dim_1, head_dim_2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(head_dim_2, head_dim_2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(head_dim_2, latent_dim * 2),
        )

        total_params = sum(p.numel() for p in self.head.parameters())
        print(f"GastroNet Head ({total_params:,} params, 768->{latent_dim}d):")
        print(f"  Linear(768 -> {head_dim_1})")
        print(f"  GELU + Dropout(0.1)")
        print(f"  Linear({head_dim_1} -> {head_dim_2})")
        print(f"  GELU + Dropout(0.1)")
        print(f"  Linear({head_dim_2} -> {head_dim_2})")
        print(f"  GELU + Dropout(0.1)")
        print(f"  Linear({head_dim_2} -> {latent_dim * 2})")
        self._init_weights_head()
        return



    # ----------------------------------------------------------------------
    # Helper: initialize small head
    # ----------------------------------------------------------------------
    def _init_weights_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    # ----------------------------------------------------------------------
    # Backbone feature extraction (CLS pooled features)
    # ----------------------------------------------------------------------
    def extract_backbone_features(self, x_chw: Tensor) -> Tensor:
        """
        x_chw: B, C, H, W in [0,1]
        Returns: B, 768 CLS features
        """
        if x_chw.shape[-2] != self.img_size_vit or x_chw.shape[-1] != self.img_size_vit:
            x_chw = F.interpolate(
                x_chw,
                size=(self.img_size_vit, self.img_size_vit),
                mode="bilinear",
                align_corners=False,
            )

        feats = self.backbone(x_chw)  # B, 768
        if feats.dim() != 2 or feats.size(1) != self.feat_dim:
            raise RuntimeError(f"GastronetViTEncoder: unexpected feats shape {feats.shape}")
        return feats

    # ----------------------------------------------------------------------
    # Head‑only encoders for cached features
    # ----------------------------------------------------------------------
    def encode_from_feats(self, feats: Tensor) -> Tuple[Tensor, Tensor]:
        """
        feats: (B*, 768)  (output of extract_backbone_features)
        Returns:
            q_means:    (B*, latent_dim)
            q_logvars:  (B*, latent_dim)
        """
        params = self.head(feats)  # B*, 2*latent_dim
        q_means = params[..., : self.latent_dim]
        q_logvars = params[..., self.latent_dim:]
        return q_means, q_logvars

    def encode_from_feats_batched(
        self,
        feats: Tensor,
        b: int,
        t: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        feats: (B*T, 768) or (N, 768) you provide (b, t).
        Returns:
            if t > 1: (B, T, latent_dim)
            else:     (B, latent_dim)
        """
        params = self.head(feats)  # (B*T, 2*latent_dim) or (N, 2*latent_dim)
        params = params.view(b, t, -1)
        q_means = params[..., : self.latent_dim]
        q_logvars = params[..., self.latent_dim:]
        if t == 1:
            q_means = q_means.squeeze(1)
            q_logvars = q_logvars.squeeze(1)
        return q_means, q_logvars

    # ----------------------------------------------------------------------
    # Forward: HWC / THWC to latents
    # ----------------------------------------------------------------------
    def forward(self, vid_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        vid_batch:
            - B, H, W, C
            - B, T, H, W, C

        Returns:
            q_means, q_logvars
                - B, latent_dim   or
                - B, T, latent_dim
        """
        if vid_batch.dim() == 5:
            # B, T, H, W, C
            b, t, h, w, c = vid_batch.shape
            x = vid_batch.reshape(-1, h, w, c)
        else:
            # B, H, W, C
            b, h, w, c = vid_batch.shape
            t = 1
            x = vid_batch

        x_chw = x.permute(0, 3, 1, 2).contiguous()  # B*, C, H, W
        feats = self.extract_backbone_features(x_chw)  # B*, 768

        params = self.head(feats)  # B*, 2*latent_dim
        params = params.view(b, t, -1)
        q_means = params[..., : self.latent_dim]
        q_logvars = params[..., self.latent_dim:]

        if t == 1:
            q_means = q_means.squeeze(1)
            q_logvars = q_logvars.squeeze(1)

        return q_means, q_logvars


class GastroMLPHead(nn.Module):
    """GastroNet-style plain MLP head."""

    def __init__(
            self,
            in_dim: int,
            latent_dim: int,
            hidden_dim: int | None = None,
            dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 1024
        mid_dim = 512
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.mid_dim = mid_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, latent_dim * 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def describe(self, label: str = "GastroNet MLP Head") -> None:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[{label}] total={total_params:,}, trainable={trainable_params:,}")
        print(f"  Linear({self.in_dim} -> {self.hidden_dim})")
        print(f"  GELU + Dropout({self.dropout})")
        print(f"  Linear({self.hidden_dim} -> {self.mid_dim})")
        print(f"  GELU + Dropout({self.dropout})")
        print(f"  Linear({self.mid_dim} -> {self.mid_dim})")
        print(f"  GELU + Dropout({self.dropout})")
        print(f"  Linear({self.mid_dim} -> {self.latent_dim * 2})")


class GastronetViTEncoderLoRABackbone(GastronetViTEncoder):
    def __init__(
        self,
        latent_dim: int = 6,
        in_channels: int = 3,
        in_size: int = 320,
        repo_id: str = "tgwboers/GastroNet-5M_Pretrained_Weights",
        img_size_vit: int = 336,
        weights_name: str = "dinov2_base.pth",
        train_backbone: bool = False,
        use_lora_backbone: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_target_keywords: Iterable[str] = ("qkv", "proj", "fc1", "fc2"),
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.1,
    ):
        super().__init__(
            latent_dim=latent_dim,
            in_channels=in_channels,
            in_size=in_size,
            repo_id=repo_id,
            img_size_vit=img_size_vit,
            weights_name=weights_name,
            train_backbone=train_backbone,
        )

        self.use_lora_backbone = use_lora_backbone
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_keywords = tuple(lora_target_keywords)

        if self.use_lora_backbone:
            self._inject_lora_into_backbone()

        self.head = GastroMLPHead(
            in_dim=self.feat_dim,
            latent_dim=latent_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

        total_head = sum(p.numel() for p in self.head.parameters())
        trainable_head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        print(
            f"[GastronetViTEncoderLoRABackbone] latent_dim={latent_dim}, in_size={in_size}, "
            f"train_backbone={train_backbone}, use_lora_backbone={use_lora_backbone}"
        )
        self.head.describe("GastroNet MLP Head")

    def _inject_lora_into_backbone(self) -> None:
        replaced: list[str] = []
        for module_name, module in list(self.backbone.named_modules()):
            for child_name, child in list(module.named_children()):
                if not isinstance(child, nn.Linear):
                    continue
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                if not any(keyword in full_name for keyword in self.lora_target_keywords):
                    continue

                lora_layer = LoRALinear(
                    child.in_features,
                    child.out_features,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    bias=child.bias is not None,
                    freeze_base=True,
                )
                with torch.no_grad():
                    lora_layer.base.weight.copy_(child.weight)
                    if child.bias is not None:
                        lora_layer.base.bias.copy_(child.bias)

                setattr(module, child_name, lora_layer)
                replaced.append(full_name)

        if not replaced:
            print(
                "[GastronetViTEncoderLoRABackbone] warning: "
                "no backbone linear layers matched LoRA targets."
            )
        else:
            print(
                f"[GastronetViTEncoderLoRABackbone] injected LoRA into {len(replaced)} backbone layers."
            )
            for name in replaced[:20]:
                print(f"  - {name}")
            if len(replaced) > 20:
                print(f"  ... and {len(replaced) - 20} more")
