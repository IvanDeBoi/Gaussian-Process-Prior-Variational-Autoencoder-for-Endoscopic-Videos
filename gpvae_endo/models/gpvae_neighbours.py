"""GP-VAE models with a neighbour-based approximation of the GP prior (Shi et al. 2025): GPVAE_HPA, GPVAE_SPA.

The exact GP KL of :mod:`gpvae_endo.models.gpvae_casale` needs all N frames at every step. These models
only look at each frame's ``n_neighbors`` nearest frames in time (fixed at construction):

* **HPA** -- KL between the encoder's distribution over the block {frame, its neighbours} and the GP prior
  restricted to that block. The block's latents are the *detached* encodings refreshed once per epoch
  (``refresh_epoch_state``), so this KL trains the GP hyper-parameters only (kept from the original).
* **SPA** -- KL between the encoder's distribution for the frame and the GP *conditional* given its
  neighbours' (detached) latents.

Both share the neighbour graph, the per-epoch refresh and the saved fields (:class:`NeighbourGPVAEModel`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from gpvae_endo.models.gpvae_casale import GPVAECasaleModel
from gpvae_endo.models.vae import _from_device, _with_device


class NeighbourGPVAEModel(GPVAECasaleModel):
    """Shared part of GPVAE_HPA and GPVAE_SPA."""

    short_name = "?"  # "HPA" / "SPA", for the messages

    def __init__(
        self,
        x: np.ndarray,
        Y: np.ndarray,
        M: np.ndarray,
        device: str = "gpu",
        latent_dim: int = 6,
        epochs: int = 10000,
        lr: float = 1e-3,
        max_beta: float = 1e-3,
        img_H: int = 32,
        img_W: Optional[int] = None,
        img_C: int = 3,
        kernel_gp_prior: str = "RBF",
        underlying_method: str = "GPVAE_neighbours",
        n_neighbors: int = 32,
        batch_size: int = 16,
        encoder: str = "not set yet",
        decoder: str = "not set yet",
        **kwargs,
    ):
        super().__init__(
            x=x, Y=Y, M=M, device=device, latent_dim=latent_dim, epochs=epochs, lr=lr, max_beta=max_beta,
            img_H=img_H, img_W=img_W, img_C=img_C, kernel_gp_prior=kernel_gp_prior,
            underlying_method=underlying_method, encoder=encoder, decoder=decoder, batch_size=batch_size,
            **kwargs,
        )
        print(f"kernel_gp_prior={kernel_gp_prior}, n_neighbors={n_neighbors}")
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

        N = self.Xtensor_norm.size(0)
        self.mu_all = torch.zeros(N, self.latent_dim, device=self.device)
        self.logvar_all = torch.zeros(N, self.latent_dim, device=self.device)

        print(f"Precomputing neighbour indices on normalized Xtensor ({self.short_name})...")
        self._build_neighbour_graph()
        print(f"Done precomputing neighbour indices for {self.short_name}.")
        print(f"{self.underlying_method} initialized successfully")

    def _build_neighbour_graph(self):
        """For every frame, the indices of its `n_neighbors` nearest frames in (normalised) frame number."""
        with torch.no_grad():
            X = self.Xtensor_norm
            N = X.size(0)
            sort_idx = torch.argsort(X.squeeze(-1)) if X.size(-1) == 1 else torch.argsort(X[:, 0])
            X_sorted = X[sort_idx]
            dist2 = torch.cdist(X_sorted, X_sorted, p=2.0) ** 2
            knn = torch.topk(dist2, k=min(self.n_neighbors + 1, N), largest=False, dim=1).indices
            neighbour_idx = sort_idx[knn[:, 1:]]  # drop the frame itself

            inv_sort = torch.empty_like(sort_idx)
            inv_sort[sort_idx] = torch.arange(N, device=self.device)
            self.sorted_idx = sort_idx
            self.inv_sorted_idx = inv_sort
            self.neighbour_idx = neighbour_idx  # (N, K), indexed by original frame position

    def refresh_epoch_state(self):
        """Once per epoch: encode every frame (no gradient) and keep the means and log-variances."""
        epoch_state_batch_size = getattr(self, "epoch_state_batch_size", None)
        if epoch_state_batch_size is None:
            epoch_state_batch_size = self.batch_size
        self.encoder.eval()
        with torch.no_grad():
            mu_all, logvar_all = self._encode_ytensor_in_batches(batch_size=epoch_state_batch_size,
                                                                 use_no_grad=True)
            logvar_all = torch.clamp(logvar_all, -10.0, 10.0)
            self.mu_all = None
            self.logvar_all = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.mu_all = mu_all.detach()
            self.logvar_all = logvar_all.detach()
        self.encoder.train()

    def _local_prior_covariance(self, idx_local: torch.Tensor) -> torch.Tensor:
        """GP prior covariance (plus noise and 1e-6 jitter) of each block of frames, (b, k, k)."""
        k = idx_local.size(1)
        K_loc = self.gp_prior.covar_module(self.Xtensor_norm[idx_local]).evaluate()
        eye = torch.eye(k, device=self.device).unsqueeze(0)
        if hasattr(self.gp_prior, "likelihood") and hasattr(self.gp_prior.likelihood, "noise"):
            K_loc = K_loc + self.gp_prior.likelihood.noise * eye
        else:
            K_loc = K_loc + 1e-4 * eye
        return K_loc + 1e-6 * eye

    # -- saving -------------------------------------------------------------------------------------------
    def state_dict_full(self):
        state = super().state_dict_full()
        state.update({
            "kernel_gp_prior": self.kernel_gp_prior,
            "gp_prior": self.gp_prior.state_dict(),
            "n_neighbors": self.n_neighbors,
            "sorted_idx": self.sorted_idx.detach().cpu(),
            "inv_sorted_idx": self.inv_sorted_idx.detach().cpu(),
            "neighbour_idx": self.neighbour_idx.detach().cpu(),
            "mu_all": self.mu_all.detach().cpu(),
            "logvar_all": self.logvar_all.detach().cpu(),
        })
        return state

    def _load_neighbour_state(self, state):
        self.kernel_gp_prior = state.get("kernel_gp_prior", self.kernel_gp_prior)
        self.n_neighbors = state.get("n_neighbors", self.n_neighbors)

    def _load_graph_and_latents(self, state):
        N = self.Xtensor_norm.size(0)
        if "sorted_idx" in state:
            self.sorted_idx = state["sorted_idx"].long().to(self.device)
        if "inv_sorted_idx" in state:
            self.inv_sorted_idx = state["inv_sorted_idx"].long().to(self.device)
        if "neighbour_idx" in state:
            self.neighbour_idx = state["neighbour_idx"].long().to(self.device)
        else:
            self._build_neighbour_graph()
        if "mu_all" in state:
            self.mu_all = state["mu_all"].float().to(self.device)
        else:
            self.mu_all = torch.zeros(N, self.latent_dim, device=self.device)
        if "logvar_all" in state:
            self.logvar_all = state["logvar_all"].float().to(self.device)
        else:
            self.logvar_all = torch.zeros(N, self.latent_dim, device=self.device)

    def checkpoint_state(self) -> dict:
        state = super().checkpoint_state()
        state["logvar_all"] = _with_device(self.logvar_all)
        return state

    def load_checkpoint_state(self, state: dict) -> None:
        super().load_checkpoint_state(state)
        self.logvar_all = _from_device(state["logvar_all"])


# ----------------------------------------------------------------------------------------------------------
class GPVAEHPAModel(NeighbourGPVAEModel):
    """GPVAE_HPA: block KL on {frame, neighbours} against the local GP prior."""

    short_name = "HPA"

    def __init__(self, *args, underlying_method: str = "GPVAE_HPA", **kwargs):
        super().__init__(*args, underlying_method=underlying_method, **kwargs)

    @staticmethod
    def _format_lengthscale(v) -> str:
        return f"{v:.4f}"

    def hpa_kl_batch(self, mu_all: torch.Tensor, sigma2_all: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """Mean over the batch's blocks and latent dimensions of KL(q(Z_block) || p(Z_block)), divided by
        the block size k.

        Faster than the original: K_loc is shared by all Q latent dimensions of a block and the encoder
        covariance S = diag(s2) is diagonal, so tr(K^-1 S) = sum_k diag(K^-1)_k * s2_k and mu^T K^-1 mu is
        one contraction with K^-1. The original copied the Cholesky factor once per latent dimension
        (Q x the memory). Same value up to float rounding (about 1e-7 relative).
        """
        b = batch_idx.size(0)
        block_global = torch.cat([batch_idx.unsqueeze(1), self.neighbour_idx[batch_idx]], dim=1)  # (b, k)
        k = block_global.size(1)

        mu_block = mu_all[block_global].permute(0, 2, 1).contiguous()     # (b, Q, k)
        s2_block = sigma2_all[block_global].permute(0, 2, 1).contiguous()  # (b, Q, k)

        L = torch.linalg.cholesky(self._local_prior_covariance(block_global))  # (b, k, k)
        logdetK = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + 1e-12), dim=-1)  # (b,)

        K_inv = torch.cholesky_inverse(L)  # (b, k, k)
        tr_term = torch.einsum("bk,bqk->bq", torch.diagonal(K_inv, dim1=-2, dim2=-1), s2_block)
        quad_term = torch.einsum("bqk,bkj,bqj->bq", mu_block, K_inv, mu_block)
        ent_term = torch.sum(torch.log(s2_block + 1e-12), dim=-1)
        kl_block = 0.5 * (tr_term + quad_term - ent_term + logdetK.unsqueeze(1) - k)  # (b, Q)
        assert kl_block.shape[0] == b
        return kl_block.mean() / k

    def compute_kl_term(self, mu, logvar, x_batch=None, mask=None, Xb_norm=None, batch_idx=None, **kwargs):
        if batch_idx is None:
            raise ValueError("GPVAEHPAShiModel.compute_kl_term requires batch_idx")
        sigma2_all = torch.exp(torch.clamp(self.logvar_all, -10.0, 10.0))
        kl_gp = self.hpa_kl_batch(self.mu_all, sigma2_all, batch_idx)
        if not torch.isfinite(kl_gp):
            raise ValueError("Non-finite HPA KL encountered")
        return kl_gp

    def load_state_dict_full(self, state):
        # Casale's loader already builds a new GP; the original HPA loader builds a second one
        # (6 numpy draws in total). Kept, because it moves the random state the evaluation then uses.
        super().load_state_dict_full(state)
        self._load_neighbour_state(state)
        self.gp_prior = self._new_gp()
        if state.get("gp_prior", None) is not None:
            self.gp_prior.load_state_dict(state["gp_prior"])
        self._load_graph_and_latents(state)


class GPVAESPAModel(NeighbourGPVAEModel):
    """GPVAE_SPA: per-frame KL against the GP conditional given the neighbours' latents."""

    short_name = "SPA"

    def __init__(self, *args, underlying_method: str = "GPVAE_SPA", **kwargs):
        super().__init__(*args, underlying_method=underlying_method, **kwargs)

    def spa_kl_batch(self, mu_enc_batch: torch.Tensor, logvar_enc_batch: torch.Tensor,
                     batch_idx: torch.Tensor) -> torch.Tensor:
        b, Q = mu_enc_batch.shape
        sigma2_enc_batch = torch.exp(torch.clamp(logvar_enc_batch, -10.0, 10.0))

        nb_idx_batch = self.neighbour_idx[batch_idx]
        K_loc = self._local_prior_covariance(torch.cat([batch_idx.unsqueeze(1), nb_idx_batch], dim=1))
        K_ii = K_loc[:, 0, 0]
        K_iN = K_loc[:, 0, 1:]
        K_NN = K_loc[:, 1:, 1:]

        # GP conditional p(z_i | z_neighbours) = N(alpha^T z_nb, v)
        L_NN = torch.linalg.cholesky(K_NN)
        alpha = torch.cholesky_solve(K_iN.unsqueeze(-1), L_NN).squeeze(-1)
        v_j = K_ii - torch.sum(K_iN * alpha, dim=-1)
        p_mu_j = torch.sum(alpha.unsqueeze(-1) * self.mu_all[nb_idx_batch], dim=1)

        mu_q = mu_enc_batch
        v_j_exp = torch.clamp(v_j.unsqueeze(1).expand_as(mu_q), min=1e-6, max=1e2)
        s2_q = torch.clamp(sigma2_enc_batch, min=1e-6, max=1e1)
        kl_all = 0.5 * (torch.log(v_j_exp / s2_q) + (s2_q + (mu_q - p_mu_j) ** 2) / v_j_exp - 1.0)
        return kl_all.mean(dim=0).sum() / Q

    def compute_kl_term(self, mu, logvar, x_batch=None, mask=None, Xb_norm=None, batch_idx=None, **kwargs):
        if batch_idx is None:
            raise ValueError("GPVAESPAShiModel.compute_kl_term requires batch_idx")
        kl_gp = self.spa_kl_batch(mu, logvar, batch_idx)
        kl_gp = (self.Ytensor.size(0) / batch_idx.size(0)) * kl_gp
        if not torch.isfinite(kl_gp):
            raise ValueError("Non-finite SPA KL encountered")
        return kl_gp

    def load_state_dict_full(self, state):
        super().load_state_dict_full(state)  # Casale's loader: one new GP (3 numpy draws)
        self._load_neighbour_state(state)
        self._load_graph_and_latents(state)
