"""GP-VAE with the exact GP prior of Casale et al. (2018), https://arxiv.org/abs/1810.11738.

Prior over the latent trajectory: p(Z | x) = N(0, K_xx (x) I_Q), with K_xx a Gaussian-process kernel over the
frame numbers x (here a Cauchy kernel). The kernel hyper-parameters (lengthscale, output scale, noise) are
learned together with the encoder and decoder, at 10x the learning rate.

Besides the KL term this class adds :meth:`time_encode` (GP regression: frame number -> latent, used to
fill in missing frames) and :meth:`get_prior_covariance` (for plots). GPVAE_HPA and GPVAE_SPA build on it.

Kept from the original on purpose (results must not change):

* finding B9: :meth:`compute_kl_term` does not accept the extra arguments the training loop passes, so
  GPVAE_Casale raises a ``TypeError`` as soon as the KL weight becomes non-zero (after the warm-up);
* finding B3: the optimiser is rebuilt here with every encoder group at ``lr`` (the LoRA LR is dropped);
* finding B15: :meth:`load_state_dict_full` builds a new GP (3 numpy random draws) and the optimiser keeps
  pointing at the old one.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from gpvae_endo.gp.gp import GP
from gpvae_endo.models.vae import VAEModel


class GPVAECasaleModel(VAEModel):
    """Full (exact) GP prior over all N training frames."""

    _save_message = "✓ Saved {method} → {path}"

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
        underlying_method: str = "GPVAECasale-fullGP",
        encoder: str = "not set yet",
        decoder: str = "not set yet",
        **kwargs,
    ):
        super().__init__(x, Y, M, device, latent_dim, epochs, lr, max_beta, img_H=img_H, img_W=img_W, img_C=img_C,
                         underlying_method=underlying_method, encoder=encoder, decoder=decoder, **kwargs)

        self.kernel_gp_prior = kernel_gp_prior
        self.gp_prior = self._new_gp()

        # Rebuild the optimiser with the GP hyper-parameters at 10x the LR (encoder groups at `lr`: B3).
        gp_lr = 10.0 * lr
        base_kernel = self.gp_prior.covar_module.base_kernel
        ls_params = [base_kernel.raw_lengthscale if hasattr(base_kernel, "raw_lengthscale") else base_kernel.lengthscale]
        os_params = ([self.gp_prior.covar_module.raw_outputscale]
                     if hasattr(self.gp_prior.covar_module, "raw_outputscale") else [])
        noise_params = ([self.gp_prior.likelihood.raw_noise]
                        if hasattr(self.gp_prior, "likelihood") and hasattr(self.gp_prior.likelihood, "raw_noise")
                        else [])
        self.optimizer = torch.optim.Adam([
            *self.get_trainable_encoder_params(lr),
            {"params": self.decoder.parameters(), "lr": lr},
            {"params": ls_params, "lr": gp_lr},
            {"params": os_params, "lr": gp_lr},
            {"params": noise_params, "lr": gp_lr},
        ], lr=self.lr)
        self._build_schedulers()  # the schedulers must follow the new optimiser

        print(f"{self.underlying_method} initialized successfully")

    def _new_gp(self) -> GP:
        """A fresh GP over the training frame numbers (draws 3 numbers from numpy's global RNG)."""
        return GP(
            train_x=self.Xtensor_norm,
            train_y=torch.zeros(self.Xtensor_norm.size(0), 1, device=self.device),  # dummy targets
            device=self.device,
            kernel=self.kernel_gp_prior,
            verbose=False,
            use_x_scaling=self.settings.use_x_scaling,
        )

    # ------------------------------------------------------------------------------------------------------
    # hooks for the training loop
    # ------------------------------------------------------------------------------------------------------
    def _gp_hyperparams_str(self) -> str:
        """Lengthscale, output scale and noise for the per-epoch log line."""
        covar = self.gp_prior.covar_module
        base = covar.base_kernel if hasattr(covar, "base_kernel") else covar
        ls = base.lengthscale.detach().cpu().view(-1)
        outputscale = covar.outputscale.detach().cpu().item() if hasattr(covar, "outputscale") else None
        noise = None
        if hasattr(self.gp_prior, "likelihood") and hasattr(self.gp_prior.likelihood, "noise"):
            noise = self.gp_prior.likelihood.noise.detach().cpu().item()
        parts = ["ls=" + ",".join(self._format_lengthscale(v) for v in ls)]
        if outputscale is not None:
            parts.append(f"os={outputscale:.4f}")
        if noise is not None:
            parts.append(f"noise={noise:.6f}")
        return "|".join(parts)

    @staticmethod
    def _format_lengthscale(v) -> str:
        return f"{v:.4}"  # 4 significant digits (GPVAE_HPA prints 4 decimals; both kept as they were)

    def get_extra_log_message(self) -> str:
        return self._gp_hyperparams_str()

    def get_generator_parameters_for_clipping(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        params += list(self.gp_prior.covar_module.parameters())
        if hasattr(self.gp_prior, "likelihood"):
            params += list(self.gp_prior.likelihood.parameters())
        return params

    def compute_kl_term(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        x_batch: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Exact GP KL over all frames (re-encodes every frame on each step).

        No ``**kwargs`` on purpose: the training loop passes ``Xb_norm`` and ``batch_idx``, so this raises a
        TypeError once the KL weight is non-zero, exactly like the original (finding B9).
        """
        self.encoder.eval()
        if self.backbone_feats is not None and hasattr(self.encoder, "encode_from_feats_batched"):
            feats_all = self.backbone_feats
            mu_all, logvar_all = self.encoder.encode_from_feats_batched(feats_all, b=feats_all.size(0), t=1)
        else:
            mu_all, logvar_all = self._encode_ytensor_in_batches(use_no_grad=False)
        self.encoder.train()
        return self.gp_prior_kl_full(mu_all, logvar_all)

    def gp_prior_kl_full(self, mu_enc, logvar_enc):
        """KL(q(Z) || N(0, K)) summed over the Q latent dimensions, divided by N*Q."""
        X = self.Xtensor_norm
        N, Q = mu_enc.shape
        K = self.gp_prior.covar_module(X).evaluate()
        if hasattr(self.gp_prior, "likelihood") and hasattr(self.gp_prior.likelihood, "noise"):
            K = K + self.gp_prior.likelihood.noise * torch.eye(N, device=self.device)
        else:
            K = K + 1e-4 * torch.eye(N, device=self.device)
        K = K + 1e-6 * torch.eye(N, device=self.device)

        L, info = torch.linalg.cholesky_ex(K)
        if torch.any(info > 0):
            print("Cholesky failed, info:", info.nonzero())
            K = K + 1e-2 * torch.eye(N, device=self.device)
            L = torch.linalg.cholesky(K)
        logdetK = 2.0 * torch.sum(torch.log(torch.diag(L) + 1e-12))

        total_kl = 0.0
        for q in range(Q):
            mu_q = mu_enc[:, q]
            sigma2_q = torch.exp(torch.clamp(logvar_enc[:, q], -10.0, 10.0))
            mu_q_col = mu_q.unsqueeze(1)
            quad_term = torch.sum(mu_q_col * torch.cholesky_solve(mu_q_col, L))      # mu^T K^-1 mu
            W = torch.cholesky_solve(torch.diag(torch.sqrt(sigma2_q + 1e-12)), L)
            trace_term = torch.sum(W * W)                                             # as in the original
            entropy_term = torch.sum(torch.log(sigma2_q + 1e-12))
            total_kl = total_kl + 0.5 * (trace_term + quad_term - entropy_term + logdetK - N)
        return total_kl / (N * Q)

    # ------------------------------------------------------------------------------------------------------
    # GP regression from frame number to latent
    # ------------------------------------------------------------------------------------------------------
    @staticmethod
    def _stable_cholesky(K: torch.Tensor, base_jitter: float = 1e-6, max_tries: int = 8) -> torch.Tensor:
        K = 0.5 * (K + K.transpose(-1, -2))
        eye = torch.eye(K.size(-1), device=K.device, dtype=K.dtype)
        jitter = base_jitter * K.diagonal(dim1=-2, dim2=-1).abs().mean().clamp_min(1.0)
        for attempt in range(max_tries):
            L, info = torch.linalg.cholesky_ex(K + jitter * eye)
            if torch.all(info == 0):
                if attempt > 0:
                    print(f"[time_encode] Cholesky succeeded with jitter={float(jitter.detach().cpu()):.3e}")
                return L
            jitter = jitter * 10.0
        min_eig = torch.linalg.eigvalsh(K).min().detach().cpu().item()
        raise RuntimeError(
            f"Cholesky failed after {max_tries} jitter attempts; "
            f"min eigenvalue={min_eig:.3e}, final jitter={float(jitter.detach().cpu()):.3e}"
        )

    def time_encode(self, x_new: np.ndarray, use_likelihood: bool = True, return_full_cov: bool = False):
        """GP posterior of the latents at new frame numbers, given the encoded training frames.

        Returns ``(Z_mean, Z_std)`` each (M, Q), plus the (M, M) covariance (shared by all latent
        dimensions) if `return_full_cov`.
        """
        self.eval()
        device = self.device
        x_new_tensor = torch.from_numpy(x_new).float().to(device)
        if x_new_tensor.dim() == 1:
            x_new_tensor = x_new_tensor.unsqueeze(-1)
        x_new_norm = (x_new_tensor - self.X_mean) / self.X_std

        X_train = self.Xtensor_norm
        N = X_train.size(0)
        M = x_new_norm.size(0)
        Q = self.latent_dim
        with torch.no_grad():
            mu_all = self.mu_all.to(device)
            Kxx = self.gp_prior.covar_module(X_train).evaluate()
            Kxs = self.gp_prior.covar_module(X_train, x_new_norm).evaluate()
            Kss = self.gp_prior.covar_module(x_new_norm).evaluate()
            if hasattr(self.gp_prior, "likelihood") and hasattr(self.gp_prior.likelihood, "noise"):
                noise = self.gp_prior.likelihood.noise if use_likelihood else 0.0
            else:
                noise = 0.0
            Kxx = Kxx + noise * torch.eye(N, device=device, dtype=Kxx.dtype)
            Lxx = self._stable_cholesky(Kxx)

            v = torch.cholesky_solve(Kxs, Lxx)
            Z_cov = Kss - Kxs.transpose(0, 1) @ v
            Z_cov = 0.5 * (Z_cov + Z_cov.transpose(0, 1))
            Z_cov = Z_cov + 1e-6 * torch.eye(M, device=device)

            Z_mean = torch.zeros(M, Q, device=device)
            for q in range(Q):
                alpha_q = torch.cholesky_solve(mu_all[:, q:q + 1], Lxx)
                Z_mean[:, q] = (Kxs.transpose(0, 1) @ alpha_q).squeeze(-1)
            Z_std = torch.sqrt(torch.diag(Z_cov).unsqueeze(1).expand(M, Q).clamp_min(1e-12))

        if return_full_cov:
            return Z_mean.detach().cpu().numpy(), Z_std.detach().cpu().numpy(), Z_cov.detach().cpu().numpy()
        return Z_mean.detach().cpu().numpy(), Z_std.detach().cpu().numpy()

    def get_prior_covariance(self, x_eval=None, which="train", add_noise=True):
        """Prior covariance K(x, x) at 200 random (sorted) training frame numbers, for the plot."""
        self.eval()
        with torch.no_grad():
            if x_eval is None:
                x_source = np.asarray(self.x) if which == "train" else np.asarray(getattr(self, "xtest", self.x))
                N = x_source.shape[0]
                idx = np.random.choice(N, size=min(200, N), replace=False)
                idx.sort()
                x_eval = x_source[idx]
            x_torch = torch.tensor(x_eval).float().to(self.device)
            if x_torch.dim() == 1:
                x_torch = x_torch.unsqueeze(-1)
            x_norm = (x_torch - self.X_mean) / self.X_std
            cov_mat = self.gp_prior.covar_module(x_norm).evaluate().cpu().numpy()
            if add_noise and hasattr(self.gp_prior, "likelihood"):
                np.fill_diagonal(cov_mat, cov_mat.diagonal() + self.gp_prior.likelihood.noise.item())
            elif add_noise:
                cov_mat += 1e-4 * np.eye(cov_mat.shape[0])
            return cov_mat

    # ------------------------------------------------------------------------------------------------------
    # saving and loading
    # ------------------------------------------------------------------------------------------------------
    def state_dict_full(self):
        state = super().state_dict_full()
        state.update({
            "gp_prior": self.gp_prior.state_dict(),
            "kernel_gp_prior": self.kernel_gp_prior,
            "mu_all": self.mu_all,
            "use_decoder_gan": self.use_decoder_gan,
            "discriminator": self.discriminator.state_dict() if self.discriminator is not None else None,
            "loss_adv_d_list": self.loss_adv_d_list,
        })
        return state

    def load_state_dict_full(self, state):
        super().load_state_dict_full(state)
        self.kernel_gp_prior = state.get("kernel_gp_prior", self.kernel_gp_prior)
        self.mu_all = state.get("mu_all", self.mu_all)
        self.loss_adv_d_list = state.get("loss_adv_d_list", [])
        self.gp_prior = self._new_gp()  # a new GP object (finding B15), as in the original
        if state.get("gp_prior", None) is not None:
            self.gp_prior.load_state_dict(state["gp_prior"])
        if self.use_decoder_gan and self.discriminator is not None and state.get("discriminator") is not None:
            self.discriminator.load_state_dict(state["discriminator"])

    def checkpoint_state(self) -> dict:
        state = super().checkpoint_state()
        state["gp_prior"] = self.gp_prior.state_dict()
        return state

    def load_checkpoint_state(self, state: dict) -> None:
        super().load_checkpoint_state(state)
        self.gp_prior.load_state_dict(state["gp_prior"])  # into the existing GP: the optimiser holds it
