"""The faster GPVAE_HPA KL formula gives the same value as the original one, up to float rounding.

``original_hpa_kl`` below is the original code (commit aaf8b6d, ``Models/GPVAE_HPA_model.py``), copied
verbatim apart from taking the model as an argument. It copies the Cholesky factor once per latent
dimension; the new code shares one inverse per block.
"""
from __future__ import annotations

import pytest
import torch

import harness
import refactored


def original_hpa_kl(model, mu_all, sigma2_all, batch_idx):
    device = model.device
    b = batch_idx.size(0)
    _, Q = mu_all.shape
    nb_idx_batch = model.neighbour_idx[batch_idx]
    block_global = torch.cat([batch_idx.unsqueeze(1), nb_idx_batch], dim=1)
    k = block_global.size(1)
    mu_block = mu_all[block_global].permute(0, 2, 1).contiguous()
    s2_block = sigma2_all[block_global].permute(0, 2, 1).contiguous()
    X_block = model.Xtensor_norm[block_global]
    K_loc = model.gp_prior.covar_module(X_block).evaluate()
    eye = torch.eye(k, device=device).unsqueeze(0)
    K_loc = K_loc + model.gp_prior.likelihood.noise * eye
    K_loc = K_loc + 1e-6 * eye
    L = torch.linalg.cholesky(K_loc)
    logdetK = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + 1e-12), dim=-1)
    BQ = b * Q
    mu_block_flat = mu_block.view(BQ, k)
    s2_block_flat = s2_block.view(BQ, k)
    L_expanded = L.unsqueeze(1).expand(b, Q, k, k).reshape(BQ, k, k)
    logdetK_flat = logdetK.unsqueeze(1).expand(b, Q).reshape(BQ)
    K_inv_mu = torch.cholesky_solve(mu_block_flat.unsqueeze(-1), L_expanded).squeeze(-1)
    K_inv_S = torch.cholesky_solve(torch.diag_embed(s2_block_flat), L_expanded)
    tr_term = torch.diagonal(K_inv_S, dim1=-2, dim2=-1).sum(dim=-1)
    quad_term = torch.sum(mu_block_flat * K_inv_mu, dim=-1)
    ent_term = torch.sum(torch.log(s2_block_flat + 1e-12), dim=-1)
    kl_block = (0.5 * (tr_term + quad_term - ent_term + logdetK_flat - k)).view(b, Q)
    return kl_block.mean() / k


@pytest.fixture(scope="module")
def hpa_model():
    torch.set_num_threads(harness.THREADS)
    harness.seed_all(0)
    args = harness.model_args("GPVAE_HPA", "EndoVAEEnc", "EndoVAEDec", epochs=1)
    return refactored.build("GPVAE_HPA", args, harness.TINY_SETTINGS)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_kl_as_original(hpa_model, seed):
    g = torch.Generator().manual_seed(seed)
    n, q = hpa_model.Ytensor.shape[0], harness.LATENT
    mu_all = torch.randn(n, q, generator=g) * 2.0
    sigma2_all = torch.exp(torch.randn(n, q, generator=g))
    batch_idx = torch.randperm(n, generator=g)[: harness.BATCH]
    with torch.no_grad():
        new = hpa_model.hpa_kl_batch(mu_all, sigma2_all, batch_idx)
        old = original_hpa_kl(hpa_model, mu_all, sigma2_all, batch_idx)
    assert torch.isfinite(new)
    torch.testing.assert_close(new, old, rtol=1e-5, atol=0.0)


def test_gradients_to_gp_hyperparameters_agree(hpa_model):
    """The HPA KL trains the GP's lengthscale, scale and noise; their gradients must agree too."""
    g = torch.Generator().manual_seed(5)
    n, q = hpa_model.Ytensor.shape[0], harness.LATENT
    mu_all = torch.randn(n, q, generator=g)
    sigma2_all = torch.exp(torch.randn(n, q, generator=g))
    batch_idx = torch.arange(harness.BATCH)
    params = dict(hpa_model.gp_prior.named_parameters())
    grads = []
    for fn in (hpa_model.hpa_kl_batch, lambda *a: original_hpa_kl(hpa_model, *a)):
        for p in params.values():
            p.grad = None
        fn(mu_all, sigma2_all, batch_idx).backward()
        # the GP mean does not enter the KL and gets no gradient in either version
        grads.append({n: p.grad.clone() for n, p in params.items() if p.grad is not None})
    new, old = grads
    assert set(new) == set(old) and any("lengthscale" in n for n in new) and any("noise" in n for n in new)
    for name in new:
        torch.testing.assert_close(new[name], old[name], rtol=1e-4, atol=1e-7, msg=name)
