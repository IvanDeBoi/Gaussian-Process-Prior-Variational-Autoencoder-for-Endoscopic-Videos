"""Build a model by name and work out the file it is saved to.

Model names (``MODELS`` in the config): ``VAE``, ``GPVAE_Casale``, ``GPVAE_SPA``, ``GPVAE_HPA``, ``MSVQ_VAE``.
The constructor arguments and the file names are exactly those of the original driver, so models saved by
the original code are found and loaded again.

Note (kept from the original): ``max_beta`` is not passed, so each model uses its own default
(VAE 1e-2; the GP models 1e-3; MSVQ has no KL term). MSVQ_VAE ignores the encoder/decoder choice and always
uses batch size 8.
"""
from __future__ import annotations

from gpvae_endo.config import ExperimentConfig, ModelSettings

MODEL_NAMES = ("VAE", "GPVAE_Casale", "GPVAE_SPA", "GPVAE_HPA", "MSVQ_VAE")


def _suffix(cfg: ExperimentConfig, test_frac: float) -> str:
    return (f"L_{cfg.must_use_lpips}_{cfg.lpips_res} M_{cfg.must_use_mask} G_{cfg.must_use_decoder_gan} "
            f"GAP_{cfg.must_use_gap_for_test}_{cfg.gap_size} TF_{int(test_frac * 100)}.pth")


def model_file_name(kind: str, encoder: str, decoder: str, cfg: ExperimentConfig, test_frac: float) -> str:
    """File name (inside ``FOLDER_TO_SAVE_MODELS``) of a trained model, as the original driver named it."""
    video, res, dim = cfg.substring_filter, cfg.resolution, cfg.latent_dim
    if kind == "VAE":
        return f"VAE {video} {res} {encoder} {decoder} {dim}D " + _suffix(cfg, test_frac)
    if kind in ("GPVAE_Casale", "GPVAE_SPA", "GPVAE_HPA"):
        return f"{kind} {video} {res} {cfg.kernel} {encoder} {decoder} {dim}D " + _suffix(cfg, test_frac)
    if kind == "MSVQ_VAE":
        return f"MSVQ_VAE {video} {res} {dim}D " + _suffix(cfg, test_frac)
    raise ValueError(f"Unknown model: {kind}")


def build_model(kind: str, encoder: str, decoder: str, x_train, Y_train, M_train, cfg: ExperimentConfig,
                settings: ModelSettings):
    """Construct (untrained) model `kind`. Construction draws random numbers (initial weights; the GP's
    initial hyper-parameters), so the order of calls matters for reproducibility."""
    M = M_train if cfg.must_use_mask else None
    res = cfg.resolution
    C = 3 if cfg.must_preserve_color else 1
    common = dict(x=x_train, Y=Y_train, M=M, device="gpu", latent_dim=cfg.latent_dim, lr=cfg.learning_rate,
                  epochs=cfg.epochs, encoder=encoder, decoder=decoder, batch_size=cfg.batch_size, settings=settings)

    if kind == "VAE":
        from gpvae_endo.models.vae import VAEModel
        return VAEModel(img_H=res, img_W=res, img_C=3, **common)
    if kind == "GPVAE_Casale":
        from gpvae_endo.models.gpvae_casale import GPVAECasaleModel
        return GPVAECasaleModel(img_H=res, img_W=res, img_C=C, kernel_gp_prior=cfg.kernel, **common)
    if kind == "GPVAE_SPA":
        from gpvae_endo.models.gpvae_neighbours import GPVAESPAModel
        return GPVAESPAModel(img_H=res, img_W=res, img_C=C, kernel_gp_prior=cfg.kernel,
                             n_neighbors=cfg.nr_of_neighbours, **common)
    if kind == "GPVAE_HPA":
        from gpvae_endo.models.gpvae_neighbours import GPVAEHPAModel
        return GPVAEHPAModel(img_H=res, img_W=res, img_C=C, kernel_gp_prior=cfg.kernel,
                             n_neighbors=cfg.nr_of_neighbours, **common)
    if kind == "MSVQ_VAE":
        from gpvae_endo.models.msvq_vae import MSVQVAEModel
        print("Using MSVQ-VAE (MSB encoder/decoder, VQ bottleneck)")
        return MSVQVAEModel(
            x=x_train, Y=Y_train, M=M, device="gpu", img_H=res, img_W=res, img_C=3,
            codebook_size=cfg.codebook_size, code_dim=cfg.latent_dim, beta_commit=0.25, ema_decay=0.99,
            epochs=cfg.epochs, lr=cfg.learning_rate, batch_size=8, settings=settings,
        )
    raise ValueError(f"Unknown model: {kind}")
