"""Build and train the tiny equivalence runs with the refactored package (the counterpart of
``record_legacy.py``, which does the same with the original code)."""
from __future__ import annotations

from gpvae_endo.config import load_model_settings
from gpvae_endo.models.gpvae_casale import GPVAECasaleModel
from gpvae_endo.models.gpvae_neighbours import GPVAEHPAModel, GPVAESPAModel
from gpvae_endo.models.msvq_vae import MSVQVAEModel
from gpvae_endo.models.vae import VAEModel

# config_endo.py seeds the global RNGs when executed, so read it now (as the original did at import),
# never after the run has been seeded.
BASE_SETTINGS = load_model_settings()

CLASSES = {"VAE": VAEModel, "GPVAE_Casale": GPVAECasaleModel, "GPVAE_HPA": GPVAEHPAModel,
           "GPVAE_SPA": GPVAESPAModel, "MSVQ_VAE": MSVQVAEModel}


def tiny_settings(overrides: dict):
    """config_endo's ModelSettings with the harness overrides (given as UPPER_CASE config names)."""
    return BASE_SETTINGS.replace(**{k.lower(): v for k, v in overrides.items()})


def build(kind, args, overrides):
    return CLASSES[kind](**args, settings=tiny_settings(overrides))


def train(model, overrides, checkpoint=None):
    if checkpoint is None:
        model.train()
    else:
        model.train(checkpoint=checkpoint)
