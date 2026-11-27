"""Settings: which video, which models, how long to train, what to compute afterwards.

Settings live in plain Python files (``config_endo.py`` and one file per video in ``configs/``), as in the
original code. This module reads such a file and turns it into two typed objects:

* :class:`ExperimentConfig` -- what to run: video, models, epochs, batch size, test split, which metrics and
  plots. Read from the file given with ``--config`` (default ``config_endo.py``).
* :class:`ModelSettings` -- training details the model code uses: LR warm-up, KL ramp, loss weights,
  early stopping, LoRA, GAN, LPIPS resolution.

**Important (kept from the original, finding B21):** the original model code imported its
:class:`ModelSettings` values from ``config_endo.py`` directly, whatever ``--config`` said. To keep results
identical, :func:`load_model_settings` still reads them from ``config_endo.py``. If the ``--config`` file
sets one of them to a different value, a warning is printed, because that value is ignored. (All files in
``configs/`` currently use the same values as ``config_endo.py``.)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, fields
from functools import lru_cache
from types import ModuleType

from gpvae_endo.paths import REPO_ROOT

BASE_CONFIG = REPO_ROOT / "config_endo.py"


def load_config_module(config_path: str | os.PathLike | None) -> tuple[str, ModuleType]:
    """Execute a Python config file and return ``(absolute path, module)``.

    ``None`` means ``config_endo.py``; a path without ``.py`` gets it appended if that file exists.
    """
    config_path = str(config_path or "config_endo.py")
    if not config_path.endswith(".py") and os.path.exists(f"{config_path}.py"):
        config_path = f"{config_path}.py"
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    config_dir = os.path.dirname(config_path)
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)
    spec = importlib.util.spec_from_file_location("runtime_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return config_path, module


def _from_module(cls, module: ModuleType, source: str):
    """Build dataclass `cls` from the UPPER_CASE names of a config module (field `foo_bar` <- `FOO_BAR`)."""
    missing = [f.name.upper() for f in fields(cls) if not hasattr(module, f.name.upper())]
    if missing:
        raise AttributeError(f"{source} is missing required config values: {', '.join(missing)}")
    return cls(**{f.name: getattr(module, f.name.upper()) for f in fields(cls)})


# --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    """What one run of the pipeline does. Field ``foo_bar`` is ``FOO_BAR`` in the config file."""

    # which video
    substring_filter: str          # video name, e.g. "c1_descending_t1_v2"
    resolution: int                # frame size in pixels (336)
    substring_normalisation: str   # which NPZ normalisation to load ("m01sG")
    img_src_folder: str
    # folders
    folder_data: str
    folder_output: str
    folder_to_save_reconstructed: str
    folder_to_save_frames_for_debugging: str
    folder_logs: str
    folder_to_save_models: str
    # what to train
    models: list
    encoders: list
    decoders: list
    allowed_pairs_per_model: dict
    epochs: int
    latent_dim: int
    kernel: str
    nr_of_neighbours: int          # neighbours per block in GPVAE_HPA / GPVAE_SPA
    codebook_size: int             # MSVQ_VAE only
    batch_size: int
    learning_rate: float
    seed: int
    device: str
    must_always_train_from_scratch: bool
    # test frames
    percentage_for_test: float
    percentage_for_tail: float
    must_use_gap_for_test: bool
    gap_size: int
    use_x_scaling: bool
    # loss switches (used for file names and the summary; the loss itself reads ModelSettings)
    must_use_lpips: bool
    lpips_res: int
    must_use_mask: bool
    must_use_decoder_gan: bool
    must_preserve_color: bool
    # what to compute and draw after training
    must_calculate_ssim: bool
    must_calculate_psnr: bool
    must_calculate_fid: bool
    must_plot_2d_latent: bool
    must_plot_pca: bool
    must_plot_covariance: bool
    must_reconstruct_all: bool
    must_save_reconstruction_grids: bool
    must_time_encode_new_images: bool

    @classmethod
    def from_module(cls, module: ModuleType, source: str = "config") -> "ExperimentConfig":
        return _from_module(cls, module, source)


@dataclass(frozen=True)
class ModelSettings:
    """Training details read by the model code (see the module docstring for where they come from)."""

    warmup_epochs: int                    # LR warm-up length; also the epochs with KL weight 0
    ramp_epochs: int                      # KL weight ramps from 0 to max over this many epochs
    lpips_res: int                        # frames are resized to this size for the LPIPS loss
    loss_weights: dict                    # {"mse", "lpips", "gan", "kl"}
    early_stopping_patience: int
    early_stopping_start_fraction: float  # early stopping only after this fraction of the epochs
    lora_rank: int
    lora_alpha: int
    lora_lr: float
    lora_target_keywords: tuple
    min_spatial: int                      # CNN encoders downsample until the feature map is this small
    use_x_scaling: bool                   # chooses the GP lengthscale bounds (frames vs [0, 1])
    must_use_decoder_gan: bool
    gan_type: str
    gan_loss: str
    gan_disc_weight: float
    gan_lr: float
    gan_betas: tuple
    gan_base_channels: int
    gan_num_layers: int
    gan_norm: str
    gan_use_mask: bool
    gan_detach_recon_for_d: bool
    gan_label_smoothing: float
    gan_disc_updates: int
    gan_warmup_epochs: int
    gan_feature_matching_weight: float

    @classmethod
    def from_module(cls, module: ModuleType, source: str = "config") -> "ModelSettings":
        return _from_module(cls, module, source)

    def replace(self, **changes) -> "ModelSettings":
        from dataclasses import replace
        return replace(self, **changes)


@lru_cache(maxsize=1)
def _base_module() -> ModuleType:
    return load_config_module(BASE_CONFIG)[1]


def load_model_settings() -> ModelSettings:
    """The :class:`ModelSettings` from ``config_endo.py`` (read once per process, like an import)."""
    return ModelSettings.from_module(_base_module(), str(BASE_CONFIG))


def warn_ignored_model_settings(run_module: ModuleType, settings: ModelSettings) -> list[str]:
    """Print a warning for every ModelSettings value the --config file sets differently (it is ignored)."""
    ignored = []
    for f in fields(ModelSettings):
        name = f.name.upper()
        if hasattr(run_module, name) and getattr(run_module, name) != getattr(settings, f.name):
            ignored.append(f"{name}={getattr(run_module, name)!r} (used: {getattr(settings, f.name)!r})")
    if ignored:
        print("WARNING: these values in the --config file are ignored; the model code uses the ones in "
              "config_endo.py (as the original code did): " + "; ".join(ignored))
    return ignored
