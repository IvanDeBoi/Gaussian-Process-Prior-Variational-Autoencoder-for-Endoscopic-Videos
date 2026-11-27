"""Settings for one experiment run: config_endo.py, with a few values set from environment variables.

Everything is taken from ``config_endo.py`` (its ``is_on_HPC`` block: batch 64, latent 256, LPIPS at 256 px,
VAE/HPA/SPA x EndoVAEEnc/GastroEnc/GaLoRaEnc with EndoVAEDec), then these values are replaced:

    EPOCHS = 300, PERCENTAGE_FOR_TEST = 20, DECODERS = ['EndoVAEDec'] (['MSVQDec'] for MSVQ_VAE + MSVQEnc)
    SUBSTRING_FILTER   <- GPVAE_VIDEO     (required: the video name, i.e. the NPZ file name before " 336 X ...")
    FOLDER_DATA        <- GPVAE_DATA_DIR  (folder with the NPZ; default "Data/")
    MODELS             <- GPVAE_MODELS    (comma-separated; default VAE,GPVAE_HPA,GPVAE_SPA)
    ENCODERS           <- GPVAE_ENCODERS  (comma-separated; default EndoVAEEnc,GastroEnc,GaLoRaEnc)
    SEED               <- GPVAE_SEED      (default 42, as in config_endo.py)
    EPOCHS             <- GPVAE_EPOCHS    (default 300; e.g. 500 for longer runs, 3 for a smoke test)

Example (Linux / macOS / Git Bash; see the README for PowerShell):

    GPVAE_VIDEO=c1_descending_t1_v2 GPVAE_MODELS=VAE GPVAE_ENCODERS=EndoVAEEnc uv run python -m gpvae_endo run configs/experiment.py

Values the model code reads from config_endo.py itself (loss weights, warm-up, ramp, LoRA, early stopping; see
the README) are not changed here, so they are the same for every run.
"""
import os as _os

_BASE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "config_endo.py")
with open(_BASE, encoding="utf-8") as _f:
    exec(compile(_f.read(), _os.path.abspath(_BASE), "exec"))

if not is_on_HPC:  # noqa: F821 (defined by config_endo.py)
    raise ValueError("configs/experiment.py expects is_on_HPC = True in config_endo.py")


def _env_list(name, default):
    value = _os.environ.get(name, "").strip()
    return [v.strip() for v in value.split(",") if v.strip()] if value else list(default)


try:
    SUBSTRING_FILTER = _os.environ["GPVAE_VIDEO"]
except KeyError:
    raise KeyError("configs/experiment.py: set GPVAE_VIDEO to the video name (the NPZ file name before ' 336 X')") from None
FOLDER_DATA = _os.environ.get("GPVAE_DATA_DIR", "Data/")
IMG_SRC_FOLDER = PREFIX + "C3VDv2/" + SUBSTRING_FILTER  # noqa: F821 (only used by data preparation)

EPOCHS = int(_os.environ.get("GPVAE_EPOCHS", "300"))  # default 300; e.g. 500, or 3 for a smoke test
PERCENTAGE_FOR_TEST = 20
MODELS = _env_list("GPVAE_MODELS", ["VAE", "GPVAE_HPA", "GPVAE_SPA"])
ENCODERS = _env_list("GPVAE_ENCODERS", ["EndoVAEEnc", "GastroEnc", "GaLoRaEnc"])
# Decoder per model: EndoVAEDec, except MSVQ_VAE, which only works with its own MSVQEnc + MSVQDec.
_DECODER_OF = {"MSVQ_VAE": "MSVQDec"}
DECODERS = list(dict.fromkeys(_DECODER_OF.get(_m, "EndoVAEDec") for _m in MODELS))
SEED = int(_os.environ.get("GPVAE_SEED", "42"))

# A typo would otherwise give a run that trains nothing (the pipeline skips pairs that are not allowed).
for _m in MODELS:
    _d = _DECODER_OF.get(_m, "EndoVAEDec")
    for _e in ENCODERS:
        if (_e, _d) not in ALLOWED_PAIRS_PER_MODEL.get(_m, set()):  # noqa: F821
            raise ValueError(f"configs/experiment.py: {_m} + {_e} + {_d} is not in ALLOWED_PAIRS_PER_MODEL")
