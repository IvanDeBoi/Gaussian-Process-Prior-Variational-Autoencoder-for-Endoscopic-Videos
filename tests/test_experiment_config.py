"""configs/experiment.py: the settings for one run, from config_endo.py plus GPVAE_* environment variables."""
from pathlib import Path

import pytest

from gpvae_endo.config import ExperimentConfig, load_config_module

REPO = Path(__file__).resolve().parents[1]


def experiment_config(monkeypatch, **env):
    for k in ("GPVAE_VIDEO", "GPVAE_DATA_DIR", "GPVAE_MODELS", "GPVAE_ENCODERS", "GPVAE_SEED", "GPVAE_EPOCHS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _, module = load_config_module(REPO / "configs" / "experiment.py")
    return ExperimentConfig.from_module(module, "experiment.py")


def combos(cfg):
    return [(m, e, d) for m in cfg.models for e in cfg.encoders for d in cfg.decoders
            if (e, d) in cfg.allowed_pairs_per_model.get(m, set())]


def test_experiment_config_defaults_give_the_nine_combinations(monkeypatch):
    cfg = experiment_config(monkeypatch, GPVAE_VIDEO="c1_descending_t1_v2", GPVAE_DATA_DIR="/data/c3vd")
    assert cfg.substring_filter == "c1_descending_t1_v2" and cfg.folder_data == "/data/c3vd"
    assert cfg.epochs == 300 and cfg.percentage_for_test == 20 and cfg.seed == 42
    assert sorted(combos(cfg)) == sorted((m, e, "EndoVAEDec") for m in ("VAE", "GPVAE_HPA", "GPVAE_SPA")
                                         for e in ("EndoVAEEnc", "GastroEnc", "GaLoRaEnc"))
    # everything else is config_endo.py's
    _, base = load_config_module(REPO / "config_endo.py")
    assert cfg.batch_size == base.BATCH_SIZE == 64 and cfg.latent_dim == base.LATENT_DIM
    assert cfg.lpips_res == base.LPIPS_RES and cfg.must_calculate_fid == base.MUST_CALCULATE_FID


def test_experiment_config_one_combination_per_job(monkeypatch):
    cfg = experiment_config(monkeypatch, GPVAE_VIDEO="v", GPVAE_MODELS="GPVAE_HPA", GPVAE_ENCODERS="GaLoRaEnc",
                     GPVAE_SEED="7")
    assert combos(cfg) == [("GPVAE_HPA", "GaLoRaEnc", "EndoVAEDec")] and cfg.seed == 7 and cfg.epochs == 300
    assert experiment_config(monkeypatch, GPVAE_VIDEO="v", GPVAE_EPOCHS="3").epochs == 3


def test_experiment_config_rejects_a_typo_and_a_missing_video(monkeypatch):
    with pytest.raises(ValueError, match="not in ALLOWED_PAIRS"):
        experiment_config(monkeypatch, GPVAE_VIDEO="v", GPVAE_MODELS="GPVAE_HAP")
    with pytest.raises(KeyError, match="GPVAE_VIDEO"):
        experiment_config(monkeypatch)


def test_a_run_creates_its_output_folders(monkeypatch, tmp_path):
    """A new run folder starts empty; the original crashed there at its first log line (no Logs/)."""
    from gpvae_endo.paths import OutputPaths

    cfg = experiment_config(monkeypatch, GPVAE_VIDEO="vid_k")
    monkeypatch.chdir(tmp_path)
    OutputPaths.from_config(cfg).create()
    for sub in ("Logs", "Pics", "Saved_Models", "Recon/vid_k/Frames", "Debug/vid_k"):
        assert (tmp_path / sub).is_dir(), sub
