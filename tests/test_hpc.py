"""Cluster scripts: configs/hpc.py, slurm/make_job_list.py, slurm/collect_results.py, and dry runs of the job
scripts (GPVAE_DRY_RUN=1: bash runs the real scripts with a simulated Slurm environment; the Python step and
sbatch are only printed). Nothing here talks to a cluster.
"""
import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from gpvae_endo.config import ExperimentConfig, load_config_module

REPO = Path(__file__).resolve().parents[1]
SLURM = REPO / "slurm"
SCRIPTS = [SLURM / "train.slurm", SLURM / "prepare.slurm", SLURM / "common.sh"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, SLURM / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_job_list = load("make_job_list")
collect_results = load("collect_results")


# --- configs/hpc.py --------------------------------------------------------------------------------------

def hpc_config(monkeypatch, **env):
    for k in ("GPVAE_VIDEO", "GPVAE_DATA_DIR", "GPVAE_MODELS", "GPVAE_ENCODERS", "GPVAE_SEED", "GPVAE_EPOCHS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _, module = load_config_module(REPO / "configs" / "hpc.py")
    return ExperimentConfig.from_module(module, "hpc.py")


def combos(cfg):
    return [(m, e, d) for m in cfg.models for e in cfg.encoders for d in cfg.decoders
            if (e, d) in cfg.allowed_pairs_per_model.get(m, set())]


def test_hpc_config_defaults_give_the_nine_combinations(monkeypatch):
    cfg = hpc_config(monkeypatch, GPVAE_VIDEO="c1_descending_t1_v2", GPVAE_DATA_DIR="/data/c3vd")
    assert cfg.substring_filter == "c1_descending_t1_v2" and cfg.folder_data == "/data/c3vd"
    assert cfg.epochs == 300 and cfg.percentage_for_test == 20 and cfg.seed == 42
    assert sorted(combos(cfg)) == sorted((m, e, "EndoVAEDec") for m in ("VAE", "GPVAE_HPA", "GPVAE_SPA")
                                         for e in ("EndoVAEEnc", "GastroEnc", "GaLoRaEnc"))
    # everything else is config_endo.py's
    _, base = load_config_module(REPO / "config_endo.py")
    assert cfg.batch_size == base.BATCH_SIZE == 64 and cfg.latent_dim == base.LATENT_DIM
    assert cfg.lpips_res == base.LPIPS_RES and cfg.must_calculate_fid == base.MUST_CALCULATE_FID


def test_hpc_config_one_combination_per_job(monkeypatch):
    cfg = hpc_config(monkeypatch, GPVAE_VIDEO="v", GPVAE_MODELS="GPVAE_HPA", GPVAE_ENCODERS="GaLoRaEnc",
                     GPVAE_SEED="7")
    assert combos(cfg) == [("GPVAE_HPA", "GaLoRaEnc", "EndoVAEDec")] and cfg.seed == 7 and cfg.epochs == 300
    assert hpc_config(monkeypatch, GPVAE_VIDEO="v", GPVAE_EPOCHS="3").epochs == 3


def test_hpc_config_rejects_a_typo_and_a_missing_video(monkeypatch):
    with pytest.raises(ValueError, match="not in ALLOWED_PAIRS"):
        hpc_config(monkeypatch, GPVAE_VIDEO="v", GPVAE_MODELS="GPVAE_HAP")
    with pytest.raises(KeyError, match="GPVAE_VIDEO"):
        hpc_config(monkeypatch)


def test_a_run_creates_its_output_folders(monkeypatch, tmp_path):
    """A cluster run folder starts empty; the original crashed there at its first log line (no Logs/)."""
    from gpvae_endo.paths import OutputPaths

    cfg = hpc_config(monkeypatch, GPVAE_VIDEO="vid_k")
    monkeypatch.chdir(tmp_path)
    OutputPaths.from_config(cfg).create()
    for sub in ("Logs", "Pics", "Saved_Models", "Recon/vid_k/Frames", "Debug/vid_k"):
        assert (tmp_path / sub).is_dir(), sub


# --- job list and results ---------------------------------------------------------------------------------

def test_make_job_list(tmp_path, capsys):
    data = tmp_path / "c3vd"
    data.mkdir()
    for v in ("vid_b", "vid_a"):
        (data / f"{v} 336 X Ym01sGM.npz").write_bytes(b"")
    (data / "vid_a 336 X Y01M.npz").write_bytes(b"")  # other normalisations are ignored
    out = tmp_path / "jobs.txt"
    assert make_job_list.main(["c3vd", "--data-root", str(tmp_path), "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 * 9 and lines[0] == "c3vd vid_a VAE EndoVAEEnc 42" and lines[1] == "c3vd vid_b VAE EndoVAEEnc 42"
    assert len(set(lines)) == len(lines)
    assert "--array=0-17%4" in capsys.readouterr().out


def test_collect_results_reads_the_pipelines_summary(tmp_path):
    from gpvae_endo.reporting.summary import build_experiment_summary_text

    class Cfg:
        substring_filter, resolution, latent_dim, nr_of_neighbours = "vid_a", 336, 256, 50
        must_use_lpips, lpips_res, must_use_decoder_gan, kernel = True, 256, False, "CAUCHY"
        must_use_gap_for_test, gap_size = False, 10

    row = dict(model_from_models="GPVAE_HPA", encoder="EndoVAEEnc", decoder="EndoVAEDec",
               psnr_test_image_encoded=31.25, psnr_test_time_encoded=27.5, ssim_test_image_encoded=0.8,
               rmse_test_encoded_image=0.01, time_sec=3600.0, epochs=300, time_per_epoch=10.5)
    run = tmp_path / "c3vd" / "vid_a" / "GPVAE_HPA-EndoVAEEnc" / "seed42"
    run.mkdir(parents=True)
    old = build_experiment_summary_text([dict(row, psnr_test_image_encoded=1.0)], Cfg, 0.2, 0, 1)
    (run / "Experiment_Summaries.txt").write_text(build_experiment_summary_text([row], Cfg, 0.2, 0, 1) + "\n" + old,
                                                  encoding="utf-8")
    (run / "DONE").write_text("x")
    (tmp_path / "c3vd" / "vid_b" / "VAE-EndoVAEEnc" / "seed42").mkdir(parents=True)  # still running

    rows, missing = collect_results.collect(tmp_path)
    assert len(rows) == 1 and len(missing) == 1
    r = rows[0]
    assert (r["dataset"], r["video"], r["model"], r["encoder"], r["seed"], r["finished"]) == \
        ("c3vd", "vid_a", "GPVAE_HPA", "EndoVAEEnc", 42, True)
    assert r["PSNR_te_im"] == 31.25 and r["PSNR_te_t"] == 27.5 and r["FID_te_im"] is None and r["epochs"] == 300
    assert collect_results.main([str(tmp_path)]) == 0 and (tmp_path / "results.csv").exists()


# --- job scripts ------------------------------------------------------------------------------------------

def find_bash():
    """Git Bash on Windows, bash elsewhere (not the WSL launcher in System32)."""
    path = shutil.which("bash")
    if path and os.name == "nt" and re.search(r"system32|windowsapps", path, re.I):
        path = None
        git = shutil.which("git")
        if git:
            for cand in (Path(git).parents[1] / "usr" / "bin" / "bash.exe", Path(git).parents[1] / "bin" / "bash.exe"):
                if cand.exists():
                    path = str(cand)
                    break
    return path


BASH = find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash found")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_lf_line_endings(path):
    assert b"\r" not in path.read_bytes()


@needs_bash
@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax(path):
    r = subprocess.run([BASH, "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def posix(p) -> str:
    return str(p).replace("\\", "/")


def dry(script, tmp_path, **env):
    e = {k: os.environ[k] for k in ("SYSTEMROOT", "TEMP", "TMP") if k in os.environ}
    e.update(VSC_SCRATCH=posix(tmp_path), GPVAE_REPO=posix(REPO), GPVAE_DRY_RUN="1", SLURM_JOB_ID="1234",
             SLURM_JOB_ACCOUNT="ap_test", SLURM_JOB_PARTITION="ampere_gpu", SLURM_JOB_NAME="gpvae-legacy",
             SLURM_JOB_START_TIME="1000", SLURM_JOB_END_TIME="4600", SLURM_ARRAY_TASK_ID="1")
    e.update({k: str(v) for k, v in env.items()})
    r = subprocess.run([BASH, str(SLURM / script)], env=e, capture_output=True, text=True, timeout=60)
    python = [ln.split("DRY-RUN python: ", 1)[1] for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN python: ")]
    sbatch = [ln.split("DRY-RUN sbatch: ", 1)[1] for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN sbatch: ")]
    return r, python, sbatch


def job_list(tmp_path):
    jobs = tmp_path / "jobs.txt"  # Windows line endings on purpose: they must not reach the fields
    jobs.write_bytes(b"# comment\r\nc3vd vid_a VAE EndoVAEEnc 42\r\n\r\n  # indented comment\r\n"
                     b"kvasir vid_k GPVAE_HPA GaLoRaEnc 1\r\n")
    return jobs


@needs_bash
def test_train_runs_the_line_of_its_array_index(tmp_path):
    runs = tmp_path / "runs"
    r, python, sbatch = dry("train.slurm", tmp_path, JOBS=posix(job_list(tmp_path)), RUNS_ROOT=posix(runs))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "kvasir / vid_k / GPVAE_HPA + GaLoRaEnc / seed 1" in r.stdout
    assert f"data  {posix(tmp_path)}/gpvae-data/kvasir" in r.stdout
    assert python == [f"python -m gpvae_endo run {posix(REPO)}/configs/hpc.py --checkpoint --deadline-margin 1800"]
    assert sbatch == []
    run_dir = runs / "kvasir" / "vid_k" / "GPVAE_HPA-GaLoRaEnc" / "seed1"
    assert (run_dir / "DONE").exists()
    # the pipeline opens Logs/<...>.txt and saves Pics/... without creating the folders itself
    for sub in ("Logs", "Pics", "Saved_Models", "Debug", "Recon"):
        assert (run_dir / sub).is_dir(), sub
    # clean-up after a finished run: checkpoint, model and frames go; summary, plots and video stay
    for keep in (1, 0):
        (run_dir / "DONE").unlink()
        for f in ("Saved_Models/m.pth", "Saved_Models/m.ckpt.pt", "Recon/vid_k/Frames/f.png", "Debug/vid_k/v.mp4",
                  "Pics/p.png", "Experiment_Summaries.txt"):
            (run_dir / f).parent.mkdir(parents=True, exist_ok=True)
            (run_dir / f).write_bytes(b"x")
        r, _, _ = dry("train.slurm", tmp_path, JOBS=posix(job_list(tmp_path)), RUNS_ROOT=posix(runs),
                      KEEP_MODELS=keep, KEEP_FRAMES=keep)
        assert r.returncode == 0, r.stdout + r.stderr
        assert not (run_dir / "Saved_Models/m.ckpt.pt").exists()
        assert (run_dir / "Saved_Models/m.pth").exists() == bool(keep)
        assert (run_dir / "Recon/vid_k/Frames/f.png").exists() == bool(keep)
        for f in ("Debug/vid_k/v.mp4", "Pics/p.png", "Experiment_Summaries.txt", "DONE"):
            assert (run_dir / f).exists(), f
    r, python, _ = dry("train.slurm", tmp_path, JOBS=posix(job_list(tmp_path)), RUNS_ROOT=posix(runs))
    assert r.returncode == 0 and python == [] and "already finished" in r.stdout


@needs_bash
def test_train_resubmits_the_same_index_on_exit_85(tmp_path):
    runs = tmp_path / "runs"
    r, _, sbatch = dry("train.slurm", tmp_path, JOBS=posix(job_list(tmp_path)), RUNS_ROOT=posix(runs),
                       GPVAE_DRY_EXIT=85, GPVAE_RESUBMITS=2, GPVAE_EPOCHS=3)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(sbatch) == 1
    s = sbatch[0]
    for part in ("--account=ap_test", "--partition=ampere_gpu", "--time=60", "--array=1",
                 "--dependency=afterany:1234", f"JOBS={posix(job_list(tmp_path))}", "GPVAE_RESUBMITS=3",
                 f"RUNS_ROOT={posix(runs)}", "GPVAE_EPOCHS=3", f"{posix(REPO)}/slurm/train.slurm"):
        assert part in s, (part, s)
    assert not (runs / "kvasir" / "vid_k" / "GPVAE_HPA-GaLoRaEnc" / "seed1" / "DONE").exists()
    r, _, sbatch = dry("train.slurm", tmp_path, JOBS=posix(job_list(tmp_path)), RUNS_ROOT=posix(runs),
                       GPVAE_DRY_EXIT=85, GPVAE_RESUBMITS=10)
    assert r.returncode == 85 and sbatch == [] and "cap reached" in r.stderr


@needs_bash
def test_train_rejects_a_bad_line(tmp_path):
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("c3vd vid_a VAE\nc3vd vid_a VAE EndoVAEEnc 42 extra\n", encoding="utf-8")
    for idx in (0, 1):
        r, python, _ = dry("train.slurm", tmp_path, JOBS=posix(jobs), SLURM_ARRAY_TASK_ID=idx)
        assert r.returncode == 2 and python == [] and "is not 'dataset video model encoder seed'" in r.stderr


@needs_bash
def test_prepare_video_and_folder(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    videos = tmp_path / "videos.txt"
    videos.write_text("kvasir k1 $VSC_SCRATCH/k1.mp4 1000 2999\n" f"c3vd c1 {posix(frames)}\n", encoding="utf-8")
    out = f"{posix(tmp_path)}/gpvae-data"
    r, python, _ = dry("prepare.slurm", tmp_path, VIDEOS=posix(videos), SLURM_ARRAY_TASK_ID=0)
    assert r.returncode == 0, r.stdout + r.stderr
    assert python == [f"python {posix(REPO)}/scripts/make_npz.py --name k1 --out {out}/kvasir "
                      f"--video {posix(tmp_path)}/k1.mp4 --first 1000 --last 2999"]
    r, python, _ = dry("prepare.slurm", tmp_path, VIDEOS=posix(videos), SLURM_ARRAY_TASK_ID=1)
    assert python == [f"python {posix(REPO)}/scripts/make_npz.py --name c1 --out {out}/c3vd --frames {posix(frames)}"]


@needs_bash
def test_prepare_the_kvasir_list(tmp_path):
    """slurm/videos_kvasir.txt: 9 videos after its comment lines, each its longest clean-mucosa segment."""
    lst = REPO / "slurm" / "videos_kvasir.txt"
    rows = [ln.split() for ln in lst.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(rows) == 9 and len({r[1] for r in rows}) == 9
    for i, (dataset, name, source, first, last) in enumerate(rows):
        assert dataset == "kvasir" and name == f"kvasir_{source.rsplit('/', 1)[1][:-4]}" and int(first) < int(last)
        r, python, _ = dry("prepare.slurm", tmp_path, VIDEOS="slurm/videos_kvasir.txt", SLURM_ARRAY_TASK_ID=i)
        assert r.returncode == 0, r.stdout + r.stderr
        video = source.replace("$VSC_SCRATCH", posix(tmp_path))
        assert python == [f"python {posix(REPO)}/scripts/make_npz.py --name {name} --out {posix(tmp_path)}/gpvae-data/kvasir "
                          f"--video {video} --first {first} --last {last}"]
    assert rows[7][1:] == ["kvasir_7ad22d50ebaf4596", "$VSC_SCRATCH/videos/kvasir/7ad22d50ebaf4596.mp4", "12362", "12631"]
