"""End-to-end check: the original driver and the refactored pipeline produce the same run.

Both are run, each in its own empty working folder, on the same tiny synthetic video (336x336 frames, as the
driver requires) with the same small config, on the CPU with one thread, twice:

1. training from scratch (compared for information only: two runs of the original itself already differ
   slightly at this size, finding T7; and the B2 fix may make the refactored run end with an earlier
   epoch's weights);
2. loading the models the original saved in step 1 and evaluating them. This must be identical: the run
   logs and the ``Experiment_Summaries.txt`` tables are compared, ignoring clock times and absolute paths.

The original driver is taken from a folder holding commit aaf8b6d (see ``record_legacy.py``), with ONE line
changed: the denormalisation fix B1 (``[:, :, c]`` -> ``[..., c]``), because the refactored code fixes B1
on purpose and every metric depends on it.

    python tests/equivalence/e2e_compare.py <folder with the original code> <scratch folder>

Takes a few minutes on a laptop CPU. Exit code 0 = identical.
"""
from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
N_FRAMES, RES = 16, 336

OVERRIDES = """
# ---- e2e test overrides ----
SUBSTRING_FILTER = "synthetic"
MODELS = ['VAE', 'GPVAE_HPA', 'GPVAE_SPA', 'MSVQ_VAE']
ENCODERS = ['EndoVAEEnc', 'MSVQEnc']
DECODERS = ['EndoVAEDec', 'MSVQDec']
EPOCHS = 3
LATENT_DIM = 8
BATCH_SIZE = 4
CODEBOOK_SIZE = 16
NR_OF_NEIGHBOURS = 4
PERCENTAGE_FOR_TEST = 25
PERCENTAGE_FOR_TAIL = 0
FOLDER_TO_SAVE_RECONSTRUCTED = "Recon/"
MUST_RECONSTRUCT_ALL = True
MUST_CALCULATE_FID = True
MUST_CALCULATE_SSIM = True
MUST_CALCULATE_PSNR = True
MUST_PLOT_COVARIANCE = True
DEVICE = "cpu"
"""


def make_npz(path: Path) -> None:
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:RES, 0:RES] / RES
    frames = []
    for k in range(N_FRAMES):
        cx, cy = 0.3 + 0.4 * k / N_FRAMES, 0.5 + 0.2 * np.sin(k / 3)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.03)
        frames.append(np.stack([0.7 * blob + 0.2, 0.4 * blob + 0.2 * xx, 0.3 * blob + 0.2 * yy], axis=-1))
    frames = np.clip(np.asarray(frames) + 0.02 * rng.standard_normal((N_FRAMES, RES, RES, 3)), 0, 1)
    means = frames.reshape(-1, 3).mean(0)
    stds = frames.reshape(-1, 3).std(0)
    Y = ((frames - means) / stds).reshape(N_FRAMES, -1).astype(np.float32)
    M = (rng.random((N_FRAMES, RES * RES)) > 0.05).astype(np.float32)
    x = np.arange(N_FRAMES, dtype=np.float32).reshape(-1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, Y=Y, M=M, means=means.astype(np.float32), stds=stds.astype(np.float32))


def prepare_workdir(workdir: Path, npz: Path) -> Path:
    if workdir.exists():
        shutil.rmtree(workdir)
    for sub in ("Data", "Logs", "Pics", "Debug", "Saved_Models", "Recon"):
        (workdir / sub).mkdir(parents=True)
    shutil.copy(npz, workdir / "Data" / npz.name)
    cfg = workdir / "config_e2e.py"
    cfg.write_text((REPO / "config_endo.py").read_text(encoding="utf-8") + OVERRIDES, encoding="utf-8")
    return cfg


def patched_legacy(legacy: Path, scratch: Path) -> Path:
    """A copy of the original code with only the B1 fix applied to its driver."""
    dst = scratch / "legacy_b1"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(legacy, dst, ignore=shutil.ignore_patterns("Weights", "__pycache__", "Data", "*.pth"))
    driver = dst / "Latent_Space_Discovery.py"
    src = driver.read_text(encoding="utf-8")
    fixed, n = re.subn(r"(denormalised_arr|normalised_image)\[:, :, (\d)\]", r"\1[..., \2]", src)
    assert n == 9, f"expected 9 B1 index expressions in the original denormalise(), found {n}"
    driver.write_text(fixed, encoding="utf-8")
    return driver


def start(cmd, cwd: Path, extra_env: dict) -> subprocess.Popen:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "GPVAE_ALLOW_CPU": "1",
           "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8", **extra_env}
    print("running:", " ".join(map(str, cmd)), "in", cwd, flush=True)
    log = open(cwd / "stdout.txt", "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait(proc: subprocess.Popen, cwd: Path) -> None:
    if proc.wait() != 0:
        print((cwd / "stdout.txt").read_text(encoding="utf-8", errors="replace")[-4000:])
        raise SystemExit(f"command failed with exit code {proc.returncode}; see {cwd / 'stdout.txt'}")


_VOLATILE = [
    (re.compile(r"\d{4}-\d{2}-\d{2}[ _]\d{2}[:-]\d{2}[:-]\d{2}"), "<datetime>"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}\b"), "<hms>"),
    (re.compile(r"Using config file: .*"), "Using config file: <path>"),
    (re.compile(r"Loading from: .*"), "Loading from: <path>"),
    (re.compile(r"\(\d+\.\d s\)"), "(<s>)"),
    (re.compile(r"\|\s+\d+\.\d{3}$"), "| <t/epoch>"),
]
# lines only one side prints: the refactored code prints the duplicate train FID once (finding B8)
_SKIP = ("WARNING: these values in the --config file are ignored", "WARNING: no CUDA GPU found")


def normalise(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        if line.startswith(_SKIP):
            continue
        for pat, rep in _VOLATILE:
            line = pat.sub(rep, line)
        out.append(line.rstrip())
    return out


def drop_first_duplicate_fid(lines: list[str]) -> list[str]:
    """The original computes and prints the training-set FID twice in a row (B8); keep the second."""
    out, i = [], 0
    marker = "Computing FID on"
    while i < len(lines):
        if (lines[i].startswith(marker) and i + 2 < len(lines) and lines[i + 2] == lines[i]
                and lines[i + 1].startswith("FID image decoded train")):
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return out


def compare_runs(old_dir: Path, new_dir: Path, label: str, show: int = 60) -> bool:
    ok = True
    a = normalise((old_dir / "Experiment_Summaries.txt").read_text(encoding="utf-8"))
    b = normalise((new_dir / "Experiment_Summaries.txt").read_text(encoding="utf-8"))
    diff = list(difflib.unified_diff(a, b, "original", "refactored", lineterm="", n=0))
    print(f"\n== {label}: summary table {'IDENTICAL' if not diff else 'DIFFERENT'}")
    print("\n".join(diff[:show]))
    ok &= not diff
    log_a = drop_first_duplicate_fid(normalise(next((old_dir / "Logs").glob("run log*.txt")).read_text("utf-8")))
    log_b = normalise(next((new_dir / "Logs").glob("run log*.txt")).read_text("utf-8"))
    diff = list(difflib.unified_diff(log_a, log_b, "original log", "refactored log", lineterm="", n=0))
    print(f"== {label}: run log ({len(log_a)} lines) {'IDENTICAL' if not diff else f'{len(diff)} diff lines'}")
    print("\n".join(diff[:show]))
    return ok and not diff


def main(argv) -> int:
    legacy, scratch = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    npz = scratch / "synthetic 336 X Ym01sGM.npz"
    make_npz(npz)
    driver = patched_legacy(legacy, scratch)
    new_env = {"PYTHONPATH": str(REPO)}

    # 1. train from scratch with both. Informational only: on this CPU even two runs of the ORIGINAL differ
    #    in the 6th-7th significant digit (finding T7), so training is compared bit for bit by the tiny
    #    runs in tests/test_equivalence.py instead.
    old_dir, new_dir = scratch / "train_original", scratch / "train_refactored"
    procs = [start([sys.executable, str(driver), "--config", str(prepare_workdir(old_dir, npz))], old_dir, {}),
             start([sys.executable, "-m", "gpvae_endo", "run", "--config", str(prepare_workdir(new_dir, npz))],
                   new_dir, new_env)]
    for proc, d in zip(procs, (old_dir, new_dir)):
        wait(proc, d)
    compare_runs(old_dir, new_dir, "training from scratch (informational)", show=12)

    # 2. load the original's saved models with both programs and evaluate them: must be identical. This
    #    covers data loading and splitting, model file names, loading, every metric, the plots' inputs and
    #    the summary table.
    eval_old, eval_new = scratch / "eval_original", scratch / "eval_refactored"
    extra = "\nMUST_ALWAYS_TRAIN_FROM_SCRATCH = False\n"
    procs = []
    for d, cmd_head, env in ((eval_old, [sys.executable, str(driver)], {}),
                             (eval_new, [sys.executable, "-m", "gpvae_endo", "run"], new_env)):
        cfg = prepare_workdir(d, npz)
        cfg.write_text(cfg.read_text(encoding="utf-8") + extra, encoding="utf-8")
        shutil.rmtree(d / "Saved_Models")
        shutil.copytree(old_dir / "Saved_Models", d / "Saved_Models")
        procs.append((start([*cmd_head, "--config", str(cfg)], d, env), d))
    for proc, d in procs:
        wait(proc, d)
    ok = compare_runs(eval_old, eval_new, "evaluating the same saved models", show=120)
    print("\nRESULT:", "identical" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
