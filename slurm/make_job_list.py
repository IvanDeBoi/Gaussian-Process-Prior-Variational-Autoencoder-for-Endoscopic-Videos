"""Write a job list for slurm/train.slurm: one line per (video, model, encoder, seed).

    python slurm/make_job_list.py c3vd                    # every video in $VSC_SCRATCH/gpvae-data/c3vd/
    python slurm/make_job_list.py kvasir --seeds 42 1 2   # more seeds
    python slurm/make_job_list.py c3vd --models VAE --encoders EndoVAEEnc   # a subset

The videos are the files "<video> 336 X Ym01sGM.npz" in <data root>/<dataset>/ (data root: --data-root, else
$GPVAE_DATA_ROOT, else $VSC_SCRATCH/gpvae-data). The list goes to slurm/jobs/<dataset>.txt (or --out); the
script prints the sbatch command for it. Standard library only, so any Python 3 can run it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUFFIX = " 336 X Ym01sGM.npz"
MODELS = ["VAE", "GPVAE_HPA", "GPVAE_SPA"]
ENCODERS = ["EndoVAEEnc", "GastroEnc", "GaLoRaEnc"]


def default_data_root() -> Path:
    if os.environ.get("GPVAE_DATA_ROOT"):
        return Path(os.environ["GPVAE_DATA_ROOT"])
    if os.environ.get("VSC_SCRATCH"):
        return Path(os.environ["VSC_SCRATCH"]) / "gpvae-data"
    return REPO / "Data"


def find_videos(data_dir: Path) -> list[str]:
    return sorted(p.name[: -len(SUFFIX)] for p in data_dir.glob("*" + SUFFIX))


def job_lines(dataset: str, videos: list[str], models: list[str], encoders: list[str], seeds: list[int]) -> list[str]:
    """Ordered by model, then encoder, then video: the cheap EndoVAEEnc runs of each model come first."""
    for v in videos:
        if any(c.isspace() for c in v) or "," in v:
            raise ValueError(f"video name with a space or comma is not supported: {v!r}")
    return [f"{dataset} {v} {m} {e} {s}" for s in seeds for m in models for e in encoders for v in videos]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="name of the data folder, e.g. c3vd, kvasir, hyperkvasir")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--encoders", nargs="+", default=ENCODERS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    data_dir = (args.data_root or default_data_root()) / args.dataset
    videos = find_videos(data_dir)
    if not videos:
        print(f"ERROR: no '*{SUFFIX}' files in {data_dir}", file=sys.stderr)
        return 1
    lines = job_lines(args.dataset, videos, args.models, args.encoders, args.seeds)
    out = args.out or REPO / "slurm" / "jobs" / f"{args.dataset}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(f"{len(videos)} videos in {data_dir}:")
    for v in videos:
        print(f"  {v}")
    print(f"{len(lines)} runs ({len(videos)} videos x {len(args.models)} models x {len(args.encoders)} encoders"
          f" x {len(args.seeds)} seeds) written to {out}")
    try:
        rel = out.resolve().relative_to(REPO)
    except ValueError:
        rel = out
    print("Submit with (from the repository folder, after `source setup.sh`):")
    print(f"  sbatch --account=$ACC --array=0-{len(lines) - 1}%4 --export=$EXP,JOBS={rel.as_posix()} "
          f"slurm/train.slurm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
