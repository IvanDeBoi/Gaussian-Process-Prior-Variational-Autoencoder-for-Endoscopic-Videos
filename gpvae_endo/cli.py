"""Command line.

    python -m gpvae_endo run                              # settings from config_endo.py
    python -m gpvae_endo run configs/config_c1_descending_t1_v2.py
    python -m gpvae_endo run --config configs/<file>.py --checkpoint     # with checkpoint/resume

Exit codes: 0 = finished; 85 = checkpoint saved because the job's time is nearly up, submit the job again
to continue (only with ``--checkpoint``); 1 = no GPU (set ``GPVAE_ALLOW_CPU=1`` to run on the CPU anyway);
anything else = error.
"""
from __future__ import annotations

import argparse
import sys

from gpvae_endo.training.checkpoint import EXIT_REQUEUE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gpvae_endo",
                                     description="GP-VAE for endoscopic videos: train, evaluate, report.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="train and evaluate every model the config file lists")
    run.add_argument("config", nargs="?", default=None,
                     help="Python config file (default: config_endo.py)")
    run.add_argument("--config", dest="config_flag", default=None,
                     help="Python config file; overrides the positional argument")
    run.add_argument("--checkpoint", action="store_true",
                     help="save training state regularly and resume from it; exit with code 85 shortly before "
                          "the Slurm job ends (SLURM_JOB_END_TIME)")
    run.add_argument("--checkpoint-interval", type=float, default=900.0, metavar="SECONDS",
                     help="seconds between checkpoints (default 900)")
    run.add_argument("--deadline-margin", type=float, default=600.0, metavar="SECONDS",
                     help="stop this many seconds before SLURM_JOB_END_TIME (default 600)")
    run.add_argument("--stop-after-epochs", type=int, default=None, metavar="N",
                     help="(testing) save and exit with 85 after N epochs in this process")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from gpvae_endo.pipeline import CheckpointOptions, run
        from gpvae_endo.training.checkpoint import RequeueRequested
        from gpvae_endo.utils.device import NoGPUError

        options = CheckpointOptions(enabled=args.checkpoint, interval_s=args.checkpoint_interval,
                                    deadline_margin_s=args.deadline_margin,
                                    stop_after_epochs=args.stop_after_epochs)
        try:
            run(args.config_flag or args.config, options)
        except RequeueRequested as exc:
            print(f"[checkpoint] {exc}. Exit code {EXIT_REQUEUE}: submit the job again to continue.")
            sys.stdout.flush()
            return EXIT_REQUEUE
        except NoGPUError as exc:
            print(f"ERROR: {exc}")
            sys.stdout.flush()
            return 1
        return 0
    return 2
