"""Kept so the old command still works:

    python Latent_Space_Discovery.py [config.py] [--config config.py]

It runs the same experiment as ``python -m gpvae_endo run``. The code that used to be in this file (3,439
lines) now lives in the ``gpvae_endo`` package; see README.md for where each part went.
"""
import sys

from gpvae_endo.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
