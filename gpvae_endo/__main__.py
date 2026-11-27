"""``python -m gpvae_endo ...`` (see :mod:`gpvae_endo.cli`)."""
import sys

# The original driver set CUDA_LAUNCH_BLOCKING=1 and TORCH_USE_CUDA_DSA=1 here: CUDA debugging switches that
# make the CPU wait for every GPU operation. They slow training down and do not change any result, so they
# are gone. To debug a CUDA error, set CUDA_LAUNCH_BLOCKING=1 in the shell for that one run.
from gpvae_endo.cli import main

sys.exit(main())
