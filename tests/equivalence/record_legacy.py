"""Record ``legacy_reference.json``: the tiny-run fingerprints of the ORIGINAL code (commit aaf8b6d).

The original code is not part of this branch any more. Extract it to a folder first, then point this script
at it::

    git archive aaf8b6d | tar -x -C <folder>          # the original code
    # the ViT cases need the pretrained weights where the original looks for them:
    #   <folder>/Models/Weights/dinov2_base.pth  (a copy or a link to your weights folder)
    python tests/equivalence/record_legacy.py <folder> [case ...]

Run it from the repository root, with the project's environment. It only ever needs re-running if
``harness.py`` changes what is fingerprinted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harness  # noqa: E402


def main(argv):
    legacy = Path(argv[1]).resolve()
    cases = argv[2:] or list(harness.CASES)
    sys.path.insert(0, str(legacy))

    import Models.VAE_model as vae_module  # the original modules
    import Models.MSVQ_VAE_model as msvq_module
    from Models.VAE_model import VAEModel
    from Models.GPVAE_Casale_model import GPVAECasaleModel
    from Models.GPVAE_HPA_model import GPVAEHPAModel
    from Models.GPVAE_SPA_model import GPVAESPAModel
    from Models.MSVQ_VAE_model import MSVQVAEModel

    assert Path(vae_module.__file__).resolve().is_relative_to(legacy), vae_module.__file__
    classes = {"VAE": VAEModel, "GPVAE_Casale": GPVAECasaleModel, "GPVAE_HPA": GPVAEHPAModel,
               "GPVAE_SPA": GPVAESPAModel, "MSVQ_VAE": MSVQVAEModel}

    def patch(settings):
        # The model files did `from config_endo import WARMUP_EPOCHS, ...`, so the values live in each
        # model module's namespace; overriding them there is what changes the behaviour.
        for module in (vae_module, msvq_module):
            for k, v in settings.items():
                setattr(module, k, v)

    def build(kind, args, settings):
        patch(settings)
        return classes[kind](**args)

    def train(model, settings):
        patch(settings)
        model.train()

    out = json.loads(harness.REFERENCE.read_text(encoding="utf-8")) if harness.REFERENCE.exists() else {}
    out["_about"] = (
        "Fingerprints of tiny CPU training runs (tests/equivalence/harness.py) made with the ORIGINAL code, "
        "commit aaf8b6d, torch threads 1. The refactored code must reproduce them exactly."
    )
    import torch
    out["torch"] = torch.__version__
    for case in cases:
        t0 = time.time()
        out[case] = harness.run_case(build, train, case)
        print(f"{case}: {time.time() - t0:.1f} s", file=sys.stderr)
        harness.REFERENCE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {harness.REFERENCE}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
