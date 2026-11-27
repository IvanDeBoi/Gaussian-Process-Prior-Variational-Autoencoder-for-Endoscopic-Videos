"""Evaluate models you already trained, without training them again (README: Quick start, step 4).

Same settings as configs/experiment.py (300-epoch models, 20 % test frames, seed 42, video/models/encoders from the
GPVAE_* environment variables), except that a model file found in Saved_Models/ is loaded and evaluated
instead of trained. If a model file is missing, that model is trained from scratch (the log then says
"No pretrained model found at ...").
"""
import os as _os

_BASE_RUN = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "experiment.py")
with open(_BASE_RUN, encoding="utf-8") as _f:
    exec(compile(_f.read(), _BASE_RUN, "exec"))

MUST_ALWAYS_TRAIN_FROM_SCRATCH = False
