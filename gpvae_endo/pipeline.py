"""One complete experiment run: load a video, train every requested model, evaluate it, write the summary.

This is the original driver (``Latent_Space_Discovery.py``) step by step:

1. read the config file; start a log file (everything printed also goes to ``Logs/run log ... .txt``);
2. seed the random generators (``SEED``);
3. load the video's NPZ file and split its frames into training and test frames;
4. for every allowed (model, encoder, decoder) combination: build the model, train it (or load it from
   ``Saved_Models/`` when ``MUST_ALWAYS_TRAIN_FROM_SCRATCH = False`` and it exists), save it, evaluate it;
5. append a results table to ``Experiment_Summaries.txt``.

New compared with the original: optional checkpoints (see :mod:`gpvae_endo.training.checkpoint`), so a
cluster job that runs out of time can save and continue in the next job.
"""
from __future__ import annotations

import gc
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import torch

from gpvae_endo.config import (
    ExperimentConfig,
    load_config_module,
    load_model_settings,
    warn_ignored_model_settings,
)
from gpvae_endo.data.dataset import load_video, scale_frame_numbers, split_frames
from gpvae_endo.evaluation.evaluate import evaluate_model
from gpvae_endo.models.factory import build_model, model_file_name
from gpvae_endo.paths import OutputPaths
from gpvae_endo.reporting.summary import build_experiment_summary_text
from gpvae_endo.training.checkpoint import CheckpointConfig
from gpvae_endo.utils.device import require_gpu_or_allowed_cpu
from gpvae_endo.utils.io import Tee, format_hms, prepend_text_file
from gpvae_endo.utils.seeding import seed_everything

# Known harmless library warnings the original driver silenced as well.
for _message in ("enable_nested_tensor.*norm_first", "Tight layout not applied", ".*pretrained.*deprecated.*",
                 ".*Arguments other than a weight enum or `None` for 'weights'.*", "xFormers is not available",
                 "Importing `spectral_angle_mapper` from `torchmetrics.functional` was deprecated"):
    warnings.filterwarnings("ignore", message=_message)
plt.rcParams["figure.constrained_layout.use"] = False



@dataclass
class CheckpointOptions:
    """Command-line checkpoint settings (``python -m gpvae_endo run --checkpoint ...``)."""

    enabled: bool = False
    interval_s: float = 900.0
    deadline_margin_s: float = 600.0
    stop_after_epochs: Optional[int] = None


def checkpoint_path(model_path: str) -> str:
    """``Saved_Models/<model file name>.ckpt.pt`` next to the model file."""
    return (model_path[:-len(".pth")] if model_path.endswith(".pth") else model_path) + ".ckpt.pt"


def run(config_path: Optional[str] = None, checkpoints: Optional[CheckpointOptions] = None) -> None:
    """Run the experiment described by `config_path` (default ``config_endo.py``).

    Raises :class:`gpvae_endo.training.checkpoint.RequeueRequested` when checkpointing is on and the job's
    time is nearly up; the command line turns that into exit code 85.
    """
    checkpoints = checkpoints or CheckpointOptions()

    # 1. settings. Both config files seed the random generators when they are read, so they are read
    #    first, before this run seeds them (as in the original, where they were imported at start-up).
    config_path, module = load_config_module(config_path)
    cfg = ExperimentConfig.from_module(module, config_path)
    settings = load_model_settings()

    paths = OutputPaths.from_config(cfg)
    paths.create()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sys.stdout = Tee(f"{cfg.folder_logs}run log {cfg.substring_filter} {cfg.resolution} {timestamp}.txt")
    print(f"Using config file: {config_path}")
    warn_ignored_model_settings(module, settings)

    # 2. seeding
    seed_everything(cfg.seed)
    torch.set_default_dtype(torch.float32)

    print(f"{cfg.substring_filter}")
    print(f"SEED: {cfg.seed}")
    print(f"DEVICE: {cfg.device}")
    print(f"MUST_ALWAYS_TRAIN_FROM_SCRATCH: {cfg.must_always_train_from_scratch}")
    print(f"MUST_USE_MASK: {cfg.must_use_mask}")
    print(f"MUST_USE_LPIPS: {cfg.must_use_lpips}")
    print(f"MUST_USE_DECODER_GAN: {cfg.must_use_decoder_gan}")
    print(f"MUST_RECONSTRUCT_GAP: {cfg.must_use_gap_for_test}")
    require_gpu_or_allowed_cpu()  # fail fast, before loading the data (finding B10)
    if checkpoints.enabled:
        print(f"Checkpoints: on (every {checkpoints.interval_s:.0f} s; "
              f"SLURM_JOB_END_TIME={os.environ.get('SLURM_JOB_END_TIME', 'not set')})")

    rows = []
    start_time_total = time.time()

    # 3. data
    print(f"\n{'=' * 62} Loading data {'=' * 64}\n")
    print(f"Start this run: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")
    video = load_video(cfg)
    split = split_frames(video, cfg)
    x_scaler, x_train_scaled, x_test_scaled = scale_frame_numbers(split, cfg.use_x_scaling)

    # 4. every allowed (model, encoder, decoder)
    for kind in cfg.models:
        for encoder in cfg.encoders:
            for decoder in cfg.decoders:
                if (encoder, decoder) not in cfg.allowed_pairs_per_model.get(kind, set()):
                    continue
                rows.append(_train_and_evaluate(kind, encoder, decoder, cfg, settings, paths, split, x_scaler,
                                                x_train_scaled, x_test_scaled, video.denormalise, checkpoints))

    # 5. summary
    summary_text = build_experiment_summary_text(rows, cfg, split.test_frac, start_time_total, time.time())
    print("\n" + summary_text)
    prepend_text_file(os.path.join(os.getcwd(), "Experiment_Summaries.txt"), summary_text)


def _train_and_evaluate(kind, encoder, decoder, cfg, settings, paths, split, x_scaler, x_train_scaled,
                        x_test_scaled, denormalise, checkpoints: CheckpointOptions) -> dict:
    start_time_this_model = time.time()
    print(f"\n\n{'=' * 42} Processing {kind} {encoder} {decoder} {'=' * 47}")
    print(f"Start time: {datetime.fromtimestamp(start_time_this_model).strftime('%Y-%m-%d %H:%M:%S')}")

    model = build_model(kind, encoder, decoder, x_train_scaled, split.Y_train, split.M_train, cfg, settings)
    model_path = cfg.folder_to_save_models + model_file_name(kind, encoder, decoder, cfg, split.test_frac)
    ckpt = None
    if checkpoints.enabled:
        ckpt = CheckpointConfig(
            path=checkpoint_path(model_path), interval_s=checkpoints.interval_s,
            deadline_margin_s=checkpoints.deadline_margin_s, stop_after_epochs=checkpoints.stop_after_epochs,
            extra_config={"seed": cfg.seed, "video": cfg.substring_filter, "model_file": os.path.basename(model_path)},
        )

    resuming = ckpt is not None and os.path.exists(ckpt.path)
    if not resuming and os.path.exists(model_path) and not cfg.must_always_train_from_scratch:
        what = "MSVQ-VAE" if kind == "MSVQ_VAE" else "model"
        print(f"Loading pretrained {what} from {model_path}")
        model.load_state_dict_full(torch.load(model_path, map_location=cfg.device, weights_only=False))
        must_train = False
    else:
        if not resuming:
            print(f"No pretrained model found at {model_path}, training from scratch")
        must_train = True

    start_time_train = end_time_train = time.time()
    if must_train:
        print(f"Start time training: {datetime.fromtimestamp(start_time_train).strftime('%H:%M:%S')}")
        if ckpt is None:
            model.train()
        else:
            model.train(checkpoint=ckpt)
        end_time_train = time.time()
        print(f"End time training: {datetime.fromtimestamp(end_time_train).strftime('%H:%M:%S')} "
              f"=> total duration {format_hms(end_time_train - start_time_train)}")
        model.save(model_path)
    end_time_this_model = time.time()

    torch.cuda.empty_cache()
    gc.collect()
    print(f"Time to train this model (hms): {format_hms(end_time_this_model - start_time_this_model)}")

    tf = int(split.test_frac * 100)
    str_with_params = (f"{cfg.substring_filter} {kind} {encoder} {decoder} {cfg.kernel} {cfg.latent_dim}D "
                       f"L_{cfg.must_use_lpips}_{cfg.lpips_res} M_{cfg.must_use_mask} G_{cfg.must_use_decoder_gan} "
                       f"GAP_{cfg.must_use_gap_for_test}_{cfg.gap_size} TF_{tf}")
    result = evaluate_model(model, kind, encoder, decoder, split, x_scaler, x_train_scaled, x_test_scaled, cfg,
                            paths, denormalise, str_with_params)

    epochs = len(model.loss_total_list)
    row = dict(substring=cfg.substring_filter, RESOLUTION=cfg.resolution, model_from_models=kind, encoder=encoder,
               decoder=decoder, **result.as_dict(),
               time_sec=float(end_time_this_model - start_time_this_model), epochs=epochs,
               time_per_epoch=float(end_time_train - start_time_train) / epochs)

    print(f"\nWrapping up {str_with_params}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return row
