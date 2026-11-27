"""Tiny synthetic training runs used to prove that the refactored code trains *exactly* like the original.

The same runs are made twice:

* once with the original code (commit aaf8b6d), by ``record_legacy.py``, which writes
  ``legacy_reference.json``;
* once with the refactored package, by ``tests/test_equivalence.py``.

On the CPU with one thread these runs are bit-reproducible (finding T4), so the two fingerprints must be
equal to the last bit. This file only *reads* model attributes, so it works with both code versions.

What a fingerprint holds: the per-epoch losses, the printed per-epoch log lines, the learning rates of
every parameter group after every epoch, the norm of every weight tensor (keyed by its name, so saved
model files stay loadable), the GP hyper-parameters, the final latents ``Z``, the keys of the saved model
file, and the outputs of the methods the evaluation uses afterwards (``image_encode``, ``decode``,
``time_encode``) on fixed inputs.
"""
from __future__ import annotations

import contextlib
import io
import json
import random
from pathlib import Path

import numpy as np
import torch

N_FRAMES = 24
IMG = 32
LATENT = 8
BATCH = 8
NEIGHBOURS = 4
THREADS = 1

# Settings the model code reads from config_endo, overridden to make the tiny runs cross every phase:
# LR warm-up -> cosine schedule, KL weight ramp -> max, best-state snapshots, early-stopping bookkeeping.
TINY_SETTINGS = {
    "WARMUP_EPOCHS": 2,
    "RAMP_EPOCHS": 2,
    "LPIPS_RES": 16,
    "EARLY_STOPPING_PATIENCE": 25,
    "EARLY_STOPPING_START_FRACTION": 0.5,
}

# name -> (model kind, encoder, decoder, epochs, extra settings). Casale only during warm-up: once the KL
# switches on it crashes with a TypeError in the original (finding B9), which is preserved and tested
# separately.
CASES = {
    "VAE_EndoVAE": ("VAE", "EndoVAEEnc", "EndoVAEDec", 8, {}),
    "VAE_CNN": ("VAE", "CNNEnc", "CNNDecRes", 6, {}),
    "HPA_EndoVAE": ("GPVAE_HPA", "EndoVAEEnc", "EndoVAEDec", 8, {}),
    "SPA_EndoVAE": ("GPVAE_SPA", "EndoVAEEnc", "EndoVAEDec", 8, {}),
    "Casale_EndoVAE": ("GPVAE_Casale", "EndoVAEEnc", "EndoVAEDec", 4, {"WARMUP_EPOCHS": 4}),
    "MSVQ": ("MSVQ_VAE", "MSVQEnc", "MSVQDec", 6, {}),
    "HPA_GastroEnc": ("GPVAE_HPA", "GastroEnc", "EndoVAEDec", 5, {}),
    "VAE_GaLoRaEnc": ("VAE", "GaLoRaEnc", "EndoVAEDec", 3, {}),
}
SLOW_CASES = ("HPA_GastroEnc", "VAE_GaLoRaEnc")  # need the DINOv2 weights, slower on the CPU

REFERENCE = Path(__file__).resolve().parent / "legacy_reference.json"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_data(n: int = N_FRAMES, size: int = IMG):
    """Deterministic smooth frames (a blob moving over time) plus a mask with a few invalid pixels."""
    rng = np.random.default_rng(1234)
    t = np.arange(n, dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size] / size
    frames = []
    for k in range(n):
        cx, cy = 0.3 + 0.4 * k / n, 0.5 + 0.2 * np.sin(k / 4)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.02)
        rgb = np.stack([0.6 * blob + 0.2, 0.4 * blob + 0.1 * xx, 0.3 * blob + 0.1 * yy], axis=-1)
        frames.append(rgb)
    frames = np.asarray(frames) + 0.02 * rng.standard_normal((n, size, size, 3))
    Y = frames.reshape(n, -1).astype(np.float32)
    M = (rng.random((n, size * size)) > 0.05).astype(np.float32)
    x = t.reshape(-1, 1).astype(np.float32)  # as the driver's IdentityScaler returns it
    return x, Y, M


def model_args(kind: str, encoder: str, decoder: str, epochs: int) -> dict:
    """Constructor arguments, as the driver passes them (device "gpu" falls back to the CPU here)."""
    x, Y, M = make_data()
    if kind == "MSVQ_VAE":
        return dict(x=x, Y=Y, M=M, device="cpu", img_H=IMG, img_W=IMG, img_C=3, codebook_size=16,
                    code_dim=LATENT, beta_commit=0.25, ema_decay=0.99, epochs=epochs, lr=1e-3, batch_size=BATCH)
    args = dict(x=x, Y=Y, M=M, device="cpu", latent_dim=LATENT, lr=1e-3, img_H=IMG, img_W=IMG, img_C=3,
                epochs=epochs, encoder=encoder, decoder=decoder, batch_size=BATCH)
    if kind != "VAE":
        args["kernel_gp_prior"] = "CAUCHY"
    if kind in ("GPVAE_HPA", "GPVAE_SPA"):
        args["n_neighbors"] = NEIGHBOURS
    return args


def _as_tensor(t):
    if t is None:
        return None
    return torch.as_tensor(np.asarray(t) if not torch.is_tensor(t) else t).double()


def _norm(t):
    t = _as_tensor(t)
    return None if t is None else float(t.norm())


def _sum(t):
    t = _as_tensor(t)
    return None if t is None else float(t.sum())


def _gp(model):
    if getattr(model, "gp_prior", None) is None:
        return None
    covar = model.gp_prior.covar_module
    return {
        "lengthscale": float(covar.base_kernel.lengthscale.detach().view(-1)[0]),
        "outputscale": float(covar.outputscale.detach()),
        "noise": float(model.gp_prior.likelihood.noise.detach().view(-1)[0]),
    }


def _state_keys(state, prefix=""):
    out = []
    for k, v in state.items():
        out.append(f"{prefix}{k}")
        if isinstance(v, dict) and k not in ("encoder", "decoder", "gp_prior", "discriminator"):
            out += _state_keys(v, f"{prefix}{k}.")
    return sorted(out)


def fingerprint(model, lr_trace, log_text: str, restores: int) -> dict:
    loss_lists = sorted(k for k in vars(model) if k.startswith("loss_") and k.endswith("_list"))
    fp = {
        "losses": {k: [float(v) for v in getattr(model, k)] for k in loss_lists},
        "epoch_lines": [ln for ln in log_text.splitlines() if ln.startswith("Epoch ")],
        "lr_trace": lr_trace,
        "lrs_final": [float(g["lr"]) for g in model.optimizer.param_groups],
        "encoder_norms": {k: _norm(v) for k, v in model.encoder.state_dict().items()},
        "decoder_norms": {k: _norm(v) for k, v in model.decoder.state_dict().items()},
        "Z_norm": _norm(model.Z),
        "Z_sum": _sum(model.Z),
        "best_state_restores": restores,
        "saved_keys": _state_keys(model.state_dict_full()),
    }
    for extra in ("quantizer", "lat_proj", "lat_unproj"):
        if hasattr(model, extra):
            fp[f"{extra}_norms"] = {k: _norm(v) for k, v in getattr(model, extra).state_dict().items()}
    for name in ("mu_all", "logvar_all"):
        v = getattr(model, name, None)
        if v is not None:
            fp[f"{name}_sum"] = _sum(v)
    gp = _gp(model)
    if gp is not None:
        fp["gp"] = gp
    # what the evaluation calls afterwards
    _, Y, _ = make_data()
    imgs = Y[:5].reshape(5, IMG, IMG, 3)
    mu, std = model.image_encode(imgs)
    fp["image_encode"] = [_sum(mu), _norm(mu), _sum(std)]
    dec = model.decode(np.asarray(mu, dtype=np.float32))
    fp["decode"] = [_sum(dec), _norm(dec)]
    if hasattr(model, "time_encode"):
        xs = np.array([[0.0], [3.5], [11.0], [30.0]], dtype=np.float32)
        zm, zs = model.time_encode(xs, use_likelihood=True, return_full_cov=False)
        zm2, zs2, zc = model.time_encode(xs, use_likelihood=True, return_full_cov=True)
        fp["time_encode"] = [_sum(zm), _norm(zm), _sum(zs), _sum(zm2), _sum(zc)]
    return fp


def trace_lrs(model) -> list:
    """Record every parameter group's LR after each `update_lr` call (no behaviour change, no RNG)."""
    trace = []
    original = model.update_lr

    def update_lr(epoch):
        original(epoch)
        trace.append([float(g["lr"]) for g in model.optimizer.param_groups])

    model.update_lr = update_lr
    return trace


def count_calls(model, name: str) -> list:
    calls = []
    original = getattr(model, name)

    def wrapper(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    setattr(model, name, wrapper)
    return calls


def run_case(build, train, case: str) -> dict:
    """`build(kind, args, settings)` constructs a model with the given tiny settings in effect;
    `train(model)` trains it. Both are code-version specific; everything else is shared."""
    kind, encoder, decoder, epochs, extra = CASES[case]
    settings = {**TINY_SETTINGS, **extra}
    torch.set_num_threads(THREADS)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        seed_all(0)
        model = build(kind, model_args(kind, encoder, decoder, epochs), settings)
        gp_init = _gp(model)
        lr_trace = trace_lrs(model) if hasattr(model, "update_lr") else []
        restores = count_calls(model, "load_state_dict_full")
        train(model, settings)
        fp = fingerprint(model, lr_trace, buf.getvalue(), len(restores))
    if gp_init is not None:
        fp["gp_init"] = gp_init
    fp["rng_after"] = [float(torch.rand(1)), float(np.random.rand())]  # RNG state left for the next model
    return fp


def flatten(prefix, obj, out):
    if isinstance(obj, str):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten(f"{prefix}[{i}]", v, out)
    else:
        out[prefix] = obj
    return out


def compare(a: dict, b: dict, rtol: float = 0.0) -> list[str]:
    """Differences between two fingerprints (empty list = identical within `rtol`)."""
    fa, fb = flatten("", a, {}), flatten("", b, {})
    problems = [f"key only in one fingerprint: {k}" for k in sorted(set(fa) ^ set(fb))]
    for k in sorted(set(fa) & set(fb)):
        va, vb = fa[k], fb[k]
        if isinstance(va, str) or isinstance(vb, str) or va is None or vb is None:
            if va != vb:
                problems.append(f"{k}: {va!r} != {vb!r}")
            continue
        va, vb = float(va), float(vb)
        if va == vb or (np.isnan(va) and np.isnan(vb)):
            continue
        if abs(va - vb) > rtol * max(abs(va), abs(vb), 1e-12):
            problems.append(f"{k}: {va!r} != {vb!r}")
    return problems


def load_reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))
