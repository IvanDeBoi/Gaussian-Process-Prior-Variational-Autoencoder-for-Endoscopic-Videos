"""The results table appended to ``Experiment_Summaries.txt`` after a run (newest run on top).

One row per trained (model, encoder, decoder). Column names: ``tr``/``te`` = training/test frames,
``im`` = image-encoded, ``t`` = time-encoded reconstruction (GP models only). Same format as the original.
"""
from __future__ import annotations

from datetime import datetime

from gpvae_endo.config import ExperimentConfig
from gpvae_endo.utils.io import format_hms


def _fmt(value, spec: str) -> str:
    return format(value, spec) if value is not None else "n/a"


def build_experiment_summary_text(rows: list[dict], cfg: ExperimentConfig, test_frac: float,
                                  start_time_total: float, end_time_total: float) -> str:
    lines = [
        "===================================================================================== Summary "
        "==================================================================================================="
        "===============================================================",
        f"Overview all models for {cfg.substring_filter} | res={cfg.resolution} | latent dim={cfg.latent_dim} | "
        f"H={cfg.nr_of_neighbours} | LPIPS={cfg.must_use_lpips} lpips res={cfg.lpips_res} | "
        f"GAN={cfg.must_use_decoder_gan} | {cfg.kernel} | gap={cfg.must_use_gap_for_test} size={cfg.gap_size} | "
        f"TF={int(test_frac * 100)}%",
        "Metrics note: RMSE, SSIM, PSNR, and FID are reported in denormalized image space.",
    ]
    header = (
        f"{'model':15s} | "
        f"{'encoder':10s} | "
        f"{'decoder':10s} | "
        f"{'RMSE_tr_im':>12s} | {'RMSE_tr_t':>12s} | "
        f"{'RMSE_te_im':>12s} | {'RMSE_te_t':>12s} | "
        f"{'FID_tr_im':>10s} | {'FID_te_im':>10s} | "
        f"{'FID_tr_t':>10s} | {'FID_te_t':>10s} | "
        f"{'SSIM_tr_im':>12s} | {'SSIM_te_im':>12s} | "
        f"{'SSIM_tr_t':>12s} | {'SSIM_te_t':>12s} | "
        f"{'PSNR_tr_im':>12s} | {'PSNR_te_im':>12s} | "
        f"{'PSNR_tr_t':>12s} | {'PSNR_te_t':>12s} | "
        f"{'t(h:m:s)':>9s} | {'epochs':>6s} | {'t/epoch(s)':>8s}"
    )
    lines += [header, "-" * len(header)]

    for e in rows:
        g = e.get
        lines.append(
            f"{e['model_from_models']:15s} | "
            f"{e['encoder']:10s} | "
            f"{e['decoder']:10s} | "
            f"{_fmt(g('rmse_train_encoded_image'), '.6f'):>12s} | {_fmt(g('rmse_train_encoded_time'), '.6f'):>12s} | "
            f"{_fmt(g('rmse_test_encoded_image'), '.6f'):>12s} | {_fmt(g('rmse_test_encoded_time'), '.6f'):>12s} | "
            f"{_fmt(g('fid_train_image_encoded'), '.3f'):>10s} | {_fmt(g('fid_test_image_encoded'), '.3f'):>10s} | "
            f"{_fmt(g('fid_train_time_encoded'), '.3f'):>10s} | {_fmt(g('fid_test_time_encoded'), '.3f'):>10s} | "
            f"{_fmt(g('ssim_train_image_encoded'), '.4f'):>12s} | {_fmt(g('ssim_test_image_encoded'), '.4f'):>12s} | "
            f"{_fmt(g('ssim_train_time_encoded'), '.4f'):>12s} | {_fmt(g('ssim_test_time_encoded'), '.4f'):>12s} | "
            f"{_fmt(g('psnr_train_image_encoded'), '.4f'):>12s} | {_fmt(g('psnr_test_image_encoded'), '.4f'):>12s} | "
            f"{_fmt(g('psnr_train_time_encoded'), '.4f'):>12s} | {_fmt(g('psnr_test_time_encoded'), '.4f'):>12s} | "
            f"{format_hms(e['time_sec']):>9s} | {e['epochs']:6d} | {e['time_per_epoch']:8.3f}"
        )

    lines.append(f"End time: {datetime.fromtimestamp(end_time_total).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total time (hms): {format_hms(end_time_total - start_time_total)}")
    lines.append("")
    return "\n".join(lines)
