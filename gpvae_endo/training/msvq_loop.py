"""The training loop of MSVQ_VAE (the original ``MSVQVAEModel.train`` unchanged, plus optional checkpointing).

Differences from the VAE loop (:mod:`gpvae_endo.training.loop`), all kept from the original:

* the frame order is shuffled once, before the first epoch, and then reused every epoch;
* on the very first batch the codebook is filled with encoder outputs ("warm start");
* the best epoch is chosen among *all* epochs (MSVQ has no KL term, so the loss is comparable throughout);
  its snapshot is a real copy. The original used ``.detach().cpu()``, which copies on a GPU but on the CPU
  returns the live tensors (B2), so on the GPU nothing changes;
* the final latents are not computed after training.
"""
from __future__ import annotations

import gc
from typing import Optional

import torch

from gpvae_endo.training.checkpoint import CheckpointConfig, TrainingCheckpointer
from gpvae_endo.training.loop import _epoch_means, _new_sums
from gpvae_endo.training.progress import ProgressTracker
from gpvae_endo.utils.tensors import cpu_copy

MSVQ_HISTORY_NAMES = (
    "loss_total_list", "loss_mse_list", "loss_lpips_list", "loss_kl_list", "loss_gan_list", "loss_fm_list",
    "loss_commit_list", "loss_codebook_list", "loss_weighted_mse_list", "loss_weighted_lpips_list",
    "loss_weighted_kl_list", "loss_weighted_gan_list", "loss_weighted_fm_list", "loss_weighted_commit_list",
    "loss_adv_d_list",
)
_PER_STEP = ("mse", "lpips", "kl", "gan", "fm", "commit", "codebook", "weighted_mse", "weighted_lpips",
             "weighted_kl", "weighted_gan", "weighted_fm", "weighted_commit")


def _warm_start_codebook(model, z_e: torch.Tensor) -> None:
    with torch.no_grad():
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, z_e.size(1))
        n_codes = model.quantizer.num_embeddings
        if flat.size(0) >= n_codes:
            perm_idx = torch.randperm(flat.size(0), device=flat.device)[:n_codes]
            model.quantizer.embedding.data.copy_(flat[perm_idx])
            model.quantizer.embed_avg.data.copy_(flat[perm_idx])
        else:
            repeats = (n_codes + flat.size(0) - 1) // flat.size(0)
            tiled = flat.repeat(repeats, 1)[:n_codes]
            model.quantizer.embedding.data.copy_(tiled)
            model.quantizer.embed_avg.data.copy_(tiled)
        model.quantizer.cluster_size.data.fill_(1.0)
    print(f"  Codebook warm-started from {flat.size(0)} encoder vectors")


def _best_snapshot(model) -> dict:
    return cpu_copy({
        "encoder": model.encoder.state_dict(),
        "decoder": model.decoder.state_dict(),
        "lat_proj": model.lat_proj.state_dict(),
        "lat_unproj": model.lat_unproj.state_dict(),
        "quantizer": model._quantizer_state(),
        "X_mean": model.X_mean,
        "X_std": model.X_std,
    })


def train_msvq(model, checkpoint: Optional[CheckpointConfig] = None):
    print(f"Starting training {model.underlying_method} ...")
    model.encoder.train()
    model.enc_proj.train()
    model.dec_in_proj.train()
    model.decoder.train()

    N = model.Ytensor.size(0)
    pt = ProgressTracker(
        patience=model.early_stopping_patience,
        max_iterations=model.epochs,
        early_stopping_start_iteration=int(model.epochs * model.early_stopping_start_fraction),
    )
    history = {name: [] for name in MSVQ_HISTORY_NAMES}
    best_loss = float("inf")
    best_state = None
    codebook_initialised = False
    start_epoch = 0

    checkpointer = None
    resumed = None
    if checkpoint is not None:
        checkpointer = TrainingCheckpointer(model, checkpoint, history)
        resumed = checkpointer.resume(pt)
        if resumed is not None and resumed["finished"]:
            print(f"{model.underlying_method} training complete.")
            return model

    if resumed is not None:
        best_loss, best_state, start_epoch = resumed["best_loss"], resumed["best_state"], resumed["next_epoch"]
        perm = resumed["extra"]["perm"]
        codebook_initialised = resumed["extra"]["codebook_initialised"]
    else:
        perm = torch.randperm(N)  # once, on the CPU (frames live on the CPU)

    for epoch in range(start_epoch, model.epochs):
        sums = _new_sums(("total",) + _PER_STEP + ("adv_d",), model.device)  # on the device, see loop.py

        # KL weight ramp (as in the VAE; MSVQ has no KL term, but beta still resets the best loss)
        if epoch < model.warmup_epochs:
            beta = 0.0
        else:
            beta = model.max_beta * min(1.0, (epoch - model.warmup_epochs) / model.ramp_epochs)
        if beta < model.max_beta:
            pt.best_loss = float("inf")

        for i in range(0, N, model.batch_size):
            idx = perm[i:i + model.batch_size]
            x_batch_hwc = model.Ytensor[idx].to(model.device, non_blocking=True)
            m_batch = model.Mtensor[idx].to(model.device, non_blocking=True) if model.Mtensor is not None else None
            x_batch_chw = x_batch_hwc.permute(0, 3, 1, 2).contiguous()

            z_e = model.latent_to_spatial(model.encode_to_latent(x_batch_chw))
            if not codebook_initialised:
                _warm_start_codebook(model, z_e)
                codebook_initialised = True

            z_q_st, z_q, _ = model.quantize(z_e)
            x_hat_chw = model.decode_from_quantized(z_q_st)

            adv_d = torch.tensor(0.0, device=model.device)
            use_gan_now = (
                model.use_decoder_gan
                and model.discriminator is not None
                and epoch >= model.gan_warmup_epochs
                and (model.loss_weights["gan"] > 0 or model.gan_feature_matching_weight > 0)
            )
            if use_gan_now:
                x_hat_hwc = x_hat_chw.permute(0, 2, 3, 1).contiguous()
                for _ in range(model.gan_disc_updates):
                    adv_d = model._train_discriminator_step(real_hwc=x_batch_hwc, fake_hwc=x_hat_hwc, mask=m_batch)

            total_loss, loss_dict = model.compute_total_loss(
                x_batch_hwc=x_batch_hwc, x_batch_chw=x_batch_chw, x_hat_chw=x_hat_chw, z_e=z_e, z_q=z_q,
                mask=m_batch, kl_beta=beta, use_gan=use_gan_now,
            )
            if not torch.isfinite(total_loss):
                print(
                    "NaN total loss at epoch", epoch, "step", i,
                    "x_hat min/max:", x_hat_chw.min().item(), x_hat_chw.max().item()
                )
                continue

            model.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder.parameters()) + list(model.enc_proj.parameters())
                + list(model.dec_in_proj.parameters()) + list(model.decoder.parameters()),
                max_norm=1.0,
            )  # note: the latent projections are not clipped (as in the original)
            model.optimizer.step()

            bs = x_batch_hwc.size(0)
            del total_loss, x_hat_chw, z_e, z_q_st, z_q, x_batch_chw, x_batch_hwc
            if use_gan_now:
                del x_hat_hwc
            if m_batch is not None:
                del m_batch

            sums["total"] += loss_dict["total"].detach().double() * bs
            for k in _PER_STEP:
                sums[k] += loss_dict[k].detach().double() * bs
            sums["adv_d"] += adv_d.detach().double() * bs
            del loss_dict, adv_d

        ep = _epoch_means(sums, N)
        history["loss_total_list"].append(float(ep["total"]))
        for name in MSVQ_HISTORY_NAMES[1:-1]:
            history[name].append(float(ep[name[len("loss_"):-len("_list")]]))
        history["loss_adv_d_list"].append(float(ep["adv_d"]))

        if ep["total"] < best_loss:
            best_loss = ep["total"]
            best_state = _best_snapshot(model)
            gc.collect()

        if epoch % pt.print_iterations == 0:
            lrs = [f"{g['lr']:.2e}" for g in model.optimizer.param_groups]
            print(
                f"Epoch {epoch:04d} | "
                f"total={ep['total']:.6f} | "
                f"wmse={ep['weighted_mse']:.6f} (mse={ep['mse']:.6f}) | "
                f"wlpips={ep['weighted_lpips']:.6f} (lpips={ep['lpips']:.6f}) | "
                f"wkl={ep['weighted_kl']:.6f} (kl={ep['kl']:.6f}, beta={beta:.2e}) | "
                f"wcommit={ep['weighted_commit']:.6f} (commit={ep['commit']:.6f}) | "
                f"wcodebook=na (codebook={ep['codebook']:.6f}) | "
                f"wgan={ep['weighted_gan']:.6f} (gan={ep['gan']:.6f}) | "
                f"wfm={ep['weighted_fm']:.6f} (fm={ep['fm']:.6f}) | "
                f"d={ep['adv_d']:.6f} | "
                f"lr={lrs[0]} | "
                f"no improvement count={pt.no_improvement_count}"
            )

        if pt.check_stop_criterion(torch.tensor(ep["total"], device=model.device), epoch):
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch < model.warmup_epochs:
            model.warmup_scheduler.step()
        else:
            model.main_scheduler.step()
        if checkpointer is not None:
            checkpointer.end_of_epoch(epoch, pt, best_loss, best_state,
                                      extra={"perm": perm, "codebook_initialised": codebook_initialised})

    if best_state is not None:  # restore the best epoch's weights
        model.encoder.load_state_dict(best_state["encoder"])
        model.decoder.load_state_dict(best_state["decoder"])
        model.lat_proj.load_state_dict(best_state["lat_proj"])
        model.lat_unproj.load_state_dict(best_state["lat_unproj"])
        model._load_quantizer(best_state["quantizer"])
        model.X_mean = best_state["X_mean"].to(model.device)
        model.X_std = best_state["X_std"].to(model.device)
        model.Xtensor_norm = (model.Xtensor - model.X_mean) / (model.X_std + 1e-6)

    for name, values in history.items():
        setattr(model, name, values)

    if checkpointer is not None:
        checkpointer.save_finished(pt, best_loss, best_state,
                                   extra={"perm": perm, "codebook_initialised": codebook_initialised})
    print(f"{model.underlying_method} training complete.")
    return model
