"""The training loop of the VAE-family models (VAE, GPVAE_Casale, GPVAE_HPA, GPVAE_SPA).

``model.train()`` calls :func:`train_vae`. The loop is the original ``VAEModel.train`` (only ``self``
became ``model``), plus optional checkpointing, with two changes: the best-epoch restore works (B2, see
below), and the loss sums stay on the GPU instead of being read back after every step (same numbers,
faster). Per epoch:

1. ``model.refresh_epoch_state()`` (HPA/SPA re-encode all frames for their neighbour blocks);
2. shuffle the frames; for each mini-batch: encode, sample a latent, decode, compute the loss
   (``model.compute_total_loss``), back-propagate, clip gradients to norm 1, take an Adam step;
3. record the average losses, remember the best epoch (a copy of its weights), print a log line;
4. stop early if the loss has not improved for a while; step the LR schedule.

At the end the best-epoch weights are loaded back and every frame is encoded once more to get the final
latents ``model.Z``. "Best" = lowest average training loss (B11: still the training loss) among the epochs
that use the full KL weight ``max_beta``. In the original the snapshot held *references* to the live
weights, so the "restore" silently kept the last epoch (finding B2); it is now a real copy on the CPU. When
training ends before the KL weight reaches its maximum, there is no snapshot and the last epoch is kept.

With a :class:`~gpvae_endo.training.checkpoint.CheckpointConfig` the loop also saves its full state at
epoch boundaries and resumes from it; without one it behaves exactly as the original.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from gpvae_endo.training.checkpoint import CheckpointConfig, TrainingCheckpointer
from gpvae_endo.training.progress import ProgressTracker
from gpvae_endo.utils.tensors import cpu_copy
from gpvae_endo.viz.training import decode_and_save_images_from_random_latents, plot_and_save_latents

LOSS_HISTORY_NAMES = (
    "loss_total_list", "loss_mse_list", "loss_lpips_list", "loss_kl_list", "loss_gan_list", "loss_fm_list",
    "loss_weighted_mse_list", "loss_weighted_lpips_list", "loss_weighted_kl_list", "loss_weighted_gan_list",
    "loss_weighted_fm_list", "loss_adv_d_list",
)  # the per-epoch loss histories a trained model carries (and saves)
_PER_STEP = ("mse", "lpips", "kl", "gan", "fm", "weighted_mse", "weighted_lpips", "weighted_kl", "weighted_gan",
             "weighted_fm")


def _new_sums(names, device) -> dict:
    """Per-epoch loss sums, kept on the training device as float64 scalars.

    The original read every loss back to the CPU after every step (``.item()``, 12 GPU syncs per step) and
    added the values up as Python floats. Adding float64 tensors on the device is the same IEEE double
    arithmetic (the float32 loss widened to float64, times the batch size, added), so the sums are
    bit-identical, but the GPU no longer stalls after every step.
    """
    return {k: torch.zeros((), dtype=torch.float64, device=device) for k in names}


def _epoch_means(sums: dict, n: int) -> dict:
    """The per-epoch averages as Python floats (one read-back per epoch)."""
    values = torch.stack(list(sums.values())).tolist()
    return {k: v / n for k, v in zip(sums, values)}


def kl_beta(model, epoch: int) -> float:
    """KL weight: 0 during the LR warm-up, then a linear ramp to ``max_beta`` over ``ramp_epochs``."""
    if epoch < model.warmup_epochs:
        return 0.0
    t = min(1.0, (epoch - model.warmup_epochs) / model.ramp_epochs)
    return model.max_beta * t


def train_vae(model, checkpoint: Optional[CheckpointConfig] = None) -> None:
    print(f"Starting training {model.underlying_method} ...")
    model.encoder.train()
    model.decoder.train()

    N = model.Ytensor.size(0)
    pt = ProgressTracker(
        patience=model.early_stopping_patience,
        max_iterations=model.epochs,
        early_stopping_start_iteration=int(model.epochs * model.early_stopping_start_fraction),
    )
    history = {name: [] for name in LOSS_HISTORY_NAMES}

    best_loss = float("inf")
    best_state = None
    start_epoch = 0

    checkpointer = None
    if checkpoint is not None:
        checkpointer = TrainingCheckpointer(model, checkpoint, history)
        resumed = checkpointer.resume(pt)
        if resumed is not None:
            if resumed["finished"]:
                print(f"{model.underlying_method} training complete.")
                return
            best_loss, best_state, start_epoch = resumed["best_loss"], resumed["best_state"], resumed["next_epoch"]

    use_vit_cache = model.uses_vit_cache()

    for epoch in range(start_epoch, model.epochs):
        model.refresh_epoch_state()

        perm = torch.randperm(N)
        sums = _new_sums(("total",) + _PER_STEP + ("adv_d",), model.device)

        beta = kl_beta(model, epoch)
        if beta < model.max_beta:
            pt.best_loss = float("inf")

        for i in range(0, N, model.batch_size):
            idx_cpu = perm[i:i + model.batch_size]
            idx = idx_cpu.to(model.device, non_blocking=model.device.type == "cuda")
            model.optimizer.zero_grad(set_to_none=True)

            x_batch = model._batch_to_device(model.Ytensor[idx_cpu])
            m_batch = model._batch_to_device(model.Mtensor[idx_cpu]) if model.Mtensor is not None else None
            Xb_norm = model.Xtensor_norm[idx]

            if use_vit_cache:
                feats_batch = model.backbone_feats[idx]
                mu_enc, logvar_enc = model.encoder.encode_from_feats_batched(feats_batch, b=feats_batch.size(0), t=1)
            else:
                mu_enc, logvar_enc = model.encoder(x_batch)

            z = model.reparameterize(mu_enc, logvar_enc)
            logits = model.decoder(z)

            adv_d = torch.tensor(0.0, device=model.device)
            use_gan_now = (
                model.use_decoder_gan
                and model.discriminator is not None
                and epoch >= model.gan_warmup_epochs
                and (model.loss_weights["gan"] > 0 or model.gan_feature_matching_weight > 0)
            )
            if use_gan_now:
                for _ in range(model.gan_disc_updates):
                    adv_d = model._train_discriminator_step(real_hwc=x_batch, fake_hwc=logits, mask=m_batch)

            total_loss, loss_dict = model.compute_total_loss(
                recon=logits, x=x_batch, mu=mu_enc, logvar=logvar_enc, mask=m_batch, kl_beta=beta,
                use_gan=use_gan_now, Xb_norm=Xb_norm, batch_idx=idx,
            )

            if not torch.isfinite(total_loss):
                print(
                    "NaN total loss at epoch", epoch, "step", i,
                    "logits min/max:", logits.min().item(), logits.max().item()
                )
                continue  # skipped batch; the epoch average still divides by N

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_generator_parameters_for_clipping(), max_norm=1.0)
            model.optimizer.step()

            bs = x_batch.size(0)
            sums["total"] += total_loss.detach().double() * bs
            for k in _PER_STEP:
                sums[k] += loss_dict[k].detach().double() * bs
            sums["adv_d"] += adv_d.detach().double() * bs

            del x_batch, m_batch, Xb_norm, mu_enc, logvar_enc, z, logits, total_loss, loss_dict, adv_d

        ep = _epoch_means(sums, N)
        history["loss_total_list"].append(ep["total"])
        for k in _PER_STEP:
            history[f"loss_{k}_list"].append(ep[k])
        history["loss_adv_d_list"].append(ep["adv_d"])

        # Best epoch (B2 fixed): a real copy, and only among epochs with the full KL weight. Before that the
        # KL weight is still 0 or ramping, so those losses are lower for that reason alone and would always win.
        if beta >= model.max_beta and ep["total"] < best_loss:
            best_loss = ep["total"]
            state = model.state_dict_full()
            for k in list(state.keys()):
                if "list" in k:
                    del state[k]
            best_state = cpu_copy(state)

        if epoch % pt.print_iterations == 0:
            lrs = [f"{g['lr']:.2e}" for g in model.optimizer.param_groups]
            extra_msg = model.get_extra_log_message()
            extra_msg = f" | {extra_msg}" if extra_msg else ""
            print(
                f"Epoch {epoch:04d} | "
                f"total={ep['total']:.6f} | "
                f"wmse={ep['weighted_mse']:.6f} (mse={ep['mse']:.6f}) | "
                f"wlpips={ep['weighted_lpips']:.6f} (lpips={ep['lpips']:.6f}) | "
                f"wgan={ep['weighted_gan']:.6f} (gan={ep['gan']:.6f}) | "
                f"wfm={ep['weighted_fm']:.6f} (fm={ep['fm']:.6f}) | "
                f"wkl={ep['weighted_kl']:.6f} (kl={ep['kl']:.6f}, beta={beta:.2e}) | "
                f"d={ep['adv_d']:.6f} | "
                f"lr={lrs[0]} | "
                f"no improvement count={pt.no_improvement_count}"
                f"{extra_msg}"
            )

        if epoch % (1000 * pt.print_iterations) == 0 and epoch > 0:
            _save_training_snapshot(model, epoch, use_vit_cache)

        if pt.check_stop_criterion(torch.tensor(ep["total"], device=model.device), epoch):
            print(f"Early stopping at epoch {epoch}")
            break

        model.update_lr(epoch)
        if checkpointer is not None:
            checkpointer.end_of_epoch(epoch, pt, best_loss, best_state)

    if best_state is None and any(loss < float("inf") for loss in history["loss_total_list"]):
        # No epoch with the full KL weight (a short run): keep the last epoch. The original still "restored"
        # here (its snapshot followed the live weights), and that restore has side effects worth keeping:
        # GP models build a new GP object, which draws random numbers (B15). So restore the current state.
        best_state = {k: v for k, v in model.state_dict_full().items() if "list" not in k}
    if best_state is not None:
        model.load_state_dict_full(best_state)

    print("Encoding Ytensor to get mu_all (using ViT cache if available)...")
    model.encoder.eval()
    with torch.no_grad():
        if model.uses_vit_cache():
            feats_all = model.backbone_feats
            mu_all, _ = model.encoder.encode_from_feats_batched(feats_all, b=feats_all.size(0), t=1)
        else:
            mu_all, _ = model._encode_ytensor_in_batches(batch_size=16, use_no_grad=True, move_to_cpu=True)

    model.mu_all = mu_all
    model.Z = mu_all.cpu().numpy()
    print(f"Finished encoding mu_all, shape {mu_all.shape}")

    for name, values in history.items():
        setattr(model, name, values)

    if checkpointer is not None:
        checkpointer.save_finished(pt, best_loss, best_state)
    print(f"{model.underlying_method} training complete.")


def _save_training_snapshot(model, epoch: int, use_vit_cache: bool) -> None:
    """Every 1000 epochs: decode one latent to a PNG and plot the latents (folders in the working dir)."""
    with torch.no_grad():
        N_sub = model.Ytensor.size(0)
        k = min(250, N_sub)
        if k == N_sub:
            idx_sub_cpu = torch.arange(N_sub)
        else:
            idx_sub_cpu = torch.from_numpy(np.linspace(0, N_sub - 1, k).round().astype(int))
        idx_sub = idx_sub_cpu.to(model.device, non_blocking=model.device.type == "cuda")
        Y_sub = model.Ytensor[idx_sub_cpu]
        if use_vit_cache:
            feats_sub = model.backbone_feats[idx_sub]
            mu_sub, _ = model.encoder.encode_from_feats_batched(feats_sub, b=feats_sub.size(0), t=1)
        else:
            mu_sub, _ = model.encoder(model._batch_to_device(Y_sub))
        Z = mu_sub.detach().cpu().numpy()

    decode_and_save_images_from_random_latents(
        model, Z, save_dir="Output_Random_Latent_Images",
        prefix=f"rand latent {model.underlying_method} {model.img_H} epoch {epoch} ", num_samples=1,
    )
    plot_and_save_latents(model, Z, epoch)
