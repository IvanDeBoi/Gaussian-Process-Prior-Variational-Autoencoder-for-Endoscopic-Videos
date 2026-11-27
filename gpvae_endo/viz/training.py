"""Training diagnostics: loss curves and the per-frame RMSE-versus-frame scatter.

Moved from the original driver (Latent_Space_Discovery.py); module globals became explicit parameters.
"""
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_losses(model, str_with_params, *, paths):
    """Save a plot of the last 80% of the training loss curves (total, MSE, LPIPS, KL, GAN, FM)."""
    print("Plotting loss function")

    total_epochs = len(model.loss_total_list)
    if total_epochs == 0:
        return

    percentage = 80
    start = int((100 - percentage) / 100 * total_epochs)
    epochs = range(start, total_epochs)

    loss_total = model.loss_total_list[start:]
    loss_weighted_mse = model.loss_weighted_mse_list[start:]
    loss_weighted_lpips = model.loss_weighted_lpips_list[start:]
    loss_weighted_kl = model.loss_weighted_kl_list[start:]
    loss_weighted_gan = model.loss_weighted_gan_list[start:]
    loss_weighted_fm = model.loss_weighted_fm_list[start:]

    plt.figure(figsize=(16, 6))
    plt.plot(epochs, loss_total, label="Total loss")
    plt.plot(epochs, loss_weighted_mse, label="Weighted MSE loss")
    plt.plot(epochs, loss_weighted_lpips, label="Weighted LPIPS loss")
    plt.plot(epochs, loss_weighted_kl, label="Weighted KL loss")
    plt.plot(epochs, loss_weighted_gan, label="Weighted GAN loss")
    plt.plot(epochs, loss_weighted_fm, label="Weighted FM loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Losses during training {str_with_params} (last {percentage}%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    ymin, ymax = plt.ylim()
    plt.ylim(bottom=0.0, top=ymax)

    plt.tight_layout()
    plt.savefig(
        paths.output + f"Losses during training {str_with_params} last {percentage} pct.png"
    )
    plt.close()


def plot_rmse_vs_frame(
        x_train_scaled,
        x_test_scaled,
        rmse_train_per_image,
        rmse_test_per_image,
        std_train,
        std_test,
        str_with_params,
        *,
        paths,
):
    """
    Save a scatter of per-frame RMSE vs frame number and a text file with per-frame uncertainty and RMSE.

    x_*_scaled: (N,1) or (N,) frame numbers (same order as Y_*)
    rmse_*_per_image: (N,)
    std_*: (N,Q) predictive std from model.time_encode
    """
    min_size = 10
    max_size = 100

    x_train = np.asarray(x_train_scaled).squeeze()
    x_test = np.asarray(x_test_scaled).squeeze()
    rmse_train = np.asarray(rmse_train_per_image).squeeze()
    rmse_test = np.asarray(rmse_test_per_image).squeeze()
    std_train = np.asarray(std_train)
    std_test = np.asarray(std_test)

    # aggregate std over latent dims
    std_train_1d = std_train.mean(axis=1)
    std_test_1d = std_test.mean(axis=1)

    all_std = np.concatenate([std_train_1d, std_test_1d])
    s_min, s_max = all_std.min(), all_std.max()
    if s_max == s_min:
        size_train = np.full_like(std_train_1d, min_size, dtype=float)
        size_test = np.full_like(std_test_1d, min_size, dtype=float)
    else:
        size_train = min_size + (std_train_1d - s_min) * (max_size - min_size) / (s_max - s_min)
        size_test = min_size + (std_test_1d - s_min) * (max_size - min_size) / (s_max - s_min)

    fig, ax = plt.subplots(figsize=(10, 6))

    sc_train = ax.scatter(
        x_train,
        rmse_train,
        c=rmse_train,  # color by error, plasma
        cmap='jet',
        s=size_train,
        alpha=0.4,
        marker='o',
        edgecolors='k',
        linewidths=0.3,
        label='Train'
    )

    sc_test = ax.scatter(
        x_test,
        rmse_test,
        c='red',
        s=size_test,
        alpha=0.7,
        marker='o',
        edgecolors='k',
        linewidths=0.3,
        label='Test'
    )

    cbar = fig.colorbar(sc_train, ax=ax, pad=0.02)
    cbar.set_label('RMSE (denormalized)')

    ax.set_xlabel('Frame number')
    ax.set_ylabel('RMSE (denormalized)')
    ax.set_title(f'RMSE (denormalized) vs Frame {str_with_params}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(paths.output + f"RMSE vs frame_{str_with_params}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save per-frame uncertainty values in frame-number order.
    records = []
    for frame_nr, uncertainty, rmse in zip(x_train, std_train_1d, rmse_train):
        records.append((int(frame_nr), "train", float(uncertainty), float(rmse)))
    for frame_nr, uncertainty, rmse in zip(x_test, std_test_1d, rmse_test):
        records.append((int(frame_nr), "test", float(uncertainty), float(rmse)))

    records.sort(key=lambda row: row[0])

    txt_path = os.path.join(
        paths.output,
        f"{str_with_params} uncertainty per frame.txt",
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("frame_nr\tsplit\tuncertainty\trmse_denormalized\n")
        for frame_nr, split, uncertainty, rmse in records:
            f.write(f"{frame_nr}\t{split}\t{uncertainty:.8f}\t{rmse:.8f}\n")


# --------------------------------------------------------------------------------------------------------
# Snapshots during training (every 1000 epochs; from the original Models/VAE_model.py)
# --------------------------------------------------------------------------------------------------------
def decode_and_save_images_from_random_latents(model, Z: np.ndarray, save_dir: str, prefix: str,
                                               num_samples: int = 1) -> None:
    """Decode a few latents of `Z` and save them as PNGs (the decoder is left in train mode, as before)."""
    model.decoder.eval()
    N = Z.shape[0]
    num_samples = min(num_samples, N)
    if num_samples == 1:
        idx = np.array([6]) if N > 6 else np.array([0])
    else:
        idx = np.linspace(0, N - 1, num_samples).round().astype(int)
    z_np = Z[idx, :]

    import torch
    with torch.no_grad():
        imgs_np = model.decode(z_np)

    os.makedirs(save_dir, exist_ok=True)
    for j in range(num_samples):
        img_hwc = imgs_np[j].reshape(model.img_H, model.img_W, 3)
        img_hwc = np.clip(img_hwc, 0.0, 1.0)
        plt.figure(figsize=(4, 4))
        plt.imshow(img_hwc)
        plt.axis("off")
        fname = os.path.join(save_dir, f"{prefix} {idx[j]:06d}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight", pad_inches=0.0)
        plt.close()
    model.decoder.train()


def plot_and_save_latents(model, Z: np.ndarray, epoch: int, out_dir: str = "Output_Latent_Plots") -> None:
    """Scatter of the first three latent dimensions."""
    Q = Z.shape[1]
    idx_ard = np.arange(3) if Q >= 3 else np.arange(Q)
    Z_ard = Z[:, idx_ard]
    fig = plt.figure(figsize=(7, 7))

    if len(idx_ard) == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(Z_ard[:, 0], Z_ard[:, 1], Z_ard[:, 2], s=5, alpha=0.4, label="Latents Z")
        ax.set_zlabel(f"Latent dim {idx_ard[2]}")
    else:
        ax = fig.add_subplot(111)
        ax.scatter(Z_ard[:, 0], Z_ard[:, 1], s=5, alpha=0.4, label="Latents Z")

    ax.set_xlabel(f"Latent dim {idx_ard[0]}")
    ax.set_ylabel(f"Latent dim {idx_ard[1]}")
    ax.set_title(f"{model.underlying_method} (epoch {epoch})\ndims {idx_ard.tolist()}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"latents_{model.underlying_method} epoch {epoch:06d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
