"""Gaussian-process plots: prior covariance heat map and the GP posterior per latent dimension.

Moved from the original driver (Latent_Space_Discovery.py); module globals became explicit parameters.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_prior_covariance(model, str_with_params, subsample=200, *, save_folder):
    """Save a heat map of the GP prior covariance (or VAE middle covariance) of the first `subsample` frames."""
    print(f"Plotting posterior covars ({subsample}x{subsample}): {str_with_params}")
    try:
        x_sub = model.x[:subsample]

        if hasattr(model, "get_prior_covariance"):
            # GP models
            cov_train = model.get_prior_covariance(x_sub, 'train')
            title = 'K_XX Train'
        elif hasattr(model, "get_middle_covariance"):
            # VAE model
            cov_train = model.get_middle_covariance(x_sub)
            title = 'Middle cov Train'
        else:
            raise AttributeError("Model has no covariance method")

        plt.figure(figsize=(8, 6))
        plt.imshow(cov_train, cmap='viridis', origin='upper')
        plt.colorbar(label='Covariance')
        plt.xlabel('Time idx');
        plt.ylabel('Time idx')
        plt.title(f'{title} - {str_with_params}')
        plt.savefig(f"{save_folder}/Prior covar train {str_with_params}.png", dpi=150)
        plt.close()
        # print("✓ Prior covars saved!")
        print("Prior covars saved")
    except Exception as e:
        print(f"Skipping prior covar {str_with_params}: {e}")


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_gp_posterior_per_latent_combined(
        model,
        x_train,
        x_test,
        z_train_ref,
        z_test_ref,
        str_with_params="",
        n_samples=10,
        n_grid=1000,
        seed=42,
        use_likelihood=True,
        *,
        save_folder,
):
    """
    Plot per-latent GP posterior on a dense synthetic grid spanning the
    combined min/max of x_train and x_test.

    Inputs
    ------
    model       : GPVAE model with time_encode(..., return_full_cov=True)
    x_train     : (Ntr,) or (Ntr,1) original/scaled x values used as train latent targets
    x_test      : (Nte,) or (Nte,1) original/scaled x values used as test latent targets
    z_train_ref : (Ntr,Q) reference train latents to scatter
    z_test_ref  : (Nte,Q) reference test latents to scatter
    """

    def _sample_gp_trajectories_from_full_cov(mean, cov, n_samples=10, seed=42,
                                              jitter=1e-6, max_tries=8):
        mean = np.asarray(mean).reshape(-1)
        cov = np.asarray(cov)

        cov = 0.5 * (cov + cov.T)

        eye = np.eye(cov.shape[0], dtype=cov.dtype)
        this_jitter = jitter
        for _ in range(max_tries):
            try:
                L = np.linalg.cholesky(cov + this_jitter * eye)
                break
            except np.linalg.LinAlgError:
                this_jitter *= 10.0
        else:
            w, V = np.linalg.eigh(cov)
            w = np.clip(w, a_min=1e-8, a_max=None)
            L = V @ np.diag(np.sqrt(w))

        rng = np.random.default_rng(seed)
        eps = rng.standard_normal((cov.shape[0], n_samples))
        samples = mean[:, None] + L @ eps
        return samples.T  # (n_samples, M)

    x_train = _to_numpy(x_train).reshape(-1)
    x_test = _to_numpy(x_test).reshape(-1)
    z_train_ref = _to_numpy(z_train_ref)
    z_test_ref = _to_numpy(z_test_ref)

    Ntr, Q = z_train_ref.shape
    Nte, Q2 = z_test_ref.shape
    assert Q == Q2, "Train/test latent dims must match"
    assert x_train.shape[0] == Ntr, "x_train and z_train_ref mismatch"
    assert x_test.shape[0] == Nte, "x_test and z_test_ref mismatch"

    # Sort train/test references for clean scatter overlay
    idx_tr = np.argsort(x_train)
    idx_te = np.argsort(x_test)

    x_train = x_train[idx_tr]
    x_test = x_test[idx_te]
    z_train_ref = z_train_ref[idx_tr]
    z_test_ref = z_test_ref[idx_te]

    # Dense synthetic grid spanning combined min/max
    x_min = min(np.min(x_train), np.min(x_test))
    x_max = max(np.max(x_train), np.max(x_test))
    x_grid = np.linspace(x_min, x_max, n_grid, dtype=np.float32).reshape(-1, 1)

    # GP posterior on dense grid
    Z_grid_mean, Z_grid_std, Z_grid_cov = model.time_encode(
        x_grid,
        use_likelihood=use_likelihood,
        return_full_cov=True,
    )

    Z_grid_mean = _to_numpy(Z_grid_mean)
    Z_grid_std = _to_numpy(Z_grid_std)
    Z_grid_cov = _to_numpy(Z_grid_cov)

    # shared posterior std over grid from full covariance diagonal
    grid_std_shared = np.sqrt(np.clip(np.diag(Z_grid_cov), a_min=0.0, a_max=None))

    fig, axes = plt.subplots(
        Q, 1,
        figsize=(18, max(3 * Q, 6)),
        sharex=True,
        constrained_layout=True,
    )
    if Q == 1:
        axes = [axes]

    train_color = "tab:blue"
    test_color = "tab:orange"
    mean_color = "black"
    sample_color = "gray"
    band_color = "0.6"

    x_grid_flat = x_grid.reshape(-1)

    for q, ax in enumerate(axes):
        mu_q = Z_grid_mean[:, q]

        samples_q = _sample_gp_trajectories_from_full_cov(
            mean=mu_q,
            cov=Z_grid_cov,
            n_samples=n_samples,
            seed=seed + q,
            jitter=1e-6,
        )

        # variance band
        ax.fill_between(
            x_grid_flat,
            mu_q - 2.0 * grid_std_shared,
            mu_q + 2.0 * grid_std_shared,
            color=band_color,
            alpha=0.25,
            label=r"GP mean $\pm$ 2 std" if q == 0 else None,
        )

        # samples
        for k in range(n_samples):
            ax.plot(
                x_grid_flat,
                samples_q[k],
                color=sample_color,
                alpha=0.18,
                linewidth=0.8,
                label="Posterior samples" if (q == 0 and k == 0) else None,
            )

        # posterior mean
        ax.plot(
            x_grid_flat,
            mu_q,
            color=mean_color,
            linewidth=2.0,
            label="GP posterior mean" if q == 0 else None,
        )

        # train/test latent targets
        ax.scatter(
            x_train,
            z_train_ref[:, q],
            s=14,
            color=train_color,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.85,
            label="Train latent target" if q == 0 else None,
        )
        ax.scatter(
            x_test,
            z_test_ref[:, q],
            s=18,
            color=test_color,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.90,
            label="Test latent target" if q == 0 else None,
        )

        ax.set_ylabel(f"z[{q}]")
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].set_title(f"GP posterior per latent {str_with_params}")
    axes[-1].set_xlabel("Time / frame coordinate")

    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        axes[0].legend(loc="upper right", ncol=4, fontsize=9)

    save_path = os.path.join(
        save_folder,
        f"GP posterior combined per latent {str_with_params}.png",
    )
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
