import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: config_endo, gpvae_endo
# --- Imports ---
import gpytorch
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from datetime import datetime
import time

import torch
from torch.optim import Adam

from Trash.GPLVM import bGPLVM
from gpvae_endo.training.progress import ProgressTracker
from config_endo import (
    SEED,
    FOLDER_DATA,
    FOLDER_OUTPUT,
    RESOLUTION,
    SUBSTRING_FILTER,
    DEVICE,
    PCA_TARGET_VARIANCE,
)

import os
import numpy as np

from skdim.id import KNN
from skdim.id import MLE
from skdim.id import TwoNN  # <-- already imported


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_pca(pca, Y, str_with_params, imgshape=None, max_components=10,
             cmap="bwr", rows=3, cols=3):

    # 1) Scree plot (explained variance ratio)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    n_comp = len(pca.explained_variance_ratio_)
    xs = np.arange(1, n_comp + 1)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax1.plot(xs, cumvar, "-o", color="red", label="Cumulative")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Cumulative explained variance", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.bar(xs, pca.explained_variance_ratio_, alpha=0.7, color="blue",
            label="Individual")
    ax2.set_ylabel("Individual explained variance ratio", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    plt.title("PCA explained variance")
    fig.tight_layout()
    fig.savefig(os.path.join(FOLDER_OUTPUT,
                             f"PCA explained variance {str_with_params}.png"))
    plt.close(fig)

    # 2) Scree with log x-scale
    fig, ax1 = plt.subplots(figsize=(8, 4))
    xs = np.arange(1, n_comp + 1)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax1.plot(xs, cumvar, "-o", color="red", label="Cumulative")
    ax1.set_xscale("log")
    ax1.set_xlabel("Principal Component (log scale)")
    ax1.set_ylabel("Cumulative explained variance", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.bar(xs, pca.explained_variance_ratio_, alpha=0.7, color="blue",
            label="Individual")
    ax2.set_ylabel("Individual explained variance ratio", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    plt.title("PCA explained variance (log x-scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(FOLDER_OUTPUT,
                             f"PCA explained variance logx {str_with_params}.png"))
    plt.close(fig)

    # X_pca: scores in PC space
    X_pca = pca.transform(Y)  # (n_samples, n_components)
    assert X_pca.shape[0] == Y.shape[0]
    if X_pca.shape[1] < 3:
        return
    idx = np.arange(Y.shape[0])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
        c=idx, cmap="viridis", s=5, alpha=0.8
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.title("Latent space 3D")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Number order")
    fig.tight_layout()
    fig.savefig(os.path.join(FOLDER_OUTPUT,
                             f"PCA latent 3D {str_with_params}.png"))
    plt.close(fig)

    # 3D interactive plot
    colors_interactive = idx
    fig = go.Figure(data=[
        # 1. Line Trace (Trajectory)
        go.Scatter3d(
            x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
            mode='lines',
            line=dict(color='black', width=2),
            name='Trajectory'
        ),
        # 2. Scatter Trace (Points, colored by time)
        go.Scatter3d(
            x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors_interactive,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Frame Index')
            ),
            text=idx,
            hoverinfo='text',
            name='Latent space PCA (capped 3D)'
        )
    ])
    fig.update_layout(
        title=f'3D Latent {str_with_params}',
        scene=dict(
            xaxis_title='Dim 1',
            yaxis_title='Dim 2',
            zaxis_title='Dim 3'
        )
    )
    fig.write_html(FOLDER_OUTPUT + f"PCA Latent 3D {str_with_params}.html")




    if imgshape is None:
        return

    # 4) Component images
    K = min(rows * cols, pca.components_.shape[0])
    plt.figure(figsize=(4 * cols, 4 * rows))
    for k in range(K):
        comp = pca.components_[k]
        comp_img = comp.reshape(imgshape)
        r = k // cols
        c = k % cols
        plt.subplot(rows, cols, k + 1)
        vmin, vmax = np.percentile(comp, [1, 99])
        if comp_img.ndim == 2:
            plt.imshow(comp_img, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            comp_norm = (comp_img - vmin) / (vmax - vmin + 1e-8)
            comp_norm = np.clip(comp_norm, 0, 1)
            plt.imshow(comp_norm)
        plt.axis("off")
        plt.title(f"PC {k + 1}")

    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER_OUTPUT,
                             f"PCA components {str_with_params}.png"))
    plt.close()


def plot_ARD_contributions(gplvm_model, label="TO BE SET"):
    # Extract lengthscales and ARD relevance
    ls = gplvm_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    inv_ls = 1.0 / ls                        # relevance: larger = more important
    rel_importance = inv_ls / inv_ls.sum()   # normalized for plotting

    # Sort relevances in descending order to find elbow
    inv_sorted = np.sort(inv_ls)[::-1]

    if inv_sorted.size > 1:
        # ratios between consecutive relevances
        ratios = inv_sorted[:-1] / inv_sorted[1:]
        elbow_idx = int(np.argmax(ratios))   # 0-based
        eff_dim = elbow_idx + 1
        # relevance cutoff at the elbow
        cutoff = inv_sorted[elbow_idx]
        active = inv_ls >= cutoff
    else:
        eff_dim = inv_sorted.size
        active = np.ones_like(inv_ls, dtype=bool)

    dims = np.arange(len(inv_ls))

    plt.figure(figsize=(7, 4))
    bars = plt.bar(dims, rel_importance, color="steelblue", alpha=0.8)

    # Highlight active vs inactive based on elbow cutoff
    for d, (b, is_active) in enumerate(zip(bars, active)):
        if is_active:
            b.set_color("tab:orange")

    plt.xlabel("Latent dimension index")
    plt.ylabel("Relative ARD importance")
    plt.title(
        f"GPLVM ARD relevance per latent dim ({label})\n"
        f"Effective dim (elbow) ≈ {eff_dim}"
    )
    plt.grid(axis="y", alpha=0.3)

    # Annotate exact values on top (optional)
    for d, val in enumerate(rel_importance):
        plt.text(d, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)

    plt.tight_layout()
    out_path = os.path.join(FOLDER_OUTPUT, f"GPLVM ARD importance {label}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_loss(loss_list, label="TO BE SET"):
    iters = np.arange(len(loss_list))
    plt.figure(figsize=(7, 4))
    plt.plot(iters, loss_list, "-b")
    plt.xlabel("Iteration")
    plt.ylabel("Negative ELBO loss")
    plt.title(f"GPLVM training loss ({label})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FOLDER_OUTPUT, f"GPLVM training loss {label}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

def gplvm_dim_from_ard(gplvm_model, min_dim=1):
    """
    Compute a global latent dimension from ARD by sorting inverse lengthscales
    and looking for the largest ratio r_i / r_{i+1} (elbow).
    """
    ls = gplvm_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    inv_ls = 1.0 / ls   # relevance: larger = more important

    # sort descending by relevance
    order = np.argsort(-inv_ls)
    inv_sorted = inv_ls[order]

    # if all equal or only 1 dim, fall back
    if inv_sorted.size <= 1:
        return inv_sorted.size, inv_ls, order

    # ratios between consecutive relevances
    ratios = inv_sorted[:-1] / inv_sorted[1:]

    # find largest ratio (elbow)
    elbow_idx = int(np.argmax(ratios))      # index in 0-based; dim = elbow_idx+1

    # enforce minimum dimension
    global_dim = max(min_dim, elbow_idx + 1)
    return global_dim, inv_ls, order


if __name__ == "__main__":

    start_time_total = time.time()
    print(f"Start time: {datetime.fromtimestamp(start_time_total).strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n***************************************************************************************************\n")

    for resolution in RESOLUTION:
        for substring_filter in SUBSTRING_FILTER:
            start_time = time.time()
            print(
                f"Start time for substring filter: {substring_filter}, resolution: {resolution}, seed: {SEED}, time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")

            str_with_params = f"{substring_filter} {resolution}"

            npz_path = os.path.join(FOLDER_DATA, f"{str_with_params} XY.npz")

            data = np.load(npz_path)
            x = data["x"]  # frame numbers, not used for PCA here
            Y = data["Y"]  # flattened, normalized images; shape (N, D)
            print(Y.shape)
            print(x.shape)

            print("Substract mean")
            Y_centered = Y - Y.mean(axis=0, keepdims=True)

            print("Perform PCA")
            pca = PCA(n_components=PCA_TARGET_VARIANCE, svd_solver="full")
            Y_pca = pca.fit_transform(Y_centered)
            print("Y_pca shape:", Y_pca.shape)
            print("Explained variance sum:", pca.explained_variance_ratio_.sum())

            print("Plot PCA results")
            # plot_pca(pca)
            plot_pca(pca, Y, str_with_params=str_with_params, imgshape=(resolution, resolution, 3),
                     max_components=10, cmap="bwr", rows=8, cols=8)

            print("Investigate lambdas")
            lambdas = pca.explained_variance_
            lambda_ratios = lambdas[:-1] / lambdas[1:]
            max_ratio_idx = int(np.argmax(lambda_ratios))
            max_ratio_value = float(lambda_ratios[max_ratio_idx])
            print("Max lambda ratio:", max_ratio_value)
            print("Index in list (0-based):", max_ratio_idx)
            print("This is λ_{} / λ_{}".format(max_ratio_idx + 1, max_ratio_idx + 2))
            print("Estimated intrinsic dimension (PCA global):", max_ratio_idx + 1)
            explained_after_two = pca.explained_variance_ratio_[: max_ratio_idx + 1].sum()
            global_instrinsic_dim = max_ratio_idx + 1
            print(
                f"Cumulative explained variance after these {global_instrinsic_dim} eigenvalues: {explained_after_two}")

            print("Investigate non-linear dimensionality methods")
            K_values = [5, 10, 15, 25, 50, 75]

            # --- containers for logging ---
            knn_ids = []
            mle_local_ids = []
            mle_global_ids = []
            lpca_ids = []  # local PCA IDs
            gplvm_ids = []

            # --- TwoNN (parameter-free) ---
            print("Trying TwoNN (global ID)")
            twonn = TwoNN()
            id_twonn = twonn.fit_transform(Y_pca)
            print(f"Estimated intrinsic dimension (TwoNN): {id_twonn:.6f}")

            # --- KNN ---
            for K in K_values:
                print(f"Trying KNN with K = {K}")
                knn_id = KNN(k=K)
                id_est = knn_id.fit_transform(Y_pca)
                if id_est > 25: #unreasonable
                    id_est = -1 #code for failure
                knn_ids.append(id_est)
                print(f"K = {K}, Estimated intrinsic dimension (KNN): {id_est:.6f}")

            # --- MLE with precomputed kNN ---
            max_K = max(K_values)
            print("Precompute knn distances")
            nbrs = NearestNeighbors(n_neighbors=max_K + 1, algorithm='auto').fit(Y_pca)
            distances, indices = nbrs.kneighbors(Y_pca)

            for K in K_values:
                print(f"Trying MLE with K = {K}, neighborhood_based=True")
                mle_id = MLE(neighborhood_based=True, K=K, integral_approximation="Haro", unbiased=True)
                id_est = mle_id.fit_transform(
                    Y_pca,
                    precomputed_knn_arrays=(distances[:, :K + 1], indices[:, :K + 1])
                )
                if id_est > 25: #unreasonable
                    id_est = -1 #code for failure
                mle_local_ids.append(id_est)
                print(f"K = {K}, Estimated intrinsic dimension (MLE, local): {id_est:.6f}")

            # for K in K_values:
            #     print(f"Trying MLE with K = {K}, neighborhood_based=False")
            #     mle_id = MLE(neighborhood_based=False, K=K, integral_approximation="Haro", unbiased=True)
            #     id_est = mle_id.fit_transform(
            #         Y_pca,
            #         precomputed_knn_arrays=(distances[:, :K + 1], indices[:, :K + 1])
            #     )
            #     mle_global_ids.append(id_est)
            #     print(f"K = {K}, Estimated intrinsic dimension (MLE, global): {id_est:.6f}")

            # --- Local PCA ID (simple implementation) ---
            # For each K, run local PCA in k-NN neighborhoods and count components
            # explaining at least `lpca_var_thresh` (global) of local variance.
            max_K = max(K_values)
            print("Precompute knn distances for local PCA")
            nbrs = NearestNeighbors(n_neighbors=max_K + 1, algorithm='auto').fit(Y_pca)
            distances, indices = nbrs.kneighbors(Y_pca)
            lpca_var_thresh = 0.75  # e.g., 95% of local variance
            for K in K_values:
                print(f"Trying local PCA with K = {K}")
                # use precomputed neighbor indices, skip self at index 0
                neigh_idx = indices[:, 1:K + 1]  # shape: (N, K)
                local_dims = []
                for i in range(Y_centered.shape[0]):
                    Xi = Y_pca[neigh_idx[i]]  # (K, D) neighbors of point i
                    # center neighborhood
                    Xi_centered = Xi - Xi.mean(axis=0, keepdims=True)
                    # PCA on local neighborhood
                    lpca = PCA(n_components=min(Xi_centered.shape[0], Xi_centered.shape[1]))
                    lpca.fit(Xi_centered)
                    cumvar = np.cumsum(lpca.explained_variance_ratio_)
                    d_loc = int(np.searchsorted(cumvar, lpca_var_thresh) + 1)
                    local_dims.append(d_loc)
                lpca_id = float(np.mean(local_dims))
                if lpca_id > 25: #unreasonable
                    lpca_id = -1 #code for failure
                lpca_ids.append(lpca_id)
                print(f"K = {K}, Estimated intrinsic dimension (local PCA): {lpca_id:.6f}")

            # GPLVM
            print("Apply PCA to reduce")
            # PCA reduce Y to min_explained_var % variance ---
            # pca_red = PCA(n_components=PCA_TARGET_VARIANCE)  # Keep min_explained_var % variance
            # Y_reduced = pca_red.fit_transform(Y_centered)  # Now (800, ~150)
            # print(f"  PCA reduced Y: {Y.shape[1]}D -> {Y_reduced.shape[1]}D ({PCA_TARGET_VARIANCE * 100}% var)")
            print(f"  PCA reduced Y: {Y.shape[1]}D -> {Y_pca.shape[1]}D ({PCA_TARGET_VARIANCE * 100}% var)")
            # print("Apply standard scaler after PCA")
            # scaler = StandardScaler(with_mean=True, with_std=True)
            # Y_std = scaler.fit_transform(Y_reduced)

            # print(f"  Y_std  {Y_std.shape}")
            for K in K_values:
                label = str_with_params + " GPLVM " + str(K) + " inducing"
                Y_torch = torch.from_numpy(Y_pca).float().to(DEVICE)
                N, data_dim = Y_torch.shape
                latent_dim = min(10, Y_pca.shape[1]) #could be lower than ?, e.g. 32x32 pixel CB
                n_inducing = K
                gplvm_model = bGPLVM(
                    n=Y_torch.shape[0],
                    data_dim=Y_torch.shape[1],
                    latent_dim=latent_dim,
                    n_inducing=n_inducing,
                    pca=False,
                    Y=Y_torch,
                    kernel="RBF"
                ).to(DEVICE)
                max_y = torch.max(Y_torch).item()
                min_y = torch.min(Y_torch).item()
                range_y = max_y - min_y
                upper_bound_noise = 0.01 * range_y #this is stupid, if the data is scaled
                init_noise = 0.5 * upper_bound_noise
                print(f"  Range Y ({max_y:.3f} - {min_y:.3f}) = {range_y:.3f} => upper bound noise {upper_bound_noise:.3f}, init noise {init_noise:.3f}")
                # upper_bound_noise = 0.01
                likelihood = GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.Interval(1e-6, upper_bound_noise),
                    noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-6, upper_bound_noise, sigma=0.5)  # Realistic prior
                ).to(DEVICE)
                likelihood.noise = torch.tensor(init_noise, device=DEVICE)
                mll = VariationalELBO(likelihood, gplvm_model, num_data=len(Y_centered), beta=1.0)
                pt = ProgressTracker(patience=250, max_iterations=10000, print_iterations=50)

                model_params = [p for n, p in gplvm_model.named_parameters() if "X" not in n]
                latent_params = gplvm_model.X.parameters()
                optimizer = Adam([
                    {'params': model_params, 'lr': 5e-3},
                    {'params': latent_params, 'lr': 1e-3},
                    {'params': likelihood.parameters(), 'lr': 5e-4},
                ])

                params = list(gplvm_model.parameters()) + list(likelihood.parameters())

                # optimizer = torch.optim.LBFGS(
                #     params,
                #     lr=0.001,  # often needs a larger lr than Adam, tune 0.1–1.0
                #     max_iter=20,  # LBFGS inner iterations per outer loop
                #     # line_search_fn="strong_wolfe"
                # )

                loss_list = []
                # batch_size = 512
                batch_size = Y.shape[0]
                print(f"   Working with batch size {batch_size}")
                print(f"   Working with {n_inducing} inducing points")
                for i in range(pt.max_iterations + 1):
                    batch_index = gplvm_model._get_batch_idx(batch_size)


                    # def closure():
                    #     optimizer.zero_grad()
                    #     sample = gplvm_model.sample_latent_variable()
                    #     sample_batch = sample[batch_index]
                    #     output_batch = gplvm_model(sample_batch)
                    #     loss = -mll(output_batch, Y_torch[batch_index].T).sum()
                    #     loss.backward()
                    #     return loss
                    # loss = optimizer.step(closure)

                    optimizer.zero_grad()
                    sample = gplvm_model.sample_latent_variable()
                    sample_batch = sample[batch_index]

                    sample_batch = gplvm_model.X.q_mu[batch_index]

                    output_batch = gplvm_model(sample_batch)
                    loss = -mll(output_batch, Y_torch[batch_index].T).sum()
                    loss_list.append(loss.item())
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(gplvm_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    if pt.check_stop_criterion(loss=loss, current_iteration=i):
                        break
                    # ---- build one-line hyperparameter string from underlying GP (interpretable) ----
                    def _hp_str_gp_decoder(model, likelihood, prefix=""):
                        covar = model.covar_module
                        base = covar.base_kernel if hasattr(covar, "base_kernel") else covar
                        ls = base.lengthscale.detach().cpu().view(-1)
                        os = covar.outputscale.detach().cpu().item()
                        noise = likelihood.noise.detach().cpu()  # [3072]
                        noise_mean = noise.mean().item()
                        noise_std = noise.std(unbiased=False).item()
                        ls_parts = [f"{v:.2f}" for v in ls]
                        parts = [
                            f"{prefix}ls=[{','.join(ls_parts)}]",
                            f"{prefix}os={os:.3f}",
                            f"{prefix}noise_mean={noise_mean:.4f}",
                            f"{prefix}noise_std={noise_std:.5f}",
                        ]
                        if hasattr(model, "X"):
                            q_mu0 = model.X.q_mu.detach().cpu().view(-1)[0]
                            q_log_sigma0 = model.X.q_log_sigma.detach().cpu().view(-1)[0]
                            parts.append(f"{prefix}q_mu0={q_mu0:.3f}")
                            parts.append(f"{prefix}q_log_sigma0={q_log_sigma0:.3f}")
                        return "|".join(parts)
                    if i % pt.print_iterations == 0:
                        hyperparams_str = _hp_str_gp_decoder(gplvm_model, likelihood)
                        # print(f"     Iter {i}: loss = {loss.item():.2f} | {hyperparams_str}")
                        print(f"   {hyperparams_str}")
                        # ---------- Plot latents and inducing points (3D, ARD-sorted) ----------
                        with torch.no_grad():
                            Z = gplvm_model.X.q_mu.detach().cpu().numpy()
                            inducing = gplvm_model.variational_strategy.inducing_points.detach().cpu().numpy()
                        if inducing.ndim == 3:
                            inducing_full = inducing[0]
                        else:
                            inducing_full = inducing
                        with torch.no_grad():
                            covar = gplvm_model.covar_module
                            base_kernel = covar.base_kernel if hasattr(covar, "base_kernel") else covar
                            ls_tensor = base_kernel.lengthscale.detach().cpu()
                        if ls_tensor.dim() > 1:
                            ls = ls_tensor.mean(dim=list(range(ls_tensor.dim() - 1))).numpy()
                        else:
                            ls = ls_tensor.numpy()
                        top_k = 3
                        idx_sorted = np.argsort(ls)
                        idx_ard = idx_sorted[:top_k]
                        Z_ard = Z[:, idx_ard]
                        inducing_ard = inducing_full[:, idx_ard]
                        fig = plt.figure(figsize=(7, 7))
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(Z_ard[:, 0], Z_ard[:, 1], Z_ard[:, 2],
                                   s=5, alpha=0.4, label="Latents Z")
                        ax.scatter(inducing_ard[:, 0], inducing_ard[:, 1], inducing_ard[:, 2],
                                   s=50, alpha=0.9, marker="x", color="red", label="Inducing")
                        ax.set_xlabel(f"Latent dim {idx_ard[0]}")
                        ax.set_ylabel(f"Latent dim {idx_ard[1]}")
                        ax.set_zlabel(f"Latent dim {idx_ard[2]}")
                        ax.set_title(f"GPLVM Latents & Inducing\nIter {i}, K {K}, ARD dims {idx_ard.tolist()}")
                        ax.legend()
                        ax.grid(True, linestyle="--", alpha=0.3)
                        # Fixed axis limits
                        # for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
                        #     set_lim(-15, 15)
                        out_dir = "Output_Latent_Plots"
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"latents_inducing_GPLVM_3d_ARD_iter_{K}_{i:06d}.png")
                        plt.savefig(out_path, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                    elif i == 1:
                        hyperparams_str = _hp_str_gp_decoder(gplvm_model, likelihood)
                        print(f"   Initial loss = {loss.item():.2f} | {hyperparams_str}")
                plot_training_loss(loss_list, label=label)
                # After training, store latent means from best model
                Z = gplvm_model.X.q_mu.detach().cpu().numpy()
                loss_list_decoder = loss_list
                # ARD-based global dim via elbow on sorted inverse lengthscales
                eff_dim_gplvm, inv_lengthscales, ard_order = gplvm_dim_from_ard(gplvm_model, min_dim=1)
                print("Inverse lengthscales (unsorted):", inv_lengthscales)
                print("ARD dims sorted by relevance:", ard_order)
                print("Effective latent dimension (Bayesian GPLVM, elbow):", eff_dim_gplvm)
                plot_ARD_contributions(gplvm_model, label=label)
                gplvm_ids.append(eff_dim_gplvm)

            # --- Plot ID vs K ---
            plt.figure(figsize=(8, 5))
            plt.axhline(y=id_twonn, color="gray", linestyle="--", label=f"TwoNN ≈ {id_twonn:.2f}")
            plt.axhline(y=global_instrinsic_dim, color="red", linestyle=":", label=f"PCA gap ≈ {global_instrinsic_dim}")
            plt.plot(K_values, knn_ids, "-o", label="KNN")
            # plt.plot(K_values, mle_local_ids, "-s", label="MLE (neighborhood_based=True)")
            # plt.plot(K_values, mle_global_ids, "-^", label="MLE (neighborhood_based=False)")
            # plt.plot(K_values, lpca_ids, "-d", label=f"Local PCA ({int(lpca_var_thresh * 100)}% var)")
            plt.plot(K_values, gplvm_ids, "-d", label=f"GPLVM")

            plt.xlabel("K (number of neighbors or nr inducing points GPLVM)")
            plt.ylabel("Estimated intrinsic dimension")
            plt.title(f"Intrinsic dimension vs K {substring_filter} {resolution}")
            plt.legend()
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(FOLDER_OUTPUT, f"ID vs K {substring_filter} {resolution}.png"))
            plt.close()

            end_time = time.time()
            print(f"\nTime this run: {(end_time - start_time):.3f}s")
            print(
                "\n***************************************************************************************************\n")

    end_time_total = time.time()
    print(f"\nTotal time: {(end_time_total - start_time_total):.3f}s")
