"""Latent-space plots: 3D latent trajectory, 2D latent with image thumbnails, and PCA variance plots.

Moved from the original driver (Latent_Space_Discovery.py); module globals became explicit parameters.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA


def plot_latent(
        Z,
        reconstructed_Z_train,
        reconstructed_Z_test,
        std_train,
        std_test,
        str_with_params,
        *,
        paths,
):
    """Save a 3D latent plot (PNG and interactive HTML) of Z and the reconstructed train/test latents."""
    print(f"Plotting Latent Space")
    min_size_3D = 8
    max_size_3D = 40
    Z = np.asarray(Z)
    reconstructed_Z_train = np.asarray(reconstructed_Z_train)
    reconstructed_Z_test = np.asarray(reconstructed_Z_test)
    std_train = np.asarray(std_train)
    std_test = np.asarray(std_test)

    N_mo, Q = Z.shape
    N_tr, Q2 = reconstructed_Z_train.shape

    # print(reconstructed_Z_test)
    # print(type(reconstructed_Z_test))
    # print(reconstructed_Z_test.shape)

    if reconstructed_Z_test is not None and reconstructed_Z_test.ndim >= 2:
        N_te, Q3 = reconstructed_Z_test.shape
    else:
        N_te = 0
        Q3 = Q2

    # ---- PCA to 3D if needed ----
    if Q < 3:
        raise ValueError(f"Latent dim {Q} < 3, cannot make 3D plot")
    elif Q == 3:
        if N_te > 0:
            Z3all = np.vstack([Z, reconstructed_Z_train, reconstructed_Z_test])
        else:
            Z3all = np.vstack([Z, reconstructed_Z_train])
    else:
        if N_te > 0:
            Zall = np.vstack([Z, reconstructed_Z_train, reconstructed_Z_test])
            Z3all = PCA(n_components=3).fit_transform(Zall)
        else:
            Zall = np.vstack([Z, reconstructed_Z_train])
            Z3all = PCA(n_components=3).fit_transform(Zall)


    Z3mo = Z3all[:N_mo]
    Z3tr = Z3all[N_mo:N_mo + N_tr]
    Z3te = Z3all[N_mo + N_tr:]

    n_frames = N_mo
    colors = np.arange(n_frames)

    # ---------- static matplotlib 3D ----------
    fig_3D = plt.figure(figsize=(10, 10))
    ax = fig_3D.add_subplot(111, projection='3d')

    sc = ax.scatter(Z3mo[:, 0], Z3mo[:, 1], Z3mo[:, 2],
                    c=colors, cmap='viridis', alpha=0.6, s=15, label="Z")
    ax.scatter(Z3tr[:, 0], Z3tr[:, 1], Z3tr[:, 2],
               c=colors, cmap='viridis', alpha=0.4, s=30, label="recon train")
    ax.scatter(Z3te[:, 0], Z3te[:, 1], Z3te[:, 2],
               c='red', alpha=0.4, s=30, label="recon test")

    fig_3D.colorbar(sc, ax=ax, pad=0.1).set_label('Frame Index')
    ax.set_xlabel('Latent 1 (PCA)' if Q > 3 else 'Dim 1')
    ax.set_ylabel('Latent 2 (PCA)' if Q > 3 else 'Dim 2')
    ax.set_zlabel('Latent 3 (PCA)' if Q > 3 else 'Dim 3')
    ax.set_title(f'3D Latent {str_with_params}')
    ax.legend()
    plt.savefig(paths.output + f"3D Latent {str_with_params}.png")
    plt.close()

    # ---------- sizes from std ----------
    std_train_1d = std_train.mean(axis=1)
    if N_te > 0:
        std_test_1d = std_test.mean(axis=1)
        all_std = np.concatenate([std_train_1d, std_test_1d])
    else:
        all_std = np.concatenate([std_train_1d])

    s_min, s_max = all_std.min(), all_std.max()
    if s_max == s_min:
        size_train = np.full_like(std_train_1d, min_size_3D)
        if N_te > 0:
            size_test = np.full_like(std_test_1d, min_size_3D)
        else:
            size_test = 1
    else:
        size_train = min_size_3D + (std_train_1d - s_min) * (max_size_3D - min_size_3D) / (s_max - s_min)
        if N_te > 0:
            size_test = min_size_3D + (std_test_1d - s_min) * (max_size_3D - min_size_3D) / (s_max - s_min)
        else:
            size_test = 1

    # ---------- interactive Plotly 3D ----------
    import plotly.graph_objects as go

    image_index_labels = [f"Image_{i + 1}" for i in range(n_frames)]

    traces = [
        go.Scatter3d(
            x=Z3mo[:, 0], y=Z3mo[:, 1], z=Z3mo[:, 2],
            mode='lines',
            line=dict(color='black', width=2),
            name='Trajectory'
        ),
        go.Scatter3d(
            x=Z3mo[:, 0], y=Z3mo[:, 1], z=Z3mo[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                symbol='diamond',
                color=colors,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title='Frame Index (Z)',
                    x=1.02
                ),
            ),
            text=image_index_labels,
            hoverinfo='text',
            name='Z'
        ),
        go.Scatter3d(
            x=Z3tr[:, 0], y=Z3tr[:, 1], z=Z3tr[:, 2],
            mode='markers',
            marker=dict(
                size=size_train,
                color=colors,
                colorscale='Plasma',
                opacity=0.3,
                colorbar=dict(
                    title='Frame Index (recon train)',
                    x=1.10
                ),
            ),
            text=[f"{image_index_labels[i]} std={std_train_1d[i]:.3f}"
                  for i in range(n_frames)],
            hoverinfo='text',
            name='recon train'
        )
    ]

    if reconstructed_Z_test is not None and Z3te is not None and len(Z3te) > 0:
        traces.append(
            go.Scatter3d(
                x=Z3te[:, 0], y=Z3te[:, 1], z=Z3te[:, 2],
                mode='markers',
                marker=dict(
                    size=size_test,
                    color='red',
                    opacity=0.3
                ),
                text=[f"Test_{i + 1} std={std_test_1d[i]:.3f}"
                      for i in range(len(Z3te))],
                hoverinfo='text',
                name='recon test'
            )
        )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=f'3D Latent {str_with_params}',
        scene=dict(
            xaxis_title='Latent 1 (PCA)' if Q > 3 else 'Dim 1',
            yaxis_title='Latent 2 (PCA)' if Q > 3 else 'Dim 2',
            zaxis_title='Latent 3 (PCA)' if Q > 3 else 'Dim 3',
        )
    )

    fig.write_html(paths.output + f"3D Latent {str_with_params}.html")


def _ensure_images(arr, N, img_shape):
    if arr.ndim == 2:
        return arr.reshape((N,) + img_shape)
    return arr


def plot_latent_with_orig_dec_2d_train_test(
        Z_model: np.ndarray,
        Z_train: np.ndarray,
        Z_test,                        # np.ndarray or None
        imgs_train_orig: np.ndarray,  # Y_train, same order as Ztrain
        imgs_test_orig,               # Y_test,  same order as Ztest (or None)
        imgs_train_dec: np.ndarray,  # Y_train_decoded, same order as Ztrain
        imgs_test_dec,               # Y_test_decoded,  same order as Ztest (or None)
        str_with_params: str = "",
        img_shape=(64, 64, 3),
        idx_tr=[50, 100, 150, 200],
        idx_te=[10, 20, 30, 50],
        zoom: float = 0.35,
        figsize=(8, 8),
        save_path_latent: str = None,
        *,
        denormalise,
):
    """Save a 2D (PCA) latent scatter with original/decoded image thumbnails for a few train and test frames."""
    # Normalise Z_test: treat None, 0-d arrays, or empty arrays all as "no test data"
    if Z_test is None or not isinstance(Z_test, np.ndarray) or Z_test.ndim < 2 or Z_test.shape[0] == 0:
        Z_test = None

    N_mo, Q = Z_model.shape
    N_tr, Q2 = Z_train.shape
    assert Q == Q2, "Latent dims of Z_model and Z_train must match"

    has_test = Z_test is not None
    if has_test:
        N_te, Q3 = Z_test.shape
        assert Q == Q3, "Latent dims of Z_test must match Z_model"
    else:
        N_te = 0

    imgs_train_orig = _ensure_images(imgs_train_orig, N_tr, img_shape)
    imgs_train_dec = _ensure_images(imgs_train_dec, N_tr, img_shape)
    if has_test:
        imgs_test_orig = _ensure_images(imgs_test_orig, N_te, img_shape)
        imgs_test_dec = _ensure_images(imgs_test_dec, N_te, img_shape)

    imgs_train_orig = denormalise(imgs_train_orig)
    imgs_train_dec = denormalise(imgs_train_dec)
    if has_test:
        imgs_test_orig = denormalise(imgs_test_orig)
        imgs_test_dec = denormalise(imgs_test_dec)

    # PCA on all latents
    Zall = np.vstack([Z_model, Z_train] + ([Z_test] if has_test else []))
    if Q == 2:
        Z2all = Zall
    else:
        Z2all = PCA(n_components=2).fit_transform(Zall)

    Z2mo = Z2all[:N_mo]
    Z2tr = Z2all[N_mo:N_mo + N_tr]
    Z2te = Z2all[N_mo + N_tr:] if has_test else None

    xmo, ymo = Z2mo[:, 0], Z2mo[:, 1]
    xtr, ytr = Z2tr[:, 0], Z2tr[:, 1]
    if has_test:
        xte, yte = Z2te[:, 0], Z2te[:, 1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xmo, ymo, s=10,
               c=np.linspace(0, 1, len(xmo)),
               cmap="viridis", alpha=0.8, label="Z from model")
    ax.scatter(xtr, ytr, s=10, c="tab:blue", alpha=0.5, label="Train Encoded")
    if has_test:
        ax.scatter(xte, yte, s=10, c="tab:orange", alpha=0.7, label="Test Encoded")

    ax.set_xlabel("Latent 1 (PCA)")
    ax.set_ylabel("Latent 2 (PCA)")
    ax.set_title(f"2D latent with orig+dec images {str_with_params}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    dx = 0.03 * (Z2all[:, 0].max() - Z2all[:, 0].min())
    dy = 0.03 * (Z2all[:, 1].max() - Z2all[:, 1].min())

    thumb_zoom = 0.16  # controls visual size in the figure #Endoslam
    gap_factor = 40.0  # controls distance between orig/dec boxes #Endoslam

    def drawpair(xp, yp, imorig, imdec, color):
        imo = np.clip(imorig, 0, 1)
        imd = np.clip(imdec, 0, 1)

        # Create images with fixed zoom (independent of RESOLUTION)
        oio = OffsetImage(imo, zoom=thumb_zoom)
        oid = OffsetImage(imd, zoom=thumb_zoom)

        # Gap in data coordinates, proportional to zoom, not to image RESOLUTION
        gap = gap_factor * thumb_zoom
        xy_orig = (xp + dx, yp + gap)
        xy_dec = (xp + dx, yp - gap)

        abo = AnnotationBbox(
            oio, xy_orig, xycoords='data',
            frameon=True, pad=0.0,
            bboxprops=dict(edgecolor=color, linewidth=2.5),
        )
        abd = AnnotationBbox(
            oid, xy_dec, xycoords='data',
            frameon=True, pad=0.0,
            bboxprops=dict(edgecolor=color, linewidth=2.5),
        )

        ax.add_artist(abo)
        ax.add_artist(abd)

        midx = 0.5 * (xy_orig[0] + xy_dec[0])
        midy = 0.5 * (xy_orig[1] + xy_dec[1])
        ax.plot([xp, midx], [yp, midy], color=color, linewidth=0.7, alpha=0.9)

    # train: Y_train[i] with Y_train_decoded[i], blue border
    for i in idx_tr:
        if i < N_tr:
            drawpair(xtr[i], ytr[i],
                     imgs_train_orig[i],
                     imgs_train_dec[i],
                     color="tab:blue")

    # test: Y_test[j] with Y_test_decoded[j], orange border (only if test data exists)
    if has_test:
        for j in idx_te:
            if j < N_te:
                drawpair(xte[j], yte[j],
                         imgs_test_orig[j],
                         imgs_test_dec[j],
                         color="tab:orange")

    plt.tight_layout()
    if save_path_latent is not None:
        plt.savefig(save_path_latent, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_pca(Y, str_with_params, *, paths):
    """Save two PCA explained-variance plots (linear and log x axis) of Y."""
    Y = np.asarray(Y)  # make sure it's a NumPy array
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)  # (N,) → (N, 1)
    if Y.ndim > 2:
        Y = Y.reshape(Y.shape[0], -1)  # flatten to (N, D)
    print(f"Plotting PCA on {Y.shape}")
    n_components = min(Y.shape[0], Y.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(Y)  # learn PCA on training

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
    plt.title("PCA explained variance\n" + str_with_params)
    fig.tight_layout()
    fig.savefig(os.path.join(paths.output,
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
    plt.title("PCA explained variance (log x-scale)\n" + str_with_params)
    fig.tight_layout()
    fig.savefig(os.path.join(paths.output,
                             f"PCA explained variance logx {str_with_params}.png"))
    plt.close(fig)
