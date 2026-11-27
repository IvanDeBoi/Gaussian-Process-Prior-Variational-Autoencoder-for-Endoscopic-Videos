import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: config_endo, gpvae_endo
# --- Imports ---
import matplotlib.pyplot as plt

from datetime import datetime
import time
import random

from config_endo import (
    SEED,
    FOLDER_DATA,
    FOLDER_OUTPUT,
    MUST_PRESERVE_COLOR,
    RESOLUTION,
    SUBSTRING_FILTER,
)

import os
import cv2
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from skimage.metrics import structural_similarity  # SSIM


def frames_from_Y(Y, resolution, must_preserve_color):
    """
    Convert flattened Y (N, D) back to image tensors (N, H, W, C) in [0,1].
    """
    print("Reconstructing frames from Y")
    N, D = Y.shape
    H = W = resolution
    C = 3 if must_preserve_color else 1
    imgs = Y.reshape(N, H, W, C)
    imgs = imgs.astype(np.float64)
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
    print(f"Frames reconstructed: {imgs.shape}")
    return imgs


def ssim_distance_matrix(imgs):
    """
    SSIM-based distance: d(i,j) = 1 - SSIM(img_i,img_j).
    """
    print("Computing SSIM distance matrix")
    N = imgs.shape[0]
    dmat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        if i % 10 == 0:
            print(f"  SSIM row {i+1}/{N}")
        for j in range(i, N):
            if imgs.shape[-1] == 1:
                img_i = imgs[i, :, :, 0]
                img_j = imgs[j, :, :, 0]
                s = structural_similarity(img_i, img_j, data_range=1.0)
            else:
                s = structural_similarity(
                    imgs[i], imgs[j],
                    channel_axis=-1,
                    data_range=1.0
                )
            d = 1.0 - float(s)
            dmat[i, j] = dmat[j, i] = d
    print("Finished SSIM distance matrix")
    return dmat


def ncc_distance_matrix(imgs):
    """
    Normalized cross-correlation distance: d = 1 - NCC.
    """
    print("Computing NCC distance matrix")
    N = imgs.shape[0]
    flat = imgs.reshape(N, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True) + 1e-8
    flat_norm = flat / std
    L = flat_norm.shape[1]
    ncc = (flat_norm @ flat_norm.T) / L
    ncc = np.clip(ncc, -1.0, 1.0)
    dmat = 1.0 - ncc
    print("Finished NCC distance matrix")
    return dmat.astype(np.float32)


def mi_distance_matrix(imgs, bins=32):
    """
    Very rough mutual-information based distance: d = 1 - MI_norm.
    """
    print("Computing MI distance matrix")
    N = imgs.shape[0]
    if imgs.shape[-1] > 1:
        gray = imgs.mean(axis=-1)
    else:
        gray = imgs[:, :, :, 0]
    gray_flat = gray.reshape(N, -1)
    dmat = np.zeros((N, N), dtype=np.float32)

    def entropy(p):
        p = p + 1e-12
        return -np.sum(p * np.log(p))

    for i in range(N):
        if i % 10 == 0:
            print(f"  MI row {i+1}/{N}")
        for j in range(i, N):
            xi = gray_flat[i]
            xj = gray_flat[j]
            joint_hist, _, _ = np.histogram2d(
                xi, xj, bins=bins, range=[[0, 1], [0, 1]]
            )
            joint_prob = joint_hist / joint_hist.sum()
            px = joint_prob.sum(axis=1)
            py = joint_prob.sum(axis=0)
            Hx = entropy(px)
            Hy = entropy(py)
            Hxy = entropy(joint_prob.ravel())
            mi = Hx + Hy - Hxy
            Hmax = np.log(bins)
            mi_norm = mi / (Hmax + 1e-8)
            mi_norm = np.clip(mi_norm, 0.0, 1.0)
            d = 1.0 - float(mi_norm)
            dmat[i, j] = dmat[j, i] = d
    print("Finished MI distance matrix")
    return dmat

def optical_flow_magnitude(img1, img2):
    """
    Compute mean optical flow magnitude between two frames in [0,1].
    Uses Farnebäck dense optical flow on grayscale.
    """
    if img1.shape[-1] == 3:
        g1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        g1 = (img1[:, :, 0] * 255).astype(np.uint8)
        g2 = (img2[:, :, 0] * 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        g1, g2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))

def visualize_optical_flow(img1, img2, title_prefix="OF", save_name=None):
    """
    Visualize dense optical flow between two frames in [0,1].

    2x2 layout:
      (1,1) frame 1
      (1,2) frame 2
      (2,1) flow vectors (quiver) on frame 1
      (2,2) flow magnitude heatmap
    """
    print(f"Visualizing optical flow: {title_prefix}")

    # Convert to uint8 grayscale for OpenCV
    if img1.shape[-1] == 3:
        g1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        g1 = (img1[:, :, 0] * 255).astype(np.uint8)
        g2 = (img2[:, :, 0] * 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        g1, g2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)

    H, W = g1.shape
    step = max(1, min(H, W) // 30)
    y, x = np.mgrid[0:H:step, 0:W:step]

    plt.figure(figsize=(10, 8))

    # (1,1) Frame 1
    plt.subplot(2, 2, 1)
    if img1.shape[-1] == 3:
        plt.imshow(img1)
    else:
        plt.imshow(img1[:, :, 0], cmap="gray")
    plt.title(f"{title_prefix} frame 1")
    plt.axis("off")

    # (1,2) Frame 2
    plt.subplot(2, 2, 2)
    if img2.shape[-1] == 3:
        plt.imshow(img2)
    else:
        plt.imshow(img2[:, :, 0], cmap="gray")
    plt.title(f"{title_prefix} frame 2")
    plt.axis("off")

    # (2,1) Quiver on frame 1
    plt.subplot(2, 2, 3)
    plt.imshow(g1, cmap="gray")
    plt.quiver(
        x, y,
        fx[::step, ::step],
        fy[::step, ::step],
        color="r",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.002,
    )
    plt.title(f"{title_prefix} flow vectors")
    plt.axis("off")

    # (2,2) Magnitude heatmap
    plt.subplot(2, 2, 4)
    plt.imshow(mag, cmap="magma")
    plt.colorbar(label="Flow magnitude")
    plt.title(f"{title_prefix} flow magnitude")
    plt.axis("off")

    plt.tight_layout()
    if save_name is not None:
        out_path = os.path.join(FOLDER_OUTPUT, save_name)
        print(f"Saving optical flow visualization to {out_path}")
        plt.savefig(out_path, dpi=150)
    plt.close()




def of_distance_matrix(imgs):
    """
    Optical-flow-based distance: mean flow magnitude between frames.
    Symmetric by averaging forward/backward magnitude.
    """
    print("Computing optical-flow distance matrix")
    N = imgs.shape[0]
    dmat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        if i % 5 == 0:
            print(f"  OF row {i+1}/{N}")
        for j in range(i + 1, N):
            d_ij = optical_flow_magnitude(imgs[i], imgs[j])
            d_ji = optical_flow_magnitude(imgs[j], imgs[i])
            d = 0.5 * (d_ij + d_ji)
            dmat[i, j] = dmat[j, i] = d
    print("Finished optical-flow distance matrix")
    return dmat

def pick_pairs_and_visualize(dmat, imgs, substring_filter, resolution):
    N = dmat.shape[0]

    # Flatten upper triangle (exclude diagonal)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_vals = dmat[i_idx, j_idx]

    # Sort indices by distance
    order = np.argsort(d_vals)
    small_pairs = list(zip(i_idx[order], j_idx[order]))
    large_pairs = list(zip(i_idx[order[::-1]], j_idx[order[::-1]]))

    # Pick up to 5 smallest‑d pairs
    k_small = min(5, len(small_pairs))
    small_sample = random.sample(small_pairs[:max(10, k_small)], k_small)

    # Pick up to 3 largest‑d pairs
    k_large = min(3, len(large_pairs))
    large_sample = random.sample(large_pairs[:max(10, k_large)], k_large)

    print("Small‑d pairs:", small_sample)
    print("Large‑d pairs:", large_sample)

    # Visualize small‑d pairs
    for idx, (i, j) in enumerate(small_sample):
        title = f"OF small d {substring_filter} {resolution} {i}-{j}"
        save = f"of small d {substring_filter} {resolution} {idx}_{i}-{j}.png"
        visualize_optical_flow(imgs[i], imgs[j], title_prefix=title, save_name=save)

    # Visualize large‑d pairs
    for idx, (i, j) in enumerate(large_sample):
        title = f"OF large d {substring_filter} {resolution} {i}-{j}"
        save = f"of large d {substring_filter} {resolution} {idx}_{i}-{j}.png"
        visualize_optical_flow(imgs[i], imgs[j], title_prefix=title, save_name=save)



def rbf_similarity_from_distance(dmat):
    print("Converting distance matrix to RBF similarity")
    sigma = np.median(dmat) + 1e-8
    print(f"  sigma (median distance) = {sigma:.6f}")
    return np.exp(-(dmat ** 2) / (2 * sigma ** 2))


def plot_matrix(mat, title, fname, cbar_label, limit_colorbar=False):
    print(f"Plotting {title}")
    plt.figure(figsize=(6, 5))
    if limit_colorbar:
        im = plt.imshow(mat, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    else:
        im = plt.imshow(mat, cmap="viridis", aspect="auto")
    plt.colorbar(im, label=cbar_label)
    plt.xlabel("Frame index")
    plt.ylabel("Frame index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER_OUTPUT, fname), dpi=150)
    plt.close()


if __name__ == "__main__":

    start_time_total = time.time()
    print(f"Start time: {datetime.fromtimestamp(start_time_total).strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n***************************************************************************************************\n")

    for resolution in RESOLUTION:
        for substring_filter in SUBSTRING_FILTER:
            start_time = time.time()
            print(
                f"\n=== New run: {substring_filter}, resolution {resolution}, seed {SEED} ==="
            )

            str_with_params = f"{substring_filter} {resolution}"
            npz_path = os.path.join(FOLDER_DATA, f"{str_with_params} XY.npz")
            print(f"Loading data from {npz_path}")
            data = np.load(npz_path)
            x = data["x"]
            Y = data["Y"]
            print(f"Y shape: {Y.shape}, x shape: {x.shape}")

            N, D = Y.shape
            base_name = f"{substring_filter} {resolution}"

            imgs = frames_from_Y(Y, resolution, MUST_PRESERVE_COLOR)

            # ---------------- Euclidean ----------------
            dist_path_euc = os.path.join(FOLDER_DATA, f"{base_name} distance euclidean.npy")
            sim_path_euc  = os.path.join(FOLDER_DATA, f"{base_name} similarity euclidean.npy")

            if os.path.exists(dist_path_euc) and os.path.exists(sim_path_euc):
                print(f"Loading Euclidean distance from {dist_path_euc}")
                print(f"Loading Euclidean similarity from {sim_path_euc}")
                diff_euc = np.load(dist_path_euc)
                sim_euc  = np.load(sim_path_euc)
            else:
                print("Computing Euclidean distance matrix")
                diff_euc = np.zeros((N, N), dtype=np.float32)
                for i in range(N):
                    if i % 10 == 0:
                        print(f"  Euclidean row {i+1}/{N}")
                    Yi = Y[i][None, :]
                    diff = Y - Yi
                    diff_euc[i] = np.linalg.norm(diff, axis=1)
                sim_euc = rbf_similarity_from_distance(diff_euc)
                np.save(dist_path_euc, diff_euc)
                np.save(sim_path_euc,  sim_euc)
                print(f"Saved Euclidean distance to {dist_path_euc}")
                print(f"Saved Euclidean similarity to {sim_path_euc}")

            plot_matrix(
                diff_euc,
                f"Euclidean distance matrix {str_with_params}",
                f"distance euclidean {substring_filter} {resolution}.png",
                "Euclidean distance",
            )
            plot_matrix(
                sim_euc,
                f"Euclidean similarity matrix {str_with_params}",
                f"similarity euclidean {substring_filter} {resolution}.png",
                "Similarity",
                limit_colorbar=True,
            )

            # ---------------- SSIM ----------------
            ### SSIM (Structural Similarity Index Measure) is an image similarity metric
            # that compares two images by how well they match in luminance, contrast, and local structure,
            # rather than by raw pixel differences.
            dist_path_ssim = os.path.join(FOLDER_DATA, f"{base_name} distance ssim.npy")
            sim_path_ssim  = os.path.join(FOLDER_DATA, f"{base_name} similarity ssim.npy")

            if os.path.exists(dist_path_ssim) and os.path.exists(sim_path_ssim):
                print(f"Loading SSIM distance from {dist_path_ssim}")
                print(f"Loading SSIM similarity from {sim_path_ssim}")
                diff_ssim = np.load(dist_path_ssim)
                sim_ssim  = np.load(sim_path_ssim)
            else:
                diff_ssim = ssim_distance_matrix(imgs)
                sim_ssim  = rbf_similarity_from_distance(diff_ssim)
                np.save(dist_path_ssim, diff_ssim)
                np.save(sim_path_ssim,  sim_ssim)
                print(f"Saved SSIM distance to {dist_path_ssim}")
                print(f"Saved SSIM similarity to {sim_path_ssim}")

            plot_matrix(
                diff_ssim,
                f"SSIM distance matrix {str_with_params}",
                f"distance ssim {substring_filter} {resolution}.png",
                "1 - SSIM",
            )
            plot_matrix(
                sim_ssim,
                f"SSIM similarity matrix {str_with_params}",
                f"similarity ssim {substring_filter} {resolution}.png",
                "Similarity",
                limit_colorbar=True,
            )

            # ---------------- NCC ----------------
            # NCC (normalized cross‑correlation) is an image similarity measure that quantifies how
            # linearly related the pixel intensities of two images are
            # after removing mean and scaling by variance.
            dist_path_ncc = os.path.join(FOLDER_DATA, f"{base_name} distance ncc.npy")
            sim_path_ncc  = os.path.join(FOLDER_DATA, f"{base_name} similarity ncc.npy")

            if os.path.exists(dist_path_ncc) and os.path.exists(sim_path_ncc):
                print(f"Loading NCC distance from {dist_path_ncc}")
                print(f"Loading NCC similarity from {sim_path_ncc}")
                diff_ncc = np.load(dist_path_ncc)
                sim_ncc  = np.load(sim_path_ncc)
            else:
                diff_ncc = ncc_distance_matrix(imgs)
                sim_ncc  = rbf_similarity_from_distance(diff_ncc)
                np.save(dist_path_ncc, diff_ncc)
                np.save(sim_path_ncc,  sim_ncc)
                print(f"Saved NCC distance to {dist_path_ncc}")
                print(f"Saved NCC similarity to {sim_path_ncc}")

            plot_matrix(
                diff_ncc,
                f"NCC distance matrix {str_with_params}",
                f"distance ncc {substring_filter} {resolution}.png",
                "1 - NCC",
            )
            plot_matrix(
                sim_ncc,
                f"NCC similarity matrix {str_with_params}",
                f"similarity ncc {substring_filter} {resolution}.png",
                "Similarity",
                limit_colorbar=True,
            )

            # ---------------- MI ----------------
            # MI here is mutual information,
            # an information‑theoretic measure of how much knowing one image reduces uncertainty about the other.
            dist_path_mi = os.path.join(FOLDER_DATA, f"{base_name} distance mi.npy")
            sim_path_mi  = os.path.join(FOLDER_DATA, f"{base_name} similarity mi.npy")

            if os.path.exists(dist_path_mi) and os.path.exists(sim_path_mi):
                print(f"Loading MI distance from {dist_path_mi}")
                print(f"Loading MI similarity from {sim_path_mi}")
                diff_mi = np.load(dist_path_mi)
                sim_mi  = np.load(sim_path_mi)
            else:
                diff_mi = mi_distance_matrix(imgs)
                sim_mi  = rbf_similarity_from_distance(diff_mi)
                np.save(dist_path_mi, diff_mi)
                np.save(sim_path_mi,  sim_mi)
                print(f"Saved MI distance to {dist_path_mi}")
                print(f"Saved MI similarity to {sim_path_mi}")

            plot_matrix(
                diff_mi,
                f"MI distance matrix {str_with_params}",
                f"distance mi {substring_filter} {resolution}.png",
                "1 - MI norm",
            )
            plot_matrix(
                sim_mi,
                f"MI similarity matrix {str_with_params}",
                f"similarity mi {substring_filter} {resolution}.png",
                "Similarity",
                limit_colorbar=True,
            )

            # ---------------- Optical Flow ----------------
            dist_path_of = os.path.join(FOLDER_DATA, f"{base_name} distance of.npy")
            sim_path_of = os.path.join(FOLDER_DATA, f"{base_name} similarity of.npy")

            if os.path.exists(dist_path_of) and os.path.exists(sim_path_of):
                print(f"Loading OF distance from {dist_path_of}")
                print(f"Loading OF similarity from {sim_path_of}")
                diff_of = np.load(dist_path_of)
                sim_of = np.load(sim_path_of)
            else:
                diff_of = of_distance_matrix(imgs)
                sim_of = rbf_similarity_from_distance(diff_of)
                np.save(dist_path_of, diff_of)
                np.save(sim_path_of, sim_of)
                print(f"Saved OF distance to {dist_path_of}")
                print(f"Saved OF similarity to {sim_path_of}")

            pick_pairs_and_visualize(diff_of, imgs, substring_filter, resolution)

            plot_matrix(
                diff_of,
                f"OF distance matrix {str_with_params}",
                f"distance of {substring_filter} {resolution}.png",
                "Mean flow magnitude",
            )
            plot_matrix(
                sim_of,
                f"OF similarity matrix {str_with_params}",
                f"similarity of {substring_filter} {resolution}.png",
                "Similarity",
                limit_colorbar=True,
            )

            end_time = time.time()
            print(f"Time this run: {(end_time - start_time):.3f}s")
            print("\n***************************************************************************************************\n")

    end_time_total = time.time()
    print(f"\nTotal time: {(end_time_total - start_time_total):.3f}s")
