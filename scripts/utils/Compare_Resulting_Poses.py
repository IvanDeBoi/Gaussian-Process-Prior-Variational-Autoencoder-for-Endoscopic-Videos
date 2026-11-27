import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import plotly.graph_objects as go
from pathlib import Path


def umeyama_alignment(source, target, estimate_scale=True):
    """
    Standard Procrustes/Umeyama algorithm to find the optimal rigid
    transformation (s, R, t) between two point sets.
    """
    n, m = source.shape
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    s_centered = source - mu_s
    t_centered = target - mu_t

    sigma2_s = np.mean(np.sum(s_centered ** 2, axis=1))
    covariance = (t_centered.T @ s_centered) / n

    U, D, Vt = np.linalg.svd(covariance)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1

    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / sigma2_s if estimate_scale else 1.0
    t = mu_t - scale * (R @ mu_s)

    aligned = (scale * R @ source.T).T + t
    return aligned


def simple_dtw(s1, s2):
    """
    Calculates the Dynamic Time Warping distance between two 3D trajectories.
    """
    n1, n2 = len(s1), len(s2)
    dtw_matrix = np.full((n1 + 1, n2 + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            cost = np.linalg.norm(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                          dtw_matrix[i, j - 1],
                                          dtw_matrix[i - 1, j - 1])
    return dtw_matrix[n1, n2]


# --- Configuration ---
gt_file = "../Poses/pose_GT c1_ascending_t1_v1 336_xyz_eulerXYZ.csv"
other_files = [
    "../Poses/c1_ascending_t1_v1 VAE EndoVAEEnc EndoVAEDec RBF 64D L_False M_True G_False GAP_False_15 TF_20 poses_xyz_eulerXYZ.csv",
    "../Poses/c1_ascending_t1_v1 GPVAE_HPA EndoVAEEnc EndoVAEDec RBF 64D L_False M_True G_False GAP_False_15 TF_20 poses_xyz_eulerXYZ.csv",
    "../Poses/c1_ascending_t1_v1 Original poses_xyz_eulerXYZ.csv"
]
str_with_params = "c1_ascending_t1_v1 336 RBF EndoVAEEnc EndoVAEDec 64 L_False M_True G_False GAP_False_15 TF_20"

pics_dir = Path("../Pics")
pics_dir.mkdir(parents=True, exist_ok=True)

# Load Ground Truth
df_gt = pd.read_csv(gt_file)
gt_pts = df_gt[['x', 'y', 'z']].values

# --- Initialize Figures ---
n_models = len(other_files)
total_cols = n_models + 1  # Column 1 is GT, followed by N models

# 1. Global Comparison Figure
fig_global = plt.figure(figsize=(12, 9))
ax_global = fig_global.add_subplot(111, projection='3d')
ax_global.plot(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], 'k--', label="Ground Truth", linewidth=2)

# 2. Raw Strip Figure (1 x 4)
fig_strip_raw = plt.figure(figsize=(6 * total_cols, 6))
ax_gt_raw = fig_strip_raw.add_subplot(1, total_cols, 1, projection='3d')
ax_gt_raw.plot(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], 'k--', linewidth=2)
ax_gt_raw.set_title("GROUND TRUTH", pad=20)
ax_gt_raw.set_xlabel('X', labelpad=10);
ax_gt_raw.set_ylabel('Y', labelpad=10);
ax_gt_raw.set_zlabel('Z', labelpad=10)

# 3. Aligned Strip Figure (1 x 4)
fig_strip_aligned = plt.figure(figsize=(6 * total_cols, 6))
ax_gt_aligned = fig_strip_aligned.add_subplot(1, total_cols, 1, projection='3d')
ax_gt_aligned.plot(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], 'k--', linewidth=2)
ax_gt_aligned.set_title("GROUND TRUTH", pad=20)
ax_gt_aligned.set_xlabel('X', labelpad=10);
ax_gt_aligned.set_ylabel('Y', labelpad=10);
ax_gt_aligned.set_zlabel('Z', labelpad=10)

# 4. Interactive Plotly
fig_plotly = go.Figure()
fig_plotly.add_trace(go.Scatter3d(x=gt_pts[:, 0], y=gt_pts[:, 1], z=gt_pts[:, 2],
                                  mode='lines', line=dict(color='black', width=4, dash='dash'), name="GT"))

# --- Process Models ---
print("Generating visualizations with increased margins...")

for i, f in enumerate(other_files):
    if not os.path.exists(f): continue
    df = pd.read_csv(f);
    pts = df[['x', 'y', 'z']].values

    # Alignment & DTW
    min_len = min(len(pts), len(gt_pts))
    aligned_pts = umeyama_alignment(pts[:min_len], gt_pts[:min_len])
    dist = simple_dtw(aligned_pts, gt_pts[:min_len])

    label = "VAE" if "VAE" in f else "GPVAE_HPA" if "GPVAE_HPA" in f else "Original"

    # Add to RAW strip (Individual model only)
    ax_raw = fig_strip_raw.add_subplot(1, total_cols, i + 2, projection='3d')
    ax_raw.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='C' + str(i), label=label)
    ax_raw.set_title(f"{label}\n(Raw Data)", pad=20)
    ax_raw.set_xlabel('X', labelpad=10);
    ax_raw.set_ylabel('Y', labelpad=10);
    ax_raw.set_zlabel('Z', labelpad=10)

    # Add to ALIGNED strip (Individual model only)
    ax_aligned = fig_strip_aligned.add_subplot(1, total_cols, i + 2, projection='3d')
    ax_aligned.plot(aligned_pts[:, 0], aligned_pts[:, 1], aligned_pts[:, 2], color='C' + str(i), label=label)
    ax_aligned.set_title(f"{label} (Aligned)\nDTW: {dist:.2f}", pad=20)
    ax_aligned.set_xlabel('X', labelpad=10);
    ax_aligned.set_ylabel('Y', labelpad=10);
    ax_aligned.set_zlabel('Z', labelpad=10)

    # Global Plots
    ax_global.plot(aligned_pts[:, 0], aligned_pts[:, 1], aligned_pts[:, 2], label=label)
    fig_plotly.add_trace(
        go.Scatter3d(x=aligned_pts[:, 0], y=aligned_pts[:, 1], z=aligned_pts[:, 2], mode='lines', name=label))

# --- Save Files with Layout Optimization ---
ax_global.set_title("3D Trajectories Aligned to GT", pad=20)
ax_global.set_xlabel('X', labelpad=15);
ax_global.set_ylabel('Y', labelpad=15);
ax_global.set_zlabel('Z', labelpad=15)
ax_global.legend()
fig_global.tight_layout(pad=4.0)
fig_global.savefig(pics_dir / f"aligned_trajectories_plot {str_with_params}.png", dpi=300)

fig_strip_raw.tight_layout(pad=5.0)
fig_strip_raw.savefig(pics_dir / f"trajectories_strip_RAW {str_with_params}.png", dpi=300)

fig_strip_aligned.tight_layout(pad=5.0)
fig_strip_aligned.savefig(pics_dir / f"trajectories_strip_ALIGNED {str_with_params}.png", dpi=300)

fig_plotly.update_layout(title="Interactive Aligned Trajectories", margin=dict(l=10, r=10, b=10, t=50))
fig_plotly.write_html(str(pics_dir / f"aligned_trajectories_interactive {str_with_params}.html"))

plt.close('all')