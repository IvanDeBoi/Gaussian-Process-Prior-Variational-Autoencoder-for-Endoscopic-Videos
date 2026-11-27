import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: config_endo, gpvae_endo
import os
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.cm as cm
import plotly.graph_objects as go


from config_endo import (
    FOLDER_POSES, SUBSTRING_FILTER, KERNEL, LATENT_DIM, MUST_USE_LPIPS, MUST_USE_MASK, MUST_USE_DECODER_GAN,
    MUST_USE_GAP_FOR_TEST, GAP_SIZE
)


def load_c3vdv2_poses(pose_txt_path: str | Path):
    """
    Load C3VDv2 camera poses from pose.txt.

    Format per line (comma-separated):
    r11,r12,r13,0,
    r21,r22,r23,0,
    r31,r32,r33,0,
    tx, ty, tz, 1

    Rotation is camera-to-world (R_c2w) and last row (tx,ty,tz) is the
    camera position in world coordinates.

    We convert to world-to-camera R_w2c for consistency with SfM code.
    """
    pose_txt_path = Path(pose_txt_path)
    lines = pose_txt_path.read_text().strip().splitlines()

    R_list = []
    C_list = []

    for line in lines:
        parts = line.strip().split(",")
        vals = [float(x) for x in parts]

        if len(vals) != 16:
            raise ValueError(f"Expected 16 values per line, got {len(vals)}")

        T = np.array(vals, dtype=np.float64).reshape(4, 4)

        # Original rotation: camera-to-world
        R_c2w = T[:3, :3]
        # Convert to world-to-camera
        R_w2c = R_c2w.T

        # Camera position in world: last row, first 3 entries
        C = T[3, :3]

        R_list.append(R_w2c)
        C_list.append(C)

    C_array = np.stack(C_list, axis=0)  # (N, 3)
    return R_list, C_array


def rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to XYZ (roll-pitch-yaw) Euler angles in radians.
    R is world-to-camera.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll_x = np.arctan2(R[2, 1], R[2, 2])
        pitch_y = np.arctan2(-R[2, 0], sy)
        yaw_z = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll_x = np.arctan2(-R[1, 2], R[1, 1])
        pitch_y = np.arctan2(-R[2, 0], sy)
        yaw_z = 0.0

    return np.array([roll_x, pitch_y, yaw_z], dtype=np.float64)


def save_xyz_euler(out_dir: Path, stem: str, centers: np.ndarray, R_list: list[np.ndarray], str_with_params):
    """
    Save frame number, per-frame x y z and roll/pitch/yaw (XYZ, radians) to a CSV file.
    """
    rows = []
    for i, (R, C) in enumerate(zip(R_list, centers)):
        euler = rotmat_to_euler_xyz(R)  # [rx, ry, rz]
        # Prepend the frame index (i) to the data
        row = np.concatenate([[i], C, euler])  # [frame, x, y, z, rx, ry, rz]
        rows.append(row)

    rows = np.stack(rows, axis=0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # UPDATED: Changed extension from .txt to .csv
    out_path = out_dir / f"{stem}_{str_with_params}_xyz_eulerXYZ.csv"

    # Header with commas
    header = "frame,x,y,z,roll_x,pitch_y,yaw_z"

    # Save using comma delimiter
    np.savetxt(out_path, rows, fmt=["%d"] + ["%.6f"] * 6, delimiter=",", header=header, comments="")

    print(f"Saved poses to CSV: {out_path}")
    return out_path


def plot_matplotlib(out_dir: Path, stem: str, centers: np.ndarray, R_list: list[np.ndarray], str_with_params):
    xs, ys, zs = centers[:, 0], centers[:, 1], centers[:, 2]

    axis_len = 1.0
    step = max(1, 3)

    cmap = plt.colormaps.get_cmap("viridis")
    t_vals = np.linspace(0.0, 1.0, len(centers))
    colors_mpl = cmap(t_vals)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xs, ys, zs, c=colors_mpl, s=15, marker="o", label="camera centers (GT)")
    ax.plot(xs, ys, zs, color="k", linewidth=1, alpha=0.5)

    # Three perpendicular rays meeting at C
    for i in range(0, len(R_list), step):
        R = R_list[i]      # world-to-camera
        C = centers[i]

        ex = R.T @ np.array([1, 0, 0])
        ey = R.T @ np.array([0, 1, 0])
        ez = R.T @ np.array([0, 0, 1])

        px = C + axis_len * ex
        py = C + axis_len * ey
        pz = C + axis_len * ez

        ax.plot([C[0], px[0]], [C[1], px[1]], [C[2], px[2]], "r-", linewidth=1)
        ax.plot([C[0], py[0]], [C[1], py[1]], [C[2], py[2]], "g-", linewidth=1)
        ax.plot([C[0], pz[0]], [C[1], pz[1]], [C[2], pz[2]], "b-", linewidth=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"GT Camera trajectory ({stem})")
    ax.legend()
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{stem} {str_with_params} TRAJ.png"
    plt.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved Matplotlib GT trajectory plot to {png_path}")
    return png_path


def plot_plotly(out_dir: Path, stem: str, centers: np.ndarray, R_list: list[np.ndarray], str_with_params):
    xs, ys, zs = centers[:, 0], centers[:, 1], centers[:, 2]

    axis_len = 1.0
    step = max(1, 3)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+lines",
        name="camera centers (GT)",
        marker=dict(
            size=4,
            color=np.linspace(0.0, 1.0, len(xs)),
            colorscale="Viridis",
            colorbar=dict(title="Frame index"),
        ),
        line=dict(color="rgba(50,50,50,0.5)", width=2),
    ))

    for i in range(0, len(R_list), step):
        R = R_list[i]
        C = centers[i]

        ex = R.T @ np.array([1, 0, 0])
        ey = R.T @ np.array([0, 1, 0])
        ez = R.T @ np.array([0, 0, 1])

        px = C + axis_len * ex
        py = C + axis_len * ey
        pz = C + axis_len * ez

        fig.add_trace(go.Scatter3d(
            x=[C[0], px[0]],
            y=[C[1], px[1]],
            z=[C[2], px[2]],
            mode="lines",
            line=dict(color="red", width=2),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[C[0], py[0]],
            y=[C[1], py[1]],
            z=[C[2], py[2]],
            mode="lines",
            line=dict(color="green", width=2),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[C[0], pz[0]],
            y=[C[1], pz[1]],
            z=[C[2], pz[2]],
            mode="lines",
            line=dict(color="blue", width=2),
            showlegend=False,
        ))

    fig.update_layout(
        title=f"GT Camera trajectory ({stem})",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"{stem} {str_with_params} TRAJ.html"
    fig.write_html(str(html_path))
    print(f"Saved Plotly GT trajectory to {html_path}")
    return html_path


if __name__ == "__main__":
    pose_file = r"C:\Endo\C3VDv2\c1_ascending_t1_v1/pose.txt"

    str_with_params = "GPVAE_HPA c1_ascending_t1_v1 336 RBF EndoVAEEnc EndoVAEDec 64D L=False M_True G_False GAP_False_15 TF_20"
    str_with_params = "GT c1_ascending_t1_v1 336"

    # Get the directory where this script is located
    script_dir = Path(__file__).resolve().parent

    # Move one level up and join with your FOLDER_POSES name
    # If FOLDER_POSES is an absolute path, this will override;
    # if it's a string like "poses", it will create it one level up.
    out_dir = script_dir.parent / Path(FOLDER_POSES).name

    print(f"Output directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    pose_file = Path(pose_file)
    stem = pose_file.stem

    R_list, centers = load_c3vdv2_poses(pose_file)


    save_xyz_euler(out_dir, stem, centers, R_list, str_with_params)
    plot_matplotlib(out_dir, stem, centers, R_list, str_with_params)
    plot_plotly(out_dir, stem, centers, R_list, str_with_params)
