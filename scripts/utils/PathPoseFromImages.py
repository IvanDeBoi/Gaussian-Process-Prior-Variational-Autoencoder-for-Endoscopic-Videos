import os
import glob
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go
import matplotlib.cm as cm


def load_images_from_folder(folder, exts=("*.png", "*.jpg", "*.jpeg")):
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
        paths.extend(glob.glob(os.path.join(folder, e.upper())))
    paths = sorted(paths)
    if not paths:
        raise RuntimeError(f"No images found in {folder}")
    print(f"Found {len(paths)} images")
    return paths


# ---------- INTRINSICS OPTIONS ----------

def guess_intrinsics_from_image_size(img_shape, focal_ratio=1.2):
    h, w = img_shape[:2]
    f = focal_ratio * max(h, w)
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    print("Guessed intrinsics (K_not_known) =\n", K)
    return K


def load_intrinsics_from_folder(calib_folder):
    calib_folder = Path(calib_folder)
    yaml_path = calib_folder / "calibration.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"calibration.yaml not found in {calib_folder}")

    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()

    if K is None or dist is None:
        raise RuntimeError(f"Could not read camera_matrix or distortion_coefficients from {yaml_path}")

    print("Loaded intrinsics (K_known) from", yaml_path)
    print("K =\n", K)
    print("dist =", dist.ravel())
    return K.astype(np.float64), dist.astype(np.float64)


# ---------- CORE SfM UTILS ----------

def detect_and_match_features(img1, img2):
    orb = cv2.ORB_create(4000)
    kps1, des1 = orb.detectAndCompute(img1, None)
    kps2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return [], [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return [], [], []

    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[: int(len(matches) * 0.5)]

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good])

    return pts1, pts2, good


def estimate_relative_pose(pts1, pts2, K):
    if len(pts1) < 8:
        raise RuntimeError("Not enough matches to estimate pose")

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )[0:2]

    if E is None:
        raise RuntimeError("Essential matrix estimation failed")

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t


def compose_poses(R_prev, t_prev, R_rel, t_rel):
    R_curr = R_rel @ R_prev
    t_curr = R_rel @ t_prev + t_rel
    return R_curr, t_curr


def poses_to_camera_centers(poses):
    centers = []
    for R, t in poses:
        C = -R.T @ t
        centers.append(C.reshape(3))
    return np.array(centers)


def rotmat_to_euler_xyz(R):
    """
    Convert 3x3 rotation matrix to roll/pitch/yaw (XYZ) in radians.
    Interprets R as world-to-camera here.
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


def run_sfm_and_save(img_paths, K, out_dir, name_tag, suffix):
    """
    Run sequential SfM on an image sequence with given K, then save
    trajectory + plots + xyz/euler using a suffix: 'K_known' or 'K_not_known'.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_img = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        raise RuntimeError(f"Failed to read {img_paths[0]}")

    R_curr = np.eye(3, dtype=np.float64)
    t_curr = np.zeros((3, 1), dtype=np.float64)
    poses = [(R_curr.copy(), t_curr.copy())]

    prev_img = first_img

    for idx in range(1, len(img_paths)):
        curr_img = cv2.imread(img_paths[idx], cv2.IMREAD_GRAYSCALE)
        if curr_img is None:
            print(f"Skipping unreadable image: {img_paths[idx]}")
            continue

        print(f"\n[{suffix}] Processing pair: {os.path.basename(img_paths[idx-1])} "
              f"-> {os.path.basename(img_paths[idx])}")

        pts1, pts2, matches = detect_and_match_features(prev_img, curr_img)
        print(f"[{suffix}] Matches: {len(matches)}")

        if len(matches) < 8:
            print(f"[{suffix}] Not enough matches, skipping frame")
            continue

        try:
            R_rel, t_rel = estimate_relative_pose(pts1, pts2, K)
        except RuntimeError as e:
            print(f"[{suffix}] Pose estimation failed:", e)
            continue

        R_curr, t_curr = compose_poses(R_curr, t_curr, R_rel, t_rel)
        poses.append((R_curr.copy(), t_curr.copy()))

        prev_img = curr_img

    save_and_plot_trajectory(poses, out_dir, f"{name_tag}_{suffix}")


def save_and_plot_trajectory(poses, out_dir, base_name):
    centers = poses_to_camera_centers(poses)
    xs, ys, zs = centers[:, 0], centers[:, 1], centers[:, 2]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Save CSV + NPZ -----
    traj_csv = out_dir / f"{base_name}_TRAJ.csv"
    poses_npz = out_dir / f"{base_name}_POSES.npz"

    np.savetxt(
        traj_csv,
        centers,
        delimiter=",",
        header="X,Y,Z",
        comments="",
    )

    Rs = np.stack([R for R, _ in poses], axis=0)
    ts = np.stack([t for _, t in poses], axis=0)
    np.savez(poses_npz, R=Rs, t=ts)

    print(f"Saved camera centers to {traj_csv}")
    print(f"Saved poses (R,t) to {poses_npz}")

    # ----- Save xyz + Euler (roll, pitch, yaw) -----
    rows = []
    for R, C in zip(Rs, centers):
        euler = rotmat_to_euler_xyz(R)  # [rx, ry, rz]
        rows.append(np.concatenate([C, euler]))
    rows = np.stack(rows, axis=0)

    xyz_euler_path = out_dir / f"{base_name}_TRAJ_xyz_eulerXYZ.txt"
    header = "x y z roll_x pitch_y yaw_z (radians)"
    np.savetxt(xyz_euler_path, rows, fmt="%.6f", header=header, comments="")
    print(f"Saved xyz + euler to {xyz_euler_path}")

    # Common helpers
    axis_len = float(np.linalg.norm(centers[-1] - centers[0]) * 0.05) if len(centers) > 1 else 0.1
    step = 3 if len(poses) >= 3 else 1  # 1 out of 3

    # Viridis colors along the trajectory
    cmap = cm.get_cmap("viridis")
    t_vals = np.linspace(0.0, 1.0, len(centers))
    colors_mpl = cmap(t_vals)  # RGBA

    # ----- Matplotlib PNG -----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xs, ys, zs, c=colors_mpl, s=15, marker='o', label="camera centers")
    ax.plot(xs, ys, zs, color="k", linewidth=1, alpha=0.5)

    for i_idx, (R, t) in enumerate(poses[::step]):
        i = i_idx * step
        C = centers[i]
        x_axis = C + R.T @ (axis_len * np.array([1, 0, 0]))
        y_axis = C + R.T @ (axis_len * np.array([0, 1, 0]))
        z_axis = C + R.T @ (axis_len * np.array([0, 0, 1]))
        ax.plot([C[0], x_axis[0]], [C[1], x_axis[1]], [C[2], x_axis[2]], "r-")
        ax.plot([C[0], y_axis[0]], [C[1], y_axis[1]], [C[2], y_axis[2]], "g-")
        ax.plot([C[0], z_axis[0]], [C[1], z_axis[1]], [C[2], z_axis[2]], "b-")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Camera trajectory ({base_name})")
    ax.legend()
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    png_path = out_dir / f"{base_name}_TRAJ.png"
    plt.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved Matplotlib trajectory plot to {png_path}")

    # ----- Plotly HTML -----
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+lines",
        name="camera centers",
        marker=dict(
            size=4,
            color=np.linspace(0.0, 1.0, len(xs)),
            colorscale="Viridis",
            colorbar=dict(title="Frame index"),
        ),
        line=dict(color="rgba(50,50,50,0.5)", width=2),
    ))

    for i_idx, (R, t) in enumerate(poses[::step]):
        i = i_idx * step
        C = centers[i]
        x_axis = C + R.T @ (axis_len * np.array([1, 0, 0]))
        y_axis = C + R.T @ (axis_len * np.array([0, 1, 0]))
        z_axis = C + R.T @ (axis_len * np.array([0, 0, 1]))

        fig_p.add_trace(go.Scatter3d(
            x=[C[0], x_axis[0]],
            y=[C[1], x_axis[1]],
            z=[C[2], x_axis[2]],
            mode="lines",
            line=dict(color="red", width=3),
            showlegend=False,
        ))
        fig_p.add_trace(go.Scatter3d(
            x=[C[0], y_axis[0]],
            y=[C[1], y_axis[1]],
            z=[C[2], y_axis[2]],
            mode="lines",
            line=dict(color="green", width=3),
            showlegend=False,
        ))
        fig_p.add_trace(go.Scatter3d(
            x=[C[0], z_axis[0]],
            y=[C[1], z_axis[1]],
            z=[C[2], z_axis[2]],
            mode="lines",
            line=dict(color="blue", width=3),
            showlegend=False,
        ))

    fig_p.update_layout(
        title=f"Camera trajectory ({base_name})",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    html_path = out_dir / f"{base_name}_TRAJ.html"
    fig_p.write_html(str(html_path))
    print(f"Saved Plotly trajectory to {html_path}")


# -------- main code: TWO OPTIONS RUN SEQUENTIALLY --------
if __name__ == "__main__":
    IMAGE_FOLDER = "../Data/C3VDv2 c1 asc t1 v1 full res/undistorted"
    # IMAGE_FOLDER = "../Data/C3VDv2 c1 asc t1 v2 full res/undistorted"

    # IMAGE_FOLDER = "../Output_Reconstructed_Frames/C3VDv2C1ascv2t1_336_GPVAE_HPA_Shi_time_encoded_new/Undistorted"

    folder_path = Path(IMAGE_FOLDER)
    folder_name = folder_path.name
    name_tag = folder_name.replace(" ", "_")

    CALIB_FOLDER = "../Data/C3VDv2 CB/calibration_results"  # set this to your calib folder

    img_paths = load_images_from_folder(IMAGE_FOLDER)

    OUTPUT_BASE = folder_path / f"SfM_{name_tag}"
    OUTPUT_BASE.mkdir(exist_ok=True)

    # OPTION 1: K_not_known
    first_img = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        raise RuntimeError(f"Failed to read {img_paths[0]}")

    K_guess = guess_intrinsics_from_image_size(first_img.shape, focal_ratio=1.2)
    out_dir_guess = OUTPUT_BASE / "K_not_known"
    run_sfm_and_save(img_paths, K_guess, out_dir_guess, name_tag, suffix="K_not_known")

    # OPTION 2: K_known
    K_known, dist_known = load_intrinsics_from_folder(CALIB_FOLDER)
    out_dir_known = OUTPUT_BASE / "K_known"
    run_sfm_and_save(img_paths, K_known, out_dir_known, name_tag, suffix="K_known")
