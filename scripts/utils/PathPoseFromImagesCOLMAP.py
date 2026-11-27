import os
import subprocess
from pathlib import Path

def run_cmd(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

def run_colmap_sfm(
    image_dir,
    workspace_dir,
    use_global=True,
    camera_model="PINHOLE",
    single_camera=True,
):
    """
    image_dir: folder with (undistorted) images
    workspace_dir: base folder where COLMAP db + sparse model will be stored
    use_global: True -> global_mapper, False -> incremental mapper
    camera_model: PINHOLE, SIMPLE_RADIAL, OPENCV, etc.
    single_camera: if True, treat all images as same camera (shared intrinsics)
    """
    image_dir = Path(image_dir)
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    db_path = workspace_dir / "database.db"
    sparse_dir = workspace_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    # 1) Feature extraction
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", camera_model,
    ]
    if single_camera:
        cmd += ["--ImageReader.single_camera", "1"]
    run_cmd(cmd)

    # 2) Matching (exhaustive; for long sequences you might prefer sequential_matcher)
    cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
    ]
    run_cmd(cmd)

    # 3) Mapping
    if use_global:
        # global SfM
        cmd = [
            "colmap", "global_mapper",
            "--database_path", str(db_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
        ]
    else:
        # incremental SfM
        cmd = [
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
        ]
    run_cmd(cmd)

    # The first model is usually in sparse/0
    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        raise RuntimeError(f"No model found in {model_dir}")

    print(f"COLMAP sparse model stored in {model_dir}")
    return model_dir


def export_poses_to_cam_files(model_dir, output_dir=None):
    """
    Use colmap model_converter to export per-image CAM files
    (simple text with intrinsics + pose). You can then parse in Python.
    """
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir / "cams_export"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    cmd = [
        "colmap", "model_converter",
        "--input_path", str(model_dir),
        "--output_path", str(output_dir),
        "--output_type", "CAM",  # one .cam file per image
    ]
    run_cmd(cmd)
    print(f"Exported per-image poses to {output_dir}")
    return output_dir


if __name__ == "__main__":
    # Example usage with your undistorted sequence
    IMAGE_FOLDER = "../Data/C3VDv2 c1 asc t1 v1 full res/Undistorted"
    WORKSPACE   = "../Data/C3VDv2_COLMAP_workspace"

    # 1) Run COLMAP SfM (global)
    model_dir = run_colmap_sfm(
        image_dir=IMAGE_FOLDER,
        workspace_dir=WORKSPACE,
        use_global=True,
        camera_model="PINHOLE",     # or OPENCV if you want radial+tan
        single_camera=True,
    )

    # 2) Export poses
    cams_dir = export_poses_to_cam_files(model_dir)

    # Now you can parse the .cam files in cams_dir to get K, R, t per image.
