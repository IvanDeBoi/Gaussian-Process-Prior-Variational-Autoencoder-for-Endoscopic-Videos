import glob
import os
import cv2
import numpy as np

# ---------- USER SETTINGS ----------
# Folder with calibration images (checkerboard)
CALIB_IMG_DIR = "../Data/C3VDv2 CB/"  # change to your folder
RESULT_DIR = "../Data/C3VDv2 CB/calibration_results"  # change to your folder

# Glob pattern for images (supports jpg, png, etc.)
CALIB_PATTERN = os.path.join(CALIB_IMG_DIR, "*.png")

# Checkerboard geometry: number of INNER corners (width, height)
# e.g. 9x6 inner corners for a board with 10x7 squares
CHECKERBOARD = (15, 10)

# Physical size of one square in meters (or mm, any consistent unit)
SQUARE_SIZE = 0.01  # 1 cm

# Output file
OUTPUT_YAML = "calibration_fisheye.yaml"

# Subfolder for undistorted images
# TO_UNDISTORT_SUBFOLDER = CALIB_IMG_DIR #Use this line as a check to see what happens with the cb images
# TO_UNDISTORT_SUBFOLDER = "../Data/C3VDv2 c1 asc t1 v1 full res/Original"
# UNDISTORTED_SUBFOLDER   = "../Data/C3VDv2 c1 asc t1 v1 full res/Undistorted"

TO_UNDISTORT_SUBFOLDER = "../Output_Reconstructed_Frames/C3VDv2C1ascv2t1_336_GPVAE_HPA_Shi_time_encoded_new/FromModels"
UNDISTORTED_SUBFOLDER   = "../Output_Reconstructed_Frames/C3VDv2C1ascv2t1_336_GPVAE_HPA_Shi_time_encoded_new/Undistorted"


# Control zoom / FOV in undistortion (fx, fy scaling)
#   1.0 = use calibrated K (more crop, less distortion)
#   <1.0 = wider view, less cropping
FOCAL_SCALE = 0.7
# -----------------------------------


if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Prepare object points for one board, scaled by SQUARE_SIZE
    # fisheye.calibrate expects: object points as (N, 1, 3)
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # list of (1, N, 3)
    imgpoints = []  # list of (1, N, 2)

    image_paths = sorted(glob.glob(CALIB_PATTERN))
    if not image_paths:
        raise RuntimeError(f"No images found with pattern: {CALIB_PATTERN}")

    print(f"Found {len(image_paths)} calibration images.")

    img_size = None

    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: could not read {fname}, skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]  # (width, height)

        ret, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  cv2.CALIB_CB_FAST_CHECK
        )
        if not ret:
            print(f"Checkerboard NOT found in {fname}, skipping.")
            continue

        # Refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 1e-6)
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria
        )

        objpoints.append(objp.copy())                          # (1, N, 3)
        imgpoints.append(corners_refined.reshape(1, -1, 2))    # (1, N, 2)
        print(f"Checkerboard detected in {fname}: {corners_refined.shape[0]} corners.")

    if len(objpoints) < 3:
        raise RuntimeError("Not enough valid calibration views (need at least 3).")

    print("Running fisheye calibration...")
    # Initialize K and D
    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    rvecs = []
    tvecs = []

    w, h = img_size  # (width, height)

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
             cv2.fisheye.CALIB_CHECK_COND +
             cv2.fisheye.CALIB_FIX_SKEW)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        (w, h),
        K,
        D,
        rvecs,
        tvecs,
        flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    )

    print(f"RMS reprojection error (fisheye): {rms:.6f}")
    print("K (fisheye camera matrix):\n", K)
    print("D (fisheye distortion coeffs):\n", D.ravel())

    # Save to YAML in OpenCV format
    yaml_path = os.path.join(RESULT_DIR, OUTPUT_YAML)
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("D", D)
    fs.release()
    print(f"Fisheye calibration saved to {yaml_path}")

    # ---------- UNDISTORT ALL IMAGES (FISHEYE) ----------
    os.makedirs(UNDISTORTED_SUBFOLDER, exist_ok=True)

    print(f"Undistorting all images from {TO_UNDISTORT_SUBFOLDER} to {UNDISTORTED_SUBFOLDER}")

    # Rectification / undistort maps with adjustable zoom via FOCAL_SCALE
    R = np.eye(3)
    new_K = K.copy()
    new_K[0, 0] *= FOCAL_SCALE
    new_K[1, 1] *= FOCAL_SCALE

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, new_K, (w, h), cv2.CV_16SC2
    )

    to_undistort_pattern = os.path.join(TO_UNDISTORT_SUBFOLDER, "*.png")  # or *.jpg, etc.
    to_undistort_paths = sorted(glob.glob(to_undistort_pattern))

    for fname in to_undistort_paths:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: could not read {fname}, skipping undistort.")
            continue

        undistorted = cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        base = os.path.basename(fname)
        out_path = os.path.join(UNDISTORTED_SUBFOLDER, base)
        cv2.imwrite(out_path, undistorted)
        print(f"Saved fisheye-undistorted image to {out_path}")

    print("Done fisheye undistorting all images.")
