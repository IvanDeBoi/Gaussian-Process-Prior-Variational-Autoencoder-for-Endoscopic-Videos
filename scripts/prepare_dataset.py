import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root: config_endo, gpvae_endo
import os
import re
import cv2
import numpy as np
import time
from PIL import Image

MUST_USE_BAND_M = False
MUST_USE_BLUR_M = False
MUST_USE_SPEC_M = False
MUST_USE_SPEC_DUCK_M = True
MUST_USE_OVER_M = False


from config_endo import (
    FOLDER_DATA,
    IMG_SRC_FOLDER,
    FOLDER_OUTPUT,
    MUST_PRESERVE_COLOR,
    RESOLUTION,
    SUBSTRING_FILTER,
    GASTRONET_MEAN,
    GASTRONET_STD
)

# --- Added: mask utilities from Utils/MaskPixels.py ---
from scripts.utils.MaskPixels import (
    detect_banding_regions,
    detect_blurry_regions_laplacian,
    detect_specular_mask,
    detect_overexposed_regions,
    detect_specular_ducknet
)


def load_and_prepare_data(
        data_folder,
        substring_filter,
        resolution,
        must_preserve_color=False,
        max_frames=None,
):
    """
    Loads, normalizes, and flattens images for PCA, computes artifact masks,
    and saves multicolor mask visualizations for quality control.
    """
    image_data_list = []

    if must_preserve_color:
        read_flag = cv2.IMREAD_COLOR
        print("Processing images as color (3 channels)")
    else:
        read_flag = cv2.IMREAD_GRAYSCALE
        print("Processing images as grayscale (1 channel)")

    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Error: Data folder '{data_folder}' not found")

    print(f"Reading {data_folder} and looking for {substring_filter} and {resolution}")
    all_files = os.listdir(data_folder)
    # print(all_files)

    # Match filenames like C3VD_full_t1_v1_336_0651.png
    candidate_files = [
        f for f in all_files
        if substring_filter in f
           and f"_{resolution}_" in f
           and f.lower().endswith(".png")
    ]

    trailing_digits_re = re.compile(r"(\d+)(?=\.png$)", re.IGNORECASE)

    files_with_numbers = []
    for f in candidate_files:
        m = trailing_digits_re.search(f)
        if m:
            num = int(m.group(1))
            files_with_numbers.append((f, num))
        else:
            print(f" WARNING: could not parse numeric suffix from {f}")

    if not files_with_numbers:
        raise ValueError("Error: No matching PNG files with numeric suffix found")

    # Sort by numeric suffix ascending
    files_with_numbers.sort(key=lambda t: t[1])

    if max_frames is not None:
        files_with_numbers = files_with_numbers[:max_frames]
        print(f"Limiting to max_frames={max_frames}, using {len(files_with_numbers)} files")

    sorted_files = [t[0] for t in files_with_numbers]
    frame_numbers_sorted = [t[1] for t in files_with_numbers]

    imgs = []
    red_pixels_all, green_pixels_all, blue_pixels_all = [], [], []

    # Prepare visualization folder
    vis_dir = os.path.join("Resulting_Masks", f"{substring_filter}_{resolution}")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Loading {len(sorted_files)} files and computing masks...")

    M_list = []
    for i, filename in enumerate(sorted_files):
        filepath = os.path.join(data_folder, filename)
        print(f"Processing {filename} ({i+1}/{len(sorted_files)})")
        img_cv = cv2.imread(filepath, read_flag)

        if img_cv is None:
            continue

        # For processing/masking, we need BGR
        if must_preserve_color:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_bgr_for_mask = img_cv.copy()  # already BGR
            normalized_img = img_rgb.astype(np.float32) / 255.0
        else:
            # If grayscale, convert to BGR for the mask detectors
            img_bgr_for_mask = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
            normalized_img = img_cv.astype(np.float32) / 255.0

        imgs.append(normalized_img)
        image_data_list.append(normalized_img.flatten())

        # --- Mask Logic ---
        band_m = np.zeros_like(img_bgr_for_mask[:, :, 0], dtype=np.uint8)
        blur_m = np.zeros_like(img_bgr_for_mask[:, :, 0], dtype=np.uint8)
        spec_m = np.zeros_like(img_bgr_for_mask[:, :, 0], dtype=np.uint8)
        spec_duck_m = np.zeros_like(img_bgr_for_mask[:, :, 0], dtype=np.uint8)
        over_m = np.zeros_like(img_bgr_for_mask[:, :, 0], dtype=np.uint8)

        if MUST_USE_BAND_M:
            band_m = detect_banding_regions(img_bgr_for_mask, resp_thresh_factor=5.0)
        if MUST_USE_BLUR_M:
            blur_m = detect_blurry_regions_laplacian(img_bgr_for_mask)
        if MUST_USE_SPEC_M:
            spec_m = detect_specular_mask(img_bgr_for_mask, min_area=50)
        if MUST_USE_SPEC_DUCK_M:
            spec_duck_m = detect_specular_ducknet(img_bgr_for_mask)
        if MUST_USE_OVER_M:
            over_m = detect_overexposed_regions(img_bgr_for_mask, v_thresh=235, s_thresh=40)

        # Artifact = 1 if any detected
        artefact_mask = (band_m | blur_m | spec_m | spec_duck_m | over_m).astype(np.uint8)
        valid_mask = (artefact_mask == 0).astype(np.uint8)
        M_list.append(valid_mask.reshape(-1))

        # --- Multicolor Visualization (Save every nth frame) ---
        if i % 10 == 0:
            vis_img = img_bgr_for_mask.copy()
            vis_img[band_m.astype(bool)] = (0, 0, 255)  # Red: Banding
            vis_img[blur_m.astype(bool)] = (0, 255, 0)  # Green: Blur
            vis_img[spec_m.astype(bool)] = (255, 255, 0)  # Cyan: Specular
            vis_img[spec_duck_m.astype(bool)] = (255, 0, 0) # Blue: Specular DuckNet
            vis_img[over_m.astype(bool)] = (255, 0, 255)  # Magenta: Overexposed

            out_name = f"check_{frame_numbers_sorted[i]:06d}.png"
            cv2.imwrite(os.path.join(vis_dir, out_name), vis_img)

        if must_preserve_color:
            red_pixels_all.append(normalized_img[:, :, 0].reshape(-1))
            green_pixels_all.append(normalized_img[:, :, 1].reshape(-1))
            blue_pixels_all.append(normalized_img[:, :, 2].reshape(-1))

    Y01 = np.array(image_data_list)
    M = np.stack(M_list, axis=0)
    x = np.array(frame_numbers_sorted, dtype=np.int32).reshape(-1, 1)

    n_frames = len(imgs)
    H, W = imgs[0].shape[:2]
    C = 3 if must_preserve_color else 1

    # --- Stats and Normalization ---
    if must_preserve_color:
        ORIGINAL_MEAN = [np.mean(red_pixels_all), np.mean(green_pixels_all), np.mean(blue_pixels_all)]
        ORIGINAL_STD = [np.std(red_pixels_all), np.std(green_pixels_all), np.std(blue_pixels_all)]

        # Ym0s1 Calculation
        Y01_res = Y01.reshape(n_frames, H, W, C)
        Ym0s1_res = np.empty_like(Y01_res)
        for c in range(3):
            Ym0s1_res[..., c] = (Y01_res[..., c] - ORIGINAL_MEAN[c]) / ORIGINAL_STD[c]
        Ym0s1 = Ym0s1_res.reshape(n_frames, -1)

        # Ym0s1G (Gastronet)
        Ym0s1G_res = np.empty_like(Y01_res)
        for c in range(3):
            Ym0s1G_res[..., c] = (Y01_res[..., c] - GASTRONET_MEAN[c]) / GASTRONET_STD[c]
        Ym0s1G = Ym0s1G_res.reshape(n_frames, -1)
    else:
        mean_g = np.mean(Y01)
        std_g = np.std(Y01)
        ORIGINAL_MEAN, ORIGINAL_STD = [mean_g], [std_g]
        Ym0s1 = (Y01 - mean_g) / std_g
        Ym0s1G = Ym0s1  # Placeholder if no Gastronet gray defined

    return Y01, Ym0s1, Ym0s1G, x, ORIGINAL_MEAN, ORIGINAL_STD, M


def create_gif(data_folder, dest_folder, substring_filter, resolution, max_frames=None):
    """Standard GIF creation logic based on sorted numeric suffixes."""
    filename_for_gif = f"{substring_filter}_{resolution}.gif"
    gif_path = os.path.join(dest_folder, filename_for_gif)
    if os.path.exists(gif_path): return

    all_files = [f for f in os.listdir(data_folder) if
                 substring_filter in f and f"_{resolution}_" in f and f.endswith(".png")]
    trailing_digits_re = re.compile(r"(\d+)(?=\.png$)", re.IGNORECASE)

    files_with_numbers = []
    for f in all_files:
        m = trailing_digits_re.search(f)
        if m: files_with_numbers.append((f, int(m.group(1))))

    files_with_numbers.sort(key=lambda t: t[1])
    if max_frames: files_with_numbers = files_with_numbers[:max_frames]

    pil_frames = [Image.open(os.path.join(data_folder, t[0])).convert('RGB') for t in files_with_numbers]
    if pil_frames:
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=50, loop=0, optimize=True)


if __name__ == "__main__":
    max_frames = 2000
    start_total = time.time()

    print(f"\n--- Processing: {SUBSTRING_FILTER} @ {RESOLUTION} ---")

    create_gif(IMG_SRC_FOLDER, FOLDER_OUTPUT, SUBSTRING_FILTER, RESOLUTION, max_frames)

    Y01, Ym0s1, Ym0s1G, x, o_mean, o_std, M = load_and_prepare_data(
        IMG_SRC_FOLDER, SUBSTRING_FILTER, RESOLUTION, MUST_PRESERVE_COLOR, max_frames
    )

    # Save NPZ files
    base_name = f"{SUBSTRING_FILTER} {RESOLUTION} X"

    np.savez(os.path.join(FOLDER_DATA, f"{base_name} Y01M.npz"), x=x, Y=Y01, M=M)
    print(f"\nSaved at {os.path.join(FOLDER_DATA, f'{base_name} Y01M.npz')}")

    np.savez(os.path.join(FOLDER_DATA, f"{base_name} Ym0s1M.npz"), x=x, Y=Ym0s1, M=M, means=o_mean, stds=o_std)
    print(f"\nSaved at {os.path.join(FOLDER_DATA, f'{base_name} Ym0s1M.npz')}")

    np.savez(os.path.join(FOLDER_DATA, f"{base_name} Ym01sGM.npz"), x=x, Y=Ym0s1G, M=M, means=GASTRONET_MEAN, stds=GASTRONET_STD)
    print(f"\nSaved at {os.path.join(FOLDER_DATA, f'{base_name} Ym01sGM.npz')}")

    print(f"\nTotal preparation time: {(time.time() - start_total):.2f}s")