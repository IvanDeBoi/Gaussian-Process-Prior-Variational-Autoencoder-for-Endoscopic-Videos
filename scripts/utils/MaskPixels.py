import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: config_endo, gpvae_endo
import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

import torch
from gpvae_endo.networks.duck_net import DuckNet
from gpvae_endo.paths import weights_dir

DUCKNET_WEIGHTS_PATH = str(weights_dir() / 'Duck16mod_Spec_99_new.pt')
# ---------------------- BAND STRIPES (PERIODIC ARTIFACTS) ----------------------

def detect_banding_regions(img_bgr,
                           edge_thresh=40,
                           max_stripe_thickness=6,
                           min_stripe_length_ratio=0.6,
                           **kwargs):
    """
    Detect narrow, nearly horizontal banding stripes.
    Returns a uint8 mask in {0,1} with sparse stripe pixels.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Slight blur: kill texture, keep banding
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)

    # 2) Horizontal-structure emphasis:
    #    use vertical gradient (dI/dy) so horizontal lines respond strongly
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gy_abs = np.abs(gy)

    # 3) Threshold gradient to get candidate horizontal edges
    #    (tune edge_thresh if needed)
    _, edge_mask = cv2.threshold(
        gy_abs.astype(np.uint8), edge_thresh, 255, cv2.THRESH_BINARY
    )

    # 4) Clean small noise, encourage thin horizontal segments
    #    kernel: wide horizontally, thin vertically
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    edge_clean = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel)

    # 5) Connected components; keep only thin, long structures
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_clean, connectivity=8
    )
    band_mask = np.zeros((h, w), dtype=np.uint8)

    min_length = int(min_stripe_length_ratio * w)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]

        # heuristics:
        # - stripe must be much wider than tall
        # - thickness (height) limited
        # - must be reasonably long horizontally
        aspect = ww / max(1, hh)
        if (hh <= max_stripe_thickness and
            ww >= min_length and
            aspect >= 8 and
            area < 0.1 * h * w):  # avoid huge anatomical blobs
            band_mask[labels == i] = 1

    return band_mask


# ---------------------- BLURRY REGIONS (LAPLACIAN) ----------------------


def detect_blurry_regions_laplacian(
        image_bgr: np.ndarray,
        window_size: int = 45,  # Increased from 25/30 to expand neighborhood
        blur_thresh_factor: float = 0.15,  # Lowered from 0.2/0.3 to be stricter
        min_area: int = 500,  # Reduced slightly to catch smaller peripheral blur
        max_fraction: float = 0.25,  # Increased so we don't discard the mask if the frame is very blurry
        exclude_bright_v: int = 230,  # Only exclude the very brightest specular cores
        dilation_iter: int = 2  # New: Expand the mask outward
):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Laplacian Variance
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)

    k = window_size
    kernel = np.ones((k, k), np.float32) / (k * k)

    # Calculate local variance
    mean_lap = cv2.filter2D(lap_abs, -1, kernel)
    mean_lap_sq = cv2.filter2D(lap_abs ** 2, -1, kernel)
    local_var = mean_lap_sq - mean_lap ** 2

    mean_var = local_var.mean()
    thr_var = blur_thresh_factor * mean_var

    # Initial raw mask
    blur_mask_raw = (local_var < thr_var).astype(np.uint8)

    # 2. Refine: Protect high-gradient edges (like veins) from being called blur
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_mean = cv2.blur(grad, (k, k))

    # If the local gradient is significantly higher than average, it's sharp
    g_thr = 0.4 * grad_mean.mean()
    blur_mask_raw[grad_mean > g_thr] = 0

    # 3. Exclude bright spots (specular handles these)
    _, _, v = cv2.split(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV))
    blur_mask_raw[v >= exclude_bright_v] = 0

    # 4. Growth & Cleaning
    # Use a larger ellipse to bridge gaps and expand the footprint
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blur_clean = cv2.morphologyEx(blur_mask_raw, cv2.MORPH_CLOSE, kernel_clean)

    # Dilate to make the green areas "bigger" as requested
    if dilation_iter > 0:
        blur_clean = cv2.dilate(blur_clean, kernel_clean, iterations=dilation_iter)

    # 5. Connected Components & Area Filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blur_clean, 8)
    final_blur = np.zeros_like(blur_clean)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            final_blur[labels == i] = 1

    # Safety check: if the whole image is "blurry", something is wrong with thresholds
    if final_blur.mean() > max_fraction:
        # Optional: return a reduced version or just keep it
        pass

    return final_blur

# ---------------------- BRIGHTNESS CLASSIFICATION (MDPI-LIKE) ----------------------


def classify_brightness_mdpi(bgr, La=112.0, t1=0.3):
    """
    Brightness classification: 'high' / 'medium' / 'low' based on average V. [page:0]
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)

    la = V.mean()
    T1 = (la - La) / La  # normalized brightness measure [page:0]

    if T1 > t1:
        return "high"
    elif T1 < -t1:
        return "low"
    else:
        return "medium"


def enhance_low_brightness_mdpi(bgr, gamma_min=0.6, gamma_max=0.9):
    """
    Simplified V-channel brightness enhancement for low-brightness images. [page:0]
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv)

    la = V.mean()
    alpha = np.clip(la / 255.0, 0.0, 1.0)
    gamma = gamma_max - (gamma_max - gamma_min) * (1.0 - alpha)

    V_norm = V / 255.0
    V_enh = np.power(V_norm, gamma) * 255.0
    V_enh = np.clip(V_enh, 0, 255)

    hsv_enh = cv2.merge([H, S, V_enh]).astype(np.uint8)
    return cv2.cvtColor(hsv_enh, cv2.COLOR_HSV2BGR)


# ---------------------- SPECULAR REFLECTIONS (MDPI-INSPIRED, LESS AGGRESSIVE) ----------------------


def detect_specular_mask(
        image_bgr: np.ndarray,
        min_area: int = 80,
        dilation_iter: int = 2,
        **kwargs
) -> np.ndarray:
    # 1. Brightness category (as before)
    cat = classify_brightness_mdpi(image_bgr)

    # 2. Channels
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(image_bgr.astype(np.float32))

    # 3. Enhanced V
    se_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    v_enh = cv2.add(v, cv2.morphologyEx(v, cv2.MORPH_TOPHAT, se_th))

    # 4. Adaptive thresholds – slightly more permissive than before
    mu_v, sigma_v = v_enh.mean(), v_enh.std()
    if cat == "high":
        t_v = max(235.0, mu_v + 1.2 * sigma_v)
        t_s = 35
        t_dev = 10.0
    elif cat == "medium":
        t_v = max(225.0, mu_v + 1.0 * sigma_v)
        t_s = 50
        t_dev = 12.0
    else:  # low
        t_v = max(215.0, mu_v + 0.8 * sigma_v)
        t_s = 70
        t_dev = 14.0

    # 5. Whiteness criterion (RGB)
    mean_ch = (b + g + r) / 3.0
    color_dev = np.sqrt(
        (b - mean_ch) ** 2 +
        (g - mean_ch) ** 2 +
        (r - mean_ch) ** 2
    )

    core_mask = (v_enh >= t_v) & (s <= t_s) & (color_dev < t_dev)

    # 6. Gradient gating – more tolerant but still vein‑safe
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    # Only consider very bright pixels as possible specular even if gradient is high
    grad_thr = grad_mag.mean() * 2.0  # slightly lower than before
    grad_candidates = (grad_mag > grad_thr) & (v > 225) & (s < 60)

    combined = np.maximum(core_mask.astype(np.uint8),
                          grad_candidates.astype(np.uint8))

    # 7. Morphological cleaning & dilation to grow mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(combined.astype(np.uint8), kernel, iterations=dilation_iter)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, 8)
    final_mask = np.zeros_like(dilated)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            final_mask[labels == i] = 1

    return final_mask

# ---------------------- LARGE OVEREXPOSED WHITE AREAS ----------------------


def detect_overexposed_regions(
    image_bgr: np.ndarray,
    v_thresh: int = 235,
    s_thresh: int = 40,
    min_area: int = 500
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    bright = v >= v_thresh
    low_sat = s <= s_thresh
    cand = (bright & low_sat).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    final = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            final[labels == i] = 1

    return final


# ---------------------- AI-BASED SPECULAR / BAD-PIXEL MASK (DUCKNET) ----------------------


def detect_specular_ducknet(
        image_bgr: np.ndarray,
        init_features: int = 16,
        prob_threshold: float = 0.1,
        intensity_thresh: int = 225,
        dilation_iter: int = 2,
        dilation_kernel_size: int = 5,
        pad: int = 25,
) -> np.ndarray:
    """
    AI-based bad-pixel detector using a pre-trained DuckNet segmentation model.

    Mirrors the pipeline in Get_Mask.py but is self-contained so it can be
    called from Prepare_Dataset alongside the classical detectors.

    Returns a uint8 mask in {0, 1} where 1 = bad pixel (to be excluded),
    consistent with the convention used by the other detect_* functions.

    Parameters
    ----------
    image_bgr       : BGR image (numpy uint8), same as every other detector.
    weights_path    : Absolute or relative path to the .pt checkpoint file.
    init_features   : DuckNet width parameter (must match the checkpoint).
    prob_threshold  : Sigmoid probability above which a pixel is flagged.
    intensity_thresh: Pixels brighter than this (grayscale) are also flagged.
    dilation_iter   : Morphological dilation iterations applied to the combined mask.
    dilation_kernel_size: Side length of the square dilation kernel.
    pad             : Border reflection padding added before resizing to 256 px
                      (reduces edge artefacts; cropped away before returning).
    """


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load model (re-used across calls if caller caches the result) ---
    model = DuckNet(
        in_channels=3,
        out_channels=1,
        depth=5,
        init_features=init_features,
        normalization=None,
        use_multiplier=False,
        out_activation=None,
    ).to(device)

    checkpoint = torch.load(DUCKNET_WEIGHTS_PATH, map_location=device, weights_only=False)
    actual_weights = (checkpoint['model_state_dict']
                      if 'model_state_dict' in checkpoint else checkpoint)
    cleaned_weights = {k.replace("module.", ""): v for k, v in actual_weights.items()}
    model.load_state_dict(cleaned_weights, strict=False)
    model.eval()

    h_orig, w_orig = image_bgr.shape[:2]

    # --- Pad with reflected border to reduce edge artefacts ---
    img_padded = cv2.copyMakeBorder(
        image_bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT
    )
    img_blurred = cv2.GaussianBlur(img_padded, (25, 25), 0)
    img_blurred[pad:-pad, pad:-pad] = image_bgr  # keep centre sharp

    # --- Resize to 256 x 256 for the network ---
    img_resized = cv2.resize(img_blurred, (256, 256))

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_rgb.astype(np.float32) / 255.0 - mean) / std
    img_tensor = (torch.from_numpy(img_normalized)
                  .permute(2, 0, 1).unsqueeze(0).float().to(device))

    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.sigmoid(logits).cpu().numpy().squeeze()

    ai_mask = (probs > prob_threshold).astype(np.uint8)

    # --- Intensity fallback (very bright pixels) ---
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, intensity_mask = cv2.threshold(gray, intensity_thresh, 1, cv2.THRESH_BINARY)
    intensity_mask = intensity_mask.astype(np.uint8)

    # --- Combine & dilate at 256 x 256 resolution ---
    combined = cv2.bitwise_or(ai_mask, intensity_mask)
    kernel   = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated  = cv2.dilate(combined, kernel, iterations=dilation_iter)

    # --- Upsample back to padded size, then crop padding off ---
    mask_padded = cv2.resize(
        dilated,
        (w_orig + 2 * pad, h_orig + 2 * pad),
        interpolation=cv2.INTER_NEAREST,
    )
    bad_mask = mask_padded[pad:-pad, pad:-pad]  # shape == (h_orig, w_orig)

    return bad_mask.astype(np.uint8)


# ---------------------- PLOTTING ----------------------


def save_side_by_side_plot(orig_bgr, combined_bgr, out_path):
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    comb_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(comb_rgb)
    plt.title("Masked")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


