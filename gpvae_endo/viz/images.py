"""Image helpers: side-by-side comparison figures, uint8/RGB conversions and text panels for frames.

Moved from the original driver (Latent_Space_Discovery.py); module globals became explicit parameters.
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compare_images(
        Y_original,
        Y_predicted,
        nrs,
        img_shape,
        folder,
        prefix="Reconstruction",
        cmap_diff="viridis",  # Changed from 'bwr' to 'viridis'
        left_title="Original",
        right_title="Reconstructed",
        M_original=None,
        show_masked=True,  # NEW: controls whether to plot Original+Mask
        *,
        denormalise,
):
    """
    Save one side-by-side figure per image (original, optional original+mask, reconstruction, difference).
    `denormalise` is the callable that undoes the dataset normalisation of one (H, W, C) image.

    Y_original, Y_predicted: (N, D) arrays
    M_original: (N, D) mask array (optional)
    img_shape: (H, W) or (H, W, C)
    """
    os.makedirs(folder, exist_ok=True)

    n_samples = Y_original.shape[0]
    has_mask = (M_original is not None) and show_masked

    n_cols = 5 if has_mask else 4
    width_ratios = [1, 1, 1, 1, 0.05] if has_mask else [1, 1, 1, 0.05]
    fig_width = 18 if has_mask else 14

    for idx in range(n_samples):
        y_orig = Y_original[idx].reshape(img_shape)
        y_orig = denormalise(y_orig)

        y_pred = Y_predicted[idx].reshape(img_shape)
        y_pred = denormalise(y_pred)

        diff = y_orig - y_pred

        fig, axes = plt.subplots(
            1, n_cols, figsize=(fig_width, 5),
            gridspec_kw={"width_ratios": width_ratios}
        )

        # 1. Original
        img_to_show = np.clip(y_orig, 0, 1) if y_orig.ndim == 3 else y_orig
        cmap = "gray" if y_orig.ndim == 2 else None
        axes[0].imshow(img_to_show, cmap=cmap)
        axes[0].set_title(left_title)
        axes[0].axis("off")

        curr_ax = 1

        # 2. Original + Mask (Blue background), only if enabled
        if has_mask:
            mask = M_original[idx].reshape(img_shape[:2])

            if img_to_show.ndim == 2:
                masked_img = np.stack([img_to_show] * 3, axis=-1)
            else:
                masked_img = img_to_show.copy()

            # Set pixels to bright blue [0, 0, 1] where mask is 0
            masked_img[mask < 0.5] = [0, 0, 1]

            axes[curr_ax].imshow(masked_img)
            axes[curr_ax].set_title("Original + Mask")
            axes[curr_ax].axis("off")
            curr_ax += 1

        # 3. Reconstructed
        axes[curr_ax].imshow(
            np.clip(y_pred, 0, 1) if y_pred.ndim == 3 else y_pred,
            cmap=cmap
        )
        axes[curr_ax].set_title(right_title)
        axes[curr_ax].axis("off")
        curr_ax += 1

        # 4. Difference
        diff_vis = np.abs(diff)
        if diff_vis.ndim == 2:
            im = axes[curr_ax].imshow(diff_vis, cmap=cmap_diff)
        else:
            im = axes[curr_ax].imshow(np.mean(diff_vis, axis=-1), cmap=cmap_diff)
        axes[curr_ax].set_title("Difference")
        axes[curr_ax].axis("off")
        curr_ax += 1

        # 5. Colorbar
        plt.colorbar(im, cax=axes[curr_ax])

        fname = os.path.join(folder, f"{prefix} {nrs[idx]}.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close(fig)


def decoded_image_to_uint8(decoded_image, img_shape, must_preserve_color, *, denormalise):
    """
    Convert a decoded flat/HWC image to uint8 using the same pipeline as compare_images:
    reshape -> denormalise -> clip to display range -> uint8.
    `denormalise` is the callable that undoes the dataset normalisation.
    """
    y = np.asarray(decoded_image).reshape(img_shape)

    if must_preserve_color:
        y = denormalise(y)
        y = np.clip(y, 0.0, 1.0)
        return (y * 255.0).astype(np.uint8)

    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def image_uint8_to_rgb(image_uint8, must_preserve_color):
    """Return the uint8 image as RGB (grey images are stacked to 3 channels)."""
    if must_preserve_color:
        return image_uint8
    if image_uint8.ndim == 2:
        return np.stack([image_uint8] * 3, axis=-1)
    return image_uint8


def mask_overlay_to_rgb(image_uint8, mask_flat, must_preserve_color):
    """Return the image as RGB with masked-out pixels (mask < 0.5) painted blue."""
    base_rgb = image_uint8_to_rgb(image_uint8, must_preserve_color).copy()
    if mask_flat is None:
        return base_rgb

    mask_2d = np.asarray(mask_flat).reshape(base_rgb.shape[:2])
    base_rgb[mask_2d < 0.5] = [0, 0, 255]
    return base_rgb


def build_info_panel(height, width, frame_nr, uncertainty, extra_info):
    """Draw a black panel with the frame number, uncertainty and run settings as text."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Heuristic for font scale and spacing based on panel height
    font_scale = 0.45
    thickness = 1
    if height >= 336:
        font_scale = 0.6
        thickness = 2

    line_spacing = int(35 * font_scale / 0.6)

    lines = [
        f"Frame: {frame_nr}",
        f"Uncert: {'n/a' if uncertainty is None else f'{uncertainty:.4f}'}",
        f"Model: {extra_info.get('model', 'n/a')}",
        f"Filter: {extra_info.get('filter', 'n/a')}",
        f"Res: {extra_info.get('res', 'n/a')}",
        f"Kernel: {extra_info.get('kernel', 'n/a')}",
        f"Enc: {extra_info.get('encoder', 'n/a')}",
        f"Dec: {extra_info.get('decoder', 'n/a')}",
        f"Latent: {extra_info.get('latent_dim', 'n/a')}D",
        f"LPIPS={extra_info.get('lpips', 'n/a')}",
        f"MASK={extra_info.get('mask', 'n/a')}",
        f"GAN={extra_info.get('gan', 'n/a')}"
    ]

    for i, line in enumerate(lines):
        y = line_spacing + i * line_spacing
        if y > height - 10:
            break
        cv2.putText(panel, line, (10, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return panel


def build_centered_text_panel(height, width, text):
    """Draw a black panel with `text` centred on it."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(10, (width - text_size[0]) // 2)
    y = max(text_size[1] + 10, (height + text_size[1]) // 2)
    cv2.putText(panel, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return panel


def add_panel_title(panel, title):
    """Return a copy of the panel with a black title bar and `title` written on top."""
    titled_panel = panel.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(title, font, font_scale, thickness)
    x = max(10, (titled_panel.shape[1] - text_size[0]) // 2)
    y = max(25, text_size[1] + 8)

    cv2.rectangle(
        titled_panel,
        (0, 0),
        (titled_panel.shape[1], 36),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        titled_panel,
        title,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return titled_panel
