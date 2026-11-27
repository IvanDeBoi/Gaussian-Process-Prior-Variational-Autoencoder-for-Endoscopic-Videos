"""Reconstruct every frame of a video (train frames via the image encoder, missing frames via the GP) and save
per-frame PNGs, optional comparison grids and a side-by-side video.
"""
import os

import cv2
import numpy as np
import torch

from gpvae_endo.viz.images import (
    add_panel_title,
    build_centered_text_panel,
    build_info_panel,
    decoded_image_to_uint8,
    image_uint8_to_rgb,
    mask_overlay_to_rgb,
)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def reconstruct_and_save_all_frames(
        model,
        x_train,
        x_test,
        Y_train,
        Y_test,
        M_train,
        M_test,
        x_scaler,
        RESOLUTION,
        SUBSTRING_FILTER,
        underlying_model,
        must_preserve_color,
        # New extra info
        encoder_name="n/a",
        decoder_name="n/a",
        latent_dim="n/a",
        kernel="n/a",
        lpips="n/a",
        mask="n/a",
        gan="n/a",
        str_with_params="not set",
        *,
        paths,
        denormalise,
):
    """Decode every frame (image-encode train, time-encode missing), then save per-frame PNGs, grids and a 2x2-panel video."""
    print("Reconstructing all frames")
    train_lookup = {}
    for i in range(x_train.shape[0]):
        frame_nr = int(x_train[i, 0])
        train_lookup[frame_nr] = {
            "Y": Y_train[i],
            "mask": None if M_train is None else M_train[i],
        }

    test_lookup = {}
    for i in range(x_test.shape[0]):
        frame_nr = int(x_test[i, 0])
        test_lookup[frame_nr] = {
            "Y": Y_test[i],
            "mask": None if M_test is None else M_test[i],
        }

    frame_numbers = sorted(set(train_lookup.keys()) | set(test_lookup.keys()))
    all_entries = []
    for frame_nr in frame_numbers:
        in_train = frame_nr in train_lookup
        source_entry = train_lookup.get(frame_nr, test_lookup.get(frame_nr))
        all_entries.append({
            "frame_nr": frame_nr,
            "Y_original": source_entry["Y"],
            "Y_train": train_lookup[frame_nr]["Y"] if in_train else None,
            "mask_train": train_lookup[frame_nr]["mask"] if in_train else None,
            "in_train": in_train,
        })

    print(f"Making reconstruction panels for {len(all_entries)} frames")

    x_all = np.array([[entry["frame_nr"]] for entry in all_entries], dtype=float)
    x_all_scaled = x_scaler.transform(x_all)

    H, W = RESOLUTION, RESOLUTION
    C = 3 if must_preserve_color else 1
    img_shape = (H, W, C) if C == 3 else (H, W)

    # if not hasattr(model, "time_encode"):
    #     print("Skipping reconstruction panels: model has no time_encode functionality.")
    #     return

    # Split entries into train (image-encode) and test/missing (time-encode)
    train_indices = [i for i, e in enumerate(all_entries) if e["in_train"]]
    test_indices = [i for i, e in enumerate(all_entries) if not e["in_train"]]

    # Allocate output arrays (decoded images + per-latent std)
    Q = None  # will be inferred from first encode call
    Y_all_decoded = np.empty((len(all_entries),), dtype=object)
    Z_all_std = np.full(len(all_entries), np.nan)  # scalar mean-std per frame

    with torch.no_grad():
        # --- Train frames: image_encode ---
        if train_indices and hasattr(model, "image_encode"):
            print(f"Image-encoding {len(train_indices)} train frames")
            Y_train_imgs = np.stack(
                [all_entries[i]["Y_original"] for i in train_indices], axis=0
            )  # (N, H*W*C) flattened — reshape to (N, C, H, W) for image_encode
            Y_train_imgs = Y_train_imgs.reshape(len(train_indices), H, W, C)
            Z_img_mean, Z_img_std = model.image_encode(Y_train_imgs)
            Y_img_decoded = _to_numpy(model.decode(Z_img_mean))
            Z_img_std_np = _to_numpy(Z_img_std)
            for out_pos, src_pos in enumerate(train_indices):
                Y_all_decoded[src_pos] = Y_img_decoded[out_pos]
                Z_all_std[src_pos] = float(np.mean(Z_img_std_np[out_pos]))
        elif train_indices:
            print("model has no image_encode; falling back to time_encode for train frames")
            x_train_scaled_sub = x_all_scaled[train_indices]
            Z_tr_mean, Z_tr_std = model.time_encode(x_train_scaled_sub, use_likelihood=True)
            Y_tr_decoded = _to_numpy(model.decode(Z_tr_mean))
            Z_tr_std_np = _to_numpy(Z_tr_std)
            for out_pos, src_pos in enumerate(train_indices):
                Y_all_decoded[src_pos] = Y_tr_decoded[out_pos]
                Z_all_std[src_pos] = float(np.mean(Z_tr_std_np[out_pos]))

        # --- Test / missing frames: time_encode ---
        if test_indices and hasattr(model, "time_encode"):
            print(f"Time-encoding {len(test_indices)} missing (test) frames")
            x_test_scaled_sub = x_all_scaled[test_indices]
            Z_te_mean, Z_te_std = model.time_encode(x_test_scaled_sub, use_likelihood=True)
            Y_te_decoded = _to_numpy(model.decode(Z_te_mean))
            Z_te_std_np = _to_numpy(Z_te_std)
            for out_pos, src_pos in enumerate(test_indices):
                Y_all_decoded[src_pos] = Y_te_decoded[out_pos]
                Z_all_std[src_pos] = float(np.mean(Z_te_std_np[out_pos]))

    video_path = os.path.join(
        paths.debug_dataset,
        f"{str_with_params}.mp4",
    )
    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (2 * W, 2 * H),
    )

    try:
        for i, entry in enumerate(all_entries):
            frame_nr = entry["frame_nr"]
            original_uint8 = decoded_image_to_uint8(
                entry["Y_original"], img_shape, must_preserve_color, denormalise=denormalise
            )
            decoded_i = Y_all_decoded[i]

            if decoded_i is None or np.asarray(decoded_i).size != np.prod(img_shape):
                print(f"Frame {frame_nr}: no time reconstruction possible, using black frame")
                if must_preserve_color:
                    reconstructed_uint8 = np.zeros((H, W, 3), dtype=np.uint8)
                else:
                    reconstructed_uint8 = np.zeros((H, W), dtype=np.uint8)
            else:
                reconstructed_uint8 = decoded_image_to_uint8(
                    decoded_i, img_shape, must_preserve_color, denormalise=denormalise
                )

            uncertainty = None if np.isnan(Z_all_std[i]) else float(Z_all_std[i])

            extra_info = {
                "model": underlying_model,
                "filter": SUBSTRING_FILTER,
                "res": RESOLUTION,
                "kernel": kernel,
                "encoder": encoder_name,
                "decoder": decoder_name,
                "latent_dim": latent_dim,
                "lpips": lpips,
                "mask": mask,
                "gan": gan,
            }
            info_panel = build_info_panel(H, W, frame_nr, uncertainty, extra_info)
            original_panel = add_panel_title(
                image_uint8_to_rgb(original_uint8, must_preserve_color),
                "original",
            )

            if entry["in_train"]:
                masked_panel = add_panel_title(
                    mask_overlay_to_rgb(
                        original_uint8, entry["mask_train"], must_preserve_color
                    ),
                    "original + mask",
                )
            else:
                masked_panel = add_panel_title(
                    build_centered_text_panel(H, W, "frame missing"),
                    "original + mask",
                )

            reconstructed_rgb = image_uint8_to_rgb(reconstructed_uint8, must_preserve_color)
            reconstructed_out_path = os.path.join(
                paths.frames,
                f"{str_with_params} {frame_nr:06d}.png"
            )
            cv2.imwrite(reconstructed_out_path, cv2.cvtColor(reconstructed_rgb, cv2.COLOR_RGB2BGR))

            reconstructed_panel = add_panel_title(
                image_uint8_to_rgb(reconstructed_uint8, must_preserve_color),
                "reconstructed",
            )

            top_row = np.concatenate(
                [
                    info_panel,
                    original_panel,
                ],
                axis=1,
            )
            bottom_row = np.concatenate(
                [
                    masked_panel,
                    reconstructed_panel,
                ],
                axis=1,
            )
            grid = np.concatenate(
                [
                    top_row,
                    bottom_row,
                ],
                axis=0,
            )

            video_writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

            if paths.save_grids:
                out_path = os.path.join(
                    paths.grids,
                    f"{str_with_params} {frame_nr:06d} grid.png",
                )
                cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    finally:
        video_writer.release()

    if paths.save_grids:
        print(
            f"Saved reconstruction grids for {len(all_entries)} frames "
            f"to {paths.grids}"
        )
    else:
        print("Skipped saving individual reconstruction grid PNGs")

    print(f"Saved reconstruction video to {video_path}")
