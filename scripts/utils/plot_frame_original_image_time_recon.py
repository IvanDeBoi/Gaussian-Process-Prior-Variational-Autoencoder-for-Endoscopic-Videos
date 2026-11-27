from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_first_dataset_image_time_latents import (
    DATASET,
    RESOLUTION,
    SAVED_MODELS_ROOT,
    WORKSPACE,
    load_dataset,
    make_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot original, image-encoded reconstruction, and time-encoded reconstruction for one frame."
    )
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--saved-models-root", type=Path, default=SAVED_MODELS_ROOT)
    parser.add_argument("--data-root", type=Path, default=WORKSPACE / "Data")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "Pics" / "latent_first_dataset_image_time")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--frame", type=int, default=50)
    return parser.parse_args()


def denormalise(image: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return image * stds.reshape(1, 1, 3) + means.reshape(1, 1, 3)


def load_normalisation(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_path)
    means = data["means"] if "means" in data.files else np.array([0.0, 0.0, 0.0])
    stds = data["stds"] if "stds" in data.files else np.array([1.0, 1.0, 1.0])
    return means.astype(np.float32), stds.astype(np.float32)


def display_image(flat_or_hwc: np.ndarray, resolution: int, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    image = np.asarray(flat_or_hwc).reshape(resolution, resolution, 3)
    image = denormalise(image, means, stds)
    return np.clip(image, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    x, y, mask = load_dataset(args.data_root, args.dataset, args.resolution)
    data_path = args.data_root / f"{args.dataset} {args.resolution} X Ym01sGM.npz"
    means, stds = load_normalisation(data_path)

    frame_indices = np.where(x.reshape(-1) == args.frame)[0]
    if len(frame_indices) != 1:
        raise ValueError(f"Expected exactly one frame {args.frame}; found {len(frame_indices)}")
    idx = int(frame_indices[0])

    _, hpa = make_models(args, x, y, mask)
    y_frame = y[idx:idx + 1].reshape(1, args.resolution, args.resolution, 3)
    x_frame = x[idx:idx + 1]

    z_img, _ = hpa.image_encode(y_frame, batch_size=1)
    z_time, _ = hpa.time_encode(x_frame, use_likelihood=True, return_full_cov=False)
    recon_img = hpa.decode(z_img)[0]
    recon_time = hpa.decode(z_time)[0]

    panels = [
        ("Original", display_image(y[idx], args.resolution, means, stds)),
        ("Image encoded", display_image(recon_img, args.resolution, means, stds)),
        ("Time encoded", display_image(recon_time, args.resolution, means, stds)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{args.dataset} frame {args.frame}")
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.dataset}_frame_{args.frame:04d}_original_image_encoded_time_encoded.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
