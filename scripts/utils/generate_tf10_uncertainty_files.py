from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from gpvae_endo.config import load_model_settings
from gpvae_endo.models.gpvae_neighbours import GPVAEHPAModel, GPVAESPAModel


DEFAULT_DATA_DIR = WORKSPACE / "Data"
DEFAULT_SAVED_MODELS_DIR = Path(r"C:\Endo\TF_10 with mask\Saved_Models")
DEFAULT_OUTPUT_DIR = Path(r"C:\Endo\TF_10 with mask\Pics\newly generated")
MODEL_RE = re.compile(
    r"^(?P<model>GPVAE_(?:HPA|SPA))\s+"
    r"(?P<dataset>\S+)\s+"
    r"(?P<resolution>\d+)\s+"
    r"(?P<kernel>\S+)\s+"
    r"(?P<encoder>\S+)\s+"
    r"(?P<decoder>\S+)\s+"
    r"(?P<latent_dim>\d+)D\s+"
    r"L_(?P<lpips>True|False)_(?P<lpips_res>\d+)\s+"
    r"M_(?P<mask>True|False)\s+"
    r"G_(?P<gan>True|False)\s+"
    r"GAP_(?P<gap>True|False)_(?P<gap_size>\d+)\s+"
    r"TF_(?P<test_percentage>\d+)\.pth$"
)


def parse_model_path(path: Path) -> dict[str, str | int | bool] | None:
    match = MODEL_RE.match(path.name)
    if not match:
        return None

    info: dict[str, str | int | bool] = match.groupdict()
    for key in ("resolution", "latent_dim", "lpips_res", "gap_size", "test_percentage"):
        info[key] = int(info[key])
    for key in ("lpips", "mask", "gan", "gap"):
        info[key] = str(info[key]) == "True"
    return info


def denormalise(images: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    output = np.empty_like(images)
    for channel in range(images.shape[-1]):
        output[..., channel] = images[..., channel] * stds[channel] + means[channel]
    return output


def rmse_denormalised(
    y_original: np.ndarray,
    y_predicted: np.ndarray,
    mask: np.ndarray | None,
    img_shape: tuple[int, int, int],
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    original = denormalise(y_original.reshape(-1, *img_shape), means, stds).reshape(y_original.shape[0], -1)
    predicted = denormalise(y_predicted.reshape(-1, *img_shape), means, stds).reshape(y_predicted.shape[0], -1)
    diff2 = (original - predicted) ** 2

    if mask is None:
        return np.sqrt(diff2.mean(axis=1))

    _, _, channels = img_shape
    full_mask = np.repeat(mask, channels, axis=1)
    valid = full_mask.sum(axis=1)
    valid = np.where(valid <= 0, 1.0, valid)
    return np.sqrt((diff2 * full_mask).sum(axis=1) / valid)


def split_data(x: np.ndarray, y: np.ndarray, mask: np.ndarray, test_percentage: int) -> tuple:
    all_idx = np.arange(y.shape[0], dtype=int)
    _, idx_test = train_test_split(
        all_idx,
        test_size=test_percentage / 100.0,
        random_state=42,
        shuffle=True,
    )
    idx_test = np.asarray(idx_test, dtype=int)
    idx_train = np.setdiff1d(all_idx, idx_test)

    x_train = x[idx_train]
    y_train = y[idx_train]
    mask_train = mask[idx_train]
    x_test = x[idx_test]
    y_test = y[idx_test]
    mask_test = mask[idx_test]

    train_sort = np.argsort(x_train, axis=0).flatten()
    test_sort = np.argsort(x_test, axis=0).flatten()
    return (
        x_train[train_sort],
        y_train[train_sort],
        mask_train[train_sort],
        x_test[test_sort],
        y_test[test_sort],
        mask_test[test_sort],
    )


def load_dataset(data_dir: Path, dataset: str, resolution: int) -> tuple[np.ndarray, ...]:
    path = data_dir / f"{dataset} {resolution} X Ym01sGM.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path)
    x = data["x"]
    y = data["Y"]
    mask = data["M"]
    means = data["means"] if "means" in data.files else np.array([0, 0, 0], dtype=np.float32)
    stds = data["stds"] if "stds" in data.files else np.array([1, 1, 1], dtype=np.float32)
    return x, y, mask, means, stds


def build_model(info: dict[str, str | int | bool], x_train: np.ndarray, y_train: np.ndarray, mask_train: np.ndarray):
    kwargs = dict(
        x=x_train.reshape(-1, 1),
        Y=y_train,
        M=mask_train,
        device="gpu",
        latent_dim=int(info["latent_dim"]),
        lr=1e-3,
        img_H=int(info["resolution"]),
        img_W=int(info["resolution"]),
        img_C=3,
        epochs=500,
        kernel_gp_prior=str(info["kernel"]),
        n_neighbors=50,
        encoder=str(info["encoder"]),
        decoder=str(info["decoder"]),
        batch_size=64,
    )
    if info["model"] == "GPVAE_HPA":
        return GPVAEHPAModel(**kwargs, settings=load_model_settings())
    if info["model"] == "GPVAE_SPA":
        return GPVAESPAModel(**kwargs, settings=load_model_settings())
    raise ValueError(f"Unsupported model: {info['model']}")


def output_name(info: dict[str, str | int | bool]) -> str:
    return (
        f"{info['dataset']} {info['model']} {info['encoder']} {info['decoder']} "
        f"{info['kernel']} {info['latent_dim']}D "
        f"L_{info['lpips']}_{info['lpips_res']} M_{info['mask']} G_{info['gan']} "
        f"GAP_{info['gap']}_{info['gap_size']} TF_{info['test_percentage']} uncertainty per frame.txt"
    )


def write_uncertainty_file(model_path: Path, info: dict[str, str | int | bool], args: argparse.Namespace) -> Path:
    x, y, mask, means, stds = load_dataset(args.data_dir, str(info["dataset"]), int(info["resolution"]))
    x_train, y_train, mask_train, x_test, y_test, mask_test = split_data(
        x=x,
        y=y,
        mask=mask,
        test_percentage=int(info["test_percentage"]),
    )

    img_shape = (int(info["resolution"]), int(info["resolution"]), 3)
    model = build_model(info, x_train, y_train, mask_train)
    state = torch.load(model_path, map_location=model.device, weights_only=False)
    try:
        model.load_state_dict_full(state)
    except RuntimeError as exc:
        message = str(exc)
        if "noise_prior" not in message or "gp_prior" not in state:
            raise
        print("  GP prior checkpoint has older/newer noise-prior buffers; reloading GP prior with strict=False")
        missing, unexpected = model.gp_prior.load_state_dict(state["gp_prior"], strict=False)
        if missing:
            print(f"  ignored missing GP keys: {list(missing)}")
        if unexpected:
            print(f"  ignored unexpected GP keys: {list(unexpected)}")
    model.eval()

    with torch.no_grad():
        z_train_mean, z_train_std = model.time_encode(x_train.reshape(-1, 1), use_likelihood=True, return_full_cov=False)
        z_test_mean, z_test_std = model.time_encode(x_test.reshape(-1, 1), use_likelihood=True, return_full_cov=False)
        y_train_decoded = model.decode(z_train_mean)
        y_test_decoded = model.decode(z_test_mean)

    train_uncertainty = np.asarray(z_train_std).mean(axis=1)
    test_uncertainty = np.asarray(z_test_std).mean(axis=1)
    train_rmse = rmse_denormalised(y_train, y_train_decoded, mask_train, img_shape, means, stds)
    test_rmse = rmse_denormalised(y_test, y_test_decoded, mask_test, img_shape, means, stds)

    records = []
    for frame_nr, uncertainty, rmse in zip(x_train.squeeze(), train_uncertainty, train_rmse):
        records.append((int(frame_nr), "train", float(uncertainty), float(rmse)))
    for frame_nr, uncertainty, rmse in zip(x_test.squeeze(), test_uncertainty, test_rmse):
        records.append((int(frame_nr), "test", float(uncertainty), float(rmse)))
    records.sort(key=lambda row: row[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / output_name(info)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("frame_nr\tsplit\tuncertainty\trmse_denormalized\n")
        for frame_nr, split, uncertainty, rmse in records:
            handle.write(f"{frame_nr}\t{split}\t{uncertainty:.8f}\t{rmse:.8f}\n")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate missing TF_10 GPVAE uncertainty per-frame text files.")
    parser.add_argument("--saved-models-dir", type=Path, default=DEFAULT_SAVED_MODELS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-percentage", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_paths = []
    for path in sorted(args.saved_models_dir.glob("GPVAE_* TF_*.pth")):
        info = parse_model_path(path)
        if info is None:
            continue
        if int(info["test_percentage"]) != args.test_percentage:
            continue
        model_paths.append((path, info))

    if args.limit is not None:
        model_paths = model_paths[: args.limit]

    print(f"Generating uncertainty files for {len(model_paths)} saved GPVAE models")
    print(f"Output directory: {args.output_dir}")
    for index, (path, info) in enumerate(model_paths, start=1):
        print(f"[{index}/{len(model_paths)}] {path.name}")
        output_path = write_uncertainty_file(path, info, args)
        print(f"  wrote {output_path}")


if __name__ == "__main__":
    main()
