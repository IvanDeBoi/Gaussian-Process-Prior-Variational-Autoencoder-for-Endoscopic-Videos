from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: config_endo, gpvae_endo

import argparse
import re
import sys
import types
from pathlib import Path

import numpy as np


WORKSPACE = Path(r"C:\Users\Ivan\OneDrive - Universiteit Antwerpen\01 Postdoc\04 Exp\15 GPVAE Endo")
SAVED_MODELS_ROOT = Path(r"C:\Endo\Full train with mask\Saved_Models")
DATASET = "c1_ascending_t1_v1"
RESOLUTION = 336
CHECKPOINT_SUFFIX = "256D L_True_256 M_True G_False GAP_False_0 TF_0.pth"
DATA_SUFFIX = "Ym01sGM.npz"

PANEL_TITLE_FONTSIZE = 30
FIGURE_TITLE_FONTSIZE = 34
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 18
COLORBAR_LABEL_FONTSIZE = 22
COLORBAR_TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 22
ZOOM_TITLE_FONTSIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot VAE image-encoded latents and GPVAE image/time "
            "encoded latents using trained checkpoints."
        )
    )
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--saved-models-root", type=Path, default=SAVED_MODELS_ROOT)
    parser.add_argument("--data-root", type=Path, default=WORKSPACE / "Data")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "Pics" / "latent_first_dataset_image_time")
    parser.add_argument("--dataset", default="all", help="Dataset name, or 'all' for every dataset with both checkpoints.")
    parser.add_argument("--gp-model", default="GPVAE_HPA", choices=["GPVAE_HPA", "GPVAE_SPA"])
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--checkpoint-suffix", default=CHECKPOINT_SUFFIX)
    parser.add_argument("--test-percent", type=float, default=None, help="Held-out frame percentage. Defaults to TF_ value in checkpoint suffix.")
    parser.add_argument("--tail-percent", type=float, default=0.0)
    parser.add_argument("--gap-size", type=int, default=0, help="Hold out a contiguous middle gap of this many frames.")
    parser.add_argument("--time-size-max-multiplier", type=float, default=10.0)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--gap-zoom", action="store_true", help="Add inset 3D zoom panels focused on the middle gap.")
    parser.add_argument("--hide-legends", action="store_true", help="Omit per-panel legends for use in combined grids.")
    return parser.parse_args()


def infer_test_percent(checkpoint_suffix: str) -> float:
    match = re.search(r"\bTF_(\d+(?:\.\d+)?)\.pth$", checkpoint_suffix)
    return float(match.group(1)) if match else 0.0


def available_datasets(root: Path, resolution: int, checkpoint_suffix: str, gp_model: str) -> list[str]:
    vae_prefix = "VAE "
    vae_suffix = f" {resolution} EndoVAEEnc EndoVAEDec {checkpoint_suffix}"
    gp_prefix = f"{gp_model} "
    gp_suffix = f" {resolution} CAUCHY EndoVAEEnc EndoVAEDec {checkpoint_suffix}"

    vae_datasets = {
        path.name[len(vae_prefix):-len(vae_suffix)]
        for path in root.glob(f"{vae_prefix}*{vae_suffix}")
        if path.name.startswith(vae_prefix) and path.name.endswith(vae_suffix)
    }
    gp_datasets = {
        path.name[len(gp_prefix):-len(gp_suffix)]
        for path in root.glob(f"{gp_prefix}*{gp_suffix}")
        if path.name.startswith(gp_prefix) and path.name.endswith(gp_suffix)
    }
    return sorted(vae_datasets & gp_datasets)


def install_lpips_stub() -> None:
    """Models instantiate LPIPS in __init__, but these plots never train or use it."""
    import torch

    if "lpips" in sys.modules:
        return

    class DummyLPIPS(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, y):
            return torch.zeros((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype)

    module = types.ModuleType("lpips")
    module.LPIPS = DummyLPIPS
    sys.modules["lpips"] = module


def install_optional_dependency_stubs() -> None:
    install_lpips_stub()
    if "huggingface_hub" not in sys.modules:
        module = types.ModuleType("huggingface_hub")

        def snapshot_download(*args, **kwargs):
            raise RuntimeError("huggingface_hub is stubbed; snapshot_download is unavailable in this plotting run")

        module.snapshot_download = snapshot_download
        sys.modules["huggingface_hub"] = module


def checkpoint_path(
    root: Path,
    model: str,
    dataset: str,
    checkpoint_suffix: str,
    encoder: str = "EndoVAEEnc",
    decoder: str = "EndoVAEDec",
) -> Path:
    if model == "VAE":
        name = f"VAE {dataset} 336 {encoder} {decoder} {checkpoint_suffix}"
    elif model.startswith("GPVAE_"):
        name = f"{model} {dataset} 336 CAUCHY {encoder} {decoder} {checkpoint_suffix}"
    else:
        raise ValueError(model)
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_dataset(data_root: Path, dataset: str, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = data_root / f"{dataset} {resolution} X {DATA_SUFFIX}"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    data = np.load(data_path)
    x = data["x"].reshape(-1, 1)
    y = data["Y"]
    mask = data["M"] if "M" in data.files else None
    sort_idx = np.argsort(x.reshape(-1))
    x = x[sort_idx]
    y = y[sort_idx]
    if mask is not None:
        mask = mask[sort_idx]
    return x, y, mask


def train_indices(n_frames: int, test_percent: float, tail_percent: float, gap_size: int = 0) -> np.ndarray:
    all_idx = np.arange(n_frames, dtype=int)
    if test_percent <= 0 and tail_percent <= 0 and gap_size <= 0:
        return all_idx

    from sklearn.model_selection import train_test_split

    idx_test = np.array([], dtype=int)
    if test_percent > 0:
        _, idx_test = train_test_split(
            all_idx,
            test_size=test_percent / 100.0,
            random_state=42,
            shuffle=True,
        )
        idx_test = np.asarray(idx_test, dtype=int)

    if gap_size > 0:
        mid = n_frames // 2
        half = gap_size // 2
        start = max(0, mid - half)
        end = min(n_frames, mid + half)
        idx_test = np.unique(np.concatenate([idx_test, all_idx[start:end]])).astype(int)

    if tail_percent > 0:
        n_tail = int(tail_percent * n_frames / 100.0)
        if n_tail > 0:
            idx_test = np.unique(np.concatenate([idx_test, np.arange(n_frames - n_tail, n_frames, dtype=int)]))

    return np.setdiff1d(all_idx, idx_test).astype(int)


def gap_indices(n_frames: int, gap_size: int) -> np.ndarray:
    if gap_size <= 0:
        return np.array([], dtype=int)
    all_idx = np.arange(n_frames, dtype=int)
    mid = n_frames // 2
    half = gap_size // 2
    start = max(0, mid - half)
    end = min(n_frames, mid + half)
    return all_idx[start:end]


def pca_fit_transform(arrays: list[np.ndarray], n_components: int = 3) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    stacked = np.vstack([np.asarray(a, dtype=np.float64) for a in arrays])
    finite = np.isfinite(stacked)
    if not finite.all():
        safe = np.where(finite, stacked, np.nan)
        col_means = np.nanmean(safe, axis=0, keepdims=True)
        col_means = np.nan_to_num(col_means, nan=0.0, posinf=0.0, neginf=0.0)
        stacked = np.where(finite, stacked, col_means)
    mean = stacked.mean(axis=0, keepdims=True)
    centered = stacked - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    eigenvalues = (singular_values ** 2) / max(stacked.shape[0] - 1, 1)
    explained = eigenvalues / eigenvalues.sum() if eigenvalues.sum() > 0 else eigenvalues
    projected = [(np.asarray(a, dtype=np.float64) - mean) @ components for a in arrays]
    return projected, explained, np.cumsum(explained)


def pca_scree(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(array, dtype=np.float64) - np.asarray(array, dtype=np.float64).mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singular_values ** 2) / max(centered.shape[0] - 1, 1)
    explained = eigenvalues / eigenvalues.sum() if eigenvalues.sum() > 0 else eigenvalues
    return explained, np.cumsum(explained)


def make_models(args: argparse.Namespace, dataset: str, x: np.ndarray, y: np.ndarray, mask: np.ndarray):
    import torch

    install_optional_dependency_stubs()
    if str(args.workspace) not in sys.path:
        sys.path.insert(0, str(args.workspace))

    from gpvae_endo.config import load_model_settings
    from gpvae_endo.models.vae import VAEModel
    from gpvae_endo.models.gpvae_neighbours import GPVAEHPAModel, GPVAESPAModel

    settings = load_model_settings()

    common = dict(
        x=x,
        Y=y,
        M=mask,
        device=args.device,
        latent_dim=args.latent_dim,
        lr=1e-3,
        img_H=args.resolution,
        img_W=args.resolution,
        img_C=3,
        epochs=1,
        encoder="EndoVAEEnc",
        decoder="EndoVAEDec",
        batch_size=args.batch_size,
    )

    gp_class = GPVAEHPAModel if args.gp_model == "GPVAE_HPA" else GPVAESPAModel

    vae = VAEModel(**common, settings=settings)
    gp_model = gp_class(
        **common,
        kernel_gp_prior="CAUCHY",
        n_neighbors=50,
        underlying_method=args.gp_model,
        settings=settings,
    )

    vae_state = torch.load(
        checkpoint_path(args.saved_models_root, "VAE", dataset, args.checkpoint_suffix),
        map_location=vae.device,
        weights_only=False,
    )
    gp_state = torch.load(
        checkpoint_path(args.saved_models_root, args.gp_model, dataset, args.checkpoint_suffix),
        map_location=gp_model.device,
        weights_only=False,
    )
    normalize_gpytorch_prior_keys(gp_state)
    vae.load_state_dict_full(vae_state)
    gp_model.load_state_dict_full(gp_state)
    vae.eval()
    gp_model.eval()
    return vae, gp_model


def normalize_gpytorch_prior_keys(state: dict) -> None:
    gp_state = state.get("gp_prior")
    if not isinstance(gp_state, dict):
        return
    replacements = {
        "_buffered_loc": "_transformed_loc",
        "_buffered_scale": "_transformed_scale",
    }
    for old, new in replacements.items():
        for key in list(gp_state.keys()):
            if old in key:
                gp_state[key.replace(old, new)] = gp_state.pop(key)


def as_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def uncertainty_sizes(std: np.ndarray, base_size: float, max_multiplier: float) -> np.ndarray:
    std_1d = np.asarray(std).mean(axis=1)
    min_std = float(std_1d.min())
    max_std = float(std_1d.max())
    if max_std <= min_std:
        return np.full(std_1d.shape, base_size)
    max_size = max_multiplier * base_size
    return base_size + (std_1d - min_std) * (max_size - base_size) / (max_std - min_std)


def uncertainty_values(std: np.ndarray) -> np.ndarray:
    return np.asarray(std).mean(axis=1)


def output_model_slug(gp_model: str) -> str:
    if gp_model == "GPVAE_HPA":
        return "hpa"
    return gp_model.lower()


def plot_latents(
    dataset: str,
    gp_model_name: str,
    vae_img: np.ndarray,
    gp_img: np.ndarray,
    gp_time: np.ndarray,
    gp_time_std: np.ndarray,
    output_dir: Path,
    test_mask: np.ndarray | None = None,
    gap_mask: np.ndarray | None = None,
    gap_zoom: bool = False,
    hide_legends: bool = False,
    time_size_max_multiplier: float = 10.0,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    (vae_xyz,), _, _ = pca_fit_transform([vae_img], n_components=3)
    (gp_img_xyz, gp_time_xyz), _, _ = pca_fit_transform([gp_img, gp_time], n_components=3)

    fig = plt.figure(figsize=(18, 8))
    ax_vae = fig.add_subplot(121, projection="3d")
    ax_gp = fig.add_subplot(122, projection="3d")
    ax_gp.computed_zorder = False
    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.06, top=0.78, wspace=0.12)

    image_size = 18.0
    test_size = image_size * 2.0
    test_square_size = test_size * 2.0
    vae_colors = np.arange(vae_xyz.shape[0])
    gp_colors = np.arange(gp_img_xyz.shape[0])
    frame_norm = colors.Normalize(vmin=0, vmax=max(vae_xyz.shape[0], gp_img_xyz.shape[0]) - 1)
    gp_time_sizes = uncertainty_sizes(gp_time_std, image_size, time_size_max_multiplier)
    gp_time_uncertainty = uncertainty_values(gp_time_std)
    uncertainty_cmap = colors.LinearSegmentedColormap.from_list(
        "test_uncertainty_green_yellow_red",
        ["#39a935", "#f5dd33", "#b92318"],
    )
    uncertainty_norm = colors.Normalize(
        vmin=float(np.nanmin(gp_time_uncertainty)),
        vmax=float(np.nanmax(gp_time_uncertainty)),
    )
    if test_mask is None:
        test_mask = np.zeros(vae_xyz.shape[0], dtype=bool)
    else:
        test_mask = np.asarray(test_mask, dtype=bool)
    if gap_mask is None:
        gap_mask = np.zeros(vae_xyz.shape[0], dtype=bool)
    else:
        gap_mask = np.asarray(gap_mask, dtype=bool)
    train_mask = ~test_mask
    uncertainty_scale_values = gp_time_uncertainty[test_mask] if test_mask.any() else gp_time_uncertainty
    uncertainty_vmin = float(np.nanmin(uncertainty_scale_values))
    uncertainty_vmax = float(np.nanmax(uncertainty_scale_values))
    if uncertainty_vmax <= uncertainty_vmin:
        uncertainty_vmax = uncertainty_vmin + 1e-12
    uncertainty_norm = colors.Normalize(vmin=uncertainty_vmin, vmax=uncertainty_vmax)

    sc_vae = ax_vae.scatter(
        vae_xyz[train_mask, 0], vae_xyz[train_mask, 1], vae_xyz[train_mask, 2],
        c=vae_colors[train_mask], cmap="viridis", norm=frame_norm, alpha=0.85, s=image_size,
        zorder=1, label="VAE image encoded train"
    )
    ax_gp.scatter(
        gp_img_xyz[train_mask, 0], gp_img_xyz[train_mask, 1], gp_img_xyz[train_mask, 2],
        c=gp_colors[train_mask], cmap="viridis", norm=frame_norm, alpha=0.85, s=image_size,
        zorder=1, label=f"{gp_model_name} image encoded train"
    )
    ax_gp.scatter(
        gp_time_xyz[train_mask, 0], gp_time_xyz[train_mask, 1], gp_time_xyz[train_mask, 2],
        c=gp_colors[train_mask], cmap="viridis", norm=frame_norm, alpha=0.45, s=gp_time_sizes[train_mask],
        zorder=1, label=f"{gp_model_name} time encoded train"
    )
    if test_mask.any():
        ax_vae.scatter(
            vae_xyz[test_mask, 0], vae_xyz[test_mask, 1], vae_xyz[test_mask, 2],
            c="black", alpha=0.98, s=test_size, edgecolors="black", linewidths=0.35, depthshade=False,
            zorder=5,
            label="VAE image encoded test"
        )
        ax_gp.scatter(
            gp_img_xyz[test_mask, 0], gp_img_xyz[test_mask, 1], gp_img_xyz[test_mask, 2],
            c=gp_time_uncertainty[test_mask], cmap=uncertainty_cmap, norm=uncertainty_norm,
            alpha=0.98, s=test_size, edgecolors="black", linewidths=0.35, depthshade=False,
            zorder=10,
            label=f"{gp_model_name} image encoded test"
        )
        ax_gp.scatter(
            gp_time_xyz[test_mask, 0], gp_time_xyz[test_mask, 1], gp_time_xyz[test_mask, 2],
            c=gp_time_uncertainty[test_mask], cmap=uncertainty_cmap, norm=uncertainty_norm,
            marker="s", alpha=0.98, s=test_square_size,
            edgecolors="black", linewidths=0.35, depthshade=False,
            zorder=30,
            label=f"{gp_model_name} time encoded test"
        )
    cax = fig.add_axes([0.91, 0.18, 0.018, 0.62])
    colorbar = fig.colorbar(sc_vae, cax=cax)
    colorbar.set_label("Frame index", fontsize=COLORBAR_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, length=5, width=1.1)

    for axis in (ax_vae, ax_gp):
        axis.set_xlabel("PC1", fontsize=AXIS_LABEL_FONTSIZE, labelpad=12)
        axis.set_ylabel("PC2", fontsize=AXIS_LABEL_FONTSIZE, labelpad=12)
        axis.set_zlabel("PC3", fontsize=AXIS_LABEL_FONTSIZE, labelpad=12)
        axis.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE, pad=4)
        axis.grid(True, alpha=0.25)
    ax_vae.set_title("VAE", fontsize=PANEL_TITLE_FONTSIZE, pad=12)
    ax_gp.set_title(gp_model_name, fontsize=PANEL_TITLE_FONTSIZE, pad=12)
    if not hide_legends:
        if gap_zoom:
            ax_vae.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
            ax_gp.legend(fontsize=LEGEND_FONTSIZE, loc="lower left")
        else:
            ax_vae.legend(fontsize=LEGEND_FONTSIZE)
            ax_gp.legend(fontsize=LEGEND_FONTSIZE)

    if gap_zoom and gap_mask.any():
        def set_zoom_limits(axis, arrays):
            stacked = np.vstack([array[gap_mask] for array in arrays if array[gap_mask].size > 0])
            mins = stacked.min(axis=0)
            maxs = stacked.max(axis=0)
            span = np.maximum(maxs - mins, 1e-6)
            pad = np.maximum(span * 0.35, 0.05)
            axis.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
            axis.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
            axis.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

        ax_vae_zoom = fig.add_axes([0.07, 0.58, 0.20, 0.28], projection="3d")
        ax_gp_zoom = fig.add_axes([0.55, 0.58, 0.20, 0.28], projection="3d")
        ax_gp_zoom.computed_zorder = False

        ax_vae_zoom.scatter(
            vae_xyz[:, 0], vae_xyz[:, 1], vae_xyz[:, 2],
            c="0.75", alpha=0.18, s=image_size * 0.6, depthshade=False, zorder=1
        )
        ax_vae_zoom.scatter(
            vae_xyz[gap_mask, 0], vae_xyz[gap_mask, 1], vae_xyz[gap_mask, 2],
            c="black", s=test_size, edgecolors="black", linewidths=0.35, depthshade=False, zorder=5
        )

        ax_gp_zoom.scatter(
            gp_img_xyz[:, 0], gp_img_xyz[:, 1], gp_img_xyz[:, 2],
            c="0.75", alpha=0.18, s=image_size * 0.6, depthshade=False, zorder=1
        )
        ax_gp_zoom.scatter(
            gp_img_xyz[gap_mask, 0], gp_img_xyz[gap_mask, 1], gp_img_xyz[gap_mask, 2],
            c=gp_time_uncertainty[gap_mask], cmap=uncertainty_cmap, norm=uncertainty_norm,
            s=test_size, edgecolors="black", linewidths=0.35, depthshade=False, zorder=10
        )
        ax_gp_zoom.scatter(
            gp_time_xyz[gap_mask, 0], gp_time_xyz[gap_mask, 1], gp_time_xyz[gap_mask, 2],
            c=gp_time_uncertainty[gap_mask], cmap=uncertainty_cmap, norm=uncertainty_norm,
            marker="s", alpha=0.98, s=test_square_size,
            edgecolors="black", linewidths=0.35, depthshade=False, zorder=30
        )

        set_zoom_limits(ax_vae_zoom, [vae_xyz])
        set_zoom_limits(ax_gp_zoom, [gp_img_xyz, gp_time_xyz])
        for axis in (ax_vae_zoom, ax_gp_zoom):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_zticks([])
            axis.set_title("gap zoom", fontsize=ZOOM_TITLE_FONTSIZE)
            axis.grid(True, alpha=0.2)

    fig.suptitle(dataset, fontsize=FIGURE_TITLE_FONTSIZE, y=0.97)
    output_dir.mkdir(parents=True, exist_ok=True)
    zoom_suffix = "_gap_zoom" if gap_zoom else ""
    output_path = output_dir / f"{dataset}_vae_image_{output_model_slug(gp_model_name)}_image_time_uncertainty_3d_pca{zoom_suffix}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    if args.dataset.lower() == "all":
        datasets = available_datasets(args.saved_models_root, args.resolution, args.checkpoint_suffix, args.gp_model)
    else:
        datasets = [args.dataset]
    if not datasets:
        raise RuntimeError(f"No datasets found with both VAE and {args.gp_model} checkpoints.")

    print(f"Plotting {len(datasets)} dataset(s) for {args.gp_model}: {', '.join(datasets)}")
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        x, y, mask = load_dataset(args.data_root, dataset, args.resolution)
        test_percent = args.test_percent if args.test_percent is not None else infer_test_percent(args.checkpoint_suffix)
        # Checkpoints with GAP_True were trained with the middle gap held out.
        # The figure can still mark the random TF split plus the gap in red.
        idx_model_train = train_indices(
            x.shape[0],
            0.0 if args.gap_size > 0 else test_percent,
            args.tail_percent,
            args.gap_size,
        )
        idx_display_train = train_indices(x.shape[0], test_percent, args.tail_percent, args.gap_size)
        test_mask = np.ones(x.shape[0], dtype=bool)
        test_mask[idx_display_train] = False
        gap_mask = np.zeros(x.shape[0], dtype=bool)
        gap_mask[gap_indices(x.shape[0], args.gap_size)] = True
        x_train = x[idx_model_train]
        y_train = y[idx_model_train]
        mask_train = mask[idx_model_train] if mask is not None else None
        y_images = y.reshape(y.shape[0], args.resolution, args.resolution, 3)
        print(
            f"Using {idx_model_train.size}/{x.shape[0]} frames as GP training support; "
            f"marking {test_mask.sum()} frames as held-out test points; "
            f"evaluating time encoding on all {x.shape[0]} timestamps"
        )
        vae, gp_model = make_models(args, dataset, x_train, y_train, mask_train)

        print("Encoding VAE images")
        vae_img, _ = vae.image_encode(y_images, batch_size=args.batch_size)
        print(f"Encoding {args.gp_model} images")
        gp_img, _ = gp_model.image_encode(y_images, batch_size=args.batch_size)
        print(f"Encoding {args.gp_model} time")
        gp_time, gp_time_std = gp_model.time_encode(x, use_likelihood=True, return_full_cov=False)

        latent_path = plot_latents(
            dataset,
            args.gp_model,
            vae_img,
            gp_img,
            gp_time,
            gp_time_std,
            args.output_dir,
            test_mask=test_mask,
            gap_mask=gap_mask,
            gap_zoom=args.gap_zoom,
            hide_legends=args.hide_legends,
            time_size_max_multiplier=args.time_size_max_multiplier,
        )
        print(f"Saved {latent_path}")


if __name__ == "__main__":
    main()
