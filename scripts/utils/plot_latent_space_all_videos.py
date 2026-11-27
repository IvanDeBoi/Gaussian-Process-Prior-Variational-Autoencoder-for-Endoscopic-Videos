from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WORKSPACE = Path(r"C:\Users\Ivan\OneDrive - Universiteit Antwerpen\01 Postdoc\04 Exp\15 GPVAE Endo")
SAVED_MODELS_ROOT = Path(r"C:\Endo\Full train with mask\Saved_Models")
CHECKPOINT_SUFFIX = "256D L_True_256 M_True G_False GAP_False_0 TF_0.pth"
DATA_SUFFIX = "Ym01sGM.npz"

DATASET_ORDER = [
    "c1_ascending_t1_v1",
    "c1_descending_t1_v2",
    "c1_descending_t2_v3",
    "c1_rectum_t2_v2",
    "c1_rectum_t4_v3",
    "c1_transverse2_t1_v2",
    "c2_cecum_t3_v2",
    "c2_sigmoid_t2_v2",
    "c2_transverse2_t1_v3",
]

DEFAULT_MODELS = ["VAE", "GPVAE_HPA"]
DEFAULT_ENCODERS = ["EndoVAEEnc"]
DEFAULT_DECODER = "EndoVAEDec"


@dataclass(frozen=True)
class Spec:
    model: str
    encoder: str
    decoder: str

    @property
    def label(self) -> str:
        return f"{self.model} {self.encoder} {self.decoder}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot stored model.Z latent trajectories for every configured C3VDv2 video "
            "using existing trained checkpoints only. Produces one plot per dataset."
        )
    )
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--saved-models-root", type=Path, default=SAVED_MODELS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "Pics" / "latent_space_all_videos")
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--checkpoint-suffix", default=CHECKPOINT_SUFFIX)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    parser.add_argument("--decoder", default=DEFAULT_DECODER)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested non-MSVQ checkpoint is missing.",
    )
    return parser.parse_args()


def configured_datasets(config_dir: Path) -> list[str]:
    datasets = []
    for config_path in sorted(config_dir.glob("config_*.py")):
        text = config_path.read_text(encoding="utf-8")
        match = re.search(r"^SUBSTRING_FILTER\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
        if match:
            datasets.append(match.group(1))

    ordered = [dataset for dataset in DATASET_ORDER if dataset in datasets]
    ordered.extend(dataset for dataset in datasets if dataset not in ordered)
    return ordered


def requested_specs(models: list[str], encoders: list[str], decoder: str) -> list[Spec]:
    specs = []
    for model in models:
        if model == "MSVQ_VAE":
            continue
        for encoder in encoders:
            if encoder == "MSVQEnc":
                continue
            specs.append(Spec(model=model, encoder=encoder, decoder=decoder))
    return specs


def find_checkpoint(root: Path, dataset: str, spec: Spec, suffix: str) -> Path | None:
    hits = []
    for path in root.glob(f"{spec.model} {dataset} *{suffix}"):
        name = path.name
        if f" {spec.encoder} {spec.decoder} " in name:
            hits.append(path)
    if len(hits) > 1:
        raise RuntimeError(f"Ambiguous checkpoints for {dataset} {spec.label}: {hits}")
    return hits[0] if hits else None


def inventory_checkpoints(
    datasets: list[str],
    specs: list[Spec],
    root: Path,
    suffix: str,
) -> tuple[dict[str, dict[Spec, Path]], list[tuple[str, Spec]]]:
    present: dict[str, dict[Spec, Path]] = {}
    missing: list[tuple[str, Spec]] = []

    for dataset in datasets:
        present[dataset] = {}
        for spec in specs:
            checkpoint = find_checkpoint(root, dataset, spec, suffix)
            if checkpoint is None:
                missing.append((dataset, spec))
            else:
                present[dataset][spec] = checkpoint
    return present, missing


def write_inventory_report(
    output_dir: Path,
    datasets: list[str],
    specs: list[Spec],
    present: dict[str, dict[Spec, Path]],
    missing: list[tuple[str, Spec]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "checkpoint_inventory.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "model", "encoder", "decoder", "status", "checkpoint"])
        for dataset in datasets:
            for spec in specs:
                checkpoint = present[dataset].get(spec)
                writer.writerow(
                    [
                        dataset,
                        spec.model,
                        spec.encoder,
                        spec.decoder,
                        "present" if checkpoint else "missing",
                        str(checkpoint) if checkpoint else "",
                    ]
                )

    print(f"Checkpoint inventory: {report_path}")
    for dataset in datasets:
        n_present = len(present[dataset])
        print(f"{dataset}: {n_present}/{len(specs)} requested checkpoints present")
    if missing:
        print("Missing checkpoints:")
        for dataset, spec in missing:
            print(f"  {dataset}: {spec.label}")


def pca_single(array: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(array, dtype=np.float64)
    centered = z - z.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    projected = centered @ components
    eigenvalues = (singular_values ** 2) / max(z.shape[0] - 1, 1)
    explained = eigenvalues / eigenvalues.sum() if eigenvalues.sum() > 0 else eigenvalues
    return projected, explained, np.cumsum(explained)


def plot_dataset(
    dataset: str,
    curves: list[tuple[Spec, np.ndarray]],
    output_dir: Path,
) -> Path:
    import matplotlib.pyplot as plt

    projected = [(spec, *pca_single(z, n_components=3)) for spec, z in curves]
    fig = plt.figure(figsize=(14, 7))
    axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]

    for axis, (spec, xy, _, _) in zip(axes, projected):
        axis.plot(xy[:, 0], xy[:, 1], xy[:, 2], linewidth=1.7)
        axis.scatter(xy[0, 0], xy[0, 1], xy[0, 2], marker="o", s=30, label="start")
        axis.scatter(xy[-1, 0], xy[-1, 1], xy[-1, 2], marker="x", s=46, label="end")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.set_zlabel("PC3")
        axis.grid(True, alpha=0.25)
        axis.text2D(0.03, 0.95, spec.model, transform=axis.transAxes)

    fig.suptitle(dataset)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_latent_trajectories_model_Z_3d_pca_side_by_side.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    scree_path = output_dir / f"{dataset}_latent_scree_model_Z.png"
    plot_scree(dataset, projected, scree_path)
    return output_path


def plot_scree(dataset: str, projected: list[tuple[Spec, np.ndarray, np.ndarray, np.ndarray]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, (spec, _, explained, cumulative) in zip(axes, projected):
        n = min(20, explained.shape[0])
        xs = np.arange(1, n + 1)
        axis.bar(xs, explained[:n], alpha=0.75, label="Explained")
        axis.plot(xs, cumulative[:n], "-o", color="black", markersize=3, label="Cumulative")
        axis.set_title(spec.model)
        axis.set_xlabel("Principal component")
        axis.set_ylabel("Explained variance ratio")
        axis.set_ylim(0, 1.02)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(dataset)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def checkpoint_z(checkpoint: Path) -> np.ndarray:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "Z" not in state or state["Z"] is None:
        raise KeyError(f"Checkpoint has no stored Z: {checkpoint}")
    z = state["Z"]
    if hasattr(z, "detach"):
        z = z.detach().cpu().numpy()
    return np.asarray(z)


def run_plotting(args: argparse.Namespace, datasets: list[str], present: dict[str, dict[Spec, Path]]) -> None:
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        curves = []

        for spec, checkpoint in present[dataset].items():
            print(f"Loading Z from {spec.label}")
            curves.append((spec, checkpoint_z(checkpoint)))

        if not curves:
            print(f"No curves available for {dataset}; skipping plot")
            continue
        output_path = plot_dataset(dataset, curves, args.output_dir)
        print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    datasets = configured_datasets(args.workspace / "configs")
    specs = requested_specs(args.models, args.encoders, args.decoder)
    present, missing = inventory_checkpoints(datasets, specs, args.saved_models_root, args.checkpoint_suffix)
    write_inventory_report(args.output_dir, datasets, specs, present, missing)

    if args.strict and missing:
        raise SystemExit("Missing requested checkpoints; rerun without --strict to plot available curves only.")
    if args.check_only:
        return

    run_plotting(args, datasets, present)


if __name__ == "__main__":
    main()
