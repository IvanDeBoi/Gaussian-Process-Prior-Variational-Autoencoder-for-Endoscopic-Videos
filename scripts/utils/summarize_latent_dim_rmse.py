from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"C:\Endo")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Pics"
LATENT_DIMS = (8, 16, 32, 64, 128, 256)
EXTRA_SUMMARY_FILES = {
    256: DEFAULT_ROOT / "Full train with mask" / "Experiment_Summaries.txt",
}
METRIC = "RMSE_tr_im"
TIME_METRIC = "seconds_per_epoch"
MIN_LATENT_DIM_POINTS = 2
IGNORED_ENCODERS = {"CNNEnc"}
IGNORED_DECODERS = {"CNNDec", "CNNDecRes"}
GRID_PLOT_NAME = "rmse_tr_im_by_latent_dim_by_dataset_grid.png"
TIME_GRID_PLOT_NAME = "seconds_per_epoch_by_latent_dim_by_dataset_grid.png"
TIME_PLOT_NAME = "seconds_per_epoch_by_latent_dim.png"
Y_AXIS_LABEL = "RMSE training images"
TIME_Y_AXIS_LABEL = "Time per epoch [s]"
LEGEND_MODEL_ORDER = ("VAE", "GPVAE_SPA", "GPVAE_HPA", "MSVQ_VAE")
MODEL_BASE_COLORS = {
    "VAE": "#d62728",
    "GPVAE_SPA": "#2ca02c",
    "GPVAE_HPA": "#1f77b4",
    "MSVQ_VAE": "#7f3c8d",
}
ENCODER_MARKERS = {
    "EndoVAEEnc": "o",
    "GastroEnc": "s",
    "GaLoRaEnc": "^",
    "MSVQEnc": "X",
}
ENCODER_SHADE = {
    "EndoVAEEnc": 0.95,
    "GastroEnc": 0.72,
    "GaLoRaEnc": 1.18,
    "MSVQEnc": 1.0,
}
ENCODER_LEGEND_ORDER = {
    "EndoVAEEnc": 0,
    "GastroEnc": 1,
    "MSVQEnc": 2,
    "GaLoRaEnc": 3,
}
DATASET_ORDER = (
    "c1_ascending_t1_v1",
    "c1_descending_t1_v2",
    "c1_descending_t2_v3",
    "c1_rectum_t2_v2",
    "c1_rectum_t4_v3",
    "c1_transverse2_t1_v2",
    "c2_cecum_t3_v2",
    "c2_sigmoid_t2_v2",
    "c2_transverse2_t1_v3",
)

GRID_TITLE_FONTSIZE = 17
GRID_AXIS_LABEL_FONTSIZE = 16
GRID_TICK_LABEL_FONTSIZE = 14
GRID_SCI_LABEL_FONTSIZE = 13
GRID_LINEWIDTH = 2.0
GRID_MARKERSIZE = 7.0
GRID_MARKEREDGEWIDTH = 1.2
LEGEND_HEADER_FONTSIZE = 16
LEGEND_LABEL_FONTSIZE = 14
LEGEND_LINEWIDTH = 2.7
LEGEND_MARKERSIZE = 8.5
RMSE_SCIENTIFIC_SCALE = 1e-3
RMSE_SCIENTIFIC_LABEL = r"$x10^{-3}$"


def parse_value(value: str) -> float | int | None | str:
    value = value.strip()
    if value.lower() == "n/a":
        return None

    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def clean_column_name(name: str) -> str:
    return name.strip().replace("t(h:m:s)", "time_hms").replace("t/epoch(s)", "seconds_per_epoch")


def latent_dim_from_folder(folder: Path) -> int | None:
    match = re.match(r"^latent\s+dim\s+(\d+)$", folder.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def latent_dim_from_overview(line: str) -> int | None:
    match = re.search(r"\blatent\s+dim\s*=\s*(\d+)\b", line, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def end_time_from_line(line: str) -> datetime | None:
    match = re.search(r"End time:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    if not match:
        return None
    return datetime.fromisoformat(match.group(1))


def discover_summary_files(root: Path, latent_dims: tuple[int, ...]) -> dict[int, Path]:
    summaries: dict[int, Path] = {}

    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        latent_dim = latent_dim_from_folder(folder)
        if latent_dim not in latent_dims:
            continue

        summary_file = folder / "Experiment_Summaries.txt"
        if summary_file.exists():
            summaries[latent_dim] = summary_file

    for latent_dim, summary_file in EXTRA_SUMMARY_FILES.items():
        if latent_dim in latent_dims and summary_file.exists():
            summaries[latent_dim] = summary_file

    missing = sorted(set(latent_dims) - set(summaries))
    if missing:
        missing_text = ", ".join(str(dim) for dim in missing)
        raise FileNotFoundError(f"Missing Experiment_Summaries.txt for latent dim(s): {missing_text}")

    return dict(sorted(summaries.items()))


def parse_experiment_summaries(path: Path, latent_dim: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_dataset: str | None = None
    columns: list[str] | None = None
    block_rows: list[dict[str, Any]] = []
    block_index = 0

    def flush_block(end_time: datetime | None = None) -> None:
        nonlocal block_rows
        for block_row in block_rows:
            block_row["end_time"] = end_time
            rows.append(block_row)
        block_rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()

        if line.startswith("Overview all models for "):
            flush_block()
            block_index += 1
            current_latent_dim = latent_dim_from_overview(line)
            current_dataset = (
                line.split("Overview all models for ", 1)[1].split(" | ", 1)[0].strip()
                if current_latent_dim == latent_dim
                else None
            )
            columns = None
            continue

        if line.startswith("End time:"):
            flush_block(end_time_from_line(line))
            current_dataset = None
            columns = None
            continue

        if line.startswith("model") and "|" in line:
            columns = [clean_column_name(part) for part in line.split("|")]
            continue

        if not columns or not current_dataset or "|" not in line:
            continue

        if set(line.strip()) <= {"-"}:
            continue

        parts = [part.strip() for part in line.split("|")]
        if len(parts) != len(columns):
            continue

        row = {"dataset": current_dataset, "latent_dim": latent_dim, "block_index": block_index}
        row.update({column: parse_value(value) for column, value in zip(columns, parts)})
        block_rows.append(row)

    flush_block()
    return rows


def model_label(row: dict[str, Any]) -> str:
    return f"{row['model']} | {row['encoder']} | {row['decoder']}"


def should_ignore_model(row: dict[str, Any]) -> bool:
    return str(row.get("encoder")) in IGNORED_ENCODERS or str(row.get("decoder")) in IGNORED_DECODERS


def is_more_recent(candidate: dict[str, Any], incumbent_end_time: Any, incumbent_block_index: Any) -> bool:
    candidate_end_time = candidate.get("end_time")
    if isinstance(candidate_end_time, datetime) and isinstance(incumbent_end_time, datetime):
        return candidate_end_time > incumbent_end_time
    if isinstance(candidate_end_time, datetime):
        return True
    if isinstance(incumbent_end_time, datetime):
        return False
    return int(candidate.get("block_index", 0)) > int(incumbent_block_index or 0)


def pivot_metric(rows: list[dict[str, Any]], latent_dims: tuple[int, ...], metric: str) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for row in rows:
        if should_ignore_model(row):
            continue

        key = (str(row["dataset"]), str(row["model"]), str(row["encoder"]), str(row["decoder"]))
        output_row = keyed.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "encoder": row["encoder"],
                "decoder": row["decoder"],
                "model_label": model_label(row),
            },
        )
        metric_column = f"{metric}_dim{row['latent_dim']}"
        end_time_column = f"end_time_dim{row['latent_dim']}"
        block_index_column = f"block_index_dim{row['latent_dim']}"
        if metric_column not in output_row or is_more_recent(
            row,
            output_row.get(end_time_column),
            output_row.get(block_index_column),
        ):
            output_row[metric_column] = row.get(metric)
            output_row[end_time_column] = row.get("end_time")
            output_row[block_index_column] = row.get("block_index")

    for output_row in keyed.values():
        for latent_dim in latent_dims:
            output_row.setdefault(f"{metric}_dim{latent_dim}", None)

    return sorted(keyed.values(), key=lambda row: (row["dataset"], row["model_label"]))


def write_csv(rows: list[dict[str, Any]], path: Path, latent_dims: tuple[int, ...], metric: str) -> None:
    fieldnames = [
        "dataset",
        "model",
        "encoder",
        "decoder",
        "model_label",
        *[f"{metric}_dim{latent_dim}" for latent_dim in latent_dims],
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def count_metric_points(row: dict[str, Any], latent_dims: tuple[int, ...], metric: str) -> int:
    return sum(isinstance(row[f"{metric}_dim{latent_dim}"], (int, float)) for latent_dim in latent_dims)


def filter_sparse_rows(
    rows: list[dict[str, Any]],
    latent_dims: tuple[int, ...],
    min_points: int,
    metric: str,
) -> list[dict[str, Any]]:
    return [row for row in rows if count_metric_points(row, latent_dims, metric) >= min_points]


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def ordered_datasets(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row["dataset"]) for row in rows}
    ordered = [dataset for dataset in DATASET_ORDER if dataset in present]
    ordered.extend(dataset for dataset in ordered_unique([str(row["dataset"]) for row in rows]) if dataset not in ordered)
    return ordered


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "dataset"


def split_model_label(model_label: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in model_label.split("|")]
    if len(parts) != 3:
        return model_label, "", ""
    return parts[0], parts[1], parts[2]


def adjust_color(hex_color: str, factor: float) -> str:
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
    adjusted = []
    for channel in rgb:
        if factor >= 1:
            value = channel + (255 - channel) * (factor - 1)
        else:
            value = channel * factor
        adjusted.append(max(0, min(255, round(value))))
    return f"#{adjusted[0]:02x}{adjusted[1]:02x}{adjusted[2]:02x}"


def style_maps(rows: list[dict[str, Any]], plt: Any) -> tuple[dict[str, Any], dict[str, str]]:
    model_labels = sorted({str(row["model_label"]) for row in rows})
    fallback_colors = plt.cm.tab20.colors
    fallback_markers = ["o", "x", "^", "s", "D", "v", "P", "*", "h", "+", "<", ">"]
    color_map = {}
    marker_map = {}

    for idx, label in enumerate(model_labels):
        model_name, encoder_name, _ = split_model_label(label)
        if model_name in MODEL_BASE_COLORS:
            color_map[label] = adjust_color(MODEL_BASE_COLORS[model_name], ENCODER_SHADE.get(encoder_name, 1.0))
        else:
            color_map[label] = fallback_colors[idx % len(fallback_colors)]
        marker_map[label] = ENCODER_MARKERS.get(encoder_name, fallback_markers[idx % len(fallback_markers)])

    return color_map, marker_map


def legend_label_for_model_column(model_label: str, model_name: str) -> str:
    prefix = f"{model_name} | "
    return model_label.removeprefix(prefix)


def legend_sort_key(model_label: str) -> tuple[int, str]:
    _, encoder_name, decoder_name = split_model_label(model_label)
    return ENCODER_LEGEND_ORDER.get(encoder_name, 99), decoder_name


def draw_grouped_legend(
    fig: Any,
    model_labels: list[str],
    color_map: dict[str, Any],
    marker_map: dict[str, str],
) -> None:
    import matplotlib.pyplot as plt

    legend_ax = fig.add_axes([0.06, 0.02, 0.88, 0.20])
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(0.0, 1.0)
    legend_ax.axis("off")

    legend_frame = plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="white", edgecolor="0.82", linewidth=0.8)
    legend_ax.add_patch(legend_frame)

    column_width = 1.0 / len(LEGEND_MODEL_ORDER)
    for col_idx, model_name in enumerate(LEGEND_MODEL_ORDER):
        x0 = col_idx * column_width + 0.02
        group_labels = sorted(
            [label for label in model_labels if label.startswith(f"{model_name} |")],
            key=legend_sort_key,
        )
        legend_ax.text(x0, 0.84, model_name, fontsize=LEGEND_HEADER_FONTSIZE, ha="left", va="center")

        for row_idx, label in enumerate(group_labels):
            y = 0.62 - row_idx * 0.22
            legend_ax.plot(
                [x0, x0 + 0.03],
                [y, y],
                color=color_map[label],
                marker=marker_map[label],
                linewidth=LEGEND_LINEWIDTH,
                markersize=LEGEND_MARKERSIZE,
                clip_on=False,
            )
            legend_ax.text(
                x0 + 0.04,
                y,
                legend_label_for_model_column(label, model_name),
                fontsize=LEGEND_LABEL_FONTSIZE,
                ha="left",
                va="center",
            )


def apply_rmse_scientific_axis(ax: Any) -> None:
    import matplotlib.ticker as mticker

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{value / RMSE_SCIENTIFIC_SCALE:g}")
    )
    ax.text(
        0.0,
        1.015,
        RMSE_SCIENTIFIC_LABEL,
        transform=ax.transAxes,
        fontsize=GRID_SCI_LABEL_FONTSIZE,
        ha="left",
        va="bottom",
    )


def plot_curves(
    rows: list[dict[str, Any]],
    path: Path,
    latent_dims: tuple[int, ...],
    metric: str,
    y_axis_label: str,
    title: str,
    alpha: float = 0.45,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.0, 7.0), constrained_layout=True)
    model_labels = sorted({str(row["model_label"]) for row in rows})
    color_map, marker_map = style_maps(rows, plt)

    for row in rows:
        values = [
            float(row[f"{metric}_dim{latent_dim}"])
            if isinstance(row[f"{metric}_dim{latent_dim}"], (int, float))
            else math.nan
            for latent_dim in latent_dims
        ]
        ax.plot(
            latent_dims,
            values,
            marker=marker_map[str(row["model_label"])],
            linewidth=1.2,
            markersize=5.0,
            markeredgewidth=1.0,
            alpha=alpha,
            color=color_map[str(row["model_label"])],
        )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=color_map[label],
            marker=marker_map[label],
            linewidth=1.8,
            markersize=5.5,
            label=label,
        )
        for label in model_labels
    ]
    ax.legend(
        handles=legend_handles,
        title="Model | encoder | decoder",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
    )

    ax.set_title(title)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(y_axis_label)
    ax.set_xticks(latent_dims)
    ax.grid(True, axis="both", alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_dataset_grid(
    rows: list[dict[str, Any]],
    path: Path,
    latent_dims: tuple[int, ...],
    metric: str,
    y_axis_label: str,
) -> Path:
    import matplotlib.pyplot as plt

    datasets = ordered_datasets(rows)
    color_map, marker_map = style_maps(rows, plt)
    fig, axes = plt.subplots(3, 3, figsize=(18.0, 13.0), sharex=True, sharey=False, constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.955, bottom=0.30, wspace=0.24, hspace=0.42)
    flat_axes = list(axes.flat)

    for ax, dataset in zip(flat_axes, datasets):
        dataset_rows = [row for row in rows if str(row["dataset"]) == dataset]
        for row in dataset_rows:
            values = [
                float(row[f"{metric}_dim{latent_dim}"])
                if isinstance(row[f"{metric}_dim{latent_dim}"], (int, float))
                else math.nan
                for latent_dim in latent_dims
            ]
            label = str(row["model_label"])
            ax.plot(
                latent_dims,
                values,
                marker=marker_map[label],
                linewidth=GRID_LINEWIDTH,
                markersize=GRID_MARKERSIZE,
                markeredgewidth=GRID_MARKEREDGEWIDTH,
                alpha=0.9,
                color=color_map[label],
            )

        ax.set_title(dataset, fontsize=GRID_TITLE_FONTSIZE, pad=10)
        ax.set_xticks(latent_dims)
        ax.tick_params(axis="both", labelsize=GRID_TICK_LABEL_FONTSIZE, length=6, width=1.2)
        ax.tick_params(axis="x", labelbottom=True)
        if metric == METRIC:
            apply_rmse_scientific_axis(ax)
        ax.grid(True, axis="both", alpha=0.25)

    for ax in flat_axes[len(datasets) :]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Latent dimension", fontsize=GRID_AXIS_LABEL_FONTSIZE)
    for ax in axes[:, 0]:
        ax.set_ylabel(y_axis_label, fontsize=GRID_AXIS_LABEL_FONTSIZE)

    model_labels = sorted({str(row["model_label"]) for row in rows})
    draw_grouped_legend(fig, model_labels, color_map, marker_map)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def build_outputs(root: Path, output_dir: Path, latent_dims: tuple[int, ...]) -> tuple[Path, Path, Path, Path, Path, int]:
    summary_files = discover_summary_files(root, latent_dims)
    all_rows = [
        row
        for latent_dim, summary_file in summary_files.items()
        for row in parse_experiment_summaries(summary_file, latent_dim)
    ]
    pivoted_rows = pivot_metric(all_rows, latent_dims, METRIC)
    pivoted_rows = filter_sparse_rows(pivoted_rows, latent_dims, MIN_LATENT_DIM_POINTS, METRIC)
    time_rows = pivot_metric(all_rows, latent_dims, TIME_METRIC)
    time_rows = filter_sparse_rows(time_rows, latent_dims, MIN_LATENT_DIM_POINTS, TIME_METRIC)

    csv_path = output_dir / "rmse_tr_im_by_latent_dim.csv"
    plot_path = output_dir / "rmse_tr_im_by_latent_dim.png"
    grid_plot_path = output_dir / GRID_PLOT_NAME
    time_plot_path = output_dir / TIME_PLOT_NAME
    time_grid_plot_path = output_dir / TIME_GRID_PLOT_NAME
    write_csv(pivoted_rows, csv_path, latent_dims, METRIC)
    plot_curves(
        pivoted_rows,
        plot_path,
        latent_dims,
        metric=METRIC,
        y_axis_label=Y_AXIS_LABEL,
        title=f"{METRIC} curves by latent dimension",
    )
    plot_dataset_grid(pivoted_rows, grid_plot_path, latent_dims, metric=METRIC, y_axis_label=Y_AXIS_LABEL)
    plot_curves(
        time_rows,
        time_plot_path,
        latent_dims,
        metric=TIME_METRIC,
        y_axis_label=TIME_Y_AXIS_LABEL,
        title="Time per epoch by latent dimension",
    )
    plot_dataset_grid(
        time_rows,
        time_grid_plot_path,
        latent_dims,
        metric=TIME_METRIC,
        y_axis_label=TIME_Y_AXIS_LABEL,
    )
    return csv_path, plot_path, grid_plot_path, time_plot_path, time_grid_plot_path, len(pivoted_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect RMSE_tr_im by dataset/model for latent dims 8, 16, 32, 64, 128, and 256."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root folder containing Latent dim subfolders.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for the CSV and PNG plot.")
    args = parser.parse_args()

    csv_path, plot_path, grid_plot_path, time_plot_path, time_grid_plot_path, n_rows = build_outputs(
        args.root,
        args.output_dir,
        LATENT_DIMS,
    )
    print(f"Wrote {n_rows} rows to {csv_path}")
    print(f"Wrote plot to {plot_path}")
    print(f"Wrote 3x3 dataset grid plot to {grid_plot_path}")
    print(f"Wrote time-per-epoch plot to {time_plot_path}")
    print(f"Wrote time-per-epoch 3x3 dataset grid plot to {time_grid_plot_path}")


if __name__ == "__main__":
    main()
