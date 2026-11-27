from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"C:\Endo")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Pics" / "test_percentage_rmse_analysis"
DEFAULT_TEST_PERCENTAGES = (10, 20, 40, 60)
METRIC = "RMSE_te_im"
TIME_METRIC = "RMSE_te_t"
PLOT_NAME = "image_encoded_rmse_vs_test_percentage.png"
RAW_CSV_NAME = "image_encoded_rmse_raw.csv"
AVERAGED_CSV_NAME = "image_encoded_rmse_averaged.csv"
TIME_PLOT_NAME = "time_encoded_rmse_vs_test_percentage.png"
UNCERTAINTY_PLOT_NAME = "gp_time_encoded_test_posterior_uncertainty_vs_test_percentage.png"
COMBINED_PLOT_NAME = "combined_image_time_uncertainty_vs_test_percentage.png"
TIME_RAW_CSV_NAME = "time_encoded_rmse_raw.csv"
TIME_AVERAGED_CSV_NAME = "time_encoded_rmse_averaged.csv"
UNCERTAINTY_RAW_CSV_NAME = "gp_time_encoded_posterior_uncertainty_raw.csv"
UNCERTAINTY_AVERAGED_CSV_NAME = "gp_time_encoded_posterior_uncertainty_averaged.csv"
Y_AXIS_LABEL = "Mean image encoded test RMSE"
TIME_Y_AXIS_LABEL = "Mean time encoded test RMSE"
UNCERTAINTY_METRIC = "posterior_uncertainty"
UNCERTAINTY_Y_AXIS_LABEL = "Mean GP posterior uncertainty\non test points"
TIME_ENCODED_MODELS = {"GPVAE_SPA", "GPVAE_HPA"}
UNCERTAINTY_FILE_RE = re.compile(
    r"^(?P<dataset>\S+)\s+"
    r"(?P<model>GPVAE_(?:HPA|SPA))\s+"
    r"(?P<encoder>\S+)\s+"
    r"(?P<decoder>\S+)\s+.*\s+"
    r"TF_(?P<test_percentage>\d+)\s+uncertainty per frame\.txt$"
)
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
PANEL_TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
LEGEND_HEADER_FONTSIZE = 13
LEGEND_LABEL_FONTSIZE = 12
LEGEND_MARKERSIZE = 7.5
LEGEND_LINEWIDTH = 2.4


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


def summary_path(root: Path, test_percentage: int) -> Path:
    return root / f"TF_{test_percentage} with mask" / "Experiment_Summaries.txt"


def pics_path(root: Path, test_percentage: int) -> Path:
    return root / f"TF_{test_percentage} with mask" / "Pics"


def parse_experiment_summaries(path: Path, test_percentage: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_dataset: str | None = None
    columns: list[str] | None = None
    block_index = 0

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.rstrip()

        if line.startswith("Overview all models for "):
            block_index += 1
            current_dataset = line.split("Overview all models for ", 1)[1].split(" | ", 1)[0].strip()
            columns = None
            continue

        if line.startswith("End time:"):
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

        row = {
            "test_percentage": test_percentage,
            "dataset": current_dataset,
            "block_index": block_index,
        }
        row.update({column: parse_value(value) for column, value in zip(columns, parts)})
        rows.append(row)

    return rows


def parse_uncertainty_file(path: Path, requested_test_percentage: int) -> dict[str, Any] | None:
    match = UNCERTAINTY_FILE_RE.match(path.name)
    if not match:
        return None

    file_test_percentage = int(match.group("test_percentage"))
    if file_test_percentage != requested_test_percentage:
        return None

    test_uncertainties: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("split", "")).strip().lower() != "test":
                continue
            try:
                test_uncertainties.append(float(str(row["uncertainty"]).strip()))
            except (KeyError, TypeError, ValueError):
                continue

    if not test_uncertainties:
        return None

    return {
        "test_percentage": file_test_percentage,
        "dataset": match.group("dataset"),
        "model": match.group("model"),
        "encoder": match.group("encoder"),
        "decoder": match.group("decoder"),
        UNCERTAINTY_METRIC: sum(test_uncertainties) / len(test_uncertainties),
        "n_test_frames": len(test_uncertainties),
        "source_file": str(path),
    }


def parse_uncertainty_files(root: Path, test_percentages: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test_percentage in test_percentages:
        folder = pics_path(root, test_percentage)
        if not folder.exists():
            print(f"WARNING: no Pics folder found for TF_{test_percentage}: {folder}")
            continue

        files = sorted(folder.rglob("*uncertainty per frame.txt"))
        if not files:
            print(f"WARNING: no uncertainty text files found for TF_{test_percentage}: {folder}")
            continue

        for path in files:
            row = parse_uncertainty_file(path, test_percentage)
            if row is not None:
                rows.append(row)

    return rows


def model_label(row: dict[str, Any]) -> str:
    return f"{row['model']} | {row['encoder']} | {row['decoder']}"


def sort_key(row: dict[str, Any]) -> tuple[int, int, str, int]:
    model = str(row["model"])
    encoder = str(row["encoder"])
    model_rank = LEGEND_MODEL_ORDER.index(model) if model in LEGEND_MODEL_ORDER else len(LEGEND_MODEL_ORDER)
    encoder_rank = ENCODER_LEGEND_ORDER.get(encoder, 99)
    return model_rank, encoder_rank, model_label(row), int(row["test_percentage"])


def shade_color(hex_color: str, factor: float) -> str:
    hex_color = hex_color.lstrip("#")
    channels = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
    adjusted = []
    for channel in channels:
        if factor >= 1:
            value = channel + (255 - channel) * (factor - 1)
        else:
            value = channel * factor
        adjusted.append(max(0, min(255, round(value))))
    return "#" + "".join(f"{channel:02x}" for channel in adjusted)


def line_style(row: dict[str, Any]) -> tuple[str, str]:
    base_color = MODEL_BASE_COLORS.get(str(row["model"]), "#555555")
    color = shade_color(base_color, ENCODER_SHADE.get(str(row["encoder"]), 1.0))
    marker = ENCODER_MARKERS.get(str(row["encoder"]), "o")
    return color, marker


def split_model_label(model_label: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in model_label.split("|")]
    if len(parts) != 3:
        return model_label, "", ""
    return parts[0], parts[1], parts[2]


def legend_entry_sort_key(label: str) -> tuple[int, int, str]:
    model, encoder, decoder = split_model_label(label)
    model_rank = LEGEND_MODEL_ORDER.index(model) if model in LEGEND_MODEL_ORDER else len(LEGEND_MODEL_ORDER)
    encoder_rank = ENCODER_LEGEND_ORDER.get(encoder, 99)
    return model_rank, encoder_rank, decoder


def legend_label_for_model_column(model_label: str, model_name: str) -> str:
    prefix = f"{model_name} | "
    return model_label[len(prefix):] if model_label.startswith(prefix) else model_label


def draw_grouped_legend(fig: Any, model_labels: list[str]) -> None:
    import matplotlib.pyplot as plt

    color_map: dict[str, str] = {}
    marker_map: dict[str, str] = {}
    for label in model_labels:
        model_name, encoder_name, _ = split_model_label(label)
        base_color = MODEL_BASE_COLORS.get(model_name, "#555555")
        color_map[label] = shade_color(base_color, ENCODER_SHADE.get(encoder_name, 1.0))
        marker_map[label] = ENCODER_MARKERS.get(encoder_name, "o")

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
            key=legend_entry_sort_key,
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


def average_metric(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    values: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)

    for row in rows:
        metric_value = row.get(metric)
        if not isinstance(metric_value, (float, int)):
            continue

        key = (
            int(row["test_percentage"]),
            str(row["model"]),
            str(row["encoder"]),
            str(row["decoder"]),
        )
        values[key].append(float(metric_value))

    averaged_rows: list[dict[str, Any]] = []
    for (test_percentage, model, encoder, decoder), metric_values in values.items():
        averaged_rows.append(
            {
                "test_percentage": test_percentage,
                "model": model,
                "encoder": encoder,
                "decoder": decoder,
                "model_label": f"{model} | {encoder} | {decoder}",
                f"mean_{metric}": sum(metric_values) / len(metric_values),
                "n_datasets": len(metric_values),
            }
        )

    return sorted(averaged_rows, key=sort_key)


def rows_for_models(rows: list[dict[str, Any]], models: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("model")) in models]


def write_raw_csv(rows: list[dict[str, Any]], output_path: Path, metric: str) -> None:
    fieldnames = ["test_percentage", "dataset", "model", "encoder", "decoder", metric]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row.get(metric) is None:
                continue
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_averaged_csv(rows: list[dict[str, Any]], output_path: Path, metric: str) -> None:
    fieldnames = [
        "test_percentage",
        "model",
        "encoder",
        "decoder",
        "model_label",
        f"mean_{metric}",
        "n_datasets",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def counts_note_for_rows(rows: list[dict[str, Any]]) -> tuple[list[int], str]:
    counts_by_percentage = {
        test_percentage: sorted({int(row["n_datasets"]) for row in rows if row["test_percentage"] == test_percentage})
        for test_percentage in sorted({int(row["test_percentage"]) for row in rows})
    }
    counts_note = ", ".join(
        f"TF_{test_percentage} n={','.join(str(count) for count in counts)}"
        for test_percentage, counts in counts_by_percentage.items()
    )
    return sorted(counts_by_percentage), counts_note


def plot_metric_on_axis(
    ax: Any,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    y_axis_label: str,
) -> None:
    metric_column = f"mean_{metric}"
    by_combo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_combo[str(row["model_label"])].append(row)

    for combo, combo_rows in sorted(by_combo.items(), key=lambda item: sort_key(item[1][0])):
        combo_rows = sorted(combo_rows, key=lambda row: int(row["test_percentage"]))
        color, marker = line_style(combo_rows[0])
        ax.plot(
            [row["test_percentage"] for row in combo_rows],
            [row[metric_column] for row in combo_rows],
            marker=marker,
            color=color,
            linewidth=1.9,
            markersize=5.5,
            label=combo,
        )

    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_xlabel("Test percentage (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(counts_note_for_rows(rows)[0])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.ticklabel_format(axis="y", style="plain")
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_averaged_rmse(
    rows: list[dict[str, Any]],
    output_path: Path,
    metric: str,
    title: str,
    y_axis_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install it or run in the project environment.") from exc

    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=160)
    plot_metric_on_axis(ax, rows, metric, title, y_axis_label)

    model_labels = sorted({str(row["model_label"]) for row in rows}, key=legend_entry_sort_key)
    fig.tight_layout(rect=(0, 0.30, 1, 1))
    draw_grouped_legend(fig, model_labels)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_combined_rmse(
    image_rows: list[dict[str, Any]],
    time_rows: list[dict[str, Any]],
    uncertainty_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install it or run in the project environment.") from exc

    fig, axes = plt.subplots(1, 3, figsize=(20.0, 6.2), dpi=160, sharex=False)
    plot_metric_on_axis(
        axes[0],
        image_rows,
        METRIC,
        "Image-Encoded Test RMSE",
        Y_AXIS_LABEL,
    )
    plot_metric_on_axis(
        axes[1],
        time_rows,
        TIME_METRIC,
        "Time-Encoded Test RMSE",
        TIME_Y_AXIS_LABEL,
    )
    plot_metric_on_axis(
        axes[2],
        uncertainty_rows,
        UNCERTAINTY_METRIC,
        "GP Posterior Uncertainty on Test Points",
        UNCERTAINTY_Y_AXIS_LABEL,
    )

    labels = []
    for ax in axes:
        _, ax_labels = ax.get_legend_handles_labels()
        for label in ax_labels:
            if label not in labels:
                labels.append(label)

    model_labels = sorted(labels, key=legend_entry_sort_key)
    fig.tight_layout(rect=(0, 0.30, 1, 1))
    draw_grouped_legend(fig, model_labels)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def dataset_coverage(rows: list[dict[str, Any]]) -> dict[int, set[str]]:
    coverage: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        coverage[int(row["test_percentage"])].add(str(row["dataset"]))
    return dict(sorted(coverage.items()))


def warn_about_missing_datasets(coverage: dict[int, set[str]]) -> None:
    if not coverage:
        return

    all_datasets = set().union(*coverage.values())
    expected_count = len(all_datasets)
    incomplete = False

    for test_percentage, datasets in coverage.items():
        missing = sorted(all_datasets - datasets)
        if not missing:
            continue

        incomplete = True
        print(
            f"WARNING: TF_{test_percentage} has {len(datasets)} out of {expected_count} dataset sections. "
            f"Missing: {', '.join(missing)}"
        )

    if incomplete:
        print("WARNING: Averages are based on the datasets present in each Experiment_Summaries.txt file.")


def write_analysis(
    rows: list[dict[str, Any]],
    output_dir: Path,
    metric: str,
    raw_csv_name: str,
    averaged_csv_name: str,
    plot_name: str,
    title: str,
    y_axis_label: str,
) -> list[dict[str, Any]]:
    averaged_rows = average_metric(rows, metric)
    raw_csv = output_dir / raw_csv_name
    averaged_csv = output_dir / averaged_csv_name
    plot_path = output_dir / plot_name

    write_raw_csv(rows, raw_csv, metric)
    write_averaged_csv(averaged_rows, averaged_csv, metric)
    plot_averaged_rmse(averaged_rows, plot_path, metric, title, y_axis_label)

    print(f"Wrote raw rows to: {raw_csv}")
    print(f"Wrote averaged rows to: {averaged_csv}")
    print(f"Wrote plot to: {plot_path}")
    return averaged_rows


def write_uncertainty_analysis(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    averaged_rows = average_metric(rows, UNCERTAINTY_METRIC)
    raw_csv = output_dir / UNCERTAINTY_RAW_CSV_NAME
    averaged_csv = output_dir / UNCERTAINTY_AVERAGED_CSV_NAME
    plot_path = output_dir / UNCERTAINTY_PLOT_NAME

    fieldnames = [
        "test_percentage",
        "dataset",
        "model",
        "encoder",
        "decoder",
        UNCERTAINTY_METRIC,
        "n_test_frames",
        "source_file",
    ]
    with raw_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_averaged_csv(averaged_rows, averaged_csv, UNCERTAINTY_METRIC)
    plot_averaged_rmse(
        averaged_rows,
        plot_path,
        UNCERTAINTY_METRIC,
        "Mean GP Posterior Uncertainty on Test Points vs Test Percentage",
        UNCERTAINTY_Y_AXIS_LABEL,
    )

    print(f"Wrote raw uncertainty rows to: {raw_csv}")
    print(f"Wrote averaged uncertainty rows to: {averaged_csv}")
    print(f"Wrote uncertainty plot to: {plot_path}")
    return averaged_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average image-encoded and time-encoded test RMSE by model/encoder/decoder across datasets and plot them against test percentage."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root containing the TF_<percent> with mask folders.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-percentages", type=int, nargs="+", default=list(DEFAULT_TEST_PERCENTAGES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for test_percentage in args.test_percentages:
        path = summary_path(args.root, test_percentage)
        if not path.exists():
            raise FileNotFoundError(path)
        all_rows.extend(parse_experiment_summaries(path, test_percentage))

    coverage = dataset_coverage(all_rows)
    warn_about_missing_datasets(coverage)

    image_averaged_rows = write_analysis(
        rows=all_rows,
        output_dir=args.output_dir,
        metric=METRIC,
        raw_csv_name=RAW_CSV_NAME,
        averaged_csv_name=AVERAGED_CSV_NAME,
        plot_name=PLOT_NAME,
        title="Mean Image-Encoded Test RMSE vs Test Percentage",
        y_axis_label=Y_AXIS_LABEL,
    )

    time_rows = rows_for_models(all_rows, TIME_ENCODED_MODELS)
    time_averaged_rows = write_analysis(
        rows=time_rows,
        output_dir=args.output_dir,
        metric=TIME_METRIC,
        raw_csv_name=TIME_RAW_CSV_NAME,
        averaged_csv_name=TIME_AVERAGED_CSV_NAME,
        plot_name=TIME_PLOT_NAME,
        title="Mean Time-Encoded Test RMSE vs Test Percentage",
        y_axis_label=TIME_Y_AXIS_LABEL,
    )

    uncertainty_rows = parse_uncertainty_files(args.root, args.test_percentages)
    uncertainty_averaged_rows = write_uncertainty_analysis(uncertainty_rows, args.output_dir)

    combined_plot_path = args.output_dir / COMBINED_PLOT_NAME
    plot_combined_rmse(image_averaged_rows, time_averaged_rows, uncertainty_averaged_rows, combined_plot_path)
    print(f"Wrote combined plot to: {combined_plot_path}")

    datasets_by_percentage = {test_percentage: len(datasets) for test_percentage, datasets in coverage.items()}

    print(f"Dataset sections per test percentage: {datasets_by_percentage}")


if __name__ == "__main__":
    main()
