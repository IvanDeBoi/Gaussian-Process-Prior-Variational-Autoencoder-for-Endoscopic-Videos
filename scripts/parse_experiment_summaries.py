from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


MENTIONED_SUMMARY_FILE = Path(r"C:\Endo\Full train with mask\Experiment_Summaries.txt")
LOCAL_SUMMARY_FILE = Path(__file__).with_name("Experiment_Summaries.txt")
DEFAULT_SUMMARY_FILE = MENTIONED_SUMMARY_FILE if MENTIONED_SUMMARY_FILE.exists() else LOCAL_SUMMARY_FILE

METRIC_COLUMNS = [
    "RMSE_tr_im",
    "RMSE_tr_t",
    "FID_tr_im",
    "FID_tr_t",
    "SSIM_tr_im",
    "SSIM_tr_t",
    "PSNR_tr_im",
    "PSNR_tr_t",
    "epochs",
]


def parse_value(value: str) -> float | int | None | str:
    value = value.strip()
    if value.lower() == "n/a":
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def clean_column_name(name: str) -> str:
    return name.strip().replace("t(h:m:s)", "time_hms").replace("t/epoch(s)", "seconds_per_epoch")


def format_metric(value: float | int | None, width: int = 14) -> str:
    if value is None:
        return f"{'n/a':>{width}s}"
    return f"{value:{width}.6f}"


def format_mean_std(mean: float | int | None, std: float | int | None, width: int = 20) -> str:
    if mean is None or std is None:
        return f"{'n/a':>{width}s}"
    return f"{mean:.6f} +/- {std:.6f}".rjust(width)


def parse_experiment_summaries(path: str | Path = DEFAULT_SUMMARY_FILE) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    current_experiment: str | None = None
    columns: list[str] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()

        if line.startswith("Overview all models for "):
            current_experiment = line.split("Overview all models for ", 1)[1].split(" | ", 1)[0].strip()
            columns = None
            continue

        if line.startswith("model") and "|" in line:
            columns = [clean_column_name(part) for part in line.split("|")]
            continue

        if not columns or not current_experiment or "|" not in line:
            continue

        if set(line.strip()) <= {"-"}:
            continue

        parts = [part.strip() for part in line.split("|")]
        if len(parts) != len(columns):
            continue

        row = {"experiment": current_experiment}
        row.update({column: parse_value(value) for column, value in zip(columns, parts)})
        rows.append(row)

    return rows


def metrics_as_variables(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {column: [row.get(column) for row in rows] for column in METRIC_COLUMNS}


GROUPED_METRICS = [
    "RMSE_tr_im",
    "RMSE_tr_t",
    "FID_tr_im",
    "FID_tr_t",
    "SSIM_tr_im",
    "SSIM_tr_t",
    "PSNR_tr_im",
    "PSNR_tr_t",
]
GROUPED_TIME_COLUMNS = ["seconds_per_epoch"]
IMAGE_TABLE_COLUMNS = [
    ("model", 12, ""),
    ("encoder", 10, ""),
    ("decoder", 10, ""),
    ("RMSE", 20, "[0,1]"),
    ("PSNR", 20, "dB"),
    ("SSIM", 20, "unitless"),
    ("FID", 20, "unitless"),
    ("Time/epoch", 20, "s"),
]
TIME_TABLE_COLUMNS = [
    ("model", 12, ""),
    ("encoder", 10, ""),
    ("decoder", 10, ""),
    ("RMSE", 20, "[0,1]"),
    ("PSNR", 20, "dB"),
    ("SSIM", 20, "unitless"),
    ("FID", 20, "unitless"),
]


def summarize_metrics_by_combo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    model_order = {"VAE": 0}
    encoder_order = {"EndoVAEEnc": 0, "GastroEnc": 1, "GaLoRaEnc": 2, "MSVQEnc": 3}

    for row in rows:
        key = (row["model"], row["encoder"], row["decoder"])
        grouped.setdefault(key, {metric: [] for metric in GROUPED_METRICS + GROUPED_TIME_COLUMNS})

        for metric in GROUPED_METRICS + GROUPED_TIME_COLUMNS:
            value = row.get(metric)
            if isinstance(value, (int, float)):
                grouped[key][metric].append(float(value))

    summaries = []
    for (model_name, encoder_name, decoder_name), values_by_metric in grouped.items():
        summary = {
            "model": model_name,
            "encoder": encoder_name,
            "decoder": decoder_name,
            "n_datasets": len(values_by_metric["RMSE_tr_im"]),
        }
        for metric in GROUPED_METRICS:
            values = values_by_metric[metric]
            summary[f"avg_{metric}"] = sum(values) / len(values) if values else None
            summary[f"std_{metric}"] = statistics.pstdev(values) if values else None

        seconds_per_epoch_values = values_by_metric["seconds_per_epoch"]
        summary["avg_t_epoch_s"] = (
            sum(seconds_per_epoch_values) / len(seconds_per_epoch_values)
            if seconds_per_epoch_values
            else None
        )
        summary["std_t_epoch_s"] = statistics.pstdev(seconds_per_epoch_values) if seconds_per_epoch_values else None

        summaries.append(summary)

    return sorted(
        summaries,
        key=lambda row: (
            encoder_order.get(row["encoder"], 99),
            row["encoder"],
            row["decoder"],
            model_order.get(row["model"], 1),
            row["model"],
        ),
    )


def average_rmse_by_combo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize_metrics_by_combo(rows)


def format_table_row(values: list[str], widths: list[int]) -> str:
    return "  " + " | ".join(f"{value:>{width}s}" for value, width in zip(values, widths))


def print_table_header(columns: list[tuple[str, int, str]]) -> None:
    widths = [width for _, width, _ in columns]
    print(format_table_row([name for name, _, _ in columns], widths))
    print(format_table_row([unit for _, _, unit in columns], widths))
    print("  " + "-" * (sum(widths) + 3 * (len(widths) - 1)))


records = parse_experiment_summaries()
metrics = metrics_as_variables(records)
metrics_by_combo = summarize_metrics_by_combo(records)
rmse_by_combo = metrics_by_combo
rmse_tr_im_by_combo = rmse_by_combo

experiment = [row["experiment"] for row in records]
model = [row["model"] for row in records]
encoder = [row["encoder"] for row in records]
decoder = [row["decoder"] for row in records]

rmse_tr_im = metrics["RMSE_tr_im"]
rmse_tr_t = metrics["RMSE_tr_t"]
fid_tr_im = metrics["FID_tr_im"]
fid_tr_t = metrics["FID_tr_t"]
ssim_tr_im = metrics["SSIM_tr_im"]
ssim_tr_t = metrics["SSIM_tr_t"]
psnr_tr_im = metrics["PSNR_tr_im"]
psnr_tr_t = metrics["PSNR_tr_t"]
epochs = metrics["epochs"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Experiment_Summaries.txt into Python variables.")
    parser.add_argument(
        "summary_file",
        nargs="?",
        default=DEFAULT_SUMMARY_FILE,
        type=Path,
        help="Path to Experiment_Summaries.txt.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to save the parsed records as JSON.",
    )
    args = parser.parse_args()

    parsed_records = parse_experiment_summaries(args.summary_file)
    parsed_metrics = metrics_as_variables(parsed_records)
    parsed_metrics_by_combo = summarize_metrics_by_combo(parsed_records)

    print(f"Loaded {len(parsed_records)} rows from {args.summary_file}")
    print("Available variables:")
    print("  records, metrics")
    print("  metrics_by_combo, rmse_by_combo, rmse_tr_im_by_combo")
    print("  experiment, model, encoder, decoder")
    print("  " + ", ".join(column.lower() for column in METRIC_COLUMNS))

    if parsed_records:
        print("\nFirst row:")
        print(json.dumps(parsed_records[0], indent=2))

    if parsed_metrics_by_combo:
        print("\nImage-encoded metrics by model + encoder + decoder:")
        print_table_header(IMAGE_TABLE_COLUMNS)
        for row in parsed_metrics_by_combo:
            print(
                f"  {row['model']:12s} | {row['encoder']:10s} | {row['decoder']:10s} | "
                f"{format_mean_std(row['avg_RMSE_tr_im'], row['std_RMSE_tr_im'])} | "
                f"{format_mean_std(row['avg_PSNR_tr_im'], row['std_PSNR_tr_im'])} | "
                f"{format_mean_std(row['avg_SSIM_tr_im'], row['std_SSIM_tr_im'])} | "
                f"{format_mean_std(row['avg_FID_tr_im'], row['std_FID_tr_im'])} | "
                f"{format_mean_std(row['avg_t_epoch_s'], row['std_t_epoch_s'])}"
            )

        print("\nTime-encoded metrics by model + encoder + decoder:")
        print_table_header(TIME_TABLE_COLUMNS)
        for row in parsed_metrics_by_combo:
            if row["avg_RMSE_tr_t"] is None:
                continue
            print(
                f"  {row['model']:12s} | {row['encoder']:10s} | {row['decoder']:10s} | "
                f"{format_mean_std(row['avg_RMSE_tr_t'], row['std_RMSE_tr_t'])} | "
                f"{format_mean_std(row['avg_PSNR_tr_t'], row['std_PSNR_tr_t'])} | "
                f"{format_mean_std(row['avg_SSIM_tr_t'], row['std_SSIM_tr_t'])} | "
                f"{format_mean_std(row['avg_FID_tr_t'], row['std_FID_tr_t'])}"
            )

    if args.json:
        args.json.write_text(json.dumps(parsed_records, indent=2), encoding="utf-8")
        print(f"\nSaved JSON to {args.json}")


if __name__ == "__main__":
    main()
