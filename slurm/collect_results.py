"""Collect the results of all cluster runs into one CSV and print a table per dataset.

    python slurm/collect_results.py                 # runs/ in the repository
    python slurm/collect_results.py /path/to/runs --out results.csv

Each run folder <runs>/<dataset>/<video>/<model>-<encoder>/seed<seed>/ has its own Experiment_Summaries.txt
(the newest table is on top). This script reads that table, writes every row to one CSV (default
<runs>/results.csv) with the columns dataset, video, model, encoder, seed, finished and the summary's metric
columns, and prints, per dataset, the mean and standard deviation over its videos for each model + encoder.
Runs without a table yet (still training, or failed) are listed separately. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHOWN = ["PSNR_te_im", "PSNR_te_t", "SSIM_te_im", "SSIM_te_t", "RMSE_te_im", "RMSE_te_t", "FID_te_im",
         "PSNR_tr_im", "epochs", "t/epoch(s)"]


def parse_value(text: str):
    text = text.strip()
    if text in ("", "n/a"):
        return None
    try:
        return float(text)
    except ValueError:
        return text


def parse_summary(text: str) -> list[dict]:
    """Rows of the newest (first) table in an Experiment_Summaries.txt."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("model ") and " | " in line:
            header = [h.strip() for h in line.split("|")]
            rows = []
            for row in lines[i + 2:]:
                if " | " not in row:
                    break
                values = [v.strip() for v in row.split("|")]
                if len(values) != len(header):
                    break
                rows.append({h: (v if h in ("model", "encoder", "decoder", "t(h:m:s)") else parse_value(v))
                             for h, v in zip(header, values)})
            return rows
    return []


def collect(runs: Path) -> tuple[list[dict], list[Path]]:
    rows, missing = [], []
    for seed_dir in sorted(runs.glob("*/*/*/seed*")):
        if not seed_dir.is_dir():
            continue
        dataset, video, combo = seed_dir.parts[-4:-1]
        summary = seed_dir / "Experiment_Summaries.txt"
        table = parse_summary(summary.read_text(encoding="utf-8", errors="replace")) if summary.exists() else []
        if not table:
            missing.append(seed_dir)
            continue
        for r in table:
            rows.append({"dataset": dataset, "video": video, "model": r.pop("model"), "encoder": r.pop("encoder"),
                         "seed": int(seed_dir.name[len("seed"):]), "finished": (seed_dir / "DONE").exists(), **r})
    return rows, missing


def fmt(values: list[float]) -> str:
    values = [v for v in values if isinstance(v, float)]
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4g}"
    return f"{statistics.mean(values):.4g} ± {statistics.stdev(values):.2g}"


def print_tables(rows: list[dict]) -> None:
    for dataset in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == dataset]
        print(f"\n== {dataset}: mean ± std over videos (and seeds)")
        head = f"{'model':10s} {'encoder':11s} {'runs':>4s} " + " ".join(f"{c:>17s}" for c in SHOWN)
        print(head)
        print("-" * len(head))
        for model in sorted({r["model"] for r in sub}):
            for enc in sorted({r["encoder"] for r in sub if r["model"] == model}):
                g = [r for r in sub if r["model"] == model and r["encoder"] == enc]
                print(f"{model:10s} {enc:11s} {len(g):4d} " + " ".join(f"{fmt([r.get(c) for r in g]):>17s}"
                                                                       for c in SHOWN))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="?", default=REPO / "runs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rows, missing = collect(args.runs)
    if not rows:
        print(f"no finished runs under {args.runs}")
        return 1
    out = args.out or args.runs / "results.csv"
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} result rows written to {out}")
    print_tables(rows)
    if missing:
        print(f"\n{len(missing)} run folders without results yet (training, waiting to resubmit, or failed):")
        for p in missing:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
