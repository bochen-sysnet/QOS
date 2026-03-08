#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "ibm_qpu_metrics.csv"
DEFAULT_OUT = ROOT / "figures" / "error_metrics_by_backend.pdf"

METRICS = [
    ("median_2q_gate_error", "2Q Gate Error", "#4C72B0", "///"),
    ("median_1q_gate_error", "1Q Gate Error", "#55A868", "\\\\\\"),
    ("median_readout_metric", "Readout Metric", "#C44E52", "xxx"),
]
BACKEND_ORDER = ["ibm_torino", "ibm_marrakesh", "ibm_fez"]


def _load_stats(csv_path: Path) -> dict[str, dict[str, tuple[float, float, int]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            backend = (row.get("backend_name") or "").strip()
            if not backend:
                continue
            for metric, *_ in METRICS:
                raw = (row.get(metric) or "").strip()
                if not raw:
                    continue
                try:
                    val = float(raw)
                except ValueError:
                    continue
                if val > 0.5:
                    continue
                values[backend][metric].append(val)

    stats: dict[str, dict[str, tuple[float, float, int]]] = {}
    for backend, by_metric in values.items():
        stats[backend] = {}
        for metric, *_ in METRICS:
            arr = by_metric.get(metric, [])
            if not arr:
                continue
            mu = float(sum(arr) / len(arr))
            sigma = float(statistics.stdev(arr)) if len(arr) > 1 else 0.0
            stats[backend][metric] = (mu, sigma, len(arr))
    return stats


def _display_backend_name(name: str) -> str:
    return name.replace("ibm_", "").capitalize()


def _plot(stats: dict[str, dict[str, tuple[float, float, int]]], out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    backends = [b for b in BACKEND_ORDER if b in stats]
    if not backends:
        raise RuntimeError("No backend stats available to plot.")

    x = np.arange(len(backends), dtype=float)
    width = 0.23

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    for i, (metric, label, color, hatch) in enumerate(METRICS):
        means = []
        stds = []
        for backend in backends:
            mu, sigma, _n = stats.get(backend, {}).get(metric, (np.nan, np.nan, 0))
            means.append(mu)
            stds.append(sigma)

        ax.bar(
            x + (i - 1) * width,
            means,
            yerr=stds,
            capsize=3,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            hatch=hatch,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_display_backend_name(b) for b in backends])
    ax.set_ylabel("Error")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot mean/std of 2Q/1Q/readout errors across IBM backends."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.input_csv}")
    stats = _load_stats(args.input_csv)
    _plot(stats, args.out_pdf)
    print(f"Wrote figure: {args.out_pdf}")


if __name__ == "__main__":
    main()
