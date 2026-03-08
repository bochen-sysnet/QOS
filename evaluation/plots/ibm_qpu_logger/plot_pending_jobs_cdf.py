#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "ibm_qpu_metrics.csv"
DEFAULT_OUT = ROOT / "figures" / "pending_jobs_cdf.pdf"

COLORS = {
    "ibm_torino": "#4C72B0",
    "ibm_marrakesh": "#55A868",
    "ibm_fez": "#C44E52",
}


def _load_pending_jobs(csv_path: Path) -> dict[str, list[float]]:
    by_backend: dict[str, list[float]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            backend = (row.get("backend_name") or "").strip()
            pending_jobs = (row.get("pending_jobs") or "").strip()
            if not backend or pending_jobs == "":
                continue
            try:
                by_backend[backend].append(float(pending_jobs))
            except ValueError:
                continue
    return dict(sorted(by_backend.items()))


def _nearest_rank_percentiles(values: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    """Return nearest-rank percentile values (always observed samples)."""
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0:
        return np.asarray([], dtype=float)
    out: list[float] = []
    for p in percentiles:
        rank = int(np.ceil((float(p) / 100.0) * n))
        rank = max(1, min(rank, n))
        out.append(float(arr[rank - 1]))
    return np.asarray(out, dtype=float)


def _plot_percentiles(data: dict[str, list[float]], out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax_pct = plt.subplots(1, 1, figsize=(8.8, 5.2), constrained_layout=True)

    percentiles = np.array([95, 99, 99.9], dtype=float)
    backends = [b for b, vals in data.items() if vals]
    x = np.arange(len(percentiles), dtype=float)
    width = 0.82 / max(1, len(backends))
    for idx, backend in enumerate(backends):
        arr = np.asarray(data[backend], dtype=float)
        pvals = _nearest_rank_percentiles(arr, percentiles)
        offsets = (idx - (len(backends) - 1) / 2.0) * width
        bars = ax_pct.bar(
            x + offsets,
            pvals,
            width=width,
            color=COLORS.get(backend),
            edgecolor="black",
            linewidth=0.7,
            label=backend.replace("ibm_", ""),
        )
        for bar, value in zip(bars, pvals):
            ax_pct.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{int(round(value))}",
                ha="center",
                va="bottom",
                fontsize=14,
                rotation=18,
            )

    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(
        [f"{int(p)}th" if float(p).is_integer() else f"{p:.1f}th" for p in percentiles]
    )
    ax_pct.set_xlabel("Percentile")
    ax_pct.set_ylabel("Pending Jobs")
    ax_pct.set_yscale("symlog", linthresh=1.0)
    ax_pct.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_pct.legend(loc="upper left", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IBM QPU pending-jobs CDF from logged CSV data.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.input_csv}")

    data = _load_pending_jobs(args.input_csv)
    if not data:
        raise RuntimeError(f"No valid pending-jobs data found in: {args.input_csv}")

    _plot_percentiles(data, args.out_pdf)
    print(f"Wrote figure: {args.out_pdf}")


if __name__ == "__main__":
    main()
