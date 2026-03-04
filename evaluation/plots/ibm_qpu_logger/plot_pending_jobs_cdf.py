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


def _plot_cdf(data: dict[str, list[float]], out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)

    for backend, values in data.items():
        if not values:
            continue
        arr = np.sort(np.asarray(values, dtype=float))
        y = np.arange(1, len(arr) + 1, dtype=float) / len(arr)
        ax.step(
            arr,
            y,
            where="post",
            linewidth=2.4,
            color=COLORS.get(backend),
            label=f"{backend} (n={len(arr)})",
        )

    ax.set_xlabel("Pending Jobs")
    ax.set_ylabel("CDF")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True)

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

    _plot_cdf(data, args.out_pdf)
    print(f"Wrote figure: {args.out_pdf}")


if __name__ == "__main__":
    main()
