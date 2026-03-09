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
DEFAULT_TORINO = ROOT / "plot_data" / "timing_torino.csv"
DEFAULT_MARRAKESH = ROOT / "plot_data" / "timing_marrakesh.csv"
DEFAULT_OUT = ROOT / "paper_figures" / "simulation_time_consumption.pdf"

METHOD_ORDER = ["FrozenQubits", "CutQC", "QOS", "QOSE"]
SIZE_ORDER = [12, 24]
BACKEND_LABELS = {
    "torino": "Torino",
    "marrakesh": "Marrakesh",
}
BACKEND_COLORS = {
    "torino": "#4C72B0",
    "marrakesh": "#55A868",
}


def _load_sim_times(path: Path) -> list[tuple[int, str, float]]:
    rows: list[tuple[int, str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_sim = (row.get("simulation") or "").strip()
            method = (row.get("method") or "").strip()
            raw_size = (row.get("size") or "").strip()
            if not raw_sim or not method or not raw_size:
                continue
            try:
                sim = float(raw_sim)
                size = int(float(raw_size))
            except ValueError:
                continue
            rows.append((size, method, sim))
    return rows


def _aggregate(
    torino_csv: Path, marrakesh_csv: Path
) -> dict[tuple[str, int, str], list[float]]:
    out: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for backend, path in (("torino", torino_csv), ("marrakesh", marrakesh_csv)):
        for size, method, sim in _load_sim_times(path):
            out[(backend, size, method)].append(sim)
    return out


def _plot(
    agg: dict[tuple[str, int, str], list[float]],
    out_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.36

    for ax, size in zip(axes, SIZE_ORDER):
        for idx, backend in enumerate(("torino", "marrakesh")):
            means = []
            stds = []
            for method in METHOD_ORDER:
                vals = agg.get((backend, size, method), [])
                if vals:
                    arr = np.asarray(vals, dtype=float)
                    means.append(float(np.mean(arr)))
                    stds.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
                else:
                    means.append(np.nan)
                    stds.append(0.0)
            ax.bar(
                x + (idx - 0.5) * width,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                color=BACKEND_COLORS[backend],
                edgecolor="black",
                linewidth=0.6,
                label=BACKEND_LABELS[backend],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="right")
        ax.set_title(f"{size}-qubit")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_yscale("log")
        ax.legend(loc="upper left", frameon=True)

    axes[0].set_ylabel("Simulation Time (s)")
    for ax in axes:
        ax.set_xlabel("Method")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot simulation-time consumption from cached full-eval timing CSVs."
    )
    parser.add_argument("--timing-torino", type=Path, default=DEFAULT_TORINO)
    parser.add_argument("--timing-marrakesh", type=Path, default=DEFAULT_MARRAKESH)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.timing_torino.exists():
        raise FileNotFoundError(f"Missing Torino timing CSV: {args.timing_torino}")
    if not args.timing_marrakesh.exists():
        raise FileNotFoundError(f"Missing Marrakesh timing CSV: {args.timing_marrakesh}")

    agg = _aggregate(args.timing_torino, args.timing_marrakesh)
    _plot(agg, args.out_pdf)
    print(f"Wrote figure: {args.out_pdf}")


if __name__ == "__main__":
    main()
