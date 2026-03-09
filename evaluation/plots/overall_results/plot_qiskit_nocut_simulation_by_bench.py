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
DEFAULT_OUT = ROOT / "paper_figures" / "qiskit_nocut_simulation_by_bench.pdf"

TARGET_METHOD = "FrozenQubits"  # Qiskit baseline (no circuit cutting)
SIZE_ORDER = [12, 24]
BENCH_ORDER = [
    "bv",
    "ghz",
    "hamsim_1",
    "qaoa_pl1",
    "qaoa_r3",
    "qsvm",
    "twolocal_1",
    "vqe_1",
    "wstate",
]


def _load(path: Path) -> list[tuple[int, str, float]]:
    rows: list[tuple[int, str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("method") or "").strip() != TARGET_METHOD:
                continue
            raw_size = (row.get("size") or "").strip()
            bench = (row.get("bench") or "").strip()
            raw_sim = (row.get("simulation") or "").strip()
            if not raw_size or not bench or not raw_sim:
                continue
            try:
                size = int(float(raw_size))
                sim = float(raw_sim)
            except ValueError:
                continue
            rows.append((size, bench, sim))
    return rows


def _aggregate(torino_csv: Path, marrakesh_csv: Path) -> dict[tuple[int, str], list[float]]:
    out: dict[tuple[int, str], list[float]] = defaultdict(list)
    for csv_path in (torino_csv, marrakesh_csv):
        for size, bench, sim in _load(csv_path):
            out[(size, bench)].append(sim)
    return out


def _plot(agg: dict[tuple[int, str], list[float]], out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    x = np.arange(len(BENCH_ORDER), dtype=float)
    bar_color = "#4C72B0"

    for ax, size in zip(axes, SIZE_ORDER):
        means: list[float] = []
        stds: list[float] = []
        for bench in BENCH_ORDER:
            vals = agg.get((size, bench), [])
            if vals:
                arr = np.asarray(vals, dtype=float)
                means.append(float(np.mean(arr)))
                stds.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
            else:
                means.append(np.nan)
                stds.append(0.0)

        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            color=bar_color,
            edgecolor="black",
            linewidth=0.7,
        )
        ax.set_title(f"{size}-Qubit Circuits")
        ax.set_xticks(x)
        ax.set_xticklabels(BENCH_ORDER, rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_xlabel("Bench")

    axes[0].set_ylabel("Simulation Time (s)")
    fig.suptitle("Qiskit (No Cut) Simulation Time by Bench", y=1.02, fontsize=15)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Qiskit no-cut simulation time for 12/24 qubits across benches."
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
