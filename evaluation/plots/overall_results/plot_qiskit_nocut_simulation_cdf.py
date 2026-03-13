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
DEFAULT_OUT = ROOT / "paper_figures" / "qiskit_nocut_simulation_cdf.pdf"

TARGET_METHOD = "FrozenQubits"  # Qiskit baseline (no cutting)
SIZE_ORDER = [12, 24]
SIZE_LABELS = {12: "12-qubit", 24: "24-qubit"}
COLORS = {12: "#4C72B0", 24: "#DD8452"}


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


def _aggregate_all_samples(
    torino_csv: Path, marrakesh_csv: Path
) -> dict[int, list[float]]:
    # Keep every raw sample from both backends (no averaging).
    per_size: dict[int, list[float]] = defaultdict(list)
    for csv_path in (torino_csv, marrakesh_csv):
        for size, _bench, sim in _load(csv_path):
            per_size[size].append(sim)
    return per_size


def _plot(per_size: dict[int, list[float]], out_pdf: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)

    for size in SIZE_ORDER:
        vals = sorted(v for v in per_size.get(size, []) if v > 0)
        if not vals:
            continue
        x = np.asarray(vals, dtype=float)
        y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
        ax.step(x, y, where="post", color=COLORS[size], linewidth=2.2, label=SIZE_LABELS[size])

    ax.set_xscale("log")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", frameon=True)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CDF of Qiskit no-cut simulation time (12 vs 24 qubits)."
    )
    parser.add_argument("--timing-torino", type=Path, default=DEFAULT_TORINO)
    parser.add_argument("--timing-marrakesh", type=Path, default=DEFAULT_MARRAKESH)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.timing_torino.exists():
        raise FileNotFoundError(f"Missing Torino timing CSV: {args.timing_torino}")
    if not args.timing_marrakesh.exists():
        raise FileNotFoundError(f"Missing Marrakesh timing CSV: {args.timing_marrakesh}")

    per_size = _aggregate_all_samples(args.timing_torino, args.timing_marrakesh)
    _plot(per_size, args.out_pdf)
    print(f"Wrote figure: {args.out_pdf}")


if __name__ == "__main__":
    main()
