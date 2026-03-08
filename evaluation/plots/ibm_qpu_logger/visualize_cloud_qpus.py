#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "figures" / "cloud_accessible_qpus.pdf"


def main() -> None:
    providers = [
        "IBM",
        "AWS",
        "Azure",
        "Google",
    ]
    qpu_counts = [
        12,
        8,
        6,
        2,
    ]

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    bars = ax.bar(
        providers,
        qpu_counts,
        color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"],
        edgecolor="black",
        linewidth=0.8,
    )

    # ax.set_title("Cloud-Accessible Universal Gate-Model QPUs (Approximate)")
    ax.set_ylabel("Number of QPUs")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, qpu_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.15,
            str(value),
            ha="center",
            va="bottom",
        )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"Wrote figure: {OUT_PDF}")


if __name__ == "__main__":
    main()
