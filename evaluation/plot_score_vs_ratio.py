import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    w = 0.02
    target_increase = 0.02
    target_time = 0.2
    k = math.log(1.0 + (1.0 - target_time) / (2.0 * w)) / target_increase

    ratios = np.linspace(0.95, 1.02, 201)
    penalties = np.exp(k * (ratios - 1.0)) - 1.0
    # Assume depth ratio == cnot ratio == r, and avg_run_time == 1.0
    scores = -(w * (penalties * 2.0) + 1.0)
    print(f"w: {w}")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(ratios, scores, linewidth=2.0)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Combined score vs depth/cnot ratio (r=r_d=r_c)")
    ax.set_xlabel("Depth/CNOT ratio")
    ax.set_ylabel("Combined score (avg_run_time=1.0)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0.95, 1.02)

    out_path = Path(__file__).resolve().parent / "plots/score_vs_ratio.pdf"
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
