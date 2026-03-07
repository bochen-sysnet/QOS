#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ABLATION_DIR = ROOT / "openevolve_ablation"
ABLATION_DIFF_DIR = ABLATION_DIR / "diff"
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIGURES_DIR = OUT_DIR / "figures"

RUNS = [
    ("No Seed", "Full", ["gem3flash_pws8_22q_noseed_low_full"]),
    ("No Seed", "Diff", ["gem3flash_pws8_22q_noseed_low_diff"]),
    ("Seed", "Full", ["gem3flash_pws8_22q_seed_low_full"]),
    ("Seed", "Diff", ["gem3flash_pws8_22q_seed_low_diff"]),
]

COLORS = {
    "Full": "#4C72B0",
    "Diff": "#C44E52",
}


def _read_combined_score(run_dir: Path) -> float:
    info_path = run_dir / "best" / "best_program_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing best_program_info.json: {info_path}")
    with info_path.open() as f:
        payload = json.load(f)
    metrics = payload.get("metrics", {})
    score = metrics.get("combined_score")
    if score is None:
        raise KeyError(f"combined_score missing in: {info_path}")
    return float(score)


def _resolve_run_dir(names: list[str]) -> Path:
    roots = [ABLATION_DIFF_DIR, ABLATION_DIR]
    for name in names:
        for root in roots:
            candidate = root / name
            if candidate.exists():
                return candidate
    tried = [str(root / name) for root in roots for name in names]
    raise FileNotFoundError(f"Could not find run dir. Tried: {tried}")


def main() -> None:
    rows: list[dict[str, object]] = []
    for seed_group, variant, run_names in RUNS:
        run_dir = _resolve_run_dir(run_names)
        rows.append(
            {
                "seed_group": seed_group,
                "variant": variant,
                "run_dir": str(run_dir),
                "combined_score": _read_combined_score(run_dir),
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "gem3flash_seed_diff_full_compare.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed_group", "variant", "run_dir", "combined_score"])
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.3), constrained_layout=True)

    groups = ["No Seed", "Seed"]
    variants = ["Full", "Diff"]
    x = np.arange(len(groups), dtype=float)
    width = 0.34

    for idx, variant in enumerate(variants):
        values = []
        for group in groups:
            row = next(item for item in rows if item["seed_group"] == group and item["variant"] == variant)
            values.append(float(row["combined_score"]))
        offset = (idx - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=variant,
            color=COLORS[variant],
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    ax.set_xticks(x, groups)
    ax.set_ylabel("Best Combined Score")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True)

    fig_path = FIGURES_DIR / "gem3flash_seed_diff_full_compare.pdf"
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"Wrote data: {csv_path}")
    print(f"Wrote figure: {fig_path}")


if __name__ == "__main__":
    main()
