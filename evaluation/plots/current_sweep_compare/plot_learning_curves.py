#!/usr/bin/env python3
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIGURES_DIR = OUT_DIR / "figures"

MODEL_FAMILIES = [
    ("gem3pro", "Gemini 3 Pro", "gem3pro_pws8_22q_seed_low_full_v", "#4C72B0"),
    ("gem3flash", "Gemini 3 Flash", "gem3flash_pws8_22q_seed_low_full_v", "#55A868"),
    ("gpt5mini", "GPT-5 mini", "gpt5mini_pws8_22q_full_v", "#C44E52"),
    ("gpt53codex", "GPT-5.3 Codex", "gpt53codex_pws8_22q_full_v", "#8172B3"),
    ("claude_sonnet46", "Claude Sonnet 4.6", "claude_sonnet46_pws8_22q_full_v", "#CCB974"),
    ("claude_opus46", "Claude Opus 4.6", "claude_opus46_pws8_22q_full_v", "#64B5CD"),
]


def _safe_float(value, default=float("nan")):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _checkpoint_number(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)$", path.name)
    if not match:
        raise ValueError(f"Unexpected checkpoint directory name: {path.name}")
    return int(match.group(1))


def _load_run_curve(run_dir: Path) -> list[dict[str, object]]:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return []
    rows: list[dict[str, object]] = []
    for ckpt in sorted(ckpt_root.glob("checkpoint_*"), key=_checkpoint_number):
        info_path = ckpt / "best_program_info.json"
        if not info_path.exists():
            continue
        try:
            data = json.loads(info_path.read_text())
        except Exception:
            continue
        metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
        score = _safe_float(metrics.get("combined_score"))
        if not math.isfinite(score):
            continue
        rows.append(
            {
                "iteration": _checkpoint_number(ckpt),
                "combined_score": score,
                "run_name": run_dir.name,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_dir = ROOT / "openevolve_output"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    per_run_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.1), constrained_layout=True)

    for family, display_name, prefix, color in MODEL_FAMILIES:
        run_curves: list[list[dict[str, object]]] = []
        for run_dir in sorted(base_dir.glob(prefix + "*")):
            curve = _load_run_curve(run_dir)
            if not curve:
                continue
            run_curves.append(curve)
            for row in curve:
                per_run_rows.append(
                    {
                        "family": family,
                        "display_name": display_name,
                        "run_name": run_dir.name,
                        "iteration": row["iteration"],
                        "combined_score": row["combined_score"],
                    }
                )

        if not run_curves:
            continue

        by_iter: dict[int, list[float]] = defaultdict(list)
        for curve in run_curves:
            for row in curve:
                by_iter[int(row["iteration"])].append(float(row["combined_score"]))

        iterations = sorted(by_iter)
        mean_vals = [float(np.mean(by_iter[it])) for it in iterations]
        min_vals = [float(np.min(by_iter[it])) for it in iterations]
        max_vals = [float(np.max(by_iter[it])) for it in iterations]

        ax.plot(
            iterations,
            mean_vals,
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=5.5,
            label=display_name,
        )
        if len(run_curves) > 1:
            ax.fill_between(iterations, min_vals, max_vals, color=color, alpha=0.18, linewidth=0)

        for it, mean_v, min_v, max_v in zip(iterations, mean_vals, min_vals, max_vals):
            aggregate_rows.append(
                {
                    "family": family,
                    "display_name": display_name,
                    "iteration": it,
                    "runs": len(by_iter[it]),
                    "score_mean": mean_v,
                    "score_min": min_v,
                    "score_max": max_v,
                }
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Combined Score So Far")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncol=2, frameon=True)

    per_run_csv = DATA_DIR / "learning_curves_per_run.csv"
    aggregate_csv = DATA_DIR / "learning_curves_aggregate.csv"
    figure_pdf = FIGURES_DIR / "learning_curves_all_models.pdf"

    _write_csv(
        per_run_csv,
        per_run_rows,
        ["family", "display_name", "run_name", "iteration", "combined_score"],
    )
    _write_csv(
        aggregate_csv,
        aggregate_rows,
        ["family", "display_name", "iteration", "runs", "score_mean", "score_min", "score_max"],
    )
    fig.savefig(figure_pdf)
    plt.close(fig)

    print(f"Wrote data: {per_run_csv}")
    print(f"Wrote data: {aggregate_csv}")
    print(f"Wrote figure: {figure_pdf}")


if __name__ == "__main__":
    main()
