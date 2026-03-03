#!/usr/bin/env python3
import csv
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

MODEL_PANEL = {
    "gem3pro": "Gemini",
    "gem3flash": "Gemini",
    "gpt5mini": "GPT",
    "gpt53codex": "GPT",
    "claude_sonnet46": "Claude",
    "claude_opus46": "Claude",
}

PANELS = ["Gemini", "GPT", "Claude"]

ITER_SUCCESS_RE = re.compile(r"Iteration\s+(\d+):\s+Program .* completed in ")
ITER_ERROR_RE = re.compile(r"Iteration\s+(\d+)\s+error:")
COMBINED_RE = re.compile(r"combined_score=([+-]?\d+(?:\.\d+)?)")


def _safe_float(value, default=float("nan")):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_run_curve_from_logs(run_dir: Path) -> list[dict[str, object]]:
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return []

    events: dict[int, dict[str, object]] = {}
    for log_path in sorted(logs_dir.glob("*.log")):
        lines = log_path.read_text(errors="ignore").splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            m_ok = ITER_SUCCESS_RE.search(line)
            if m_ok:
                iteration = int(m_ok.group(1))
                score = float("nan")
                if i + 1 < len(lines):
                    m_score = COMBINED_RE.search(lines[i + 1])
                    if m_score:
                        score = _safe_float(m_score.group(1))
                        i += 1
                events[iteration] = {"iteration": iteration, "status": "success", "score": score}
                i += 1
                continue

            m_err = ITER_ERROR_RE.search(line)
            if m_err:
                iteration = int(m_err.group(1))
                events[iteration] = {"iteration": iteration, "status": "error", "score": float("nan")}
            i += 1

    if not events:
        return []

    max_iter = max(events)
    best_so_far = float("-inf")
    curve: list[dict[str, object]] = []
    for iteration in range(1, max_iter + 1):
        event = events.get(iteration)
        status = "missing"
        raw_score = float("nan")
        if event is not None:
            status = str(event["status"])
            raw_score = _safe_float(event["score"])
            if status == "success" and math.isfinite(raw_score):
                best_so_far = max(best_so_far, raw_score)
        best_score = best_so_far if best_so_far != float("-inf") else float("nan")
        curve.append(
            {
                "iteration": iteration,
                "status": status,
                "raw_score": raw_score,
                "best_score_so_far": best_score,
                "run_name": run_dir.name,
            }
        )
    return curve


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
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True, constrained_layout=True)
    panel_axes = {panel: axes[idx] for idx, panel in enumerate(PANELS)}

    for family, display_name, prefix, color in MODEL_FAMILIES:
        run_curves: list[list[dict[str, object]]] = []
        for run_dir in sorted(base_dir.glob(prefix + "*")):
            curve = _load_run_curve_from_logs(run_dir)
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
                        "status": row["status"],
                        "raw_score": row["raw_score"],
                        "best_score_so_far": row["best_score_so_far"],
                    }
                )

        if not run_curves:
            continue

        max_iter = max(len(curve) for curve in run_curves)
        by_iter: dict[int, list[float]] = defaultdict(list)
        for curve in run_curves:
            last_best = float("nan")
            for row in curve:
                last_best = _safe_float(row["best_score_so_far"], last_best)
                if math.isfinite(last_best):
                    by_iter[int(row["iteration"])].append(last_best)
            # extend flat after the last logged iteration, if needed
            if curve:
                last_iter = int(curve[-1]["iteration"])
                if math.isfinite(last_best):
                    for iteration in range(last_iter + 1, max_iter + 1):
                        by_iter[iteration].append(last_best)

        iterations = sorted(by_iter)
        mean_vals = [float(np.mean(by_iter[it])) for it in iterations]
        min_vals = [float(np.min(by_iter[it])) for it in iterations]
        max_vals = [float(np.max(by_iter[it])) for it in iterations]

        ax = panel_axes[MODEL_PANEL[family]]
        ax.plot(
            iterations,
            mean_vals,
            color=color,
            linewidth=2.3,
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

    for idx, panel in enumerate(PANELS):
        ax = panel_axes[panel]
        ax.set_xlabel("Iteration")
        if idx == 0:
            ax.set_ylabel("Best Combined Score So Far")
        ax.set_title(panel)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower right", frameon=True)

    per_run_csv = DATA_DIR / "learning_curves_per_run.csv"
    aggregate_csv = DATA_DIR / "learning_curves_aggregate.csv"
    figure_pdf = FIGURES_DIR / "learning_curves_all_models.pdf"

    _write_csv(
        per_run_csv,
        per_run_rows,
        [
            "family",
            "display_name",
            "run_name",
            "iteration",
            "status",
            "raw_score",
            "best_score_so_far",
        ],
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
