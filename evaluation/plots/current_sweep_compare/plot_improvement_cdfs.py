#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIGURES_DIR = OUT_DIR / "figures"
BASE_DIR = ROOT / "openevolve_output"

MAX_ITER = 100
NO_IMPROVEMENT_SENTINEL = MAX_ITER + 1

ITER_SUCCESS_RE = re.compile(r"Iteration\s+(\d+):\s+Program .* completed in ")
ITER_ERROR_RE = re.compile(r"Iteration\s+(\d+)\s+error:")
COMBINED_RE = re.compile(r"combined_score=([+-]?\d+(?:\.\d+)?)")
EVAL_SCORE_RE = re.compile(r"Evaluated program .*combined_score=([+-]?\d+(?:\.\d+)?)")
NEW_BEST_RE = re.compile(
    r"New best program .* \(combined_score:\s*([+-]?\d+(?:\.\d+)?)\s*→\s*([+-]?\d+(?:\.\d+)?)"
)

MODEL_SPECS = [
    ("gem3pro", "Gemini 3 Pro", "#4C72B0"),
    ("gem3flash", "Gemini 3 Flash", "#55A868"),
    ("gpt5mini", "GPT-5 mini", "#C44E52"),
    ("gpt53codex", "GPT-5.3 Codex", "#8172B3"),
    ("claude_sonnet46", "Claude Sonnet 4.6", "#CCB974"),
    ("claude_opus46", "Claude Opus 4.6", "#64B5CD"),
]


def _safe_float(v: str | None) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _model_family(name: str) -> str:
    lower = name.lower()
    for key, _, _ in MODEL_SPECS:
        if key in lower:
            return key
    m = re.match(r"(.+)_v\d+$", name)
    return m.group(1) if m else name


def _parse_run_improvements(run_dir: Path) -> Tuple[float, float, int, Dict[int, float]]:
    """
    Returns:
    - first_improvement_iteration (1..100, or 101 when no improvement in first 100)
    - last_improvement_iteration (1..100, or 101 when no improvement in first 100)
    - number_of_improvements_within_100_iterations
    - iteration->score map for successful iterations
    """
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return float(NO_IMPROVEMENT_SENTINEL), float(NO_IMPROVEMENT_SENTINEL), 0, {}

    events: Dict[int, Tuple[str, float]] = {}
    first_eval_score = float("nan")
    first_newbest_old = float("nan")

    for log_path in sorted(logs_dir.glob("*.log")):
        lines = log_path.read_text(errors="ignore").splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            if not math.isfinite(first_eval_score):
                m_eval = EVAL_SCORE_RE.search(line)
                if m_eval:
                    first_eval_score = _safe_float(m_eval.group(1))

            if not math.isfinite(first_newbest_old):
                m_nb = NEW_BEST_RE.search(line)
                if m_nb:
                    first_newbest_old = _safe_float(m_nb.group(1))

            m_ok = ITER_SUCCESS_RE.search(line)
            if m_ok:
                iteration = int(m_ok.group(1))
                score = float("nan")
                if i + 1 < len(lines):
                    m_score = COMBINED_RE.search(lines[i + 1])
                    if m_score:
                        score = _safe_float(m_score.group(1))
                        i += 1
                events[iteration] = ("success", score)
                i += 1
                continue

            m_err = ITER_ERROR_RE.search(line)
            if m_err:
                iteration = int(m_err.group(1))
                events[iteration] = ("error", float("nan"))

            i += 1

    # Baseline at iteration 0.
    if math.isfinite(first_newbest_old):
        baseline = first_newbest_old
    elif math.isfinite(first_eval_score):
        baseline = first_eval_score
    else:
        # Fallback if initial score cannot be recovered from logs.
        first_success = min(
            (it for it, (st, sc) in events.items() if st == "success" and math.isfinite(sc)),
            default=None,
        )
        baseline = events[first_success][1] if first_success is not None else float("nan")

    improvements = 0
    first_improvement_iter = NO_IMPROVEMENT_SENTINEL
    last_improvement_iter = NO_IMPROVEMENT_SENTINEL
    best_so_far = baseline
    successful_scores: Dict[int, float] = {}
    eps = 1e-12

    for iteration in range(1, MAX_ITER + 1):
        status, score = events.get(iteration, ("missing", float("nan")))
        if status == "success" and math.isfinite(score):
            successful_scores[iteration] = score
            if not math.isfinite(best_so_far):
                best_so_far = score
            elif score > best_so_far + eps:
                improvements += 1
                if first_improvement_iter == NO_IMPROVEMENT_SENTINEL:
                    first_improvement_iter = iteration
                last_improvement_iter = iteration
                best_so_far = score

    return float(first_improvement_iter), float(last_improvement_iter), improvements, successful_scores


def _cdf_xy(values: List[float]) -> Tuple[List[float], List[float]]:
    if not values:
        return [], []
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    x, y = [], []
    for i, v in enumerate(sorted_vals, start=1):
        x.append(v)
        y.append(i / n)
    return x, y


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_display = {k: name for k, name, _ in MODEL_SPECS}
    model_color = {k: color for k, _, color in MODEL_SPECS}

    run_rows: List[Dict[str, object]] = []
    first_improv_by_model: Dict[str, List[float]] = defaultdict(list)
    last_improv_by_model: Dict[str, List[float]] = defaultdict(list)
    improv_count_by_model: Dict[str, List[float]] = defaultdict(list)

    for run_dir in sorted(BASE_DIR.glob("*_v*")):
        if not run_dir.is_dir():
            continue
        family = _model_family(run_dir.name)
        first_iter, last_iter, n_impr, _ = _parse_run_improvements(run_dir)

        first_improv_by_model[family].append(first_iter)
        last_improv_by_model[family].append(last_iter)
        improv_count_by_model[family].append(float(n_impr))
        run_rows.append(
            {
                "run_name": run_dir.name,
                "model_family": family,
                "first_improvement_iteration": int(first_iter),
                "last_improvement_iteration": int(last_iter),
                "improvements_within_100_iter": int(n_impr),
            }
        )

    # Add aggregated "all models" view.
    # Exclude no-improvement runs from both CDFs.
    all_first = [
        v
        for vals in first_improv_by_model.values()
        for v in vals
        if v <= MAX_ITER
    ]
    all_last = [
        v
        for vals in last_improv_by_model.values()
        for v in vals
        if v <= MAX_ITER
    ]
    all_counts = [
        v
        for vals in improv_count_by_model.values()
        for v in vals
        if v > 0
    ]

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(17.8, 4.3), constrained_layout=True)
    ax_first, ax_last, ax_count = axes

    # Deterministic plotting order: known models first, then unknown.
    if all_first and all_last and all_counts:
        x1, y1 = _cdf_xy(all_first)
        x_last, y_last = _cdf_xy(all_last)
        x2, y2 = _cdf_xy(all_counts)
        ax_first.step(x1, y1, where="post", linewidth=2.8, color="black", label="All Runs")
        ax_last.step(x_last, y_last, where="post", linewidth=2.8, color="black", label="All Runs")
        ax_count.step(x2, y2, where="post", linewidth=2.8, color="black", label="All Runs")

    ax_first.set_xlabel("Iteration of First Improvement")
    ax_first.set_ylabel("CDF")
    if all_first:
        x_min = max(1, int(min(all_first)) - 1)
        x_max = min(MAX_ITER, int(max(all_first)) + 1)
        if x_max <= x_min:
            x_max = min(MAX_ITER, x_min + 1)
        ax_first.set_xlim(x_min, x_max)
        span = x_max - x_min
        if span <= 10:
            ticks = list(range(x_min, x_max + 1))
        else:
            n = 6
            step = span / float(n - 1)
            ticks = sorted({int(round(x_min + i * step)) for i in range(n)})
            if ticks[0] != x_min:
                ticks = [x_min] + ticks
            if ticks[-1] != x_max:
                ticks = ticks + [x_max]
        ax_first.set_xticks(ticks)
    else:
        ax_first.set_xlim(1, MAX_ITER)
        ax_first.set_xticks([1, 20, 40, 60, 80, 100])
    ax_first.grid(axis="both", linestyle="--", alpha=0.3)

    ax_last.set_xlabel("Iteration of Last Improvement")
    ax_last.set_ylabel("CDF")
    if all_last:
        lx_min = max(1, int(min(all_last)) - 1)
        lx_max = min(MAX_ITER, int(max(all_last)) + 1)
        if lx_max <= lx_min:
            lx_max = min(MAX_ITER, lx_min + 1)
        ax_last.set_xlim(lx_min, lx_max)
        lspan = lx_max - lx_min
        if lspan <= 10:
            lticks = list(range(lx_min, lx_max + 1))
        else:
            n = 6
            step = lspan / float(n - 1)
            lticks = sorted({int(round(lx_min + i * step)) for i in range(n)})
            if lticks[0] != lx_min:
                lticks = [lx_min] + lticks
            if lticks[-1] != lx_max:
                lticks = lticks + [lx_max]
        ax_last.set_xticks(lticks)
    else:
        ax_last.set_xlim(1, MAX_ITER)
        ax_last.set_xticks([1, 20, 40, 60, 80, 100])
    ax_last.grid(axis="both", linestyle="--", alpha=0.3)

    ax_count.set_xlabel("Number of Improvements in 100 Iterations")
    ax_count.set_ylabel("CDF")
    if all_counts:
        ax_count.set_xlim(min(all_counts), max(all_counts))
    ax_count.grid(axis="both", linestyle="--", alpha=0.3)

    # Single-series figure: no legend by request.

    per_run_csv = DATA_DIR / "improvement_stats_per_run.csv"
    figure_pdf = FIGURES_DIR / "improvement_cdfs_all_models.pdf"
    _write_csv(
        per_run_csv,
        sorted(run_rows, key=lambda r: str(r["run_name"])),
        [
            "run_name",
            "model_family",
            "first_improvement_iteration",
            "last_improvement_iteration",
            "improvements_within_100_iter",
        ],
    )
    fig.savefig(figure_pdf)
    plt.close(fig)

    print(f"Wrote data: {per_run_csv}")
    print(f"Wrote figure: {figure_pdf}")


if __name__ == "__main__":
    main()
