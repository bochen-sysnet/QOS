#!/usr/bin/env python3
"""Plot per-iteration combined score and mark new-best iterations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = (
    ROOT
    / "openevolve_ablation"
    / "gem3flash_pws8_22q_noseed_no_cases_no_summary_v2"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "figures"
FONT_SIZE = 24
LEGEND_FONT_SIZE = 20


@dataclass
class IterScore:
    iteration: int
    score: float
    time_ratio: float


def _to_float(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _load_scores(csv_path: Path) -> list[IterScore]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    per_iteration: dict[int, IterScore] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                iteration = int((row.get("iteration") or "").strip())
            except ValueError:
                continue
            score = _to_float((row.get("combined_score") or "").strip())
            time_ratio = _to_float((row.get("avg_run_time") or "").strip())
            if score is None or time_ratio is None:
                continue
            # Keep latest seen row for an iteration if duplicated.
            per_iteration[iteration] = IterScore(
                iteration=iteration,
                score=score,
                time_ratio=time_ratio,
            )

    scores = sorted(per_iteration.values(), key=lambda x: x.iteration)
    if not scores:
        raise RuntimeError(f"No valid combined_score rows found in {csv_path}")
    return scores


def _new_best_points(scores: Iterable[IterScore]) -> list[IterScore]:
    best = float("-inf")
    improved: list[IterScore] = []
    for item in scores:
        if item.score > best:
            best = item.score
            improved.append(item)
    return improved


def _find_latest_checkpoint(run_dir: Path) -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints dir: {checkpoints_dir}")

    candidates: list[tuple[int, Path]] = []
    for ckpt in checkpoints_dir.glob("checkpoint_*"):
        if not ckpt.is_dir():
            continue
        try:
            idx = int(ckpt.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((idx, ckpt))
    if not candidates:
        raise RuntimeError(f"No checkpoint_* folders under {checkpoints_dir}")
    _, latest_ckpt = max(candidates, key=lambda x: x[0])
    return latest_ckpt


def _find_initial_metric(run_dir: Path, metric_key: str) -> float:
    latest_ckpt = _find_latest_checkpoint(run_dir)

    for prog_json in sorted((latest_ckpt / "programs").glob("*.json")):
        try:
            row = json.load(prog_json.open("r", encoding="utf-8"))
        except Exception:
            continue
        if int(row.get("generation", -1)) != 0:
            continue
        metric = _to_float((row.get("metrics", {}) or {}).get(metric_key))
        if metric is not None:
            return metric
    raise RuntimeError(f"Could not find generation-0 initial program metric '{metric_key}' in {latest_ckpt}")


def _load_iteration_child_parent(run_dir: Path) -> dict[int, tuple[str, str]]:
    import re

    iter_pat = re.compile(
        r"Iteration\s+(\d+):\s+Program\s+([0-9a-fA-F-]+)\s+\(parent:\s+([0-9a-fA-F-]+)\)"
    )
    iter_to_child_parent: dict[int, tuple[str, str]] = {}
    for log_path in sorted((run_dir / "logs").glob("openevolve_*.log")):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            m = iter_pat.search(line)
            if not m:
                continue
            it = int(m.group(1))
            iter_to_child_parent[it] = (m.group(2), m.group(3))
    return iter_to_child_parent


def _load_generation_by_iteration(run_dir: Path) -> dict[int, int]:
    # Known generation hints from all checkpoint snapshots (not only latest).
    id_to_gen: dict[str, int] = {}
    checkpoints_dir = run_dir / "checkpoints"
    for prog_json in checkpoints_dir.glob("checkpoint_*/programs/*.json"):
        try:
            row = json.load(prog_json.open("r", encoding="utf-8"))
            pid = str(row.get("id", "")).strip()
            gen = int(row.get("generation"))
        except Exception:
            continue
        if pid:
            id_to_gen[pid] = gen

    iter_to_child_parent = _load_iteration_child_parent(run_dir)
    child_to_parent = {child: parent for child, parent in iter_to_child_parent.values()}
    memo: dict[str, int] = dict(id_to_gen)

    def _resolve_generation(program_id: str) -> int | None:
        if program_id in memo:
            return memo[program_id]
        parent_id = child_to_parent.get(program_id)
        if not parent_id:
            return None
        parent_gen = _resolve_generation(parent_id)
        if parent_gen is None:
            return None
        memo[program_id] = parent_gen + 1
        return memo[program_id]

    by_it: dict[int, int] = {}
    for it, (child_id, _parent_id) in iter_to_child_parent.items():
        gen = _resolve_generation(child_id)
        if gen is not None:
            by_it[it] = int(gen)
    return by_it


def _plot(
    scores: list[IterScore],
    output_path: Path,
    initial_score: float,
    initial_time_ratio: float,
    generation_by_iteration: dict[int, int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": FONT_SIZE})

    by_it = {item.iteration: item.score for item in scores}
    missing = [i for i in range(1, 101) if i not in by_it]
    if missing:
        raise RuntimeError(f"Missing iterations in CSV: {missing}")

    x = [0] + list(range(1, 101))
    y = [initial_score] + [by_it[i] for i in range(1, 101)]
    time_by_it = {item.iteration: item.time_ratio for item in scores}

    improved_x: list[int] = []
    improved_y: list[float] = []
    no_improve_x: list[int] = []
    no_improve_y: list[float] = []
    best_so_far = initial_score
    for i in range(1, 101):
        score = by_it[i]
        if score > best_so_far:
            improved_x.append(i)
            improved_y.append(score)
            best_so_far = score
        else:
            no_improve_x.append(i)
            no_improve_y.append(score)

    fig, (ax_gen, ax) = plt.subplots(1, 2, figsize=(13.8, 4.8))

    x_gen = [0] + list(range(1, 101))
    missing_gen = [i for i in range(1, 101) if i not in generation_by_iteration]
    if missing_gen:
        raise RuntimeError(
            "Missing generation entries for iterations: "
            + ",".join(str(i) for i in missing_gen[:20])
            + ("..." if len(missing_gen) > 20 else "")
        )
    y_gen = [0] + [int(generation_by_iteration[i]) for i in range(1, 101)]
    ax_gen.plot(x_gen, y_gen, color="black", linewidth=1.6, zorder=1)

    no_improve_gen = [int(generation_by_iteration[i]) for i in no_improve_x]
    improved_gen = [int(generation_by_iteration[i]) for i in improved_x]
    ax_gen.scatter([0], [0], color="#1f77b4", marker="D", s=110, zorder=4, label="Init")
    ax_gen.scatter(
        no_improve_x,
        no_improve_gen,
        color="#7f7f7f",
        s=35,
        marker="o",
        zorder=2,
        label="No Impr",
    )
    ax_gen.scatter(improved_x, improved_gen, color="#d62728", marker="*", s=110, zorder=5, label="Impr")
    ax_gen.set_xlabel("Iteration", fontsize=FONT_SIZE)
    ax_gen.set_ylabel("Generation", fontsize=FONT_SIZE)
    ax_gen.tick_params(axis="both", labelsize=FONT_SIZE)
    ax_gen.grid(axis="y", linestyle="--", alpha=0.35)
    ax_gen.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, frameon=True)

    ax.plot(x, y, color="black", linewidth=1.8, zorder=1)
    ax.scatter([0], [initial_score], color="#1f77b4", marker="D", s=110, zorder=4, label="Init")
    ax.scatter(no_improve_x, no_improve_y, color="#7f7f7f", marker="o", s=35, zorder=2, label="No Impr")
    ax.scatter(improved_x, improved_y, color="#d62728", marker="*", s=110, zorder=5, label="Impr")

    # Annotate each improvement point with its time ratio.
    ann_font = LEGEND_FONT_SIZE + 2
    prev_it: int | None = None
    for idx, (it, score) in enumerate(zip(improved_x, improved_y)):
        tr = time_by_it[it]
        x_text = float(it)
        # Avoid overlap when improvement iterations are very close (e.g., first two points).
        if prev_it is not None and (it - prev_it) <= 3:
            x_text += 1.2 if (idx % 2 == 0) else -1.2
        ax.annotate(
            f"r={tr:.2f}",
            xy=(it, score),
            xytext=(x_text, -18),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=ann_font,
            rotation=90,
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#444444"},
            color="#222222",
        )
        prev_it = it
    ax.annotate(
        f"r={initial_time_ratio:.2f}",
        xy=(0, initial_score),
        xytext=(0, -18),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=ann_font,
        rotation=90,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#444444"},
        color="#222222",
    )

    ax.set_xlabel("Iteration", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, frameon=True)
    ax.set_ylabel("Reward", fontsize=FONT_SIZE, labelpad=8)
    ymin = min(min(y), -16.0)
    ymax = max(y)
    pad = max(0.3, 0.05 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.14, top=0.99, wspace=0.30)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="OpenEvolve run folder containing runtime_metrics.iterations.csv",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Explicit CSV path. Defaults to <run-dir>/runtime_metrics.iterations.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. Default: motivation_results/figures/<run-name>_iter_progress.pdf",
    )
    args = parser.parse_args()

    csv_path = args.csv or (args.run_dir / "runtime_metrics.iterations.csv")
    run_name = args.run_dir.name.rstrip("/")
    output_path = args.output or (DEFAULT_OUT_DIR / "iter_progress.pdf")

    scores = _load_scores(csv_path)
    initial_score = _find_initial_metric(args.run_dir, "combined_score")
    initial_time_ratio = _find_initial_metric(args.run_dir, "avg_run_time")
    generation_by_iteration = _load_generation_by_iteration(args.run_dir)
    _plot(scores, output_path, initial_score, initial_time_ratio, generation_by_iteration)
    print(f"Wrote figure: {output_path}")


if __name__ == "__main__":
    main()
