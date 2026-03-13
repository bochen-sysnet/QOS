#!/usr/bin/env python3
"""Compare random-qubit evolution metrics vs size-sweep averages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = ROOT / "openevolve_ablation" / "qubits"
RUN_PREFIX = "gem3flash_pws8_12to24_random_seed_low_full_v"
SIZE_SWEEP_DIR = (
    ROOT / "evaluation" / "plots" / "ablation_results" / "data" / "qubit_sampling_size_sweep" / "size_sweep"
)

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIGURES_DIR = OUT_DIR / "figures"

METRICS = [
    ("combined_score", "Combined Score"),
    ("qose_depth", "Depth Ratio"),
    ("qose_cnot", "CNOT Ratio"),
    ("avg_run_time", "Time Ratio"),
]


def _safe_float(raw: Any) -> float:
    try:
        if raw is None or raw == "":
            return float("nan")
        return float(raw)
    except Exception:
        return float("nan")


def _find_runs(runs_root: Path, run_prefix: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for d in sorted(runs_root.glob(f"{run_prefix}*")):
        if not d.is_dir() or "_v" not in d.name:
            continue
        suffix = d.name.rsplit("_v", 1)[-1]
        if suffix.isdigit():
            out[int(suffix)] = d
    return out


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_checkpoint_best_info(run_dir: Path) -> Path | None:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return None
    best_idx = -1
    best_path: Path | None = None
    for d in ckpt_root.glob("checkpoint_*"):
        if not d.is_dir():
            continue
        suffix = d.name.split("checkpoint_", 1)[-1]
        if not suffix.isdigit():
            continue
        idx = int(suffix)
        candidate = d / "best_program_info.json"
        if idx > best_idx and candidate.exists():
            best_idx = idx
            best_path = candidate
    return best_path


def _load_evolution_metrics(run_dir: Path) -> tuple[dict[str, float], str]:
    direct = run_dir / "best" / "best_program_info.json"
    source = "best"
    info_path = direct
    if not info_path.exists():
        fallback = _latest_checkpoint_best_info(run_dir)
        if fallback is None:
            return {}, "missing"
        info_path = fallback
        source = "checkpoint_fallback"

    payload = _load_json(info_path)
    metrics = payload.get("metrics", {})
    out = {k: _safe_float(metrics.get(k)) for k, _ in METRICS}
    return out, source


def _load_size_sweep_avg(size_sweep_dir: Path, run_name: str) -> dict[str, float]:
    path = size_sweep_dir / f"size_sweep_{run_name}_metrics.csv"
    if not path.exists():
        return {}

    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    out: dict[str, float] = {}
    for key, _label in METRICS:
        vals = [_safe_float(r.get(key)) for r in rows]
        finite = [v for v in vals if math.isfinite(v)]
        out[key] = float(sum(finite) / len(finite)) if finite else float("nan")
    return out


def _summarize(values: list[float]) -> tuple[float, float, int]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan"), float("nan"), 0
    mean = float(statistics.mean(finite))
    std = float(statistics.pstdev(finite) if len(finite) > 1 else 0.0)
    return mean, std, len(finite)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary_rows: list[dict[str, Any]], out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 24,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    metric_labels = {
        "combined_score": "Reward",
        "qose_depth": "Depth",
        "qose_cnot": "CNOT",
        "avg_run_time": "Time",
    }
    metric_order = [k for k, _ in METRICS]
    sources = [("evolution", "Random"), ("size_sweep_avg", "Full")]

    means_by_source: dict[str, list[float]] = {src: [] for src, _ in sources}
    stds_by_source: dict[str, list[float]] = {src: [] for src, _ in sources}
    for metric in metric_order:
        for src, _src_label in sources:
            row = next((r for r in summary_rows if r["metric"] == metric and r["source"] == src), None)
            means_by_source[src].append(_safe_float(row["mean"]) if row is not None else float("nan"))
            stds_by_source[src].append(_safe_float(row["std"]) if row is not None else float("nan"))

    x = np.arange(len(metric_order), dtype=float)
    width = 0.34
    # Keep original figure shape.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    bars_random = ax.bar(
        x - width / 2.0,
        means_by_source["evolution"],
        width=width,
        yerr=stds_by_source["evolution"],
        capsize=5,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.9,
        label="Random",
    )
    bars_full = ax.bar(
        x + width / 2.0,
        means_by_source["size_sweep_avg"],
        width=width,
        yerr=stds_by_source["size_sweep_avg"],
        capsize=5,
        color="#F58518",
        edgecolor="black",
        linewidth=0.9,
        label="Full",
    )

    def _annotate(bars, vals: list[float], errs: list[float]) -> None:
        for bar, value, err in zip(bars, vals, errs):
            if not math.isfinite(value):
                continue
            err = err if math.isfinite(err) else 0.0
            y = value + max(err, 0.0)
            pad = 0.015 * max(1.0, abs(y))
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y + pad,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
            )

    _annotate(bars_random, means_by_source["evolution"], stds_by_source["evolution"])
    _annotate(bars_full, means_by_source["size_sweep_avg"], stds_by_source["size_sweep_avg"])

    ax.set_xticks(x, [metric_labels.get(m, m) for m in metric_order])
    ax.set_ylabel("Metric Value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower center", frameon=True, fontsize=16)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bar plot of random-qubit run metrics: evolution best vs size-sweep average, "
            "aggregated across all versions under openevolve_ablation/qubits."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--run-prefix", default=RUN_PREFIX)
    parser.add_argument("--size-sweep-dir", type=Path, default=SIZE_SWEEP_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURES_DIR / "randq_evolution_vs_size_sweep_bar.pdf",
    )
    args = parser.parse_args()

    versions = _find_runs(args.runs_root, args.run_prefix)
    if not versions:
        raise RuntimeError(f"No runs found under {args.runs_root} with prefix {args.run_prefix}")

    per_version_rows: list[dict[str, Any]] = []
    missing_sweep: list[str] = []
    for version in sorted(versions.keys()):
        run_dir = versions[version]
        run_name = run_dir.name

        evo, evo_source = _load_evolution_metrics(run_dir)
        sweep = _load_size_sweep_avg(args.size_sweep_dir, run_name)
        if not sweep:
            missing_sweep.append(run_name)

        row: dict[str, Any] = {
            "version": version,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "evolution_source": evo_source,
            "has_size_sweep": int(bool(sweep)),
        }
        for key, _ in METRICS:
            row[f"evolution_{key}"] = evo.get(key, float("nan"))
            row[f"size_sweep_avg_{key}"] = sweep.get(key, float("nan"))
        per_version_rows.append(row)

    # Keep versions where both evolution and size-sweep data exist.
    aligned_rows = [r for r in per_version_rows if int(r["has_size_sweep"]) == 1]
    if not aligned_rows:
        raise RuntimeError("No runs have both evolution and size-sweep data.")

    summary_rows: list[dict[str, Any]] = []
    for metric, _label in METRICS:
        evo_vals = [_safe_float(r[f"evolution_{metric}"]) for r in aligned_rows]
        sweep_vals = [_safe_float(r[f"size_sweep_avg_{metric}"]) for r in aligned_rows]

        evo_mean, evo_std, evo_n = _summarize(evo_vals)
        sweep_mean, sweep_std, sweep_n = _summarize(sweep_vals)
        summary_rows.append(
            {
                "metric": metric,
                "source": "evolution",
                "n": evo_n,
                "mean": evo_mean,
                "std": evo_std,
                "min": min(v for v in evo_vals if math.isfinite(v)) if evo_n else float("nan"),
                "max": max(v for v in evo_vals if math.isfinite(v)) if evo_n else float("nan"),
            }
        )
        summary_rows.append(
            {
                "metric": metric,
                "source": "size_sweep_avg",
                "n": sweep_n,
                "mean": sweep_mean,
                "std": sweep_std,
                "min": min(v for v in sweep_vals if math.isfinite(v)) if sweep_n else float("nan"),
                "max": max(v for v in sweep_vals if math.isfinite(v)) if sweep_n else float("nan"),
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(
        DATA_DIR / "randq_evolution_vs_size_sweep_per_version.csv",
        per_version_rows,
        [
            "version",
            "run_name",
            "run_dir",
            "evolution_source",
            "has_size_sweep",
            *[f"evolution_{k}" for k, _ in METRICS],
            *[f"size_sweep_avg_{k}" for k, _ in METRICS],
        ],
    )
    _write_csv(
        DATA_DIR / "randq_evolution_vs_size_sweep_summary.csv",
        summary_rows,
        ["metric", "source", "n", "mean", "std", "min", "max"],
    )
    _plot(summary_rows, args.output)

    print(f"Wrote figure: {args.output}")
    print(f"Wrote data: {DATA_DIR / 'randq_evolution_vs_size_sweep_per_version.csv'}")
    print(f"Wrote data: {DATA_DIR / 'randq_evolution_vs_size_sweep_summary.csv'}")
    if missing_sweep:
        print("Missing size-sweep metrics for runs:")
        for name in missing_sweep:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
