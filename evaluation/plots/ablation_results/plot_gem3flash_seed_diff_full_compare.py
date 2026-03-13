#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qos.error_mitigator import evaluator as evaluator_module


ABLATION_DIR = ROOT / "openevolve_ablation"
ABLATION_DIFF_DIR = ABLATION_DIR / "diff"
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIGURES_DIR = OUT_DIR / "figures"
SIZE_SWEEP_DIR = DATA_DIR / "seed_diff_size_sweep"

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

METRICS_FIELDS = [
    "size",
    "score_mode",
    "qose_depth",
    "qose_cnot",
    "qose_overhead",
    "avg_run_time",
    "qose_run_sec_avg",
    "qos_run_sec_avg",
    "qose_over_qos_run_time_sum_ratio",
    "combined_score",
    "eval_elapsed_sec",
    "failure_reason",
]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _load_rows_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _set_env(key: str, value: str | None) -> tuple[str, str | None]:
    prev = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    return key, prev


def _restore_env(items: list[tuple[str, str | None]]) -> None:
    for key, prev in reversed(items):
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _default_cache_path_for_sweep(sizes: list[int], sample_seed: str) -> str:
    size_min = min(sizes)
    size_max = max(sizes)
    if size_max <= 12:
        base = "qos_baseline_12q"
    elif size_min == 12:
        base = "qos_baseline_all"
    else:
        base = "qos_baseline_24q"
    if sample_seed:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_seed)
        return f"openevolve_output/baselines/{base}_seed{safe}.json"
    return f"openevolve_output/baselines/{base}.json"


def _extract_metrics_artifacts(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(result, "metrics") and hasattr(result, "artifacts"):
        return dict(result.metrics), dict(result.artifacts)
    if isinstance(result, dict):
        return dict(result.get("metrics", {})), dict(result.get("artifacts", {}))
    return {}, {}


def _resolve_run_dir(names: list[str]) -> Path:
    roots = [ABLATION_DIFF_DIR, ABLATION_DIR]
    for name in names:
        for root in roots:
            candidate = root / name
            if candidate.exists():
                return candidate
    tried = [str(root / name) for root in roots for name in names]
    raise FileNotFoundError(f"Could not find run dir. Tried: {tried}")


def _latest_checkpoint_best_program(run_dir: Path) -> Path | None:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return None
    best_path: Path | None = None
    best_idx = -1
    for d in ckpt_root.glob("checkpoint_*"):
        if not d.is_dir():
            continue
        suffix = d.name.split("checkpoint_", 1)[-1]
        if not suffix.isdigit():
            continue
        idx = int(suffix)
        candidate = d / "best_program.py"
        if candidate.exists() and idx > best_idx:
            best_idx = idx
            best_path = candidate
    return best_path


def _resolve_best_program(run_dir: Path) -> Path:
    direct = run_dir / "best" / "best_program.py"
    if direct.exists():
        return direct
    fallback = _latest_checkpoint_best_program(run_dir)
    if fallback is not None:
        return fallback
    raise FileNotFoundError(f"No best program found in {run_dir}")


def _load_metrics_map(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _load_rows_csv(path):
        try:
            size = int(row.get("size", ""))
        except Exception:
            continue
        norm = dict(row)
        norm["size"] = size
        for k in METRICS_FIELDS:
            if k in ("size", "failure_reason", "score_mode"):
                continue
            norm[k] = _safe_float(norm.get(k))
        out[size] = norm
    return out


def _evaluate_one_size(
    program_path: Path,
    size: int,
    benches_csv: str,
    sample_seed: str,
    cache_path: str,
    score_mode: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    env_edits = [
        _set_env("QOSE_SIZE_MIN", str(size)),
        _set_env("QOSE_SIZE_MAX", str(size)),
        _set_env("QOSE_STRATIFIED_SIZES", "0"),
        _set_env("QOSE_SAMPLES_PER_BENCH", "1"),
        _set_env("QOSE_DISTINCT_SIZES_PER_BENCH", "1"),
        _set_env("QOSE_BENCHES", benches_csv),
        _set_env("QOSE_BASELINE_CACHE_PATH", cache_path),
        _set_env("QOSE_SCORE_MODE", score_mode),
    ]
    if sample_seed:
        env_edits.append(_set_env("QOSE_SAMPLE_SEED", sample_seed))
    else:
        env_edits.append(_set_env("QOSE_SAMPLE_SEED", None))
    try:
        result = evaluator_module.evaluate(str(program_path))
        metrics, artifacts = _extract_metrics_artifacts(result)
    finally:
        _restore_env(env_edits)
    elapsed = time.perf_counter() - t0

    summary = artifacts.get("summary", {}) if isinstance(artifacts.get("summary"), dict) else {}
    qose_run_sec_avg = _safe_float(summary.get("qose_run_sec_avg", artifacts.get("qose_run_sec_avg")))
    qos_run_sec_avg = _safe_float(summary.get("qos_run_sec_avg", artifacts.get("qos_run_sec_avg")))
    return {
        "size": int(size),
        "score_mode": str(summary.get("score_mode", score_mode)),
        "qose_depth": _safe_float(metrics.get("qose_depth")),
        "qose_cnot": _safe_float(metrics.get("qose_cnot")),
        "qose_overhead": _safe_float(metrics.get("qose_overhead")),
        "avg_run_time": _safe_float(metrics.get("avg_run_time")),
        "qose_run_sec_avg": qose_run_sec_avg,
        "qos_run_sec_avg": qos_run_sec_avg,
        "qose_over_qos_run_time_sum_ratio": (
            qose_run_sec_avg / qos_run_sec_avg if qos_run_sec_avg > 0 else float("nan")
        ),
        "combined_score": _safe_float(metrics.get("combined_score")),
        "eval_elapsed_sec": elapsed,
        "failure_reason": str(metrics.get("failure_reason", "")),
    }


def _evaluate_program_across_sizes(
    run_name: str,
    program_path: Path,
    sizes: list[int],
    benches_csv: str,
    sample_seed: str,
    score_mode: str,
    force_recompute: bool,
) -> Path:
    metrics_path = SIZE_SWEEP_DIR / f"size_sweep_{run_name}_metrics.csv"
    rows_by_size = _load_metrics_map(metrics_path) if (metrics_path.exists() and not force_recompute) else {}
    missing_sizes = [s for s in sizes if s not in rows_by_size]
    cache_path = _default_cache_path_for_sweep(sizes, sample_seed)

    for idx, size in enumerate(missing_sizes, start=1):
        print(f"[size-sweep] run={run_name} ({idx}/{len(missing_sizes)}) size={size}", flush=True)
        rows_by_size[size] = _evaluate_one_size(
            program_path=program_path,
            size=size,
            benches_csv=benches_csv,
            sample_seed=sample_seed,
            cache_path=cache_path,
            score_mode=score_mode,
        )
        ordered_rows = [rows_by_size[s] for s in sorted(rows_by_size.keys())]
        _write_rows_csv(metrics_path, ordered_rows, METRICS_FIELDS)
    return metrics_path


def _avg_combined_from_metrics(metrics_csv: Path, sizes: list[int]) -> tuple[float | None, int]:
    size_set = set(sizes)
    vals: list[float] = []
    for row in _load_rows_csv(metrics_csv):
        try:
            size = int(row.get("size", ""))
        except Exception:
            continue
        if size not in size_set:
            continue
        score = _safe_float(row.get("combined_score"))
        if math.isfinite(score):
            vals.append(score)
    if not vals:
        return None, 0
    return float(sum(vals) / len(vals)), len(vals)


def _parse_sizes(raw: str) -> list[int]:
    out = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("Expected at least one size.")
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Seed/Diff variants using average size-sweep reward."
    )
    parser.add_argument("--sizes", default="12,14,16,18,20,22,24")
    parser.add_argument("--benches", default="")
    parser.add_argument("--sample-seed", default="")
    parser.add_argument("--score-mode", choices=("piecewise", "legacy"), default="piecewise")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    sizes = _parse_sizes(args.sizes)
    benches = (
        [token.strip() for token in args.benches.split(",") if token.strip()]
        if args.benches.strip()
        else [bench for bench, _ in evaluator_module.BENCHES]
    )
    benches_csv = ",".join(benches)

    rows: list[dict[str, object]] = []
    for seed_group, variant, run_names in RUNS:
        run_dir = _resolve_run_dir(run_names)
        run_name = run_dir.name
        program = _resolve_best_program(run_dir)
        metrics_csv = SIZE_SWEEP_DIR / f"size_sweep_{run_name}_metrics.csv"
        if not args.skip_eval:
            metrics_csv = _evaluate_program_across_sizes(
                run_name=run_name,
                program_path=program,
                sizes=sizes,
                benches_csv=benches_csv,
                sample_seed=args.sample_seed.strip(),
                score_mode=args.score_mode,
                force_recompute=args.force_recompute,
            )
        avg_reward, n_sizes = _avg_combined_from_metrics(metrics_csv, sizes)
        rows.append(
            {
                "seed_group": seed_group,
                "variant": variant,
                "run_dir": str(run_dir),
                "metrics_csv": str(metrics_csv),
                "avg_size_sweep_reward": "" if avg_reward is None else float(avg_reward),
                "n_sizes": int(n_sizes),
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "gem3flash_seed_diff_full_compare.csv"
    _write_rows_csv(
        csv_path,
        rows,
        ["seed_group", "variant", "run_dir", "metrics_csv", "avg_size_sweep_reward", "n_sizes"],
    )

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
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
            values.append(_safe_float(row["avg_size_sweep_reward"]))
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
            if not math.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    0.01,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=20,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=20,
                )

    ax.set_xticks(x, groups)
    ax.set_ylabel("Reward")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True)

    fig_path = FIGURES_DIR / "gem3flash_seed_diff_full_compare.pdf"
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"Wrote data: {csv_path}")
    print(f"Wrote figure: {fig_path}")


if __name__ == "__main__":
    main()
