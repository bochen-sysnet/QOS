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


OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
SIZE_SWEEP_DIR = DATA_DIR / "size_sweep_all_versions"
LEGACY_SIZE_SWEEP_DIRS = [
    OUT_DIR / "data" / "size_sweep",
    ROOT
    / "evaluation"
    / "plots"
    / "ablation_results"
    / "data"
    / "qubit_sampling_size_sweep"
    / "size_sweep",
]

MODEL_FAMILIES = [
    ("gem3pro", "Gemini 3 Pro", "gem3pro_pws8_22q_seed_low_full_v"),
    ("gem3flash", "Gemini 3 Flash", "gem3flash_pws8_22q_seed_low_full_v"),
    ("gpt5mini", "GPT-5 mini", "gpt5mini_pws8_22q_full_v"),
    ("gpt53codex", "GPT-5.3 Codex", "gpt53codex_pws8_22q_full_v"),
    ("claude_sonnet46", "Claude Sonnet 4.6", "claude_sonnet46_pws8_22q_full_v"),
    ("claude_opus46", "Claude Opus 4.6", "claude_opus46_pws8_22q_full_v"),
]

FAMILY_COLORS = {
    "gem3pro": "#4C72B0",
    "gem3flash": "#55A868",
    "gpt5mini": "#C44E52",
    "gpt53codex": "#8172B3",
    "claude_sonnet46": "#CCB974",
    "claude_opus46": "#64B5CD",
}


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _combined_piecewise_from_ratios(depth_ratio: float, cnot_ratio: float, time_ratio: float) -> float:
    # Matches qos/error_mitigator/evaluator.py:_combined_score_from_ratios (piecewise mode).
    struct_delta = 1.0 - ((depth_ratio + cnot_ratio) / 2.0)
    time_delta = 1.0 - time_ratio
    slope_pos = 1.0
    slope_neg = 8.0
    struct_term = slope_pos * struct_delta if struct_delta >= 0.0 else slope_neg * struct_delta
    return float(struct_term + time_delta)


def _parse_csv_ints(raw: str) -> list[int]:
    vals: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    if not vals:
        raise ValueError("No size values provided")
    return vals


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


def _extract_metrics(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(result, "metrics") and hasattr(result, "artifacts"):
        return dict(result.metrics), dict(result.artifacts)
    if isinstance(result, dict):
        return dict(result.get("metrics", {})), dict(result.get("artifacts", {}))
    return {}, {}


def _load_rows_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _discover_runs(base_dir: Path, family_prefix: str) -> list[Path]:
    runs: list[tuple[int, Path]] = []
    for run_dir in sorted(base_dir.glob(family_prefix + "*")):
        if not run_dir.is_dir():
            continue
        tail = run_dir.name.removeprefix(family_prefix)
        if not tail.isdigit():
            continue
        if not (run_dir / "best" / "best_program.py").exists():
            continue
        runs.append((int(tail), run_dir))
    return [p for _v, p in sorted(runs, key=lambda x: x[0])]


def _row_complete(rows_by_size: dict[int, dict[str, Any]], size: int) -> bool:
    row = rows_by_size.get(size)
    if row is None:
        return False
    required = ("combined_score", "qose_depth", "qose_cnot", "avg_run_time")
    return all(math.isfinite(_safe_float(row.get(k))) for k in required)


def _evaluate_program_across_sizes(
    program: Path,
    run_name: str,
    sizes: list[int],
    benches: list[str],
    sample_seed: str,
    force_recompute: bool,
) -> Path:
    SIZE_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = SIZE_SWEEP_DIR / f"size_sweep_{run_name}_metrics.csv"

    rows_by_size: dict[int, dict[str, Any]] = {}
    if out_csv.exists() and not force_recompute:
        for row in _load_rows_csv(out_csv):
            try:
                size = int(row.get("size", ""))
            except Exception:
                continue
            rows_by_size[size] = dict(row)

    missing_sizes = [s for s in sizes if not _row_complete(rows_by_size, s)]
    if not missing_sizes:
        return out_csv

    bench_csv = ",".join(benches)
    cache_path = _default_cache_path_for_sweep(sizes, sample_seed)

    for idx, size in enumerate(missing_sizes, start=1):
        t0 = time.perf_counter()
        print(f"[eval] run={run_name} ({idx}/{len(missing_sizes)}) size={size}", flush=True)
        env_edits = [
            _set_env("QOSE_SCORE_MODE", "piecewise"),
            _set_env("QOSE_SIZE_MIN", str(size)),
            _set_env("QOSE_SIZE_MAX", str(size)),
            _set_env("QOSE_STRATIFIED_SIZES", "0"),
            _set_env("QOSE_SAMPLES_PER_BENCH", "1"),
            _set_env("QOSE_DISTINCT_SIZES_PER_BENCH", "1"),
            _set_env("QOSE_BENCHES", bench_csv),
            _set_env("QOSE_BASELINE_CACHE_PATH", cache_path),
        ]
        if sample_seed:
            env_edits.append(_set_env("QOSE_SAMPLE_SEED", sample_seed))
        else:
            env_edits.append(_set_env("QOSE_SAMPLE_SEED", None))
        try:
            result = evaluator_module.evaluate(str(program))
            metrics, artifacts = _extract_metrics(result)
        finally:
            _restore_env(env_edits)
        elapsed = time.perf_counter() - t0
        summary = artifacts.get("summary", {}) if isinstance(artifacts.get("summary", {}), dict) else {}
        qose_run_sec_avg = _safe_float(
            summary.get("qose_run_sec_avg", artifacts.get("qose_run_sec_avg"))
        )
        qos_run_sec_avg = _safe_float(
            summary.get("qos_run_sec_avg", artifacts.get("qos_run_sec_avg"))
        )
        rows_by_size[size] = {
            "size": size,
            "combined_score": _safe_float(metrics.get("combined_score")),
            "qose_depth": _safe_float(metrics.get("qose_depth")),
            "qose_cnot": _safe_float(metrics.get("qose_cnot")),
            "avg_run_time": _safe_float(metrics.get("avg_run_time")),
            "qose_overhead": _safe_float(metrics.get("qose_overhead")),
            "qose_run_sec_avg": qose_run_sec_avg,
            "qos_run_sec_avg": qos_run_sec_avg,
            "qose_over_qos_run_time_sum_ratio": (
                qose_run_sec_avg / qos_run_sec_avg if qos_run_sec_avg > 0 else float("nan")
            ),
            "eval_elapsed_sec": elapsed,
            "failure_reason": metrics.get("failure_reason", ""),
        }

        ordered_rows = [rows_by_size[s] for s in sizes if s in rows_by_size]
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "size",
                    "combined_score",
                    "qose_depth",
                    "qose_cnot",
                    "avg_run_time",
                    "qose_overhead",
                    "qose_run_sec_avg",
                    "qos_run_sec_avg",
                    "qose_over_qos_run_time_sum_ratio",
                    "eval_elapsed_sec",
                    "failure_reason",
                ],
            )
            writer.writeheader()
            writer.writerows(ordered_rows)

    return out_csv


def _ensure_cache_file(run_name: str) -> Path:
    target = SIZE_SWEEP_DIR / f"size_sweep_{run_name}_metrics.csv"
    if target.exists():
        return target
    for src_dir in LEGACY_SIZE_SWEEP_DIRS:
        src = src_dir / f"size_sweep_{run_name}_metrics.csv"
        if src.exists():
            target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            return target
    return target


def _plot_metrics_with_band(
    stats_rows: list[dict[str, Any]],
    sizes: list[int],
    out_pdf: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    metric_specs = [
        ("combined_score", "Combined Score (piecewise)", False),
        ("qose_depth", "Depth Ratio", True),
        ("qose_cnot", "CNOT Ratio", True),
        ("avg_run_time", "Time Ratio", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2), constrained_layout=True, sharex=True)
    axes_flat = axes.flatten()
    legend_handles = []
    legend_labels = []

    for ax, (metric_key, y_label, draw_ref) in zip(axes_flat, metric_specs):
        for family, display, _prefix in MODEL_FAMILIES:
            fam_rows = [r for r in stats_rows if r["family"] == family]
            by_size = {int(r["size"]): r for r in fam_rows}
            x: list[int] = []
            y_mean: list[float] = []
            y_std: list[float] = []
            for s in sizes:
                row = by_size.get(s)
                if not row:
                    continue
                mean_v = _safe_float(row.get(f"{metric_key}_mean"))
                std_v = _safe_float(row.get(f"{metric_key}_std"), 0.0)
                if not math.isfinite(mean_v):
                    continue
                x.append(s)
                y_mean.append(mean_v)
                y_std.append(std_v if math.isfinite(std_v) else 0.0)
            if not x:
                continue

            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y_mean, dtype=float)
            s_arr = np.asarray(y_std, dtype=float)
            color = FAMILY_COLORS.get(family, "#333333")
            line = ax.plot(
                x_arr,
                y_arr,
                marker="o",
                linewidth=2.0,
                markersize=5.5,
                color=color,
                label=display,
            )[0]
            ax.fill_between(x_arr, y_arr - s_arr, y_arr + s_arr, color=color, alpha=0.20, linewidth=0)
            if metric_key == "combined_score":
                legend_handles.append(line)
                legend_labels.append(display)

        if draw_ref:
            ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(y_label)
        ax.set_xticks(sizes)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes_flat[2].set_xlabel("# Qubits")
    axes_flat[3].set_xlabel("# Qubits")
    if legend_handles:
        fig.legend(legend_handles, legend_labels, ncol=3, loc="upper center", frameon=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run/resume size sweeps across all versions of six model families and plot "
            "combined-score generalization curves with mean±std bands."
        )
    )
    parser.add_argument("--sizes", default="12,14,16,18,20,22,24")
    parser.add_argument("--benches", default="")
    parser.add_argument("--sample-seed", default="")
    parser.add_argument("--families", default="")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    sizes = _parse_csv_ints(args.sizes)
    benches = (
        [t.strip() for t in args.benches.split(",") if t.strip()]
        if args.benches.strip()
        else [bench for bench, _label in evaluator_module.BENCHES]
    )
    selected_families = (
        {t.strip() for t in args.families.split(",") if t.strip()}
        if args.families.strip()
        else {f for f, _d, _p in MODEL_FAMILIES}
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SIZE_SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    per_run_rows: list[dict[str, Any]] = []
    for family, display, prefix in MODEL_FAMILIES:
        if family not in selected_families:
            continue
        run_dirs = _discover_runs(ROOT / "openevolve_output", prefix)
        if not run_dirs:
            print(f"[warn] no runs discovered for family={family} prefix={prefix}")
            continue
        for run_dir in run_dirs:
            run_name = run_dir.name
            best_program = run_dir / "best" / "best_program.py"
            metrics_csv = _ensure_cache_file(run_name)
            if not args.skip_eval:
                metrics_csv = _evaluate_program_across_sizes(
                    program=best_program,
                    run_name=run_name,
                    sizes=sizes,
                    benches=benches,
                    sample_seed=args.sample_seed,
                    force_recompute=args.force_recompute,
                )
            if not metrics_csv.exists():
                print(f"[warn] missing metrics for run={run_name}: {metrics_csv}")
                continue
            for row in _load_rows_csv(metrics_csv):
                size = int(row.get("size", "0") or 0)
                if size not in sizes:
                    continue
                qose_depth = _safe_float(row.get("qose_depth"))
                qose_cnot = _safe_float(row.get("qose_cnot"))
                avg_run_time = _safe_float(row.get("avg_run_time"))
                combined_score_reported = _safe_float(row.get("combined_score"))
                combined_score = (
                    _combined_piecewise_from_ratios(qose_depth, qose_cnot, avg_run_time)
                    if all(math.isfinite(v) for v in (qose_depth, qose_cnot, avg_run_time))
                    else float("nan")
                )
                per_run_rows.append(
                    {
                        "family": family,
                        "display_name": display,
                        "run_name": run_name,
                        "size": size,
                        "combined_score": combined_score,
                        "combined_score_reported": combined_score_reported,
                        "qose_depth": qose_depth,
                        "qose_cnot": qose_cnot,
                        "avg_run_time": avg_run_time,
                    }
                )

    per_run_csv = DATA_DIR / "combined_score_per_run_size.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "display_name",
                "run_name",
                "size",
                "combined_score",
                "combined_score_reported",
                "qose_depth",
                "qose_cnot",
                "avg_run_time",
            ],
        )
        writer.writeheader()
        writer.writerows(per_run_rows)
    print(f"[done] wrote per-run metrics: {per_run_csv}")

    stats_rows: list[dict[str, Any]] = []
    metric_keys = ["combined_score", "qose_depth", "qose_cnot", "avg_run_time"]
    for family, display, _prefix in MODEL_FAMILIES:
        if family not in selected_families:
            continue
        fam_rows = [r for r in per_run_rows if r["family"] == family]
        for size in sizes:
            row_out: dict[str, Any] = {
                "family": family,
                "display_name": display,
                "size": size,
            }
            run_count = 0
            has_any = False
            for key in metric_keys:
                vals = [
                    _safe_float(r.get(key))
                    for r in fam_rows
                    if int(r.get("size", -1)) == size
                ]
                vals = [v for v in vals if math.isfinite(v)]
                if vals:
                    has_any = True
                    arr = np.asarray(vals, dtype=float)
                    row_out[f"{key}_mean"] = float(np.mean(arr))
                    row_out[f"{key}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                    run_count = max(run_count, int(arr.size))
                else:
                    row_out[f"{key}_mean"] = float("nan")
                    row_out[f"{key}_std"] = float("nan")
            row_out["num_runs"] = run_count
            if has_any:
                stats_rows.append(row_out)

    stats_csv = DATA_DIR / "combined_score_generalization_stats.csv"
    with stats_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "display_name",
                "size",
                "combined_score_mean",
                "combined_score_std",
                "qose_depth_mean",
                "qose_depth_std",
                "qose_cnot_mean",
                "qose_cnot_std",
                "avg_run_time_mean",
                "avg_run_time_std",
                "num_runs",
            ],
        )
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"[done] wrote stats: {stats_csv}")

    fig_path = FIG_DIR / "combined_score_generalization_with_band.pdf"
    _plot_metrics_with_band(stats_rows, sizes, fig_path)
    print(f"[done] wrote figure: {fig_path}")


if __name__ == "__main__":
    main()
