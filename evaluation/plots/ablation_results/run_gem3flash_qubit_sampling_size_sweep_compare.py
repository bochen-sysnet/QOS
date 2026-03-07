#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
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
DATA_DIR = OUT_DIR / "data" / "qubit_sampling_size_sweep"
FIGURES_DIR = OUT_DIR / "figures"

VARIANT_SPECS = [
    {
        "key": "12q",
        "label": "12q",
        "root": ROOT / "openevolve_ablation",
        "prefix": "gem3flash_pws8_12q_seed_low_full_v",
        "color": "#4C78A8",
        "hatch": "//",
    },
    {
        "key": "24q",
        "label": "24q",
        "root": ROOT / "openevolve_ablation",
        "prefix": "gem3flash_pws8_24q_seed_low_full_v",
        "color": "#F58518",
        "hatch": "\\\\",
    },
    {
        "key": "randq",
        "label": "RandQ",
        "root": ROOT / "openevolve_ablation",
        "prefix": "gem3flash_pws8_12to24_random_seed_low_full_v",
        "color": "#54A24B",
        "hatch": "..",
    },
    {
        "key": "22q",
        "label": "22q (Ours)",
        "root": ROOT / "openevolve_output",
        "prefix": "gem3flash_pws8_22q_seed_low_full_v",
        "color": "#B279A2",
        "hatch": "xx",
    },
]

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

CASE_FIELDS = [
    "size",
    "bench",
    "case_index",
    "qose_depth",
    "qos_depth",
    "depth_ratio",
    "qose_cnot",
    "qos_cnot",
    "cnot_ratio",
    "qose_run_sec",
    "qos_run_sec",
    "run_time_ratio",
    "qose_num_circuits",
    "qos_num_circuits",
    "overhead_ratio",
    "qose_input_size",
    "qose_output_size",
    "qose_method",
    "qose_gv_cost_calls",
    "qose_wc_cost_calls",
    "score_mode",
    "sample_seed",
    "benches",
]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _parse_sizes(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("Expected at least one size.")
    return sorted(set(out))


def _list_version_dirs(root: Path, prefix: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for d in sorted(root.glob(f"{prefix}*")):
        if not d.is_dir() or "_v" not in d.name:
            continue
        suffix = d.name.rsplit("_v", 1)[-1]
        if suffix.isdigit():
            out[int(suffix)] = d
    return out


def _read_rows_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    prev = None
    if key in __import__("os").environ:
        prev = __import__("os").environ.get(key)
    if value is None:
        __import__("os").environ.pop(key, None)
    else:
        __import__("os").environ[key] = value
    return key, prev


def _restore_env(edits: list[tuple[str, str | None]]) -> None:
    env = __import__("os").environ
    for key, prev in reversed(edits):
        if prev is None:
            env.pop(key, None)
        else:
            env[key] = prev


def _extract_metrics_artifacts(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(result, "metrics") and hasattr(result, "artifacts"):
        return dict(result.metrics), dict(result.artifacts)
    if isinstance(result, dict):
        return dict(result.get("metrics", {})), dict(result.get("artifacts", {}))
    return {}, {}


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


def _resolve_best_program(run_dir: Path) -> tuple[Path | None, str]:
    direct = run_dir / "best" / "best_program.py"
    if direct.exists():
        return direct, "best"
    fallback = _latest_checkpoint_best_program(run_dir)
    if fallback is not None:
        return fallback, "checkpoint_fallback"
    return None, "missing"


def _load_metrics_map(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_rows_csv(path):
        try:
            size = int(row.get("size", ""))
        except Exception:
            continue
        norm = dict(row)
        norm["size"] = size
        for k in METRICS_FIELDS:
            if k in ("size", "failure_reason"):
                continue
            norm[k] = _safe_float(norm.get(k))
        out[size] = norm
    return out


def _row_complete(row: dict[str, Any], retry_failed: bool, score_mode: str) -> bool:
    # Default behavior is strict no-rerun: if a row exists for this size, treat as complete.
    # Use --retry-failed to recompute rows with non-finite combined_score.
    row_mode = str(row.get("score_mode", "")).strip().lower()
    if row_mode and row_mode != score_mode:
        return False
    if not row_mode:
        # Old caches may not have score_mode; recompute once for explicit provenance.
        return False
    if not retry_failed:
        return True
    return math.isfinite(_safe_float(row.get("combined_score")))


def _evaluate_one_size(
    program_path: Path,
    size: int,
    benches: str,
    sample_seed: str,
    cache_path: str,
    score_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t0 = time.perf_counter()
    env_edits = [
        _set_env("QOSE_SIZE_MIN", str(size)),
        _set_env("QOSE_SIZE_MAX", str(size)),
        _set_env("QOSE_STRATIFIED_SIZES", "0"),
        _set_env("QOSE_SAMPLES_PER_BENCH", "1"),
        _set_env("QOSE_DISTINCT_SIZES_PER_BENCH", "1"),
        _set_env("QOSE_BENCHES", benches),
        _set_env("QOSE_BASELINE_CACHE_PATH", cache_path),
        _set_env("QOSE_SCORE_MODE", score_mode),
        _set_env("QOSE_INCLUDE_CASES_ARTIFACT", "1"),
        _set_env("QOSE_INCLUDE_SUMMARY_ARTIFACT", "1"),
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
    out_score_mode = str(summary.get("score_mode", score_mode))
    qose_run_sec_avg = _safe_float(summary.get("qose_run_sec_avg", artifacts.get("qose_run_sec_avg")))
    qos_run_sec_avg = _safe_float(summary.get("qos_run_sec_avg", artifacts.get("qos_run_sec_avg")))
    cases = artifacts.get("cases", [])
    if not isinstance(cases, list):
        cases = []

    row = {
        "size": int(size),
        "score_mode": out_score_mode,
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
    case_rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            continue
        qose_depth = _safe_float(case.get("qose_depth"))
        qos_depth = _safe_float(case.get("qos_depth"))
        qose_cnot = _safe_float(case.get("qose_cnot"))
        qos_cnot = _safe_float(case.get("qos_cnot"))
        qose_run_sec = _safe_float(case.get("qose_run_sec"))
        qos_run_sec = _safe_float(case.get("qos_run_sec"))
        qose_num_circuits = _safe_float(case.get("qose_num_circuits"))
        qos_num_circuits = _safe_float(case.get("qos_num_circuits"))

        gv_trace = case.get("qose_gv_cost_trace")
        wc_trace = case.get("qose_wc_cost_trace")
        gv_calls = len(gv_trace) if isinstance(gv_trace, list) else float("nan")
        wc_calls = len(wc_trace) if isinstance(wc_trace, list) else float("nan")

        case_rows.append(
            {
                "size": int(size),
                "bench": str(case.get("bench", "")),
                "case_index": i,
                "qose_depth": qose_depth,
                "qos_depth": qos_depth,
                "depth_ratio": (_safe_float(qose_depth / qos_depth) if qos_depth > 0 else float("nan")),
                "qose_cnot": qose_cnot,
                "qos_cnot": qos_cnot,
                "cnot_ratio": (_safe_float(qose_cnot / qos_cnot) if qos_cnot > 0 else float("nan")),
                "qose_run_sec": qose_run_sec,
                "qos_run_sec": qos_run_sec,
                "run_time_ratio": (
                    _safe_float(qose_run_sec / qos_run_sec) if qos_run_sec > 0 else float("nan")
                ),
                "qose_num_circuits": qose_num_circuits,
                "qos_num_circuits": qos_num_circuits,
                "overhead_ratio": (
                    _safe_float(qose_num_circuits / qos_num_circuits)
                    if qos_num_circuits > 0
                    else float("nan")
                ),
                "qose_input_size": _safe_float(case.get("qose_input_size")),
                "qose_output_size": _safe_float(case.get("qose_output_size")),
                "qose_method": str(case.get("qose_method", "")),
                "qose_gv_cost_calls": gv_calls,
                "qose_wc_cost_calls": wc_calls,
                "score_mode": out_score_mode,
                "sample_seed": sample_seed,
                "benches": benches,
            }
        )
    return row, case_rows


def _evaluate_program_across_sizes(
    run_name: str,
    program_path: Path,
    sizes: list[int],
    benches: str,
    sample_seed: str,
    force_recompute: bool,
    retry_failed: bool,
    score_mode: str,
) -> tuple[Path, Path]:
    metrics_path = DATA_DIR / "size_sweep" / f"size_sweep_{run_name}_metrics.csv"
    cases_path = DATA_DIR / "size_sweep_cases" / f"size_sweep_{run_name}_cases.csv"
    rows_by_size = _load_metrics_map(metrics_path) if (metrics_path.exists() and not force_recompute) else {}
    run_case_rows = _read_rows_csv(cases_path) if (cases_path.exists() and not force_recompute) else []
    missing_sizes = [
        size
        for size in sizes
        if size not in rows_by_size
        or not _row_complete(rows_by_size[size], retry_failed=retry_failed, score_mode=score_mode)
    ]
    cache_path = _default_cache_path_for_sweep(sizes, sample_seed)

    if missing_sizes:
        print(f"[size-sweep] run={run_name} missing_sizes={missing_sizes}", flush=True)
    else:
        print(f"[size-sweep] run={run_name} already complete for requested sizes", flush=True)

    for i, size in enumerate(missing_sizes, start=1):
        print(f"[size-sweep] run={run_name} ({i}/{len(missing_sizes)}) size={size}", flush=True)
        row, case_rows = _evaluate_one_size(
            program_path=program_path,
            size=size,
            benches=benches,
            sample_seed=sample_seed,
            cache_path=cache_path,
            score_mode=score_mode,
        )
        rows_by_size[size] = row

        # Preserve cached rows for sizes outside current request; avoid data loss
        # when running a subset (e.g., --sizes 22 for sanity checks).
        ordered_rows = [rows_by_size[s] for s in sorted(rows_by_size.keys())]
        _write_rows_csv(metrics_path, ordered_rows, METRICS_FIELDS)
        run_case_rows = [
            r
            for r in run_case_rows
            if int(r.get("size", "-1")) != int(size)
        ]
        run_case_rows.extend(case_rows)
        run_case_rows_sorted = sorted(
            run_case_rows,
            key=lambda r: (int(r.get("size", "0")), str(r.get("bench", "")), int(r.get("case_index", "0"))),
        )
        _write_rows_csv(cases_path, run_case_rows_sorted, CASE_FIELDS)
        print(
            "[size-sweep] run=%s size=%s combined=%.4f depth=%.4f cnot=%.4f elapsed=%.1fs"
            % (
                run_name,
                size,
                _safe_float(row.get("combined_score")),
                _safe_float(row.get("qose_depth")),
                _safe_float(row.get("qose_cnot")),
                _safe_float(row.get("eval_elapsed_sec")),
            ),
            flush=True,
        )
    return metrics_path, cases_path


def _mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _summarize_variant_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[float]] = {}
    labels: dict[str, str] = {}
    for row in run_rows:
        key = str(row["variant_key"])
        by_variant.setdefault(key, []).append(_safe_float(row["avg_combined_score"]))
        labels[key] = str(row["variant_label"])

    summary: list[dict[str, Any]] = []
    for spec in VARIANT_SPECS:
        key = spec["key"]
        vals = [v for v in by_variant.get(key, []) if math.isfinite(v)]
        if not vals:
            summary.append(
                {
                    "variant_key": key,
                    "variant_label": labels.get(key, spec["label"]),
                    "n_runs": 0,
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                }
            )
            continue
        summary.append(
            {
                "variant_key": key,
                "variant_label": labels.get(key, spec["label"]),
                "n_runs": len(vals),
                "mean": float(statistics.mean(vals)),
                "std": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
                "min": float(min(vals)),
                "max": float(max(vals)),
            }
        )
    return summary


def _plot_variant_summary(summary_rows: list[dict[str, Any]], out_pdf: Path) -> None:
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

    ordered_rows = []
    for spec in VARIANT_SPECS:
        found = next((r for r in summary_rows if r["variant_key"] == spec["key"]), None)
        if found is None:
            found = {
                "variant_key": spec["key"],
                "variant_label": spec["label"],
                "n_runs": 0,
                "mean": "",
                "std": "",
            }
        ordered_rows.append(found)

    x = np.arange(len(ordered_rows), dtype=float)
    means = [
        _safe_float(row["mean"]) if str(row.get("mean", "")).strip() != "" else float("nan")
        for row in ordered_rows
    ]
    stds = [
        _safe_float(row["std"], 0.0) if str(row.get("std", "")).strip() != "" else 0.0
        for row in ordered_rows
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.5), constrained_layout=True)
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=[spec["color"] for spec in VARIANT_SPECS],
        edgecolor="black",
        linewidth=0.9,
    )
    for bar, spec in zip(bars, VARIANT_SPECS):
        bar.set_hatch(spec["hatch"])

    for idx, (bar, m, s) in enumerate(zip(bars, means, stds)):
        if not math.isfinite(m):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                0.01,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=11,
            )
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            m + max(s, 0.0) + 0.01,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_xticks(x, [spec["label"] for spec in VARIANT_SPECS])
    ax.set_ylabel("Avg Combined Score Across 12-24 Qubits")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def _summarize_per_size_metrics(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = [
        ("combined_score", "Combined Score"),
        ("qose_cnot", "CNOT Ratio"),
        ("qose_depth", "Depth Ratio"),
        ("avg_run_time", "Time Ratio"),
    ]
    grouped: dict[tuple[str, str, int, str, str], list[float]] = {}
    for row in raw_rows:
        variant_key = str(row["variant_key"])
        variant_label = str(row["variant_label"])
        size = int(row["size"])
        for metric_key, metric_label in metric_keys:
            value = _safe_float(row.get(metric_key))
            if not math.isfinite(value):
                continue
            group_key = (variant_key, variant_label, size, metric_key, metric_label)
            grouped.setdefault(group_key, []).append(value)

    out_rows: list[dict[str, Any]] = []
    for (variant_key, variant_label, size, metric_key, metric_label), values in sorted(grouped.items()):
        out_rows.append(
            {
                "variant_key": variant_key,
                "variant_label": variant_label,
                "size": size,
                "metric_key": metric_key,
                "metric_label": metric_label,
                "n_runs": len(values),
                "mean": float(statistics.mean(values)),
                "std": float(statistics.pstdev(values) if len(values) > 1 else 0.0),
                "min": float(min(values)),
                "max": float(max(values)),
            }
        )
    return out_rows


def _plot_per_size_metrics(
    per_size_rows: list[dict[str, Any]],
    sizes: list[int],
    out_pdf: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    metric_panels = [
        ("combined_score", "Combined Score"),
        ("qose_cnot", "CNOT Ratio (QSA/QOS)"),
        ("qose_depth", "Depth Ratio (QSA/QOS)"),
        ("avg_run_time", "Time Ratio (QSA/QOS)"),
    ]
    marker_map = {"12q": "o", "24q": "s", "randq": "^", "22q": "D"}

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2), constrained_layout=True)
    flat_axes = axes.flatten()

    for ax, (metric_key, y_label) in zip(flat_axes, metric_panels):
        for spec in VARIANT_SPECS:
            means: list[float] = []
            stds: list[float] = []
            for size in sizes:
                rec = next(
                    (
                        row
                        for row in per_size_rows
                        if row["variant_key"] == spec["key"]
                        and int(row["size"]) == int(size)
                        and row["metric_key"] == metric_key
                    ),
                    None,
                )
                if rec is None:
                    means.append(float("nan"))
                    stds.append(float("nan"))
                else:
                    means.append(_safe_float(rec.get("mean")))
                    stds.append(max(0.0, _safe_float(rec.get("std"), 0.0)))

            ax.errorbar(
                sizes,
                means,
                yerr=stds,
                color=spec["color"],
                marker=marker_map.get(spec["key"], "o"),
                markersize=5.2,
                linewidth=1.9,
                capsize=3.5,
                label=spec["label"],
                alpha=0.95,
            )

        ax.set_xticks(sizes)
        ax.set_xlabel("Qubits")
        ax.set_ylabel(y_label)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="best", frameon=True, ncol=1)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Size-sweep best programs from the gem3flash qubit-sampling ablation runs "
            "(12q, 24q, random-12-24, and 22q original), then compare averaged scores "
            "with error bars across versions."
        )
    )
    parser.add_argument(
        "--sizes",
        default="12,14,16,18,20,22,24",
        help="Comma-separated evaluation sizes.",
    )
    parser.add_argument(
        "--benches",
        default="",
        help="Comma-separated benches. Empty means evaluator defaults.",
    )
    parser.add_argument(
        "--sample-seed",
        default="",
        help="Optional fixed sample seed for evaluator runs.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute cached size-sweep rows even if CSV exists.",
    )
    parser.add_argument(
        "--max-runs-per-variant",
        type=int,
        default=0,
        help="Optional limit for quick tests (0 means all).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Re-evaluate size rows whose cached combined_score is non-finite. "
            "Default is no-rerun for any existing size row."
        ),
    )
    parser.add_argument(
        "--score-mode",
        choices=("piecewise", "legacy"),
        default="piecewise",
        help="Scoring mode passed to evaluator via QOSE_SCORE_MODE (default: piecewise).",
    )
    args = parser.parse_args()

    sizes = _parse_sizes(args.sizes)
    benches = args.benches.strip()
    sample_seed = args.sample_seed.strip()
    score_mode = args.score_mode.strip().lower()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    run_summary_rows: list[dict[str, Any]] = []
    cases_rows: list[dict[str, Any]] = []
    size_set = set(sizes)

    for spec in VARIANT_SPECS:
        version_dirs = _list_version_dirs(spec["root"], spec["prefix"])
        versions = sorted(version_dirs.keys())
        if args.max_runs_per_variant > 0:
            versions = versions[: args.max_runs_per_variant]
        if not versions:
            print(f"[skip] No runs found for {spec['label']} ({spec['prefix']}*)", flush=True)
            continue

        for version in versions:
            run_dir = version_dirs[version]
            run_name = run_dir.name
            program_path, source = _resolve_best_program(run_dir)
            if program_path is None:
                print(f"[skip] run={run_name} no best program found", flush=True)
                continue
            print(f"[run] variant={spec['label']} run={run_name} source={source}", flush=True)
            metrics_csv, cases_csv = _evaluate_program_across_sizes(
                run_name=run_name,
                program_path=program_path,
                sizes=sizes,
                benches=benches,
                sample_seed=sample_seed,
                force_recompute=args.force_recompute,
                retry_failed=args.retry_failed,
                score_mode=score_mode,
            )
            metrics_rows = _read_rows_csv(metrics_csv)
            run_cases = _read_rows_csv(cases_csv)
            per_size_scores: list[float] = []
            for row in metrics_rows:
                size = int(row["size"])
                if size not in size_set:
                    continue
                combined = _safe_float(row.get("combined_score"))
                per_size_scores.append(combined)
                raw_rows.append(
                    {
                        "variant_key": spec["key"],
                        "variant_label": spec["label"],
                        "version": version,
                    "run_name": run_name,
                    "size": size,
                    "score_mode": score_mode,
                    "combined_score": combined,
                        "qose_depth": _safe_float(row.get("qose_depth")),
                        "qose_cnot": _safe_float(row.get("qose_cnot")),
                        "avg_run_time": _safe_float(row.get("avg_run_time")),
                        "qose_over_qos_run_time_sum_ratio": _safe_float(
                            row.get("qose_over_qos_run_time_sum_ratio")
                        ),
                        "metrics_csv": str(metrics_csv),
                    }
                )
            for case in run_cases:
                try:
                    case_size = int(case.get("size", ""))
                except Exception:
                    continue
                if case_size not in size_set:
                    continue
                one = dict(case)
                one["variant_key"] = spec["key"]
                one["variant_label"] = spec["label"]
                one["version"] = version
                one["run_name"] = run_name
                cases_rows.append(one)
            run_summary_rows.append(
                {
                    "variant_key": spec["key"],
                    "variant_label": spec["label"],
                    "version": version,
                    "run_name": run_name,
                    "program_path": str(program_path),
                    "program_source": source,
                    "score_mode": score_mode,
                    "avg_combined_score": _mean(per_size_scores),
                    "n_sizes": sum(1 for v in per_size_scores if math.isfinite(v)),
                    "sizes": ",".join(str(s) for s in sizes),
                }
            )

    variant_summary_rows = _summarize_variant_rows(run_summary_rows)
    per_size_metric_summary_rows = _summarize_per_size_metrics(raw_rows)

    _write_rows_csv(
        DATA_DIR / "qubit_sampling_size_sweep_raw.csv",
        raw_rows,
        [
            "variant_key",
            "variant_label",
            "version",
            "run_name",
            "size",
            "score_mode",
            "combined_score",
            "qose_depth",
            "qose_cnot",
            "avg_run_time",
            "qose_over_qos_run_time_sum_ratio",
            "metrics_csv",
        ],
    )
    _write_rows_csv(
        DATA_DIR / "qubit_sampling_size_sweep_run_summary.csv",
        run_summary_rows,
        [
            "variant_key",
            "variant_label",
            "version",
            "run_name",
            "program_path",
            "program_source",
            "score_mode",
            "avg_combined_score",
            "n_sizes",
            "sizes",
        ],
    )
    _write_rows_csv(
        DATA_DIR / "qubit_sampling_size_sweep_cases_raw.csv",
        cases_rows,
        [
            "variant_key",
            "variant_label",
            "version",
            "run_name",
            *CASE_FIELDS,
        ],
    )
    _write_rows_csv(
        DATA_DIR / "qubit_sampling_size_sweep_variant_summary.csv",
        variant_summary_rows,
        ["variant_key", "variant_label", "n_runs", "mean", "std", "min", "max"],
    )
    _write_rows_csv(
        DATA_DIR / "qubit_sampling_size_sweep_per_size_metric_summary.csv",
        per_size_metric_summary_rows,
        [
            "variant_key",
            "variant_label",
            "size",
            "metric_key",
            "metric_label",
            "n_runs",
            "mean",
            "std",
            "min",
            "max",
        ],
    )

    _plot_variant_summary(
        variant_summary_rows,
        FIGURES_DIR / "gem3flash_qubit_sampling_size_sweep_compare.pdf",
    )
    _plot_per_size_metrics(
        per_size_metric_summary_rows,
        sizes=sizes,
        out_pdf=FIGURES_DIR / "gem3flash_qubit_sampling_size_sweep_per_size_metrics.pdf",
    )

    print("Wrote data under:", DATA_DIR)
    print("Wrote figure:", FIGURES_DIR / "gem3flash_qubit_sampling_size_sweep_compare.pdf")
    print("Wrote figure:", FIGURES_DIR / "gem3flash_qubit_sampling_size_sweep_per_size_metrics.pdf")


if __name__ == "__main__":
    main()
