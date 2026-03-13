#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
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
DATA_DIR = OUT_DIR / "data"
FIGURES_DIR = OUT_DIR / "figures"
SIZE_SWEEP_DIR = DATA_DIR / "size_sweep"
SIZE_SWEEP_ALL_VERSIONS_DIR = DATA_DIR / "size_sweep_all_versions"

MODEL_FAMILIES = [
    ("gem3pro", "Gemini 3 Pro", "gem3pro_pws8_22q_seed_low_full_v"),
    ("gem3flash", "Gemini 3 Flash", "gem3flash_pws8_22q_seed_low_full_v"),
    ("gpt5mini", "GPT-5 mini", "gpt5mini_pws8_22q_full_v"),
    ("gpt53codex", "GPT-5.3 Codex", "gpt53codex_pws8_22q_full_v"),
    ("claude_sonnet46", "Claude Sonnet 4.6", "claude_sonnet46_pws8_22q_full_v"),
    ("claude_opus46", "Claude Opus 4.6", "claude_opus46_pws8_22q_full_v"),
]

METRIC_GROUPS = (
    ("time_ratio_mean", "Time (mean of ratio)"),
    ("time_ratio_sum", "Time (ratio of mean)"),
    ("depth_ratio", "Depth"),
    ("cnot_ratio", "CNOT"),
)

# Official published token prices as checked on 2026-03-04.
# Sources:
# - Gemini Developer API pricing: https://ai.google.dev/pricing
# - OpenAI API pricing: https://platform.openai.com/docs/pricing/
# - Anthropic model pages:
#   https://www.anthropic.com/claude/sonnet
#   https://www.anthropic.com/claude/opus
MODEL_PRICING_USD_PER_1M = {
    "gem3pro": {"input": 2.00, "output": 12.00, "cached_input": 0.20},
    "gem3flash": {"input": 0.50, "output": 3.00, "cached_input": 0.05},
    "gpt5mini": {"input": 0.125, "output": 1.00, "cached_input": 0.0125},  # flex tier
    "gpt53codex": {"input": 1.75, "output": 14.00, "cached_input": 0.175},  # standard tier
    "claude_sonnet46": {"input": 3.00, "output": 15.00, "cached_input": 0.0},
    "claude_opus46": {"input": 5.00, "output": 25.00, "cached_input": 0.0},
}


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _parse_csv_ints(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


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


def _extract_version_from_name(name: str) -> int:
    match = re.search(r"_v(\d+)$", name)
    return int(match.group(1)) if match else -1


def _select_all_runs(base_dir: Path, selected_families: set[str], max_versions_per_family: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for family_key, display_name, prefix in MODEL_FAMILIES:
        if family_key not in selected_families:
            continue
        family_runs: list[dict[str, Any]] = []
        for run_dir in sorted(base_dir.glob(prefix + "*")):
            info_path = run_dir / "best" / "best_program_info.json"
            program_path = run_dir / "best" / "best_program.py"
            if not info_path.exists() or not program_path.exists():
                continue
            version = _extract_version_from_name(run_dir.name)
            family_runs.append(
                {
                    "family": family_key,
                    "display_name": display_name,
                    "run_name": run_dir.name,
                    "version": version,
                    "best_program": str(program_path),
                }
            )
        family_runs = sorted(
            family_runs,
            key=lambda row: (int(row["version"]), str(row["run_name"])),
        )
        if max_versions_per_family > 0:
            family_runs = family_runs[:max_versions_per_family]
        selected.extend(family_runs)
    return selected


def _select_best_runs(base_dir: Path, selected_families: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for family_key, display_name, prefix in MODEL_FAMILIES:
        if family_key not in selected_families:
            continue
        best_row: dict[str, Any] | None = None
        for run_dir in sorted(base_dir.glob(prefix + "*")):
            info_path = run_dir / "best" / "best_program_info.json"
            program_path = run_dir / "best" / "best_program.py"
            if not info_path.exists() or not program_path.exists():
                continue
            try:
                data = json.loads(info_path.read_text())
            except Exception:
                continue
            metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
            score = _safe_float(metrics.get("combined_score"))
            if not math.isfinite(score):
                continue
            row = {
                "family": family_key,
                "display_name": display_name,
                "run_name": run_dir.name,
                "combined_score_22q": score,
                "best_program": str(program_path),
            }
            if best_row is None or score > best_row["combined_score_22q"]:
                best_row = row
        if best_row is None:
            raise FileNotFoundError(f"No valid best program found for family '{family_key}'")
        selected.append(best_row)
    return selected


def _write_selected_models(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["family", "display_name", "run_name", "combined_score_22q", "best_program"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_program_across_sizes(
    program: Path,
    run_name: str,
    sizes: list[int],
    benches: list[str],
    sample_seed: str,
    force_recompute: bool,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = output_dir / f"size_sweep_{run_name}_metrics.csv"
    rows_by_size: dict[int, dict[str, Any]] = {}
    if metrics_csv_path.exists() and not force_recompute:
        for row in _load_rows_csv(metrics_csv_path):
            try:
                size = int(row.get("size", ""))
            except Exception:
                continue
            norm = dict(row)
            norm["size"] = size
            for key in (
                "qose_depth",
                "qose_cnot",
                "qose_overhead",
                "avg_run_time",
                "qose_run_sec_avg",
                "qos_run_sec_avg",
                "qose_over_qos_run_time_sum_ratio",
                "combined_score",
                "eval_elapsed_sec",
            ):
                norm[key] = _safe_float(norm.get(key))
            rows_by_size[size] = norm

    def _row_complete(size: int) -> bool:
        row = rows_by_size.get(size)
        if row is None:
            return False
        required = (
            "qose_depth",
            "qose_cnot",
            "avg_run_time",
            "qose_run_sec_avg",
            "qos_run_sec_avg",
            "qose_over_qos_run_time_sum_ratio",
            "combined_score",
        )
        return all(math.isfinite(_safe_float(row.get(key))) for key in required)

    missing_sizes = [size for size in sizes if not _row_complete(size)]
    bench_csv = ",".join(benches)
    cache_path = _default_cache_path_for_sweep(sizes, sample_seed)

    if missing_sizes:
        print(f"[eval] run={run_name} missing_sizes={missing_sizes}", flush=True)
    else:
        print(f"[eval] run={run_name} all requested sizes already cached", flush=True)

    for idx, size in enumerate(missing_sizes, start=1):
        t0 = time.perf_counter()
        print(f"[eval] run={run_name} ({idx}/{len(missing_sizes)}) size={size}", flush=True)
        env_edits = [
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
        row = {
            "size": size,
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
            "failure_reason": metrics.get("failure_reason", ""),
        }
        rows_by_size[size] = row
        ordered_rows = [rows_by_size[s] for s in sizes if s in rows_by_size]
        with metrics_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "size",
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
                ],
            )
            writer.writeheader()
            writer.writerows(ordered_rows)
        print(
            "[eval] run=%s size=%s combined=%.4f depth=%.4f cnot=%.4f time=%.4f elapsed=%.1fs"
            % (
                run_name,
                size,
                row["combined_score"],
                row["qose_depth"],
                row["qose_cnot"],
                row["avg_run_time"],
                elapsed,
            ),
            flush=True,
        )

    return metrics_csv_path


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [_safe_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _load_summary_from_metrics(metrics_csv: Path, sizes: list[int]) -> dict[str, float]:
    rows = _load_rows_csv(metrics_csv)
    selected_rows = []
    size_set = set(sizes)
    for row in rows:
        try:
            size = int(row.get("size", ""))
        except Exception:
            continue
        if size in size_set:
            selected_rows.append(row)
    if not selected_rows:
        raise ValueError(f"No matching rows found in {metrics_csv}")
    qose_sum = sum(
        value
        for value in (_safe_float(row.get("qose_run_sec_avg")) for row in selected_rows)
        if math.isfinite(value)
    )
    qos_sum = sum(
        value
        for value in (_safe_float(row.get("qos_run_sec_avg")) for row in selected_rows)
        if math.isfinite(value)
    )
    return {
        "depth_ratio": _mean_metric(selected_rows, "qose_depth"),
        "cnot_ratio": _mean_metric(selected_rows, "qose_cnot"),
        "time_ratio_mean": _mean_metric(selected_rows, "avg_run_time"),
        "time_ratio_sum": (
            qose_sum / qos_sum
            if qose_sum > 0.0 and qos_sum > 0.0
            else _mean_metric(selected_rows, "qose_over_qos_run_time_sum_ratio")
        ),
        "combined_score": _mean_metric(selected_rows, "combined_score"),
    }


def _plot_summary(summary_rows: list[dict[str, Any]], out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(summary_rows))
    width = 0.15
    metrics = [
        ("Depth Ratio", "depth_ratio", "#4C72B0"),
        ("CNOT Ratio", "cnot_ratio", "#DD8452"),
        ("Time Ratio (ratio, then mean)", "time_ratio_mean", "#55A868"),
        ("Time Ratio (sum, then ratio)", "time_ratio_sum", "#8172B3"),
    ]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    fig, ax = plt.subplots(figsize=(14.0, 4.6), constrained_layout=True)
    for (label, key, color), off in zip(metrics, offsets):
        values = [_safe_float(row.get(key)) for row in summary_rows]
        bars = ax.bar(
            x + off,
            values,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            if not math.isfinite(value):
                continue
            y_pad = 0.01 if value >= 0 else -0.01
            va = "bottom" if value >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + y_pad,
                f"{value:.3f}",
                ha="center",
                va=va,
                fontsize=9,
                rotation=90,
            )

    ax.axhline(1.0, color="black", linewidth=0.9, linestyle=":", alpha=0.6)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("Average Across 12-24 Qubits")
    ax.set_xticks(x)
    ax.set_xticklabels([row["display_name"] for row in summary_rows], rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def _summarize_family_metric(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in raw_rows:
        for metric_key, _metric_label in METRIC_GROUPS:
            value = _safe_float(row.get(metric_key))
            grouped.setdefault((str(row["family"]), str(row["display_name"]), metric_key), []).append(value)

    summary_rows: list[dict[str, Any]] = []
    for (family, display_name, metric_key), values in sorted(grouped.items()):
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            summary_rows.append(
                {
                    "family": family,
                    "display_name": display_name,
                    "metric_key": metric_key,
                    "n_versions": 0,
                    "mean": "",
                    "std": "",
                }
            )
            continue
        summary_rows.append(
            {
                "family": family,
                "display_name": display_name,
                "metric_key": metric_key,
                "n_versions": len(finite),
                "mean": float(statistics.mean(finite)),
                "std": float(statistics.pstdev(finite) if len(finite) > 1 else 0.0),
            }
        )
    return summary_rows


def _plot_grouped_component_summary(summary_rows: list[dict[str, Any]], out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 20,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    metric_defs = list(METRIC_GROUPS)
    family_order = [family for family, _display, _prefix in MODEL_FAMILIES]
    display_name_map = {family: display for family, display, _prefix in MODEL_FAMILIES}
    color_map = {
        family: plt.get_cmap("tab10")(idx % 10) for idx, family in enumerate(family_order)
    }
    hatch_cycle = ["", "//", "\\\\", "xx", "..", "++", "--", "oo"]
    hatch_map = {
        family: hatch_cycle[idx % len(hatch_cycle)] for idx, family in enumerate(family_order)
    }

    fig, ax = plt.subplots(figsize=(14.8, 4.8), constrained_layout=True)
    x = np.arange(len(metric_defs), dtype=float)
    n_models = len(family_order)
    width = 0.13
    offsets = [(idx - (n_models - 1) / 2.0) * width for idx in range(n_models)]

    for model_idx, family in enumerate(family_order):
        means: list[float] = []
        stds: list[float] = []
        for metric_key, _metric_label in metric_defs:
            row = next(
                (
                    rec
                    for rec in summary_rows
                    if rec["family"] == family and rec["metric_key"] == metric_key
                ),
                None,
            )
            means.append(_safe_float(row.get("mean")) if row is not None else float("nan"))
            stds.append(max(0.0, _safe_float(row.get("std"), 0.0)) if row is not None else float("nan"))

        bars = ax.bar(
            x + offsets[model_idx],
            means,
            width=width,
            yerr=stds,
            capsize=3.0,
            color=color_map[family],
            hatch=hatch_map[family],
            edgecolor="black",
            linewidth=0.5,
            label=display_name_map[family],
            alpha=0.95,
        )
        for bar, value, std in zip(bars, means, stds):
            if not math.isfinite(value):
                continue
            std_val = max(0.0, _safe_float(std, 0.0))
            if value >= 0:
                y_text = value + std_val + 0.02
                va = "bottom"
            else:
                y_text = value - std_val - 0.02
                va = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_text,
                f"{value:.2f}",
                ha="center",
                va=va,
                fontsize=20,
                rotation=90,
            )

    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle=":", alpha=0.9)
    ax.text(
        0.6,
        1.0,
        "QOS",
        va="bottom",
        ha="center",
        fontsize=20,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _key, label in metric_defs], rotation=0, ha="center")
    ax.set_ylabel("Ratio of Metrics")

    ax.legend(loc="lower right", ncol=1, frameon=True)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_best_overall_across_sizes(
    metrics_csv: Path,
    display_name: str,
    run_name: str,
    sizes: list[int],
    out_pdf: Path,
) -> None:
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

    rows = _load_rows_csv(metrics_csv)
    rows_by_size: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            size = int(row.get("size", ""))
        except Exception:
            continue
        rows_by_size[size] = row

    x_sizes = [size for size in sizes if size in rows_by_size]
    depth_vals = [_safe_float(rows_by_size[size].get("qose_depth")) for size in x_sizes]
    cnot_vals = [_safe_float(rows_by_size[size].get("qose_cnot")) for size in x_sizes]
    time_ratio_mean_vals = [_safe_float(rows_by_size[size].get("avg_run_time")) for size in x_sizes]
    time_ratio_sum_vals = [
        _safe_float(rows_by_size[size].get("qose_over_qos_run_time_sum_ratio")) for size in x_sizes
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2), constrained_layout=True)

    axes[0].plot(x_sizes, depth_vals, color="#4C72B0", marker="o", linewidth=2.2, markersize=6)
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    axes[0].set_xlabel("Qubits")
    axes[0].set_ylabel("Depth Ratio")
    axes[0].set_xticks(x_sizes)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[1].plot(x_sizes, cnot_vals, color="#DD8452", marker="o", linewidth=2.2, markersize=6)
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    axes[1].set_xlabel("Qubits")
    axes[1].set_ylabel("CNOT Ratio")
    axes[1].set_xticks(x_sizes)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[2].plot(
        x_sizes,
        time_ratio_mean_vals,
        color="#55A868",
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Ratio, then Mean",
    )
    axes[2].plot(
        x_sizes,
        time_ratio_sum_vals,
        color="#8172B3",
        marker="s",
        linewidth=2.2,
        markersize=5.5,
        label="Sum, then Ratio",
    )
    axes[2].axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    axes[2].set_xlabel("Qubits")
    axes[2].set_ylabel("Time Ratio")
    axes[2].set_xticks(x_sizes)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[2].legend(loc="upper right", ncol=1, frameon=True)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def _load_runtime_summary_for_run(run_name: str) -> dict[str, float]:
    runtime_csv = ROOT / "openevolve_output" / run_name / "runtime_metrics.iterations.csv"
    if not runtime_csv.exists():
        raise FileNotFoundError(f"Runtime metrics CSV not found for {run_name}: {runtime_csv}")

    totals = {
        "llm_elapsed_time_sec_total": 0.0,
        "evaluation_elapsed_time_sec_total": 0.0,
        "prompt_tokens_total": 0.0,
        "completion_tokens_total": 0.0,
        "total_tokens_total": 0.0,
        "reasoning_tokens_total": 0.0,
        "cached_tokens_total": 0.0,
        "iterations_total": 0.0,
    }

    rows = _load_rows_csv(runtime_csv)
    for row in rows:
        totals["llm_elapsed_time_sec_total"] += max(
            0.0, _safe_float(row.get("llm_elapsed_time_sec"), 0.0)
        )
        totals["evaluation_elapsed_time_sec_total"] += max(
            0.0, _safe_float(row.get("evaluation_elapsed_time_sec"), 0.0)
        )
        totals["prompt_tokens_total"] += max(0.0, _safe_float(row.get("prompt_tokens"), 0.0))
        totals["completion_tokens_total"] += max(
            0.0, _safe_float(row.get("completion_tokens"), 0.0)
        )
        totals["total_tokens_total"] += max(0.0, _safe_float(row.get("total_tokens"), 0.0))
        totals["reasoning_tokens_total"] += max(
            0.0, _safe_float(row.get("reasoning_tokens"), 0.0)
        )
        totals["cached_tokens_total"] += max(0.0, _safe_float(row.get("cached_tokens"), 0.0))
        totals["iterations_total"] += 1.0

    totals["runtime_metrics_csv"] = str(runtime_csv)
    return totals


def _plot_runtime_breakdown(summary_rows: list[dict[str, Any]], out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [str(row["display_name"]) for row in summary_rows]
    llm = [max(0.0, _safe_float(row.get("llm_elapsed_time_sec_total"), 0.0)) for row in summary_rows]
    eval_t = [
        max(0.0, _safe_float(row.get("evaluation_elapsed_time_sec_total"), 0.0))
        for row in summary_rows
    ]
    totals = [a + b for a, b in zip(llm, eval_t)]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
    ax.bar(
        x,
        llm,
        width=0.7,
        label="LLM",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x,
        eval_t,
        width=0.7,
        bottom=llm,
        label="Evaluation",
        color="#55A868",
        edgecolor="black",
        linewidth=0.6,
    )
    for xpos, total in zip(x, totals):
        ax.text(xpos, total * 1.01 if total > 0 else 0.0, f"{total/3600.0:.2f}h", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Total Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncol=2, frameon=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_token_breakdown(summary_rows: list[dict[str, Any]], out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [str(row["display_name"]) for row in summary_rows]
    prompt = [max(0.0, _safe_float(row.get("prompt_tokens_total"), 0.0)) for row in summary_rows]
    completion = [
        max(0.0, _safe_float(row.get("completion_tokens_total"), 0.0)) for row in summary_rows
    ]
    totals = [a + b for a, b in zip(prompt, completion)]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
    ax.bar(
        x,
        prompt,
        width=0.7,
        label="Prompt",
        color="#DD8452",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x,
        completion,
        width=0.7,
        bottom=prompt,
        label="Completion",
        color="#8172B3",
        edgecolor="black",
        linewidth=0.6,
    )
    for xpos, total in zip(x, totals):
        ax.text(xpos, total * 1.01 if total > 0 else 0.0, f"{total/1000.0:.1f}k", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Total Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncol=2, frameon=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def _build_price_rows(runtime_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    price_rows: list[dict[str, Any]] = []
    for row in runtime_rows:
        family = str(row["family"])
        price = MODEL_PRICING_USD_PER_1M[family]
        prompt_tokens = max(0.0, _safe_float(row.get("prompt_tokens_total"), 0.0))
        completion_tokens = max(0.0, _safe_float(row.get("completion_tokens_total"), 0.0))
        cached_tokens = max(0.0, _safe_float(row.get("cached_tokens_total"), 0.0))
        non_cached_prompt = max(0.0, prompt_tokens - cached_tokens)
        prompt_cost = (
            non_cached_prompt * price["input"] + cached_tokens * price["cached_input"]
        ) / 1_000_000.0
        completion_cost = (completion_tokens * price["output"]) / 1_000_000.0
        total_cost = prompt_cost + completion_cost
        price_rows.append(
            {
                "family": family,
                "display_name": row["display_name"],
                "run_name": row["run_name"],
                "prompt_tokens_total": prompt_tokens,
                "completion_tokens_total": completion_tokens,
                "cached_tokens_total": cached_tokens,
                "input_price_per_1m_usd": price["input"],
                "cached_input_price_per_1m_usd": price["cached_input"],
                "output_price_per_1m_usd": price["output"],
                "prompt_cost_usd": prompt_cost,
                "completion_cost_usd": completion_cost,
                "total_cost_usd": total_cost,
            }
        )
    return price_rows


def _plot_price_breakdown(price_rows: list[dict[str, Any]], out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [str(row["display_name"]) for row in price_rows]
    prompt_cost = [max(0.0, _safe_float(row.get("prompt_cost_usd"), 0.0)) for row in price_rows]
    completion_cost = [
        max(0.0, _safe_float(row.get("completion_cost_usd"), 0.0)) for row in price_rows
    ]
    totals = [a + b for a, b in zip(prompt_cost, completion_cost)]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
    ax.bar(
        x,
        prompt_cost,
        width=0.7,
        label="Prompt/Input Cost",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x,
        completion_cost,
        width=0.7,
        bottom=prompt_cost,
        label="Completion/Output Cost",
        color="#DD8452",
        edgecolor="black",
        linewidth=0.6,
    )
    for xpos, total in zip(x, totals):
        ax.text(xpos, total * 1.01 if total > 0 else 0.0, f"${total:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Estimated Total Cost (USD)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncol=2, frameon=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the best run from each current sweep model family across all sizes/benches, "
            "cache per-model size-sweep metrics, and build a combined comparison figure."
        )
    )
    parser.add_argument(
        "--sizes",
        default="12,14,16,18,20,22,24",
        help="Comma-separated qubit sizes to evaluate.",
    )
    parser.add_argument(
        "--benches",
        default="",
        help="Optional comma-separated bench names; default uses evaluator BENCHES.",
    )
    parser.add_argument(
        "--families",
        default="",
        help="Optional comma-separated family keys to limit the run.",
    )
    parser.add_argument(
        "--sample-seed",
        default="",
        help="Optional QOSE_SAMPLE_SEED used for evaluator sampling/cache partitioning.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached per-model size-sweep CSVs and recompute requested sizes.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not run evaluation; require cached CSVs and only rebuild summary/figure.",
    )
    parser.add_argument(
        "--max-versions-per-family",
        type=int,
        default=0,
        help="Optional limit of versions per family for all-version comparison (0 means all).",
    )
    args = parser.parse_args()

    sizes = _parse_csv_ints(args.sizes)
    selected_families = (
        {token.strip() for token in args.families.split(",") if token.strip()}
        if args.families.strip()
        else {family for family, _display, _prefix in MODEL_FAMILIES}
    )
    benches = (
        [token.strip() for token in args.benches.split(",") if token.strip()]
        if args.benches.strip()
        else [bench for bench, _label in evaluator_module.BENCHES]
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    selected = _select_best_runs(ROOT / "openevolve_output", selected_families)
    selected_models_csv = DATA_DIR / "selected_models.csv"
    _write_selected_models(selected_models_csv, selected)
    print(f"[start] wrote selected models: {selected_models_csv}")

    # Build current_sweep_best_compare.pdf from all available versions:
    # per run -> size sweep average over requested sizes -> aggregate mean/std across versions.
    all_runs = _select_all_runs(
        ROOT / "openevolve_output",
        selected_families=selected_families,
        max_versions_per_family=args.max_versions_per_family,
    )
    grouped_raw_rows: list[dict[str, Any]] = []
    for run_row in all_runs:
        program = Path(str(run_row["best_program"]))
        run_name = str(run_row["run_name"])
        metrics_csv = SIZE_SWEEP_ALL_VERSIONS_DIR / f"size_sweep_{run_name}_metrics.csv"
        if not args.skip_eval:
            metrics_csv = _evaluate_program_across_sizes(
                program=program,
                run_name=run_name,
                sizes=sizes,
                benches=benches,
                sample_seed=args.sample_seed,
                force_recompute=args.force_recompute,
                output_dir=SIZE_SWEEP_ALL_VERSIONS_DIR,
            )
        elif not metrics_csv.exists():
            print(f"[skip] all-version cached metrics not found for {run_name}: {metrics_csv}")
            continue

        per_run_summary = _load_summary_from_metrics(metrics_csv, sizes)
        grouped_raw_rows.append(
            {
                "family": run_row["family"],
                "display_name": run_row["display_name"],
                "run_name": run_name,
                "version": run_row["version"],
                "metrics_csv": str(metrics_csv),
                **per_run_summary,
            }
        )

    grouped_raw_csv = DATA_DIR / "current_sweep_best_compare_grouped_raw.csv"
    with grouped_raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "display_name",
                "run_name",
                "version",
                "metrics_csv",
                "combined_score",
                "depth_ratio",
                "cnot_ratio",
                "time_ratio_mean",
                "time_ratio_sum",
            ],
        )
        writer.writeheader()
        writer.writerows(grouped_raw_rows)
    print(f"[done] wrote grouped raw: {grouped_raw_csv}")

    grouped_summary_rows = _summarize_family_metric(grouped_raw_rows)
    grouped_summary_csv = DATA_DIR / "current_sweep_best_compare_grouped_summary.csv"
    with grouped_summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "display_name",
                "metric_key",
                "n_versions",
                "mean",
                "std",
            ],
        )
        writer.writeheader()
        writer.writerows(grouped_summary_rows)
    print(f"[done] wrote grouped summary: {grouped_summary_csv}")

    if grouped_summary_rows:
        grouped_fig = FIGURES_DIR / "current_sweep_best_compare.pdf"
        _plot_grouped_component_summary(grouped_summary_rows, grouped_fig)
        print(f"[done] wrote figure: {grouped_fig}")

    summary_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for row in selected:
        program = Path(row["best_program"])
        run_name = row["run_name"]
        runtime_rows.append(
            {
                "family": row["family"],
                "display_name": row["display_name"],
                "run_name": run_name,
                **_load_runtime_summary_for_run(run_name),
            }
        )
        metrics_csv = SIZE_SWEEP_DIR / f"size_sweep_{run_name}_metrics.csv"
        if not args.skip_eval:
            metrics_csv = _evaluate_program_across_sizes(
                program=program,
                run_name=run_name,
                sizes=sizes,
                benches=benches,
                sample_seed=args.sample_seed,
                force_recompute=args.force_recompute,
                output_dir=SIZE_SWEEP_DIR,
            )
        elif not metrics_csv.exists():
            print(f"[skip] cached metrics not found for {run_name}: {metrics_csv}")
            continue

        summary = _load_summary_from_metrics(metrics_csv, sizes)
        summary_rows.append(
            {
                "family": row["family"],
                "display_name": row["display_name"],
                "run_name": run_name,
                "metrics_csv": str(metrics_csv),
                **summary,
            }
        )

    summary_csv = DATA_DIR / "current_sweep_best_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "display_name",
                "run_name",
                "metrics_csv",
                "depth_ratio",
                "cnot_ratio",
                "time_ratio_mean",
                "time_ratio_sum",
                "combined_score",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[done] wrote summary: {summary_csv}")

    runtime_summary_csv = DATA_DIR / "current_sweep_runtime_summary.csv"
    with runtime_summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "display_name",
                "run_name",
                "runtime_metrics_csv",
                "iterations_total",
                "llm_elapsed_time_sec_total",
                "evaluation_elapsed_time_sec_total",
                "prompt_tokens_total",
                "completion_tokens_total",
                "total_tokens_total",
                "reasoning_tokens_total",
                "cached_tokens_total",
            ],
        )
        writer.writeheader()
        writer.writerows(runtime_rows)
    print(f"[done] wrote runtime summary: {runtime_summary_csv}")

    runtime_fig = FIGURES_DIR / "current_sweep_runtime_breakdown.pdf"
    _plot_runtime_breakdown(runtime_rows, runtime_fig)
    print(f"[done] wrote figure: {runtime_fig}")

    token_fig = FIGURES_DIR / "current_sweep_token_breakdown.pdf"
    _plot_token_breakdown(runtime_rows, token_fig)
    print(f"[done] wrote figure: {token_fig}")

    price_rows = _build_price_rows(runtime_rows)
    price_summary_csv = DATA_DIR / "current_sweep_price_summary.csv"
    with price_summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "display_name",
                "run_name",
                "prompt_tokens_total",
                "completion_tokens_total",
                "cached_tokens_total",
                "input_price_per_1m_usd",
                "cached_input_price_per_1m_usd",
                "output_price_per_1m_usd",
                "prompt_cost_usd",
                "completion_cost_usd",
                "total_cost_usd",
            ],
        )
        writer.writeheader()
        writer.writerows(price_rows)
    print(f"[done] wrote price summary: {price_summary_csv}")

    price_fig = FIGURES_DIR / "current_sweep_price_breakdown.pdf"
    _plot_price_breakdown(price_rows, price_fig)
    print(f"[done] wrote figure: {price_fig}")

    if summary_rows:
        best_overall = max(summary_rows, key=lambda row: _safe_float(row.get("combined_score"), float("-inf")))
        best_overall_csv = DATA_DIR / "best_overall_selected.csv"
        with best_overall_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "family",
                    "display_name",
                    "run_name",
                    "metrics_csv",
                    "depth_ratio",
                    "cnot_ratio",
                    "time_ratio_mean",
                    "time_ratio_sum",
                    "combined_score",
                ],
            )
            writer.writeheader()
            writer.writerow(best_overall)
        print(f"[done] wrote best-overall selection: {best_overall_csv}")

        best_overall_fig = FIGURES_DIR / "best_overall_across_sizes.pdf"
        _plot_best_overall_across_sizes(
            metrics_csv=Path(str(best_overall["metrics_csv"])),
            display_name=str(best_overall["display_name"]),
            run_name=str(best_overall["run_name"]),
            sizes=sizes,
            out_pdf=best_overall_fig,
        )
        print(f"[done] wrote figure: {best_overall_fig}")


if __name__ == "__main__":
    main()
