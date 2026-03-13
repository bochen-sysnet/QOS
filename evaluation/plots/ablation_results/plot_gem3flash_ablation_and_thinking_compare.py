#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
SIZE_SWEEP_DIR = DATA_DIR / "ablation_size_sweep"

OUTPUT_ROOT = ROOT / "openevolve_output"
ABLATION_ROOT = ROOT / "openevolve_ablation"
ABLATION_ARTIFACT_ROOT = ABLATION_ROOT / "artifact"
ABLATION_THINKING_ROOT = ABLATION_ROOT / "thinking"

# Keep same figure size as plot_gem3flash_seed_diff_full_compare.py
FIGSIZE = (6.6, 4.3)


def _list_version_dirs(root: Path, prefix: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for d in sorted(root.glob(f"{prefix}*")):
        if not d.is_dir() or "_v" not in d.name:
            continue
        suffix = d.name.rsplit("_v", 1)[-1]
        if suffix.isdigit():
            out[int(suffix)] = d
    return out


def _read_score_from_info(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        val = payload.get("metrics", {}).get("combined_score")
        return float(val) if val is not None else None
    except Exception:
        return None


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one size.")
    return sorted(set(values))


def _latest_checkpoint_best_info(run_dir: Path) -> Path | None:
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
        candidate = d / "best_program_info.json"
        if idx > best_idx and candidate.exists():
            best_idx = idx
            best_path = candidate
    return best_path


def _read_combined_score_with_fallback(run_dir: Path) -> tuple[float | None, str]:
    direct = run_dir / "best" / "best_program_info.json"
    score = _read_score_from_info(direct)
    if score is not None:
        return score, "best"
    fallback = _latest_checkpoint_best_info(run_dir)
    if fallback is not None:
        score = _read_score_from_info(fallback)
        if score is not None:
            return score, "checkpoint_fallback"
    return None, "missing"


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
        if idx > best_idx and candidate.exists():
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


def _load_rows_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _set_env(key: str, value: str | None) -> tuple[str, str | None]:
    prev = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    return key, prev


def _restore_env(edits: list[tuple[str, str | None]]) -> None:
    for key, prev in reversed(edits):
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


def _extract_metrics(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(result, "metrics") and hasattr(result, "artifacts"):
        return dict(result.metrics), dict(result.artifacts)
    if isinstance(result, dict):
        return dict(result.get("metrics", {})), dict(result.get("artifacts", {}))
    return {}, {}


def _evaluate_program_across_sizes(
    program: Path,
    run_name: str,
    sizes: list[int],
    benches: list[str],
    sample_seed: str,
    force_recompute: bool,
) -> Path:
    SIZE_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = SIZE_SWEEP_DIR / f"size_sweep_{run_name}_metrics.csv"
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
            "combined_score",
        )
        return all(math.isfinite(_safe_float(row.get(key))) for key in required)

    missing_sizes = [size for size in sizes if not _row_complete(size)]
    if not missing_sizes:
        return metrics_csv_path

    cache_path = _default_cache_path_for_sweep(sizes, sample_seed)
    bench_csv = ",".join(benches)
    for idx, size in enumerate(missing_sizes, start=1):
        print(f"[size-sweep] run={run_name} ({idx}/{len(missing_sizes)}) size={size}", flush=True)
        t0 = time.perf_counter()
        env_edits = [
            _set_env("QOSE_SIZE_MIN", str(size)),
            _set_env("QOSE_SIZE_MAX", str(size)),
            _set_env("QOSE_STRATIFIED_SIZES", "0"),
            _set_env("QOSE_SAMPLES_PER_BENCH", "1"),
            _set_env("QOSE_DISTINCT_SIZES_PER_BENCH", "1"),
            _set_env("QOSE_BENCHES", bench_csv),
            _set_env("QOSE_BASELINE_CACHE_PATH", cache_path),
            _set_env("QOSE_SCORE_MODE", "piecewise"),
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
        summary = artifacts.get("summary", {}) if isinstance(artifacts.get("summary"), dict) else {}
        qose_run_sec_avg = _safe_float(summary.get("qose_run_sec_avg", artifacts.get("qose_run_sec_avg")))
        qos_run_sec_avg = _safe_float(summary.get("qos_run_sec_avg", artifacts.get("qos_run_sec_avg")))
        rows_by_size[size] = {
            "size": int(size),
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
        ordered_rows = [rows_by_size[s] for s in sorted(rows_by_size.keys())]
        with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
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
    return metrics_csv_path


def _load_avg_metric_from_size_sweep(
    metrics_csv: Path, sizes: list[int], metric_key: str
) -> tuple[float | None, int]:
    size_set = set(sizes)
    if metric_key == "qose_over_qos_run_time_sum_ratio":
        qose_sum = 0.0
        qos_sum = 0.0
        n = 0
        for row in _load_rows_csv(metrics_csv):
            try:
                size = int(row.get("size", ""))
            except Exception:
                continue
            if size not in size_set:
                continue
            qose = _safe_float(row.get("qose_run_sec_avg"))
            qos = _safe_float(row.get("qos_run_sec_avg"))
            if math.isfinite(qose) and math.isfinite(qos) and qos > 0.0:
                qose_sum += qose
                qos_sum += qos
                n += 1
        if n == 0 or qos_sum <= 0.0:
            return None, 0
        return float(qose_sum / qos_sum), n

    values: list[float] = []
    for row in _load_rows_csv(metrics_csv):
        try:
            size = int(row.get("size", ""))
        except Exception:
            continue
        if size not in size_set:
            continue
        value = _safe_float(row.get(metric_key))
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None, 0
    return float(sum(values) / len(values)), len(values)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _collect_series_rows(series_specs: list[tuple[str, list[Path], str]]) -> list[dict]:
    rows: list[dict] = []
    for label, roots, prefix in series_specs:
        # Prefer the first root in `roots` when the same version exists in multiple roots.
        versions: dict[int, Path] = {}
        for root in roots:
            version_dirs = _list_version_dirs(root, prefix)
            for version in sorted(version_dirs.keys()):
                versions.setdefault(version, version_dirs[version])
        for version, run_dir in sorted(versions.items()):
            score, source = _read_combined_score_with_fallback(run_dir)
            rows.append(
                {
                    "series": label,
                    "version": int(version),
                    "run_dir": str(run_dir),
                    "score_source": source,
                    "combined_score": "" if score is None else float(score),
                    "available": int(score is not None),
                }
            )
    return rows


def _collect_series_rows_from_size_sweep(
    series_specs: list[tuple[str, list[Path], str]],
    sizes: list[int],
    benches: list[str],
    sample_seed: str,
    force_recompute: bool,
    skip_eval: bool,
    metric_key: str,
) -> list[dict]:
    rows: list[dict] = []
    for label, roots, prefix in series_specs:
        versions: dict[int, Path] = {}
        for root in roots:
            version_dirs = _list_version_dirs(root, prefix)
            for version in sorted(version_dirs.keys()):
                versions.setdefault(version, version_dirs[version])

        for version, run_dir in sorted(versions.items()):
            program, program_source = _resolve_best_program(run_dir)
            if program is None:
                rows.append(
                    {
                        "series": label,
                        "version": int(version),
                        "run_dir": str(run_dir),
                        "program_source": "missing",
                        "score_source": "missing_program",
                        "combined_score": "",
                        "n_sizes": 0,
                        "available": 0,
                    }
                )
                continue

            metrics_csv = SIZE_SWEEP_DIR / f"size_sweep_{run_dir.name}_metrics.csv"
            if not skip_eval:
                metrics_csv = _evaluate_program_across_sizes(
                    program=program,
                    run_name=run_dir.name,
                    sizes=sizes,
                    benches=benches,
                    sample_seed=sample_seed,
                    force_recompute=force_recompute,
                )
            elif not metrics_csv.exists():
                rows.append(
                    {
                        "series": label,
                        "version": int(version),
                        "run_dir": str(run_dir),
                        "program_source": program_source,
                        "score_source": "missing_size_sweep",
                        "combined_score": "",
                        "n_sizes": 0,
                        "available": 0,
                    }
                )
                continue

            avg_score, n_sizes = _load_avg_metric_from_size_sweep(metrics_csv, sizes, metric_key)
            rows.append(
                {
                    "series": label,
                    "version": int(version),
                    "run_dir": str(run_dir),
                    "program_source": program_source,
                    "score_source": "size_sweep_avg" if avg_score is not None else "missing_size_values",
                    "combined_score": "" if avg_score is None else float(avg_score),
                    "metric_key": metric_key,
                    "n_sizes": int(n_sizes),
                    "available": int(avg_score is not None),
                    "metrics_csv": str(metrics_csv),
                }
            )
    return rows


def _summarize_series(rows: list[dict], order: list[str], value_key: str = "combined_score") -> list[dict]:
    summary: list[dict] = []
    for series in order:
        vals: list[float] = []
        for r in rows:
            if r["series"] != series:
                continue
            raw = r[value_key]
            if isinstance(raw, (int, float)):
                vals.append(float(raw))
            elif str(raw).strip():
                vals.append(float(raw))
        if not vals:
            summary.append(
                {
                    "series": series,
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
                "series": series,
                "n_runs": len(vals),
                "mean": float(statistics.mean(vals)),
                "std": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
                "min": float(min(vals)),
                "max": float(max(vals)),
            }
        )
    return summary


def _plot_aggregated_bars(
    summary_rows: list[dict],
    order: list[str],
    colors: dict[str, str],
    out_pdf: Path,
    title: str = "",
    y_label: str = "Best Combined Score",
) -> None:
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []
    for name in order:
        rec = next((r for r in summary_rows if r["series"] == name), None)
        if rec is None or str(rec.get("mean", "")).strip() == "":
            means.append(float("nan"))
            stds.append(0.0)
            counts.append(0)
        else:
            means.append(float(rec["mean"]))
            stds.append(float(rec["std"]))
            counts.append(int(rec["n_runs"]))

    x = np.arange(len(order), dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[colors.get(k, "#777777") for k in order],
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, m, s, n in zip(bars, means, stds, counts):
        if math.isnan(m):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.01,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                m + max(s, 0.0) + 0.01,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=20,
            )

    ax.set_xticks(x, order)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate gem3flash ablation/thinking results across versions. "
            "Ablation figure uses average combined score across size sweep."
        )
    )
    parser.add_argument(
        "--sizes",
        default="12,14,16,18,20,22,24",
        help="Comma-separated sizes used to average size-sweep combined score for ablation figure.",
    )
    parser.add_argument(
        "--benches",
        default="",
        help="Optional comma-separated benches; default uses evaluator BENCHES.",
    )
    parser.add_argument(
        "--sample-seed",
        default="",
        help="Optional QOSE_SAMPLE_SEED for ablation size sweeps.",
    )
    parser.add_argument(
        "--skip-ablation-size-sweep-eval",
        action="store_true",
        help="Do not run evaluator for ablation size sweep; use only cached CSVs.",
    )
    parser.add_argument(
        "--force-recompute-ablation-size-sweep",
        action="store_true",
        help="Recompute ablation size-sweep CSVs even if cached.",
    )
    args = parser.parse_args()

    sizes = _parse_csv_ints(args.sizes)
    benches = (
        [token.strip() for token in args.benches.split(",") if token.strip()]
        if args.benches.strip()
        else [bench for bench, _label in evaluator_module.BENCHES]
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Ablation variants aggregated across all available versions.
    ablation_specs = [
        ("1111", [OUTPUT_ROOT], "gem3flash_pws8_22q_seed_low_full_v"),
        ("0111", [ABLATION_ARTIFACT_ROOT, ABLATION_ROOT], "gem3flash_pws8_22q_noseed_full_v"),
        ("0011", [ABLATION_ARTIFACT_ROOT, ABLATION_ROOT], "gem3flash_pws8_22q_noseed_no_cases_v"),
        (
            "0001",
            [ABLATION_ARTIFACT_ROOT, ABLATION_ROOT],
            "gem3flash_pws8_22q_noseed_no_cases_no_summary_v",
        ),
        (
            "0000",
            [ABLATION_ARTIFACT_ROOT, ABLATION_ROOT],
            "gem3flash_pws8_22q_noseed_no_cases_no_summary_no_metadata_v",
        ),
    ]
    ablation_order = [s[0] for s in ablation_specs]
    ablation_rows = _collect_series_rows_from_size_sweep(
        ablation_specs,
        sizes=sizes,
        benches=benches,
        sample_seed=args.sample_seed,
        force_recompute=args.force_recompute_ablation_size_sweep,
        skip_eval=args.skip_ablation_size_sweep_eval,
        metric_key="qose_over_qos_run_time_sum_ratio",
    )
    ablation_summary = _summarize_series(ablation_rows, ablation_order)
    _write_csv(DATA_DIR / "gem3flash_ablation_versions_time_ratio_raw.csv", ablation_rows)
    _write_csv(DATA_DIR / "gem3flash_ablation_versions_time_ratio_summary.csv", ablation_summary)
    _plot_aggregated_bars(
        ablation_summary,
        ablation_order,
        {
            "1111": "#4C78A8",
            "0111": "#F58518",
            "0011": "#54A24B",
            "0001": "#B279A2",
            "0000": "#72B7B2",
        },
        FIGURES_DIR / "gem3flash_ablation_versions_combined_score_aggregated.pdf",
        "",
        y_label="Time Ratio",
    )

    # Figure 2: Thinking levels aggregated across all available versions.
    thinking_specs = [
        ("Low", [OUTPUT_ROOT], "gem3flash_pws8_22q_seed_low_full_v"),
        ("Medium", [ABLATION_THINKING_ROOT, ABLATION_ROOT], "gem3flash_pws8_22q_seed_medium_full_v"),
        ("High", [ABLATION_THINKING_ROOT, ABLATION_ROOT], "gem3flash_pws8_22q_seed_high_full_v"),
    ]
    thinking_order = [s[0] for s in thinking_specs]
    thinking_rows = _collect_series_rows_from_size_sweep(
        thinking_specs,
        sizes=sizes,
        benches=benches,
        sample_seed=args.sample_seed,
        force_recompute=args.force_recompute_ablation_size_sweep,
        skip_eval=args.skip_ablation_size_sweep_eval,
        metric_key="qose_over_qos_run_time_sum_ratio",
    )
    thinking_summary = _summarize_series(thinking_rows, thinking_order)
    _write_csv(DATA_DIR / "gem3flash_thinking_versions_time_ratio_raw.csv", thinking_rows)
    _write_csv(DATA_DIR / "gem3flash_thinking_versions_time_ratio_summary.csv", thinking_summary)
    _plot_aggregated_bars(
        thinking_summary,
        thinking_order,
        {
            "Low": "#4C78A8",
            "Medium": "#F58518",
            "High": "#54A24B",
        },
        FIGURES_DIR / "gem3flash_thinking_versions_combined_score_aggregated.pdf",
        "",
        y_label="Time Ratio",
    )

    print("Wrote figures and CSVs under:", OUT_DIR)


if __name__ == "__main__":
    main()
