#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qos.error_mitigator import evaluator as evaluator_module


def _parse_sizes(raw: str) -> list[int]:
    if not raw:
        return [12, 14, 16, 18, 20, 22, 24]
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError("No valid sizes provided")
    return vals


def _extract_metrics(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(result, "metrics") and hasattr(result, "artifacts"):
        return dict(result.metrics), dict(result.artifacts)
    if isinstance(result, dict):
        return dict(result.get("metrics", {})), dict(result.get("artifacts", {}))
    return {}, {}


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


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_ratio_for_score(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float(numerator)
    return float(numerator) / float(denominator)


def _score_mode_for_cases() -> str:
    raw = os.getenv("QOSE_SCORE_MODE", "legacy").strip().lower()
    if raw in {"pwl", "piecewise", "piecewise_linear"}:
        return "piecewise"
    return "legacy"


def _combined_score_case(
    depth_ratio: float, cnot_ratio: float, time_ratio: float, overhead_ratio: float
) -> float:
    mode = _score_mode_for_cases()
    if mode == "piecewise":
        struct_delta = 1.0 - ((depth_ratio + cnot_ratio) / 2.0)
        time_delta = 1.0 - time_ratio
        struct_term = struct_delta if struct_delta >= 0 else 8.0 * struct_delta
        return struct_term + time_delta
    return -(depth_ratio + cnot_ratio + overhead_ratio + time_ratio)


def _extract_case_rows(artifacts: dict[str, Any], fallback_size: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = artifacts.get("cases", [])
    if not isinstance(cases, list):
        return out
    for case in cases:
        if not isinstance(case, dict):
            continue
        bench = str(case.get("bench", "")).strip()
        if not bench:
            continue
        try:
            size = int(case.get("size", fallback_size))
        except Exception:
            size = fallback_size
        qose_run_sec = _safe_float(case.get("qose_run_sec", float("nan")))
        qos_run_sec = _safe_float(case.get("qos_run_sec", float("nan")))
        if math.isfinite(qose_run_sec) and math.isfinite(qos_run_sec) and qos_run_sec > 0:
            time_ratio = qose_run_sec / qos_run_sec
        else:
            time_ratio = float("nan")
        qose_depth = _safe_float(case.get("qose_depth", float("nan")))
        qos_depth = _safe_float(case.get("qos_depth", float("nan")))
        qose_cnot = _safe_float(case.get("qose_cnot", float("nan")))
        qos_cnot = _safe_float(case.get("qos_cnot", float("nan")))
        qose_num_circuits = _safe_float(case.get("qose_num_circuits", float("nan")))
        qos_num_circuits = _safe_float(case.get("qos_num_circuits", float("nan")))
        depth_ratio = _safe_ratio_for_score(qose_depth, qos_depth)
        cnot_ratio = _safe_ratio_for_score(qose_cnot, qos_cnot)
        overhead_ratio = _safe_ratio_for_score(qose_num_circuits, qos_num_circuits)
        if math.isfinite(depth_ratio) and math.isfinite(cnot_ratio) and math.isfinite(time_ratio):
            score_case = _combined_score_case(depth_ratio, cnot_ratio, time_ratio, overhead_ratio)
        else:
            score_case = float("nan")
        out.append(
            {
                "bench": bench,
                "size": size,
                "qose_depth": depth_ratio,
                "qose_cnot": cnot_ratio,
                "qose_run_sec": qose_run_sec,
                "qos_run_sec": qos_run_sec,
                "qose_over_qos_run_time_case_ratio": time_ratio,
                "qose_overhead_case_ratio": overhead_ratio,
                "combined_score_case": score_case,
                "qose_method": str(case.get("qose_method", "")),
                "qose_output_size": case.get("qose_output_size", ""),
                "qose_num_circuits": case.get("qose_num_circuits", ""),
                "qos_num_circuits": case.get("qos_num_circuits", ""),
            }
        )
    return out


def _infer_latest_evolution_log(program: Path) -> Path | None:
    # Expect layout: openevolve_output/<run>/best/best_program.py
    if program.name != "best_program.py":
        return None
    if program.parent.name != "best":
        return None
    run_dir = program.parent.parent
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return None
    logs = sorted(logs_dir.glob("*.log"))
    if not logs:
        return None
    return logs[-1]


def _infer_run_dir(program: Path) -> Path | None:
    # Expect layout: openevolve_output/<run>/best/best_program.py
    if program.name != "best_program.py":
        return None
    if program.parent.name != "best":
        return None
    return program.parent.parent


def _parse_selected_pairs_from_log(log_path: Path) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = defaultdict(int)
    pair_re = re.compile(r"\(([^,\s]+),\s*(\d+)\)")
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Evaluating bench/size pairs:" not in line:
            continue
        for bench, size_s in pair_re.findall(line):
            try:
                counts[(bench, int(size_s))] += 1
            except Exception:
                continue
    return dict(counts)


def _parse_selected_pairs_from_checkpoints(program: Path) -> dict[tuple[str, int], int]:
    run_dir = _infer_run_dir(program)
    if run_dir is None:
        return {}
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return {}
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for path in ckpt_root.glob("checkpoint_*/programs/*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        artifacts_json = data.get("artifacts_json")
        if not artifacts_json:
            continue
        try:
            artifacts = json.loads(artifacts_json)
        except Exception:
            continue
        cases = artifacts.get("cases", [])
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            bench = str(case.get("bench", "")).strip()
            if not bench:
                continue
            try:
                size = int(case.get("size", -1))
            except Exception:
                continue
            if size < 0:
                continue
            counts[(bench, size)] += 1
    return dict(counts)


def _plot_case_metrics_heatmap(
    case_rows: list[dict[str, Any]],
    benches: list[str],
    sizes: list[int],
    output_dir: Path,
    figure_format: str,
    run_name: str,
    selected_counts: dict[tuple[str, int], int] | None = None,
) -> None:
    if not case_rows:
        print("[warn] no case rows available for case-metric heatmap")
        return

    bench_label_map = {b: label for b, label in evaluator_module.BENCHES}
    bench_to_i = {b: i for i, b in enumerate(benches)}
    size_to_j = {s: j for j, s in enumerate(sizes)}

    def _build_matrix(key: str) -> Any:
        import numpy as np

        mat = np.full((len(benches), len(sizes)), np.nan, dtype=float)
        for row in case_rows:
            bench = str(row.get("bench", ""))
            try:
                size = int(row.get("size", -1))
            except Exception:
                continue
            if bench not in bench_to_i or size not in size_to_j:
                continue
            mat[bench_to_i[bench], size_to_j[size]] = _safe_float(row.get(key, float("nan")))
        return mat

    mats = [
        ("Depth Ratio (QOSE/QOS)", _build_matrix("qose_depth")),
        ("CNOT Ratio (QOSE/QOS)", _build_matrix("qose_cnot")),
        ("Time Ratio (QOSE/QOS, case)", _build_matrix("qose_over_qos_run_time_case_ratio")),
        ("Combined Score (per-case formula)", _build_matrix("combined_score_case")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(19.0, 11.0), constrained_layout=True)
    axes = axes.ravel()
    for ax, (title, mat) in zip(axes, mats):
        im = ax.imshow(mat, aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha="right")
        ax.set_yticks(range(len(benches)))
        ax.set_yticklabels([bench_label_map.get(b, b) for b in benches])
        ax.set_xlabel("Qubit Size")
        if ax is axes[0]:
            ax.set_ylabel("Benchmark")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        for i in range(len(benches)):
            for j in range(len(sizes)):
                val = mat[i, j]
                if math.isfinite(float(val)):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val > 1.15 else "black",
                    )

                if selected_counts:
                    bench = benches[i]
                    size = sizes[j]
                    cnt = selected_counts.get((bench, size), 0)
                    if cnt > 0:
                        ax.scatter(j, i, marker="*", s=45, c="red", edgecolors="black", linewidths=0.4)
                        ax.text(
                            j - 0.42,
                            i - 0.33,
                            str(cnt),
                            fontsize=6,
                            color="red",
                            ha="left",
                            va="top",
                            fontweight="bold",
                        )

    fig.suptitle(
        "Per-circuit Metrics Across All Sizes (63 circuits)"
        + ("; red * = selected in evolution logs" if selected_counts else ""),
        fontsize=12,
    )
    pdf_path = output_dir / f"size_sweep_{run_name}_cases_heatmap.pdf"
    png_path = output_dir / f"size_sweep_{run_name}_cases_heatmap.png"
    if figure_format in {"pdf", "both"}:
        fig.savefig(pdf_path)
        print(f"[done] wrote case heatmap: {pdf_path}")
    if figure_format in {"png", "both"}:
        fig.savefig(png_path, dpi=180)
        print(f"[done] wrote case heatmap: {png_path}")
    plt.close(fig)


def _plot(
    rows: list[dict[str, Any]], output_dir: Path, figure_format: str, run_name: str
) -> None:
    def _annotate_bars(ax, bars, fmt: str = "{:.3f}") -> None:
        for bar in bars:
            h = bar.get_height()
            if h is None:
                continue
            try:
                if not math.isfinite(float(h)):
                    continue
            except Exception:
                continue
            x = bar.get_x() + bar.get_width() / 2.0
            if h >= 0:
                y = h
                va = "bottom"
                dy = 2
            else:
                y = h
                va = "top"
                dy = -2
            ax.annotate(
                fmt.format(h),
                (x, y),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=7,
                rotation=90,
            )

    sizes = [r["size"] for r in rows]
    depth = [r["qose_depth"] for r in rows]
    cnot = [r["qose_cnot"] for r in rows]
    run_time = [r["avg_run_time"] for r in rows]
    run_time_sum_ratio = [r["qose_over_qos_run_time_sum_ratio"] for r in rows]
    combined = [r["combined_score"] for r in rows]

    x = list(range(len(sizes)))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    depth_bars = ax1.bar([v - 1.5 * width for v in x], depth, width, label="Depth Ratio (QOSE/QOS)")
    cnot_bars = ax1.bar([v - 0.5 * width for v in x], cnot, width, label="CNOT Ratio (QOSE/QOS)")
    time_bars = ax1.bar([v + 0.5 * width for v in x], run_time, width, label="Time Ratio (QOSE/QOS, mean)")
    sum_time_bars = ax1.bar(
        [v + 1.5 * width for v in x],
        run_time_sum_ratio,
        width,
        label="Time Ratio (QOSE/QOS, sum)",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_ylabel("Ratio")
    ax1.set_title("Depth/CNOT/Time Ratios vs Qubit Size")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend()
    _annotate_bars(ax1, depth_bars)
    _annotate_bars(ax1, cnot_bars)
    _annotate_bars(ax1, time_bars)
    _annotate_bars(ax1, sum_time_bars)

    combined_bars = ax2.bar(x, combined, width=0.6, color="#4C72B0")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_xlabel("Qubit Size")
    ax2.set_ylabel("Combined Score")
    ax2.set_title("Combined Score vs Qubit Size")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    _annotate_bars(ax2, combined_bars)

    pdf_path = output_dir / f"size_sweep_{run_name}.pdf"
    png_path = output_dir / f"size_sweep_{run_name}.png"
    if figure_format in {"pdf", "both"}:
        fig.savefig(pdf_path)
        print(f"[done] wrote plot: {pdf_path}")
    if figure_format in {"png", "both"}:
        fig.savefig(png_path, dpi=180)
        print(f"[done] wrote plot: {png_path}")


def _load_rows_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _load_case_rows_csv(path: Path) -> list[dict[str, Any]]:
    return _load_rows_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one evolved program with qos.error_mitigator.evaluator "
            "across qubit sizes and plot metric trends."
        )
    )
    parser.add_argument("--program", required=True, help="Path to evolved program .py")
    parser.add_argument(
        "--sizes",
        default="12,14,16,18,20,22,24",
        help="Comma-separated qubit sizes (default: 12,14,...,24)",
    )
    parser.add_argument(
        "--benches",
        default="",
        help="Optional comma-separated bench names; default uses evaluator BENCHES",
    )
    parser.add_argument(
        "--sample-seed",
        default="",
        help="Optional QOSE_SAMPLE_SEED for deterministic pair sampling/cache partitioning",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/plots/size_sweep",
        help="Output directory. Default: evaluation/plots/size_sweep",
    )
    parser.add_argument(
        "--figure-format",
        choices=("pdf", "png", "both"),
        default="pdf",
        help="Figure output format (default: pdf)",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Also write CSV/JSON metrics files (default: off)",
    )
    parser.add_argument(
        "--save-case-metrics",
        action="store_true",
        help="Write per-circuit case metrics (bench,size rows) as CSV/JSON.",
    )
    parser.add_argument(
        "--plot-case-metrics",
        action="store_true",
        help="Plot per-circuit case-metric heatmaps across all sizes/benches.",
    )
    parser.add_argument(
        "--evolution-log",
        default="",
        help="Optional evolution log path to annotate selected (bench,size) pairs.",
    )
    parser.add_argument(
        "--no-annotate-selected",
        action="store_true",
        help="Disable annotation for (bench,size) pairs selected in evolution logs.",
    )
    parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="Do not reuse existing size_sweep_* metrics/cases; recompute all requested sizes.",
    )
    args = parser.parse_args()

    program = Path(args.program)
    if not program.exists():
        raise FileNotFoundError(f"Program not found: {program}")

    sizes = _parse_sizes(args.sizes)
    run_name = program.parent.parent.name if program.parent.name == "best" else program.stem
    output_dir = Path(args.output_dir) if args.output_dir.strip() else Path("evaluation/plots/size_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.benches.strip():
        benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    else:
        benches = [b for b, _label in evaluator_module.BENCHES]
    bench_csv = ",".join(benches)

    print(f"[start] program={program}")
    print(f"[start] benches={bench_csv}")
    print(f"[start] sizes={sizes}")
    if args.sample_seed:
        print(f"[start] sample_seed={args.sample_seed}")
    cache_path = _default_cache_path_for_sweep(sizes, args.sample_seed)
    print(f"[start] baseline_cache={cache_path}")

    metrics_csv_path = output_dir / f"size_sweep_{run_name}_metrics.csv"
    case_csv_path = output_dir / f"size_sweep_{run_name}_cases.csv"

    rows_by_size: dict[int, dict[str, Any]] = {}
    case_rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}

    if not args.no_reuse_existing:
        existing_rows = _load_rows_csv(metrics_csv_path)
        for r in existing_rows:
            try:
                s = int(r.get("size", ""))
            except Exception:
                continue
            r_norm = dict(r)
            r_norm["size"] = s
            for key in (
                "qose_depth",
                "qose_cnot",
                "qose_overhead",
                "avg_run_time",
                "combined_score",
                "qose_run_sec_avg",
                "qos_run_sec_avg",
                "qose_over_qos_run_time_sum_ratio",
                "eval_elapsed_sec",
            ):
                r_norm[key] = _safe_float(r_norm.get(key, float("nan")))
            rows_by_size[s] = r_norm
        existing_case_rows = _load_case_rows_csv(case_csv_path)
        for r in existing_case_rows:
            bench = str(r.get("bench", "")).strip()
            try:
                s = int(r.get("size", ""))
            except Exception:
                continue
            if not bench:
                continue
            r_norm = dict(r)
            r_norm["size"] = s
            for key in (
                "qose_depth",
                "qose_cnot",
                "qose_run_sec",
                "qos_run_sec",
                "qose_over_qos_run_time_case_ratio",
                "qose_overhead_case_ratio",
                "combined_score_case",
            ):
                r_norm[key] = _safe_float(r_norm.get(key, float("nan")))
            case_rows_by_key[(bench, s)] = r_norm
        if rows_by_size or case_rows_by_key:
            print(
                f"[start] reuse loaded metrics_sizes={len(rows_by_size)} "
                f"case_pairs={len(case_rows_by_key)}"
            )

    def _size_has_full_case_coverage(size_val: int) -> bool:
        expected = {(b, size_val) for b in benches}
        return expected.issubset(set(case_rows_by_key.keys()))

    needs_case_rows = bool(args.save_case_metrics or args.plot_case_metrics)
    missing_sizes: list[int] = []
    for s in sizes:
        has_row = s in rows_by_size
        has_cases = _size_has_full_case_coverage(s)
        if has_row and (has_cases or not needs_case_rows):
            continue
        missing_sizes.append(s)

    if missing_sizes:
        print(f"[start] evaluating missing sizes: {missing_sizes}")
    else:
        print("[start] all requested sizes already available; reusing existing results.")

    run_start = time.perf_counter()
    for idx, size in enumerate(missing_sizes, start=1):
        t0 = time.perf_counter()
        print(f"[progress] ({idx}/{len(missing_sizes)}) evaluating size={size} ...", flush=True)

        env_edits = [
            _set_env("QOSE_SIZE_MIN", str(size)),
            _set_env("QOSE_SIZE_MAX", str(size)),
            _set_env("QOSE_STRATIFIED_SIZES", "0"),
            _set_env("QOSE_SAMPLES_PER_BENCH", "1"),
            _set_env("QOSE_DISTINCT_SIZES_PER_BENCH", "1"),
            _set_env("QOSE_BENCHES", bench_csv),
            _set_env("QOSE_BASELINE_CACHE_PATH", cache_path),
        ]
        if args.sample_seed:
            env_edits.append(_set_env("QOSE_SAMPLE_SEED", args.sample_seed))
        else:
            env_edits.append(_set_env("QOSE_SAMPLE_SEED", None))

        try:
            result = evaluator_module.evaluate(str(program))
            metrics, artifacts = _extract_metrics(result)
        finally:
            _restore_env(env_edits)

        elapsed = time.perf_counter() - t0
        summary = artifacts.get("summary", {}) if isinstance(artifacts.get("summary", {}), dict) else {}
        qose_run_sec_avg = float(
            summary.get("qose_run_sec_avg", artifacts.get("qose_run_sec_avg", float("nan")))
        )
        qos_run_sec_avg = float(
            summary.get("qos_run_sec_avg", artifacts.get("qos_run_sec_avg", float("nan")))
        )
        row = {
            "size": size,
            "qose_depth": float(metrics.get("qose_depth", float("nan"))),
            "qose_cnot": float(metrics.get("qose_cnot", float("nan"))),
            "qose_overhead": float(metrics.get("qose_overhead", float("nan"))),
            "avg_run_time": float(metrics.get("avg_run_time", float("nan"))),
            "combined_score": float(metrics.get("combined_score", float("nan"))),
            "qose_run_sec_avg": qose_run_sec_avg,
            "qos_run_sec_avg": qos_run_sec_avg,
            "qose_over_qos_run_time_sum_ratio": (
                qose_run_sec_avg / qos_run_sec_avg
                if qos_run_sec_avg > 0.0
                else float("nan")
            ),
            "eval_elapsed_sec": elapsed,
            "failure_reason": metrics.get("failure_reason", ""),
        }
        rows_by_size[size] = row
        size_cases = _extract_case_rows(artifacts, size)
        for cr in size_cases:
            case_rows_by_key[(str(cr.get("bench", "")).strip(), int(cr.get("size", size)))] = cr
        print(
            "[progress] done size=%s combined=%.4f depth=%.4f cnot=%.4f time=%.4f elapsed=%.1fs"
            % (
                size,
                row["combined_score"],
                row["qose_depth"],
                row["qose_cnot"],
                row["avg_run_time"],
                elapsed,
            ),
            flush=True,
        )
        if size_cases:
            print(f"[progress] collected per-circuit cases size={size}: {len(size_cases)}", flush=True)

    rows = [rows_by_size[s] for s in sizes if s in rows_by_size]
    case_rows_all = [case_rows_by_key[(b, s)] for s in sizes for b in benches if (b, s) in case_rows_by_key]
    _plot(rows, output_dir, args.figure_format, run_name)
    total = time.perf_counter() - run_start
    persist_metrics = bool(args.save_metrics or not args.no_reuse_existing)
    persist_case_metrics = bool(
        args.save_case_metrics or args.plot_case_metrics or not args.no_reuse_existing
    )

    if persist_metrics:
        csv_path = output_dir / f"size_sweep_{run_name}_metrics.csv"
        json_path = output_dir / f"size_sweep_{run_name}_metrics.json"
        fields = [
            "size",
            "qose_depth",
            "qose_cnot",
            "qose_overhead",
            "avg_run_time",
            "combined_score",
            "qose_run_sec_avg",
            "qos_run_sec_avg",
            "qose_over_qos_run_time_sum_ratio",
            "eval_elapsed_sec",
            "failure_reason",
        ]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with json_path.open("w") as f:
            json.dump(rows, f, indent=2)
        print(f"[done] wrote metrics: {csv_path}")
        print(f"[done] wrote metrics: {json_path}")

    selected_counts: dict[tuple[str, int], int] = {}
    if not args.no_annotate_selected:
        log_path: Path | None
        if args.evolution_log.strip():
            log_path = Path(args.evolution_log)
        else:
            log_path = _infer_latest_evolution_log(program)
        if log_path and log_path.exists():
            selected_counts = _parse_selected_pairs_from_log(log_path)
            print(
                f"[done] parsed selected pairs from log: {log_path} "
                f"(unique_pairs={len(selected_counts)})"
            )
            if not selected_counts:
                selected_counts = _parse_selected_pairs_from_checkpoints(program)
                if selected_counts:
                    print(
                        "[done] log had no pair lines; parsed selected pairs from checkpoints "
                        f"(unique_pairs={len(selected_counts)})"
                    )
        else:
            if args.evolution_log.strip():
                print(f"[warn] evolution log not found: {args.evolution_log}")
            else:
                print("[warn] could not infer evolution log; skipping selection annotations")
            selected_counts = _parse_selected_pairs_from_checkpoints(program)
            if selected_counts:
                print(
                    "[done] parsed selected pairs from checkpoints "
                    f"(unique_pairs={len(selected_counts)})"
                )

    if persist_case_metrics:
        case_json_path = output_dir / f"size_sweep_{run_name}_cases.json"
        case_fields = [
            "bench",
            "size",
            "qose_depth",
            "qose_cnot",
            "qose_run_sec",
            "qos_run_sec",
            "qose_over_qos_run_time_case_ratio",
            "qose_overhead_case_ratio",
            "combined_score_case",
            "qose_method",
            "qose_output_size",
            "qose_num_circuits",
            "qos_num_circuits",
        ]
        with case_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=case_fields)
            writer.writeheader()
            writer.writerows(case_rows_all)
        with case_json_path.open("w") as f:
            json.dump(case_rows_all, f, indent=2)
        print(f"[done] wrote case metrics: {case_csv_path}")
        print(f"[done] wrote case metrics: {case_json_path}")

    if args.plot_case_metrics:
        _plot_case_metrics_heatmap(
            case_rows_all,
            benches,
            sizes,
            output_dir,
            args.figure_format,
            run_name,
            selected_counts=selected_counts if selected_counts else None,
        )

    if case_rows_all:
        expected = len(benches) * len(sizes)
        print(f"[done] per-circuit cases collected: {len(case_rows_all)} (expected ~{expected})")
    print(f"[done] total elapsed: {total:.1f}s")


if __name__ == "__main__":
    main()
