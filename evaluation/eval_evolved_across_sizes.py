#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
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


def _plot(
    rows: list[dict[str, Any]], output_dir: Path, figure_format: str, run_name: str
) -> None:
    sizes = [r["size"] for r in rows]
    depth = [r["qose_depth"] for r in rows]
    cnot = [r["qose_cnot"] for r in rows]
    run_time = [r["avg_run_time"] for r in rows]
    combined = [r["combined_score"] for r in rows]

    x = list(range(len(sizes)))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    ax1.bar([v - width for v in x], depth, width, label="Depth Ratio")
    ax1.bar(x, cnot, width, label="CNOT Ratio")
    ax1.bar([v + width for v in x], run_time, width, label="Time Ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_ylabel("Ratio (QOSE / QOS)")
    ax1.set_title("Depth/CNOT/Time Ratio vs Qubit Size")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend()

    ax2.bar(x, combined, width=0.6, color="#4C72B0")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_xlabel("Qubit Size")
    ax2.set_ylabel("Combined Score")
    ax2.set_title("Combined Score vs Qubit Size")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

    pdf_path = output_dir / f"size_sweep_{run_name}.pdf"
    png_path = output_dir / f"size_sweep_{run_name}.png"
    if figure_format in {"pdf", "both"}:
        fig.savefig(pdf_path)
        print(f"[done] wrote plot: {pdf_path}")
    if figure_format in {"png", "both"}:
        fig.savefig(png_path, dpi=180)
        print(f"[done] wrote plot: {png_path}")


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

    rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    for idx, size in enumerate(sizes, start=1):
        t0 = time.perf_counter()
        print(f"[progress] ({idx}/{len(sizes)}) evaluating size={size} ...", flush=True)

        env_edits = [
            _set_env("QOSE_SIZE_MIN", str(size)),
            _set_env("QOSE_SIZE_MAX", str(size)),
            _set_env("QOSE_STRATIFIED_SIZES", "0"),
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
        row = {
            "size": size,
            "qose_depth": float(metrics.get("qose_depth", float("nan"))),
            "qose_cnot": float(metrics.get("qose_cnot", float("nan"))),
            "qose_overhead": float(metrics.get("qose_overhead", float("nan"))),
            "avg_run_time": float(metrics.get("avg_run_time", float("nan"))),
            "combined_score": float(metrics.get("combined_score", float("nan"))),
            "qose_run_sec_avg": float(artifacts.get("qose_run_sec_avg", float("nan"))),
            "qos_run_sec_avg": float(artifacts.get("qos_run_sec_avg", float("nan"))),
            "eval_elapsed_sec": elapsed,
            "failure_reason": metrics.get("failure_reason", ""),
        }
        rows.append(row)
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

    _plot(rows, output_dir, args.figure_format, run_name)
    total = time.perf_counter() - run_start
    if args.save_metrics:
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
    print(f"[done] total elapsed: {total:.1f}s")


if __name__ == "__main__":
    main()
