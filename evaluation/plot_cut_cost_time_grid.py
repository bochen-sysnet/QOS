import argparse
import csv
import datetime as dt
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.full_eval import BENCHES, _load_qasm_circuit
from qos.error_mitigator.run import compute_gv_cost, compute_wc_cost
from qos.types.types import Qernel


def _import_matplotlib():
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore

    return plt, Line2D


def _parse_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def _parse_benches(value: str) -> list[tuple[str, str]]:
    if value.strip().lower() == "all":
        return BENCHES
    wanted = [b.strip() for b in value.split(",") if b.strip()]
    wanted_set = set(wanted)
    out = [(b, lbl) for (b, lbl) in BENCHES if b in wanted_set]
    missing = [b for b in wanted if b not in {x for x, _ in out}]
    if missing:
        raise ValueError(f"Unknown bench(es): {', '.join(missing)}")
    if not out:
        raise ValueError("No valid benches selected.")
    return out


def _compute_once(
    bench: str,
    qubits: int,
    size_to_reach: int,
    method: str,
    clingo_timeout_sec: int,
    max_partition_tries: int,
) -> tuple[float, float]:
    os.environ["QVM_CLINGO_TIMEOUT_SEC"] = str(clingo_timeout_sec)
    os.environ["QVM_MAX_PARTITION_TRIES"] = str(max_partition_tries)
    qc = _load_qasm_circuit(bench, qubits)
    q = Qernel(qc.copy())
    t0 = time.perf_counter()
    if method == "GV":
        cost, _ = compute_gv_cost(q, size_to_reach, timeout_sec=0)
        cost = float(cost)
    elif method == "WC":
        cost, _ = compute_wc_cost(q, size_to_reach, timeout_sec=0)
        cost = float(cost)
    else:
        raise ValueError(f"Unknown method: {method}")
    elapsed = time.perf_counter() - t0
    return cost, elapsed


def _worker(
    result_queue: mp.Queue,
    bench: str,
    qubits: int,
    size_to_reach: int,
    method: str,
    clingo_timeout_sec: int,
    max_partition_tries: int,
) -> None:
    try:
        cost, elapsed = _compute_once(
            bench,
            qubits,
            size_to_reach,
            method,
            clingo_timeout_sec,
            max_partition_tries,
        )
        result_queue.put({"ok": True, "cost": cost, "elapsed_sec": elapsed})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


def _compute_with_optional_timeout(
    bench: str,
    qubits: int,
    size_to_reach: int,
    method: str,
    timeout_sec: int,
    clingo_timeout_sec: int,
    max_partition_tries: int,
) -> dict[str, Any]:
    if timeout_sec <= 0:
        try:
            cost, elapsed = _compute_once(
                bench,
                qubits,
                size_to_reach,
                method,
                clingo_timeout_sec,
                max_partition_tries,
            )
            return {
                "cost": cost,
                "elapsed_sec": elapsed,
                "timed_out": False,
                "error": "",
            }
        except Exception as exc:
            return {
                "cost": float("nan"),
                "elapsed_sec": float("nan"),
                "timed_out": False,
                "error": str(exc),
            }

    mp_ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    result_queue: mp.Queue = mp_ctx.Queue()
    proc = mp_ctx.Process(
        target=_worker,
        args=(
            result_queue,
            bench,
            qubits,
            size_to_reach,
            method,
            clingo_timeout_sec,
            max_partition_tries,
        ),
    )
    proc.start()
    proc.join(timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "cost": float("nan"),
            "elapsed_sec": float("nan"),
            "timed_out": True,
            "error": f"timeout after {timeout_sec}s",
        }

    try:
        result = result_queue.get_nowait()
    except queue_mod.Empty:
        return {
            "cost": float("nan"),
            "elapsed_sec": float("nan"),
            "timed_out": False,
            "error": "no result returned",
        }

    if not result.get("ok"):
        return {
            "cost": float("nan"),
            "elapsed_sec": float("nan"),
            "timed_out": False,
            "error": result.get("error", "unknown error"),
        }
    return {
        "cost": float(result["cost"]),
        "elapsed_sec": float(result["elapsed_sec"]),
        "timed_out": False,
        "error": "",
    }


def _plot_grid(
    records: list[dict[str, Any]],
    benches: list[tuple[str, str]],
    qubits: list[int],
    out_pdf: Path,
) -> None:
    plt, Line2D = _import_matplotlib()
    nrows = len(benches)
    ncols = len(qubits)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 2.9 * nrows),
        squeeze=False,
        sharex=False,
    )

    for r_idx, (bench, bench_label) in enumerate(benches):
        for c_idx, nq in enumerate(qubits):
            ax = axes[r_idx, c_idx]
            subset = [
                r
                for r in records
                if r["bench"] == bench and int(r["qubits"]) == int(nq)
            ]
            x_vals = sorted({int(r["size_to_reach"]) for r in subset})

            def _series(method: str, field: str) -> list[float]:
                out: list[float] = []
                for x in x_vals:
                    row = next(
                        (
                            rr
                            for rr in subset
                            if rr["method"] == method and int(rr["size_to_reach"]) == x
                        ),
                        None,
                    )
                    out.append(float(row[field]) if row is not None else float("nan"))
                return out

            gv_cost = _series("GV", "cost")
            wc_cost = _series("WC", "cost")
            gv_time = _series("GV", "elapsed_sec")
            wc_time = _series("WC", "elapsed_sec")

            ax.plot(x_vals, gv_cost, color="#1f77b4", marker="o", linewidth=1.3, markersize=3)
            ax.plot(x_vals, wc_cost, color="#ff7f0e", marker="o", linewidth=1.3, markersize=3)
            ax.set_title(f"{bench_label} | {nq}q", fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            if c_idx == 0:
                ax.set_ylabel("Cost", fontsize=8)

            ax2 = ax.twinx()
            ax2.plot(
                x_vals,
                gv_time,
                color="#1f77b4",
                linestyle="--",
                marker="x",
                linewidth=1.1,
                markersize=3,
            )
            ax2.plot(
                x_vals,
                wc_time,
                color="#ff7f0e",
                linestyle="--",
                marker="x",
                linewidth=1.1,
                markersize=3,
            )
            if c_idx == ncols - 1:
                ax2.set_ylabel("Time (s)", fontsize=8)

            if r_idx == nrows - 1:
                ax.set_xlabel("size_to_reach", fontsize=8)

            ax.tick_params(axis="both", labelsize=7)
            ax2.tick_params(axis="y", labelsize=7)

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", marker="o", linewidth=1.3, markersize=4, label="GV cost"),
        Line2D([0], [0], color="#ff7f0e", marker="o", linewidth=1.3, markersize=4, label="WC cost"),
        Line2D([0], [0], color="#1f77b4", linestyle="--", marker="x", linewidth=1.1, markersize=4, label="GV time"),
        Line2D([0], [0], color="#ff7f0e", linestyle="--", marker="x", linewidth=1.1, markersize=4, label="WC time"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot GV/WC cost and runtime across size_to_reach over benches and qubit sizes."
        )
    )
    parser.add_argument(
        "--benches",
        default="all",
        help="Comma-separated bench ids or 'all'.",
    )
    parser.add_argument(
        "--qubits",
        default="12,14,16,18,20,22,24",
        help="Comma-separated qubit sizes (columns).",
    )
    parser.add_argument(
        "--size-to-reach-values",
        default="",
        help="Comma-separated size_to_reach values. If empty, default range is 3..24.",
    )
    parser.add_argument("--size-to-reach-min", type=int, default=3)
    parser.add_argument("--size-to-reach-max", type=int, default=24)
    parser.add_argument(
        "--cost-timeout-sec",
        type=int,
        default=0,
        help="Timeout per GV/WC cost call; 0 disables timeout wrapper.",
    )
    parser.add_argument(
        "--clingo-timeout-sec",
        type=int,
        default=0,
        help="Pass-through to QVM_CLINGO_TIMEOUT_SEC.",
    )
    parser.add_argument(
        "--max-partition-tries",
        type=int,
        default=0,
        help="Pass-through to QVM_MAX_PARTITION_TRIES.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "evaluation" / "plots"),
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional output tag suffix.",
    )
    args = parser.parse_args()

    benches = _parse_benches(args.benches)
    qubits = _parse_int_csv(args.qubits)
    if args.size_to_reach_values.strip():
        size_values = _parse_int_csv(args.size_to_reach_values)
    else:
        size_values = list(range(args.size_to_reach_min, args.size_to_reach_max + 1))
    size_values = sorted({int(v) for v in size_values if int(v) >= 2})
    if not size_values:
        raise ValueError("No valid size_to_reach values (need >=2).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag.strip() else ""
    out_pdf = out_dir / f"cut_cost_time_grid_{timestamp}{tag}.pdf"
    out_csv = out_dir / f"cut_cost_time_grid_{timestamp}{tag}.csv"

    total_points = len(benches) * len(qubits) * len(size_values) * 2
    done = 0
    print(
        f"[start] benches={len(benches)} qubits={qubits} size_to_reach={size_values} total_calls={total_points}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    for bench, bench_label in benches:
        for nq in qubits:
            print(f"[progress] bench={bench} ({bench_label}) qubits={nq}", flush=True)
            valid_size_values = [s for s in size_values if s <= nq]
            if not valid_size_values:
                print(
                    f"[progress] skip bench={bench} qubits={nq}: no size_to_reach <= qubits",
                    flush=True,
                )
                continue
            for s in valid_size_values:
                for method in ("GV", "WC"):
                    result = _compute_with_optional_timeout(
                        bench=bench,
                        qubits=nq,
                        size_to_reach=s,
                        method=method,
                        timeout_sec=args.cost_timeout_sec,
                        clingo_timeout_sec=args.clingo_timeout_sec,
                        max_partition_tries=args.max_partition_tries,
                    )
                    row = {
                        "bench": bench,
                        "bench_label": bench_label,
                        "qubits": nq,
                        "size_to_reach": s,
                        "method": method,
                        "cost": result["cost"],
                        "elapsed_sec": result["elapsed_sec"],
                        "timed_out": int(bool(result["timed_out"])),
                        "error": result["error"],
                    }
                    records.append(row)
                    done += 1
                    if result["error"]:
                        print(
                            f"[progress] {done}/{total_points} {bench} {nq}q s={s} {method} "
                            f"error={result['error']}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[progress] {done}/{total_points} {bench} {nq}q s={s} {method} "
                            f"cost={result['cost']:.2f} sec={result['elapsed_sec']:.2f}",
                            flush=True,
                        )

    _plot_grid(records, benches, qubits, out_pdf)
    print(f"[done] wrote figure: {out_pdf}", flush=True)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bench",
                "bench_label",
                "qubits",
                "size_to_reach",
                "method",
                "cost",
                "elapsed_sec",
                "timed_out",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"[done] wrote data: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
