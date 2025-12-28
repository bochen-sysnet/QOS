import argparse
import datetime as dt
import os
from pathlib import Path
import sys
import time
from multiprocessing import Process, Value

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.full_eval import _load_qasm_circuit
from qos.error_mitigator.optimiser import GVOptimalDecompositionPass, OptimalWireCuttingPass
from qos.types.types import Qernel


def _import_matplotlib():
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def _measure_cost(pass_cls, qernel, size_to_reach, repeats, timeout_sec):
    total = 0.0
    for _ in range(repeats):
        box = Value("i", 1000)
        t0 = time.perf_counter()
        proc = Process(target=pass_cls(size_to_reach).cost, args=(qernel, box))
        proc.start()
        proc.join(timeout_sec)
        if proc.is_alive():
            proc.terminate()
            proc.join()
        total += time.perf_counter() - t0
    return total / max(1, repeats)


def _parse_sizes(args):
    if args.sizes:
        return [int(s) for s in args.sizes.split(",") if s.strip()]
    return list(range(args.size_min, args.size_max + 1, args.size_step))


def _parse_sizes_to_reach(args):
    if args.size_to_reach_values:
        return [int(s) for s in args.size_to_reach_values.split(",") if s.strip()]
    return list(range(args.size_to_reach_min, args.size_to_reach_max + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="qaoa_r3")
    parser.add_argument("--sizes", default="12,16,20,24", help="Comma-separated circuit sizes.")
    parser.add_argument("--size-min", type=int, default=12)
    parser.add_argument("--size-max", type=int, default=24)
    parser.add_argument("--size-step", type=int, default=2)
    parser.add_argument("--size-to-reach-values", default="7,9,11,13,15", help="Comma-separated size-to-reach values.")
    parser.add_argument("--size-to-reach-min", type=int, default=3)
    parser.add_argument("--size-to-reach-max", type=int, default=12)
    parser.add_argument("--fixed-size-to-reach", type=int, default=7)
    parser.add_argument("--fixed-qubits", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cost-timeout-sec", type=int, default=600)
    parser.add_argument("--clingo-timeout-sec", type=int, default=0)
    parser.add_argument("--max-partition-tries", type=int, default=0)
    parser.add_argument("--out-dir", default=str(ROOT / "evaluation" / "plots"))
    args = parser.parse_args()

    os.environ["QVM_CLINGO_TIMEOUT_SEC"] = str(args.clingo_timeout_sec)
    os.environ["QVM_MAX_PARTITION_TRIES"] = str(args.max_partition_tries)

    sizes = _parse_sizes(args)
    sizes_to_reach = _parse_sizes_to_reach(args)

    fixed_str = args.fixed_size_to_reach
    fixed_qubits = args.fixed_qubits

    gv_fixed = []
    wc_fixed = []
    for size in sizes:
        print(f"[timing] size={size} fixed_size_to_reach={fixed_str}", flush=True)
        qc = _load_qasm_circuit(args.bench, size)
        q = Qernel(qc.copy())
        t0 = time.perf_counter()
        gv_val = _measure_cost(
            GVOptimalDecompositionPass, q, fixed_str, args.repeats, args.cost_timeout_sec
        )
        gv_fixed.append(gv_val)
        print(f"[timing]  GV sec={gv_val:.2f} total={time.perf_counter() - t0:.2f}", flush=True)
        t0 = time.perf_counter()
        wc_val = _measure_cost(
            OptimalWireCuttingPass, q, fixed_str, args.repeats, args.cost_timeout_sec
        )
        wc_fixed.append(wc_val)
        print(f"[timing]  WC sec={wc_val:.2f} total={time.perf_counter() - t0:.2f}", flush=True)

    gv_sweep = []
    wc_sweep = []
    qc = _load_qasm_circuit(args.bench, fixed_qubits)
    q = Qernel(qc.copy())
    for size_to_reach in sizes_to_reach:
        print(f"[timing] fixed_qubits={fixed_qubits} size_to_reach={size_to_reach}", flush=True)
        t0 = time.perf_counter()
        gv_val = _measure_cost(
            GVOptimalDecompositionPass, q, size_to_reach, args.repeats, args.cost_timeout_sec
        )
        gv_sweep.append(gv_val)
        print(f"[timing]  GV sec={gv_val:.2f} total={time.perf_counter() - t0:.2f}", flush=True)
        t0 = time.perf_counter()
        wc_val = _measure_cost(
            OptimalWireCuttingPass, q, size_to_reach, args.repeats, args.cost_timeout_sec
        )
        wc_sweep.append(wc_val)
        print(f"[timing]  WC sec={wc_val:.2f} total={time.perf_counter() - t0:.2f}", flush=True)

    plt = _import_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=False)

    axes[0].plot(sizes, gv_fixed, marker="o", linewidth=1.2, markersize=4, label="GV")
    axes[0].plot(sizes, wc_fixed, marker="o", linewidth=1.2, markersize=4, label="WC")
    axes[0].set_title(f"Cost time vs qubits (size_to_reach={fixed_str})")
    axes[0].set_xlabel("Circuit size (qubits)")
    axes[0].set_ylabel("Time (s)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8)

    axes[1].plot(sizes_to_reach, gv_sweep, marker="o", linewidth=1.2, markersize=4, label="GV")
    axes[1].plot(sizes_to_reach, wc_sweep, marker="o", linewidth=1.2, markersize=4, label="WC")
    axes[1].set_title(f"Cost time vs size_to_reach (qubits={fixed_qubits})")
    axes[1].set_xlabel("size_to_reach")
    axes[1].set_ylabel("Time (s)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"cut_pass_timing_{args.bench}_{timestamp}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote timing figure: {out_path}")


if __name__ == "__main__":
    main()
