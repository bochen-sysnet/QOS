import argparse
import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from itertools import combinations
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.full_eval import BENCHES, _load_qasm_circuit, _run_mitigator


METHODS = ["QF", "GV", "WC", "QR"]


def _import_matplotlib():
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def _combo_label(combo):
    return "+".join(combo)


def _run_combo_eval(bench, size, args):
    qc = _load_qasm_circuit(bench, size)
    rows = []
    for r in range(1, len(METHODS) + 1):
        for combo in combinations(METHODS, r):
            metrics, _timings, _circs = _run_mitigator(qc, list(combo), args)
            rows.append(
                {
                    "bench": bench,
                    "size": size,
                    "methods": _combo_label(combo),
                    "depth": metrics["depth"],
                    "cnot": metrics["num_nonlocal_gates"],
                }
            )
    return rows


def _plot_all(rows, benches, size, out_dir, timestamp):
    plt = _import_matplotlib()
    method_labels = [_combo_label(combo) for r in range(1, len(METHODS) + 1) for combo in combinations(METHODS, r)]
    x = np.arange(len(method_labels))

    fig, axes = plt.subplots(len(benches), 2, figsize=(16, 3.2 * len(benches)), sharex=True)
    if len(benches) == 1:
        axes = np.array([axes])

    rows_by_bench = {}
    for row in rows:
        rows_by_bench.setdefault(row["bench"], []).append(row)

    for idx, (bench, label) in enumerate(benches):
        bench_rows = rows_by_bench.get(bench, [])
        bench_map = {row["methods"]: row for row in bench_rows}
        depths = [bench_map.get(m, {}).get("depth", 0) for m in method_labels]
        cnots = [bench_map.get(m, {}).get("cnot", 0) for m in method_labels]

        ax_depth = axes[idx, 0]
        ax_cnot = axes[idx, 1]

        ax_depth.bar(x, depths, color="#4C78A8")
        ax_depth.set_title(f"{label} depth (size {size})")
        ax_depth.set_ylabel("Depth")
        ax_depth.grid(axis="y", linestyle="--", alpha=0.35)

        ax_cnot.bar(x, cnots, color="#F58518")
        ax_cnot.set_title(f"{label} CNOT (size {size})")
        ax_cnot.set_ylabel("CNOT")
        ax_cnot.grid(axis="y", linestyle="--", alpha=0.35)

    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(method_labels, rotation=45, ha="right", fontsize=8)
    axes[-1, 1].set_xticks(x)
    axes[-1, 1].set_xticklabels(method_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"methods_combo_all_{size}_{timestamp}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="qaoa_pl1")
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--all-circuits", action="store_true")
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--size-to-reach", type=int, default=7)
    parser.add_argument("--ideal-size-to-reach", type=int, default=2)
    parser.add_argument("--output-dir", default=str(Path("evaluation") / "plots"))
    parser.add_argument("--timeout-sec", type=int, default=300)
    args = parser.parse_args()

    eval_args = SimpleNamespace(
        size_to_reach=args.size_to_reach,
        ideal_size_to_reach=args.ideal_size_to_reach,
        budget=args.budget,
        qos_cost_search=False,
        collect_timing=False,
        timeout_sec=args.timeout_sec,
        metric_mode="fragment",
        metrics_baseline="kolkata",
        metrics_optimization_level=3,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []
    if args.all_circuits:
        for bench, _label in BENCHES:
            rows = _run_combo_eval(bench, args.size, eval_args)
            all_rows.extend(rows)
        fig_path = _plot_all(all_rows, BENCHES, args.size, out_dir, timestamp)
        print(f"Wrote plot: {fig_path}")
    else:
        rows = _run_combo_eval(args.bench, args.size, eval_args)
        fig_path = _plot_all(rows, [(args.bench, args.bench)], args.size, out_dir, timestamp)
        print(f"Wrote plot: {fig_path}")


if __name__ == "__main__":
    main()
