import argparse
import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from itertools import combinations, permutations
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.full_eval import BENCHES, _load_qasm_circuit, _run_mitigator, _analyze_qernel
from qos.error_mitigator.analyser import (
    BasicAnalysisPass,
    SupermarqFeaturesAnalysisPass,
    IsQAOACircuitPass,
    QAOAAnalysisPass,
)
from qos.error_mitigator.optimiser import FrozenQubitsPass
from qos.error_mitigator.run import ErrorMitigator
from qos.types.types import Qernel


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


def _apply_qf(mitigator, q):
    is_qaoa_pass = IsQAOACircuitPass()
    if not is_qaoa_pass.run(q):
        return q
    qaoa_analysis_pass = QAOAAnalysisPass()
    qaoa_analysis_pass.run(q)
    metadata = q.get_metadata()
    num_cnots = metadata.get("num_nonlocal_gates", 0)
    hotspots = list(metadata.get("hotspot_nodes", {}).values())
    qubits_to_freeze = 0
    for i in range(min(2, len(hotspots))):
        if num_cnots > 0 and hotspots[i] / num_cnots >= 0.07:
            qubits_to_freeze += 1
    qubits_to_freeze = min(qubits_to_freeze, mitigator.budget)
    if qubits_to_freeze > 0:
        qf_pass = FrozenQubitsPass(qubits_to_freeze)
        q = qf_pass.run(q)
    return q


def _run_permutation_eval(bench, size, args):
    qc = _load_qasm_circuit(bench, size)
    rows = []
    for perm in permutations(METHODS):
        q = Qernel(qc.copy())
        BasicAnalysisPass().run(q)
        SupermarqFeaturesAnalysisPass().run(q)
        mitigator = ErrorMitigator(
            size_to_reach=args.size_to_reach,
            ideal_size_to_reach=args.ideal_size_to_reach,
            budget=args.budget,
            methods=METHODS,
            use_cost_search=False,
            collect_timing=False,
        )
        def _can_apply_cut(stage_name):
            try:
                costs = mitigator.computeCuttingCosts(q, args.size_to_reach)
            except Exception as exc:
                return False, f"{stage_name} cost failed: {exc}"
            return costs.get(stage_name, 9999) <= mitigator.budget, None

        error = None
        for stage in perm:
            try:
                if stage == "QF":
                    q = _apply_qf(mitigator, q)
                elif stage == "GV":
                    ok, err = _can_apply_cut("GV")
                    if ok:
                        q = mitigator.applyGV(q, args.size_to_reach)
                    else:
                        error = error or err or "GV skipped (cost too high)"
                elif stage == "WC":
                    ok, err = _can_apply_cut("WC")
                    if ok:
                        q = mitigator.applyWC(q, args.size_to_reach)
                    else:
                        error = error or err or "WC skipped (cost too high)"
                elif stage == "QR":
                    q = mitigator.applyQR(q, args.size_to_reach)
            except Exception as exc:
                error = str(exc)
                break
        metrics = _analyze_qernel(
            q,
            args.metric_mode,
            args.metrics_baseline,
            args.metrics_optimization_level,
        )
        rows.append(
            {
                "bench": bench,
                "size": size,
                "methods": " > ".join(perm),
                "depth": metrics["depth"],
                "cnot": metrics["num_nonlocal_gates"],
                "error": error,
            }
        )
    return rows


def _plot_all(rows, benches, size, out_dir, timestamp, method_labels):
    plt = _import_matplotlib()
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


def _plot_all_combined(
    combo_rows,
    perm_rows,
    benches,
    size,
    out_dir,
    timestamp,
    combo_labels,
    perm_labels,
):
    plt = _import_matplotlib()
    combo_x = np.arange(len(combo_labels))
    perm_x = np.arange(len(perm_labels))

    fig, axes = plt.subplots(len(benches), 4, figsize=(24, 3.2 * len(benches)), sharex="col")
    if len(benches) == 1:
        axes = np.array([axes])

    combo_by_bench = {}
    for row in combo_rows:
        combo_by_bench.setdefault(row["bench"], []).append(row)
    perm_by_bench = {}
    for row in perm_rows:
        perm_by_bench.setdefault(row["bench"], []).append(row)

    for idx, (bench, label) in enumerate(benches):
        combo_map = {row["methods"]: row for row in combo_by_bench.get(bench, [])}
        perm_map = {row["methods"]: row for row in perm_by_bench.get(bench, [])}

        combo_depths = [combo_map.get(m, {}).get("depth", 0) for m in combo_labels]
        combo_cnots = [combo_map.get(m, {}).get("cnot", 0) for m in combo_labels]
        perm_depths = [perm_map.get(m, {}).get("depth", 0) for m in perm_labels]
        perm_cnots = [perm_map.get(m, {}).get("cnot", 0) for m in perm_labels]

        ax_cd = axes[idx, 0]
        ax_cc = axes[idx, 1]
        ax_pd = axes[idx, 2]
        ax_pc = axes[idx, 3]

        ax_cd.bar(combo_x, combo_depths, color="#4C78A8")
        ax_cd.set_title(f"{label} combo depth (size {size})")
        ax_cd.set_ylabel("Depth")
        ax_cd.grid(axis="y", linestyle="--", alpha=0.35)

        ax_cc.bar(combo_x, combo_cnots, color="#F58518")
        ax_cc.set_title(f"{label} combo CNOT (size {size})")
        ax_cc.set_ylabel("CNOT")
        ax_cc.grid(axis="y", linestyle="--", alpha=0.35)

        ax_pd.bar(perm_x, perm_depths, color="#54A24B")
        ax_pd.set_title(f"{label} perm depth (size {size})")
        ax_pd.set_ylabel("Depth")
        ax_pd.grid(axis="y", linestyle="--", alpha=0.35)

        ax_pc.bar(perm_x, perm_cnots, color="#E45756")
        ax_pc.set_title(f"{label} perm CNOT (size {size})")
        ax_pc.set_ylabel("CNOT")
        ax_pc.grid(axis="y", linestyle="--", alpha=0.35)

    axes[-1, 0].set_xticks(combo_x)
    axes[-1, 0].set_xticklabels(combo_labels, rotation=45, ha="right", fontsize=8)
    axes[-1, 1].set_xticks(combo_x)
    axes[-1, 1].set_xticklabels(combo_labels, rotation=45, ha="right", fontsize=8)
    axes[-1, 2].set_xticks(perm_x)
    axes[-1, 2].set_xticklabels(perm_labels, rotation=45, ha="right", fontsize=7)
    axes[-1, 3].set_xticks(perm_x)
    axes[-1, 3].set_xticklabels(perm_labels, rotation=45, ha="right", fontsize=7)

    fig.tight_layout()
    out_path = out_dir / f"methods_combo_perm_all_{size}_{timestamp}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="qaoa_pl1")
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--all-circuits", action="store_true")
    parser.add_argument("--permutations", action="store_true")
    parser.add_argument("--combined", action="store_true")
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
        if args.combined:
            combo_rows = []
            perm_rows = []
            for bench, _label in BENCHES:
                combo_rows.extend(_run_combo_eval(bench, args.size, eval_args))
                perm_rows.extend(_run_permutation_eval(bench, args.size, eval_args))
            combo_labels = [
                _combo_label(combo)
                for r in range(1, len(METHODS) + 1)
                for combo in combinations(METHODS, r)
            ]
            perm_labels = [" > ".join(p) for p in permutations(METHODS)]
            fig_path = _plot_all_combined(
                combo_rows,
                perm_rows,
                BENCHES,
                args.size,
                out_dir,
                timestamp,
                combo_labels,
                perm_labels,
            )
            print(f"Wrote plot: {fig_path}")
        else:
            for bench, _label in BENCHES:
                if args.permutations:
                    rows = _run_permutation_eval(bench, args.size, eval_args)
                else:
                    rows = _run_combo_eval(bench, args.size, eval_args)
                all_rows.extend(rows)
            if args.permutations:
                method_labels = [" > ".join(p) for p in permutations(METHODS)]
            else:
                method_labels = [
                    _combo_label(combo)
                    for r in range(1, len(METHODS) + 1)
                    for combo in combinations(METHODS, r)
                ]
            fig_path = _plot_all(all_rows, BENCHES, args.size, out_dir, timestamp, method_labels)
            print(f"Wrote plot: {fig_path}")
    else:
        if args.combined:
            combo_rows = _run_combo_eval(args.bench, args.size, eval_args)
            perm_rows = _run_permutation_eval(args.bench, args.size, eval_args)
            combo_labels = [
                _combo_label(combo)
                for r in range(1, len(METHODS) + 1)
                for combo in combinations(METHODS, r)
            ]
            perm_labels = [" > ".join(p) for p in permutations(METHODS)]
            fig_path = _plot_all_combined(
                combo_rows,
                perm_rows,
                [(args.bench, args.bench)],
                args.size,
                out_dir,
                timestamp,
                combo_labels,
                perm_labels,
            )
            print(f"Wrote plot: {fig_path}")
        else:
            if args.permutations:
                rows = _run_permutation_eval(args.bench, args.size, eval_args)
                method_labels = [" > ".join(p) for p in permutations(METHODS)]
            else:
                rows = _run_combo_eval(args.bench, args.size, eval_args)
                method_labels = [
                    _combo_label(combo)
                    for r in range(1, len(METHODS) + 1)
                    for combo in combinations(METHODS, r)
                ]
            fig_path = _plot_all(rows, [(args.bench, args.bench)], args.size, out_dir, timestamp, method_labels)
            print(f"Wrote plot: {fig_path}")


if __name__ == "__main__":
    main()
