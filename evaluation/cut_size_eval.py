import argparse
import datetime as dt
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.full_eval import (
    BENCHES,
    _load_qasm_circuit,
    _analyze_qernel,
    _analyze_circuit,
)
from qos.error_mitigator.analyser import BasicAnalysisPass, SupermarqFeaturesAnalysisPass
from qos.error_mitigator.run import ErrorMitigator
from qos.types.types import Qernel


def _import_matplotlib():
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def _qos_choice(qc, args):
    try:
        from qos.error_mitigator import evolution_target
    except Exception:
        evolution_target = None
    evolved_run = getattr(evolution_target, "evolved_run", None) if evolution_target else None

    q = Qernel(qc.copy())
    BasicAnalysisPass().run(q)
    SupermarqFeaturesAnalysisPass().run(q)
    mitigator = ErrorMitigator(
        size_to_reach=args.size_to_reach,
        ideal_size_to_reach=args.ideal_size_to_reach,
        budget=args.budget,
        methods=[],
        use_cost_search=True,
        collect_timing=False,
    )
    choice = {"method": None, "size": None}

    def _wrap(fn, name):
        def _wrapped(qernel, size_to_reach):
            choice["method"] = name
            choice["size"] = size_to_reach
            return fn(qernel, size_to_reach)

        return _wrapped

    mitigator.applyGV = _wrap(mitigator.applyGV, "GV")
    mitigator.applyWC = _wrap(mitigator.applyWC, "WC")
    try:
        if evolved_run is None:
            q = mitigator.run(q)
        else:
            q = evolved_run(mitigator, q)
    except Exception:
        return choice, None
    metrics = _analyze_qernel(
        q,
        args.metric_mode,
        args.metrics_baseline,
        args.metrics_optimization_level,
    )
    return choice, metrics


def _run_fixed(qc, method, size_to_reach, args):
    q = Qernel(qc.copy())
    BasicAnalysisPass().run(q)
    SupermarqFeaturesAnalysisPass().run(q)
    mitigator = ErrorMitigator(
        size_to_reach=size_to_reach,
        ideal_size_to_reach=args.ideal_size_to_reach,
        budget=args.budget,
        methods=[method],
        use_cost_search=False,
        collect_timing=False,
    )
    try:
        q = mitigator.run(q)
    except Exception:
        return None
    return _analyze_qernel(
        q,
        args.metric_mode,
        args.metrics_baseline,
        args.metrics_optimization_level,
    )


def _relative_metrics(metrics, base):
    if metrics is None or base is None:
        return None
    base_depth = max(1, base.get("depth", 0))
    base_cnot = max(1, base.get("num_nonlocal_gates", 0))
    return {
        "depth": metrics["depth"] / base_depth,
        "num_nonlocal_gates": metrics["num_nonlocal_gates"] / base_cnot,
    }


def _plot(bench_rows, benches, size, sizes_to_reach, out_dir, timestamp):
    plt = _import_matplotlib()
    fig, axes = plt.subplots(len(benches), 2, figsize=(14, 3.2 * len(benches)), sharex=True)
    if len(benches) == 1:
        axes = np.array([axes])

    for idx, (bench, label) in enumerate(benches):
        row = bench_rows[bench]
        gv_depths = row["gv_depths"]
        wc_depths = row["wc_depths"]
        gv_cnots = row["gv_cnots"]
        wc_cnots = row["wc_cnots"]
        qos_choice = row["qos_choice"]
        qos_metrics = row["qos_metrics"]

        ax_d = axes[idx, 0]
        ax_c = axes[idx, 1]

        ax_d.plot(sizes_to_reach, gv_depths, marker="o", linewidth=1.2, markersize=3, label="GV")
        ax_d.plot(sizes_to_reach, wc_depths, marker="o", linewidth=1.2, markersize=3, label="WC")
        ax_d.set_title(f"{label} depth vs size_to_reach (size {size})")
        ax_d.set_ylabel("Depth (rel. to Qiskit)")
        ax_d.grid(axis="y", linestyle="--", alpha=0.35)

        ax_c.plot(sizes_to_reach, gv_cnots, marker="o", linewidth=1.2, markersize=3, label="GV")
        ax_c.plot(sizes_to_reach, wc_cnots, marker="o", linewidth=1.2, markersize=3, label="WC")
        ax_c.set_title(f"{label} CNOT vs size_to_reach (size {size})")
        ax_c.set_ylabel("CNOT (rel. to Qiskit)")
        ax_c.grid(axis="y", linestyle="--", alpha=0.35)

        if qos_choice["size"] is not None and qos_choice["method"] in {"GV", "WC"}:
            color = "#E45756"
            ax_d.axvline(qos_choice["size"], color=color, linestyle="--", linewidth=1)
            ax_c.axvline(qos_choice["size"], color=color, linestyle="--", linewidth=1)
            if qos_metrics:
                qos_rel = _relative_metrics(qos_metrics, row.get("baseline"))
                if qos_rel is None:
                    qos_rel = qos_metrics
                ax_d.scatter(
                    [qos_choice["size"]],
                    [qos_rel["depth"]],
                    color=color,
                    marker="*",
                    s=60,
                    zorder=3,
                )
                ax_c.scatter(
                    [qos_choice["size"]],
                    [qos_rel["num_nonlocal_gates"]],
                    color=color,
                    marker="*",
                    s=60,
                    zorder=3,
                )
            ax_d.text(
                qos_choice["size"],
                np.nanmax(gv_depths + wc_depths),
                f"QOS {qos_choice['method']}@{qos_choice['size']}",
                color=color,
                fontsize=8,
                ha="left",
                va="bottom",
            )
            ax_c.text(
                qos_choice["size"],
                np.nanmax(gv_cnots + wc_cnots),
                f"QOS {qos_choice['method']}@{qos_choice['size']}",
                color=color,
                fontsize=8,
                ha="left",
                va="bottom",
            )

        ax_d.legend(fontsize=7)

    axes[-1, 0].set_xlabel("size_to_reach")
    axes[-1, 1].set_xlabel("size_to_reach")
    fig.tight_layout()
    out_path = out_dir / f"cut_size_eval_{size}_{timestamp}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="12", help="Comma-separated qubit sizes.")
    parser.add_argument("--size-min", type=int, default=3)
    parser.add_argument("--size-max", type=int, default=7)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--size-to-reach", type=int, default=7)
    parser.add_argument("--ideal-size-to-reach", type=int, default=2)
    parser.add_argument("--metric-mode", default="fragment")
    parser.add_argument("--metrics-baseline", default="kolkata")
    parser.add_argument("--metrics-optimization-level", type=int, default=3)
    parser.add_argument("--benches", default="all")
    parser.add_argument("--out-dir", default=str(ROOT / "evaluation" / "plots"))
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    if args.benches == "all":
        benches = BENCHES
    else:
        selected = {b.strip() for b in args.benches.split(",") if b.strip()}
        benches = [(b, label) for b, label in BENCHES if b in selected]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    for size in sizes:
        sizes_to_reach = list(range(args.size_min, args.size_max + 1))
        bench_rows = {}
        qos_cache = {}
        for bench, _label in benches:
            qc = _load_qasm_circuit(bench, size)
            baseline = _analyze_circuit(
                qc,
                args.metric_mode,
                args.metrics_baseline,
                args.metrics_optimization_level,
            )
            cache_key = (bench, size)
            if cache_key in qos_cache:
                qos_choice, qos_metrics = qos_cache[cache_key]
            else:
                qos_choice, qos_metrics = _qos_choice(qc, args)
                qos_cache[cache_key] = (qos_choice, qos_metrics)
            qos_rel = _relative_metrics(qos_metrics, baseline)
            gv_depths = []
            wc_depths = []
            gv_cnots = []
            wc_cnots = []
            for s in sizes_to_reach:
                gv_metrics = _run_fixed(qc, "GV", s, args)
                wc_metrics = _run_fixed(qc, "WC", s, args)
                gv_rel = _relative_metrics(gv_metrics, baseline)
                wc_rel = _relative_metrics(wc_metrics, baseline)
                gv_depths.append(
                    float("nan") if gv_rel is None else gv_rel["depth"]
                )
                wc_depths.append(
                    float("nan") if wc_rel is None else wc_rel["depth"]
                )
                gv_cnots.append(
                    float("nan") if gv_rel is None else gv_rel["num_nonlocal_gates"]
                )
                wc_cnots.append(
                    float("nan") if wc_rel is None else wc_rel["num_nonlocal_gates"]
                )
            bench_rows[bench] = {
                "gv_depths": gv_depths,
                "wc_depths": wc_depths,
                "gv_cnots": gv_cnots,
                "wc_cnots": wc_cnots,
                "qos_choice": qos_choice,
                "qos_metrics": qos_rel,
                "baseline": baseline,
            }

        fig_path = _plot(bench_rows, benches, size, sizes_to_reach, out_dir, timestamp)
        print(f"Wrote cut size sweep: {fig_path}")


if __name__ == "__main__":
    main()
