import argparse
import csv
import datetime as dt
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qos.error_mitigator.analyser import BasicAnalysisPass, IsQAOACircuitPass
from qos.error_mitigator.run import ErrorMitigator
from qos.types.types import Qernel
from qvm.virtual_circuit import VirtualCircuit
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit


def _benchmarks_dir() -> Path:
    return Path(__file__).resolve().parent / "benchmarks"


def _list_benchmarks() -> List[str]:
    benches = []
    for p in _benchmarks_dir().iterdir():
        if p.is_dir() and any(p.glob("*.qasm")):
            benches.append(p.name)
    return sorted(benches)


def _list_qubit_sizes(bench: str) -> List[int]:
    sizes = []
    for qasm in (_benchmarks_dir() / bench).glob("*.qasm"):
        stem = qasm.stem
        if stem.isdigit():
            sizes.append(int(stem))
    return sorted(set(sizes))


def _load_qasm_circuit(bench: str, nqubits: int) -> QuantumCircuit:
    circuits_dir = _benchmarks_dir() / bench
    candidates = [
        p for p in circuits_dir.glob("*.qasm") if p.stem.isdigit() and int(p.stem) == nqubits
    ]
    if not candidates:
        raise ValueError(f"No qasm file found for {bench} with {nqubits} qubits.")
    circuit = QuantumCircuit.from_qasm_file(str(candidates[0]))
    dag = circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")
    return dag_to_circuit(dag)


def _parse_qubits(arg: str, available: List[int]) -> List[int]:
    if arg == "all":
        return available
    requested = [int(x) for x in arg.split(",") if x.strip()]
    return [q for q in requested if q in available]


def _normalize_circuit(circuit):
    if isinstance(circuit, VirtualCircuit):
        return circuit._circuit
    return circuit


def _extract_metrics(circuits: Iterable) -> Dict[str, float]:
    metrics = {
        "total_qubits": 0,
        "total_clbits": 0,
        "total_nonlocal_gates": 0,
        "total_cnot_gates": 0,
        "total_measurements": 0,
        "total_instructions": 0,
        "max_depth": 0,
        "num_fragments": 0,
    }

    analysis = BasicAnalysisPass()
    for circuit in circuits:
        qc = _normalize_circuit(circuit)
        q = Qernel(qc)
        analysis.run(q)
        m = q.get_metadata()
        metrics["total_qubits"] += m.get("num_qubits", 0)
        metrics["total_clbits"] += m.get("num_clbits", 0)
        metrics["total_nonlocal_gates"] += m.get("num_nonlocal_gates", 0)
        metrics["total_cnot_gates"] += m.get("num_cnot_gates", 0)
        metrics["total_measurements"] += m.get("num_measurements", 0)
        metrics["total_instructions"] += m.get("number_instructions", 0)
        metrics["max_depth"] = max(metrics["max_depth"], m.get("depth", 0))
        metrics["num_fragments"] += 1

    return metrics


def _qernel_circuits(q: Qernel) -> List:
    vsqs = q.get_virtual_subqernels()
    if vsqs:
        return [vsq.get_circuit() for vsq in vsqs]
    return [q.get_circuit()]


def _is_qaoa_circuit(qc) -> bool:
    q = Qernel(qc)
    return IsQAOACircuitPass().run(q)


def _frozenqubits_available() -> bool:
    try:
        import FrozenQubits.helper_FrozenQubits  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def _clingo_available() -> bool:
    try:
        import clingo  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def run_eval(
    benchmarks: List[str],
    qubits_arg: str,
    size_to_reach: int,
    ideal_size_to_reach: int,
    budget: int,
    out_dir: Path,
) -> Tuple[Path, List[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"error_mitigator_eval_{timestamp}.csv"

    frozenq_available = _frozenqubits_available()
    clingo_available = _clingo_available()
    variants = {
        "raw": None,
        "QR": ["QR"],
    }
    if clingo_available:
        variants["GV"] = ["GV"]
        variants["WC"] = ["WC"]

    if frozenq_available:
        variants["QF"] = ["QF"]

    if clingo_available and frozenq_available:
        variants["QOS"] = []
    elif clingo_available and not frozenq_available:
        variants["QOS_no_QF"] = ["GV", "WC", "QR"]
    elif not clingo_available and frozenq_available:
        variants["QOS_no_GV_WC"] = ["QF", "QR"]

    rows: List[Dict[str, object]] = []

    for bench in benchmarks:
        available_qubits = _list_qubit_sizes(bench)
        qubit_sizes = _parse_qubits(qubits_arg, available_qubits)
        if not qubit_sizes:
            continue

        for n in qubit_sizes:
            qc = _load_qasm_circuit(bench, n)
            is_qaoa = _is_qaoa_circuit(qc)

            for variant, methods in variants.items():
                if variant == "QF" and not is_qaoa:
                    continue

                if methods is None:
                    q = Qernel(qc)
                else:
                    q = Qernel(qc.copy())
                    mitigator = ErrorMitigator(
                        size_to_reach=size_to_reach,
                        ideal_size_to_reach=ideal_size_to_reach,
                        budget=budget,
                        methods=methods,
                    )
                    q = mitigator.run(q)

                metrics = _extract_metrics(_qernel_circuits(q))
                row = {
                    "benchmark": bench,
                    "qubits": n,
                    "variant": variant,
                    "is_qaoa": is_qaoa,
                    **metrics,
                }
                rows.append(row)

    if not rows:
        raise RuntimeError("No rows generated. Check benchmarks/qubits arguments.")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plot_paths = _plot_summary(rows, out_dir, timestamp)
    return csv_path, plot_paths


def _plot_summary(rows: List[Dict[str, object]], out_dir: Path, timestamp: str) -> List[Path]:
    variants = sorted({row["variant"] for row in rows})
    metrics = ["total_nonlocal_gates", "max_depth", "total_instructions"]

    plot_paths = []
    for metric in metrics:
        variant_vals = []
        for v in variants:
            vals = [float(r[metric]) for r in rows if r["variant"] == v]
            if vals:
                variant_vals.append(sum(vals) / len(vals))
            else:
                variant_vals.append(0.0)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(variants, variant_vals, color="#2D6A4F")
        ax.set_title(f"{metric} (avg across circuits)")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        plot_path = out_dir / f"error_mitigator_{metric}_{timestamp}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QOS error mitigator on benchmarks.")
    parser.add_argument(
        "--benchmarks",
        default="twolocal_3,qaoa_r3",
        help="Comma-separated benchmark names, or 'all' for every benchmark.",
    )
    parser.add_argument(
        "--qubits",
        default="all",
        help="Comma-separated qubit sizes or 'all' for available sizes.",
    )
    parser.add_argument("--size-to-reach", type=int, default=7)
    parser.add_argument("--ideal-size-to-reach", type=int, default=2)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "plots"),
        help="Output directory for CSV and plots.",
    )
    args = parser.parse_args()

    if args.benchmarks == "all":
        benchmarks = _list_benchmarks()
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    csv_path, plot_paths = run_eval(
        benchmarks=benchmarks,
        qubits_arg=args.qubits,
        size_to_reach=args.size_to_reach,
        ideal_size_to_reach=args.ideal_size_to_reach,
        budget=args.budget,
        out_dir=Path(args.out_dir),
    )

    print(f"Wrote metrics: {csv_path}")
    for p in plot_paths:
        print(f"Wrote figure: {p}")


if __name__ == "__main__":
    main()
