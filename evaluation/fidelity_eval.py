import argparse
import datetime as dt
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qos.error_mitigator.run import ErrorMitigator
from qos.types.types import Qernel
from qvm.quasi_distr import QuasiDistr
from qvm.virtual_circuit import VirtualCircuit, generate_instantiations
from qvm.virtual_gates import VirtualBinaryGate, VirtualMove, WireCut, VirtualGateEndpoint
from qiskit.circuit.library import CXGate


BENCHES = [
    ("qaoa_r3", "QAOA-R3"),
    ("bv", "BV"),
    ("ghz", "GHZ"),
    ("hamsim_1", "HS-1"),
    ("qaoa_pl1", "QAOA-P1"),
    ("qsvm", "QSVM"),
    ("twolocal_1", "TL-1"),
    ("vqe_1", "VQE-1"),
    ("wstate", "W-STATE"),
]


class SerialPool:
    def map(self, fn, iterable):
        return list(map(fn, iterable))

    def starmap(self, fn, iterable):
        return [fn(*args) for args in iterable]


def _benchmarks_dir() -> Path:
    return Path(__file__).resolve().parent / "benchmarks"


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


def _ensure_measurements(circuit: QuantumCircuit | VirtualCircuit) -> QuantumCircuit:
    if isinstance(circuit, VirtualCircuit):
        circuit = circuit._circuit
    if circuit.num_clbits == 0:
        creg = ClassicalRegister(circuit.num_qubits)
        circuit = circuit.copy()
        circuit.add_register(creg)
        circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))
        return circuit
    if not any(instr.operation.name == "measure" for instr in circuit.data):
        circuit = circuit.copy()
        circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))
    return circuit


def _has_virtual_ops(circuit: QuantumCircuit) -> bool:
    if isinstance(circuit, VirtualCircuit):
        circuit = circuit._circuit
    for instr in circuit.data:
        op = instr.operation
        if isinstance(op, (VirtualBinaryGate, VirtualMove, VirtualGateEndpoint, WireCut)):
            return True
    return False


def _replace_virtual_ops(circuit: QuantumCircuit) -> QuantumCircuit:
    if isinstance(circuit, VirtualCircuit):
        circuit = circuit._circuit
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for instr in circuit.data:
        op = instr.operation
        if isinstance(op, (VirtualBinaryGate, VirtualMove, VirtualGateEndpoint)):
            if len(instr.qubits) == 2:
                new_circuit.append(CXGate(), instr.qubits, instr.clbits)
        elif isinstance(op, WireCut):
            continue
        else:
            new_circuit.append(op, instr.qubits, instr.clbits)
    return new_circuit


def _noise_model(p1: float, p2: float) -> NoiseModel:
    noise = NoiseModel()
    err1 = depolarizing_error(p1, 1)
    err2 = depolarizing_error(p2, 2)
    one_q = ["x", "y", "z", "h", "rx", "ry", "rz", "sx", "id", "s", "sdg"]
    two_q = ["cx", "cz", "swap", "rzz", "cp", "ecr"]
    noise.add_all_qubit_quantum_error(err1, one_q)
    noise.add_all_qubit_quantum_error(err2, two_q)
    return noise


def _counts_to_probs(counts: Dict[str, int]) -> Dict[str, float]:
    shots = max(1, sum(counts.values()))
    return {k: v / shots for k, v in counts.items()}


def _hellinger_fidelity_from_counts(ideal: Dict[str, int], noisy: Dict[str, int]) -> float:
    p = _counts_to_probs(ideal)
    q = _counts_to_probs(noisy)
    if not p or not q:
        return 0.0
    keys = set(p.keys()) | set(q.keys())
    bc = 0.0
    for k in keys:
        bc += (p.get(k, 0.0) * q.get(k, 0.0)) ** 0.5
    return bc * bc


def _simulate_counts(circuit: QuantumCircuit, shots: int, noise: NoiseModel | None, seed: int) -> Dict[str, int]:
    sim = AerSimulator(noise_model=noise) if noise else AerSimulator()
    circ = _ensure_measurements(circuit)
    result = sim.run(circ, shots=shots, seed_simulator=seed, seed_transpiler=seed).result()
    return result.get_counts()


def _simulate_virtual_counts(
    circuit: QuantumCircuit | VirtualCircuit,
    shots: int,
    noise: NoiseModel | None,
    seed: int,
    max_instantiations: int,
) -> Dict[str, int]:
    vc = circuit if isinstance(circuit, VirtualCircuit) else VirtualCircuit(circuit)
    results: Dict = {}
    for frag, frag_circ in vc.fragment_circuits.items():
        inst_labels = vc.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circ, inst_labels)
        if max_instantiations > 0 and max_instantiations < len(instantiations):
            raise ValueError("max_instantiations must be 0 or >= full instantiation count.")
        distrs = []
        for inst in instantiations:
            counts = _simulate_counts(inst, shots, noise, seed)
            distrs.append(QuasiDistr.from_counts(counts))
        results[frag] = distrs
    quasi = vc.knit(results, SerialPool())
    return quasi.to_counts(vc._circuit.num_clbits, shots)


def _fidelity_for_circuit(
    circuit: QuantumCircuit,
    shots: int,
    noise: NoiseModel,
    seed: int,
    max_instantiations: int,
) -> float:
    if _has_virtual_ops(circuit):
        ideal = _simulate_virtual_counts(circuit, shots, None, seed, max_instantiations)
        noisy = _simulate_virtual_counts(circuit, shots, noise, seed, max_instantiations)
    else:
        ideal = _simulate_counts(circuit, shots, None, seed)
        noisy = _simulate_counts(circuit, shots, noise, seed)
    if not ideal or not noisy:
        return 0.0
    return _hellinger_fidelity_from_counts(ideal, noisy)


def _run_mitigator(qc: QuantumCircuit, methods: List[str], args) -> List[QuantumCircuit]:
    q = Qernel(qc.copy())
    mitigator = ErrorMitigator(
        size_to_reach=args.size_to_reach,
        ideal_size_to_reach=args.ideal_size_to_reach,
        budget=args.budget,
        methods=methods,
        use_cost_search=bool(methods == [] and args.qos_cost_search),
    )
    q = mitigator.run(q)
    vsqs = q.get_virtual_subqernels()
    if vsqs:
        return [vsq.get_circuit() for vsq in vsqs]
    return [q.get_circuit()]


def _average_fidelity(
    circuits: List[QuantumCircuit],
    shots: int,
    noise: NoiseModel,
    seed: int,
    max_instantiations: int,
) -> float:
    vals = [_fidelity_for_circuit(c, shots, noise, seed, max_instantiations) for c in circuits]
    return float(sum(vals) / max(1, len(vals)))


def _plot(
    results: Dict[Tuple[str, int, str], float],
    sizes: List[int],
    benches: List[Tuple[str, str]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    methods = ["Qiskit", "CutQC", "QOS", "FrozenQubits"]
    bench_labels = [label for _, label in benches]
    x = np.arange(len(bench_labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for s_idx, size in enumerate(sizes):
        for m_idx, method in enumerate(methods):
            vals = []
            for bench, _label in benches:
                vals.append(results[(bench, size, method)])
            offset = (m_idx - (len(methods) - 1) / 2) * width + (s_idx - (len(sizes) - 1) / 2) * (len(methods) * width + 0.1)
            ax.bar(x + offset, vals, width, label=f"{method} {size}")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fidelity")
    ax.set_title("Fidelity vs Benchmarks")
    ax.set_xticks(x)
    ax.set_xticklabels(bench_labels)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate fidelity for Qiskit, CutQC, QOS, FrozenQubits.")
    parser.add_argument("--sizes", default="12", help="Comma-separated qubit sizes (e.g., 12,24).")
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--p1", type=float, default=0.001, help="1-qubit depolarizing error.")
    parser.add_argument("--p2", type=float, default=0.01, help="2-qubit depolarizing error.")
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--size-to-reach", type=int, default=9)
    parser.add_argument("--ideal-size-to-reach", type=int, default=2)
    parser.add_argument(
        "--qos-cost-search",
        dest="qos_cost_search",
        action="store_true",
        default=True,
        help="Enable cost-search (default).",
    )
    parser.add_argument(
        "--no-qos-cost-search",
        dest="qos_cost_search",
        action="store_false",
        help="Disable cost-search.",
    )
    parser.add_argument("--clingo-timeout-sec", type=int, default=10)
    parser.add_argument("--max-partition-tries", type=int, default=5)
    parser.add_argument("--max-instantiations", type=int, default=0)
    parser.add_argument("--benches", default="all", help="Comma-separated benchmark names or 'all'.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", default="", help="Optional tag for output filename.")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "evaluation" / "plots"),
        help="Output directory for figure and CSV.",
    )
    args = parser.parse_args()

    os.environ["QVM_CLINGO_TIMEOUT_SEC"] = str(args.clingo_timeout_sec)
    os.environ["QVM_MAX_PARTITION_TRIES"] = str(args.max_partition_tries)

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag.strip()
    tag_suffix = f"_{tag}" if tag else ""

    noise = _noise_model(args.p1, args.p2)

    if args.benches == "all":
        benches = BENCHES
    else:
        selected = {b.strip() for b in args.benches.split(",") if b.strip()}
        benches = [(b, label) for b, label in BENCHES if b in selected]

    results: Dict[Tuple[str, int, str], float] = {}
    for size in sizes:
        for bench, _label in benches:
            if args.verbose:
                print(f"size={size} bench={bench}", flush=True)
            qc = _load_qasm_circuit(bench, size)
            results[(bench, size, "Qiskit")] = _average_fidelity(
                [qc], args.shots, noise, args.seed, args.max_instantiations
            )
            results[(bench, size, "CutQC")] = _average_fidelity(
                _run_mitigator(qc, ["WC"], args),
                args.shots,
                noise,
                args.seed,
                args.max_instantiations,
            )
            results[(bench, size, "QOS")] = _average_fidelity(
                _run_mitigator(qc, [], args),
                args.shots,
                noise,
                args.seed,
                args.max_instantiations,
            )
            results[(bench, size, "FrozenQubits")] = _average_fidelity(
                _run_mitigator(qc, ["QF"], args),
                args.shots,
                noise,
                args.seed,
                args.max_instantiations,
            )

    fig_path = out_dir / f"dt_fidelities_sim_{timestamp}{tag_suffix}.pdf"
    _plot(results, sizes, benches, fig_path)
    csv_path = out_dir / f"dt_fidelities_sim_{timestamp}{tag_suffix}.csv"
    with csv_path.open("w") as f:
        f.write("bench,size,method,fidelity\n")
        for (bench, size, method), val in results.items():
            f.write(f"{bench},{size},{method},{val}\n")
    print(f"Wrote figure: {fig_path}")
    print(f"Wrote metrics: {csv_path}")


if __name__ == "__main__":
    main()
