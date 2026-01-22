import argparse
import csv
import datetime as dt
import importlib.util
from pathlib import Path
import multiprocessing as mp
import logging
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import site
import signal
import os
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure conda site-packages precedes user site-packages.
USER_SITE = site.getusersitepackages()
if USER_SITE in sys.path:
    sys.path.remove(USER_SITE)
    sys.path.append(USER_SITE)

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qos.error_mitigator.analyser import BasicAnalysisPass, SupermarqFeaturesAnalysisPass
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

_EVAL_LOGGER = logging.getLogger(__name__)


def _is_eval_verbose() -> bool:
    raw = os.getenv("QOS_EVAL_VERBOSE", os.getenv("QOS_VERBOSE", ""))
    return raw.lower() in {"1", "true", "yes", "y"}

TARGET_REL = {
    "depth": {
        "QOS": 1 - 0.46,
        "CutQC": 1 - 0.386,
        "FrozenQubits": 1 - 0.294,
    },
    "nonlocal": {
        "QOS": 1 - 0.705,
        "CutQC": 1 - 0.66,
        "FrozenQubits": 1 - 0.566,
    },
}

METHOD_ORDER = ["FrozenQubits", "CutQC", "QOS", "QOSN", "QOSE"]
METHOD_ALIASES = {
    "frozenqubit": "FrozenQubits",
    "frozenqubits": "FrozenQubits",
    "fq": "FrozenQubits",
    "cutqc": "CutQC",
    "qos": "QOS",
    "qosn": "QOSN",
    "qose": "QOSE",
}


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


def _normalize_circuit(circuit):
    if isinstance(circuit, VirtualCircuit):
        return circuit._circuit
    return circuit


def _has_virtual_ops(circuit: QuantumCircuit) -> bool:
    qc = _normalize_circuit(circuit)
    for instr in qc.data:
        op = instr.operation
        if isinstance(op, (VirtualBinaryGate, VirtualMove, VirtualGateEndpoint, WireCut)):
            return True
    return False


def _replace_virtual_ops(circuit: QuantumCircuit, metric_mode: str) -> QuantumCircuit:
    qc = _normalize_circuit(circuit)
    new_circuit = QuantumCircuit(*qc.qregs, *qc.cregs)
    for instr in qc.data:
        op = instr.operation
        if isinstance(op, (VirtualBinaryGate, VirtualMove, VirtualGateEndpoint)):
            if metric_mode == "cutqc":
                continue
            if len(instr.qubits) == 2:
                new_circuit.append(CXGate(), instr.qubits, instr.clbits)
            else:
                continue
        elif isinstance(op, WireCut):
            continue
        else:
            new_circuit.append(op, instr.qubits, instr.clbits)
    return new_circuit


def _max_metric(vals: List[int]) -> int:
    if not vals:
        return 0
    return int(max(vals))


_METRICS_BACKENDS: Dict[str, object] = {}


def _get_metrics_backend(baseline: str):
    if baseline == "raw":
        return None
    if baseline in _METRICS_BACKENDS:
        return _METRICS_BACKENDS[baseline]

    if baseline == "kolkata":
        try:
            from qiskit.providers.fake_provider import FakeKolkataV2

            backend = FakeKolkataV2()
        except Exception:
            try:
                from qiskit_ibm_runtime.fake_provider import FakeKolkataV2

                backend = FakeKolkataV2()
            except Exception as exc:
                raise RuntimeError(
                    "Kolkata metrics baseline requested but FakeKolkataV2 is unavailable."
                ) from exc
    elif baseline == "torino":
        try:
            from qiskit_ibm_runtime.fake_provider import FakeTorino

            backend = FakeTorino()
        except Exception as exc:
            raise RuntimeError(
                "Torino metrics baseline requested but FakeTorino is unavailable."
            ) from exc
    elif baseline == "marrakesh":
        try:
            from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

            backend = FakeMarrakesh()
        except Exception as exc:
            raise RuntimeError(
                "Marrakesh metrics baseline requested but FakeMarrakesh is unavailable."
            ) from exc
    else:
        raise ValueError(f"Unknown metrics baseline: {baseline}")

    _METRICS_BACKENDS[baseline] = backend
    return backend


def _apply_metrics_baseline(
    circuit: QuantumCircuit, baseline: str, opt_level: int
) -> QuantumCircuit:
    if baseline == "raw":
        return circuit
    backend = _get_metrics_backend(baseline)
    if _is_eval_verbose():
        t0 = time.perf_counter()
        _EVAL_LOGGER.info(
            "Metrics transpile start baseline=%s opt=%s ops=%s",
            baseline,
            opt_level,
            len(circuit.data),
        )
        out = transpile(circuit, backend=backend, optimization_level=opt_level)
        _EVAL_LOGGER.info(
            "Metrics transpile done baseline=%s opt=%s sec=%.2f",
            baseline,
            opt_level,
            time.perf_counter() - t0,
        )
        return out
    return transpile(circuit, backend=backend, optimization_level=opt_level)


def _resolve_program_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    if candidate.exists():
        return candidate.resolve()
    rel = (ROOT / candidate).resolve()
    return rel if rel.exists() else None


def _find_qose_program(path_hint: str) -> Optional[Path]:
    candidates = []
    if path_hint:
        candidates.append(path_hint)
    env_hint = os.getenv("QOSE_PROGRAM", "").strip()
    if env_hint:
        candidates.append(env_hint)
    candidates.extend(
        [
            str(ROOT / "openevolve_output" / "best" / "best_program.py"),
            str(ROOT / "openaievolve" / "evolved_target.py"),
            str(ROOT / "qos" / "error_mitigator" / "evolution_target.py"),
        ]
    )
    for entry in candidates:
        resolved = _resolve_program_path(entry)
        if resolved:
            return resolved
    return None


def _load_qose_run(qose_program: Path) -> Callable:
    spec = importlib.util.spec_from_file_location("qose_program", str(qose_program))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load QOSE program at {qose_program}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evolved_cost_search = getattr(module, "evolved_cost_search", None)
    if callable(evolved_cost_search):
        def _wrapped(mitigator: ErrorMitigator, q: Qernel):
            mitigator._cost_search_impl = evolved_cost_search.__get__(
                mitigator, ErrorMitigator
            )
            return mitigator.run(q)

        return _wrapped
    evolved_run = getattr(module, "evolved_run", None)
    if not callable(evolved_run):
        raise AttributeError(
            f"Program {qose_program} has no callable evolved_cost_search() or evolved_run()"
        )
    return evolved_run


def _relative_methods(include_qose: bool) -> List[str]:
    methods = ["FrozenQubits", "CutQC", "QOS", "QOSN"]
    if include_qose:
        methods.append("QOSE")
    return methods


def _fidelity_methods(include_qose: bool) -> List[str]:
    methods = ["Qiskit", "FrozenQubits", "CutQC", "QOS", "QOSN"]
    if include_qose:
        methods.append("QOSE")
    return methods


def _timing_methods(include_qose: bool) -> List[str]:
    methods = ["FrozenQubits", "CutQC", "QOS", "QOSN"]
    if include_qose:
        methods.append("QOSE")
    return methods


def _cut_methods(include_qose: bool) -> List[str]:
    methods = ["Qiskit", "QOS", "QOSN"]
    if include_qose:
        methods.append("QOSE")
    methods.extend(["FrozenQubits", "CutQC"])
    return methods


def _parse_methods(value: str, include_qose: bool) -> List[str]:
    raw = value.strip()
    if not raw or raw.lower() in {"all", "default"}:
        return _relative_methods(include_qose)

    requested = []
    for entry in raw.split(","):
        key = entry.strip().lower()
        if not key:
            continue
        if key == "qiskit":
            continue
        method = METHOD_ALIASES.get(key)
        if method is None:
            raise ValueError(f"Unknown method: {entry}")
        requested.append(method)

    ordered = []
    seen = set()
    for method in METHOD_ORDER:
        if method in requested and method not in seen:
            ordered.append(method)
            seen.add(method)
    return ordered


def _append_cost_search_log(path: str, row: Dict[str, object]) -> None:
    if not path:
        return
    needs_header = True
    if os.path.exists(path):
        try:
            needs_header = os.path.getsize(path) == 0
        except OSError:
            needs_header = True
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


class SerialPool:
    def map(self, fn, iterable):
        return list(map(fn, iterable))

    def starmap(self, fn, iterable):
        return [fn(*args) for args in iterable]


def _cleanup_children() -> None:
    children = mp.active_children()
    if not children:
        return
    for child in children:
        try:
            child.terminate()
        except Exception:
            pass
    for child in children:
        try:
            child.join(timeout=1)
        except Exception:
            pass


class _CountingMitigator(ErrorMitigator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_search_calls = 0

    def computeCuttingCosts(self, q: Qernel, size_to_reach: int):
        self.cost_search_calls += 1
        costs = super().computeCuttingCosts(q, size_to_reach)
        log_path = getattr(self, "_trace_log_path", "")
        if log_path:
            row = {
                "bench": getattr(self, "_trace_bench", ""),
                "size": getattr(self, "_trace_size", ""),
                "method": getattr(self, "_trace_method", ""),
                "cost_search_call": self.cost_search_calls,
                "size_to_reach": size_to_reach,
                "budget": self.budget,
                "gv_cost": costs.get("GV"),
                "wc_cost": costs.get("WC"),
                "gv_sec": getattr(self, "_last_gv_sec", 0.0),
                "wc_sec": getattr(self, "_last_wc_sec", 0.0),
            }
            _append_cost_search_log(log_path, row)
        if getattr(self, "_trace_verbose", False):
            print(
                f"[FullEval] cost_search call={self.cost_search_calls} "
                f"size_to_reach={size_to_reach} budget={self.budget} "
                f"GV={costs.get('GV')} ({getattr(self, '_last_gv_sec', 0.0):.2f}s) "
                f"WC={costs.get('WC')} ({getattr(self, '_last_wc_sec', 0.0):.2f}s)",
                flush=True,
            )
        return costs

    def cost_search(self, q: Qernel, size_to_reach: int, budget: int):
        size, method, cost_time, timed_out = super().cost_search(
            q, size_to_reach, budget
        )
        if timed_out:
            log_path = getattr(self, "_trace_log_path", "")
            if log_path:
                gv_cost = None
                wc_cost = None
                if getattr(self, "_qose_gv_cost_trace", None):
                    gv_cost = self._qose_gv_cost_trace[-1]
                if getattr(self, "_qose_wc_cost_trace", None):
                    wc_cost = self._qose_wc_cost_trace[-1]
                row = {
                    "bench": getattr(self, "_trace_bench", ""),
                    "size": getattr(self, "_trace_size", ""),
                    "method": getattr(self, "_trace_method", ""),
                    "cost_search_call": self.cost_search_calls,
                    "size_to_reach": size,
                    "budget": self.budget,
                    "gv_cost": gv_cost,
                    "wc_cost": wc_cost,
                    "gv_sec": -1.0,
                    "wc_sec": -1.0,
                }
                _append_cost_search_log(log_path, row)
            if getattr(self, "_trace_verbose", False):
                print(
                    f"[FullEval] cost_search timeout size_to_reach={size} budget={self.budget} "
                    "GV=-1.00s WC=-1.00s",
                    flush=True,
                )
        return size, method, cost_time, timed_out


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


def _import_aer():
    try:
        from qiskit_aer import AerSimulator  # type: ignore
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError  # type: ignore
        return AerSimulator, NoiseModel, depolarizing_error, ReadoutError
    except ModuleNotFoundError:
        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            sys.path.append(user_site)
        from qiskit_aer import AerSimulator  # type: ignore
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError  # type: ignore
        return AerSimulator, NoiseModel, depolarizing_error, ReadoutError


def _noise_model(p1: float, p2: float, readout: float):
    AerSimulator, NoiseModel, depolarizing_error, ReadoutError = _import_aer()
    noise = NoiseModel()
    err1 = depolarizing_error(p1, 1)
    err2 = depolarizing_error(p2, 2)
    one_q = ["x", "y", "z", "h", "rx", "ry", "rz", "sx", "id", "s", "sdg"]
    two_q = ["cx", "cz", "swap", "rzz", "cp", "ecr"]
    noise.add_all_qubit_quantum_error(err1, one_q)
    noise.add_all_qubit_quantum_error(err2, two_q)
    if readout > 0:
        ro = ReadoutError([[1 - readout, readout], [readout, 1 - readout]])
        noise.add_all_qubit_readout_error(ro)
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


def _simulate_counts(circuit: QuantumCircuit, shots: int, noise, seed: int) -> Dict[str, int]:
    AerSimulator, _, _, _ = _import_aer()
    sim = AerSimulator(noise_model=noise) if noise else AerSimulator()
    circ = _ensure_measurements(circuit)
    result = sim.run(circ, shots=shots, seed_simulator=seed, seed_transpiler=seed).result()
    return result.get_counts()


def _simulate_virtual_counts(
    circuit: QuantumCircuit | VirtualCircuit,
    shots: int,
    noise,
    seed: int,
) -> Dict[str, int]:
    vc = circuit if isinstance(circuit, VirtualCircuit) else VirtualCircuit(circuit)
    results: Dict = {}
    for frag, frag_circ in vc.fragment_circuits.items():
        inst_labels = vc.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circ, inst_labels)
        distrs = []
        for inst in instantiations:
            counts = _simulate_counts(inst, shots, noise, seed)
            distrs.append(QuasiDistr.from_counts(counts))
        results[frag] = distrs
    quasi = vc.knit(results, SerialPool())
    return quasi.to_counts(vc._circuit.num_clbits, shots)


_REAL_BACKEND_CACHE: Dict[str, object] = {}


def _get_real_backend(backend_name: str):
    if backend_name in _REAL_BACKEND_CACHE:
        return _REAL_BACKEND_CACHE[backend_name]
    token = (
        os.getenv("IBM_TOKEN")
        or os.getenv("QISKIT_IBM_TOKEN")
        or os.getenv("IBMQ_TOKEN")
    )
    if not token:
        token_path = Path("IBM_token.key")
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        raise RuntimeError("IBM_TOKEN (or QISKIT_IBM_TOKEN/IBMQ_TOKEN) is not set.")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore

        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.backend(backend_name)
    except Exception as exc:
        try:
            from qiskit_ibm_provider import IBMProvider  # type: ignore

            provider = IBMProvider(token=token)
            backend = provider.get_backend(backend_name)
        except Exception as provider_exc:
            raise RuntimeError(
                "Failed to initialize IBM backend. Install qiskit-ibm-provider "
                "or verify QiskitRuntimeService credentials."
            ) from provider_exc
    _REAL_BACKEND_CACHE[backend_name] = backend
    return backend


def _quasi_to_counts(quasi, shots: int, num_bits: int) -> Dict[str, int]:
    if hasattr(quasi, "binary_probabilities"):
        probs = quasi.binary_probabilities(num_bits=num_bits)
    else:
        probs = {}
        for key, val in quasi.items():
            if isinstance(key, int):
                bitstr = format(key, f"0{num_bits}b")
            else:
                bitstr = str(key)
            probs[bitstr] = float(val)
    counts = {bitstr: int(round(prob * shots)) for bitstr, prob in probs.items()}
    total = sum(counts.values())
    if counts and total != shots:
        max_key = max(counts, key=counts.get)
        counts[max_key] += shots - total
    return counts


def _sampler_result_to_counts(result, shots: int, num_bits: int) -> Dict[str, int]:
    quasi_dists = getattr(result, "quasi_dists", None)
    if quasi_dists is not None:
        return _quasi_to_counts(quasi_dists[0], shots, num_bits)

    pub = None
    try:
        pub = result[0]
    except Exception:
        pub = None

    if pub is not None:
        if hasattr(pub, "join_data"):
            try:
                bitarray = pub.join_data()
                if hasattr(bitarray, "get_counts"):
                    return bitarray.get_counts()
            except Exception:
                pass

        data = getattr(pub, "data", None)
        if data is not None:
            try:
                for name in data:
                    val = data[name]
                    if hasattr(val, "get_counts"):
                        return val.get_counts()
            except Exception:
                pass
            if hasattr(data, "meas"):
                val = data.meas
                if hasattr(val, "get_counts"):
                    return val.get_counts()

    if hasattr(result, "items"):
        return _quasi_to_counts(result, shots, num_bits)

    raise RuntimeError("Unsupported sampler result type for counts extraction.")


def _real_counts(circuit: QuantumCircuit, shots: int, backend_name: str) -> Dict[str, int]:
    backend = _get_real_backend(backend_name)
    circ = _ensure_measurements(circuit)
    tcirc = transpile(circ, backend=backend, optimization_level=0)
    try:
        try:
            from qiskit_ibm_runtime import Sampler  # type: ignore
        except ImportError:
            from qiskit_ibm_runtime import SamplerV2 as Sampler  # type: ignore

        sampler = Sampler(mode=backend)
        job = sampler.run([tcirc], shots=shots)
        result = job.result()
        return _sampler_result_to_counts(result, shots, tcirc.num_clbits)
    except Exception as exc:
        raise RuntimeError(
            "Real-backend execution failed. "
            "Ensure your plan supports job execution with primitives."
        ) from exc


def _real_virtual_counts(
    circuit: QuantumCircuit | VirtualCircuit,
    shots: int,
    backend_name: str,
) -> Dict[str, int]:
    vc = circuit if isinstance(circuit, VirtualCircuit) else VirtualCircuit(circuit)
    results: Dict = {}
    for frag, frag_circ in vc.fragment_circuits.items():
        inst_labels = vc.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circ, inst_labels)
        distrs = []
        for inst in instantiations:
            counts = _real_counts(inst, shots, backend_name)
            distrs.append(QuasiDistr.from_counts(counts))
        results[frag] = distrs
    quasi = vc.knit(results, SerialPool())
    return quasi.to_counts(vc._circuit.num_clbits, shots)


def _count_real_jobs(circuit: QuantumCircuit) -> int:
    if _has_virtual_ops(circuit):
        vc = VirtualCircuit(_normalize_circuit(circuit))
        total = 0
        for frag, frag_circ in vc.fragment_circuits.items():
            inst_labels = vc.get_instance_labels(frag)
            instantiations = generate_instantiations(frag_circ, inst_labels)
            total += len(instantiations)
        return total
    return 1


def _real_fidelity_for_circuit(
    circuit: QuantumCircuit,
    shots: int,
    backend_name: str,
    seed: int,
) -> float:
    if _has_virtual_ops(circuit):
        ideal = _simulate_virtual_counts(circuit, shots, None, seed)
        noisy = _real_virtual_counts(circuit, shots, backend_name)
    else:
        ideal = _simulate_counts(circuit, shots, None, seed)
        noisy = _real_counts(circuit, shots, backend_name)
    if not ideal or not noisy:
        return 0.0
    return _hellinger_fidelity_from_counts(ideal, noisy)


def _average_real_fidelity(
    circuits: List[QuantumCircuit],
    shots: int,
    backend_name: str,
    seed: int,
) -> float:
    total = len(circuits)
    vals = []
    for idx, circuit in enumerate(circuits, start=1):
        job_count = _count_real_jobs(circuit)
        _EVAL_LOGGER.warning(
            "Real QPU progress %s/%s backend=%s jobs=%s",
            idx,
            total,
            backend_name,
            job_count,
        )
        vals.append(_real_fidelity_for_circuit(circuit, shots, backend_name, seed))
    return float(sum(vals) / max(1, len(vals)))


def _fidelity_for_circuit(
    circuit: QuantumCircuit,
    shots: int,
    noise,
    seed: int,
) -> float:
    if _has_virtual_ops(circuit):
        ideal = _simulate_virtual_counts(circuit, shots, None, seed)
        noisy = _simulate_virtual_counts(circuit, shots, noise, seed)
    else:
        ideal = _simulate_counts(circuit, shots, None, seed)
        noisy = _simulate_counts(circuit, shots, noise, seed)
    if not ideal or not noisy:
        return 0.0
    return _hellinger_fidelity_from_counts(ideal, noisy)


def _average_fidelity(
    circuits: List[QuantumCircuit],
    shots: int,
    noise,
    seed: int,
) -> float:
    vals = [_fidelity_for_circuit(c, shots, noise, seed) for c in circuits]
    return float(sum(vals) / max(1, len(vals)))


def _analyze_circuit(
    circuit: QuantumCircuit,
    metric_mode: str,
    metrics_baseline: str,
    metrics_opt_level: int,
) -> Dict[str, int]:
    if _has_virtual_ops(circuit):
        if _is_eval_verbose():
            _EVAL_LOGGER.info(
                "Analyze circuit virtual ops mode=%s baseline=%s opt=%s",
                metric_mode,
                metrics_baseline,
                metrics_opt_level,
            )
        if metric_mode == "virtual" or metric_mode == "cutqc":
            effective = _replace_virtual_ops(_normalize_circuit(circuit), metric_mode)
            effective = _apply_metrics_baseline(effective, metrics_baseline, metrics_opt_level)
            q = Qernel(effective)
            BasicAnalysisPass().run(q)
            m = q.get_metadata()
            cnot = int(m.get("num_cnot_gates", m.get("num_nonlocal_gates", 0)))
            return {
                "depth": int(m.get("depth", 0)),
                "num_nonlocal_gates": cnot,
            }

        vc = VirtualCircuit(_normalize_circuit(circuit))
        if _is_eval_verbose():
            _EVAL_LOGGER.info(
                "Analyze circuit fragments=%s mode=%s baseline=%s opt=%s",
                len(vc.fragment_circuits),
                metric_mode,
                metrics_baseline,
                metrics_opt_level,
            )
        depths = []
        nonlocals = []
        for frag_circ in vc.fragment_circuits.values():
            effective = _replace_virtual_ops(frag_circ, metric_mode)
            effective = _apply_metrics_baseline(effective, metrics_baseline, metrics_opt_level)
            q = Qernel(effective)
            BasicAnalysisPass().run(q)
            m = q.get_metadata()
            depths.append(int(m.get("depth", 0)))
            nonlocals.append(int(m.get("num_cnot_gates", m.get("num_nonlocal_gates", 0))))
        return {
            "depth": max(depths) if depths else 0,
            "num_nonlocal_gates": _max_metric(nonlocals),
        }
    if _is_eval_verbose():
        _EVAL_LOGGER.info(
            "Analyze circuit mode=%s baseline=%s opt=%s ops=%s",
            metric_mode,
            metrics_baseline,
            metrics_opt_level,
            len(_normalize_circuit(circuit).data),
        )
    effective = _replace_virtual_ops(circuit, metric_mode)
    effective = _apply_metrics_baseline(effective, metrics_baseline, metrics_opt_level)
    q = Qernel(effective)
    BasicAnalysisPass().run(q)
    m = q.get_metadata()
    cnot = int(m.get("num_cnot_gates", m.get("num_nonlocal_gates", 0)))
    return {
        "depth": int(m.get("depth", 0)),
        "num_nonlocal_gates": cnot,
    }


def _extract_circuits(q: Qernel) -> List[QuantumCircuit]:
    vsqs = q.get_virtual_subqernels()
    if vsqs:
        return [vsq.get_circuit() for vsq in vsqs]
    return [q.get_circuit()]


def _analyze_qernel(
    q: Qernel,
    metric_mode: str,
    metrics_baseline: str,
    metrics_opt_level: int,
) -> Dict[str, int]:
    circuits = []
    vsqs = q.get_virtual_subqernels()
    if vsqs:
        circuits = [_normalize_circuit(vsq.get_circuit()) for vsq in vsqs]
    else:
        circuits = [_normalize_circuit(q.get_circuit())]
    if _is_eval_verbose():
        _EVAL_LOGGER.info(
            "Analyze qernel circuits=%s mode=%s baseline=%s opt=%s",
            len(circuits),
            metric_mode,
            metrics_baseline,
            metrics_opt_level,
        )

    depths = []
    nonlocals = []
    for idx, c in enumerate(circuits, start=1):
        t0 = time.perf_counter() if _is_eval_verbose() else 0.0
        m = _analyze_circuit(
            c,
            metric_mode,
            metrics_baseline,
            metrics_opt_level,
        )
        if _is_eval_verbose():
            _EVAL_LOGGER.info(
                "Analyze qernel circuit %s/%s sec=%.2f",
                idx,
                len(circuits),
                time.perf_counter() - t0,
            )
        depths.append(m["depth"])
        nonlocals.append(m["num_nonlocal_gates"])

    return {
        "depth": max(depths) if depths else 0,
        "num_nonlocal_gates": _max_metric(nonlocals),
    }


def _run_mitigator(
    qc: QuantumCircuit,
    methods: List[str],
    args,
    use_cost_search_override: Optional[bool] = None,
    bench_name: str = "",
    size_label: str | int | None = None,
) -> Tuple[Dict[str, int], Dict[str, float], List[QuantumCircuit]]:
    base_size = qc.num_qubits if args.size_to_reach <= 0 else args.size_to_reach
    size_candidates = [base_size, qc.num_qubits]
    if size_candidates[0] == size_candidates[1]:
        size_candidates = [size_candidates[0]]
    budget = args.budget

    for size_to_reach in size_candidates:
        q = Qernel(qc.copy())
        if use_cost_search_override is None:
            use_cost_search = bool(methods == [] and args.qos_cost_search)
        else:
            use_cost_search = bool(methods == [] and use_cost_search_override)
        mitigator_cls = ErrorMitigator
        if methods == [] and use_cost_search:
            mitigator_cls = _CountingMitigator
        mitigator = mitigator_cls(
            size_to_reach=size_to_reach,
            ideal_size_to_reach=args.ideal_size_to_reach,
            budget=budget,
            methods=methods,
            use_cost_search=use_cost_search,
            collect_timing=args.collect_timing,
        )
        if isinstance(mitigator, _CountingMitigator):
            mitigator._trace_log_path = getattr(args, "cost_search_log", "")
            mitigator._trace_verbose = bool(args.verbose)
            mitigator._trace_bench = bench_name
            mitigator._trace_size = size_label if size_label is not None else qc.num_qubits
            mitigator._trace_method = "QOS"
        try:
            use_alarm = not (methods == [] or "WC" in methods or "GV" in methods)
            if use_alarm:
                signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError()))
                signal.alarm(args.timeout_sec)
            start = time.perf_counter() if args.collect_timing else 0.0
            q = mitigator.run(q)
            if args.collect_timing and "total" not in mitigator.timings:
                mitigator.timings["total"] = time.perf_counter() - start
            if args.collect_timing and hasattr(mitigator, "cost_search_calls"):
                mitigator.timings["cost_search_calls"] = mitigator.cost_search_calls
            if use_alarm:
                signal.alarm(0)
            return (
                _analyze_qernel(
                    q,
                    args.metric_mode,
                    args.metrics_baseline,
                    args.metrics_optimization_level,
                ),
                mitigator.timings,
                _extract_circuits(q),
            )
        except (ValueError, TimeoutError):
            if "SIGALRM" in signal.Signals.__members__:
                signal.alarm(0)
            continue

    return (
        _analyze_circuit(
            qc,
            args.metric_mode,
            args.metrics_baseline,
            args.metrics_optimization_level,
        ),
        {},
        [qc],
    )


def _run_qose(
    qc: QuantumCircuit, args, evolved_run: Callable
) -> Tuple[Optional[Dict[str, int]], Dict[str, float], List[QuantumCircuit]]:
    q = Qernel(qc.copy())
    BasicAnalysisPass().run(q)
    SupermarqFeaturesAnalysisPass().run(q)
    base_size = qc.num_qubits if args.size_to_reach <= 0 else args.size_to_reach
    mitigator = ErrorMitigator(
        size_to_reach=base_size,
        ideal_size_to_reach=args.ideal_size_to_reach,
        budget=args.budget,
        methods=[],
        use_cost_search=bool(args.qos_cost_search),
        collect_timing=args.collect_timing,
    )
    try:
        if "SIGALRM" in signal.Signals.__members__:
            signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(args.timeout_sec)
        start = time.perf_counter() if args.collect_timing else 0.0
        q = evolved_run(mitigator, q)
        if args.collect_timing and "total" not in mitigator.timings:
            mitigator.timings["total"] = time.perf_counter() - start
        if "SIGALRM" in signal.Signals.__members__:
            signal.alarm(0)
        return (
            _analyze_qernel(
                q,
                args.metric_mode,
                args.metrics_baseline,
                args.metrics_optimization_level,
            ),
            mitigator.timings,
            _extract_circuits(q),
        )
    except (ValueError, TimeoutError):
        if "SIGALRM" in signal.Signals.__members__:
            signal.alarm(0)
        return None, {}, [qc]


def _relative(value: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0
    return value / baseline


def _safe_ratio(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return value / baseline


def _plot_panel(
    ax,
    title: str,
    rel_data: Dict[str, Dict[str, float]],
    benches: List[Tuple[str, str]],
    methods: List[str],
    ylabel: str,
    show_avg: bool = False,
) -> None:
    x = np.arange(len(methods))
    width = 0.08
    all_vals: List[float] = []
    for i, (bench, label) in enumerate(benches):
        vals = [rel_data[bench].get(m, 1.0) for m in methods]
        all_vals.extend(vals)
        ax.bar(x + (i - len(benches) / 2) * width, vals, width, label=label)

    if show_avg:
        max_val = max(all_vals) if all_vals else 0.0
        pad = max(0.02, max_val * 0.06)
        if max_val > 0:
            ax.set_ylim(top=max(ax.get_ylim()[1], max_val + pad * 2))
        for idx, method in enumerate(methods):
            vals = [rel_data[bench].get(method, 1.0) for bench, _ in benches]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            y = max(vals) + pad
            ax.text(
                x[idx],
                y,
                f"avg {avg:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def _plot_combined(
    rel_by_size: Dict[int, Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]],
    benches: List[Tuple[str, str]],
    out_dir: Path,
    timestamp: str,
    fidelity_by_size: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
    real_fidelity_by_size: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
    methods: Optional[List[str]] = None,
    fidelity_methods: Optional[List[str]] = None,
) -> Path:
    plt = _import_matplotlib()
    if methods is None:
        methods = _relative_methods(False)
    if fidelity_methods is None:
        fidelity_methods = _fidelity_methods(False)
    sizes = sorted(rel_by_size.keys())
    rows = max(1, len(sizes))
    cols = 2 + (1 if fidelity_by_size else 0) + (1 if real_fidelity_by_size else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.6 * rows))
    if rows == 1:
        axes = np.array([axes])

    for row, size in enumerate(sizes):
        rel_depth, rel_nonlocal = rel_by_size[size]
        _plot_panel(
            axes[row, 0],
            f"Depth - {size} qubits (lower is better)",
            rel_depth,
            benches,
            methods,
            "Relative to Qiskit",
            show_avg=True,
        )
        _plot_panel(
            axes[row, 1],
            f"Number of CNOT gates - {size} qubits (lower is better)",
            rel_nonlocal,
            benches,
            methods,
            "Relative to Qiskit",
            show_avg=True,
        )
        if fidelity_by_size:
            fidelity = fidelity_by_size.get(size, {})
            _plot_panel(
                axes[row, 2],
                f"Hellinger fidelity - {size} qubits (higher is better)",
                fidelity,
                benches,
                fidelity_methods,
                "Fidelity",
                show_avg=True,
            )
        if real_fidelity_by_size:
            col = 3 if fidelity_by_size else 2
            fidelity = real_fidelity_by_size.get(size, {})
            _plot_panel(
                axes[row, col],
                f"Real Hellinger fidelity - {size} qubits (higher is better)",
                fidelity,
                benches,
                fidelity_methods,
                "Fidelity",
                show_avg=True,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=8, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / f"relative_properties_compare_{timestamp}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_timing(
    timing_rows: List[Dict[str, object]],
    out_dir: Path,
    timestamp: str,
    methods: List[str],
) -> Path:
    plt = _import_matplotlib()
    if not timing_rows:
        raise ValueError("No timing data to plot.")

    stages = ["analysis", "qaoa_analysis", "qf", "cost_search", "gv", "wc", "qr", "cut_select"]
    skip_stages = {"total", "overall", "simulation", "cost_search_calls"}
    seen = set(stages)
    for row in timing_rows:
        for k in row.keys():
            if k in {"bench", "size", "method"} or k in skip_stages:
                continue
            if k not in seen:
                stages.append(k)
                seen.add(k)

    def _row_total(row: Dict[str, object]) -> float:
        if "total" in row:
            try:
                return float(row["total"])
            except (TypeError, ValueError):
                return 0.0
        total = 0.0
        for k, v in row.items():
            if k in {"bench", "size", "method"}:
                continue
            try:
                total += float(v)
            except (TypeError, ValueError):
                continue
        return total

    sizes = sorted({int(r["size"]) for r in timing_rows})
    methods = list(methods)

    fig, axes = plt.subplots(len(sizes), 4, figsize=(22.0, 3.8 * len(sizes)))
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, 4)

    for row_idx, size in enumerate(sizes):
        ax = axes[row_idx, 0]
        x = np.arange(len(methods))
        totals = []
        for method in methods:
            method_rows = [
                r for r in timing_rows if int(r["size"]) == size and r["method"] == method
            ]
            vals = [_row_total(r) for r in method_rows]
            totals.append(sum(vals) / max(1, len(vals)))
        ax.bar(x, totals)
        ax.set_title(f"Total Timing (Avg) - {size} qubits")
        ax.set_ylabel("Seconds")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for idx, total in enumerate(totals):
            ax.text(x[idx], total, f"{total:.2f}s", ha="center", va="bottom", fontsize=7)

        ax = axes[row_idx, 1]
        bench_labels = []
        for bench, label in BENCHES:
            if any(
                int(r["size"]) == size and r["method"] == "QOS" and r["bench"] == bench
                for r in timing_rows
            ):
                bench_labels.append((bench, label))
        if not bench_labels:
            ax.set_visible(False)
            continue
        x = np.arange(len(bench_labels))
        vals = []
        for bench, _label in bench_labels:
            bench_rows = [
                r
                for r in timing_rows
                if int(r["size"]) == size and r["method"] == "QOS" and r["bench"] == bench
            ]
            totals = [_row_total(r) for r in bench_rows]
            vals.append(sum(totals) / max(1, len(totals)))
        ax.bar(x, vals)
        ax.set_title(f"QOS Timing by Circuit (Total) - {size} qubits")
        ax.set_ylabel("Seconds")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _bench, label in bench_labels], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        ax = axes[row_idx, 2]
        bench_labels = []
        for bench, label in BENCHES:
            if any(
                int(r["size"]) == size and r["method"] == "QOS" and r["bench"] == bench
                for r in timing_rows
            ):
                bench_labels.append((bench, label))
        if not bench_labels:
            ax.set_visible(False)
            continue
        x = np.arange(len(bench_labels))
        bottom = np.zeros(len(bench_labels))
        for stage in stages:
            vals = []
            for bench, _label in bench_labels:
                stage_vals = [
                    float(r.get(stage, 0.0))
                    for r in timing_rows
                    if int(r["size"]) == size
                    and r["method"] == "QOS"
                    and r["bench"] == bench
                    and stage in r
                ]
                vals.append(sum(stage_vals) / max(1, len(stage_vals)))
            ax.bar(x, vals, bottom=bottom, label=stage)
            bottom = bottom + np.array(vals)
        ax.set_title(f"QOS Timing Breakdown by Circuit - {size} qubits")
        ax.set_ylabel("Seconds")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _bench, label in bench_labels], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)

        ax = axes[row_idx, 3]
        call_vals = []
        for bench, _label in bench_labels:
            vals = [
                float(r.get("cost_search_calls", 0.0))
                for r in timing_rows
                if int(r["size"]) == size
                and r["method"] == "QOS"
                and r["bench"] == bench
                and "cost_search_calls" in r
            ]
            call_vals.append(sum(vals) / max(1, len(vals)))
        if not any(call_vals):
            ax.set_visible(False)
            continue
        x = np.arange(len(bench_labels))
        ax.bar(x, call_vals)
        ax.set_title(f"Cost Search Calls by Circuit - {size} qubits")
        ax.set_ylabel("Calls")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _bench, label in bench_labels], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = out_dir / f"timing_breakdown_{timestamp}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_overheads(
    rows: List[Dict[str, object]],
    benches: List[Tuple[str, str]],
    sizes: List[int],
    out_dir: Path,
    timestamp: str,
    methods: List[str],
) -> Path:
    plt = _import_matplotlib()
    min_factor = 1e-3

    def _val(row, key):
        v = row.get(key, 0.0)
        if v == "" or v is None:
            return min_factor
        try:
            v = float(v)
        except (TypeError, ValueError):
            return min_factor
        if v <= 0:
            return min_factor
        return v

    fig, axes = plt.subplots(len(sizes), 3, figsize=(16.5, 3.8 * len(sizes)))
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, 3)

    methods = list(methods)
    method_keys = {
        "FrozenQubits": ("fq_classical_overhead", "fq_quantum_overhead"),
        "CutQC": ("cutqc_classical_overhead", "cutqc_quantum_overhead"),
        "QOS": ("qos_classical_overhead", "qos_quantum_overhead"),
        "QOSN": ("qosn_classical_overhead", "qosn_quantum_overhead"),
    }
    if "QOSE" in methods:
        method_keys["QOSE"] = ("qose_classical_overhead", "qose_quantum_overhead")

    for row_idx, size in enumerate(sizes):
        size_rows = [r for r in rows if int(r["size"]) == size]
        by_bench = {r["bench"]: r for r in size_rows}
        bench_labels = [(b, label) for b, label in benches if b in by_bench]
        x = np.arange(len(bench_labels))
        width = 0.8 / max(1, len(methods))

        ax = axes[row_idx, 0]
        for idx, method in enumerate(methods):
            vals = []
            key, _qkey = method_keys[method]
            for bench, _label in bench_labels:
                vals.append(_val(by_bench[bench], key))
            offset = (idx - (len(methods) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=method)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"Classical Overhead - {size} qubits")
        ax.set_ylabel("Factor vs Qiskit")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _b, label in bench_labels], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        ax = axes[row_idx, 1]
        for idx, method in enumerate(methods):
            vals = []
            _ckey, qkey = method_keys[method]
            for bench, _label in bench_labels:
                vals.append(_val(by_bench[bench], qkey))
            offset = (idx - (len(methods) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=method)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"Quantum Overhead - {size} qubits")
        ax.set_ylabel("Factor vs Qiskit")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _b, label in bench_labels], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        ax = axes[row_idx, 2]
        if "QOS" in methods:
            qos_class_vals = []
            qos_quant_vals = []
            for bench, _label in bench_labels:
                row = by_bench[bench]
                qos_class_vals.append(_val(row, "qos_classical_overhead"))
                qos_quant_vals.append(_val(row, "qos_quantum_overhead"))
            ax.bar(x - width / 2, qos_class_vals, width, label="Classical")
            ax.bar(x + width / 2, qos_quant_vals, width, label="Quantum")
            ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_title(f"QOS Overhead by Circuit - {size} qubits")
            ax.set_ylabel("Factor vs Qiskit")
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([label for _b, label in bench_labels], rotation=45, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend(fontsize=7)
        else:
            ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=8, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_path = out_dir / f"overhead_breakdown_{timestamp}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _fragment_fidelity_sweep(
    circuits: List[QuantumCircuit],
    shots_list: List[int],
    noise,
    seed: int,
) -> Dict[str, object]:
    data: Dict[str, object] = {"shots": shots_list, "fragments": {}}
    fragments: Dict[str, List[float]] = {}
    for idx, circuit in enumerate(circuits):
        label = f"frag{idx}"
        vals = []
        for shots in shots_list:
            vals.append(_fidelity_for_circuit(circuit, shots, noise, seed))
        fragments[label] = vals
    data["fragments"] = fragments
    return data


def _plot_fragment_fidelity_sweep(
    fragment_data: Dict[Tuple[int, str], Dict[str, object]],
    benches: List[Tuple[str, str]],
    sizes: List[int],
    out_dir: Path,
    timestamp: str,
) -> List[Path]:
    plt = _import_matplotlib()
    out_paths: List[Path] = []
    for size in sizes:
        fig, axes = plt.subplots(3, 3, figsize=(12.5, 10.5), sharex=True, sharey=True)
        axes = np.array(axes).reshape(3, 3)
        for idx, (bench, label) in enumerate(benches):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            data = fragment_data.get((size, bench))
            if not data:
                ax.set_visible(False)
                continue
            shots = data["shots"]
            fragments = data["fragments"]
            for frag_label, vals in fragments.items():
                ax.plot(shots, vals, marker="o", linewidth=1, markersize=3, label=frag_label)
            ax.set_title(label)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_xlabel("Shots")
            ax.set_ylabel("Hellinger fidelity")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, ncol=4, fontsize=7, loc="upper center")
            fig.tight_layout(rect=(0, 0, 1, 0.9))
        else:
            fig.tight_layout()
        out_path = out_dir / f"fragment_fidelity_{timestamp}_s{size}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def _summarize(rows: List[Dict[str, object]], methods: List[str]) -> Dict[str, float]:
    def _avg(key: str) -> Optional[float]:
        vals = []
        for r in rows:
            if key not in r:
                continue
            val = r[key]
            if val == "" or val is None:
                continue
            try:
                vals.append(float(val))
            except (TypeError, ValueError):
                continue
        if not vals:
            return None
        return sum(vals) / len(vals)

    rel_depth_keys = {
        "QOS": "rel_depth_qos",
        "QOSN": "rel_depth_qosn",
        "FrozenQubits": "rel_depth_fq",
        "CutQC": "rel_depth_cutqc",
        "QOSE": "rel_depth_qose",
    }
    rel_nonlocal_keys = {
        "QOS": "rel_nonlocal_qos",
        "QOSN": "rel_nonlocal_qosn",
        "FrozenQubits": "rel_nonlocal_fq",
        "CutQC": "rel_nonlocal_cutqc",
        "QOSE": "rel_nonlocal_qose",
    }

    summary: Dict[str, float] = {}
    for method in methods:
        depth_key = rel_depth_keys.get(method)
        if depth_key:
            avg = _avg(depth_key)
            if avg is not None:
                summary[depth_key] = avg
        nonlocal_key = rel_nonlocal_keys.get(method)
        if nonlocal_key:
            avg = _avg(nonlocal_key)
            if avg is not None:
                summary[nonlocal_key] = avg

    dist = 0.0
    for method in methods:
        depth_key = rel_depth_keys.get(method)
        if depth_key and depth_key in summary and method in TARGET_REL["depth"]:
            dist += (summary[depth_key] - TARGET_REL["depth"][method]) ** 2
        nonlocal_key = rel_nonlocal_keys.get(method)
        if nonlocal_key and nonlocal_key in summary and method in TARGET_REL["nonlocal"]:
            dist += (summary[nonlocal_key] - TARGET_REL["nonlocal"][method]) ** 2
    summary["target_distance"] = dist ** 0.5
    return summary


def _run_eval(args, benches, sizes, qose_run: Optional[Callable], methods: List[str]):
    all_rows: List[Dict[str, object]] = []
    rel_by_size: Dict[int, Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]] = {}
    timing_rows = []
    fidelity_by_size: Dict[int, Dict[str, Dict[str, float]]] = {}
    real_fidelity_by_size: Dict[int, Dict[str, Dict[str, float]]] = {}
    cut_circuits: Dict[Tuple[int, str, str], List[QuantumCircuit]] = {}
    fragment_fidelity: Dict[Tuple[int, str], Dict[str, object]] = {}
    include_qose = qose_run is not None and "QOSE" in methods
    run_qos = "QOS" in methods
    run_qosn = "QOSN" in methods
    run_fq = "FrozenQubits" in methods
    run_cutqc = "CutQC" in methods

    noise = None
    if args.with_fidelity or args.fragment_fidelity_sweep:
        if args.metrics_baseline == "torino":
            p1 = args.fidelity_p1 or 2.59e-3
            p2 = args.fidelity_p2 or 7.91e-3
            ro = args.fidelity_readout or 3.113e-2
        elif args.metrics_baseline == "marrakesh":
            p1 = args.fidelity_p1 or 2.67e-3
            p2 = args.fidelity_p2 or 5.57e-3
            ro = args.fidelity_readout or 1.172e-2
        else:
            p1 = args.fidelity_p1
            p2 = args.fidelity_p2
            ro = args.fidelity_readout
        noise = _noise_model(p1, p2, ro)

    for size in sizes:
        rel_depth: Dict[str, Dict[str, float]] = {bench: {} for bench, _ in BENCHES}
        rel_nonlocal: Dict[str, Dict[str, float]] = {bench: {} for bench, _ in BENCHES}
        if args.with_fidelity:
            fidelity_by_size[size] = {bench: {} for bench, _ in BENCHES}
        if args.with_real_fidelity:
            real_fidelity_by_size[size] = {bench: {} for bench, _ in BENCHES}

        for bench, _label in benches:
            if args.verbose:
                print(f"size={size} bench={bench}", flush=True)
            bench_start = time.perf_counter()
            qc = _load_qasm_circuit(bench, size)
            if args.verbose:
                print(f"  load_qasm_sec={time.perf_counter() - bench_start:.2f}", flush=True)
            base = _analyze_circuit(
                qc,
                args.metric_mode,
                args.metrics_baseline,
                args.metrics_optimization_level,
            )
            if args.verbose:
                print(
                    f"  baseline depth={base['depth']} cnot={base['num_nonlocal_gates']} "
                    f"sec={time.perf_counter() - bench_start:.2f}",
                    flush=True,
                )
            qos_m = qos_t = qos_circs = None
            qosn_m = qosn_t = qosn_circs = None
            fq_m = fq_t = fq_circs = None
            cutqc_m = cutqc_t = cutqc_circs = None

            if run_qos:
                qos_t0 = time.perf_counter()
                qos_m, qos_t, qos_circs = _run_mitigator(
                    qc, [], args, bench_name=bench, size_label=size
                )
                if args.verbose:
                    print(
                        f"  qos depth={qos_m['depth']} cnot={qos_m['num_nonlocal_gates']} "
                        f"sec={time.perf_counter() - qos_t0:.2f}",
                        flush=True,
                    )
            if run_qosn:
                qosn_t0 = time.perf_counter()
                qosn_m, qosn_t, qosn_circs = _run_mitigator(
                    qc,
                    [],
                    args,
                    use_cost_search_override=False,
                    bench_name=bench,
                    size_label=size,
                )
                if args.verbose:
                    print(
                        f"  qosn depth={qosn_m['depth']} cnot={qosn_m['num_nonlocal_gates']} "
                        f"sec={time.perf_counter() - qosn_t0:.2f}",
                        flush=True,
                    )
            if run_fq:
                fq_t0 = time.perf_counter()
                fq_m, fq_t, fq_circs = _run_mitigator(
                    qc, ["QF"], args, bench_name=bench, size_label=size
                )
                if args.verbose:
                    print(
                        f"  frozen depth={fq_m['depth']} cnot={fq_m['num_nonlocal_gates']} "
                        f"sec={time.perf_counter() - fq_t0:.2f}",
                        flush=True,
                    )

            if run_cutqc:
                cutqc_methods = ["GV"] if args.cutqc_method == "gv" else ["WC"]
                cutqc_t0 = time.perf_counter()
                cutqc_m, cutqc_t, cutqc_circs = _run_mitigator(
                    qc, cutqc_methods, args, bench_name=bench, size_label=size
                )
                if args.verbose:
                    print(
                        f"  cutqc depth={cutqc_m['depth']} cnot={cutqc_m['num_nonlocal_gates']} "
                        f"sec={time.perf_counter() - cutqc_t0:.2f}",
                        flush=True,
                    )

            qose_m = None
            qose_t = {}
            qose_circs = []
            if include_qose:
                qose_t0 = time.perf_counter()
                qose_m, qose_t, qose_circs = _run_qose(qc, args, qose_run)
                if qose_m is None:
                    if run_qos and qos_m is not None:
                        qose_m = qos_m
                        qose_circs = qos_circs
                    elif run_qosn and qosn_m is not None:
                        qose_m = qosn_m
                        qose_circs = qosn_circs
                    else:
                        qose_m = base
                        qose_circs = [qc]
                    qose_t = {}
                if args.verbose:
                    print(
                        f"  qose depth={qose_m['depth']} cnot={qose_m['num_nonlocal_gates']} "
                        f"sec={time.perf_counter() - qose_t0:.2f}",
                        flush=True,
                    )

            if args.fragment_fidelity_sweep and run_qos and qos_circs is not None:
                fragment_fidelity[(size, bench)] = _fragment_fidelity_sweep(
                    qos_circs,
                    args.fragment_fidelity_shots,
                    noise,
                    args.fidelity_seed,
                )

            if run_qos and qos_m is not None:
                rel_depth[bench]["QOS"] = _relative(qos_m["depth"], base["depth"])
                rel_nonlocal[bench]["QOS"] = _relative(
                    qos_m["num_nonlocal_gates"], base["num_nonlocal_gates"]
                )
            if run_fq and fq_m is not None:
                rel_depth[bench]["FrozenQubits"] = _relative(fq_m["depth"], base["depth"])
                rel_nonlocal[bench]["FrozenQubits"] = _relative(
                    fq_m["num_nonlocal_gates"], base["num_nonlocal_gates"]
                )
            if run_cutqc and cutqc_m is not None:
                rel_depth[bench]["CutQC"] = _relative(cutqc_m["depth"], base["depth"])
                rel_nonlocal[bench]["CutQC"] = _relative(
                    cutqc_m["num_nonlocal_gates"], base["num_nonlocal_gates"]
                )
            if run_qosn and qosn_m is not None:
                rel_depth[bench]["QOSN"] = _relative(qosn_m["depth"], base["depth"])
                rel_nonlocal[bench]["QOSN"] = _relative(
                    qosn_m["num_nonlocal_gates"], base["num_nonlocal_gates"]
                )
            if include_qose and qose_m is not None:
                rel_depth[bench]["QOSE"] = _relative(qose_m["depth"], base["depth"])
                rel_nonlocal[bench]["QOSE"] = _relative(
                    qose_m["num_nonlocal_gates"], base["num_nonlocal_gates"]
                )

            if args.with_fidelity:
                if args.verbose:
                    print("  fidelity sim start", flush=True)
                sim_times = {}
                t0 = time.perf_counter()
                base_fidelity = _average_fidelity([qc], args.fidelity_shots, noise, args.fidelity_seed)
                sim_times["Qiskit"] = time.perf_counter() - t0
                fidelity_by_size[size][bench] = {"Qiskit": base_fidelity}
                rel_fidelity = {}

                qos_fidelity = ""
                qosn_fidelity = ""
                fq_fidelity = ""
                cutqc_fidelity = ""
                qose_fidelity = ""

                if run_qos and qos_circs is not None:
                    t0 = time.perf_counter()
                    qos_fidelity = _average_fidelity(
                        qos_circs, args.fidelity_shots, noise, args.fidelity_seed
                    )
                    sim_times["QOS"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["QOS"] = qos_fidelity
                    rel_fidelity["QOS"] = _relative(qos_fidelity, base_fidelity)
                if run_qosn and qosn_circs is not None:
                    t0 = time.perf_counter()
                    qosn_fidelity = _average_fidelity(
                        qosn_circs, args.fidelity_shots, noise, args.fidelity_seed
                    )
                    sim_times["QOSN"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["QOSN"] = qosn_fidelity
                    rel_fidelity["QOSN"] = _relative(qosn_fidelity, base_fidelity)
                if run_fq and fq_circs is not None:
                    t0 = time.perf_counter()
                    fq_fidelity = _average_fidelity(
                        fq_circs, args.fidelity_shots, noise, args.fidelity_seed
                    )
                    sim_times["FrozenQubits"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["FrozenQubits"] = fq_fidelity
                    rel_fidelity["FrozenQubits"] = _relative(fq_fidelity, base_fidelity)
                if run_cutqc and cutqc_circs is not None:
                    t0 = time.perf_counter()
                    cutqc_fidelity = _average_fidelity(
                        cutqc_circs, args.fidelity_shots, noise, args.fidelity_seed
                    )
                    sim_times["CutQC"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["CutQC"] = cutqc_fidelity
                    rel_fidelity["CutQC"] = _relative(cutqc_fidelity, base_fidelity)
                if include_qose and qose_circs is not None:
                    t0 = time.perf_counter()
                    qose_fidelity = _average_fidelity(
                        qose_circs, args.fidelity_shots, noise, args.fidelity_seed
                    )
                    sim_times["QOSE"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["QOSE"] = qose_fidelity
                    rel_fidelity["QOSE"] = _relative(qose_fidelity, base_fidelity)

                if args.verbose:
                    print(f"  fidelity sim sec={sum(sim_times.values()):.2f}", flush=True)

                rel_fidelity_qos = rel_fidelity.get("QOS", "")
                rel_fidelity_qosn = rel_fidelity.get("QOSN", "")
                rel_fidelity_fq = rel_fidelity.get("FrozenQubits", "")
                rel_fidelity_cutqc = rel_fidelity.get("CutQC", "")
                rel_fidelity_qose = rel_fidelity.get("QOSE", "")
            else:
                sim_times = {}
                base_fidelity = ""
                qos_fidelity = ""
                qosn_fidelity = ""
                fq_fidelity = ""
                cutqc_fidelity = ""
                qose_fidelity = ""
                rel_fidelity_qos = ""
                rel_fidelity_qosn = ""
                rel_fidelity_fq = ""
                rel_fidelity_cutqc = ""
                rel_fidelity_qose = ""

            if args.with_real_fidelity:
                real_base = _average_real_fidelity(
                    [qc],
                    args.real_fidelity_shots,
                    args.real_backend,
                    args.fidelity_seed,
                )
                real_fidelity_by_size[size][bench]["Qiskit"] = real_base
                if run_qos and qos_circs is not None:
                    real_fidelity_by_size[size][bench]["QOS"] = _average_real_fidelity(
                        qos_circs,
                        args.real_fidelity_shots,
                        args.real_backend,
                        args.fidelity_seed,
                    )
                if run_qosn and qosn_circs is not None:
                    real_fidelity_by_size[size][bench]["QOSN"] = _average_real_fidelity(
                        qosn_circs,
                        args.real_fidelity_shots,
                        args.real_backend,
                        args.fidelity_seed,
                    )
                if run_fq and fq_circs is not None:
                    real_fidelity_by_size[size][bench]["FrozenQubits"] = _average_real_fidelity(
                        fq_circs,
                        args.real_fidelity_shots,
                        args.real_backend,
                        args.fidelity_seed,
                    )
                if run_cutqc and cutqc_circs is not None:
                    real_fidelity_by_size[size][bench]["CutQC"] = _average_real_fidelity(
                        cutqc_circs,
                        args.real_fidelity_shots,
                        args.real_backend,
                        args.fidelity_seed,
                    )
                if include_qose and qose_circs is not None:
                    real_fidelity_by_size[size][bench]["QOSE"] = _average_real_fidelity(
                        qose_circs,
                        args.real_fidelity_shots,
                        args.real_backend,
                        args.fidelity_seed,
                    )

            qiskit_sim = float(sim_times.get("Qiskit", 0.0))

            row = {
                "bench": bench,
                "size": size,
                "baseline_depth": base["depth"],
                "baseline_nonlocal": base["num_nonlocal_gates"],
                "baseline_fidelity": base_fidelity,
                "baseline_sim_time": qiskit_sim,
            }

            if run_qos and qos_m is not None:
                qos_sim = float(sim_times.get("QOS", 0.0))
                qos_num_circuits = max(1, len(qos_circs))
                row.update(
                    {
                        "qos_depth": qos_m["depth"],
                        "qos_nonlocal": qos_m["num_nonlocal_gates"],
                        "qos_fidelity": qos_fidelity,
                        "qos_sim_time": qos_sim,
                        "qos_num_circuits": qos_num_circuits,
                        "qos_classical_overhead": _safe_ratio(qos_sim, qiskit_sim),
                        "qos_quantum_overhead": float(qos_num_circuits),
                        "rel_depth_qos": rel_depth[bench]["QOS"],
                        "rel_nonlocal_qos": rel_nonlocal[bench]["QOS"],
                        "rel_fidelity_qos": rel_fidelity_qos,
                    }
                )
            if run_qosn and qosn_m is not None:
                qosn_sim = float(sim_times.get("QOSN", 0.0))
                qosn_num_circuits = max(1, len(qosn_circs))
                row.update(
                    {
                        "qosn_depth": qosn_m["depth"],
                        "qosn_nonlocal": qosn_m["num_nonlocal_gates"],
                        "qosn_fidelity": qosn_fidelity,
                        "qosn_sim_time": qosn_sim,
                        "qosn_num_circuits": qosn_num_circuits,
                        "qosn_classical_overhead": _safe_ratio(qosn_sim, qiskit_sim),
                        "qosn_quantum_overhead": float(qosn_num_circuits),
                        "rel_depth_qosn": rel_depth[bench]["QOSN"],
                        "rel_nonlocal_qosn": rel_nonlocal[bench]["QOSN"],
                        "rel_fidelity_qosn": rel_fidelity_qosn,
                    }
                )
            if run_fq and fq_m is not None:
                fq_sim = float(sim_times.get("FrozenQubits", 0.0))
                fq_num_circuits = max(1, len(fq_circs))
                row.update(
                    {
                        "fq_depth": fq_m["depth"],
                        "fq_nonlocal": fq_m["num_nonlocal_gates"],
                        "fq_fidelity": fq_fidelity,
                        "fq_sim_time": fq_sim,
                        "fq_num_circuits": fq_num_circuits,
                        "fq_classical_overhead": _safe_ratio(fq_sim, qiskit_sim),
                        "fq_quantum_overhead": float(fq_num_circuits),
                        "rel_depth_fq": rel_depth[bench]["FrozenQubits"],
                        "rel_nonlocal_fq": rel_nonlocal[bench]["FrozenQubits"],
                        "rel_fidelity_fq": rel_fidelity_fq,
                    }
                )
            if run_cutqc and cutqc_m is not None:
                cutqc_sim = float(sim_times.get("CutQC", 0.0))
                cutqc_num_circuits = max(1, len(cutqc_circs))
                row.update(
                    {
                        "cutqc_depth": cutqc_m["depth"],
                        "cutqc_nonlocal": cutqc_m["num_nonlocal_gates"],
                        "cutqc_fidelity": cutqc_fidelity,
                        "cutqc_sim_time": cutqc_sim,
                        "cutqc_num_circuits": cutqc_num_circuits,
                        "cutqc_classical_overhead": _safe_ratio(cutqc_sim, qiskit_sim),
                        "cutqc_quantum_overhead": float(cutqc_num_circuits),
                        "rel_depth_cutqc": rel_depth[bench]["CutQC"],
                        "rel_nonlocal_cutqc": rel_nonlocal[bench]["CutQC"],
                        "rel_fidelity_cutqc": rel_fidelity_cutqc,
                    }
                )
            if include_qose and qose_m is not None:
                qose_sim = float(sim_times.get("QOSE", 0.0))
                qose_num_circuits = max(1, len(qose_circs))
                row.update(
                    {
                        "qose_depth": qose_m["depth"],
                        "qose_nonlocal": qose_m["num_nonlocal_gates"],
                        "qose_fidelity": qose_fidelity,
                        "qose_sim_time": qose_sim,
                        "qose_num_circuits": qose_num_circuits,
                        "qose_classical_overhead": _safe_ratio(qose_sim, qiskit_sim),
                        "qose_quantum_overhead": float(qose_num_circuits),
                        "rel_depth_qose": rel_depth[bench]["QOSE"],
                        "rel_nonlocal_qose": rel_nonlocal[bench]["QOSE"],
                        "rel_fidelity_qose": rel_fidelity_qose,
                    }
                )
            all_rows.append(row)

            if args.collect_timing:
                timing_methods = []
                if run_qos and qos_t is not None:
                    timing_methods.append(("QOS", qos_t))
                if run_qosn and qosn_t is not None:
                    timing_methods.append(("QOSN", qosn_t))
                if run_fq and fq_t is not None:
                    timing_methods.append(("FrozenQubits", fq_t))
                if run_cutqc and cutqc_t is not None:
                    timing_methods.append(("CutQC", cutqc_t))
                if include_qose and qose_t is not None:
                    timing_methods.append(("QOSE", qose_t))
                for method, timing in timing_methods:
                    row = {"bench": bench, "size": size, "method": method}
                    row.update(timing)
                    if args.with_fidelity:
                        row["simulation"] = sim_times.get(method, 0.0)
                    timing_rows.append(row)
            if args.cut_visualization:
                cut_circuits[(size, bench, "Qiskit")] = [qc]
                if run_qos and qos_circs is not None:
                    cut_circuits[(size, bench, "QOS")] = qos_circs
                if run_qosn and qosn_circs is not None:
                    cut_circuits[(size, bench, "QOSN")] = qosn_circs
                if include_qose and qose_circs is not None:
                    cut_circuits[(size, bench, "QOSE")] = qose_circs
                if run_fq and fq_circs is not None:
                    cut_circuits[(size, bench, "FrozenQubits")] = fq_circs
                if run_cutqc and cutqc_circs is not None:
                    cut_circuits[(size, bench, "CutQC")] = cutqc_circs
            if args.verbose:
                print(f"  total_bench_sec={time.perf_counter() - bench_start:.2f}", flush=True)

        rel_by_size[size] = (rel_depth, rel_nonlocal)

    return (
        all_rows,
        rel_by_size,
        timing_rows,
        fidelity_by_size,
        real_fidelity_by_size,
        cut_circuits,
        fragment_fidelity,
    )


def _load_qose_run_from_args(args) -> Tuple[Optional[Callable], Optional[Path]]:
    qose_program = _find_qose_program(getattr(args, "qose_program", ""))
    if not qose_program:
        return None, None
    try:
        return _load_qose_run(qose_program), qose_program
    except Exception as exc:
        if getattr(args, "verbose", False):
            print(f"Skipping QOSE program {qose_program}: {exc}", file=sys.stderr)
        return None, None


def _plot_cut_visualization(
    cut_circuits: Dict[Tuple[int, str, str], List[QuantumCircuit]],
    benches: List[Tuple[str, str]],
    sizes: List[int],
    out_dir: Path,
    timestamp: str,
    methods: List[str],
) -> Path:
    plt = _import_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
    from qiskit.visualization import circuit_drawer  # type: ignore

    out_path = out_dir / f"cut_visualization_{timestamp}.pdf"
    methods = list(methods)

    with PdfPages(out_path) as pdf:
        for size in sizes:
            for bench, label in benches:
                for method in methods:
                    circuits = cut_circuits.get((size, bench, method), [])
                    if not circuits:
                        continue
                    for idx, circuit in enumerate(circuits):
                        circuit = _normalize_circuit(circuit)
                        title = f"{label} ({size}q) - {method}"
                        if len(circuits) > 1:
                            title = f"{title} [{idx + 1}/{len(circuits)}]"
                        fig = circuit_drawer(circuit, output="mpl", fold=30, idle_wires=False)
                        fig.suptitle(title, fontsize=10)
                        pdf.savefig(fig)
                        plt.close(fig)

    return out_path


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except ModuleNotFoundError:
        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            sys.path.append(user_site)
        import matplotlib.pyplot as plt  # type: ignore
        return plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare QOS, QOSN, FrozenQubits, CutQC, and optional QOSE vs Qiskit baseline."
    )
    parser.add_argument("--sizes", default="12,24", help="Comma-separated qubit sizes.")
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--size-to-reach", type=int, default=0)
    parser.add_argument("--ideal-size-to-reach", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--clingo-timeout-sec", type=int, default=0, help="0 disables clingo timeout.")
    parser.add_argument("--max-partition-tries", type=int, default=0, help="0 disables partition-try limit.")
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
    parser.add_argument("--qos-cost-search-max-iters", type=int, default=0, help="0 disables cost-search iteration limit.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", default="", help="Optional tag to include in output filenames.")
    parser.add_argument("--collect-timing", action="store_true")
    parser.add_argument("--timing-csv", default="", help="Optional CSV path for timing breakdown.")
    parser.add_argument("--timing-plot", action="store_true")
    parser.add_argument("--overhead-plot", action="store_true")
    parser.add_argument("--with-fidelity", action="store_true")
    parser.add_argument("--fidelity-shots", type=int, default=200)
    parser.add_argument("--fidelity-seed", type=int, default=7)
    parser.add_argument("--fidelity-p1", type=float, default=0.0, help="1-qubit depolarizing error.")
    parser.add_argument("--fidelity-p2", type=float, default=0.0, help="2-qubit depolarizing error.")
    parser.add_argument("--fidelity-readout", type=float, default=0.0, help="Readout error.")
    parser.add_argument("--with-real-fidelity", action="store_true")
    parser.add_argument("--real-fidelity-shots", type=int, default=1000)
    parser.add_argument("--real-backend", default="ibm_torino")
    parser.add_argument("--fragment-fidelity-sweep", action="store_true")
    parser.add_argument(
        "--fragment-fidelity-shots",
        default="100,200,400,600,800,1000",
        help="Comma-separated shots list for fragment fidelity sweep.",
    )
    parser.add_argument("--cut-visualization", action="store_true")
    parser.add_argument(
        "--cutqc-method",
        choices=["gv", "wc"],
        default="gv",
        help="CutQC baseline method: gv (gate cutting) or wc (wire cutting).",
    )
    parser.add_argument(
        "--metrics-baseline",
        choices=["raw", "kolkata", "torino", "marrakesh"],
        default="torino",
        help="Baseline for metrics: raw circuit or transpiled to a fake backend.",
    )
    parser.add_argument("--metrics-optimization-level", type=int, default=3)
    parser.add_argument(
        "--metric-mode",
        choices=["virtual", "fragment", "cutqc"],
        default="fragment",
        help="Metric mode: virtual (paper-style), fragment (sum/max), or cutqc (ignore cut ops).",
    )
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-budgets", default="3", help="Comma-separated budgets.")
    parser.add_argument("--sweep-sizes-to-reach", default="6,7,8,9", help="Comma-separated size-to-reach values.")
    parser.add_argument("--sweep-clingo", default="5,10,20", help="Comma-separated clingo timeouts.")
    parser.add_argument("--sweep-max-tries", default="3,5,8", help="Comma-separated max partition tries.")
    parser.add_argument("--sweep-cost-iters", default="2,3,4", help="Comma-separated cost-search max iters.")
    parser.add_argument(
        "--benches",
        default="all",
        help="Comma-separated benchmark names or 'all'.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated methods to run: FrozenQubits,CutQC,QOS,QOSN,QOSE "
            "(or 'all' for defaults)."
        ),
    )
    parser.add_argument(
        "--cost-search-log",
        default="",
        help="Optional path to write cost-search trace CSV (default: auto in out-dir).",
    )
    parser.add_argument(
        "--qose-program",
        default="",
        help=(
            "Optional path to evolved QOSE program (evolved_run). "
            "If unset, auto-detect qose_program.py or openevolve_output/best_program.py."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "evaluation" / "plots"),
        help="Output directory for figures and CSV.",
    )
    args = parser.parse_args()
    os.environ["QVM_CLINGO_TIMEOUT_SEC"] = str(args.clingo_timeout_sec)
    os.environ["QVM_MAX_PARTITION_TRIES"] = str(args.max_partition_tries)
    os.environ["QOS_COST_SEARCH_MAX_ITERS"] = str(args.qos_cost_search_max_iters)

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    args.fragment_fidelity_shots = [
        int(s.strip())
        for s in str(args.fragment_fidelity_shots).split(",")
        if s.strip()
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag.strip()
    tag_suffix = f"_{tag}" if tag else ""
    if not args.cost_search_log:
        args.cost_search_log = str(
            out_dir / f"cost_search_trace_{timestamp}{tag_suffix}.csv"
        )

    all_rows: List[Dict[str, object]] = []
    rel_by_size: Dict[int, Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]] = {}

    if args.benches == "all":
        benches = BENCHES
    else:
        selected = {b.strip() for b in args.benches.split(",") if b.strip()}
        benches = [(b, label) for b, label in BENCHES if b in selected]

    qose_run, qose_program = _load_qose_run_from_args(args)
    selected_methods = _parse_methods(args.methods, qose_run is not None)
    if "QOSE" in selected_methods and qose_run is None:
        print("Requested QOSE but no QOSE program was found; skipping QOSE.", file=sys.stderr)
        selected_methods = [m for m in selected_methods if m != "QOSE"]
    include_qose = qose_run is not None and "QOSE" in selected_methods
    if args.verbose and qose_program and include_qose:
        print(f"Using QOSE program: {qose_program}", file=sys.stderr)

    if args.sweep:
        sweep_rows = []
        best = None
        budgets = [int(x) for x in args.sweep_budgets.split(",") if x.strip()]
        size_to_reach_vals = [int(x) for x in args.sweep_sizes_to_reach.split(",") if x.strip()]
        clingo_vals = [int(x) for x in args.sweep_clingo.split(",") if x.strip()]
        max_tries_vals = [int(x) for x in args.sweep_max_tries.split(",") if x.strip()]
        cost_iters = [int(x) for x in args.sweep_cost_iters.split(",") if x.strip()]

        for budget in budgets:
            for size_to_reach in size_to_reach_vals:
                for clingo in clingo_vals:
                    for max_tries in max_tries_vals:
                        for cost_iter in cost_iters:
                            args.budget = budget
                            args.size_to_reach = size_to_reach
                            args.clingo_timeout_sec = clingo
                            args.max_partition_tries = max_tries
                            args.qos_cost_search = True
                            args.qos_cost_search_max_iters = cost_iter
                            os.environ["QVM_CLINGO_TIMEOUT_SEC"] = str(args.clingo_timeout_sec)
                            os.environ["QVM_MAX_PARTITION_TRIES"] = str(args.max_partition_tries)
                            os.environ["QOS_COST_SEARCH_MAX_ITERS"] = str(args.qos_cost_search_max_iters)

                            rows, _rel, _timing, _fid, _cuts, _frag = _run_eval(
                                args, benches, sizes, qose_run, selected_methods
                            )
                            summary = _summarize(rows, selected_methods)
                            sweep_rows.append(
                                {
                                    "budget": budget,
                                    "size_to_reach": size_to_reach,
                                    "clingo_timeout_sec": clingo,
                                    "max_partition_tries": max_tries,
                                    "cost_search_max_iters": cost_iter,
                                    **summary,
                                }
                            )
                            if best is None or summary["target_distance"] < best[0]:
                                best = (summary["target_distance"], budget, size_to_reach, clingo, max_tries, cost_iter)

        sweep_path = out_dir / f"relative_properties_sweep_{timestamp}{tag_suffix}.csv"
        with sweep_path.open("w", newline="") as f:
            fieldnames = sorted({k for row in sweep_rows for k in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sweep_rows)

        if best is None:
            raise RuntimeError("No sweep results produced.")

        _, best_budget, best_size, best_clingo, best_max_tries, best_cost_iter = best
        args.budget = best_budget
        args.size_to_reach = best_size
        args.clingo_timeout_sec = best_clingo
        args.max_partition_tries = best_max_tries
        args.qos_cost_search = True
        args.qos_cost_search_max_iters = best_cost_iter
        os.environ["QVM_CLINGO_TIMEOUT_SEC"] = str(args.clingo_timeout_sec)
        os.environ["QVM_MAX_PARTITION_TRIES"] = str(args.max_partition_tries)
        os.environ["QOS_COST_SEARCH_MAX_ITERS"] = str(args.qos_cost_search_max_iters)

        (
            all_rows,
            rel_by_size,
            timing_rows,
            fidelity_by_size,
            real_fidelity_by_size,
            cut_circuits,
            fragment_fidelity,
        ) = _run_eval(args, benches, sizes, qose_run, selected_methods)
        print(f"Wrote sweep: {sweep_path}")
        print(
            "Best config:"
            f" budget={best_budget} size_to_reach={best_size}"
            f" clingo={best_clingo} max_tries={best_max_tries}"
            f" cost_iters={best_cost_iter}"
        )
    else:
        (
            all_rows,
            rel_by_size,
            timing_rows,
            fidelity_by_size,
            real_fidelity_by_size,
            cut_circuits,
            fragment_fidelity,
        ) = _run_eval(args, benches, sizes, qose_run, selected_methods)

    fidelity_methods = ["Qiskit"] + [m for m in selected_methods if m != "Qiskit"]
    combined_path = _plot_combined(
        rel_by_size,
        benches,
        out_dir,
        f"{timestamp}{tag_suffix}",
        fidelity_by_size if args.with_fidelity else None,
        real_fidelity_by_size if args.with_real_fidelity else None,
        methods=selected_methods,
        fidelity_methods=fidelity_methods,
    )

    print(f"Wrote figure: {combined_path}")
    if args.cut_visualization and cut_circuits:
        cut_methods = ["Qiskit"] + [m for m in selected_methods if m != "Qiskit"]
        cut_fig = _plot_cut_visualization(
            cut_circuits,
            benches,
            sizes,
            out_dir,
            f"{timestamp}{tag_suffix}",
            cut_methods,
        )
        print(f"Wrote cut visualization: {cut_fig}")
    if args.collect_timing and timing_rows and args.timing_csv:
        timing_path = Path(args.timing_csv)
        with timing_path.open("w", newline="") as f:
            fieldnames = sorted({k for row in timing_rows for k in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(timing_rows)
        print(f"Wrote timing: {timing_path}")
    if args.collect_timing and timing_rows and args.timing_plot:
        timing_fig = _plot_timing(timing_rows, out_dir, f"{timestamp}{tag_suffix}", selected_methods)
        print(f"Wrote timing figure: {timing_fig}")
    if args.overhead_plot:
        if not args.with_fidelity:
            raise RuntimeError("Overhead plot requires --with-fidelity to measure simulation time.")
        overhead_fig = _plot_overheads(
            all_rows, benches, sizes, out_dir, f"{timestamp}{tag_suffix}", selected_methods
        )
        print(f"Wrote overhead figure: {overhead_fig}")
    if args.fragment_fidelity_sweep:
        frag_paths = _plot_fragment_fidelity_sweep(
            fragment_fidelity, benches, sizes, out_dir, f"{timestamp}{tag_suffix}"
        )
        for path in frag_paths:
            print(f"Wrote fragment fidelity figure: {path}")
    _cleanup_children()


if __name__ == "__main__":
    main()
