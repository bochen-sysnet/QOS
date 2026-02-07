import argparse
import csv
import datetime as dt
import importlib.util
import json
import pickle
import shlex
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
_REAL_DRY_RUN = False
_REAL_DRY_RUN_CALLS = 0
_RESUME_STATE_VERSION = 1
_REAL_JOB_RESULT_CACHE = None


class RealQPUWaitTimeout(RuntimeError):
    pass


class _RealJobResultCache:
    def __init__(self, path: Path):
        self.path = path
        self._results: Dict[str, Dict[str, object]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = str(rec.get("job_key", ""))
                    if not key:
                        continue
                    counts = rec.get("counts")
                    if isinstance(counts, dict):
                        self._results[key] = counts
        except Exception:
            self._results = {}

    def count(self) -> int:
        return len(self._results)

    def get(self, job_key: str) -> Optional[Dict[str, int]]:
        counts = self._results.get(job_key)
        if counts is None:
            return None
        try:
            return {str(k): int(v) for k, v in counts.items()}
        except Exception:
            return None

    def put(self, job_key: str, counts: Dict[str, int], meta: Dict[str, object]) -> None:
        self._results[job_key] = {str(k): int(v) for k, v in counts.items()}
        rec = {"job_key": job_key, "counts": self._results[job_key], **meta}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                f.flush()
        except Exception:
            pass


def _dry_run_submit(ctx: Dict[str, int], method_ctx: Dict[str, int | str]) -> None:
    global _REAL_DRY_RUN_CALLS
    _REAL_DRY_RUN_CALLS += 1
    method_ctx["submitted"] = int(method_ctx.get("submitted", 0)) + 1
    method_ctx["completed"] = int(method_ctx.get("completed", 0)) + 1


def _dry_run_traverse(
    circuit: QuantumCircuit, ctx: Dict[str, int], method_ctx: Dict[str, int | str]
) -> None:
    vc = circuit if isinstance(circuit, VirtualCircuit) else None
    if vc is None and not _has_virtual_ops(circuit):
        _dry_run_submit(ctx, method_ctx)
        return
    if vc is None:
        vc = VirtualCircuit(_normalize_circuit(circuit))
    frag_total = len(vc.fragment_circuits)
    for frag_idx, (frag, frag_circ) in enumerate(vc.fragment_circuits.items(), start=1):
        inst_labels = vc.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circ, inst_labels)
        inst_total = len(instantiations)
        for inst_idx in range(1, inst_total + 1):
            inst_ctx = {
                **ctx,
                "fragment_idx": frag_idx,
                "fragment_total": frag_total,
                "inst_idx": inst_idx,
                "inst_total": inst_total,
            }
            _dry_run_submit(inst_ctx, method_ctx)


def _is_eval_verbose() -> bool:
    raw = os.getenv("QOS_EVAL_VERBOSE", os.getenv("QOS_VERBOSE", ""))
    return raw.lower() in {"1", "true", "yes", "y"}


def _bench_key(size: int, bench: str) -> str:
    return f"{int(size)}::{bench}"


def _safe_float(val, default=0.0):
    try:
        if val == "":
            return default
        return float(val)
    except Exception:
        return default


def _safe_int(val, default=0):
    try:
        if val == "":
            return default
        return int(val)
    except Exception:
        return default


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resume_signature(args, benches, sizes, methods: List[str]) -> Dict[str, object]:
    return {
        "sizes": [int(s) for s in sizes],
        "benches": [b for b, _ in benches],
        "methods": list(methods),
        "with_fidelity": bool(args.with_fidelity),
        "with_real_fidelity": bool(args.with_real_fidelity),
        "real_backend": str(args.real_backend),
        "real_fidelity_shots": int(args.real_fidelity_shots),
        "metric_mode": str(args.metric_mode),
        "metrics_baseline": str(args.metrics_baseline),
        "metrics_optimization_level": int(args.metrics_optimization_level),
        "cutqc_method": str(args.cutqc_method),
        "budget": int(args.budget),
        "size_to_reach": int(args.size_to_reach),
        "ideal_size_to_reach": int(args.ideal_size_to_reach),
    }


def _load_resume_state(path: Path, signature: Dict[str, object]) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if int(data.get("version", 0)) != _RESUME_STATE_VERSION:
        return None
    if data.get("signature") != signature:
        return None
    return data


def _save_resume_state(
    path: Path,
    signature: Dict[str, object],
    completed_keys: set[str],
    rows: List[Dict[str, object]],
    timing_rows: List[Dict[str, object]],
    status: str,
    note: str = "",
) -> None:
    payload: Dict[str, object] = {
        "version": _RESUME_STATE_VERSION,
        "signature": signature,
        "completed_keys": sorted(completed_keys),
        "rows": rows,
        "timing_rows": timing_rows,
        "status": status,
        "note": note,
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write_json(path, payload)


def _parse_real_job_timeout_sec() -> int:
    raw = os.getenv("QOS_REAL_JOB_TIMEOUT_SEC", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except Exception:
        return 0


def _safe_token(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _method_cache_paths(cache_dir: Optional[Path], size: int, bench: str, method: str) -> Tuple[Optional[Path], Optional[Path]]:
    if cache_dir is None:
        return None, None
    stem = f"s{int(size)}_{_safe_token(bench)}_{_safe_token(method)}"
    return cache_dir / f"{stem}.json", cache_dir / f"{stem}.pkl"


def _save_method_cache(
    cache_dir: Optional[Path],
    size: int,
    bench: str,
    method: str,
    metrics: Dict[str, int],
    circuits: List[QuantumCircuit],
) -> None:
    metrics_path, circuits_path = _method_cache_paths(cache_dir, size, bench, method)
    if metrics_path is None or circuits_path is None:
        return
    try:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "depth": int(metrics.get("depth", 0)),
            "num_nonlocal_gates": int(metrics.get("num_nonlocal_gates", 0)),
        }
        _atomic_write_json(metrics_path, payload)
        tmp = circuits_path.with_suffix(circuits_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(circuits, f)
        tmp.replace(circuits_path)
    except Exception:
        return


def _load_method_cache(
    cache_dir: Optional[Path],
    size: int,
    bench: str,
    method: str,
) -> Optional[Tuple[Dict[str, int], List[QuantumCircuit]]]:
    metrics_path, circuits_path = _method_cache_paths(cache_dir, size, bench, method)
    if metrics_path is None or circuits_path is None:
        return None
    if not metrics_path.exists() or not circuits_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text())
        with circuits_path.open("rb") as f:
            circuits = pickle.load(f)
        if not isinstance(circuits, list):
            return None
        metrics = {
            "depth": int(data.get("depth", 0)),
            "num_nonlocal_gates": int(data.get("num_nonlocal_gates", 0)),
        }
        return metrics, circuits
    except Exception:
        return None

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

METHOD_ORDER = ["FrozenQubits", "CutQC", "QOS", "QOSN", "QOSE", "qwen", "gemini", "gpt"]
METHOD_ALIASES = {
    "frozenqubit": "FrozenQubits",
    "frozenqubits": "FrozenQubits",
    "fq": "FrozenQubits",
    "cutqc": "CutQC",
    "qos": "QOS",
    "qosn": "QOSN",
    "qose": "QOSE",
    "qwen": "qwen",
    "gem": "gemini",
    "gemini": "gemini",
    "gpt": "gpt",
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


def _model_tokens(model: str) -> List[str]:
    model = model.lower().strip()
    if model == "gemini":
        return ["gemini", "gem"]
    if model == "gpt":
        return ["gpt"]
    if model == "qwen":
        return ["qwen"]
    return [model]


def _find_best_qose_programs(models: List[str], root: Path) -> Dict[str, Path]:
    best: Dict[str, Tuple[float, Path]] = {}
    for info_path in root.rglob("best_program_info.json"):
        try:
            data = json.loads(info_path.read_text())
        except Exception:
            continue
        metrics = data.get("metrics") or {}
        score = metrics.get("combined_score")
        if score is None:
            continue
        lower_path = str(info_path).lower()
        for model in models:
            tokens = _model_tokens(model)
            if not any(token in lower_path for token in tokens):
                continue
            current = best.get(model)
            if current is None or score > current[0]:
                best_program = info_path.parent / "best_program.py"
                if best_program.exists():
                    best[model] = (float(score), best_program)
            break
    return {model: path for model, (score, path) in best.items()}


def _normalize_qose_method(name: str) -> str:
    lower = name.strip().lower()
    if lower in {"qose", "qosee"}:
        return "QOSE"
    if lower in {"gem", "gemini"}:
        return "gemini"
    if lower == "gpt":
        return "gpt"
    if lower == "qwen":
        return "qwen"
    return name.strip()


def _resolve_best_program_path(root: Path, hint: str) -> Optional[Path]:
    candidate = Path(hint)
    if candidate.is_absolute() and candidate.exists():
        if candidate.is_file():
            return candidate
        for sub in ("best/best_program.py", "best_program.py"):
            path = candidate / sub
            if path.exists():
                return path
        return None
    if candidate.exists():
        if candidate.is_file():
            return candidate.resolve()
        for sub in ("best/best_program.py", "best_program.py"):
            path = candidate / sub
            if path.exists():
                return path.resolve()
        return None
    # Treat hint as a folder under the root.
    for sub in ("best/best_program.py", "best_program.py"):
        path = root / hint / sub
        if path.exists():
            return path.resolve()
    return None


def _parse_manual_qose_mapping(raw: str, root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        method = _normalize_qose_method(key)
        value = value.strip()
        if not value:
            continue
        program = _resolve_best_program_path(root, value)
        if program:
            mapping[method] = program
    return mapping


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


def _relative_methods(include_qose: bool, qose_methods: Optional[List[str]] = None) -> List[str]:
    methods = ["FrozenQubits", "CutQC", "QOS", "QOSN"]
    if include_qose:
        if qose_methods:
            methods.extend(qose_methods)
        else:
            methods.append("QOSE")
    return methods


def _fidelity_methods(include_qose: bool, qose_methods: Optional[List[str]] = None) -> List[str]:
    methods = ["Qiskit", "FrozenQubits", "CutQC", "QOS", "QOSN"]
    if include_qose:
        if qose_methods:
            methods.extend(qose_methods)
        else:
            methods.append("QOSE")
    return methods


def _timing_methods(include_qose: bool, qose_methods: Optional[List[str]] = None) -> List[str]:
    methods = ["FrozenQubits", "CutQC", "QOS", "QOSN"]
    if include_qose:
        if qose_methods:
            methods.extend(qose_methods)
        else:
            methods.append("QOSE")
    return methods


def _cut_methods(include_qose: bool, qose_methods: Optional[List[str]] = None) -> List[str]:
    methods = ["Qiskit", "QOS", "QOSN"]
    if include_qose:
        if qose_methods:
            methods.extend(qose_methods)
        else:
            methods.append("QOSE")
    methods.extend(["FrozenQubits", "CutQC"])
    return methods


def _parse_methods(value: str, include_qose: bool, qose_methods: Optional[List[str]] = None) -> List[str]:
    raw = value.strip()
    if not raw or raw.lower() in {"all", "default"}:
        return _relative_methods(include_qose, qose_methods)

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
    method_order = list(METHOD_ORDER)
    if qose_methods:
        for method in qose_methods:
            if method not in method_order:
                method_order.append(method)
    for method in method_order:
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


def _call_with_alarm_timeout(fn: Callable[[], object], timeout_sec: int, timeout_msg: str):
    if timeout_sec <= 0:
        return fn()
    if not hasattr(signal, "SIGALRM"):
        return fn()

    def _handler(signum, frame):
        raise TimeoutError(timeout_msg)

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
        return fn()
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass


def _job_quantum_seconds(job) -> Optional[float]:
    # Runtime API varies by version; prefer usage_estimation then metadata usage.
    try:
        usage_est = getattr(job, "usage_estimation", None)
        if isinstance(usage_est, dict):
            qsec = usage_est.get("quantum_seconds", None)
            if qsec is not None:
                return float(qsec)
    except Exception:
        pass
    try:
        metrics_fn = getattr(job, "metrics", None)
        if callable(metrics_fn):
            metrics = metrics_fn() or {}
            usage = metrics.get("usage", {}) if isinstance(metrics, dict) else {}
            if isinstance(usage, dict):
                for key in ("quantum_seconds", "seconds"):
                    if usage.get(key) is not None:
                        return float(usage[key])
    except Exception:
        pass
    return None


def _real_counts(
    circuit: QuantumCircuit,
    shots: int,
    backend_name: str,
    ctx: Dict[str, int],
    method_ctx: Dict[str, int | str],
) -> Dict[str, int]:
    global _REAL_JOB_RESULT_CACHE
    job_key = (
        f"{backend_name}|{shots}|{method_ctx.get('method')}|{method_ctx.get('bench')}|"
        f"{method_ctx.get('size')}|c{ctx.get('circuit_idx',1)}|f{ctx.get('fragment_idx',1)}|i{ctx.get('inst_idx',1)}"
    )
    if _REAL_JOB_RESULT_CACHE is not None:
        cached_counts = _REAL_JOB_RESULT_CACHE.get(job_key)
        if cached_counts is not None:
            method_ctx["reused"] = int(method_ctx.get("reused", 0)) + 1
            method_ctx["completed"] = int(method_ctx.get("completed", 0)) + 1
            _EVAL_LOGGER.warning(
                "Real QPU job cache-hit method=%s bench=%s size=%s completed=%s/%s",
                method_ctx.get("method"),
                method_ctx.get("bench"),
                method_ctx.get("size"),
                method_ctx.get("completed"),
                method_ctx.get("expected_total"),
            )
            return cached_counts

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
        job_id = None
        try:
            job_id = job.job_id()
        except Exception:
            pass
        method_ctx["submitted"] = int(method_ctx.get("submitted", 0)) + 1
        _EVAL_LOGGER.warning(
            "Real QPU job submitted method=%s bench=%s size=%s circuit=%s/%s fragment=%s/%s inst=%s/%s job_id=%s submitted=%s/%s",
            method_ctx.get("method"),
            method_ctx.get("bench"),
            method_ctx.get("size"),
            ctx.get("circuit_idx", 1),
            ctx.get("circuit_total", 1),
            ctx.get("fragment_idx", 1),
            ctx.get("fragment_total", 1),
            ctx.get("inst_idx", 1),
            ctx.get("inst_total", 1),
            job_id,
            method_ctx.get("submitted"),
            method_ctx.get("expected_total"),
        )
        wait_start = time.perf_counter()
        timeout_sec = _parse_real_job_timeout_sec()
        timeout_msg = (
            f"Real QPU job wait timeout after {timeout_sec}s "
            f"(method={method_ctx.get('method')} bench={method_ctx.get('bench')} "
            f"size={method_ctx.get('size')} job_id={job_id})"
        )
        result = _call_with_alarm_timeout(job.result, timeout_sec, timeout_msg)
        elapsed = time.perf_counter() - wait_start
        method_ctx["completed"] = int(method_ctx.get("completed", 0)) + 1
        qpu_sec = _job_quantum_seconds(job)
        _EVAL_LOGGER.warning(
            "Real QPU job finished method=%s bench=%s size=%s job_id=%s elapsed_sec=%.2f qpu_sec=%s completed=%s/%s",
            method_ctx.get("method"),
            method_ctx.get("bench"),
            method_ctx.get("size"),
            job_id,
            elapsed,
            f"{qpu_sec:.2f}" if qpu_sec is not None else "n/a",
            method_ctx.get("completed"),
            method_ctx.get("expected_total"),
        )
        counts = _sampler_result_to_counts(result, shots, tcirc.num_clbits)
        if _REAL_JOB_RESULT_CACHE is not None:
            _REAL_JOB_RESULT_CACHE.put(
                job_key,
                counts,
                {
                    "job_id": job_id,
                    "backend": backend_name,
                    "shots": shots,
                    "method": method_ctx.get("method"),
                    "bench": method_ctx.get("bench"),
                    "size": method_ctx.get("size"),
                    "circuit_idx": ctx.get("circuit_idx", 1),
                    "fragment_idx": ctx.get("fragment_idx", 1),
                    "inst_idx": ctx.get("inst_idx", 1),
                    "qpu_sec": qpu_sec,
                    "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
                },
            )
        return counts
    except TimeoutError as exc:
        try:
            job.cancel()
        except Exception:
            pass
        raise RealQPUWaitTimeout(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(
            "Real-backend execution failed. "
            "Ensure your plan supports job execution with primitives."
        ) from exc


def _real_virtual_counts(
    circuit: QuantumCircuit | VirtualCircuit,
    shots: int,
    backend_name: str,
    ctx: Dict[str, int],
    method_ctx: Dict[str, int | str],
) -> Dict[str, int]:
    vc = circuit if isinstance(circuit, VirtualCircuit) else VirtualCircuit(circuit)
    results: Dict = {}
    frag_total = len(vc.fragment_circuits)
    for frag_idx, (frag, frag_circ) in enumerate(vc.fragment_circuits.items(), start=1):
        inst_labels = vc.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circ, inst_labels)
        distrs = []
        inst_total = len(instantiations)
        for inst_idx, inst in enumerate(instantiations, start=1):
            inst_ctx = {
                **ctx,
                "fragment_idx": frag_idx,
                "fragment_total": frag_total,
                "inst_idx": inst_idx,
                "inst_total": inst_total,
            }
            counts = _real_counts(inst, shots, backend_name, inst_ctx, method_ctx)
            distrs.append(QuasiDistr.from_counts(counts))
        results[frag] = distrs
    quasi = vc.knit(results, SerialPool())
    return quasi.to_counts(vc._circuit.num_clbits, shots)


def _count_real_jobs(circuit: QuantumCircuit) -> int:
    vc = circuit if isinstance(circuit, VirtualCircuit) else None
    if vc is None and _has_virtual_ops(circuit):
        vc = VirtualCircuit(_normalize_circuit(circuit))
    if vc is None:
        return 1
    total = 0
    for frag, frag_circ in vc.fragment_circuits.items():
        inst_labels = vc.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circ, inst_labels)
        total += len(instantiations)
    return total


def _real_job_breakdown(circuit: QuantumCircuit) -> Dict[str, object]:
    vc = circuit if isinstance(circuit, VirtualCircuit) else None
    if vc is None and _has_virtual_ops(circuit):
        vc = VirtualCircuit(_normalize_circuit(circuit))
    if vc is None:
        return {"fragments": 1, "instantiations": [1], "jobs_total": 1}
    inst_counts = []
    for frag, frag_circ in vc.fragment_circuits.items():
        inst_labels = vc.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circ, inst_labels)
        inst_counts.append(len(instantiations))
    return {
        "fragments": len(vc.fragment_circuits),
        "instantiations": inst_counts,
        "jobs_total": int(sum(inst_counts)),
    }


def _real_fidelity_for_circuit(
    circuit: QuantumCircuit,
    shots: int,
    backend_name: str,
    seed: int,
    ctx: Dict[str, int],
    method_ctx: Dict[str, int | str],
) -> float:
    if _REAL_DRY_RUN:
        _dry_run_traverse(circuit, ctx, method_ctx)
        return 0.0
    if _has_virtual_ops(circuit):
        ideal = _simulate_virtual_counts(circuit, shots, None, seed)
        noisy = _real_virtual_counts(circuit, shots, backend_name, ctx, method_ctx)
    else:
        ideal = _simulate_counts(circuit, shots, None, seed)
        noisy = _real_counts(circuit, shots, backend_name, ctx, method_ctx)
    if not ideal or not noisy:
        return 0.0
    return _hellinger_fidelity_from_counts(ideal, noisy)


def _average_real_fidelity(
    circuits: List[QuantumCircuit],
    shots: int,
    backend_name: str,
    seed: int,
    method: str,
    bench: str,
    size: int,
) -> float:
    mean, _std = _real_fidelity_stats(
        circuits, shots, backend_name, seed, method, bench, size
    )
    return mean


def _real_fidelity_stats(
    circuits: List[QuantumCircuit],
    shots: int,
    backend_name: str,
    seed: int,
    method: str,
    bench: str,
    size: int,
) -> Tuple[float, float]:
    total = len(circuits)
    jobs_total = sum(_count_real_jobs(c) for c in circuits)
    global _REAL_DRY_RUN_CALLS
    _EVAL_LOGGER.warning(
        "Real QPU jobs method=%s bench=%s size=%s circuits_total=%s jobs_total=%s",
        method,
        bench,
        size,
        total,
        jobs_total,
    )
    method_ctx: Dict[str, int | str] = {
        "method": method,
        "bench": bench,
        "size": size,
        "expected_total": jobs_total,
        "submitted": 0,
        "completed": 0,
        "reused": 0,
        "progress_step": max(1, jobs_total // 10),
    }
    if _REAL_DRY_RUN and _is_eval_verbose():
        breakdown = [_real_job_breakdown(c) for c in circuits]
        _EVAL_LOGGER.info(
            "Real QPU jobs breakdown method=%s bench=%s size=%s fragments=%s instantiations=%s",
            method,
            bench,
            size,
            [b["fragments"] for b in breakdown],
            [b["instantiations"] for b in breakdown],
        )
    vals = []
    for idx, circuit in enumerate(circuits, start=1):
        ctx = {
            "circuit_idx": idx,
            "circuit_total": total,
            "fragment_idx": 1,
            "fragment_total": 1,
            "inst_idx": 1,
            "inst_total": 1,
        }
        vals.append(
            _real_fidelity_for_circuit(circuit, shots, backend_name, seed, ctx, method_ctx)
        )
    if int(method_ctx.get("completed", 0)) != jobs_total:
        _EVAL_LOGGER.warning(
            "Real QPU jobs mismatch method=%s bench=%s size=%s expected=%s completed=%s submitted=%s reused=%s",
            method,
            bench,
            size,
            jobs_total,
            method_ctx.get("completed"),
            method_ctx.get("submitted"),
            method_ctx.get("reused"),
        )
    if _REAL_DRY_RUN:
        return 0.0, 0.0
    mean = float(sum(vals) / max(1, len(vals)))
    std = float(np.std(vals)) if len(vals) > 1 else 0.0
    return mean, std


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


def _fidelity_stats(
    circuits: List[QuantumCircuit],
    shots: int,
    noise,
    seed: int,
) -> Tuple[float, float]:
    vals = [_fidelity_for_circuit(c, shots, noise, seed) for c in circuits]
    mean = float(sum(vals) / max(1, len(vals)))
    std = float(np.std(vals)) if len(vals) > 1 else 0.0
    return mean, std


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
    err_data: Optional[Dict[str, Dict[str, float]]] = None,
    show_avg: bool = False,
) -> None:
    x = np.arange(len(methods))
    width = 0.08
    all_vals: List[float] = []
    for i, (bench, label) in enumerate(benches):
        vals = [rel_data[bench].get(m, np.nan) for m in methods]
        all_vals.extend([v for v in vals if np.isfinite(v)])
        errs = None
        if err_data is not None:
            errs = [err_data.get(bench, {}).get(m, 0.0) for m in methods]
        if errs and any(e > 0 for e in errs):
            ax.bar(
                x + (i - len(benches) / 2) * width,
                vals,
                width,
                label=label,
                yerr=errs,
                capsize=2,
                ecolor="black",
                linewidth=0.5,
            )
        else:
            ax.bar(x + (i - len(benches) / 2) * width, vals, width, label=label)

    if show_avg:
        max_val = max(all_vals) if all_vals else 0.0
        pad = max(0.02, max_val * 0.06)
        if max_val > 0:
            ax.set_ylim(top=max(ax.get_ylim()[1], max_val + pad * 2))
        for idx, method in enumerate(methods):
            vals = [rel_data[bench].get(method, np.nan) for bench, _ in benches]
            vals = [v for v in vals if np.isfinite(v)]
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
    fidelity_err_by_size: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
    real_fidelity_by_size: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
    real_fidelity_err_by_size: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
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
            fidelity_err = fidelity_err_by_size.get(size, {}) if fidelity_err_by_size else None
            _plot_panel(
                axes[row, 2],
                f"Hellinger fidelity - {size} qubits (higher is better)",
                fidelity,
                benches,
                fidelity_methods,
                "Fidelity",
                err_data=fidelity_err,
                show_avg=True,
            )
        if real_fidelity_by_size:
            col = 3 if fidelity_by_size else 2
            fidelity = real_fidelity_by_size.get(size, {})
            fidelity_err = (
                real_fidelity_err_by_size.get(size, {}) if real_fidelity_err_by_size else None
            )
            _plot_panel(
                axes[row, col],
                f"Real Hellinger fidelity - {size} qubits (higher is better)",
                fidelity,
                benches,
                fidelity_methods,
                "Fidelity",
                err_data=fidelity_err,
                show_avg=True,
            )

    if axes.ndim == 1:
        handles, labels = axes[0].get_legend_handles_labels()
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=8, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / f"relative_properties_compare_{timestamp}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_job_counts(
    job_counts_by_size: Dict[int, Dict[str, Dict[str, float]]],
    benches: List[Tuple[str, str]],
    out_dir: Path,
    timestamp: str,
    methods: List[str],
) -> Path:
    plt = _import_matplotlib()
    sizes = sorted(job_counts_by_size.keys())
    rows = max(1, len(sizes))
    fig, axes = plt.subplots(rows, 1, figsize=(6.2, 3.6 * rows))
    if rows == 1:
        axes = np.array([axes])

    for row, size in enumerate(sizes):
        counts = job_counts_by_size.get(size, {})
        _plot_panel(
            axes[row],
            f"Real-QPU jobs (estimated) - {size} qubits",
            counts,
            benches,
            methods,
            "Jobs",
            show_avg=True,
        )

    if axes.ndim == 1:
        handles, labels = axes[0].get_legend_handles_labels()
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=8, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / f"real_job_counts_compare_{timestamp}.pdf"
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
    has_cost_calls = any("cost_search_calls" in r for r in timing_rows)
    ncols = 4 if has_cost_calls else 3

    fig, axes = plt.subplots(len(sizes), ncols, figsize=(5.5 * ncols, 3.8 * len(sizes)))
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, ncols)

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
        else:
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
        if not bench_labels:
            ax.set_visible(False)
        else:
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

        if has_cost_calls:
            ax = axes[row_idx, 3]
            if not bench_labels:
                ax.set_visible(False)
            else:
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
                else:
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
    for method in methods:
        if method in rel_depth_keys:
            continue
        key = method.lower()
        rel_depth_keys[method] = f"rel_depth_{key}"
        rel_nonlocal_keys[method] = f"rel_nonlocal_{key}"

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


def _method_prefix(method: str) -> str:
    if method == "QOS":
        return "qos"
    if method == "QOSN":
        return "qosn"
    if method == "FrozenQubits":
        return "fq"
    if method == "CutQC":
        return "cutqc"
    return method.lower()


def _build_progress_maps_from_rows(
    rows: List[Dict[str, object]],
    benches: List[Tuple[str, str]],
    sizes: List[int],
    methods: List[str],
    with_fidelity: bool,
    with_real_fidelity: bool,
):
    rel_by_size: Dict[int, Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]] = {}
    fidelity_by_size: Dict[int, Dict[str, Dict[str, float]]] = {}
    fidelity_err_by_size: Dict[int, Dict[str, Dict[str, float]]] = {}
    real_fidelity_by_size: Dict[int, Dict[str, Dict[str, float]]] = {}
    real_fidelity_err_by_size: Dict[int, Dict[str, Dict[str, float]]] = {}

    for size in sizes:
        rel_depth = {bench: {} for bench, _ in benches}
        rel_nonlocal = {bench: {} for bench, _ in benches}
        rel_by_size[size] = (rel_depth, rel_nonlocal)
        if with_fidelity:
            fidelity_by_size[size] = {bench: {} for bench, _ in benches}
            fidelity_err_by_size[size] = {bench: {} for bench, _ in benches}
        if with_real_fidelity:
            real_fidelity_by_size[size] = {bench: {} for bench, _ in benches}
            real_fidelity_err_by_size[size] = {bench: {} for bench, _ in benches}

    for row in rows:
        try:
            size = int(row.get("size"))
            bench = str(row.get("bench"))
        except Exception:
            continue
        if size not in rel_by_size:
            continue
        rel_depth, rel_nonlocal = rel_by_size[size]
        if bench not in rel_depth:
            continue
        for method in methods:
            prefix = _method_prefix(method)
            dkey = f"rel_depth_{prefix}"
            nkey = f"rel_nonlocal_{prefix}"
            dval = row.get(dkey, "")
            nval = row.get(nkey, "")
            if dval not in {"", None}:
                rel_depth[bench][method] = _safe_float(dval, float("nan"))
            if nval not in {"", None}:
                rel_nonlocal[bench][method] = _safe_float(nval, float("nan"))
            if with_fidelity:
                fkey = f"{prefix}_fidelity"
                fskey = f"{prefix}_fidelity_std"
                if row.get(fkey, "") not in {"", None}:
                    fidelity_by_size[size][bench][method] = _safe_float(row.get(fkey), float("nan"))
                if row.get(fskey, "") not in {"", None}:
                    fidelity_err_by_size[size][bench][method] = _safe_float(
                        row.get(fskey), float("nan")
                    )
            if with_real_fidelity:
                rfkey = f"{prefix}_real_fidelity"
                rfskey = f"{prefix}_real_fidelity_std"
                if row.get(rfkey, "") not in {"", None}:
                    real_fidelity_by_size[size][bench][method] = _safe_float(
                        row.get(rfkey), float("nan")
                    )
                if row.get(rfskey, "") not in {"", None}:
                    real_fidelity_err_by_size[size][bench][method] = _safe_float(
                        row.get(rfskey), float("nan")
                    )

        if with_fidelity and row.get("baseline_fidelity", "") not in {"", None}:
            fidelity_by_size[size][bench]["Qiskit"] = _safe_float(
                row.get("baseline_fidelity"), float("nan")
            )
            if row.get("baseline_fidelity_std", "") not in {"", None}:
                fidelity_err_by_size[size][bench]["Qiskit"] = _safe_float(
                    row.get("baseline_fidelity_std"), float("nan")
                )
        if with_real_fidelity and row.get("baseline_real_fidelity", "") not in {"", None}:
            real_fidelity_by_size[size][bench]["Qiskit"] = _safe_float(
                row.get("baseline_real_fidelity"), float("nan")
            )
            if row.get("baseline_real_fidelity_std", "") not in {"", None}:
                real_fidelity_err_by_size[size][bench]["Qiskit"] = _safe_float(
                    row.get("baseline_real_fidelity_std"), float("nan")
                )

    return (
        rel_by_size,
        (fidelity_by_size if with_fidelity else None),
        (fidelity_err_by_size if with_fidelity else None),
        (real_fidelity_by_size if with_real_fidelity else None),
        (real_fidelity_err_by_size if with_real_fidelity else None),
    )


def _resume_row_complete(row: Dict[str, object], args, methods: List[str]) -> bool:
    for method in methods:
        prefix = _method_prefix(method)
        if f"{prefix}_depth" not in row or f"{prefix}_nonlocal" not in row:
            return False
        if args.with_fidelity and f"{prefix}_fidelity" not in row:
            return False
        if args.with_real_fidelity and f"{prefix}_real_fidelity" not in row:
            return False
    if args.with_fidelity and "baseline_fidelity" not in row:
        return False
    if args.with_real_fidelity and "baseline_real_fidelity" not in row:
        return False
    return True


def _restore_row_aggregates(
    row: Dict[str, object],
    bench: str,
    size: int,
    methods: List[str],
    rel_depth: Dict[str, Dict[str, float]],
    rel_nonlocal: Dict[str, Dict[str, float]],
    fidelity_by_size: Dict[int, Dict[str, Dict[str, float]]],
    fidelity_err_by_size: Dict[int, Dict[str, Dict[str, float]]],
    real_fidelity_by_size: Dict[int, Dict[str, Dict[str, float]]],
    real_fidelity_err_by_size: Dict[int, Dict[str, Dict[str, float]]],
    real_job_counts_by_size: Dict[int, Dict[str, Dict[str, float]]],
    with_fidelity: bool,
    with_real_fidelity: bool,
) -> None:
    if with_fidelity:
        fidelity_by_size[size][bench]["Qiskit"] = _safe_float(row.get("baseline_fidelity", ""), 0.0)
        fidelity_err_by_size[size][bench]["Qiskit"] = _safe_float(
            row.get("baseline_fidelity_std", ""), 0.0
        )
    if with_real_fidelity:
        real_fidelity_by_size[size][bench]["Qiskit"] = _safe_float(
            row.get("baseline_real_fidelity", ""), 0.0
        )
        real_fidelity_err_by_size[size][bench]["Qiskit"] = _safe_float(
            row.get("baseline_real_fidelity_std", ""), 0.0
        )
    has_job_counts = size in real_job_counts_by_size and bench in real_job_counts_by_size[size]
    if has_job_counts:
        real_job_counts_by_size[size][bench]["Qiskit"] = 1.0

    for method in methods:
        prefix = _method_prefix(method)
        rel_d_key = f"rel_depth_{prefix}"
        rel_n_key = f"rel_nonlocal_{prefix}"
        if rel_d_key in row and row[rel_d_key] != "":
            rel_depth[bench][method] = _safe_float(row.get(rel_d_key, 0.0), 0.0)
        if rel_n_key in row and row[rel_n_key] != "":
            rel_nonlocal[bench][method] = _safe_float(row.get(rel_n_key, 0.0), 0.0)
        if with_fidelity and f"{prefix}_fidelity" in row:
            fidelity_by_size[size][bench][method] = _safe_float(
                row.get(f"{prefix}_fidelity", ""), 0.0
            )
            fidelity_err_by_size[size][bench][method] = _safe_float(
                row.get(f"{prefix}_fidelity_std", ""), 0.0
            )
        if with_real_fidelity and f"{prefix}_real_fidelity" in row:
            real_fidelity_by_size[size][bench][method] = _safe_float(
                row.get(f"{prefix}_real_fidelity", ""), 0.0
            )
            real_fidelity_err_by_size[size][bench][method] = _safe_float(
                row.get(f"{prefix}_real_fidelity_std", ""), 0.0
            )
        num_key = f"{prefix}_num_circuits"
        if has_job_counts and num_key in row:
            real_job_counts_by_size[size][bench][method] = float(_safe_int(row.get(num_key, 0), 0))


def _run_eval(
    args,
    benches,
    sizes,
    qose_runs: Optional[Dict[str, Callable]],
    methods: List[str],
    initial_rows: Optional[List[Dict[str, object]]] = None,
    initial_timing_rows: Optional[List[Dict[str, object]]] = None,
    completed_keys: Optional[set[str]] = None,
    method_cache_dir: Optional[Path] = None,
    on_bench_complete: Optional[Callable[[int, str, List[Dict[str, object]], List[Dict[str, object]], set[str]], None]] = None,
):
    all_rows: List[Dict[str, object]] = list(initial_rows or [])
    rel_by_size: Dict[int, Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]] = {}
    timing_rows = list(initial_timing_rows or [])
    fidelity_by_size: Dict[int, Dict[str, Dict[str, float]]] = {}
    fidelity_err_by_size: Dict[int, Dict[str, Dict[str, float]]] = {}
    real_fidelity_by_size: Dict[int, Dict[str, Dict[str, float]]] = {}
    real_fidelity_err_by_size: Dict[int, Dict[str, Dict[str, float]]] = {}
    real_job_counts_by_size: Dict[int, Dict[str, Dict[str, float]]] = {}
    cut_circuits: Dict[Tuple[int, str, str], List[QuantumCircuit]] = {}
    fragment_fidelity: Dict[Tuple[int, str], Dict[str, object]] = {}
    qose_runs = qose_runs or {}
    qose_methods = [m for m in methods if m in qose_runs]
    include_qose = bool(qose_methods)
    run_qos = "QOS" in methods
    run_qosn = "QOSN" in methods
    run_fq = "FrozenQubits" in methods
    run_cutqc = "CutQC" in methods
    completed_keys = set(completed_keys or set())
    existing_row_map: Dict[str, Dict[str, object]] = {}
    for _row in all_rows:
        try:
            existing_row_map[_bench_key(int(_row["size"]), str(_row["bench"]))] = _row
        except Exception:
            continue
    total_benches = len(sizes) * len(benches)

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
            fidelity_err_by_size[size] = {bench: {} for bench, _ in BENCHES}
        if args.with_real_fidelity:
            real_fidelity_by_size[size] = {bench: {} for bench, _ in BENCHES}
            real_fidelity_err_by_size[size] = {bench: {} for bench, _ in BENCHES}
        if args.with_real_fidelity or _REAL_DRY_RUN:
            job_methods = ["Qiskit"] + [m for m in methods if m != "Qiskit"]
            real_job_counts_by_size[size] = {
                bench: {m: 0.0 for m in job_methods} for bench, _ in BENCHES
            }

        for bench, _label in benches:
            bench_key = _bench_key(size, bench)
            cached_row = existing_row_map.get(bench_key)
            if (
                bench_key in completed_keys
                and cached_row is not None
                and _resume_row_complete(cached_row, args, methods)
            ):
                _restore_row_aggregates(
                    cached_row,
                    bench,
                    size,
                    methods,
                    rel_depth,
                    rel_nonlocal,
                    fidelity_by_size,
                    fidelity_err_by_size,
                    real_fidelity_by_size,
                    real_fidelity_err_by_size,
                    real_job_counts_by_size,
                    bool(args.with_fidelity),
                    bool(args.with_real_fidelity),
                )
                print(
                    f"[progress] resume-skip size={size} bench={bench} completed={len(completed_keys)}/{total_benches}",
                    flush=True,
                )
                continue
            print(
                f"[progress] start size={size} bench={bench} completed={len(completed_keys)}/{total_benches}",
                flush=True,
            )
            bench_start = time.perf_counter()
            qc = _load_qasm_circuit(bench, size)
            base = _analyze_circuit(
                qc,
                args.metric_mode,
                args.metrics_baseline,
                args.metrics_optimization_level,
            )
            qos_m = qos_t = qos_circs = None
            qosn_m = qosn_t = qosn_circs = None
            fq_m = fq_t = fq_circs = None
            cutqc_m = cutqc_t = cutqc_circs = None

            if run_qos:
                cached = _load_method_cache(method_cache_dir, size, bench, "QOS")
                if cached is not None:
                    qos_m, qos_circs = cached
                    qos_t = {}
                    print(f"[progress] size={size} bench={bench} method=QOS source=cache", flush=True)
                else:
                    print(f"[progress] size={size} bench={bench} method=QOS source=compute", flush=True)
                    qos_m, qos_t, qos_circs = _run_mitigator(
                        qc, [], args, bench_name=bench, size_label=size
                    )
                    _save_method_cache(method_cache_dir, size, bench, "QOS", qos_m, qos_circs)
            if run_qosn:
                cached = _load_method_cache(method_cache_dir, size, bench, "QOSN")
                if cached is not None:
                    qosn_m, qosn_circs = cached
                    qosn_t = {}
                    print(f"[progress] size={size} bench={bench} method=QOSN source=cache", flush=True)
                else:
                    print(f"[progress] size={size} bench={bench} method=QOSN source=compute", flush=True)
                    qosn_m, qosn_t, qosn_circs = _run_mitigator(
                        qc,
                        [],
                        args,
                        use_cost_search_override=False,
                        bench_name=bench,
                        size_label=size,
                    )
                    _save_method_cache(method_cache_dir, size, bench, "QOSN", qosn_m, qosn_circs)
            if run_fq:
                cached = _load_method_cache(method_cache_dir, size, bench, "FrozenQubits")
                if cached is not None:
                    fq_m, fq_circs = cached
                    fq_t = {}
                    print(f"[progress] size={size} bench={bench} method=FrozenQubits source=cache", flush=True)
                else:
                    print(f"[progress] size={size} bench={bench} method=FrozenQubits source=compute", flush=True)
                    fq_m, fq_t, fq_circs = _run_mitigator(
                        qc, ["QF"], args, bench_name=bench, size_label=size
                    )
                    _save_method_cache(method_cache_dir, size, bench, "FrozenQubits", fq_m, fq_circs)

            if run_cutqc:
                cached = _load_method_cache(method_cache_dir, size, bench, "CutQC")
                if cached is not None:
                    cutqc_m, cutqc_circs = cached
                    cutqc_t = {}
                    print(f"[progress] size={size} bench={bench} method=CutQC source=cache", flush=True)
                else:
                    print(f"[progress] size={size} bench={bench} method=CutQC source=compute", flush=True)
                    cutqc_methods = ["GV"] if args.cutqc_method == "gv" else ["WC"]
                    cutqc_m, cutqc_t, cutqc_circs = _run_mitigator(
                        qc, cutqc_methods, args, bench_name=bench, size_label=size
                    )
                    _save_method_cache(method_cache_dir, size, bench, "CutQC", cutqc_m, cutqc_circs)

            qose_results: Dict[str, Dict[str, object]] = {}
            if include_qose:
                for method in qose_methods:
                    cached = _load_method_cache(method_cache_dir, size, bench, method)
                    if cached is not None:
                        qose_m, qose_circs = cached
                        qose_t = {}
                        qose_elapsed = 0.0
                        print(f"[progress] size={size} bench={bench} method={method} source=cache", flush=True)
                    else:
                        print(f"[progress] size={size} bench={bench} method={method} source=compute", flush=True)
                        qose_t0 = time.perf_counter()
                        qose_m, qose_t, qose_circs = _run_qose(qc, args, qose_runs[method])
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
                        qose_elapsed = time.perf_counter() - qose_t0
                        _save_method_cache(method_cache_dir, size, bench, method, qose_m, qose_circs)
                    qose_results[method] = {
                        "metrics": qose_m,
                        "timing": qose_t,
                        "circs": qose_circs,
                        "elapsed": qose_elapsed,
                    }

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
            if include_qose:
                for method, qose_data in qose_results.items():
                    qose_m = qose_data.get("metrics")
                    if not qose_m:
                        continue
                    rel_depth[bench][method] = _relative(qose_m["depth"], base["depth"])
                    rel_nonlocal[bench][method] = _relative(
                        qose_m["num_nonlocal_gates"], base["num_nonlocal_gates"]
                    )

            if args.with_fidelity:
                rel_fidelity = {}
                if args.verbose:
                    print("  fidelity sim start", flush=True)
                sim_times = {}
                t0 = time.perf_counter()
                base_fidelity, base_std = _fidelity_stats(
                    [qc], args.fidelity_shots, noise, args.fidelity_seed
                )
                sim_times["Qiskit"] = time.perf_counter() - t0
                fidelity_by_size[size][bench] = {"Qiskit": base_fidelity}
                fidelity_err_by_size[size][bench]["Qiskit"] = base_std
                rel_fidelity = {}

                qos_fidelity = ""
                qosn_fidelity = ""
                fq_fidelity = ""
                cutqc_fidelity = ""
                qose_fidelity: Dict[str, float] = {}
                qose_std: Dict[str, float] = {}
                rel_fidelity = {}

                if run_qos and qos_circs is not None:
                    t0 = time.perf_counter()
                    qos_fidelity, qos_std = _fidelity_stats(
                        qos_circs, args.fidelity_shots, noise, args.fidelity_seed
                    )
                    sim_times["QOS"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["QOS"] = qos_fidelity
                    fidelity_err_by_size[size][bench]["QOS"] = qos_std
                    rel_fidelity["QOS"] = _relative(qos_fidelity, base_fidelity)
                if run_qosn and qosn_circs is not None:
                    t0 = time.perf_counter()
                    qosn_fidelity, qosn_std = _fidelity_stats(
                        qosn_circs, args.fidelity_shots, noise, args.fidelity_seed
                    )
                    sim_times["QOSN"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["QOSN"] = qosn_fidelity
                    fidelity_err_by_size[size][bench]["QOSN"] = qosn_std
                    rel_fidelity["QOSN"] = _relative(qosn_fidelity, base_fidelity)
                if run_fq and fq_circs is not None:
                    t0 = time.perf_counter()
                    fq_fidelity, fq_std = _fidelity_stats(
                        fq_circs, args.fidelity_shots, noise, args.fidelity_seed
                    )
                    sim_times["FrozenQubits"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["FrozenQubits"] = fq_fidelity
                    fidelity_err_by_size[size][bench]["FrozenQubits"] = fq_std
                    rel_fidelity["FrozenQubits"] = _relative(fq_fidelity, base_fidelity)
                if run_cutqc and cutqc_circs is not None:
                    t0 = time.perf_counter()
                    cutqc_fidelity, cutqc_std = _fidelity_stats(
                        cutqc_circs, args.fidelity_shots, noise, args.fidelity_seed
                    )
                    sim_times["CutQC"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["CutQC"] = cutqc_fidelity
                    fidelity_err_by_size[size][bench]["CutQC"] = cutqc_std
                    rel_fidelity["CutQC"] = _relative(cutqc_fidelity, base_fidelity)
                if include_qose:
                    for method, qose_data in qose_results.items():
                        qose_circs = qose_data.get("circs")
                        if not qose_circs:
                            continue
                        t0 = time.perf_counter()
                        qose_val, qose_std_val = _fidelity_stats(
                            qose_circs, args.fidelity_shots, noise, args.fidelity_seed
                        )
                        sim_times[method] = time.perf_counter() - t0
                        qose_fidelity[method] = qose_val
                        qose_std[method] = qose_std_val
                        fidelity_by_size[size][bench][method] = qose_val
                        fidelity_err_by_size[size][bench][method] = qose_std_val
                        rel_fidelity[method] = _relative(qose_val, base_fidelity)

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
                base_std = ""
                qos_fidelity = ""
                qosn_fidelity = ""
                fq_fidelity = ""
                cutqc_fidelity = ""
                qose_fidelity = {}
                qose_std = {}
                rel_fidelity = {}
                rel_fidelity_qos = ""
                rel_fidelity_qosn = ""
                rel_fidelity_fq = ""
                rel_fidelity_cutqc = ""
                rel_fidelity_qose = ""

            if args.with_real_fidelity:
                print(f"[progress] size={size} bench={bench} method=Qiskit stage=real_fidelity", flush=True)
                real_base, real_base_std = _real_fidelity_stats(
                    [qc],
                    args.real_fidelity_shots,
                    args.real_backend,
                    args.fidelity_seed,
                    "Qiskit",
                    bench,
                    size,
                )
                real_fidelity_by_size[size][bench]["Qiskit"] = real_base
                real_fidelity_err_by_size[size][bench]["Qiskit"] = real_base_std
                if run_qos and qos_circs is not None:
                    print(f"[progress] size={size} bench={bench} method=QOS stage=real_fidelity", flush=True)
                    real_qos, real_qos_std = _real_fidelity_stats(
                        qos_circs,
                        args.real_fidelity_shots,
                        args.real_backend,
                        args.fidelity_seed,
                        "QOS",
                        bench,
                        size,
                    )
                    real_fidelity_by_size[size][bench]["QOS"] = real_qos
                    real_fidelity_err_by_size[size][bench]["QOS"] = real_qos_std
                if run_qosn and qosn_circs is not None:
                    print(f"[progress] size={size} bench={bench} method=QOSN stage=real_fidelity", flush=True)
                    real_qosn, real_qosn_std = _real_fidelity_stats(
                        qosn_circs,
                        args.real_fidelity_shots,
                        args.real_backend,
                        args.fidelity_seed,
                        "QOSN",
                        bench,
                        size,
                    )
                    real_fidelity_by_size[size][bench]["QOSN"] = real_qosn
                    real_fidelity_err_by_size[size][bench]["QOSN"] = real_qosn_std
                if run_fq and fq_circs is not None:
                    print(f"[progress] size={size} bench={bench} method=FrozenQubits stage=real_fidelity", flush=True)
                    real_fq, real_fq_std = _real_fidelity_stats(
                        fq_circs,
                        args.real_fidelity_shots,
                        args.real_backend,
                        args.fidelity_seed,
                        "FrozenQubits",
                        bench,
                        size,
                    )
                    real_fidelity_by_size[size][bench]["FrozenQubits"] = real_fq
                    real_fidelity_err_by_size[size][bench]["FrozenQubits"] = real_fq_std
                if run_cutqc and cutqc_circs is not None:
                    print(f"[progress] size={size} bench={bench} method=CutQC stage=real_fidelity", flush=True)
                    real_cut, real_cut_std = _real_fidelity_stats(
                        cutqc_circs,
                        args.real_fidelity_shots,
                        args.real_backend,
                        args.fidelity_seed,
                        "CutQC",
                        bench,
                        size,
                    )
                    real_fidelity_by_size[size][bench]["CutQC"] = real_cut
                    real_fidelity_err_by_size[size][bench]["CutQC"] = real_cut_std
                if include_qose:
                    for method, qose_data in qose_results.items():
                        qose_circs = qose_data.get("circs")
                        if not qose_circs:
                            continue
                        print(f"[progress] size={size} bench={bench} method={method} stage=real_fidelity", flush=True)
                        real_qose, real_qose_std = _real_fidelity_stats(
                            qose_circs,
                            args.real_fidelity_shots,
                            args.real_backend,
                            args.fidelity_seed,
                            method,
                            bench,
                            size,
                        )
                        real_fidelity_by_size[size][bench][method] = real_qose
                        real_fidelity_err_by_size[size][bench][method] = real_qose_std
            else:
                real_base = ""
                real_base_std = ""
                real_qos = ""
                real_qos_std = ""
                real_qosn = ""
                real_qosn_std = ""
                real_fq = ""
                real_fq_std = ""
                real_cut = ""
                real_cut_std = ""
                real_qose: Dict[str, float] = {}
                real_qose_std: Dict[str, float] = {}
            if args.with_real_fidelity or _REAL_DRY_RUN:
                real_job_counts_by_size[size][bench]["Qiskit"] = float(
                    sum(_count_real_jobs(c) for c in [qc])
                )
                if run_qos and qos_circs is not None:
                    real_job_counts_by_size[size][bench]["QOS"] = float(
                        sum(_count_real_jobs(c) for c in qos_circs)
                    )
                if run_qosn and qosn_circs is not None:
                    real_job_counts_by_size[size][bench]["QOSN"] = float(
                        sum(_count_real_jobs(c) for c in qosn_circs)
                    )
                if run_fq and fq_circs is not None:
                    real_job_counts_by_size[size][bench]["FrozenQubits"] = float(
                        sum(_count_real_jobs(c) for c in fq_circs)
                    )
                if run_cutqc and cutqc_circs is not None:
                    real_job_counts_by_size[size][bench]["CutQC"] = float(
                        sum(_count_real_jobs(c) for c in cutqc_circs)
                    )
                if include_qose:
                    for method, qose_data in qose_results.items():
                        qose_circs = qose_data.get("circs")
                        if not qose_circs:
                            continue
                        real_job_counts_by_size[size][bench][method] = float(
                            sum(_count_real_jobs(c) for c in qose_circs)
                        )

            qiskit_sim = float(sim_times.get("Qiskit", 0.0))

            row = {
                "bench": bench,
                "size": size,
                "baseline_depth": base["depth"],
                "baseline_nonlocal": base["num_nonlocal_gates"],
                "baseline_fidelity": base_fidelity,
                "baseline_fidelity_std": base_std,
                "baseline_real_fidelity": real_base,
                "baseline_real_fidelity_std": real_base_std,
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
                        "qos_fidelity_std": qos_std if args.with_fidelity else "",
                        "qos_real_fidelity": real_qos if args.with_real_fidelity else "",
                        "qos_real_fidelity_std": real_qos_std if args.with_real_fidelity else "",
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
                        "qosn_fidelity_std": qosn_std if args.with_fidelity else "",
                        "qosn_real_fidelity": real_qosn if args.with_real_fidelity else "",
                        "qosn_real_fidelity_std": real_qosn_std if args.with_real_fidelity else "",
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
                        "fq_fidelity_std": fq_std if args.with_fidelity else "",
                        "fq_real_fidelity": real_fq if args.with_real_fidelity else "",
                        "fq_real_fidelity_std": real_fq_std if args.with_real_fidelity else "",
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
                        "cutqc_fidelity_std": cutqc_std if args.with_fidelity else "",
                        "cutqc_real_fidelity": real_cut if args.with_real_fidelity else "",
                        "cutqc_real_fidelity_std": real_cut_std if args.with_real_fidelity else "",
                        "cutqc_sim_time": cutqc_sim,
                        "cutqc_num_circuits": cutqc_num_circuits,
                        "cutqc_classical_overhead": _safe_ratio(cutqc_sim, qiskit_sim),
                        "cutqc_quantum_overhead": float(cutqc_num_circuits),
                        "rel_depth_cutqc": rel_depth[bench]["CutQC"],
                        "rel_nonlocal_cutqc": rel_nonlocal[bench]["CutQC"],
                        "rel_fidelity_cutqc": rel_fidelity_cutqc,
                    }
                )
            if include_qose:
                for method, qose_data in qose_results.items():
                    qose_m = qose_data.get("metrics")
                    qose_circs = qose_data.get("circs") or []
                    if not qose_m:
                        continue
                    qose_sim = float(sim_times.get(method, 0.0))
                    qose_num_circuits = max(1, len(qose_circs))
                    prefix = method.lower()
                    row.update(
                        {
                            f"{prefix}_depth": qose_m["depth"],
                            f"{prefix}_nonlocal": qose_m["num_nonlocal_gates"],
                            f"{prefix}_fidelity": qose_fidelity.get(method, ""),
                            f"{prefix}_fidelity_std": qose_std.get(method, ""),
                            f"{prefix}_real_fidelity": real_qose.get(method, "") if args.with_real_fidelity else "",
                            f"{prefix}_real_fidelity_std": real_qose_std.get(method, "") if args.with_real_fidelity else "",
                            f"{prefix}_sim_time": qose_sim,
                            f"{prefix}_num_circuits": qose_num_circuits,
                            f"{prefix}_classical_overhead": _safe_ratio(qose_sim, qiskit_sim),
                            f"{prefix}_quantum_overhead": float(qose_num_circuits),
                            f"rel_depth_{prefix}": rel_depth[bench].get(method, ""),
                            f"rel_nonlocal_{prefix}": rel_nonlocal[bench].get(method, ""),
                            f"rel_fidelity_{prefix}": rel_fidelity.get(method, ""),
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
                if include_qose:
                    for method, qose_data in qose_results.items():
                        qose_t = qose_data.get("timing")
                        if qose_t is not None:
                            timing_methods.append((method, qose_t))
                for method, timing in timing_methods:
                    row = {"bench": bench, "size": size, "method": method}
                    row.update(timing)
                    if args.with_fidelity:
                        row["simulation"] = sim_times.get(method, 0.0)
                    timing_rows.append(row)
            completed_keys.add(bench_key)
            existing_row_map[bench_key] = row
            if on_bench_complete is not None:
                on_bench_complete(size, bench, all_rows, timing_rows, completed_keys)
            if args.cut_visualization:
                cut_circuits[(size, bench, "Qiskit")] = [qc]
                if run_qos and qos_circs is not None:
                    cut_circuits[(size, bench, "QOS")] = qos_circs
                if run_qosn and qosn_circs is not None:
                    cut_circuits[(size, bench, "QOSN")] = qosn_circs
                if include_qose:
                    for method, qose_data in qose_results.items():
                        qose_circs = qose_data.get("circs")
                        if qose_circs is not None:
                            cut_circuits[(size, bench, method)] = qose_circs
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
        fidelity_err_by_size,
        real_fidelity_by_size,
        real_fidelity_err_by_size,
        cut_circuits,
        fragment_fidelity,
        real_job_counts_by_size,
    )


def _load_qose_runs_from_args(
    args,
) -> Tuple[Dict[str, Callable], Dict[str, Path]]:
    qose_runs: Dict[str, Callable] = {}
    qose_programs: Dict[str, Path] = {}
    manual_map = getattr(args, "qose_manual", "").strip()
    if manual_map:
        root = Path(getattr(args, "qose_best_root", str(ROOT / "openevolve_output")))
        manual_paths = _parse_manual_qose_mapping(manual_map, root)
        for method, program in manual_paths.items():
            try:
                qose_runs[method] = _load_qose_run(program)
                qose_programs[method] = program
            except Exception as exc:
                if getattr(args, "verbose", False):
                    print(f"Skipping QOSE program {program}: {exc}", file=sys.stderr)
        return qose_runs, qose_programs
    if getattr(args, "qose_auto_best", False):
        models = [m.strip() for m in getattr(args, "qose_models", "").split(",") if m.strip()]
        root = Path(getattr(args, "qose_best_root", str(ROOT / "openevolve_output")))
        best_programs = _find_best_qose_programs(models, root)
        for model in models:
            program = best_programs.get(model)
            if not program:
                continue
            try:
                qose_runs[model] = _load_qose_run(program)
                qose_programs[model] = program
            except Exception as exc:
                if getattr(args, "verbose", False):
                    print(f"Skipping QOSE program {program}: {exc}", file=sys.stderr)
    else:
        qose_program = _find_qose_program(getattr(args, "qose_program", ""))
        if not qose_program:
            return qose_runs, qose_programs
        try:
            qose_runs["QOSE"] = _load_qose_run(qose_program)
            qose_programs["QOSE"] = qose_program
        except Exception as exc:
            if getattr(args, "verbose", False):
                print(f"Skipping QOSE program {qose_program}: {exc}", file=sys.stderr)
    return qose_runs, qose_programs


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
    parser.add_argument(
        "--real-fidelity-dry-run",
        action="store_true",
        help="Log real-QPU job counts without submitting jobs.",
    )
    parser.add_argument("--real-fidelity-shots", type=int, default=1000)
    parser.add_argument("--real-backend", default="ibm_torino")
    parser.add_argument(
        "--real-job-timeout-sec",
        type=int,
        default=0,
        help="Abort if a single real-QPU job waits longer than this many seconds (0 disables).",
    )
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
            "Comma-separated methods to run: FrozenQubits,CutQC,QOS,QOSN,QOSE,qwen,gemini,gpt "
            "(or 'all' for defaults)."
        ),
    )
    parser.add_argument(
        "--cost-search-log",
        default="",
        help="Optional path to write cost-search trace CSV.",
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
        "--qose-manual",
        default="",
        help=(
            "Manual model mapping: qwen=folder,gemini=folder,gpt=folder. "
            "Values can be folders under openevolve_output or explicit paths. "
            "If set, overrides --qose-auto-best."
        ),
    )
    parser.add_argument(
        "--qose-auto-best",
        action="store_true",
        help="Auto-select best evolved programs per model from openevolve_output.",
    )
    parser.add_argument(
        "--qose-models",
        default="qwen,gemini,gpt",
        help="Comma-separated model names to load when using --qose-auto-best.",
    )
    parser.add_argument(
        "--qose-best-root",
        default=str(ROOT / "openevolve_output"),
        help="Root directory to search for best_program_info.json files.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "evaluation" / "plots"),
        help="Output directory for figures and CSV.",
    )
    parser.add_argument(
        "--resume-state",
        default="",
        help="Path to resume-state JSON. Default: <out-dir>/full_eval_progress.json",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing resume-state and run from scratch.",
    )
    parser.add_argument(
        "--reset-resume",
        action="store_true",
        help="Delete existing resume-state before starting.",
    )
    args = parser.parse_args()
    global _REAL_DRY_RUN
    _REAL_DRY_RUN = bool(args.real_fidelity_dry_run or not args.with_real_fidelity)
    os.environ["QVM_CLINGO_TIMEOUT_SEC"] = str(args.clingo_timeout_sec)
    os.environ["QVM_MAX_PARTITION_TRIES"] = str(args.max_partition_tries)
    os.environ["QOS_COST_SEARCH_MAX_ITERS"] = str(args.qos_cost_search_max_iters)
    os.environ["QOS_REAL_JOB_TIMEOUT_SEC"] = str(max(0, int(args.real_job_timeout_sec)))

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
    all_rows: List[Dict[str, object]] = []
    rel_by_size: Dict[int, Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]] = {}

    if args.benches == "all":
        benches = BENCHES
    else:
        selected = {b.strip() for b in args.benches.split(",") if b.strip()}
        benches = [(b, label) for b, label in BENCHES if b in selected]

    qose_runs, qose_programs = _load_qose_runs_from_args(args)
    selected_methods = _parse_methods(args.methods, bool(qose_runs), list(qose_runs.keys()))
    missing_qose = [
        m
        for m in selected_methods
        if m in {"QOSE", "qwen", "gemini", "gpt"} and m not in qose_runs
    ]
    if missing_qose:
        print(
            f"Requested QOSE methods not available: {', '.join(missing_qose)}; skipping.",
            file=sys.stderr,
        )
        selected_methods = [m for m in selected_methods if m not in missing_qose]
    include_qose = bool(qose_runs) and any(m in qose_runs for m in selected_methods)
    if args.verbose and qose_programs and include_qose:
        for method, program in qose_programs.items():
            if method in selected_methods:
                print(f"Using QOSE program ({method}): {program}", file=sys.stderr)

    resume_state_path = (
        Path(args.resume_state) if args.resume_state.strip() else out_dir / "full_eval_progress.json"
    )
    if args.reset_resume and resume_state_path.exists():
        resume_state_path.unlink()
    job_cache_path = resume_state_path.with_name(f"{resume_state_path.stem}_job_cache.jsonl")
    if args.reset_resume and job_cache_path.exists():
        job_cache_path.unlink()
    global _REAL_JOB_RESULT_CACHE
    _REAL_JOB_RESULT_CACHE = _RealJobResultCache(job_cache_path)
    print(
        f"Real-job cache: {job_cache_path} entries={_REAL_JOB_RESULT_CACHE.count()}",
        flush=True,
    )
    signature = _resume_signature(args, benches, sizes, selected_methods)
    initial_rows: List[Dict[str, object]] = []
    initial_timing_rows: List[Dict[str, object]] = []
    completed_keys: set[str] = set()
    if not args.no_resume:
        loaded = _load_resume_state(resume_state_path, signature)
        if loaded is not None:
            initial_rows = list(loaded.get("rows", []))
            initial_timing_rows = list(loaded.get("timing_rows", []))
            completed_keys = set(loaded.get("completed_keys", []))
            print(
                f"Resume state loaded: {resume_state_path} completed={len(completed_keys)}",
                flush=True,
            )
        else:
            print(f"Resume state: starting fresh ({resume_state_path})", flush=True)
    else:
        print("Resume disabled (--no-resume): starting fresh", flush=True)

    partial_csv_path = out_dir / f"relative_properties_partial{tag_suffix}.csv"
    partial_timing_path = out_dir / f"timing_partial{tag_suffix}.csv"
    method_cache_dir = resume_state_path.with_name(f"{resume_state_path.stem}_method_cache")
    method_cache_dir.mkdir(parents=True, exist_ok=True)
    progress_plot_stamp = f"progress{tag_suffix}" if tag_suffix else "progress"
    progress_total = len(sizes) * len(benches)

    def _on_bench_complete(
        size: int,
        bench: str,
        rows: List[Dict[str, object]],
        timing: List[Dict[str, object]],
        completed: set[str],
    ) -> None:
        initial_rows[:] = rows
        initial_timing_rows[:] = timing
        completed_keys.clear()
        completed_keys.update(completed)
        _save_resume_state(
            resume_state_path,
            signature,
            completed,
            rows,
            timing,
            status="running",
            note=f"completed size={size} bench={bench}",
        )
        _write_rows_csv(partial_csv_path, rows)
        if args.collect_timing and timing:
            _write_rows_csv(partial_timing_path, timing)
        (
            progress_rel_by_size,
            progress_fidelity_by_size,
            progress_fidelity_err_by_size,
            progress_real_fidelity_by_size,
            progress_real_fidelity_err_by_size,
        ) = _build_progress_maps_from_rows(
            rows,
            benches,
            sizes,
            selected_methods,
            bool(args.with_fidelity),
            bool(args.with_real_fidelity),
        )
        _plot_combined(
            progress_rel_by_size,
            benches,
            out_dir,
            progress_plot_stamp,
            progress_fidelity_by_size,
            progress_fidelity_err_by_size,
            progress_real_fidelity_by_size,
            progress_real_fidelity_err_by_size,
            methods=selected_methods,
            fidelity_methods=["Qiskit"] + [m for m in selected_methods if m != "Qiskit"],
        )
        print(
            f"[progress] completed size={size} bench={bench} total={len(completed)}/{progress_total} saved={resume_state_path}",
            flush=True,
        )

    resume_cmd = " ".join(shlex.quote(x) for x in [sys.executable] + sys.argv)

    if args.sweep:
        if not args.no_resume:
            print("Sweep mode: resume-state is ignored for parameter grid search.", flush=True)
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

                            (
                                rows,
                                _rel,
                                _timing,
                                _fid,
                                _fid_err,
                                _real_fid,
                                _real_fid_err,
                                _cuts,
                                _frag,
                                _real_jobs,
                            ) = _run_eval(
                                args, benches, sizes, qose_runs, selected_methods
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
            fidelity_err_by_size,
            real_fidelity_by_size,
            real_fidelity_err_by_size,
            cut_circuits,
            fragment_fidelity,
            real_job_counts_by_size,
        ) = _run_eval(args, benches, sizes, qose_runs, selected_methods)
        print(f"Wrote sweep: {sweep_path}")
        print(
            "Best config:"
            f" budget={best_budget} size_to_reach={best_size}"
            f" clingo={best_clingo} max_tries={best_max_tries}"
            f" cost_iters={best_cost_iter}"
        )
    else:
        try:
            (
                all_rows,
                rel_by_size,
                timing_rows,
                fidelity_by_size,
                fidelity_err_by_size,
                real_fidelity_by_size,
                real_fidelity_err_by_size,
                cut_circuits,
                fragment_fidelity,
                real_job_counts_by_size,
            ) = _run_eval(
                args,
                benches,
                sizes,
                qose_runs,
                selected_methods,
                initial_rows=initial_rows,
                initial_timing_rows=initial_timing_rows,
                completed_keys=completed_keys,
                method_cache_dir=method_cache_dir,
                on_bench_complete=_on_bench_complete,
            )
        except RealQPUWaitTimeout as exc:
            _save_resume_state(
                resume_state_path,
                signature,
                completed_keys,
                initial_rows,
                initial_timing_rows,
                status="stalled",
                note=str(exc),
            )
            if initial_rows:
                _write_rows_csv(partial_csv_path, initial_rows)
            if args.collect_timing and initial_timing_rows:
                _write_rows_csv(partial_timing_path, initial_timing_rows)
            print(f"Stopped due to real-QPU wait timeout: {exc}", file=sys.stderr)
            print(f"Resume state saved: {resume_state_path}", file=sys.stderr)
            print("Set a new IBM key, then resume with:", file=sys.stderr)
            print(resume_cmd, file=sys.stderr)
            _cleanup_children()
            return

    fidelity_methods = ["Qiskit"] + [m for m in selected_methods if m != "Qiskit"]
    combined_path = _plot_combined(
        rel_by_size,
        benches,
        out_dir,
        f"{timestamp}{tag_suffix}",
        fidelity_by_size if args.with_fidelity else None,
        fidelity_err_by_size if args.with_fidelity else None,
        real_fidelity_by_size if args.with_real_fidelity else None,
        real_fidelity_err_by_size if args.with_real_fidelity else None,
        methods=selected_methods,
        fidelity_methods=fidelity_methods,
    )

    print(f"Wrote figure: {combined_path}")
    if real_job_counts_by_size:
        job_methods = ["Qiskit"] + [m for m in selected_methods if m != "Qiskit"]
        job_counts_fig = _plot_job_counts(
            real_job_counts_by_size, benches, out_dir, f"{timestamp}{tag_suffix}", job_methods
        )
        print(f"Wrote job counts figure: {job_counts_fig}")
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
    final_csv_path = out_dir / f"relative_properties_{timestamp}{tag_suffix}.csv"
    if all_rows:
        _write_rows_csv(final_csv_path, all_rows)
        print(f"Wrote relative properties CSV: {final_csv_path}")
    _save_resume_state(
        resume_state_path,
        signature,
        {_bench_key(int(r["size"]), str(r["bench"])) for r in all_rows if "size" in r and "bench" in r},
        all_rows,
        timing_rows,
        status="done",
        note="evaluation completed",
    )
    print(f"Resume state updated: {resume_state_path}")
    if _REAL_DRY_RUN and _REAL_DRY_RUN_CALLS:
        print(f"Real QPU dry-run sampler_run_calls_total={_REAL_DRY_RUN_CALLS}")
    if _REAL_JOB_RESULT_CACHE is not None:
        try:
            print(f"Real-job cache entries: {_REAL_JOB_RESULT_CACHE.count()}")
        except Exception:
            pass
    print("Full evaluation finished successfully.")
    _cleanup_children()


if __name__ == "__main__":
    main()
