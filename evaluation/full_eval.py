import argparse
import csv
import datetime as dt
import importlib.util
import json
import pickle
import shutil
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
_IDEAL_RESULT_CACHE = None
_RUN_QPU_SEC = 0.0
_RUN_QPU_SEC_KNOWN_JOBS = 0
_RUN_QPU_SEC_UNKNOWN_JOBS = 0


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

    def count_prefix(self, prefix: str) -> int:
        try:
            return sum(1 for key in self._results.keys() if str(key).startswith(prefix))
        except Exception:
            return 0

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


def _is_real_sim_progress_enabled() -> bool:
    raw = os.getenv("QOS_REAL_SIM_PROGRESS", "1").strip().lower()
    return raw in {"1", "true", "yes", "y"}


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


def _row_has_nonempty_value(row: Dict[str, object], key: str) -> bool:
    if key not in row:
        return False
    val = row.get(key)
    return val not in {"", None}


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


def _normalize_backend_name(raw: str) -> str:
    name = str(raw or "").strip().lower()
    if not name:
        return "generic"
    if "torino" in name:
        return "torino"
    if "marrakesh" in name:
        return "marrakesh"
    for prefix in ("ibm_", "fake_", "backend_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    cleaned = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in name)
    cleaned = cleaned.strip("_-")
    return cleaned or "generic"


def _state_backend_label(args) -> str:
    label = str(getattr(args, "state_backend_label", "") or "").strip()
    if label:
        return _normalize_backend_name(label)
    if bool(args.with_real_fidelity):
        return _normalize_backend_name(str(args.real_backend))
    if str(args.metrics_baseline) in {"torino", "marrakesh"}:
        return _normalize_backend_name(str(args.metrics_baseline))
    return "generic"


def _signature_real_backend(args) -> str:
    label = str(getattr(args, "state_backend_label", "") or "").strip()
    if label:
        return label
    return str(args.real_backend)


def _infer_state_backend(args) -> str:
    return _state_backend_label(args)


def _infer_state_stage(args) -> str:
    with_fid = bool(args.with_fidelity)
    with_real = bool(args.with_real_fidelity)
    if with_fid and with_real:
        return "mixed"
    if with_real and not with_fid:
        return "real_only"
    if with_fid and not with_real:
        return "sim_only"
    return "metrics_only"


def _resume_companion_paths(resume_path: Path) -> Tuple[Path, Path, Path]:
    if resume_path.name == "resume.json":
        return (
            resume_path.with_name("cache_job_real.jsonl"),
            resume_path.with_name("cache_ideal_sim.jsonl"),
            resume_path.with_name("method_cache"),
        )
    stem = resume_path.stem
    return (
        resume_path.with_name(f"{stem}_job_cache.jsonl"),
        resume_path.with_name(f"{stem}_ideal_cache.jsonl"),
        resume_path.with_name(f"{stem}_method_cache"),
    )


def _copy_path_if_missing(src: Path, dst: Path) -> bool:
    if not src.exists() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def _dedup_paths(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _seed_resume_from_candidates(
    *,
    primary_resume: Path,
    signature: Dict[str, object],
    candidates: List[Path],
) -> Optional[Path]:
    if primary_resume.exists():
        return None
    primary_job, primary_ideal, primary_method = _resume_companion_paths(primary_resume)
    for candidate in candidates:
        if candidate == primary_resume or not candidate.exists():
            continue
        loaded = _load_resume_state(candidate, signature)
        if loaded is None:
            continue
        primary_resume.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, primary_resume)
        candidate_job, candidate_ideal, candidate_method = _resume_companion_paths(candidate)
        _copy_path_if_missing(candidate_job, primary_job)
        _copy_path_if_missing(candidate_ideal, primary_ideal)
        _copy_path_if_missing(candidate_method, primary_method)
        return candidate
    return None


def _link_or_copy(target: Path, link: Path) -> None:
    if not target.exists():
        return
    # Avoid self-linking (can happen when caller already passes the destination path).
    try:
        if target.absolute() == link.absolute():
            return
    except Exception:
        pass
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        try:
            link.unlink()
        except IsADirectoryError:
            shutil.rmtree(link, ignore_errors=True)
    try:
        rel = os.path.relpath(target, link.parent)
        link.symlink_to(rel)
    except Exception:
        shutil.copy2(target, link)


def _copy_replace(target: Path, dst: Path) -> None:
    if not target.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        try:
            dst.unlink()
        except IsADirectoryError:
            shutil.rmtree(dst, ignore_errors=True)
    shutil.copy2(target, dst)


def _publish_full_eval_views(
    *,
    out_dir: Path,
    artifact_dir: Path,
    panel_paths: Optional[List[Path]] = None,
    breakdown_paths: Optional[List[Path]] = None,
    final_csv_path: Optional[Path] = None,
    timing_path: Optional[Path] = None,
    backend_label: str = "",
) -> None:
    figures_paper = out_dir / "figures" / "paper"
    figures_debug = out_dir / "figures" / "debug"
    tables_paper = out_dir / "tables" / "paper"
    tables_debug = out_dir / "tables" / "debug"
    plot_inputs_torino = out_dir / "plot_inputs" / "torino"
    plot_inputs_marrakesh = out_dir / "plot_inputs" / "marrakesh"
    for folder in [
        figures_paper,
        figures_debug,
        tables_paper,
        tables_debug,
        plot_inputs_torino,
        plot_inputs_marrakesh,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    backend_hint = _normalize_backend_name(backend_label) if str(backend_label).strip() else ""

    if panel_paths:
        for panel in panel_paths:
            _link_or_copy(panel, figures_debug / panel.name)
            panel_name = panel.name.lower()
            if "depth_cnot" in panel_name:
                _copy_replace(panel, figures_paper / "panel_depth_cnot.pdf")
            elif "real_fidelity" in panel_name:
                _copy_replace(panel, figures_paper / "panel_real_fidelity.pdf")
            elif "sim_fidelity_timing" in panel_name:
                _copy_replace(panel, figures_paper / "panel_sim_fidelity_timing.pdf")
                hint = backend_hint
                if not hint:
                    if "marrakesh" in panel_name:
                        hint = "marrakesh"
                    elif "torino" in panel_name:
                        hint = "torino"
                if hint in {"torino", "marrakesh"}:
                    _copy_replace(panel, figures_paper / f"panel_sim_fidelity_timing_{hint}.pdf")

    if breakdown_paths:
        for path in breakdown_paths:
            if path.suffix.lower() != ".pdf":
                continue
            _link_or_copy(path, figures_debug / path.name)
            lower = path.name.lower()
            if lower.startswith("qose_time_breakdown_"):
                _copy_replace(path, figures_paper / "qose_time_breakdown.pdf")
            elif lower.startswith("mitigation_stage_breakdown_qos_qose_"):
                _copy_replace(path, figures_paper / "mitigation_stage_breakdown_qos_qose.pdf")
            elif lower.startswith("real_jobs_avg_per_bench_"):
                _copy_replace(path, figures_paper / "panel_avg_jobs_per_bench.pdf")

    if final_csv_path is not None and final_csv_path.exists():
        _link_or_copy(final_csv_path, tables_debug / final_csv_path.name)
        _copy_replace(final_csv_path, tables_paper / "relative_properties_latest.csv")
        if backend_hint in {"torino", "marrakesh"}:
            if backend_hint == "torino":
                _link_or_copy(final_csv_path, plot_inputs_torino / "relative_properties_latest.csv")
            else:
                _link_or_copy(final_csv_path, plot_inputs_marrakesh / "relative_properties_latest.csv")
        else:
            lower = final_csv_path.name.lower()
            if "torino" in lower:
                _link_or_copy(final_csv_path, plot_inputs_torino / "relative_properties_latest.csv")
            if "marrakesh" in lower:
                _link_or_copy(final_csv_path, plot_inputs_marrakesh / "relative_properties_latest.csv")

    if timing_path is not None and timing_path.exists():
        _link_or_copy(timing_path, tables_debug / timing_path.name)
        _copy_replace(timing_path, tables_paper / "timing_latest.csv")
        if backend_hint in {"torino", "marrakesh"}:
            if backend_hint == "torino":
                _link_or_copy(timing_path, plot_inputs_torino / "timing_latest.csv")
            else:
                _link_or_copy(timing_path, plot_inputs_marrakesh / "timing_latest.csv")
        else:
            lower = timing_path.name.lower()
            if "torino" in lower:
                _link_or_copy(timing_path, plot_inputs_torino / "timing_latest.csv")
            if "marrakesh" in lower:
                _link_or_copy(timing_path, plot_inputs_marrakesh / "timing_latest.csv")


def _resume_signature(args, benches, sizes, methods: List[str]) -> Dict[str, object]:
    return {
        "sizes": [int(s) for s in sizes],
        "benches": [b for b, _ in benches],
        "methods": list(methods),
        "with_fidelity": bool(args.with_fidelity),
        "with_real_fidelity": bool(args.with_real_fidelity),
        "real_backend": _signature_real_backend(args),
        "real_fidelity_shots": int(args.real_fidelity_shots),
        "metric_mode": str(args.metric_mode),
        "metrics_baseline": str(args.metrics_baseline),
        "metrics_optimization_level": int(args.metrics_optimization_level),
        "cutqc_method": str(args.cutqc_method),
        "budget": int(args.budget),
        "size_to_reach": int(args.size_to_reach),
        "ideal_size_to_reach": int(args.ideal_size_to_reach),
    }


def _resume_signature_compatible(
    saved_signature: Optional[Dict[str, object]],
    requested_signature: Dict[str, object],
) -> bool:
    if not isinstance(saved_signature, dict):
        return False
    # Allow method superset resume: previously computed methods can be reused
    # when current run requests additional methods (e.g., add QOSE later).
    saved_with_real = bool(saved_signature.get("with_real_fidelity", False))
    req_with_real = bool(requested_signature.get("with_real_fidelity", False))
    for key, req_val in requested_signature.items():
        if key == "methods":
            continue
        if key in {"with_fidelity", "with_real_fidelity"}:
            # These can move in either direction; partial row reuse handles missing fields.
            continue
        if key == "real_backend" and saved_with_real and req_with_real:
            saved_backend = _normalize_backend_name(str(saved_signature.get(key, "")))
            req_backend = _normalize_backend_name(str(req_val))
            if saved_backend != req_backend:
                return False
            continue
        if key in {"real_backend", "real_fidelity_shots"} and (not saved_with_real or not req_with_real):
            # Real-fidelity settings are irrelevant unless both signatures include real fidelity.
            continue
        if saved_signature.get(key) != req_val:
            return False
    saved_methods = set(saved_signature.get("methods", []) or [])
    req_methods = set(requested_signature.get("methods", []) or [])
    # Allow both expansions (add methods later) and narrowed follow-up runs
    # (e.g., run only QOSE simulated fidelity after a full real-only run).
    return saved_methods.issubset(req_methods) or req_methods.issubset(saved_methods)


def _load_resume_state(path: Path, signature: Dict[str, object]) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if int(data.get("version", 0)) != _RESUME_STATE_VERSION:
        return None
    if not _resume_signature_compatible(data.get("signature"), signature):
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


def _parse_real_fidelity_compute_methods(value: str) -> Optional[set[str]]:
    raw = value.strip()
    if not raw or raw.lower() in {"all", "default"}:
        return None
    requested: set[str] = set()
    for entry in raw.split(","):
        key = entry.strip().lower()
        if not key:
            continue
        if key in {"qiskit", "baseline"}:
            requested.add("Qiskit")
            continue
        method = METHOD_ALIASES.get(key)
        if method is None:
            raise ValueError(f"Unknown method in --real-fidelity-compute-methods: {entry}")
        requested.add(method)
    return requested


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


def _simulate_counts(
    circuit: QuantumCircuit,
    shots: int,
    noise,
    seed: int,
    ctx: Optional[Dict[str, int]] = None,
    method_ctx: Optional[Dict[str, int | str]] = None,
    sim_kind: str = "sim",
) -> Dict[str, int]:
    AerSimulator, _, _, _ = _import_aer()
    sim = AerSimulator(noise_model=noise) if noise else AerSimulator()
    circ = _ensure_measurements(circuit)
    show_progress = bool(method_ctx) and _is_real_sim_progress_enabled()
    t0 = time.perf_counter()
    if show_progress:
        _EVAL_LOGGER.warning(
            "Ideal sim start method=%s bench=%s size=%s kind=%s circuit=%s/%s shots=%s",
            method_ctx.get("method"),
            method_ctx.get("bench"),
            method_ctx.get("size"),
            sim_kind,
            (ctx or {}).get("circuit_idx", 1),
            (ctx or {}).get("circuit_total", 1),
            shots,
        )
    result = sim.run(circ, shots=shots, seed_simulator=seed, seed_transpiler=seed).result()
    if show_progress:
        _EVAL_LOGGER.warning(
            "Ideal sim done method=%s bench=%s size=%s kind=%s circuit=%s/%s elapsed_sec=%.2f",
            method_ctx.get("method"),
            method_ctx.get("bench"),
            method_ctx.get("size"),
            sim_kind,
            (ctx or {}).get("circuit_idx", 1),
            (ctx or {}).get("circuit_total", 1),
            time.perf_counter() - t0,
        )
    return result.get_counts()


def _simulate_virtual_counts(
    circuit: QuantumCircuit | VirtualCircuit,
    shots: int,
    noise,
    seed: int,
    ctx: Optional[Dict[str, int]] = None,
    method_ctx: Optional[Dict[str, int | str]] = None,
) -> Dict[str, int]:
    vc = circuit if isinstance(circuit, VirtualCircuit) else VirtualCircuit(circuit)
    results: Dict = {}
    frag_work = []
    total_inst = 0
    for frag, frag_circ in vc.fragment_circuits.items():
        inst_labels = vc.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circ, inst_labels)
        frag_work.append((frag, instantiations))
        total_inst += len(instantiations)

    show_progress = bool(method_ctx) and _is_real_sim_progress_enabled()
    if show_progress:
        _EVAL_LOGGER.warning(
            "Ideal sim start method=%s bench=%s size=%s circuit=%s/%s fragments=%s inst_total=%s shots=%s",
            method_ctx.get("method"),
            method_ctx.get("bench"),
            method_ctx.get("size"),
            (ctx or {}).get("circuit_idx", 1),
            (ctx or {}).get("circuit_total", 1),
            len(frag_work),
            total_inst,
            shots,
        )

    global_done = 0
    t_sim_start = time.perf_counter()
    for frag_idx, (frag, instantiations) in enumerate(frag_work, start=1):
        distrs = []
        inst_total = len(instantiations)
        progress_step = max(1, inst_total // 10)
        t_frag_start = time.perf_counter()
        for inst_idx, inst in enumerate(instantiations, start=1):
            counts = _simulate_counts(inst, shots, noise, seed)
            distrs.append(QuasiDistr.from_counts(counts))
            global_done += 1
            if show_progress and (inst_idx == 1 or inst_idx == inst_total or inst_idx % progress_step == 0):
                _EVAL_LOGGER.warning(
                    "Ideal sim progress method=%s bench=%s size=%s fragment=%s/%s inst=%s/%s global=%s/%s",
                    method_ctx.get("method"),
                    method_ctx.get("bench"),
                    method_ctx.get("size"),
                    frag_idx,
                    len(frag_work),
                    inst_idx,
                    inst_total,
                    global_done,
                    total_inst,
                )
        results[frag] = distrs
        if show_progress:
            _EVAL_LOGGER.warning(
                "Ideal sim fragment done method=%s bench=%s size=%s fragment=%s/%s elapsed_sec=%.2f",
                method_ctx.get("method"),
                method_ctx.get("bench"),
                method_ctx.get("size"),
                frag_idx,
                len(frag_work),
                time.perf_counter() - t_frag_start,
            )
    if show_progress:
        _EVAL_LOGGER.warning(
            "Ideal sim knit start method=%s bench=%s size=%s",
            method_ctx.get("method"),
            method_ctx.get("bench"),
            method_ctx.get("size"),
        )
    t_knit_start = time.perf_counter()
    quasi = vc.knit(results, SerialPool())
    if show_progress:
        _EVAL_LOGGER.warning(
            "Ideal sim done method=%s bench=%s size=%s sim_elapsed_sec=%.2f knit_elapsed_sec=%.2f",
            method_ctx.get("method"),
            method_ctx.get("bench"),
            method_ctx.get("size"),
            time.perf_counter() - t_sim_start,
            time.perf_counter() - t_knit_start,
        )
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


def _ideal_counts_key(
    shots: int,
    seed: int,
    circuit: QuantumCircuit,
    ctx: Dict[str, int],
    method_ctx: Dict[str, int | str],
) -> str:
    return (
        f"ideal|shots={shots}|seed={seed}|method={method_ctx.get('method')}|"
        f"bench={method_ctx.get('bench')}|size={method_ctx.get('size')}|"
        f"c={ctx.get('circuit_idx',1)}|virtual={int(_has_virtual_ops(circuit))}"
    )


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
        global _RUN_QPU_SEC, _RUN_QPU_SEC_KNOWN_JOBS, _RUN_QPU_SEC_UNKNOWN_JOBS
        if qpu_sec is not None:
            _RUN_QPU_SEC += float(qpu_sec)
            _RUN_QPU_SEC_KNOWN_JOBS += 1
        else:
            _RUN_QPU_SEC_UNKNOWN_JOBS += 1
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
                    "elapsed_sec": elapsed,
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
    global _IDEAL_RESULT_CACHE
    if _REAL_DRY_RUN:
        _dry_run_traverse(circuit, ctx, method_ctx)
        return 0.0

    # Submit/collect real counts first so resumed real-only runs start submitting
    # immediately; ideal simulation can be cached/recomputed afterward.
    if _has_virtual_ops(circuit):
        noisy = _real_virtual_counts(circuit, shots, backend_name, ctx, method_ctx)
    else:
        noisy = _real_counts(circuit, shots, backend_name, ctx, method_ctx)

    ideal_key = _ideal_counts_key(shots, seed, circuit, ctx, method_ctx)
    ideal = None
    if _IDEAL_RESULT_CACHE is not None:
        ideal = _IDEAL_RESULT_CACHE.get(ideal_key)
        if ideal is not None:
            _EVAL_LOGGER.warning(
                "Ideal sim cache-hit method=%s bench=%s size=%s circuit=%s/%s",
                method_ctx.get("method"),
                method_ctx.get("bench"),
                method_ctx.get("size"),
                ctx.get("circuit_idx", 1),
                ctx.get("circuit_total", 1),
            )
    if ideal is None and _has_virtual_ops(circuit):
        ideal = _simulate_virtual_counts(circuit, shots, None, seed, ctx=ctx, method_ctx=method_ctx)
    elif ideal is None:
        ideal = _simulate_counts(
            circuit,
            shots,
            None,
            seed,
            ctx=ctx,
            method_ctx=method_ctx,
            sim_kind="ideal",
        )
    if _IDEAL_RESULT_CACHE is not None and ideal is not None:
        _IDEAL_RESULT_CACHE.put(
            ideal_key,
            ideal,
            {
                "kind": "ideal_counts",
                "shots": shots,
                "seed": seed,
                "method": method_ctx.get("method"),
                "bench": method_ctx.get("bench"),
                "size": method_ctx.get("size"),
                "circuit_idx": ctx.get("circuit_idx", 1),
                "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
            },
        )
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
    if _REAL_JOB_RESULT_CACHE is not None:
        cache_prefix = f"{backend_name}|{shots}|{method}|{bench}|{size}|"
        cached_jobs = int(_REAL_JOB_RESULT_CACHE.count_prefix(cache_prefix))
        if cached_jobs > 0:
            _EVAL_LOGGER.warning(
                "Real QPU cache resume method=%s bench=%s size=%s cached_jobs=%s expected_total=%s",
                method,
                bench,
                size,
                cached_jobs,
                jobs_total,
            )
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
    ctx: Optional[Dict[str, int]] = None,
    method_ctx: Optional[Dict[str, int | str]] = None,
) -> float:
    if _has_virtual_ops(circuit):
        ideal = _simulate_virtual_counts(circuit, shots, None, seed, ctx=ctx, method_ctx=method_ctx)
        noisy = _simulate_virtual_counts(circuit, shots, noise, seed, ctx=ctx, method_ctx=method_ctx)
    else:
        ideal = _simulate_counts(
            circuit,
            shots,
            None,
            seed,
            ctx=ctx,
            method_ctx=method_ctx,
            sim_kind="ideal",
        )
        noisy = _simulate_counts(
            circuit,
            shots,
            noise,
            seed,
            ctx=ctx,
            method_ctx=method_ctx,
            sim_kind="noisy",
        )
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
    method: Optional[str] = None,
    bench: Optional[str] = None,
    size: Optional[int] = None,
) -> Tuple[float, float]:
    total = len(circuits)
    method_ctx: Optional[Dict[str, int | str]] = None
    if method is not None and bench is not None and size is not None:
        method_ctx = {
            "method": method,
            "bench": bench,
            "size": int(size),
        }
        _EVAL_LOGGER.warning(
            "Ideal sim jobs method=%s bench=%s size=%s circuits_total=%s",
            method,
            bench,
            size,
            total,
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
            _fidelity_for_circuit(
                circuit,
                shots,
                noise,
                seed,
                ctx=ctx,
                method_ctx=method_ctx,
            )
        )
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
        # Keep timing comparable between QOS and QOSE by default.
        # Only enable counting/instrumented mitigator when explicit cost-search logging is requested.
        if methods == [] and use_cost_search and getattr(args, "cost_search_log", "").strip():
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
            start = time.perf_counter() if args.collect_timing else 0.0
            q = mitigator.run(q)
            if args.collect_timing:
                mitigator.timings["total"] = time.perf_counter() - start
            if args.collect_timing and hasattr(mitigator, "cost_search_calls"):
                mitigator.timings["cost_search_calls"] = mitigator.cost_search_calls
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
        except ValueError:
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
) -> Tuple[Dict[str, int], Dict[str, float], List[QuantumCircuit]]:
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
        start = time.perf_counter() if args.collect_timing else 0.0
        q = evolved_run(mitigator, q)
        if args.collect_timing:
            mitigator.timings["total"] = time.perf_counter() - start
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
    except ValueError:
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
    hatch_patterns = ["///", "\\\\\\", "...", "xxx", "+++", "---", "|||", "ooo", "***"]
    for i, (bench, label) in enumerate(benches):
        vals = [rel_data[bench].get(m, np.nan) for m in methods]
        all_vals.extend([v for v in vals if np.isfinite(v)])
        errs = None
        if err_data is not None:
            errs = [err_data.get(bench, {}).get(m, 0.0) for m in methods]
        hatch = hatch_patterns[i % len(hatch_patterns)]
        if errs and any(e > 0 for e in errs):
            ax.bar(
                x + (i - len(benches) / 2) * width,
                vals,
                width,
                label=label,
                yerr=errs,
                capsize=2,
                ecolor="black",
                linewidth=0.6,
                edgecolor="black",
                hatch=hatch,
            )
        else:
            ax.bar(
                x + (i - len(benches) / 2) * width,
                vals,
                width,
                label=label,
                linewidth=0.6,
                edgecolor="black",
                hatch=hatch,
            )

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
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="y", labelsize=11)
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


def _read_rows_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _has_value(val: object) -> bool:
    if val in {"", None}:
        return False
    if isinstance(val, str):
        return val.strip().lower() not in {"", "nan", "none", "null"}
    try:
        return bool(np.isfinite(float(val)))
    except Exception:
        return True


def _row_key_from_fields(row: Dict[str, object], key_fields: List[str]) -> Optional[Tuple[str, ...]]:
    key_parts: List[str] = []
    for field in key_fields:
        val = row.get(field, "")
        if not _has_value(val):
            return None
        if field == "size":
            key_parts.append(str(_safe_int(val, -1)))
        else:
            key_parts.append(str(val))
    return tuple(key_parts)


def _backend_hint_from_path(path: Path) -> str:
    name = path.name.lower()
    if "marrakesh" in name:
        return "marrakesh"
    if "torino" in name:
        return "torino"
    return "unknown"


def _collect_fallback_csvs(
    primary_csv: Path,
    *,
    kind: str,
    backend: Optional[str] = None,
) -> List[Path]:
    pattern = "relative_properties*.csv" if kind == "relative" else "timing*.csv"
    candidates: List[Path] = []
    seen: set[Path] = set()

    def _add_path(path: Path) -> None:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved == primary_csv.resolve():
            return
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    # Prefer local artifacts first.
    for p in sorted(primary_csv.parent.glob(pattern)):
        _add_path(p)

    # Then include saved collections.
    collections_root = ROOT / "evaluation" / "plots" / "full_eval_collections"
    if collections_root.exists():
        for p in sorted(collections_root.rglob(pattern)):
            _add_path(p)

    filtered: List[Path] = []
    for p in candidates:
        if backend == "torino":
            # Torino fallback can use explicit Torino files and unlabeled legacy files.
            if _backend_hint_from_path(p) == "marrakesh":
                continue
        elif backend == "marrakesh":
            # Marrakesh fallback should only use explicitly labeled Marrakesh files.
            if _backend_hint_from_path(p) != "marrakesh":
                continue
        filtered.append(p)

    # Newest first gives best chance of matching current schema.
    filtered.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return filtered


def _merge_rows_from_fallback(
    primary_rows: List[Dict[str, object]],
    fallback_rows: List[Dict[str, object]],
    *,
    key_fields: List[str],
) -> Tuple[List[Dict[str, object]], int, int]:
    merged = [dict(r) for r in primary_rows]
    row_idx: Dict[Tuple[str, ...], int] = {}
    for idx, row in enumerate(merged):
        key = _row_key_from_fields(row, key_fields)
        if key is not None:
            row_idx[key] = idx

    filled_cells = 0
    added_rows = 0
    for frow in fallback_rows:
        key = _row_key_from_fields(frow, key_fields)
        if key is None:
            continue
        if key in row_idx:
            target = merged[row_idx[key]]
            for col, val in frow.items():
                if col in key_fields:
                    continue
                if _has_value(val) and not _has_value(target.get(col)):
                    target[col] = val
                    filled_cells += 1
        else:
            merged.append(dict(frow))
            row_idx[key] = len(merged) - 1
            added_rows += 1

    return merged, filled_cells, added_rows


def _auto_fill_rows_from_csv_fallbacks(
    primary_rows: List[Dict[str, object]],
    primary_csv: Path,
    *,
    kind: str,
    key_fields: List[str],
    backend: Optional[str] = None,
    sizes: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, object]], int, int]:
    merged = [dict(r) for r in primary_rows]
    fallback_paths = _collect_fallback_csvs(primary_csv, kind=kind, backend=backend)
    used_files = 0
    total_filled = 0

    size_filter = set(int(s) for s in sizes or [])
    method_filter = set(methods or [])
    for path in fallback_paths:
        try:
            rows = _read_rows_csv(path)
        except Exception:
            continue

        filtered_rows: List[Dict[str, object]] = []
        for row in rows:
            if size_filter:
                row_size = _safe_int(row.get("size", ""), -1)
                if row_size not in size_filter:
                    continue
            if method_filter and "method" in key_fields:
                row_method = str(row.get("method", ""))
                if row_method not in method_filter:
                    continue
            filtered_rows.append(row)

        if not filtered_rows:
            continue

        merged, filled_cells, added_rows = _merge_rows_from_fallback(
            merged, filtered_rows, key_fields=key_fields
        )
        if filled_cells > 0 or added_rows > 0:
            used_files += 1
            total_filled += filled_cells

    return merged, used_files, total_filled


def _plot_timing_total_panel(
    ax,
    timing_rows: List[Dict[str, object]],
    size: int,
    methods: List[str],
    title: str,
) -> None:
    skip_stages = {"bench", "size", "method", "total", "overall", "simulation", "cost_search_calls"}

    def _row_total(row: Dict[str, object]) -> float:
        if "total" in row:
            try:
                return float(row["total"])
            except (TypeError, ValueError):
                return 0.0
        total = 0.0
        for key, value in row.items():
            if key in skip_stages:
                continue
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue
        return total

    x = np.arange(len(methods))
    totals: List[float] = []
    hatch_patterns = ["///", "\\\\\\", "...", "xxx", "+++", "---", "|||", "ooo", "***"]
    for method in methods:
        method_rows = [r for r in timing_rows if _safe_int(r.get("size", 0), -1) == size and r.get("method") == method]
        vals = [_row_total(r) for r in method_rows]
        totals.append(sum(vals) / max(1, len(vals)))
    for idx, total in enumerate(totals):
        ax.bar(
            x[idx],
            total,
            hatch=hatch_patterns[idx % len(hatch_patterns)],
            edgecolor="black",
            linewidth=0.6,
        )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Seconds", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for idx, total in enumerate(totals):
        ax.text(x[idx], total, f"{total:.2f}s", ha="center", va="bottom", fontsize=9)


def _reorder_methods_for_panels(methods: List[str]) -> List[str]:
    # Keep panel order consistent with evaluation comparison preference:
    # FrozenQubits, CutQC, QOS, QOSE (then any other methods).
    priority = ["FrozenQubits", "CutQC", "QOS", "QOSE", "QOSN", "qwen", "gemini", "gpt"]
    ordered: List[str] = []
    seen = set()
    for method in priority:
        if method in methods and method not in seen:
            ordered.append(method)
            seen.add(method)
    for method in methods:
        if method not in seen:
            ordered.append(method)
            seen.add(method)
    return ordered


def _plot_cached_panels(
    *,
    simtiming_csv: Path,
    real_torino_csv: Path,
    real_marrakesh_csv: Path,
    timing_csv: Path,
    out_dir: Path,
    timestamp: str,
    tag: str,
    sizes: List[int],
    methods: List[str],
) -> List[Path]:
    plt = _import_matplotlib()
    benches = BENCHES
    methods = _reorder_methods_for_panels([m for m in methods if m != "Qiskit"])
    fidelity_methods = ["Qiskit"] + methods
    tag_suffix = f"_{tag}" if tag else ""
    panel_tag = f"{timestamp}{tag_suffix}"

    sim_rows = _read_rows_csv(simtiming_csv)
    torino_rows = _read_rows_csv(real_torino_csv)
    marrakesh_rows = _read_rows_csv(real_marrakesh_csv)
    timing_rows = _read_rows_csv(timing_csv)

    # Auto-fill missing cached values from prior runs so users do not need
    # manual CSV merges when combining reused baselines with new QOSE results.
    sim_rows, sim_used, sim_filled = _auto_fill_rows_from_csv_fallbacks(
        sim_rows,
        simtiming_csv,
        kind="relative",
        key_fields=["size", "bench"],
        sizes=sizes,
    )
    torino_rows, tor_used, tor_filled = _auto_fill_rows_from_csv_fallbacks(
        torino_rows,
        real_torino_csv,
        kind="relative",
        key_fields=["size", "bench"],
        backend="torino",
        sizes=sizes,
    )
    marrakesh_rows, mar_used, mar_filled = _auto_fill_rows_from_csv_fallbacks(
        marrakesh_rows,
        real_marrakesh_csv,
        kind="relative",
        key_fields=["size", "bench"],
        backend="marrakesh",
        sizes=sizes,
    )
    timing_rows, timing_used, timing_filled = _auto_fill_rows_from_csv_fallbacks(
        timing_rows,
        timing_csv,
        kind="timing",
        key_fields=["size", "bench", "method"],
        sizes=sizes,
        methods=methods,
    )
    if any(x > 0 for x in [sim_used, tor_used, mar_used, timing_used]):
        print(
            "[panel-cache] auto-fill "
            f"sim(files={sim_used},cells={sim_filled}) "
            f"torino(files={tor_used},cells={tor_filled}) "
            f"marrakesh(files={mar_used},cells={mar_filled}) "
            f"timing(files={timing_used},cells={timing_filled})",
            flush=True,
        )

    sim_rel_by_size, sim_fid_by_size, sim_fid_err_by_size, _, _ = _build_progress_maps_from_rows(
        sim_rows,
        benches,
        sizes,
        methods,
        with_fidelity=True,
        with_real_fidelity=False,
    )
    _, _, _, tor_real_by_size, tor_real_err_by_size = _build_progress_maps_from_rows(
        torino_rows,
        benches,
        sizes,
        methods,
        with_fidelity=False,
        with_real_fidelity=True,
    )
    _, _, _, mar_real_by_size, mar_real_err_by_size = _build_progress_maps_from_rows(
        marrakesh_rows,
        benches,
        sizes,
        methods,
        with_fidelity=False,
        with_real_fidelity=True,
    )

    timing_methods = [m for m in methods if any(_safe_int(r.get("size", 0), -1) in sizes and r.get("method") == m for r in timing_rows)]
    if not timing_methods:
        raise RuntimeError(
            f"No timing rows for requested methods={methods} in {timing_csv}."
        )

    out_paths: List[Path] = []

    # Figure 1: depth / CNOT in one row (12-depth, 12-cnot, 24-depth, 24-cnot).
    fig, axes = plt.subplots(1, len(sizes) * 2, figsize=(6.2 * len(sizes) * 2, 4.0))
    axes = np.array(axes).reshape(1, len(sizes) * 2)
    for idx, size in enumerate(sizes):
        rel_depth, rel_nonlocal = sim_rel_by_size[size]
        _plot_panel(
            axes[0, idx * 2],
            f"Depth - {size} qubits (lower is better)",
            rel_depth,
            benches,
            methods,
            "Relative to Qiskit",
            show_avg=True,
        )
        _plot_panel(
            axes[0, idx * 2 + 1],
            f"Number of CNOT gates - {size} qubits (lower is better)",
            rel_nonlocal,
            benches,
            methods,
            "Relative to Qiskit",
            show_avg=True,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=max(1, len(labels)), fontsize=11, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    depth_cnot_path = out_dir / f"relative_properties_panels_depth_cnot_{panel_tag}.pdf"
    fig.savefig(depth_cnot_path)
    plt.close(fig)
    out_paths.append(depth_cnot_path)

    # Figure 2: real fidelity Torino / Marrakesh in one row.
    fig, axes = plt.subplots(1, len(sizes) * 2, figsize=(6.2 * len(sizes) * 2, 4.0))
    axes = np.array(axes).reshape(1, len(sizes) * 2)
    for idx, size in enumerate(sizes):
        _plot_panel(
            axes[0, idx * 2],
            f"Real Hellinger fidelity - {size} qubits (Torino)",
            tor_real_by_size.get(size, {}),
            benches,
            fidelity_methods,
            "Fidelity",
            err_data=(tor_real_err_by_size or {}).get(size, {}),
            show_avg=True,
        )
        _plot_panel(
            axes[0, idx * 2 + 1],
            f"Real Hellinger fidelity - {size} qubits (Marrakesh)",
            mar_real_by_size.get(size, {}),
            benches,
            fidelity_methods,
            "Fidelity",
            err_data=(mar_real_err_by_size or {}).get(size, {}),
            show_avg=True,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=max(1, len(labels)), fontsize=11, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    real_path = out_dir / f"relative_properties_panels_real_fidelity_{panel_tag}.pdf"
    fig.savefig(real_path)
    plt.close(fig)
    out_paths.append(real_path)

    # Figure 3: sim fidelity + timing in one row.
    fig, axes = plt.subplots(1, len(sizes) * 2, figsize=(6.2 * len(sizes) * 2, 4.0))
    axes = np.array(axes).reshape(1, len(sizes) * 2)
    for idx, size in enumerate(sizes):
        _plot_panel(
            axes[0, idx * 2],
            f"Hellinger fidelity - {size} qubits (higher is better)",
            (sim_fid_by_size or {}).get(size, {}),
            benches,
            fidelity_methods,
            "Fidelity",
            err_data=(sim_fid_err_by_size or {}).get(size, {}),
            show_avg=True,
        )
        _plot_timing_total_panel(
            axes[0, idx * 2 + 1],
            timing_rows,
            size,
            timing_methods,
            f"Total Timing (Avg) - {size} qubits",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=max(1, len(labels)), fontsize=11, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    sim_timing_path = out_dir / f"relative_properties_panels_sim_fidelity_timing_{panel_tag}.pdf"
    fig.savefig(sim_timing_path)
    plt.close(fig)
    out_paths.append(sim_timing_path)

    return out_paths


def _timing_row_total(row: Dict[str, object]) -> float:
    skip_stages = {"bench", "size", "method", "total", "overall", "simulation", "cost_search_calls"}
    if _has_value(row.get("total")):
        try:
            return float(row.get("total", 0.0))
        except Exception:
            pass
    total = 0.0
    for key, value in row.items():
        if key in skip_stages:
            continue
        try:
            total += float(value)
        except Exception:
            continue
    return total


def _read_jsonl_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _latest_file(path_glob: List[Path]) -> Optional[Path]:
    if not path_glob:
        return None
    return sorted(path_glob, key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)[0]


def _strict_dry_run_real_job_counts(
    *,
    args,
    sizes: List[int],
    benches: List[str],
    methods: List[str],
    backend_hint: str,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    bench_set = set(benches)
    eval_benches = [(b, label) for b, label in BENCHES if b in bench_set]
    if not eval_benches:
        raise RuntimeError("No benches selected for strict dry-run job counting.")

    requested_methods = [m for m in methods if m and m != "Qiskit"]
    # Keep only methods supported by evaluator path.
    requested_methods = [
        m
        for m in requested_methods
        if m in {"FrozenQubits", "CutQC", "QOS", "QOSN", "QOSE", "qwen", "gemini", "gpt"}
    ]

    dry_args = argparse.Namespace(**vars(args))
    dry_args.with_fidelity = False
    dry_args.with_real_fidelity = False
    dry_args.collect_timing = False
    dry_args.fragment_fidelity_sweep = False
    dry_args.cut_visualization = False
    # Align transpile/noise defaults with the target backend when possible.
    if backend_hint in {"torino", "marrakesh"}:
        dry_args.metrics_baseline = backend_hint

    qose_runs, _qose_programs = _load_qose_runs_from_args(dry_args)
    selected_methods: List[str] = []
    missing_qose: List[str] = []
    for method in requested_methods:
        if method in {"QOSE", "qwen", "gemini", "gpt"} and method not in qose_runs:
            missing_qose.append(method)
            continue
        selected_methods.append(method)

    if missing_qose:
        _EVAL_LOGGER.warning(
            "Strict dry-run job-count: missing QOSE program for methods=%s; skipping them.",
            ",".join(missing_qose),
        )

    global _REAL_DRY_RUN
    prev_dry_run = _REAL_DRY_RUN
    _REAL_DRY_RUN = True
    try:
        (
            _rows,
            _rel_by_size,
            _timing_rows,
            _fidelity_by_size,
            _fidelity_err_by_size,
            _real_fidelity_by_size,
            _real_fidelity_err_by_size,
            _cut_circuits,
            _fragment_fidelity,
            real_job_counts_by_size,
        ) = _run_eval(
            dry_args,
            eval_benches,
            sizes,
            qose_runs,
            selected_methods,
            initial_rows=[],
            initial_timing_rows=[],
            completed_keys=set(),
            method_cache_dir=None,
            on_bench_complete=None,
        )
    finally:
        _REAL_DRY_RUN = prev_dry_run

    # Convert from size->bench->method to size->method->bench for plotting helper.
    converted: Dict[int, Dict[str, Dict[str, float]]] = {int(s): {} for s in sizes}
    for size in sizes:
        bench_map = real_job_counts_by_size.get(int(size), {})
        for bench, by_method in bench_map.items():
            for method, val in by_method.items():
                converted[int(size)].setdefault(str(method), {})[str(bench)] = float(val)
    return converted


def _plot_time_breakdowns(
    *,
    timing_csv: Path,
    job_cache_jsonl: Path,
    out_dir: Path,
    timestamp: str,
    tag: str,
    sizes: List[int],
    methods: List[str],
    benches: List[str],
    secondary_timing_csv: Optional[Path] = None,
    secondary_job_cache_jsonl: Optional[Path] = None,
    primary_label: str = "",
    secondary_label: str = "",
    strict_job_counts_primary: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
    strict_job_counts_secondary: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
    job_methods_override: Optional[List[str]] = None,
) -> List[Path]:
    plt = _import_matplotlib()
    tag_suffix = f"_{tag}" if tag else ""
    panel_tag = f"{timestamp}{tag_suffix}"

    timing_rows = _read_rows_csv(timing_csv)
    timing_rows, timing_used, timing_filled = _auto_fill_rows_from_csv_fallbacks(
        timing_rows,
        timing_csv,
        kind="timing",
        key_fields=["size", "bench", "method"],
        sizes=sizes,
        methods=methods,
    )
    if timing_used > 0:
        print(
            f"[time-breakdown] timing auto-fill files={timing_used} cells={timing_filled}",
            flush=True,
        )
    job_rows = _read_jsonl_rows(job_cache_jsonl)
    primary_job_keys = {str(r.get("job_key", "")) for r in job_rows if str(r.get("job_key", ""))}
    fallback_job_files = [
        p
        for p in sorted(
            out_dir.rglob("*job_cache*.jsonl"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
        if p != job_cache_jsonl
    ]
    fallback_jobs_added = 0
    fallback_files_used = 0
    for fp in fallback_job_files:
        try:
            rows = _read_jsonl_rows(fp)
        except Exception:
            continue
        added_this_file = 0
        for rec in rows:
            key = str(rec.get("job_key", ""))
            if not key or key in primary_job_keys:
                continue
            primary_job_keys.add(key)
            job_rows.append(rec)
            added_this_file += 1
        if added_this_file > 0:
            fallback_files_used += 1
            fallback_jobs_added += added_this_file
    if fallback_files_used > 0:
        print(
            f"[time-breakdown] job-cache auto-fill files={fallback_files_used} jobs={fallback_jobs_added}",
            flush=True,
        )
    size_set = {int(s) for s in sizes}
    bench_set = set(benches)
    mitigation_methods = [m for m in methods if m in {"QOS", "QOSE"}]
    if not mitigation_methods:
        raise RuntimeError("No supported methods for breakdown (expected QOS and/or QOSE).")

    def _collect_qose_breakdown(
        curr_timing_rows: List[Dict[str, object]],
        curr_job_rows: List[Dict[str, object]],
    ) -> Dict[str, object]:
        qose_by_size: Dict[int, Dict[str, float]] = {
            int(s): {
                "mitigation_sec_total": 0.0,
                "real_elapsed_sec_total": 0.0,
                "real_qpu_sec_total": 0.0,
                "real_wait_sec_total": 0.0,
                "real_jobs": 0.0,
                "real_jobs_qpu_unknown": 0.0,
            }
            for s in sizes
        }
        mitigation_by_bench: Dict[int, Dict[str, float]] = {int(s): {} for s in sizes}
        real_elapsed_by_bench: Dict[int, Dict[str, float]] = {int(s): {} for s in sizes}
        real_qpu_by_bench: Dict[int, Dict[str, float]] = {int(s): {} for s in sizes}
        real_wait_by_bench: Dict[int, Dict[str, float]] = {int(s): {} for s in sizes}
        elapsed_samples: List[float] = []
        qpu_samples: List[float] = []

        for row in curr_timing_rows:
            method = str(row.get("method", ""))
            size = _safe_int(row.get("size", ""), -1)
            bench = str(row.get("bench", ""))
            if method != "QOSE" or size not in size_set or bench not in bench_set:
                continue
            mitigation_sec = max(_timing_row_total(row), 0.0)
            qose_by_size[size]["mitigation_sec_total"] += mitigation_sec
            mitigation_by_bench[size][bench] = mitigation_by_bench[size].get(bench, 0.0) + mitigation_sec

        job_dedup: Dict[str, Dict[str, object]] = {}
        for rec in curr_job_rows:
            key = str(rec.get("job_key", ""))
            if key:
                job_dedup[key] = rec
        for rec in job_dedup.values():
            method = str(rec.get("method", ""))
            size = _safe_int(rec.get("size", ""), -1)
            bench = str(rec.get("bench", ""))
            if method != "QOSE" or size not in size_set or bench not in bench_set:
                continue
            elapsed = max(_safe_float(rec.get("elapsed_sec", ""), 0.0), 0.0)
            has_qpu = _has_value(rec.get("qpu_sec"))
            qpu_sec = max(_safe_float(rec.get("qpu_sec", ""), 0.0), 0.0) if has_qpu else 0.0
            wait_sec = max(elapsed - qpu_sec, 0.0) if has_qpu else elapsed
            qose_by_size[size]["real_elapsed_sec_total"] += elapsed
            qose_by_size[size]["real_qpu_sec_total"] += qpu_sec
            qose_by_size[size]["real_wait_sec_total"] += wait_sec
            qose_by_size[size]["real_jobs"] += 1.0
            real_elapsed_by_bench[size][bench] = real_elapsed_by_bench[size].get(bench, 0.0) + elapsed
            real_qpu_by_bench[size][bench] = real_qpu_by_bench[size].get(bench, 0.0) + qpu_sec
            real_wait_by_bench[size][bench] = real_wait_by_bench[size].get(bench, 0.0) + wait_sec
            elapsed_samples.append(elapsed)
            if has_qpu:
                qpu_samples.append(qpu_sec)
            if not has_qpu:
                qose_by_size[size]["real_jobs_qpu_unknown"] += 1.0

        mitigation_samples = []
        for s in sizes:
            mitigation_samples.extend(list(mitigation_by_bench[s].values()))

        return {
            "qose_by_size": qose_by_size,
            "mitigation_by_bench": mitigation_by_bench,
            "real_elapsed_by_bench": real_elapsed_by_bench,
            "real_qpu_by_bench": real_qpu_by_bench,
            "real_wait_by_bench": real_wait_by_bench,
            "elapsed_samples": elapsed_samples,
            "qpu_samples": qpu_samples,
            "mitigation_samples": mitigation_samples,
        }

    def _pretty_backend_label(raw: str, fallback: str) -> str:
        x = _normalize_backend_name(raw)
        if x == "generic":
            x = _normalize_backend_name(fallback)
        if x == "generic":
            return "Primary"
        return x.replace("_", " ").title()

    primary_stats = _collect_qose_breakdown(timing_rows, job_rows)
    primary_name = _pretty_backend_label(primary_label, str(timing_csv))

    secondary_stats: Optional[Dict[str, object]] = None
    secondary_name = ""
    if secondary_timing_csv is not None and secondary_job_cache_jsonl is not None:
        secondary_timing_rows = _read_rows_csv(secondary_timing_csv)
        secondary_timing_rows, sec_used, sec_filled = _auto_fill_rows_from_csv_fallbacks(
            secondary_timing_rows,
            secondary_timing_csv,
            kind="timing",
            key_fields=["size", "bench", "method"],
            sizes=sizes,
            methods=methods,
        )
        if sec_used > 0:
            print(
                f"[time-breakdown] secondary timing auto-fill files={sec_used} cells={sec_filled}",
                flush=True,
            )
        secondary_job_rows = _read_jsonl_rows(secondary_job_cache_jsonl)
        secondary_stats = _collect_qose_breakdown(secondary_timing_rows, secondary_job_rows)
        secondary_name = _pretty_backend_label(secondary_label, str(secondary_timing_csv))

    x_labels = [str(s) for s in sizes]

    def _draw_qose_panels(ax_bar, ax_dist, stats: Dict[str, object], title_prefix: str) -> None:
        qose_by_size = stats["qose_by_size"]
        mitigation_by_bench = stats["mitigation_by_bench"]
        real_wait_by_bench = stats["real_wait_by_bench"]
        real_qpu_by_bench = stats["real_qpu_by_bench"]

        components = [
            (
                "Mitigation avg/bench",
                [
                    float(np.mean(list(mitigation_by_bench[s].values())))
                    if mitigation_by_bench[s]
                    else 0.0
                    for s in sizes
                ],
                "#4C78A8",
            ),
            (
                "Real Wait avg/bench",
                [
                    float(np.mean(list(real_wait_by_bench[s].values())))
                    if real_wait_by_bench[s]
                    else 0.0
                    for s in sizes
                ],
                "#F58518",
            ),
            (
                "Real QPU avg/bench",
                [
                    float(np.mean(list(real_qpu_by_bench[s].values())))
                    if real_qpu_by_bench[s]
                    else 0.0
                    for s in sizes
                ],
                "#54A24B",
            ),
        ]

        x = np.arange(len(x_labels))
        width = 0.24
        for i, (name, vals, color) in enumerate(components):
            vals_np = np.array(vals, dtype=float)
            bars = ax_bar.bar(
                x + (i - 1) * width,
                vals_np,
                width=width,
                color=color,
                label=name,
                edgecolor="black",
                linewidth=0.6,
            )
            ymax = float(vals_np.max()) if len(vals_np) else 0.0
            offset = 0.01 * ymax if ymax > 0 else 0.0
            for bar, val in zip(bars, vals_np):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + offset,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax_bar.set_title(f"QOSE Breakdown ({title_prefix})", fontsize=13)
        ax_bar.set_xlabel("Qubit Size", fontsize=11)
        ax_bar.set_ylabel("Seconds", fontsize=11)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(x_labels, fontsize=10)
        ax_bar.tick_params(axis="y", labelsize=10)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.35)
        ax_bar.legend(fontsize=9, loc="upper left", frameon=True)

        dist_specs = [
            ("Per-job Elapsed", stats["elapsed_samples"], "#E45756"),
            ("Per-job QPU", stats["qpu_samples"], "#54A24B"),
            ("Per-bench Mitigation", stats["mitigation_samples"], "#4C78A8"),
        ]
        box_labels: List[str] = []
        box_values: List[List[float]] = []
        box_colors: List[str] = []
        for label, raw_vals, color in dist_specs:
            cleaned: List[float] = []
            for v in raw_vals:
                try:
                    x = float(v)
                except Exception:
                    continue
                if not np.isfinite(x) or x <= 0:
                    continue
                cleaned.append(x)
            if cleaned:
                box_labels.append(f"{label}\n(n={len(cleaned)})")
                box_values.append(cleaned)
                box_colors.append(color)

        if box_values:
            boxplot_kwargs = dict(
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"linewidth": 1.0},
                capprops={"linewidth": 1.0},
            )
            try:
                bp = ax_dist.boxplot(
                    box_values,
                    tick_labels=box_labels,
                    **boxplot_kwargs,
                )
            except TypeError:
                bp = ax_dist.boxplot(
                    box_values,
                    labels=box_labels,
                    **boxplot_kwargs,
                )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)
            # Label each box with its median value for quick comparison.
            for idx, vals in enumerate(box_values, start=1):
                med = float(np.median(np.array(vals, dtype=float)))
                ax_dist.text(
                    idx + 0.18,
                    med,
                    f"{med:.2f}",
                    ha="left",
                    va="center",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
                )
            ax_dist.set_yscale("log")
            ax_dist.tick_params(axis="x", labelsize=9, rotation=10)
            ax_dist.tick_params(axis="y", labelsize=10)
        else:
            ax_dist.text(0.5, 0.5, "No timing samples", ha="center", va="center", fontsize=10)
            ax_dist.set_xticks([])
            ax_dist.set_yticks([])

        ax_dist.set_title(f"Distribution ({title_prefix})", fontsize=13)
        ax_dist.set_ylabel("Seconds (log scale)", fontsize=11)
        ax_dist.grid(True, axis="y", linestyle="--", alpha=0.35)

        total_real_jobs = int(sum(v["real_jobs"] for v in qose_by_size.values()))
        total_unknown_qpu = int(sum(v["real_jobs_qpu_unknown"] for v in qose_by_size.values()))
        return total_real_jobs, total_unknown_qpu

    if secondary_stats is None:
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
        jobs, unknown = _draw_qose_panels(axes[0], axes[1], primary_stats, primary_name)
        fig.suptitle(
            f"QOSE Timing Diagnostics ({primary_name}; real jobs={jobs}, qpu_sec unknown={unknown})",
            fontsize=14,
            y=1.06,
        )
    else:
        fig, axes = plt.subplots(1, 4, figsize=(27.0, 5.0))
        jobs1, unknown1 = _draw_qose_panels(axes[0], axes[1], primary_stats, primary_name)
        jobs2, unknown2 = _draw_qose_panels(axes[2], axes[3], secondary_stats, secondary_name)
        fig.suptitle(
            (
                f"QOSE Timing Diagnostics "
                f"({primary_name}: jobs={jobs1}, unknown={unknown1}; "
                f"{secondary_name}: jobs={jobs2}, unknown={unknown2})"
            ),
            fontsize=14,
            y=1.06,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig1 = out_dir / f"qose_time_breakdown_{panel_tag}.pdf"
    fig.savefig(fig1)
    plt.close(fig)

    rows1: List[Dict[str, object]] = []
    datasets = [(primary_name, primary_stats)]
    if secondary_stats is not None:
        datasets.append((secondary_name, secondary_stats))
    for backend_name, stats in datasets:
        qose_by_size = stats["qose_by_size"]
        mitigation_by_bench = stats["mitigation_by_bench"]
        real_wait_by_bench = stats["real_wait_by_bench"]
        real_qpu_by_bench = stats["real_qpu_by_bench"]
        real_elapsed_by_bench = stats["real_elapsed_by_bench"]
        for s in sizes:
            row = {
                "backend": backend_name,
                "size": s,
                "mitigation_sec_total": qose_by_size[s]["mitigation_sec_total"],
                "mitigation_sec_avg_per_bench": float(np.mean(list(mitigation_by_bench[s].values())))
                if mitigation_by_bench[s]
                else 0.0,
                "real_wait_sec_total": qose_by_size[s]["real_wait_sec_total"],
                "real_wait_sec_avg_per_bench": float(np.mean(list(real_wait_by_bench[s].values())))
                if real_wait_by_bench[s]
                else 0.0,
                "real_qpu_sec_total": qose_by_size[s]["real_qpu_sec_total"],
                "real_qpu_sec_avg_per_bench": float(np.mean(list(real_qpu_by_bench[s].values())))
                if real_qpu_by_bench[s]
                else 0.0,
                "real_elapsed_sec_total": qose_by_size[s]["real_elapsed_sec_total"],
                "real_jobs": qose_by_size[s]["real_jobs"],
                "real_jobs_qpu_unknown": qose_by_size[s]["real_jobs_qpu_unknown"],
                "benches_with_mitigation": len(mitigation_by_bench[s]),
                "benches_with_real_jobs": len(real_elapsed_by_bench[s]),
            }
            rows1.append(row)
    csv1 = out_dir / f"qose_time_breakdown_{panel_tag}.csv"
    _write_rows_csv(csv1, rows1)

    # Figure 2: mitigation-stage breakdown by bench for QOS/QOSE @ requested sizes.
    stage_skip = {"bench", "size", "method", "total", "overall", "simulation", "cost_search_calls"}
    stage_alias = {"qaoa_analysis": "analysis"}
    stage_order_pref = ["analysis", "qr", "qf", "gv", "wc", "cost_search"]

    def _stage_values_merged(row: Dict[str, object]) -> Dict[str, float]:
        merged_vals: Dict[str, float] = {}
        for key, val in row.items():
            if key in stage_skip:
                continue
            stage = stage_alias.get(key, key)
            sec = max(_safe_float(val, 0.0), 0.0)
            if sec <= 0.0:
                continue
            merged_vals[stage] = merged_vals.get(stage, 0.0) + sec
        return merged_vals

    stage_set = set()
    for row in timing_rows:
        method = str(row.get("method", ""))
        size = _safe_int(row.get("size", ""), -1)
        bench = str(row.get("bench", ""))
        if method not in mitigation_methods or size not in size_set or bench not in bench_set:
            continue
        for key in _stage_values_merged(row).keys():
            stage_set.add(key)
    stage_order = [s for s in stage_order_pref if s in stage_set] + sorted(stage_set - set(stage_order_pref))
    if not stage_order:
        stage_order = ["analysis"]

    bench_order = [b for b, _ in BENCHES if b in bench_set]
    fig_rows = 1
    fig_cols = max(1, len(sizes) * len(mitigation_methods))
    fig2, axes2 = plt.subplots(fig_rows, fig_cols, figsize=(5.2 * fig_cols, 5.1), squeeze=False)
    cmap = plt.get_cmap("tab20")

    rows2: List[Dict[str, object]] = []
    subplot_index = 0
    for size in sizes:
        for method in mitigation_methods:
            axm = axes2[0, subplot_index]
            subplot_index += 1
            method_rows = [
                row
                for row in timing_rows
                if _safe_int(row.get("size", ""), -1) == int(size)
                and str(row.get("method", "")) == method
                and str(row.get("bench", "")) in bench_set
            ]
            bench_to_row = {str(row.get("bench", "")): row for row in method_rows}
            x_bench = np.arange(len(bench_order))
            bottom = np.zeros(len(bench_order), dtype=float)
            for si, stage in enumerate(stage_order):
                vals = []
                for bench in bench_order:
                    stage_vals = _stage_values_merged(bench_to_row.get(bench, {}))
                    val = stage_vals.get(stage, 0.0)
                    vals.append(max(val, 0.0))
                    rows2.append(
                        {
                            "size": int(size),
                            "method": method,
                            "bench": bench,
                            "stage": stage,
                            "sec": max(val, 0.0),
                        }
                    )
                vals_np = np.array(vals, dtype=float)
                if float(vals_np.sum()) <= 0.0:
                    continue
                axm.bar(
                    x_bench,
                    vals_np,
                    bottom=bottom,
                    color=cmap(si % 20),
                    label=stage if subplot_index == 1 else None,
                    linewidth=0.4,
                    edgecolor="black",
                )
                bottom += vals_np
            axm.set_title(f"{method} @ {size} qubits", fontsize=13)
            axm.set_xticks(x_bench)
            axm.set_xticklabels(bench_order, rotation=35, ha="right", fontsize=10)
            axm.tick_params(axis="y", labelsize=10)
            axm.set_ylabel("Seconds", fontsize=11)
            axm.grid(axis="y", linestyle="--", alpha=0.3)

    handles, labels = axes2[0, 0].get_legend_handles_labels()
    if handles:
        fig2.legend(
            handles,
            labels,
            ncol=max(1, len(labels)),
            fontsize=12,
            loc="upper center",
            frameon=False,
        )
        fig2.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        fig2.tight_layout()
    fig2_path = out_dir / f"mitigation_stage_breakdown_qos_qose_{panel_tag}.pdf"
    fig2.savefig(fig2_path)
    plt.close(fig2)

    csv2 = out_dir / f"mitigation_stage_breakdown_qos_qose_{panel_tag}.csv"
    _write_rows_csv(csv2, rows2)

    # Figure 3: average real-job count per bench by method (Torino/Marrakesh, 12/24).
    def _collect_real_job_counts(curr_job_rows: List[Dict[str, object]]) -> Dict[int, Dict[str, Dict[str, float]]]:
        counts: Dict[int, Dict[str, Dict[str, float]]] = {
            int(s): {} for s in sizes
        }
        seen_keys: set[str] = set()
        for rec in curr_job_rows:
            key = str(rec.get("job_key", "")).strip()
            if key:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            size = _safe_int(rec.get("size", ""), -1)
            bench = str(rec.get("bench", ""))
            method = str(rec.get("method", ""))
            if size not in size_set or bench not in bench_set or not method:
                continue
            by_method = counts[size].setdefault(method, {})
            by_method[bench] = float(by_method.get(bench, 0.0) + 1.0)
        return counts

    primary_job_counts = (
        strict_job_counts_primary
        if strict_job_counts_primary is not None
        else _collect_real_job_counts(job_rows)
    )
    secondary_job_counts: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None
    if strict_job_counts_secondary is not None:
        secondary_job_counts = strict_job_counts_secondary
    elif secondary_stats is not None and secondary_job_cache_jsonl is not None:
        secondary_job_rows = _read_jsonl_rows(secondary_job_cache_jsonl)
        secondary_job_counts = _collect_real_job_counts(secondary_job_rows)

    job_datasets: List[Tuple[str, Dict[int, Dict[str, Dict[str, float]]]]] = [
        (primary_name, primary_job_counts)
    ]
    if secondary_job_counts is not None:
        job_datasets.append((secondary_name, secondary_job_counts))

    method_priority = (
        list(job_methods_override)
        if job_methods_override
        else ["Qiskit", "FrozenQubits", "CutQC", "QOS", "QOSE", "QOSN", "qwen", "gemini", "gpt"]
    )
    method_seen: set[str] = set()
    job_methods: List[str] = []
    for m in method_priority:
        if any(m in ds_counts.get(int(s), {}) for _label, ds_counts in job_datasets for s in sizes):
            job_methods.append(m)
            method_seen.add(m)
    other_methods = sorted(
        {
            m
            for _label, ds_counts in job_datasets
            for s in sizes
            for m in ds_counts.get(int(s), {}).keys()
            if m not in method_seen
        }
    )
    job_methods.extend(other_methods)
    bench_order = [b for b, _ in BENCHES if b in bench_set]

    # One-row layout: backend-size panels laid out left-to-right
    # (e.g., Torino-12, Torino-24, Marrakesh-12, Marrakesh-24).
    panel_specs: List[Tuple[str, int, Dict[int, Dict[str, Dict[str, float]]]]] = []
    for backend_name, ds_counts in job_datasets:
        for size in sizes:
            panel_specs.append((backend_name, int(size), ds_counts))
    n_rows = 1
    n_cols = max(1, len(panel_specs))
    fig3, axes3 = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4.6),
        squeeze=False,
    )
    cmap_jobs = plt.get_cmap("tab10")
    hatch_patterns = ["///", "\\\\\\", "...", "xxx", "+++", "---", "|||", "ooo", "***"]
    rows3: List[Dict[str, object]] = []

    legend_handles = []
    legend_labels: List[str] = []
    y_eps = 1e-2
    for panel_idx, (backend_name, size, ds_counts) in enumerate(panel_specs):
        axj = axes3[0, panel_idx]
        x = np.arange(len(job_methods))
        avg_vals_active: List[float] = []
        avg_vals_all: List[float] = []
        plot_vals: List[float] = []
        for method in job_methods:
            bench_counts = ds_counts.get(int(size), {}).get(method, {})
            per_bench_all = [float(bench_counts.get(b, 0.0)) for b in bench_order]
            per_bench_active = [v for v in per_bench_all if v > 0.0]
            avg_jobs_active = float(np.mean(per_bench_active)) if per_bench_active else 0.0
            avg_jobs_all = float(np.mean(per_bench_all)) if per_bench_all else 0.0
            avg_vals_active.append(avg_jobs_active)
            avg_vals_all.append(avg_jobs_all)
            plot_vals.append(max(avg_jobs_active, y_eps))
            for bench in bench_order:
                rows3.append(
                    {
                        "backend": backend_name,
                        "size": int(size),
                        "method": method,
                        "bench": bench,
                        "jobs": float(bench_counts.get(bench, 0.0)),
                        "avg_jobs_per_bench_active": avg_jobs_active,
                        "avg_jobs_per_bench_all": avg_jobs_all,
                        "benches_with_jobs": len(per_bench_active),
                        "benches_total": len(per_bench_all),
                    }
                )

        colors = [cmap_jobs(i % 10) for i in range(len(job_methods))]
        bars = axj.bar(
            x,
            np.array(plot_vals, dtype=float),
            color=colors,
            edgecolor="black",
            linewidth=0.7,
        )
        for i, bar in enumerate(bars):
            bar.set_hatch(hatch_patterns[i % len(hatch_patterns)])
            if panel_idx == 0:
                legend_handles.append(bar)
                legend_labels.append(job_methods[i])
            val = float(avg_vals_active[i])
            y_text = max(plot_vals[i], y_eps) * 1.08
            axj.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_text,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axj.set_title(f"{backend_name} - {int(size)} qubits", fontsize=13)
        axj.set_ylabel("Avg jobs per bench", fontsize=11)
        axj.set_xticks(x)
        axj.set_xticklabels(job_methods, rotation=32, ha="right", fontsize=9)
        axj.tick_params(axis="y", labelsize=10)
        axj.set_yscale("log")
        axj.grid(axis="y", linestyle="--", alpha=0.3)

    if legend_handles:
        fig3.legend(
            legend_handles,
            legend_labels,
            ncol=max(1, len(legend_labels)),
            fontsize=10,
            loc="upper center",
            frameon=False,
        )
        fig3.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        fig3.tight_layout()
    fig3_path = out_dir / f"real_jobs_avg_per_bench_{panel_tag}.pdf"
    fig3.savefig(fig3_path)
    plt.close(fig3)

    csv3 = out_dir / f"real_jobs_avg_per_bench_{panel_tag}.csv"
    _write_rows_csv(csv3, rows3)

    return [fig1, fig2_path, fig3_path, csv1, csv2, csv3]


def _resume_row_complete(row: Dict[str, object], args, methods: List[str]) -> bool:
    for method in methods:
        prefix = _method_prefix(method)
        if not _row_has_nonempty_value(row, f"{prefix}_depth") or not _row_has_nonempty_value(row, f"{prefix}_nonlocal"):
            return False
        if args.with_fidelity and not _row_has_nonempty_value(row, f"{prefix}_fidelity"):
            return False
        if args.with_real_fidelity and not _row_has_nonempty_value(row, f"{prefix}_real_fidelity"):
            return False
    if args.with_fidelity and not _row_has_nonempty_value(row, "baseline_fidelity"):
        return False
    if args.with_real_fidelity and not _row_has_nonempty_value(row, "baseline_real_fidelity"):
        return False
    return True


def _row_has_method_metrics(row: Dict[str, object], args, method: str) -> bool:
    prefix = _method_prefix(method)
    if not _row_has_nonempty_value(row, f"{prefix}_depth") or not _row_has_nonempty_value(row, f"{prefix}_nonlocal"):
        return False
    if args.with_fidelity and not _row_has_nonempty_value(row, f"{prefix}_fidelity"):
        return False
    if args.with_real_fidelity and not _row_has_nonempty_value(row, f"{prefix}_real_fidelity"):
        return False
    return True


def _row_has_baseline_metrics(row: Dict[str, object], args) -> bool:
    if args.with_fidelity and not _row_has_nonempty_value(row, "baseline_fidelity"):
        return False
    if args.with_real_fidelity and not _row_has_nonempty_value(row, "baseline_real_fidelity"):
        return False
    return True


def _row_has_method_metrics_no_real(row: Dict[str, object], args, method: str) -> bool:
    prefix = _method_prefix(method)
    if not _row_has_nonempty_value(row, f"{prefix}_depth") or not _row_has_nonempty_value(row, f"{prefix}_nonlocal"):
        return False
    if args.with_fidelity and not _row_has_nonempty_value(row, f"{prefix}_fidelity"):
        return False
    return True


def _row_has_baseline_metrics_no_real(row: Dict[str, object], args) -> bool:
    if args.with_fidelity and not _row_has_nonempty_value(row, "baseline_fidelity"):
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
    timing_row_idx: Dict[Tuple[int, str, str], int] = {}
    for idx, trow in enumerate(timing_rows):
        try:
            tkey = (int(trow.get("size")), str(trow.get("bench")), str(trow.get("method")))
        except Exception:
            continue
        timing_row_idx[tkey] = idx
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
    existing_row_idx: Dict[str, int] = {}
    for idx, _row in enumerate(all_rows):
        try:
            key = _bench_key(int(_row["size"]), str(_row["bench"]))
            existing_row_map[key] = _row
            existing_row_idx[key] = idx
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
            reusable_methods: List[str] = []
            reusable_methods_no_real: List[str] = []
            if cached_row is not None:
                reusable_methods = [
                    m for m in methods if _row_has_method_metrics(cached_row, args, m)
                ]
                reusable_methods_no_real = [
                    m for m in methods if _row_has_method_metrics_no_real(cached_row, args, m)
                ]
            reusable_method_set = set(reusable_methods)
            reusable_baseline = cached_row is not None and _row_has_baseline_metrics(cached_row, args)
            reusable_baseline_no_real = (
                cached_row is not None and _row_has_baseline_metrics_no_real(cached_row, args)
            )
            if (
                bench_key in completed_keys
                and cached_row is not None
                and reusable_baseline
                and len(reusable_method_set) == len(methods)
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
            if cached_row is not None and reusable_methods_no_real:
                _restore_row_aggregates(
                    cached_row,
                    bench,
                    size,
                    reusable_methods_no_real,
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
                missing = [m for m in methods if m not in reusable_method_set]
                print(
                    f"[progress] partial-reuse size={size} bench={bench} reused={','.join(reusable_methods_no_real)}"
                    + (f" compute={','.join(missing)}" if missing else ""),
                    flush=True,
                )
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
            run_qos_now = run_qos and ("QOS" not in reusable_method_set)
            run_qosn_now = run_qosn and ("QOSN" not in reusable_method_set)
            run_fq_now = run_fq and ("FrozenQubits" not in reusable_method_set)
            run_cutqc_now = run_cutqc and ("CutQC" not in reusable_method_set)
            qose_methods_now = [m for m in qose_methods if m not in reusable_method_set]
            include_qose_now = bool(qose_methods_now)

            if run_qos_now:
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
            if run_qosn_now:
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
            if run_fq_now:
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

            if run_cutqc_now:
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
            if include_qose_now:
                for method in qose_methods_now:
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

            if run_qos_now and qos_m is not None:
                rel_depth[bench]["QOS"] = _relative(qos_m["depth"], base["depth"])
                rel_nonlocal[bench]["QOS"] = _relative(
                    qos_m["num_nonlocal_gates"], base["num_nonlocal_gates"]
                )
            if run_fq_now and fq_m is not None:
                rel_depth[bench]["FrozenQubits"] = _relative(fq_m["depth"], base["depth"])
                rel_nonlocal[bench]["FrozenQubits"] = _relative(
                    fq_m["num_nonlocal_gates"], base["num_nonlocal_gates"]
                )
            if run_cutqc_now and cutqc_m is not None:
                rel_depth[bench]["CutQC"] = _relative(cutqc_m["depth"], base["depth"])
                rel_nonlocal[bench]["CutQC"] = _relative(
                    cutqc_m["num_nonlocal_gates"], base["num_nonlocal_gates"]
                )
            if run_qosn_now and qosn_m is not None:
                rel_depth[bench]["QOSN"] = _relative(qosn_m["depth"], base["depth"])
                rel_nonlocal[bench]["QOSN"] = _relative(
                    qosn_m["num_nonlocal_gates"], base["num_nonlocal_gates"]
                )
            if include_qose_now:
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
                if reusable_baseline_no_real and cached_row is not None:
                    base_fidelity = _safe_float(cached_row.get("baseline_fidelity", ""), 0.0)
                    base_std = _safe_float(cached_row.get("baseline_fidelity_std", ""), 0.0)
                    sim_times["Qiskit"] = _safe_float(cached_row.get("baseline_sim_time", ""), 0.0)
                else:
                    t0 = time.perf_counter()
                    base_fidelity, base_std = _fidelity_stats(
                        [qc],
                        args.fidelity_shots,
                        noise,
                        args.fidelity_seed,
                        method="Qiskit",
                        bench=bench,
                        size=size,
                    )
                    sim_times["Qiskit"] = time.perf_counter() - t0
                fidelity_by_size[size][bench] = {"Qiskit": base_fidelity}
                fidelity_err_by_size[size][bench]["Qiskit"] = base_std
                rel_fidelity = {}

                qos_fidelity = ""
                qos_std = 0.0
                qosn_fidelity = ""
                qosn_std = 0.0
                fq_fidelity = ""
                fq_std = 0.0
                cutqc_fidelity = ""
                cutqc_std = 0.0
                qose_fidelity: Dict[str, float] = {}
                qose_std: Dict[str, float] = {}
                rel_fidelity = {}

                if run_qos_now and qos_circs is not None:
                    if cached_row is not None and _row_has_nonempty_value(cached_row, "qos_fidelity"):
                        qos_fidelity = _safe_float(cached_row.get("qos_fidelity", ""), 0.0)
                        qos_std = _safe_float(cached_row.get("qos_fidelity_std", ""), 0.0)
                        sim_times["QOS"] = _safe_float(cached_row.get("qos_sim_time", ""), 0.0)
                    else:
                        t0 = time.perf_counter()
                        qos_fidelity, qos_std = _fidelity_stats(
                            qos_circs,
                            args.fidelity_shots,
                            noise,
                            args.fidelity_seed,
                            method="QOS",
                            bench=bench,
                            size=size,
                        )
                        sim_times["QOS"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["QOS"] = qos_fidelity
                    fidelity_err_by_size[size][bench]["QOS"] = qos_std
                    rel_fidelity["QOS"] = _relative(qos_fidelity, base_fidelity)
                if run_qosn_now and qosn_circs is not None:
                    if cached_row is not None and _row_has_nonempty_value(cached_row, "qosn_fidelity"):
                        qosn_fidelity = _safe_float(cached_row.get("qosn_fidelity", ""), 0.0)
                        qosn_std = _safe_float(cached_row.get("qosn_fidelity_std", ""), 0.0)
                        sim_times["QOSN"] = _safe_float(cached_row.get("qosn_sim_time", ""), 0.0)
                    else:
                        t0 = time.perf_counter()
                        qosn_fidelity, qosn_std = _fidelity_stats(
                            qosn_circs,
                            args.fidelity_shots,
                            noise,
                            args.fidelity_seed,
                            method="QOSN",
                            bench=bench,
                            size=size,
                        )
                        sim_times["QOSN"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["QOSN"] = qosn_fidelity
                    fidelity_err_by_size[size][bench]["QOSN"] = qosn_std
                    rel_fidelity["QOSN"] = _relative(qosn_fidelity, base_fidelity)
                if run_fq_now and fq_circs is not None:
                    if cached_row is not None and _row_has_nonempty_value(cached_row, "fq_fidelity"):
                        fq_fidelity = _safe_float(cached_row.get("fq_fidelity", ""), 0.0)
                        fq_std = _safe_float(cached_row.get("fq_fidelity_std", ""), 0.0)
                        sim_times["FrozenQubits"] = _safe_float(cached_row.get("fq_sim_time", ""), 0.0)
                    else:
                        t0 = time.perf_counter()
                        fq_fidelity, fq_std = _fidelity_stats(
                            fq_circs,
                            args.fidelity_shots,
                            noise,
                            args.fidelity_seed,
                            method="FrozenQubits",
                            bench=bench,
                            size=size,
                        )
                        sim_times["FrozenQubits"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["FrozenQubits"] = fq_fidelity
                    fidelity_err_by_size[size][bench]["FrozenQubits"] = fq_std
                    rel_fidelity["FrozenQubits"] = _relative(fq_fidelity, base_fidelity)
                if run_cutqc_now and cutqc_circs is not None:
                    if cached_row is not None and _row_has_nonempty_value(cached_row, "cutqc_fidelity"):
                        cutqc_fidelity = _safe_float(cached_row.get("cutqc_fidelity", ""), 0.0)
                        cutqc_std = _safe_float(cached_row.get("cutqc_fidelity_std", ""), 0.0)
                        sim_times["CutQC"] = _safe_float(cached_row.get("cutqc_sim_time", ""), 0.0)
                    else:
                        t0 = time.perf_counter()
                        cutqc_fidelity, cutqc_std = _fidelity_stats(
                            cutqc_circs,
                            args.fidelity_shots,
                            noise,
                            args.fidelity_seed,
                            method="CutQC",
                            bench=bench,
                            size=size,
                        )
                        sim_times["CutQC"] = time.perf_counter() - t0
                    fidelity_by_size[size][bench]["CutQC"] = cutqc_fidelity
                    fidelity_err_by_size[size][bench]["CutQC"] = cutqc_std
                    rel_fidelity["CutQC"] = _relative(cutqc_fidelity, base_fidelity)
                if include_qose_now:
                    for method, qose_data in qose_results.items():
                        qose_circs = qose_data.get("circs")
                        if not qose_circs:
                            continue
                        prefix = method.lower()
                        if cached_row is not None and _row_has_nonempty_value(cached_row, f"{prefix}_fidelity"):
                            qose_val = _safe_float(cached_row.get(f"{prefix}_fidelity", ""), 0.0)
                            qose_std_val = _safe_float(cached_row.get(f"{prefix}_fidelity_std", ""), 0.0)
                            sim_times[method] = _safe_float(cached_row.get(f"{prefix}_sim_time", ""), 0.0)
                        else:
                            t0 = time.perf_counter()
                            qose_val, qose_std_val = _fidelity_stats(
                                qose_circs,
                                args.fidelity_shots,
                                noise,
                                args.fidelity_seed,
                                method=method,
                                bench=bench,
                                size=size,
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
                rel_fidelity = {}
                if cached_row is not None:
                    base_fidelity = cached_row.get("baseline_fidelity", "")
                    base_std = cached_row.get("baseline_fidelity_std", "")
                    sim_times["Qiskit"] = _safe_float(cached_row.get("baseline_sim_time", ""), 0.0)
                    qos_fidelity = cached_row.get("qos_fidelity", "")
                    qosn_fidelity = cached_row.get("qosn_fidelity", "")
                    fq_fidelity = cached_row.get("fq_fidelity", "")
                    cutqc_fidelity = cached_row.get("cutqc_fidelity", "")
                    qose_fidelity = {}
                    qose_std = {}
                    for method in qose_methods:
                        prefix = method.lower()
                        qose_fidelity[method] = cached_row.get(f"{prefix}_fidelity", "")
                        qose_std[method] = _safe_float(
                            cached_row.get(f"{prefix}_fidelity_std", ""),
                            0.0,
                        )
                        sim_times[method] = _safe_float(cached_row.get(f"{prefix}_sim_time", ""), 0.0)
                    rel_fidelity_qos = cached_row.get("rel_fidelity_qos", "")
                    rel_fidelity_qosn = cached_row.get("rel_fidelity_qosn", "")
                    rel_fidelity_fq = cached_row.get("rel_fidelity_fq", "")
                    rel_fidelity_cutqc = cached_row.get("rel_fidelity_cutqc", "")
                    rel_fidelity_qose = cached_row.get("rel_fidelity_qose", "")
                else:
                    base_fidelity = ""
                    base_std = ""
                    qos_fidelity = ""
                    qosn_fidelity = ""
                    fq_fidelity = ""
                    cutqc_fidelity = ""
                    qose_fidelity = {}
                    qose_std = {}
                    rel_fidelity_qos = ""
                    rel_fidelity_qosn = ""
                    rel_fidelity_fq = ""
                    rel_fidelity_cutqc = ""
                    rel_fidelity_qose = ""

            if args.with_real_fidelity:
                real_allowed: Optional[set[str]] = getattr(args, "_real_fidelity_compute_methods", None)

                def _real_enabled(method_name: str) -> bool:
                    return real_allowed is None or method_name in real_allowed

                def _cached_real_pair(prefix: str) -> Tuple[Optional[float], Optional[float]]:
                    if cached_row is None:
                        return None, None
                    key = f"{prefix}_real_fidelity"
                    std_key = f"{prefix}_real_fidelity_std"
                    val = cached_row.get(key, "")
                    if val in {"", None}:
                        return None, None
                    return _safe_float(val, 0.0), _safe_float(cached_row.get(std_key, ""), 0.0)

                real_qose: Dict[str, float] = {}
                real_qose_std: Dict[str, float] = {}

                real_base_val: Optional[float] = None
                real_base_std_val: Optional[float] = None
                real_qos_val: Optional[float] = None
                real_qos_std_val: Optional[float] = None
                real_qosn_val: Optional[float] = None
                real_qosn_std_val: Optional[float] = None
                real_fq_val: Optional[float] = None
                real_fq_std_val: Optional[float] = None
                real_cut_val: Optional[float] = None
                real_cut_std_val: Optional[float] = None

                # Baseline (Qiskit)
                if cached_row is not None:
                    val = cached_row.get("baseline_real_fidelity", "")
                    if val not in {"", None}:
                        real_base_val = _safe_float(val, 0.0)
                        real_base_std_val = _safe_float(cached_row.get("baseline_real_fidelity_std", ""), 0.0)
                if real_base_val is not None:
                    print(
                        f"[progress] size={size} bench={bench} method=Qiskit stage=real_fidelity source=reuse",
                        flush=True,
                    )
                elif _real_enabled("Qiskit"):
                    print(f"[progress] size={size} bench={bench} method=Qiskit stage=real_fidelity", flush=True)
                    real_base_val, real_base_std_val = _real_fidelity_stats(
                        [qc],
                        args.real_fidelity_shots,
                        args.real_backend,
                        args.fidelity_seed,
                        "Qiskit",
                        bench,
                        size,
                    )
                else:
                    print(
                        f"[progress] size={size} bench={bench} method=Qiskit stage=real_fidelity source=skip-disabled",
                        flush=True,
                    )
                if real_base_val is not None:
                    real_fidelity_by_size[size][bench]["Qiskit"] = real_base_val
                    real_fidelity_err_by_size[size][bench]["Qiskit"] = float(real_base_std_val or 0.0)

                # QOS / QOSN / FrozenQubits / CutQC
                if run_qos_now and qos_circs is not None:
                    real_qos_val, real_qos_std_val = _cached_real_pair("qos")
                    if real_qos_val is not None:
                        print(
                            f"[progress] size={size} bench={bench} method=QOS stage=real_fidelity source=reuse",
                            flush=True,
                        )
                    elif _real_enabled("QOS"):
                        print(f"[progress] size={size} bench={bench} method=QOS stage=real_fidelity", flush=True)
                        real_qos_val, real_qos_std_val = _real_fidelity_stats(
                            qos_circs,
                            args.real_fidelity_shots,
                            args.real_backend,
                            args.fidelity_seed,
                            "QOS",
                            bench,
                            size,
                        )
                    else:
                        print(
                            f"[progress] size={size} bench={bench} method=QOS stage=real_fidelity source=skip-disabled",
                            flush=True,
                        )
                    if real_qos_val is not None:
                        real_fidelity_by_size[size][bench]["QOS"] = real_qos_val
                        real_fidelity_err_by_size[size][bench]["QOS"] = float(real_qos_std_val or 0.0)

                if run_qosn_now and qosn_circs is not None:
                    real_qosn_val, real_qosn_std_val = _cached_real_pair("qosn")
                    if real_qosn_val is not None:
                        print(
                            f"[progress] size={size} bench={bench} method=QOSN stage=real_fidelity source=reuse",
                            flush=True,
                        )
                    elif _real_enabled("QOSN"):
                        print(f"[progress] size={size} bench={bench} method=QOSN stage=real_fidelity", flush=True)
                        real_qosn_val, real_qosn_std_val = _real_fidelity_stats(
                            qosn_circs,
                            args.real_fidelity_shots,
                            args.real_backend,
                            args.fidelity_seed,
                            "QOSN",
                            bench,
                            size,
                        )
                    else:
                        print(
                            f"[progress] size={size} bench={bench} method=QOSN stage=real_fidelity source=skip-disabled",
                            flush=True,
                        )
                    if real_qosn_val is not None:
                        real_fidelity_by_size[size][bench]["QOSN"] = real_qosn_val
                        real_fidelity_err_by_size[size][bench]["QOSN"] = float(real_qosn_std_val or 0.0)

                if run_fq_now and fq_circs is not None:
                    real_fq_val, real_fq_std_val = _cached_real_pair("fq")
                    if real_fq_val is not None:
                        print(
                            f"[progress] size={size} bench={bench} method=FrozenQubits stage=real_fidelity source=reuse",
                            flush=True,
                        )
                    elif _real_enabled("FrozenQubits"):
                        print(
                            f"[progress] size={size} bench={bench} method=FrozenQubits stage=real_fidelity",
                            flush=True,
                        )
                        real_fq_val, real_fq_std_val = _real_fidelity_stats(
                            fq_circs,
                            args.real_fidelity_shots,
                            args.real_backend,
                            args.fidelity_seed,
                            "FrozenQubits",
                            bench,
                            size,
                        )
                    else:
                        print(
                            f"[progress] size={size} bench={bench} method=FrozenQubits stage=real_fidelity source=skip-disabled",
                            flush=True,
                        )
                    if real_fq_val is not None:
                        real_fidelity_by_size[size][bench]["FrozenQubits"] = real_fq_val
                        real_fidelity_err_by_size[size][bench]["FrozenQubits"] = float(real_fq_std_val or 0.0)

                if run_cutqc_now and cutqc_circs is not None:
                    real_cut_val, real_cut_std_val = _cached_real_pair("cutqc")
                    if real_cut_val is not None:
                        print(
                            f"[progress] size={size} bench={bench} method=CutQC stage=real_fidelity source=reuse",
                            flush=True,
                        )
                    elif _real_enabled("CutQC"):
                        print(f"[progress] size={size} bench={bench} method=CutQC stage=real_fidelity", flush=True)
                        real_cut_val, real_cut_std_val = _real_fidelity_stats(
                            cutqc_circs,
                            args.real_fidelity_shots,
                            args.real_backend,
                            args.fidelity_seed,
                            "CutQC",
                            bench,
                            size,
                        )
                    else:
                        print(
                            f"[progress] size={size} bench={bench} method=CutQC stage=real_fidelity source=skip-disabled",
                            flush=True,
                        )
                    if real_cut_val is not None:
                        real_fidelity_by_size[size][bench]["CutQC"] = real_cut_val
                        real_fidelity_err_by_size[size][bench]["CutQC"] = float(real_cut_std_val or 0.0)

                if include_qose_now:
                    for method, qose_data in qose_results.items():
                        qose_circs = qose_data.get("circs")
                        if not qose_circs:
                            continue
                        prefix = method.lower()
                        cached_val, cached_std = _cached_real_pair(prefix)
                        if cached_val is not None:
                            print(
                                f"[progress] size={size} bench={bench} method={method} stage=real_fidelity source=reuse",
                                flush=True,
                            )
                            real_qose[method] = cached_val
                            real_qose_std[method] = float(cached_std or 0.0)
                        elif _real_enabled(method):
                            print(f"[progress] size={size} bench={bench} method={method} stage=real_fidelity", flush=True)
                            real_qose_val, real_qose_std_val = _real_fidelity_stats(
                                qose_circs,
                                args.real_fidelity_shots,
                                args.real_backend,
                                args.fidelity_seed,
                                method,
                                bench,
                                size,
                            )
                            real_qose[method] = real_qose_val
                            real_qose_std[method] = real_qose_std_val
                        else:
                            print(
                                f"[progress] size={size} bench={bench} method={method} stage=real_fidelity source=skip-disabled",
                                flush=True,
                            )
                        if method in real_qose:
                            real_fidelity_by_size[size][bench][method] = real_qose[method]
                            real_fidelity_err_by_size[size][bench][method] = real_qose_std.get(method, 0.0)

                real_base = real_base_val if real_base_val is not None else ""
                real_base_std = real_base_std_val if real_base_std_val is not None else ""
                real_qos = real_qos_val if real_qos_val is not None else ""
                real_qos_std = real_qos_std_val if real_qos_std_val is not None else ""
                real_qosn = real_qosn_val if real_qosn_val is not None else ""
                real_qosn_std = real_qosn_std_val if real_qosn_std_val is not None else ""
                real_fq = real_fq_val if real_fq_val is not None else ""
                real_fq_std = real_fq_std_val if real_fq_std_val is not None else ""
                real_cut = real_cut_val if real_cut_val is not None else ""
                real_cut_std = real_cut_std_val if real_cut_std_val is not None else ""
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
                if run_qos_now and qos_circs is not None:
                    real_job_counts_by_size[size][bench]["QOS"] = float(
                        sum(_count_real_jobs(c) for c in qos_circs)
                    )
                if run_qosn_now and qosn_circs is not None:
                    real_job_counts_by_size[size][bench]["QOSN"] = float(
                        sum(_count_real_jobs(c) for c in qosn_circs)
                    )
                if run_fq_now and fq_circs is not None:
                    real_job_counts_by_size[size][bench]["FrozenQubits"] = float(
                        sum(_count_real_jobs(c) for c in fq_circs)
                    )
                if run_cutqc_now and cutqc_circs is not None:
                    real_job_counts_by_size[size][bench]["CutQC"] = float(
                        sum(_count_real_jobs(c) for c in cutqc_circs)
                    )
                if include_qose_now:
                    for method, qose_data in qose_results.items():
                        qose_circs = qose_data.get("circs")
                        if not qose_circs:
                            continue
                        real_job_counts_by_size[size][bench][method] = float(
                            sum(_count_real_jobs(c) for c in qose_circs)
                        )

            qiskit_sim = float(sim_times.get("Qiskit", 0.0))

            row = dict(cached_row) if cached_row is not None else {}
            row.update(
                {
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
            )

            if run_qos_now and qos_m is not None:
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
            if run_qosn_now and qosn_m is not None:
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
            if run_fq_now and fq_m is not None:
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
            if run_cutqc_now and cutqc_m is not None:
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
            if include_qose_now:
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
            if bench_key in existing_row_idx:
                all_rows[existing_row_idx[bench_key]] = row
            else:
                existing_row_idx[bench_key] = len(all_rows)
                all_rows.append(row)

            if args.collect_timing:
                timing_methods = []
                if run_qos_now and qos_t is not None:
                    timing_methods.append(("QOS", qos_t))
                if run_qosn_now and qosn_t is not None:
                    timing_methods.append(("QOSN", qosn_t))
                if run_fq_now and fq_t is not None:
                    timing_methods.append(("FrozenQubits", fq_t))
                if run_cutqc_now and cutqc_t is not None:
                    timing_methods.append(("CutQC", cutqc_t))
                if include_qose_now:
                    for method, qose_data in qose_results.items():
                        qose_t = qose_data.get("timing")
                        if qose_t is not None:
                            timing_methods.append((method, qose_t))
                for method, timing in timing_methods:
                    trow = {"bench": bench, "size": size, "method": method}
                    trow.update(timing)
                    if args.with_fidelity:
                        trow["simulation"] = sim_times.get(method, 0.0)
                    tkey = (size, bench, method)
                    if tkey in timing_row_idx:
                        timing_rows[timing_row_idx[tkey]] = trow
                    else:
                        timing_row_idx[tkey] = len(timing_rows)
                        timing_rows.append(trow)
            completed_keys.add(bench_key)
            existing_row_map[bench_key] = row
            if on_bench_complete is not None:
                on_bench_complete(size, bench, all_rows, timing_rows, completed_keys)
            if args.cut_visualization:
                cut_circuits[(size, bench, "Qiskit")] = [qc]
                if run_qos_now and qos_circs is not None:
                    cut_circuits[(size, bench, "QOS")] = qos_circs
                if run_qosn_now and qosn_circs is not None:
                    cut_circuits[(size, bench, "QOSN")] = qosn_circs
                if include_qose_now:
                    for method, qose_data in qose_results.items():
                        qose_circs = qose_data.get("circs")
                        if qose_circs is not None:
                            cut_circuits[(size, bench, method)] = qose_circs
                if run_fq_now and fq_circs is not None:
                    cut_circuits[(size, bench, "FrozenQubits")] = fq_circs
                if run_cutqc_now and cutqc_circs is not None:
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
        "--state-backend-label",
        default="",
        help=(
            "Optional backend label for state/cache namespace and plot-input routing "
            "(e.g., use 'marrakesh' while running real jobs on ibm_fez)."
        ),
    )
    parser.add_argument(
        "--real-fidelity-compute-methods",
        default="all",
        help=(
            "Comma-separated methods that are allowed to submit real-QPU jobs. "
            "Use 'all' (default) for legacy behavior, or e.g. 'QOSE' to submit only QOSE. "
            "Accepted names: baseline/Qiskit,FrozenQubits,CutQC,QOS,QOSN,QOSE,qwen,gemini,gpt."
        ),
    )
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
        "--artifact-dir",
        default="",
        help=(
            "Optional directory to store all full-eval artifacts (state, caches, CSV, figures). "
            "If unset, uses --out-dir (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--resume-state",
        default="",
        help="Path to resume-state JSON. Default: <artifact-dir or out-dir>/full_eval_progress.json",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing resume-state and run from scratch.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Disable resume/job/ideal/method caches for this run "
            "(no cache files or resume-state writes)."
        ),
    )
    parser.add_argument(
        "--reset-resume",
        action="store_true",
        help="Delete existing resume-state before starting.",
    )
    parser.add_argument(
        "--plot-cached-panels",
        action="store_true",
        help="Plot 1x4 cached panel figures from existing CSV outputs, then exit.",
    )
    parser.add_argument(
        "--plot-cached-panels-after-run",
        action="store_true",
        help=(
            "After normal full_eval completes, generate cached-panel figures in the same run. "
            "Defaults: sim/timing CSV from this run; real CSVs must be provided via "
            "--panel-real-torino-csv and --panel-real-marrakesh-csv."
        ),
    )
    parser.add_argument(
        "--panel-simtiming-csv",
        default="",
        help="CSV with depth/CNOT/sim-fidelity rows (e.g. relative_properties_*_simtiming.csv).",
    )
    parser.add_argument(
        "--panel-real-torino-csv",
        default="",
        help="CSV with Torino real-fidelity rows (e.g. relative_properties_*_real.csv).",
    )
    parser.add_argument(
        "--panel-real-marrakesh-csv",
        default="",
        help="CSV with Marrakesh real-fidelity rows (e.g. relative_properties_*_real_marrakesh.csv).",
    )
    parser.add_argument(
        "--panel-timing-csv",
        default="",
        help="CSV with timing rows (e.g. timing_partial_*_simtiming.csv or --timing-csv output).",
    )
    parser.add_argument(
        "--panel-sizes",
        default="12,24",
        help="Comma-separated qubit sizes for panel plotting mode.",
    )
    parser.add_argument(
        "--panel-methods",
        default="QOS,CutQC,FrozenQubits",
        help="Comma-separated methods for panel plotting mode.",
    )
    parser.add_argument(
        "--plot-time-breakdowns",
        action="store_true",
        help=(
            "Plot time breakdown figures from cached full-eval timing/job-cache data, then exit. "
            "Outputs: 1) QOSE end-to-end time components, "
            "2) QOS/QOSE mitigation stage breakdown, "
            "3) avg real jobs per bench by method."
        ),
    )
    parser.add_argument(
        "--timebreak-timing-csv",
        default="",
        help="Timing CSV for --plot-time-breakdowns. If unset, picks latest timing_*.csv under --out-dir.",
    )
    parser.add_argument(
        "--timebreak-job-cache",
        default="",
        help=(
            "Real-job cache JSONL for --plot-time-breakdowns. "
            "If unset, uses cache from --resume-state (if set) else latest *job_cache*.jsonl under --out-dir."
        ),
    )
    parser.add_argument(
        "--timebreak-job-count-source",
        choices=["cache", "dryrun"],
        default="cache",
        help=(
            "Source for avg-jobs-per-bench panel: "
            "'cache' uses submitted-job cache; "
            "'dryrun' recomputes strict expected job counts without reuse."
        ),
    )
    parser.add_argument(
        "--timebreak-job-methods",
        default="Qiskit,FrozenQubits,CutQC,QOS,QOSE",
        help=(
            "Comma-separated methods for strict dry-run job counting. "
            "Used when --timebreak-job-count-source=dryrun."
        ),
    )
    parser.add_argument(
        "--timebreak-secondary-timing-csv",
        default="",
        help=(
            "Optional second timing CSV for backend comparison in QOSE breakdown "
            "(produces a 1x4 figure: primary+secondary)."
        ),
    )
    parser.add_argument(
        "--timebreak-secondary-job-cache",
        default="",
        help="Optional second real-job cache JSONL matching --timebreak-secondary-timing-csv.",
    )
    parser.add_argument(
        "--timebreak-primary-label",
        default="",
        help="Optional display label for the primary QOSE breakdown backend.",
    )
    parser.add_argument(
        "--timebreak-secondary-label",
        default="",
        help="Optional display label for the secondary QOSE breakdown backend.",
    )
    parser.add_argument(
        "--timebreak-sizes",
        default="12,24",
        help="Comma-separated qubit sizes for --plot-time-breakdowns.",
    )
    parser.add_argument(
        "--timebreak-methods",
        default="QOS,QOSE",
        help="Comma-separated methods for mitigation-stage breakdown figure.",
    )
    parser.add_argument(
        "--timebreak-benches",
        default="all",
        help="Comma-separated benches for --plot-time-breakdowns, or 'all'.",
    )
    args = parser.parse_args()

    if args.plot_cached_panels:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        sizes = [int(s.strip()) for s in str(args.panel_sizes).split(",") if s.strip()]
        methods = [m.strip() for m in str(args.panel_methods).split(",") if m.strip()]
        if not sizes:
            raise RuntimeError("--panel-sizes is empty.")
        if not methods:
            raise RuntimeError("--panel-methods is empty.")
        required = {
            "--panel-simtiming-csv": args.panel_simtiming_csv,
            "--panel-real-torino-csv": args.panel_real_torino_csv,
            "--panel-real-marrakesh-csv": args.panel_real_marrakesh_csv,
            "--panel-timing-csv": args.panel_timing_csv,
        }
        missing = [flag for flag, path in required.items() if not str(path).strip()]
        if missing:
            raise RuntimeError(
                "Missing required args for --plot-cached-panels: " + ", ".join(missing)
            )
        out_paths = _plot_cached_panels(
            simtiming_csv=Path(args.panel_simtiming_csv),
            real_torino_csv=Path(args.panel_real_torino_csv),
            real_marrakesh_csv=Path(args.panel_real_marrakesh_csv),
            timing_csv=Path(args.panel_timing_csv),
            out_dir=out_dir,
            timestamp=timestamp,
            tag=args.tag.strip(),
            sizes=sizes,
            methods=methods,
        )
        for path in out_paths:
            print(f"Wrote figure: {path}")
        publish_root = out_dir.parent if out_dir.name == "full_eval_artifacts" else out_dir
        _publish_full_eval_views(
            out_dir=publish_root,
            artifact_dir=out_dir,
            panel_paths=out_paths,
            final_csv_path=Path(args.panel_simtiming_csv),
            timing_path=Path(args.panel_timing_csv),
            backend_label=_state_backend_label(args),
        )
        if not args.plot_time_breakdowns:
            return

    if args.plot_time_breakdowns:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        if str(args.timebreak_timing_csv).strip():
            timing_csv = Path(args.timebreak_timing_csv)
        else:
            timing_csv = _latest_file(list(out_dir.rglob("timing_*.csv")))
            if timing_csv is None:
                raise RuntimeError(
                    "Could not find timing CSV automatically. Pass --timebreak-timing-csv."
                )

        if str(args.timebreak_job_cache).strip():
            job_cache = Path(args.timebreak_job_cache)
        elif str(args.resume_state).strip():
            resume_path = Path(args.resume_state)
            job_cache, _, _ = _resume_companion_paths(resume_path)
        else:
            job_cache = _latest_file(list(out_dir.rglob("*job_cache*.jsonl")))
            if job_cache is None:
                raise RuntimeError(
                    "Could not find job cache automatically. Pass --timebreak-job-cache or --resume-state."
                )

        secondary_timing_csv: Optional[Path] = None
        secondary_job_cache: Optional[Path] = None
        has_secondary_timing = bool(str(args.timebreak_secondary_timing_csv).strip())
        has_secondary_job = bool(str(args.timebreak_secondary_job_cache).strip())
        if has_secondary_timing != has_secondary_job:
            raise RuntimeError(
                "Provide both --timebreak-secondary-timing-csv and --timebreak-secondary-job-cache, or neither."
            )
        if has_secondary_timing and has_secondary_job:
            secondary_timing_csv = Path(args.timebreak_secondary_timing_csv)
            secondary_job_cache = Path(args.timebreak_secondary_job_cache)

        sizes = [int(s.strip()) for s in str(args.timebreak_sizes).split(",") if s.strip()]
        methods = [m.strip() for m in str(args.timebreak_methods).split(",") if m.strip()]
        if str(args.timebreak_benches).strip().lower() == "all":
            benches = [b for b, _ in BENCHES]
        else:
            benches = [b.strip() for b in str(args.timebreak_benches).split(",") if b.strip()]
        if not sizes:
            raise RuntimeError("--timebreak-sizes is empty.")
        if not methods:
            raise RuntimeError("--timebreak-methods is empty.")
        if not benches:
            raise RuntimeError("--timebreak-benches is empty.")

        job_methods_override = [
            m.strip()
            for m in str(args.timebreak_job_methods).split(",")
            if m.strip()
        ]
        if not job_methods_override:
            job_methods_override = ["Qiskit", "FrozenQubits", "CutQC", "QOS", "QOSE"]

        strict_job_counts_primary = None
        strict_job_counts_secondary = None
        if str(args.timebreak_job_count_source).strip().lower() == "dryrun":
            primary_backend_hint = _normalize_backend_name(str(args.timebreak_primary_label))
            if primary_backend_hint == "generic":
                primary_backend_hint = _backend_hint_from_path(timing_csv)
            strict_job_counts_primary = _strict_dry_run_real_job_counts(
                args=args,
                sizes=sizes,
                benches=benches,
                methods=job_methods_override,
                backend_hint=primary_backend_hint,
            )
            if secondary_timing_csv is not None:
                secondary_backend_hint = _normalize_backend_name(str(args.timebreak_secondary_label))
                if secondary_backend_hint == "generic":
                    secondary_backend_hint = _backend_hint_from_path(secondary_timing_csv)
                strict_job_counts_secondary = _strict_dry_run_real_job_counts(
                    args=args,
                    sizes=sizes,
                    benches=benches,
                    methods=job_methods_override,
                    backend_hint=secondary_backend_hint,
                )
            print(
                "[time-breakdown] avg-jobs panel source=dryrun (strict, no resume/cache reuse)",
                flush=True,
            )

        out_paths = _plot_time_breakdowns(
            timing_csv=timing_csv,
            job_cache_jsonl=job_cache,
            out_dir=out_dir,
            timestamp=timestamp,
            tag=args.tag.strip(),
            sizes=sizes,
            methods=methods,
            benches=benches,
            secondary_timing_csv=secondary_timing_csv,
            secondary_job_cache_jsonl=secondary_job_cache,
            primary_label=str(args.timebreak_primary_label).strip(),
            secondary_label=str(args.timebreak_secondary_label).strip(),
            strict_job_counts_primary=strict_job_counts_primary,
            strict_job_counts_secondary=strict_job_counts_secondary,
            job_methods_override=job_methods_override,
        )
        for path in out_paths:
            print(f"Wrote: {path}")
        publish_root = out_dir.parent if out_dir.name == "full_eval_artifacts" else out_dir
        _publish_full_eval_views(
            out_dir=publish_root,
            artifact_dir=out_dir,
            breakdown_paths=[p for p in out_paths if p.suffix.lower() == ".pdf"],
            backend_label=_state_backend_label(args),
        )
        return

    global _REAL_DRY_RUN, _RUN_QPU_SEC, _RUN_QPU_SEC_KNOWN_JOBS, _RUN_QPU_SEC_UNKNOWN_JOBS
    _REAL_DRY_RUN = bool(args.real_fidelity_dry_run or not args.with_real_fidelity)
    _RUN_QPU_SEC = 0.0
    _RUN_QPU_SEC_KNOWN_JOBS = 0
    _RUN_QPU_SEC_UNKNOWN_JOBS = 0
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
    artifact_dir = Path(args.artifact_dir) if str(args.artifact_dir).strip() else out_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
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
    args._real_fidelity_compute_methods = _parse_real_fidelity_compute_methods(
        str(args.real_fidelity_compute_methods)
    )
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

    state_backend = _infer_state_backend(args)
    state_stage = _infer_state_stage(args)
    explicit_resume_state = bool(args.resume_state.strip())
    if explicit_resume_state:
        resume_state_path = Path(args.resume_state)
    else:
        state_dir = out_dir / "state" / state_backend / state_stage
        state_dir.mkdir(parents=True, exist_ok=True)
        resume_state_path = state_dir / "resume.json"
        print(f"Full-eval state dir: {state_dir}", flush=True)
    if str(args.artifact_dir).strip() and not explicit_resume_state:
        print(f"Full-eval artifact dir: {artifact_dir}", flush=True)
    use_cache = not args.no_cache
    signature = _resume_signature(args, benches, sizes, selected_methods)
    job_cache_path, ideal_cache_path, default_method_cache_dir = _resume_companion_paths(
        resume_state_path
    )

    fallback_resume_candidates: List[Path] = []
    if not explicit_resume_state:
        state_root = out_dir / "state" / state_backend
        if state_stage == "mixed":
            fallback_resume_candidates.extend(
                [state_root / "sim_only" / "resume.json", state_root / "real_only" / "resume.json"]
            )
        elif state_stage == "real_only":
            fallback_resume_candidates.extend(
                [state_root / "mixed" / "resume.json", state_root / "sim_only" / "resume.json"]
            )
        elif state_stage == "sim_only":
            fallback_resume_candidates.extend([state_root / "mixed" / "resume.json"])
        sizes_token = "_".join(str(int(s)) for s in sorted(set(sizes)))
        fallback_resume_candidates.extend(
            [
                artifact_dir / "full_eval_progress.json",
                artifact_dir / f"full_eval_progress_{sizes_token}.json",
            ]
        )
        fallback_resume_candidates.extend(sorted(artifact_dir.glob("full_eval_progress*.json")))
        fallback_resume_candidates = _dedup_paths(fallback_resume_candidates)

    if args.reset_resume and resume_state_path.exists():
        resume_state_path.unlink()
    if args.reset_resume and job_cache_path.exists():
        job_cache_path.unlink()
    if args.reset_resume and ideal_cache_path.exists():
        ideal_cache_path.unlink()
    if use_cache and not args.no_resume and not args.reset_resume and fallback_resume_candidates:
        seeded_from = _seed_resume_from_candidates(
            primary_resume=resume_state_path,
            signature=signature,
            candidates=fallback_resume_candidates,
        )
        if seeded_from is not None:
            print(f"Seeded resume/caches from: {seeded_from}", flush=True)

    global _REAL_JOB_RESULT_CACHE, _IDEAL_RESULT_CACHE
    if use_cache:
        _REAL_JOB_RESULT_CACHE = _RealJobResultCache(job_cache_path)
        _IDEAL_RESULT_CACHE = _RealJobResultCache(ideal_cache_path)
        print(
            f"Real-job cache: {job_cache_path} entries={_REAL_JOB_RESULT_CACHE.count()}",
            flush=True,
        )
        print(
            f"Ideal-sim cache: {ideal_cache_path} entries={_IDEAL_RESULT_CACHE.count()}",
            flush=True,
        )
    else:
        _REAL_JOB_RESULT_CACHE = None
        _IDEAL_RESULT_CACHE = None
        print("Caches disabled (--no-cache): running without persisted caches/resume.", flush=True)
    initial_rows: List[Dict[str, object]] = []
    initial_timing_rows: List[Dict[str, object]] = []
    completed_keys: set[str] = set()
    if use_cache and not args.no_resume:
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
        if use_cache:
            print("Resume disabled (--no-resume): starting fresh", flush=True)
        else:
            print("Starting fresh (--no-cache)", flush=True)

    partial_csv_path = artifact_dir / f"relative_properties_partial{tag_suffix}.csv"
    partial_timing_path = artifact_dir / f"timing_partial{tag_suffix}.csv"
    method_cache_dir: Optional[Path] = None
    use_method_cache = bool(use_cache)
    if use_method_cache:
        method_cache_dir = default_method_cache_dir
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
        if use_cache:
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
            f"[progress] completed size={size} bench={bench} total={len(completed)}/{progress_total}"
            + (f" saved={resume_state_path}" if use_cache else ""),
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

        sweep_path = artifact_dir / f"relative_properties_sweep_{timestamp}{tag_suffix}.csv"
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
            if use_cache:
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
            print(
                "Accumulated qpu_sec this run (submitted jobs only): "
                f"{_RUN_QPU_SEC:.2f} "
                f"(known_jobs={_RUN_QPU_SEC_KNOWN_JOBS}, unknown_jobs={_RUN_QPU_SEC_UNKNOWN_JOBS})",
                file=sys.stderr,
            )
            if use_cache:
                print(f"Resume state saved: {resume_state_path}", file=sys.stderr)
                print("Set a new IBM key, then resume with:", file=sys.stderr)
                print(resume_cmd, file=sys.stderr)
            _cleanup_children()
            return

    fidelity_methods = ["Qiskit"] + [m for m in selected_methods if m != "Qiskit"]
    combined_path = _plot_combined(
        rel_by_size,
        benches,
        artifact_dir,
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
            real_job_counts_by_size, benches, artifact_dir, f"{timestamp}{tag_suffix}", job_methods
        )
        print(f"Wrote job counts figure: {job_counts_fig}")
    if args.cut_visualization and cut_circuits:
        cut_methods = ["Qiskit"] + [m for m in selected_methods if m != "Qiskit"]
        cut_fig = _plot_cut_visualization(
            cut_circuits,
            benches,
            sizes,
            artifact_dir,
            f"{timestamp}{tag_suffix}",
            cut_methods,
        )
        print(f"Wrote cut visualization: {cut_fig}")
    if args.collect_timing and timing_rows:
        timing_path = (
            Path(args.timing_csv)
            if args.timing_csv
            else artifact_dir / f"timing_{timestamp}{tag_suffix}.csv"
        )
        _write_rows_csv(timing_path, timing_rows)
        print(f"Wrote timing: {timing_path}")
    else:
        timing_path = None
    if args.collect_timing and timing_rows and args.timing_plot:
        timing_fig = _plot_timing(
            timing_rows, artifact_dir, f"{timestamp}{tag_suffix}", selected_methods
        )
        print(f"Wrote timing figure: {timing_fig}")
    if args.overhead_plot:
        if not args.with_fidelity:
            raise RuntimeError("Overhead plot requires --with-fidelity to measure simulation time.")
        overhead_fig = _plot_overheads(
            all_rows, benches, sizes, artifact_dir, f"{timestamp}{tag_suffix}", selected_methods
        )
        print(f"Wrote overhead figure: {overhead_fig}")
    if args.fragment_fidelity_sweep:
        frag_paths = _plot_fragment_fidelity_sweep(
            fragment_fidelity, benches, sizes, artifact_dir, f"{timestamp}{tag_suffix}"
        )
        for path in frag_paths:
            print(f"Wrote fragment fidelity figure: {path}")
    final_csv_path = artifact_dir / f"relative_properties_{timestamp}{tag_suffix}.csv"
    if all_rows:
        _write_rows_csv(final_csv_path, all_rows)
        print(f"Wrote relative properties CSV: {final_csv_path}")
    panel_out: List[Path] = []
    if args.plot_cached_panels_after_run:
        panel_sizes = [int(s.strip()) for s in str(args.panel_sizes).split(",") if s.strip()]
        panel_methods = [m.strip() for m in str(args.panel_methods).split(",") if m.strip()]
        if not panel_sizes:
            raise RuntimeError("--panel-sizes is empty.")
        if not panel_methods:
            raise RuntimeError("--panel-methods is empty.")
        panel_sim_csv = Path(args.panel_simtiming_csv) if str(args.panel_simtiming_csv).strip() else final_csv_path
        panel_timing_csv = (
            Path(args.panel_timing_csv)
            if str(args.panel_timing_csv).strip()
            else timing_path
        )
        if panel_timing_csv is None:
            raise RuntimeError(
                "--plot-cached-panels-after-run requires timing CSV. "
                "Enable --collect-timing or pass --panel-timing-csv."
            )
        if not str(args.panel_real_torino_csv).strip() or not str(args.panel_real_marrakesh_csv).strip():
            raise RuntimeError(
                "--plot-cached-panels-after-run requires --panel-real-torino-csv and "
                "--panel-real-marrakesh-csv."
            )
        panel_out = _plot_cached_panels(
            simtiming_csv=panel_sim_csv,
            real_torino_csv=Path(args.panel_real_torino_csv),
            real_marrakesh_csv=Path(args.panel_real_marrakesh_csv),
            timing_csv=panel_timing_csv,
            out_dir=artifact_dir,
            timestamp=timestamp,
            tag=tag,
            sizes=panel_sizes,
            methods=panel_methods,
        )
        for path in panel_out:
            print(f"Wrote figure: {path}")
    _publish_full_eval_views(
        out_dir=out_dir,
        artifact_dir=artifact_dir,
        panel_paths=panel_out,
        final_csv_path=final_csv_path if all_rows else None,
        timing_path=timing_path,
        backend_label=state_backend,
    )
    if use_cache:
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
    if _IDEAL_RESULT_CACHE is not None:
        try:
            print(f"Ideal-sim cache entries: {_IDEAL_RESULT_CACHE.count()}")
        except Exception:
            pass
    if args.with_real_fidelity and not _REAL_DRY_RUN:
        print(
            "Accumulated qpu_sec this run (submitted jobs only): "
            f"{_RUN_QPU_SEC:.2f} "
            f"(known_jobs={_RUN_QPU_SEC_KNOWN_JOBS}, unknown_jobs={_RUN_QPU_SEC_UNKNOWN_JOBS})"
        )
    print("Full evaluation finished successfully.")
    _cleanup_children()


if __name__ == "__main__":
    main()
