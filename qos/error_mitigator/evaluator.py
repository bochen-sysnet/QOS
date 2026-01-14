import importlib.util
import logging
import numbers
import os
import random
import time
from types import SimpleNamespace
from openevolve.evaluation_result import EvaluationResult

from evaluation.full_eval import (
    BENCHES,
    _load_qasm_circuit,
    _analyze_qernel,
    _extract_circuits,
)
from qos.error_mitigator.analyser import BasicAnalysisPass, SupermarqFeaturesAnalysisPass
from qos.error_mitigator.optimiser import GVOptimalDecompositionPass, OptimalWireCuttingPass
from qos.error_mitigator.run import ErrorMitigator
from qos.types.types import Qernel

logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("qiskit.transpiler").setLevel(logging.WARNING)
logging.getLogger("qiskit.compiler").setLevel(logging.WARNING)
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)

def _round_float_values(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        return round(float(value), 4)
    if isinstance(value, dict):
        return {k: _round_float_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_float_values(v) for v in value]
    return value

def _safe_ratio(numerator, denominator):
    if denominator <= 0:
        return float(numerator)
    return float(numerator) / float(denominator)

def _evaluate_impl(program_path):
    # Load candidate
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    evolved_cost_search = getattr(candidate, "evolved_cost_search", None)
    if evolved_cost_search is None:
        return {"combined_score": -1000.0}, {"info": "Missing evolved_cost_search"}

    args = SimpleNamespace(
        size_to_reach=int(os.getenv("QOSE_SIZE_TO_REACH", "7")),
        ideal_size_to_reach=int(os.getenv("QOSE_IDEAL_SIZE_TO_REACH", "2")),
        budget=int(os.getenv("QOSE_BUDGET", "3")),
        qos_cost_search=True,
        collect_timing=False,
        metric_mode="fragment",
        metrics_baseline="kolkata",
        metrics_optimization_level=3,
    )
    bench_env = os.getenv("QOSE_BENCHES", "").strip()
    if bench_env:
        benches = [b.strip() for b in bench_env.split(",") if b.strip()]
    else:
        bench_choices = [b for b, _label in BENCHES]
        sample_count = int(os.getenv("QOSE_NUM_SAMPLES", "9"))
        seed = None
        if seed is None:
            if sample_count <= 1:
                benches = [bench_choices[0]]
            elif sample_count <= len(bench_choices):
                benches = bench_choices[:sample_count]
            else:
                benches = [
                    bench_choices[idx % len(bench_choices)] for idx in range(sample_count)
                ]
        else:
            rng = random.Random(int(seed))
            if sample_count <= 1:
                benches = [rng.choice(bench_choices)]
            elif sample_count <= len(bench_choices):
                benches = rng.sample(bench_choices, sample_count)
            else:
                benches = [rng.choice(bench_choices) for _ in range(sample_count)]

    valid_benches = {b for b, _label in BENCHES}
    unknown = [b for b in benches if b not in valid_benches]
    if unknown:
        return {"combined_score": -1000.0}, {"info": f"Unknown benches: {', '.join(unknown)}"}
    sizes = [int(s) for s in os.getenv("QOSE_SIZES", "12").split(",") if s]
    depth_sum = cnot_sum = overhead_sum = run_time_sum = 0.0
    qos_depth_sum = qos_cnot_sum = qos_overhead_sum = qos_run_time_sum = 0.0
    count = 0
    cases = []

    feature_keys = [
        "depth",
        "num_qubits",
        "num_clbits",
        "num_nonlocal_gates",
        "num_connected_components",
        "number_instructions",
        "num_measurements",
        "num_cnot_gates",
        "program_communication",
        "liveness",
        "parallelism",
        "measurement",
        "entanglement_ratio",
        "critical_depth",
    ]

    total_run_time = 0.0
    total_qos_run_time = 0.0
    for bench in benches:
        for size in sizes:
            qc = _load_qasm_circuit(bench, size)

            # Baseline QOS
            qos_q = Qernel(qc.copy())
            qos_mitigator = ErrorMitigator(
                size_to_reach=args.size_to_reach,
                ideal_size_to_reach=args.ideal_size_to_reach,
                budget=args.budget,
                methods=[],
                use_cost_search=args.qos_cost_search,
                collect_timing=False,
            )
            t0 = time.perf_counter()
            qos_q = qos_mitigator.run(qos_q)
            qos_run_time = time.perf_counter() - t0
            qos_m = _analyze_qernel(
                qos_q,
                args.metric_mode,
                args.metrics_baseline,
                args.metrics_optimization_level,
            )
            qos_circs = _extract_circuits(qos_q)

            # Evolved QOSE
            q = Qernel(qc.copy())
            BasicAnalysisPass().run(q)
            SupermarqFeaturesAnalysisPass().run(q)
            input_meta = q.get_metadata()
            input_features = {k: input_meta.get(k) for k in feature_keys if k in input_meta}
            mitigator = ErrorMitigator(
                size_to_reach=args.size_to_reach,
                ideal_size_to_reach=args.ideal_size_to_reach,
                budget=args.budget,
                methods=[],
                use_cost_search=args.qos_cost_search,
                collect_timing=False,
            )
            mitigator._cost_search_impl = evolved_cost_search.__get__(
                mitigator, ErrorMitigator
            )
            mitigator._gv_cost_calls = 0
            mitigator._wc_cost_calls = 0
            mitigator._cost_search_calls = 0
            mitigator._cost_search_time = 0.0
            mitigator._qose_cost_search_input_size = None
            mitigator._qose_cost_search_budget = None
            mitigator._qose_cost_search_output_size = None
            mitigator._qose_cost_search_method = None
            mitigator._qose_gv_cost_trace = None
            mitigator._qose_wc_cost_trace = None
            orig_cost_search = mitigator.cost_search
            gv_cost_orig = GVOptimalDecompositionPass.cost
            wc_cost_orig = OptimalWireCuttingPass.cost

            def _wrap_cost_search(*args, **kwargs):
                t0 = time.perf_counter()
                size, method, cost_time, timed_out = orig_cost_search(*args, **kwargs)
                dt = time.perf_counter() - t0
                mitigator._cost_search_calls += 1
                mitigator._cost_search_time += dt
                return size, method, cost_time, timed_out

            mitigator.cost_search = _wrap_cost_search
            def _gv_cost(self, *args, **kwargs):
                mitigator._gv_cost_calls += 1
                return gv_cost_orig(self, *args, **kwargs)

            def _wc_cost(self, *args, **kwargs):
                mitigator._wc_cost_calls += 1
                return wc_cost_orig(self, *args, **kwargs)

            GVOptimalDecompositionPass.cost = _gv_cost
            OptimalWireCuttingPass.cost = _wc_cost
            try:
                t0 = time.perf_counter()
                q = mitigator.run(q)
                run_time = time.perf_counter() - t0
            finally:
                GVOptimalDecompositionPass.cost = gv_cost_orig
                OptimalWireCuttingPass.cost = wc_cost_orig
            qose_m = _analyze_qernel(
                q,
                args.metric_mode,
                args.metrics_baseline,
                args.metrics_optimization_level,
            )
            qose_circs = _extract_circuits(q)
            total_run_time += run_time
            total_qos_run_time += qos_run_time

            qose_depth = qose_m["depth"]
            qose_cnot = qose_m["num_nonlocal_gates"]
            qose_overhead = len(qose_circs)
            qos_depth = qos_m["depth"]
            qos_cnot = qos_m["num_nonlocal_gates"]
            qos_overhead = len(qos_circs)

            rel_depth = _safe_ratio(qose_depth, qos_depth)
            rel_cnot = _safe_ratio(qose_cnot, qos_cnot)
            rel_overhead = _safe_ratio(qose_overhead, qos_overhead)
            rel_run_time = _safe_ratio(run_time, qos_run_time)

            depth_sum += rel_depth
            cnot_sum += rel_cnot
            overhead_sum += rel_overhead
            run_time_sum += rel_run_time
            qos_depth_sum += qos_depth
            qos_cnot_sum += qos_cnot
            qos_overhead_sum += qos_overhead
            qos_run_time_sum += qos_run_time
            cases.append(
                {
                    "bench": bench,
                    "size": size,
                    "qose_depth": qose_depth,
                    "qos_depth": qos_depth,
                    "qose_cnot": qose_cnot,
                    "qos_cnot": qos_cnot,
                    "qose_num_circuits": qose_overhead,
                    "qos_num_circuits": qos_overhead,
                    "qose_run_sec": run_time,
                    "qos_run_sec": qos_run_time,
                    "qose_output_size": mitigator._qose_cost_search_output_size,
                    "qose_method": mitigator._qose_cost_search_method,
                    "qose_gv_cost_trace": mitigator._qose_gv_cost_trace,
                    "qose_wc_cost_trace": mitigator._qose_wc_cost_trace,
                    "gv_cost_calls": mitigator._gv_cost_calls,
                    "wc_cost_calls": mitigator._wc_cost_calls,
                    "input_features": input_features,
                }
            )
            count += 1

    if count == 0:
        return {"combined_score": -1000.0}, {"info": "No benches/sizes"}

    avg_depth = depth_sum / count
    avg_cnot = cnot_sum / count
    avg_overhead = overhead_sum / count
    avg_run_time = run_time_sum / count
    combined_score = -(
        avg_depth + avg_cnot + avg_overhead + avg_run_time * 0.1
    )
    metrics = {
        "qose_depth": avg_depth,
        "qose_cnot": avg_cnot,
        "qose_overhead": avg_overhead,
        "avg_run_time": avg_run_time,
        "combined_score": combined_score,
    }
    artifacts = {
        "qose_input_size": args.size_to_reach,
        "qose_budget": args.budget,
        "qose_run_sec_avg": (total_run_time / count) if count else 0.0,
        "qos_run_sec_avg": (total_qos_run_time / count) if count else 0.0,
        "gv_cost_calls_total": sum(c["gv_cost_calls"] for c in cases),
        "wc_cost_calls_total": sum(c["wc_cost_calls"] for c in cases),
        "qos_depth_avg": (qos_depth_sum / count) if count else 0.0,
        "qos_cnot_avg": (qos_cnot_sum / count) if count else 0.0,
        "qos_overhead_avg": (qos_overhead_sum / count) if count else 0.0,
        "cases": cases,
    }
    return _round_float_values(metrics), _round_float_values(artifacts)


def evaluate(program_path):
    try:
        metrics, artifacts = _evaluate_impl(program_path)
    except Exception as exc:
        metrics = {"combined_score": -1000.0}
        artifacts = {"info": f"Evaluation failed: {exc}"}
    return EvaluationResult(
        metrics=_round_float_values(metrics),
        artifacts=_round_float_values(artifacts),
    )
