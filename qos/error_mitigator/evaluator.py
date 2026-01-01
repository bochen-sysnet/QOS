import importlib.util
import logging
import multiprocessing as mp
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

def _evaluate_impl(program_path):
    # Load candidate
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    evolved_cost_search = getattr(candidate, "evolved_cost_search", None)
    if evolved_cost_search is None:
        return {"combined_score": 0.0}, {"info": "Missing evolved_cost_search"}

    args = SimpleNamespace(
        size_to_reach=int(os.getenv("QOSE_SIZE_TO_REACH", "7")),
        ideal_size_to_reach=int(os.getenv("QOSE_IDEAL_SIZE_TO_REACH", "2")),
        budget=int(os.getenv("QOSE_BUDGET", "3")),
        qos_cost_search=True,
        collect_timing=False,
        timeout_sec=int(os.getenv("QOSE_TIMEOUT_SEC", "45")),
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
        seed = os.getenv("QOSE_SEED")
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
        return {"combined_score": 0.0}, {"info": f"Unknown benches: {', '.join(unknown)}"}
    sizes = [int(s) for s in os.getenv("QOSE_SIZES", "12").split(",") if s]
    depth_sum = cnot_sum = overhead_sum = 0.0
    count = 0
    cases = []

    feature_keys = [
        "depth",
        "num_qubits",
        "num_nonlocal_gates",
        "program_communication",
        "liveness",
        "parallelism",
        "measurement",
        "entanglement_ratio",
        "critical_depth",
    ]

    total_run_time = 0.0
    for bench in benches:
        for size in sizes:
            qc = _load_qasm_circuit(bench, size)

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
            orig_cost_search = mitigator.cost_search
            gv_cost_orig = GVOptimalDecompositionPass.cost
            wc_cost_orig = OptimalWireCuttingPass.cost

            def _wrap_cost_search(*args, **kwargs):
                t0 = time.perf_counter()
                size, costs, cost_time, timed_out = orig_cost_search(*args, **kwargs)
                dt = time.perf_counter() - t0
                mitigator._cost_search_calls += 1
                mitigator._cost_search_time += dt
                return size, costs, cost_time, timed_out

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

            depth_sum += qose_m["depth"]
            cnot_sum += qose_m["num_nonlocal_gates"]
            overhead_sum += len(qose_circs)
            cases.append(
                {
                    "bench": bench,
                    "size": size,
                    "qose_depth": qose_m["depth"],
                    "qose_cnot": qose_m["num_nonlocal_gates"],
                    "qose_num_circuits": len(qose_circs),
                    "evolved_run_sec": run_time,
                    "cost_search_sec": mitigator._cost_search_time,
                    "cost_search_calls": mitigator._cost_search_calls,
                    "gv_cost_calls": mitigator._gv_cost_calls,
                    "wc_cost_calls": mitigator._wc_cost_calls,
                    "input_features": input_features,
                }
            )
            count += 1

    if count == 0:
        return {"combined_score": 0.0}, {"info": "No benches/sizes"}

    avg_depth = depth_sum / count
    avg_cnot = cnot_sum / count
    avg_overhead = overhead_sum / count

    avg_run_time = (total_run_time / count) if count else 0.0
    combined_score = 1.0 / (
        1.0 + avg_depth + avg_cnot + (avg_overhead * 10.0) + avg_run_time
    )
    metrics = {
        "qose_depth": avg_depth,
        "qose_cnot": avg_cnot,
        "qose_overhead": avg_overhead,
        "avg_run_time": avg_run_time,
        "combined_score": combined_score,
    }
    avg_cost_search = 0.0
    total_cost_search_calls = sum(c["cost_search_calls"] for c in cases)
    if total_cost_search_calls:
        total_cost_search = sum(c["cost_search_sec"] for c in cases)
        avg_cost_search = total_cost_search / total_cost_search_calls
    artifacts = {
        "evolved_run_sec_avg": (total_run_time / count) if count else 0.0,
        "cost_search_sec_avg": avg_cost_search,
        "gv_cost_calls_total": sum(c["gv_cost_calls"] for c in cases),
        "wc_cost_calls_total": sum(c["wc_cost_calls"] for c in cases),
        "cases": cases,
    }
    return metrics, artifacts


def _evaluate_worker(program_path, queue):
    try:
        metrics, artifacts = _evaluate_impl(program_path)
    except Exception as exc:
        metrics = {"combined_score": 0.0}
        artifacts = {"info": f"Evaluation failed: {exc}"}
    queue.put({"metrics": metrics, "artifacts": artifacts})


def evaluate(program_path):
    timeout_sec = int(os.getenv("QOSE_TIMEOUT_SEC", "100"))
    if timeout_sec <= 0:
        metrics, artifacts = _evaluate_impl(program_path)
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    queue = mp.Queue()
    proc = mp.Process(target=_evaluate_worker, args=(program_path, queue))
    proc.start()
    proc.join(timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"info": f"Timeout after {timeout_sec}s"},
        )
    if queue.empty():
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"info": "No result returned"},
        )
    result = queue.get()
    metrics = result.get("metrics", {"combined_score": 0.0})
    artifacts = result.get("artifacts", {})
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
