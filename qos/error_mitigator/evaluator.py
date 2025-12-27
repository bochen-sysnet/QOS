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
    _analyze_circuit,
    _analyze_qernel,
    _extract_circuits,
    _run_mitigator,
)
from qos.error_mitigator.analyser import BasicAnalysisPass, SupermarqFeaturesAnalysisPass
from qos.error_mitigator.optimiser import FrozenQubitsPass
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
    evolved_run = getattr(candidate, "evolved_run", None)
    if evolved_run is None:
        return {"combined_score": 0.0}, {"error": "Missing evolved_run"}

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
        sample_count = int(os.getenv("QOSE_NUM_SAMPLES", "3"))
        seed = os.getenv("QOSE_SEED")
        rng = random.Random(int(seed)) if seed is not None else random.SystemRandom()
        if sample_count <= 1:
            benches = [rng.choice(bench_choices)]
        elif sample_count <= len(bench_choices):
            benches = rng.sample(bench_choices, sample_count)
        else:
            benches = [rng.choice(bench_choices) for _ in range(sample_count)]

    valid_benches = {b for b, _label in BENCHES}
    unknown = [b for b in benches if b not in valid_benches]
    if unknown:
        return {"combined_score": 0.0}, {"error": f"Unknown benches: {', '.join(unknown)}"}
    sizes = [int(s) for s in os.getenv("QOSE_SIZES", "12").split(",") if s]
    baseline_mode = os.getenv("QOSE_BASELINE", "qos").strip().lower()
    if baseline_mode not in {"qos", "qiskit"}:
        return {"combined_score": 0.0}, {"error": f"Unknown QOSE_BASELINE: {baseline_mode}"}

    rel_depth = rel_cnot = rel_overhead = 0.0
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

            if baseline_mode == "qiskit":
                base = _analyze_circuit(
                    qc,
                    args.metric_mode,
                    args.metrics_baseline,
                    args.metrics_optimization_level,
                )
            else:
                base = None
                qos_m, _t, qos_circs = _run_mitigator(qc, [], args)

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
            stage_counts = {}
            stage_order = []
            stage_names = (
                "computeCuttingCosts",
                "applyGV",
                "applyWC",
                "applyQR",
                "applyBestCut",
            )
            for stage_name in stage_names:
                if not hasattr(mitigator, stage_name):
                    continue
                orig = getattr(mitigator, stage_name)

                def _wrap_stage(orig_fn, name):
                    def _wrapped(*args, **kwargs):
                        stage_counts[name] = stage_counts.get(name, 0) + 1
                        stage_order.append(name)
                        return orig_fn(*args, **kwargs)

                    return _wrapped

                setattr(mitigator, stage_name, _wrap_stage(orig, stage_name))
            fq_orig = FrozenQubitsPass.run

            def _fq_run(self, *args, **kwargs):
                stage_counts["FrozenQubitsPass"] = stage_counts.get("FrozenQubitsPass", 0) + 1
                stage_order.append("FrozenQubitsPass")
                return fq_orig(self, *args, **kwargs)

            FrozenQubitsPass.run = _fq_run
            try:
                t0 = time.perf_counter()
                q = evolved_run(mitigator, q)
                run_time = time.perf_counter() - t0
            finally:
                FrozenQubitsPass.run = fq_orig
            qose_m = _analyze_qernel(
                q,
                args.metric_mode,
                args.metrics_baseline,
                args.metrics_optimization_level,
            )
            qose_circs = _extract_circuits(q)
            total_run_time += run_time

            if baseline_mode == "qiskit":
                rel_depth += qose_m["depth"] / max(1, base["depth"])
                rel_cnot += qose_m["num_nonlocal_gates"] / max(1, base["num_nonlocal_gates"])
                rel_overhead += float(max(1, len(qose_circs)))
                baseline_depth = base["depth"]
                baseline_cnot = base["num_nonlocal_gates"]
                baseline_label = "qiskit"
            else:
                rel_depth += qose_m["depth"] / max(1, qos_m["depth"])
                rel_cnot += qose_m["num_nonlocal_gates"] / max(
                    1, qos_m["num_nonlocal_gates"]
                )
                rel_overhead += len(qose_circs) / max(1, len(qos_circs))
                baseline_depth = qos_m["depth"]
                baseline_cnot = qos_m["num_nonlocal_gates"]
                baseline_label = "qos"
            cases.append(
                {
                    "bench": bench,
                    "size": size,
                    "baseline_mode": baseline_label,
                    "baseline_depth": baseline_depth,
                    "baseline_cnot": baseline_cnot,
                    "qose_depth": qose_m["depth"],
                    "qose_cnot": qose_m["num_nonlocal_gates"],
                    "qose_num_circuits": len(qose_circs),
                    "evolved_run_sec": run_time,
                    "input_features": input_features,
                    "stage_counts": stage_counts,
                    "stage_order": stage_order,
                }
            )
            count += 1

    if count == 0:
        return {"combined_score": 0.0}, {"error": "No benches/sizes"}

    rel_depth /= count
    rel_cnot /= count
    rel_overhead /= count

    combined_score = 1.0 / (1.0 + rel_depth + rel_cnot + rel_overhead/10.)
    metrics = {
        "rel_depth": rel_depth,
        "rel_cnot": rel_cnot,
        "rel_overhead": rel_overhead,
        "combined_score": combined_score,
    }
    artifacts = {
        "baseline_mode": baseline_mode,
        "evolved_run_sec_avg": (total_run_time / count) if count else 0.0,
        "cases": cases,
    }
    return metrics, artifacts


def _evaluate_worker(program_path, queue):
    try:
        metrics, artifacts = _evaluate_impl(program_path)
    except Exception as exc:
        metrics = {"combined_score": 0.0}
        artifacts = {"error": f"Evaluation failed: {exc}"}
    queue.put({"metrics": metrics, "artifacts": artifacts})


def evaluate(program_path):
    timeout_sec = int(os.getenv("QOSE_TIMEOUT_SEC", "300"))
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
            artifacts={"error": f"Timeout after {timeout_sec}s"},
        )
    if queue.empty():
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": "No result returned"},
        )
    result = queue.get()
    metrics = result.get("metrics", {"combined_score": 0.0})
    artifacts = result.get("artifacts", {})
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
