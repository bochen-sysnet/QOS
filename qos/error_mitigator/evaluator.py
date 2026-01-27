import importlib.util
import logging
import numbers
import os
import random
import time
import traceback
import multiprocessing as mp
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
logger = logging.getLogger(__name__)

def _is_eval_verbose() -> bool:
    raw = os.getenv("QOS_EVAL_VERBOSE", os.getenv("QOS_VERBOSE", ""))
    return raw.lower() in {"1", "true", "yes", "y"}

logger.setLevel(logging.INFO if _is_eval_verbose() else logging.WARNING)

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
        size_to_reach=int(os.getenv("QOSE_SIZE_TO_REACH", "0")),
        ideal_size_to_reach=int(os.getenv("QOSE_IDEAL_SIZE_TO_REACH", "2")),
        budget=int(os.getenv("QOSE_BUDGET", "3")),
        qos_cost_search=True,
        collect_timing=False,
        metric_mode="fragment",
        metrics_baseline="torino",
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

    size_min = int(os.getenv("QOSE_SIZE_MIN", "12"))
    size_max = int(os.getenv("QOSE_SIZE_MAX", "24"))
    candidate_pairs = []
    for bench in benches:
        for size in range(size_min, size_max + 1):
            try:
                _load_qasm_circuit(bench, size)
            except Exception:
                continue
            candidate_pairs.append((bench, size))
    if not candidate_pairs:
        return {"combined_score": -1000.0}, {"info": "No valid (bench,size) pairs found"}
    pair_count = int(os.getenv("QOSE_NUM_SAMPLES", "10"))
    if seed is None:
        rng = random.Random()
    else:
        rng = random.Random(int(seed))
    if pair_count <= 1:
        bench_size_pairs = [rng.choice(candidate_pairs)]
    elif pair_count <= len(candidate_pairs):
        bench_size_pairs = rng.sample(candidate_pairs, pair_count)
    else:
        bench_size_pairs = [rng.choice(candidate_pairs) for _ in range(pair_count)]
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
    total_gv_calls = 0
    total_wc_calls = 0
    total_qos_gv_calls = 0
    total_qos_wc_calls = 0
    failure_traces = []
    for bench, size in bench_size_pairs:
        if count >= pair_count:
            break
        logger.warning("Evaluating bench=%s size=%s", bench, size)
        current_stage = "load_qasm"
        try:
            qc = _load_qasm_circuit(bench, size)
            effective_size_to_reach = (
                size if args.size_to_reach <= 0 else args.size_to_reach
            )

            # Baseline QOS
            current_stage = "baseline_qos_run"
            logger.warning("Baseline QOS run start bench=%s size=%s", bench, size)
            qos_q = Qernel(qc.copy())
            qos_mitigator = ErrorMitigator(
                size_to_reach=effective_size_to_reach,
                ideal_size_to_reach=args.ideal_size_to_reach,
                budget=args.budget,
                methods=[],
                use_cost_search=args.qos_cost_search,
                collect_timing=False,
            )
            qos_mitigator._gv_cost_calls = 0
            qos_mitigator._wc_cost_calls = 0
            t0 = time.perf_counter()
            qos_q = qos_mitigator.run(qos_q)
            qos_run_time = time.perf_counter() - t0
            logger.warning(
                "Baseline QOS run done bench=%s size=%s sec=%.2f",
                bench,
                size,
                qos_run_time,
            )
            total_qos_gv_calls += getattr(qos_mitigator, "_gv_cost_calls", 0)
            total_qos_wc_calls += getattr(qos_mitigator, "_wc_cost_calls", 0)
            current_stage = "baseline_qos_analyze"
            logger.warning("Baseline QOS analyze start bench=%s size=%s", bench, size)
            qos_m = _analyze_qernel(
                qos_q,
                args.metric_mode,
                args.metrics_baseline,
                args.metrics_optimization_level,
            )
            logger.warning("Baseline QOS analyze done bench=%s size=%s", bench, size)
            current_stage = "baseline_qos_extract"
            logger.warning("Baseline QOS extract start bench=%s size=%s", bench, size)
            qos_circs = _extract_circuits(qos_q)
            logger.warning("Baseline QOS extract done bench=%s size=%s", bench, size)

            # Evolved QOSE
            current_stage = "qose_analysis"
            logger.warning("QOSE analysis start bench=%s size=%s", bench, size)
            q = Qernel(qc.copy())
            BasicAnalysisPass().run(q)
            SupermarqFeaturesAnalysisPass().run(q)
            logger.warning("QOSE analysis done bench=%s size=%s", bench, size)
            input_meta = q.get_metadata()
            input_features = {k: input_meta.get(k) for k in feature_keys if k in input_meta}
            mitigator = ErrorMitigator(
                size_to_reach=effective_size_to_reach,
                ideal_size_to_reach=args.ideal_size_to_reach,
                budget=args.budget,
                methods=[],
                use_cost_search=args.qos_cost_search,
                collect_timing=True,
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
            mitigator._qose_gv_time_trace = None
            mitigator._qose_wc_time_trace = None
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
                current_stage = "qose_run"
                logger.warning("QOSE run start bench=%s size=%s", bench, size)
                t0 = time.perf_counter()
                q = mitigator.run(q)
                run_time = time.perf_counter() - t0
            finally:
                GVOptimalDecompositionPass.cost = gv_cost_orig
                OptimalWireCuttingPass.cost = wc_cost_orig
            if getattr(mitigator, "_qose_cost_search_error", None):
                return (
                    {"combined_score": -1000.0},
                    {"info": f"Cost search failed: {mitigator._qose_cost_search_error}"},
                )
            logger.warning("QOSE run done bench=%s size=%s sec=%.2f", bench, size, run_time)
            timings = getattr(mitigator, "timings", {}) or {}
            total_time = float(timings.get("total", run_time))
            analysis_time = float(timings.get("analysis", 0.0))
            qaoa_analysis_time = float(timings.get("qaoa_analysis", 0.0))
            qf_time = float(timings.get("qf", 0.0))
            cut_select_time = float(timings.get("cut_select", 0.0))
            cost_search_time = float(timings.get("cost_search", 0.0))
            gv_time = float(timings.get("gv", 0.0))
            wc_time = float(timings.get("wc", 0.0))
            qr_time = float(timings.get("qr", 0.0))
            accounted = (
                analysis_time
                + qaoa_analysis_time
                + qf_time
                + cut_select_time
                + cost_search_time
                + gv_time
                + wc_time
                + qr_time
            )
            other_time = max(0.0, total_time - accounted)
            logger.warning(
                "QOSE timing bench=%s size=%s total=%.2f analysis=%.2f qaoa=%.2f "
                "qf=%.2f cut_select=%.2f cost_search=%.2f gv=%.2f wc=%.2f qr=%.2f other=%.2f",
                bench,
                size,
                total_time,
                analysis_time,
                qaoa_analysis_time,
                qf_time,
                cut_select_time,
                cost_search_time,
                gv_time,
                wc_time,
                qr_time,
                other_time,
            )
            current_stage = "qose_analyze"
            logger.warning("QOSE analyze start bench=%s size=%s", bench, size)
            qose_m = _analyze_qernel(
                q,
                args.metric_mode,
                args.metrics_baseline,
                args.metrics_optimization_level,
            )
            logger.warning("QOSE analyze done bench=%s size=%s", bench, size)
            qose_circs = _extract_circuits(q)
        except Exception as exc:
            trace = traceback.format_exc()
            failure_traces.append(
                f"Stage={current_stage} bench={bench} size={size}\n{trace}"
            )
            logger.error(
                "Evaluation failed bench=%s size=%s stage=%s: %s\n%s",
                bench,
                size,
                current_stage,
                exc,
                trace,
            )
            continue
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
        gv_calls = mitigator._gv_cost_calls
        wc_calls = mitigator._wc_cost_calls
        gv_trace = mitigator._qose_gv_cost_trace
        wc_trace = mitigator._qose_wc_cost_trace
        if gv_trace is not None and not isinstance(gv_trace, (str, bytes)):
            try:
                gv_calls = len(gv_trace)
            except TypeError:
                pass
        if wc_trace is not None and not isinstance(wc_trace, (str, bytes)):
            try:
                wc_calls = len(wc_trace)
            except TypeError:
                pass
        total_gv_calls += gv_calls
        total_wc_calls += wc_calls
        cases.append(
            {
                "bench": bench,
                "size": size,
                "qose_input_size": effective_size_to_reach,
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
                "qose_gv_time_trace": mitigator._qose_gv_time_trace,
                "qose_wc_time_trace": mitigator._qose_wc_time_trace,
                "input_features": input_features,
            }
        )
        count += 1

    if count < pair_count:
        trace_blob = "\n\n".join(failure_traces)
        return (
            {"combined_score": -1000.0},
            {
                "info": (
                    "Evaluation failed: "
                    f"only {count}/{pair_count} successful runs.\n{trace_blob}"
                )
            },
        )

    avg_depth = depth_sum / count
    avg_cnot = cnot_sum / count
    avg_overhead = overhead_sum / count
    avg_run_time = run_time_sum / count
    combined_score = -(avg_depth + avg_cnot + avg_overhead + avg_run_time)
    metrics = {
        "qose_depth": avg_depth,
        "qose_cnot": avg_cnot,
        "qose_overhead": avg_overhead,
        "avg_run_time": avg_run_time,
        "combined_score": combined_score,
    }
    artifacts = {
        "qose_budget": args.budget,
        "qose_run_sec_avg": (total_run_time / count) if count else 0.0,
        "qos_run_sec_avg": (total_qos_run_time / count) if count else 0.0,
        "gv_cost_calls_total": total_gv_calls,
        "wc_cost_calls_total": total_wc_calls,
        "qos_gv_cost_calls_total": total_qos_gv_calls,
        "qos_wc_cost_calls_total": total_qos_wc_calls,
        "qos_depth_avg": (qos_depth_sum / count) if count else 0.0,
        "qos_cnot_avg": (qos_cnot_sum / count) if count else 0.0,
        "qos_overhead_avg": (qos_overhead_sum / count) if count else 0.0,
        "cases": cases,
    }
    return _round_float_values(metrics), _round_float_values(artifacts)


def evaluate(program_path):
    timeout_sec = int(os.getenv("QOSE_EVAL_TIMEOUT_SEC", "2400"))
    if timeout_sec <= 0:
        try:
            metrics, artifacts = _evaluate_impl(program_path)
        except Exception as exc:
            trace = traceback.format_exc()
            metrics = {"combined_score": -1000.0}
            artifacts = {"info": f"Evaluation failed: {exc}\n{trace}"}
            logger.error("Evaluation failed with traceback:\n%s", trace)
    else:
        mp_ctx = (
            mp.get_context("fork")
            if "fork" in mp.get_all_start_methods()
            else mp.get_context()
        )
        result_queue = mp_ctx.Queue()
        proc = mp_ctx.Process(
            target=_evaluate_worker, args=(program_path, result_queue)
        )
        proc.start()
        proc.join(timeout_sec)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            metrics = {"combined_score": -1000.0}
            artifacts = {"info": f"Evaluation timed out after {timeout_sec}s"}
        else:
            try:
                result = result_queue.get_nowait()
            except Exception:
                metrics = {"combined_score": -1000.0}
                artifacts = {"info": "Evaluation failed: no result returned"}
            else:
                if result.get("ok"):
                    metrics = result.get("metrics", {"combined_score": -1000.0})
                    artifacts = result.get("artifacts", {})
                else:
                    info = result.get("error", "Evaluation failed")
                    trace = result.get("trace", "")
                    metrics = {"combined_score": -1000.0}
                    artifacts = {"info": f"{info}\n{trace}"}
    if float(metrics.get("combined_score", 0.0)) <= -999.0 and "info" in artifacts:
        metrics = dict(metrics)
        metrics["failure_reason"] = artifacts.get("info")
        logging.getLogger(__name__).warning(
            "Evaluation returned low score: %s", artifacts.get("info")
        )
    return EvaluationResult(
        metrics=_round_float_values(metrics),
        artifacts=_round_float_values(artifacts),
    )


def _evaluate_worker(program_path, result_queue):
    try:
        metrics, artifacts = _evaluate_impl(program_path)
        result_queue.put({"ok": True, "metrics": metrics, "artifacts": artifacts})
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "error": f"Evaluation failed: {exc}",
                "trace": traceback.format_exc(),
            }
        )
