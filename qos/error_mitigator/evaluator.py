import importlib.util
import logging
import numbers
import os
import random
import time
import traceback
import multiprocessing as mp
import json
import ast
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

_SCORE_MODE_LEGACY = "legacy"
_SCORE_MODE_PIECEWISE = "piecewise"


def _is_eval_verbose() -> bool:
    raw = os.getenv("QOS_EVAL_VERBOSE", os.getenv("QOS_VERBOSE", ""))
    return raw.lower() in {"1", "true", "yes", "y"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _include_summary_artifact() -> bool:
    return _env_bool("QOSE_INCLUDE_SUMMARY_ARTIFACT", True)


def _include_cases_artifact() -> bool:
    return _env_bool("QOSE_INCLUDE_CASES_ARTIFACT", True)


def _failure_artifacts(info: str) -> dict:
    return {"info": info}


def _failure_result(info: str):
    return {"combined_score": -1000.0}, _failure_artifacts(info)


def _sample_pairs_target_count(benches: list[str]) -> int:
    samples_per_bench = int(os.getenv("QOSE_SAMPLES_PER_BENCH", "1"))
    if samples_per_bench <= 0:
        raise ValueError("QOSE_SAMPLES_PER_BENCH must be >= 1")
    return max(1, len(benches) * samples_per_bench)


def _fixed_bench_size_pairs_from_env(benches: list[str]) -> list[tuple[str, int]] | None:
    raw = os.getenv("QOSE_FIXED_BENCH_SIZE_PAIRS", "").strip()
    if not raw:
        return None

    payloads = [raw]
    # Be tolerant to shell-escaped doubled quotes like [[""qaoa_r3"",22], ...]
    if '""' in raw:
        payloads.append(raw.replace('""', '"'))

    parsed = None
    for text in payloads:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                break
            except Exception:
                continue
        if parsed is not None:
            break

    if parsed is None:
        raise ValueError(
            "QOSE_FIXED_BENCH_SIZE_PAIRS must be a list of [bench, size] pairs; "
            "example: [[\"qaoa_r3\",22],[\"bv\",20]]"
        )
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("QOSE_FIXED_BENCH_SIZE_PAIRS must parse to a list")

    valid_benches = {b for b, _ in BENCHES}
    enabled_benches = set(benches)
    fixed_pairs: list[tuple[str, int]] = []
    seen = set()

    for idx, item in enumerate(parsed):
        if isinstance(item, dict):
            bench = str(item.get("bench", "")).strip()
            size_raw = item.get("size", None)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            bench = str(item[0]).strip()
            size_raw = item[1]
        else:
            raise ValueError(
                f"QOSE_FIXED_BENCH_SIZE_PAIRS entry #{idx} must be [bench,size] or "
                "{'bench':..., 'size':...}"
            )

        if bench not in valid_benches:
            raise ValueError(f"Unknown bench in QOSE_FIXED_BENCH_SIZE_PAIRS: {bench}")
        if bench not in enabled_benches:
            raise ValueError(
                f"Bench {bench} is not enabled by QOSE_BENCHES for this run"
            )

        try:
            size = int(size_raw)
        except Exception:
            try:
                size = int(float(size_raw))
            except Exception:
                raise ValueError(
                    f"Invalid size in QOSE_FIXED_BENCH_SIZE_PAIRS for bench={bench}: {size_raw}"
                )

        pair = (bench, size)
        if pair in seen:
            continue

        try:
            _load_qasm_circuit(bench, size)
        except Exception as exc:
            raise ValueError(
                f"Invalid/unavailable circuit in QOSE_FIXED_BENCH_SIZE_PAIRS: ({bench},{size}): {exc}"
            )

        seen.add(pair)
        fixed_pairs.append(pair)

    if not fixed_pairs:
        raise ValueError("QOSE_FIXED_BENCH_SIZE_PAIRS is empty after parsing")
    return fixed_pairs


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


def _score_mode() -> str:
    raw = os.getenv("QOSE_SCORE_MODE", _SCORE_MODE_LEGACY).strip().lower()
    if raw in {"pwl", "piecewise", "piecewise_linear"}:
        return _SCORE_MODE_PIECEWISE
    return _SCORE_MODE_LEGACY


def _combined_score_from_ratios(
    avg_depth: float, avg_cnot: float, avg_run_time: float, avg_overhead: float
) -> tuple[float, str]:
    mode = _score_mode()
    if mode == _SCORE_MODE_PIECEWISE:
        # Piecewise-linear score (uses depth/cnot/time only):
        # - score is 0 when all ratios == 1
        # - stronger penalty when structure worsens (avg depth/cnot > 1)
        # - with these slopes, 5% depth/cnot increase needs ~80% time reduction to be > 0
        # - and 5% time increase needs ~5% depth/cnot reduction to be > 0
        struct_delta = 1.0 - ((avg_depth + avg_cnot) / 2.0)
        time_delta = 1.0 - avg_run_time
        slope_pos = 1
        slope_neg = 8
        struct_term = slope_pos * struct_delta if struct_delta >= 0 else slope_neg * struct_delta
        return struct_term + time_delta, mode
    # Legacy score.
    return -(avg_depth + avg_cnot + avg_overhead + avg_run_time), mode


def _build_args():
    return SimpleNamespace(
        size_to_reach=int(os.getenv("QOSE_SIZE_TO_REACH", "0")),
        ideal_size_to_reach=int(os.getenv("QOSE_IDEAL_SIZE_TO_REACH", "2")),
        budget=int(os.getenv("QOSE_BUDGET", "3")),
        qos_cost_search=True,
        collect_timing=False,
        metric_mode="fragment",
        metrics_baseline="torino",
        metrics_optimization_level=3,
    )

def _select_bench_size_pairs(args, benches):
    size_min = int(os.getenv("QOSE_SIZE_MIN", "12"))
    size_max = int(os.getenv("QOSE_SIZE_MAX", "24"))
    candidate_pairs = _collect_candidate_pairs(benches, size_min, size_max)
    if not candidate_pairs:
        return []
    samples_per_bench = int(os.getenv("QOSE_SAMPLES_PER_BENCH", "1"))
    if samples_per_bench <= 0:
        raise ValueError("QOSE_SAMPLES_PER_BENCH must be >= 1")
    sample_seed_raw = os.getenv("QOSE_SAMPLE_SEED", "").strip()
    sample_seed = int(sample_seed_raw) if sample_seed_raw else 123
    rng = random.Random(sample_seed)
    bench_to_sizes = {}
    for bench, size in candidate_pairs:
        bench_to_sizes.setdefault(bench, set()).add(size)

    bench_size_pairs = []
    for bench in benches:
        all_sizes = sorted(bench_to_sizes.get(bench, set()))
        if not all_sizes:
            raise ValueError(f"No valid sizes found for bench={bench}")

        # Keep per-bench sample size fixed and non-identical by qubit size.
        if len(all_sizes) < samples_per_bench:
            raise ValueError(
                f"Bench={bench} has only {len(all_sizes)} valid sizes, "
                f"cannot draw {samples_per_bench} distinct samples"
            )
        picked_sizes = rng.sample(all_sizes, samples_per_bench)
        for size in picked_sizes:
            bench_size_pairs.append((bench, size))
    return bench_size_pairs

def _collect_candidate_pairs(benches, size_min, size_max, size_step=1):
    candidate_pairs = []
    for bench in benches:
        for size in range(size_min, size_max + 1, max(1, size_step)):
            try:
                _load_qasm_circuit(bench, size)
            except Exception:
                continue
            candidate_pairs.append((bench, size))
    return candidate_pairs


def _evaluate_bench_size_pairs(
    evolved_cost_search,
    args,
    bench_size_pairs,
    include_cases: bool = True,
):
    depth_sum = cnot_sum = run_time_sum = 0.0
    qos_depth_sum = qos_cnot_sum = qos_run_time_sum = 0.0
    qose_depth_sum = qose_cnot_sum = 0.0
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
    overhead_sum = 0.0
    qose_overhead_sum = 0.0
    qos_overhead_sum = 0.0
    failure_traces = []
    bench_pairs_str = ", ".join(f"({b},{s})" for b, s in bench_size_pairs)
    logger.warning("Evaluating bench/size pairs: %s", bench_pairs_str)

    for bench, size in bench_size_pairs:
        current_stage = "load_qasm"
        try:
            qc = _load_qasm_circuit(bench, size)
            effective_size_to_reach = (
                size if args.size_to_reach <= 0 else args.size_to_reach
            )

            current_stage = "baseline_qos_run"
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
            # Baseline cost-search may execute in a child process. In that mode,
            # parent-side _gv/_wc counters can stay 0, so we count from traces.
            qos_gv_calls = 0
            qos_wc_calls = 0
            qos_cost_search_seen = 0
            qos_cost_search_orig = qos_mitigator.cost_search

            def _wrap_qos_cost_search(*c_args, **c_kwargs):
                nonlocal qos_gv_calls, qos_wc_calls, qos_cost_search_seen
                qos_cost_search_seen += 1
                prev_gv = int(getattr(qos_mitigator, "_gv_cost_calls", 0))
                prev_wc = int(getattr(qos_mitigator, "_wc_cost_calls", 0))
                size_out, method_out, cost_time_out, timed_out_out = qos_cost_search_orig(
                    *c_args, **c_kwargs
                )
                gv_trace = getattr(qos_mitigator, "_qose_gv_cost_trace", None)
                wc_trace = getattr(qos_mitigator, "_qose_wc_cost_trace", None)

                gv_len = None
                wc_len = None
                if gv_trace is not None and not isinstance(gv_trace, (str, bytes)):
                    try:
                        gv_len = len(gv_trace)
                    except TypeError:
                        gv_len = None
                if wc_trace is not None and not isinstance(wc_trace, (str, bytes)):
                    try:
                        wc_len = len(wc_trace)
                    except TypeError:
                        wc_len = None

                if gv_len is not None:
                    qos_gv_calls += gv_len
                else:
                    qos_gv_calls += max(
                        0, int(getattr(qos_mitigator, "_gv_cost_calls", 0)) - prev_gv
                    )
                if wc_len is not None:
                    qos_wc_calls += wc_len
                else:
                    qos_wc_calls += max(
                        0, int(getattr(qos_mitigator, "_wc_cost_calls", 0)) - prev_wc
                    )
                return size_out, method_out, cost_time_out, timed_out_out

            qos_mitigator.cost_search = _wrap_qos_cost_search
            t0 = time.perf_counter()
            qos_q = qos_mitigator.run(qos_q)
            qos_run_time = time.perf_counter() - t0
            if qos_cost_search_seen == 0:
                # No cost-search invocations observed; fallback to direct counters.
                qos_gv_calls = int(getattr(qos_mitigator, "_gv_cost_calls", 0))
                qos_wc_calls = int(getattr(qos_mitigator, "_wc_cost_calls", 0))
            total_qos_gv_calls += qos_gv_calls
            total_qos_wc_calls += qos_wc_calls
            current_stage = "baseline_qos_analyze"
            qos_m = _analyze_qernel(
                qos_q,
                args.metric_mode,
                args.metrics_baseline,
                args.metrics_optimization_level,
            )
            current_stage = "baseline_qos_extract"
            qos_circs = _extract_circuits(qos_q)
            qos_depth = qos_m["depth"]
            qos_cnot = qos_m["num_nonlocal_gates"]
            qos_num_circuits = len(qos_circs)

            current_stage = "qose_analysis"
            q = Qernel(qc.copy())
            BasicAnalysisPass().run(q)
            SupermarqFeaturesAnalysisPass().run(q)
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
                t0 = time.perf_counter()
                q = mitigator.run(q)
                run_time = time.perf_counter() - t0
            finally:
                GVOptimalDecompositionPass.cost = gv_cost_orig
                OptimalWireCuttingPass.cost = wc_cost_orig
            if getattr(mitigator, "_qose_cost_search_error", None):
                return {
                    "ok": False,
                    "error": f"Cost search failed: {mitigator._qose_cost_search_error}",
                }
            current_stage = "qose_analyze"
            qose_m = _analyze_qernel(
                q,
                args.metric_mode,
                args.metrics_baseline,
                args.metrics_optimization_level,
            )
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

        rel_depth = _safe_ratio(qose_depth, qos_depth)
        rel_cnot = _safe_ratio(qose_cnot, qos_cnot)
        rel_run_time = _safe_ratio(run_time, qos_run_time)
        qose_num_circuits = len(qose_circs)
        rel_overhead = _safe_ratio(qose_num_circuits, qos_num_circuits)
        depth_sum += rel_depth
        cnot_sum += rel_cnot
        run_time_sum += rel_run_time
        overhead_sum += rel_overhead
        qose_depth_sum += qose_depth
        qose_cnot_sum += qose_cnot
        qose_overhead_sum += qose_num_circuits
        qos_depth_sum += qos_depth
        qos_cnot_sum += qos_cnot
        qos_run_time_sum += qos_run_time
        qos_overhead_sum += qos_num_circuits
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
        if include_cases:
            cases.append(
                {
                    "bench": bench,
                    "size": size,
                    "qose_input_size": effective_size_to_reach,
                    "qose_depth": qose_depth,
                    "qos_depth": qos_depth,
                    "qose_cnot": qose_cnot,
                    "qos_cnot": qos_cnot,
                    "qose_num_circuits": qose_num_circuits,
                    "qos_num_circuits": qos_num_circuits,
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

    expected = len(bench_size_pairs)
    if count < expected:
        trace_blob = "\n\n".join(failure_traces)
        return {
            "ok": False,
            "error": (
                "Evaluation failed: "
                f"only {count}/{expected} successful runs.\n{trace_blob}"
            ),
        }

    avg_depth = depth_sum / count
    avg_cnot = cnot_sum / count
    avg_run_time = run_time_sum / count
    avg_overhead = overhead_sum / count
    combined_score_raw, score_mode = _combined_score_from_ratios(
        avg_depth, avg_cnot, avg_run_time, avg_overhead
    )
    return {
        "ok": True,
        "avg_depth": avg_depth,
        "avg_cnot": avg_cnot,
        "avg_overhead": avg_overhead,
        "avg_run_time": avg_run_time,
        "combined_score_raw": combined_score_raw,
        "score_mode": score_mode,
        "qose_run_sec_avg": (total_run_time / count) if count else 0.0,
        "qos_run_sec_avg": (total_qos_run_time / count) if count else 0.0,
        "gv_cost_calls_total": total_gv_calls,
        "wc_cost_calls_total": total_wc_calls,
        "qos_gv_cost_calls_total": total_qos_gv_calls,
        "qos_wc_cost_calls_total": total_qos_wc_calls,
        "qose_depth_avg": (qose_depth_sum / count) if count else 0.0,
        "qose_cnot_avg": (qose_cnot_sum / count) if count else 0.0,
        "qose_overhead_avg": (qose_overhead_sum / count) if count else 0.0,
        "qos_depth_avg": (qos_depth_sum / count) if count else 0.0,
        "qos_cnot_avg": (qos_cnot_sum / count) if count else 0.0,
        "qos_overhead_avg": (qos_overhead_sum / count) if count else 0.0,
        "cases": cases,
        "num_cases": count,
    }


def _evaluate_impl(program_path):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    evolved_cost_search = getattr(candidate, "evolved_cost_search", None)
    if evolved_cost_search is None:
        return _failure_result("Missing evolved_cost_search")

    args = _build_args()
    bench_env = os.getenv("QOSE_BENCHES", "").strip()
    if bench_env:
        benches = [b.strip() for b in bench_env.split(",") if b.strip()]
    else:
        benches = [b for b, _label in BENCHES]
    benches = list(dict.fromkeys(benches))

    valid_benches = {b for b, _label in BENCHES}
    unknown = [b for b in benches if b not in valid_benches]
    if unknown:
        return _failure_result(f"Unknown benches: {', '.join(unknown)}")

    size_min = int(os.getenv("QOSE_SIZE_MIN", "12"))
    size_max = int(os.getenv("QOSE_SIZE_MAX", "24"))
    try:
        fixed_pairs = _fixed_bench_size_pairs_from_env(benches)
    except ValueError as exc:
        return _failure_result(str(exc))

    candidate_pairs = _collect_candidate_pairs(benches, size_min, size_max)
    if not candidate_pairs and not fixed_pairs:
        return _failure_result("No valid (bench,size) pairs found")

    try:
        sample_pairs: list[tuple[str, int]] = []
        if fixed_pairs:
            sample_pairs = fixed_pairs
            logger.warning(
                "Using fixed bench/size pairs from QOSE_FIXED_BENCH_SIZE_PAIRS: %s",
                ", ".join(f"({b},{s})" for b, s in sample_pairs),
            )
        if not sample_pairs:
            sample_pairs = _select_bench_size_pairs(args, benches)
    except ValueError as exc:
        return _failure_result(str(exc))
    if not sample_pairs:
        return _failure_result("No valid (bench,size) pairs found")

    sample_res = _evaluate_bench_size_pairs(
        evolved_cost_search=evolved_cost_search,
        args=args,
        bench_size_pairs=sample_pairs,
        include_cases=True,
    )
    if not sample_res.get("ok", False):
        return _failure_result(sample_res.get("error", "Evaluation failed"))

    sample_combined_raw = float(sample_res["combined_score_raw"])
    score_mode = str(sample_res["score_mode"])
    combined_score_final = sample_combined_raw

    metrics = {
        "qose_depth": float(sample_res["avg_depth"]),
        "qose_cnot": float(sample_res["avg_cnot"]),
        "qose_overhead": float(sample_res["avg_overhead"]),
        "avg_run_time": float(sample_res["avg_run_time"]),
        # Final score used by evolution for ranking/comparison.
        "combined_score": float(combined_score_final),
    }

    summary = {
        "qose_budget": args.budget,
        "score_mode": score_mode,
        "qose_run_sec_avg": sample_res["qose_run_sec_avg"],
        "qos_run_sec_avg": sample_res["qos_run_sec_avg"],
        "gv_cost_calls_total": sample_res["gv_cost_calls_total"],
        "wc_cost_calls_total": sample_res["wc_cost_calls_total"],
        "qos_gv_cost_calls_total": sample_res["qos_gv_cost_calls_total"],
        "qos_wc_cost_calls_total": sample_res["qos_wc_cost_calls_total"],
        "qose_depth_avg": sample_res["qose_depth_avg"],
        "qose_cnot_avg": sample_res["qose_cnot_avg"],
        "qose_overhead_avg": sample_res["qose_overhead_avg"],
        "qos_depth_avg": sample_res["qos_depth_avg"],
        "qos_cnot_avg": sample_res["qos_cnot_avg"],
        "qos_overhead_avg": sample_res["qos_overhead_avg"],
    }
    artifacts = {}
    if _include_summary_artifact():
        artifacts["summary"] = summary
    if _include_cases_artifact():
        artifacts["cases"] = sample_res["cases"]
    return _round_float_values(metrics), _round_float_values(artifacts)


def evaluate(program_path):
    timeout_sec = int(os.getenv("QOSE_EVAL_TIMEOUT_SEC", "6000"))
    if timeout_sec <= 0:
        try:
            metrics, artifacts = _evaluate_impl(program_path)
        except Exception as exc:
            trace = traceback.format_exc()
            metrics = {"combined_score": -1000.0}
            artifacts = _failure_artifacts(f"Evaluation failed: {exc}\n{trace}")
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
            artifacts = _failure_artifacts(f"Evaluation timed out after {timeout_sec}s")
        else:
            try:
                result = result_queue.get_nowait()
            except Exception:
                metrics = {"combined_score": -1000.0}
                artifacts = _failure_artifacts("Evaluation failed: no result returned")
            else:
                if result.get("ok"):
                    metrics = result.get("metrics", {"combined_score": -1000.0})
                    artifacts = result.get("artifacts", {})
                else:
                    info = result.get("error", "Evaluation failed")
                    trace = result.get("trace", "")
                    metrics = {"combined_score": -1000.0}
                    artifacts = _failure_artifacts(f"{info}\n{trace}")
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


if __name__ == "__main__":
    pass
