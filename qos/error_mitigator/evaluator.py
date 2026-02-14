import importlib.util
import logging
import numbers
import os
import random
import time
import traceback
import multiprocessing as mp
import csv
import hashlib
import json
import fcntl
from pathlib import Path
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("qiskit.transpiler").setLevel(logging.WARNING)
logging.getLogger("qiskit.compiler").setLevel(logging.WARNING)
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_SCORE_MODE_LEGACY = "legacy"
_SCORE_MODE_PIECEWISE = "piecewise"
_SURROGATE_MIN_TRAIN_ROWS_DEFAULT = 5

_SURROGATE_FEATURE_COLUMNS = [
    "sample_qose_depth",
    "sample_qose_cnot",
    "sample_qose_overhead",
    "sample_avg_run_time",
    "sample_combined_score_raw",
]

def _is_eval_verbose() -> bool:
    raw = os.getenv("QOS_EVAL_VERBOSE", os.getenv("QOS_VERBOSE", ""))
    return raw.lower() in {"1", "true", "yes", "y"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _surrogate_enabled() -> bool:
    return _env_bool("QOSE_SURROGATE_ENABLE", True)


def _surrogate_warmup_iters() -> int:
    return max(0, int(os.getenv("QOSE_SURROGATE_WARMUP_ITERS", "20")))


def _surrogate_ml_model_name() -> str:
    return os.getenv("QOSE_SURROGATE_ML_MODEL", "random_forest").strip().lower()


def _surrogate_min_train_rows() -> int:
    return max(
        2,
        int(os.getenv("QOSE_SURROGATE_MIN_TRAIN_ROWS", str(_SURROGATE_MIN_TRAIN_ROWS_DEFAULT))),
    )


def _surrogate_state_csv() -> Path:
    p = os.getenv(
        "QOSE_SURROGATE_STATE_CSV",
        "openevolve_output/baselines/qose_surrogate_state.csv",
    )
    return Path(p)


def _surrogate_meta_json(csv_path: Path) -> Path:
    raw = os.getenv("QOSE_SURROGATE_META_JSON", "").strip()
    if raw:
        return Path(raw)
    return csv_path.with_suffix(csv_path.suffix + ".meta.json")


def _program_hash(program_path: str) -> str:
    try:
        data = Path(program_path).read_bytes()
    except Exception:
        data = str(program_path).encode("utf-8", errors="ignore")
    return hashlib.sha1(data).hexdigest()


def _with_file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = lock_path.open("a+")
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    return f


def _claim_warmup_slot(meta_json: Path, warmup_iters: int) -> tuple[bool, int]:
    if warmup_iters <= 0:
        return False, 0
    lock = _with_file_lock(meta_json.with_suffix(meta_json.suffix + ".lock"))
    try:
        if meta_json.exists():
            try:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        else:
            meta = {}
        claimed = int(meta.get("full_eval_claimed", 0))
        if claimed < warmup_iters:
            claimed += 1
            meta["full_eval_claimed"] = claimed
            meta_json.parent.mkdir(parents=True, exist_ok=True)
            meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return True, claimed
        return False, claimed
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _append_surrogate_row(csv_path: Path, row: dict) -> None:
    lock = _with_file_lock(csv_path.with_suffix(csv_path.suffix + ".lock"))
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        fields = [
            "program_hash",
            "timestamp_sec",
            "sample_qose_depth",
            "sample_qose_cnot",
            "sample_qose_overhead",
            "sample_avg_run_time",
            "sample_combined_score_raw",
            "global_combined_score",
            "score_mode",
        ]
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fields})
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _load_surrogate_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(value, default=float("nan")):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _fit_surrogate_models(rows: list[dict], ml_model_name: str):
    labeled = [r for r in rows if not (str(r.get("global_combined_score", "")).strip() in {"", "nan"})]
    if len(labeled) < _surrogate_min_train_rows():
        return None, None, len(labeled)

    x = []
    y = []
    for r in labeled:
        feat = [_safe_float(r.get(c)) for c in _SURROGATE_FEATURE_COLUMNS]
        if any(val != val for val in feat):
            continue
        target = _safe_float(r.get("global_combined_score"))
        if target != target:
            continue
        x.append(feat)
        y.append(target)
    if len(x) < _surrogate_min_train_rows():
        return None, None, len(x)

    corr_model = LinearRegression()
    corr_model.fit([[row[4]] for row in x], y)  # use raw sampled score only

    if ml_model_name == "random_forest":
        est = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    else:
        est = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    ml_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", est),
        ]
    )
    ml_model.fit(x, y)
    return corr_model, ml_model, len(x)


def _predict_surrogate_scores(
    corr_model, ml_model, sample_features: dict
) -> tuple[float, float]:
    feat = [_safe_float(sample_features.get(c)) for c in _SURROGATE_FEATURE_COLUMNS]
    if any(v != v for v in feat):
        return float("nan"), float("nan")
    pred_corr = float("nan")
    pred_ml = float("nan")
    try:
        if corr_model is not None:
            pred_corr = float(corr_model.predict([[feat[4]]])[0])
    except Exception:
        pred_corr = float("nan")
    try:
        if ml_model is not None:
            pred_ml = float(ml_model.predict([feat])[0])
    except Exception:
        pred_ml = float("nan")
    return pred_corr, pred_ml

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


def _score_metadata(mode: str) -> dict[str, str]:
    if mode == _SCORE_MODE_PIECEWISE:
        return {
            "score_formula": (
                "struct_delta=1-((qose_depth+qose_cnot)/2); time_delta=1-avg_run_time; "
                "struct_term=1*struct_delta if struct_delta>=0 else 8*struct_delta; "
                "combined_score=struct_term+time_delta"
            ),
        }
    return {
        "score_formula": "combined_score=-(qose_depth+qose_cnot+qose_overhead+avg_run_time)",
    }

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
    stratified_sizes = os.getenv("QOSE_STRATIFIED_SIZES", "1").strip().lower() in {"1", "true", "yes", "y"}
    samples_per_bench = int(os.getenv("QOSE_SAMPLES_PER_BENCH", "1"))
    if samples_per_bench <= 0:
        raise ValueError("QOSE_SAMPLES_PER_BENCH must be >= 1")
    distinct_sizes = (
        os.getenv("QOSE_DISTINCT_SIZES_PER_BENCH", "1").strip().lower()
        in {"1", "true", "yes", "y"}
    )
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

        # Keep old stratified preference when requested.
        preferred_sizes = all_sizes
        if stratified_sizes:
            parity_sizes = [s for s in all_sizes if (s - size_min) % 2 == 0]
            if parity_sizes:
                preferred_sizes = parity_sizes

        sample_pool = preferred_sizes
        if distinct_sizes and len(sample_pool) < samples_per_bench and len(all_sizes) >= samples_per_bench:
            sample_pool = all_sizes
        if distinct_sizes and len(sample_pool) < samples_per_bench:
            raise ValueError(
                f"Bench={bench} has only {len(sample_pool)} valid sizes, "
                f"cannot draw {samples_per_bench} distinct samples"
            )

        if distinct_sizes:
            picked_sizes = rng.sample(sample_pool, samples_per_bench)
        else:
            picked_sizes = [rng.choice(sample_pool) for _ in range(samples_per_bench)]
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
            t0 = time.perf_counter()
            qos_q = qos_mitigator.run(qos_q)
            qos_run_time = time.perf_counter() - t0
            qos_gv_calls = getattr(qos_mitigator, "_gv_cost_calls", 0)
            qos_wc_calls = getattr(qos_mitigator, "_wc_cost_calls", 0)
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
        return {"combined_score": -1000.0}, {"info": "Missing evolved_cost_search"}

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
        return {"combined_score": -1000.0}, {"info": f"Unknown benches: {', '.join(unknown)}"}

    try:
        sample_pairs = _select_bench_size_pairs(args, benches)
    except ValueError as exc:
        return {"combined_score": -1000.0}, {"info": str(exc)}
    if not sample_pairs:
        return {"combined_score": -1000.0}, {"info": "No valid (bench,size) pairs found"}

    sample_res = _evaluate_bench_size_pairs(
        evolved_cost_search=evolved_cost_search,
        args=args,
        bench_size_pairs=sample_pairs,
        include_cases=True,
    )
    if not sample_res.get("ok", False):
        return {"combined_score": -1000.0}, {"info": sample_res.get("error", "Evaluation failed")}

    sample_combined_raw = float(sample_res["combined_score_raw"])
    score_mode = str(sample_res["score_mode"])
    score_meta = _score_metadata(score_mode)

    sample_features = {
        "sample_qose_depth": float(sample_res["avg_depth"]),
        "sample_qose_cnot": float(sample_res["avg_cnot"]),
        "sample_qose_overhead": float(sample_res["avg_overhead"]),
        "sample_avg_run_time": float(sample_res["avg_run_time"]),
        "sample_combined_score_raw": sample_combined_raw,
    }

    surrogate_enabled = _surrogate_enabled()
    surrogate_warmup_iters = _surrogate_warmup_iters()
    surrogate_model_name = _surrogate_ml_model_name()
    state_csv = _surrogate_state_csv()
    meta_json = _surrogate_meta_json(state_csv)
    program_hash = _program_hash(program_path)

    full_eval_claimed = False
    full_eval_claim_index = 0
    global_combined_score = float("nan")
    global_num_cases = 0
    surrogate_source = "sample_raw"
    predicted_corr = float("nan")
    predicted_ml = float("nan")
    surrogate_train_rows = 0
    full_eval_error = ""

    if surrogate_enabled:
        full_eval_claimed, full_eval_claim_index = _claim_warmup_slot(
            meta_json=meta_json,
            warmup_iters=surrogate_warmup_iters,
        )
        if full_eval_claimed:
            size_min = int(os.getenv("QOSE_SIZE_MIN", "12"))
            size_max = int(os.getenv("QOSE_SIZE_MAX", "24"))
            full_pairs = _collect_candidate_pairs(benches, size_min, size_max)
            if full_pairs:
                full_res = _evaluate_bench_size_pairs(
                    evolved_cost_search=evolved_cost_search,
                    args=args,
                    bench_size_pairs=full_pairs,
                    include_cases=False,
                )
                if full_res.get("ok", False):
                    global_combined_score = float(full_res["combined_score_raw"])
                    global_num_cases = int(full_res.get("num_cases", 0))
                else:
                    full_eval_error = str(full_res.get("error", "Full evaluation failed"))
            else:
                full_eval_error = "No full-eval (bench,size) pairs found."

        _append_surrogate_row(
            state_csv,
            {
                "program_hash": program_hash,
                "timestamp_sec": time.time(),
                "sample_qose_depth": sample_features["sample_qose_depth"],
                "sample_qose_cnot": sample_features["sample_qose_cnot"],
                "sample_qose_overhead": sample_features["sample_qose_overhead"],
                "sample_avg_run_time": sample_features["sample_avg_run_time"],
                "sample_combined_score_raw": sample_combined_raw,
                "global_combined_score": (
                    global_combined_score if global_combined_score == global_combined_score else ""
                ),
                "score_mode": score_mode,
            },
        )
        train_rows = _load_surrogate_rows(state_csv)
        corr_model, ml_model, surrogate_train_rows = _fit_surrogate_models(
            train_rows, ml_model_name=surrogate_model_name
        )
        predicted_corr, predicted_ml = _predict_surrogate_scores(
            corr_model=corr_model,
            ml_model=ml_model,
            sample_features=sample_features,
        )

    if global_combined_score == global_combined_score:
        combined_score_final = global_combined_score
        surrogate_source = "global_full_eval"
    elif predicted_ml == predicted_ml:
        combined_score_final = predicted_ml
        surrogate_source = "ml_predict"
    elif predicted_corr == predicted_corr:
        combined_score_final = predicted_corr
        surrogate_source = "corr_predict"
    else:
        combined_score_final = sample_combined_raw
        surrogate_source = "sample_raw"

    metrics = {
        "qose_depth": float(sample_res["avg_depth"]),
        "qose_cnot": float(sample_res["avg_cnot"]),
        "qose_overhead": float(sample_res["avg_overhead"]),
        "avg_run_time": float(sample_res["avg_run_time"]),
        # Final score used by evolution for ranking/comparison.
        "combined_score": float(combined_score_final),
        # Keep original sampled score for analysis.
        "combined_score_sample_raw": float(sample_combined_raw),
    }
    if global_combined_score == global_combined_score:
        metrics["combined_score_global"] = float(global_combined_score)
    if predicted_corr == predicted_corr:
        metrics["combined_score_pred_corr"] = float(predicted_corr)
    if predicted_ml == predicted_ml:
        metrics["combined_score_pred_ml"] = float(predicted_ml)

    artifacts = {
        "qose_budget": args.budget,
        "qose_run_sec_avg": sample_res["qose_run_sec_avg"],
        "qos_run_sec_avg": sample_res["qos_run_sec_avg"],
        "gv_cost_calls_total": sample_res["gv_cost_calls_total"],
        "wc_cost_calls_total": sample_res["wc_cost_calls_total"],
        "qos_gv_cost_calls_total": sample_res["qos_gv_cost_calls_total"],
        "qos_wc_cost_calls_total": sample_res["qos_wc_cost_calls_total"],
        "qos_depth_avg": sample_res["qos_depth_avg"],
        "qos_cnot_avg": sample_res["qos_cnot_avg"],
        "qos_overhead_avg": sample_res["qos_overhead_avg"],
        "cases": sample_res["cases"],
        "combined_score_source": surrogate_source,
        "surrogate_enabled": surrogate_enabled,
        "surrogate_warmup_iters": surrogate_warmup_iters,
        "surrogate_ml_model": surrogate_model_name,
        "surrogate_train_rows": surrogate_train_rows,
        "surrogate_state_csv": str(state_csv),
        "surrogate_meta_json": str(meta_json),
        "surrogate_full_eval_claimed": full_eval_claimed,
        "surrogate_full_eval_claim_index": full_eval_claim_index,
        "surrogate_full_eval_num_cases": global_num_cases,
        "surrogate_full_eval_error": full_eval_error,
    }
    if predicted_corr == predicted_corr:
        artifacts["surrogate_pred_corr"] = predicted_corr
    if predicted_ml == predicted_ml:
        artifacts["surrogate_pred_ml"] = predicted_ml
    if global_combined_score == global_combined_score:
        artifacts["surrogate_global_score"] = global_combined_score
    artifacts.update(score_meta)
    artifacts["combined_score_used_formula"] = (
        "if global_full_eval available: use global_combined_score; "
        "else if ml model available: use ml prediction from sampled features; "
        "else if corr model available: use corr prediction from sampled raw score; "
        "else use sampled combined_score_raw"
    )
    return _round_float_values(metrics), _round_float_values(artifacts)


def evaluate(program_path):
    timeout_sec = int(os.getenv("QOSE_EVAL_TIMEOUT_SEC", "6000"))
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


if __name__ == "__main__":
    pass
