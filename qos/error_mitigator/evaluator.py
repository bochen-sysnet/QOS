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
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x_arr = [float(v) for v in xs]
    y_arr = [float(v) for v in ys]
    x_mean = sum(x_arr) / len(x_arr)
    y_mean = sum(y_arr) / len(y_arr)
    x_centered = [v - x_mean for v in x_arr]
    y_centered = [v - y_mean for v in y_arr]
    num = sum(a * b for a, b in zip(x_centered, y_centered))
    den_x = sum(a * a for a in x_centered) ** 0.5
    den_y = sum(b * b for b in y_centered) ** 0.5
    if den_x <= 1e-12 or den_y <= 1e-12:
        return float("nan")
    return num / (den_x * den_y)

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


def _surrogate_prediction_enabled() -> bool:
    return _env_bool("QOSE_SURROGATE_USE_ML_PREDICTION", True)


def _surrogate_refit_every() -> int:
    return max(0, int(os.getenv("QOSE_SURROGATE_REFIT_EVERY", "10")))


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


def _surrogate_cases_csv(state_csv: Path) -> Path:
    return state_csv.with_suffix(state_csv.suffix + ".cases.csv")


def _surrogate_model_compare_csv(state_csv: Path) -> Path:
    raw = os.getenv("QOSE_SURROGATE_MODEL_COMPARE_CSV", "").strip()
    if raw:
        return Path(raw)
    return state_csv.with_suffix(state_csv.suffix + ".model_compare.csv")


def _surrogate_model_compare_pdf(state_csv: Path) -> Path:
    raw = os.getenv("QOSE_SURROGATE_MODEL_COMPARE_PDF", "").strip()
    if raw:
        return Path(raw)
    return state_csv.with_suffix(state_csv.suffix + ".model_compare.pdf")


def _surrogate_selection_csv(state_csv: Path) -> Path:
    raw = os.getenv("QOSE_SURROGATE_SELECTION_CSV", "").strip()
    if raw:
        return Path(raw)
    return state_csv.with_suffix(state_csv.suffix + ".selection.csv")


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


def _load_meta_json(meta_json: Path) -> dict:
    if meta_json.exists():
        try:
            return json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _update_meta_json(meta_json: Path, updater):
    lock = _with_file_lock(meta_json.with_suffix(meta_json.suffix + ".lock"))
    try:
        meta = _load_meta_json(meta_json)
        meta = updater(meta) or meta
        meta_json.parent.mkdir(parents=True, exist_ok=True)
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _claim_full_eval_slot(
    meta_json: Path,
    warmup_iters: int,
    refit_every: int,
) -> tuple[bool, int, int, str]:
    """
    Returns:
      should_full_eval, full_eval_claim_index, eval_index, reason
    """
    result = {
        "should": False,
        "full_idx": 0,
        "eval_idx": 0,
        "reason": "",
    }

    def _update(meta: dict) -> dict:
        eval_count = int(meta.get("eval_count", 0)) + 1
        full_claimed = int(meta.get("full_eval_claimed", 0))
        should = False
        reason = ""

        if full_claimed < warmup_iters:
            full_claimed += 1
            should = True
            reason = "warmup"
        elif refit_every > 0 and eval_count > warmup_iters:
            if ((eval_count - warmup_iters) % refit_every) == 0:
                full_claimed += 1
                should = True
                reason = "periodic_refit"

        meta["eval_count"] = eval_count
        meta["full_eval_claimed"] = full_claimed
        meta["last_full_eval_reason"] = reason
        result["should"] = should
        result["full_idx"] = full_claimed
        result["eval_idx"] = eval_count
        result["reason"] = reason
        return meta

    _update_meta_json(meta_json, _update)
    return (
        bool(result["should"]),
        int(result["full_idx"]),
        int(result["eval_idx"]),
        str(result["reason"]),
    )


def _append_surrogate_row(csv_path: Path, row: dict) -> None:
    lock = _with_file_lock(csv_path.with_suffix(csv_path.suffix + ".lock"))
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        fields = [
            "eval_index",
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


def _append_surrogate_case_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    lock = _with_file_lock(csv_path.with_suffix(csv_path.suffix + ".lock"))
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        fields = [
            "program_hash",
            "timestamp_sec",
            "bench",
            "size",
            "depth_ratio",
            "cnot_ratio",
            "time_ratio",
            "overhead_ratio",
            "case_score_piecewise",
            "case_score_legacy",
            "global_combined_score",
            "score_mode",
        ]
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _append_surrogate_selection_row(csv_path: Path, row: dict) -> None:
    lock = _with_file_lock(csv_path.with_suffix(csv_path.suffix + ".lock"))
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        fields = [
            "eval_index",
            "timestamp_sec",
            "program_hash",
            "selection_mode",
            "score_source",
            "sample_count",
            "sample_pairs_json",
            "corr_support",
            "corr_top_abs_corr",
            "full_eval_claimed",
            "full_eval_reason",
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


def _load_surrogate_case_rows(csv_path: Path) -> list[dict]:
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


def _build_pipeline(estimator, use_scaler: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _surrogate_model_factories(random_state: int = 42):
    return {
        "ridge": lambda: _build_pipeline(
            Ridge(alpha=1.0, random_state=random_state), use_scaler=True
        ),
        "elasticnet": lambda: _build_pipeline(
            ElasticNet(
                alpha=0.005,
                l1_ratio=0.5,
                max_iter=20000,
                random_state=random_state,
            ),
            use_scaler=True,
        ),
        "random_forest": lambda: _build_pipeline(
            RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1),
            use_scaler=False,
        ),
        "extra_trees": lambda: _build_pipeline(
            ExtraTreesRegressor(n_estimators=500, random_state=random_state, n_jobs=-1),
            use_scaler=False,
        ),
        "gradient_boosting": lambda: _build_pipeline(
            GradientBoostingRegressor(random_state=random_state), use_scaler=False
        ),
    }


def _prepare_surrogate_xy(rows: list[dict]) -> tuple[list[list[float]], list[float], list[float]]:
    x = []
    y = []
    order_key = []
    for r in rows:
        feat = [_safe_float(r.get(c)) for c in _SURROGATE_FEATURE_COLUMNS]
        target = _safe_float(r.get("global_combined_score"))
        t = _safe_float(r.get("timestamp_sec"), default=0.0)
        eval_idx = _safe_float(r.get("eval_index"), default=float("nan"))
        if any(v != v for v in feat) or target != target:
            continue
        x.append(feat)
        y.append(target)
        if eval_idx == eval_idx:
            order_key.append((int(eval_idx), t))
        else:
            order_key.append((10**12, t))
    order = sorted(range(len(x)), key=lambda i: order_key[i])
    return [x[i] for i in order], [y[i] for i in order], [float(order_key[i][1]) for i in order]


def _rolling_one_step_mae(
    x: list[list[float]],
    y: list[float],
    model_factory,
    min_train: int,
) -> tuple[float, int]:
    if len(x) <= max(1, min_train):
        return float("nan"), 0
    errs = []
    for i in range(min_train, len(x)):
        try:
            model = model_factory()
            model.fit(x[:i], y[:i])
            pred = float(model.predict([x[i]])[0])
            errs.append(abs(pred - y[i]))
        except Exception:
            continue
    if not errs:
        return float("nan"), 0
    return float(sum(errs) / len(errs)), len(errs)


def _compute_surrogate_model_stats(rows: list[dict]) -> tuple[dict[str, dict], int]:
    x, y, _ = _prepare_surrogate_xy(rows)
    min_train = _surrogate_min_train_rows()
    factories = _surrogate_model_factories()
    stats = {}
    for name, factory in factories.items():
        mae, nsteps = _rolling_one_step_mae(x, y, factory, min_train=min_train)
        stats[name] = {
            "rolling_mae": mae,
            "rolling_steps": nsteps,
        }
    return stats, len(x)


def _choose_best_surrogate_model(stats: dict[str, dict], fallback: str) -> str:
    best_name = ""
    best_mae = float("inf")
    for name, d in stats.items():
        mae = _safe_float(d.get("rolling_mae"))
        steps = int(_safe_float(d.get("rolling_steps"), default=0))
        if mae != mae or steps <= 0:
            continue
        if mae < best_mae:
            best_mae = mae
            best_name = name
    if best_name:
        return best_name
    return fallback


def _fit_surrogate_model(rows: list[dict], ml_model_name: str):
    x, y, _ = _prepare_surrogate_xy(rows)
    if len(x) < _surrogate_min_train_rows():
        return None, len(x)
    factories = _surrogate_model_factories()
    model_name = ml_model_name if ml_model_name in factories else "random_forest"
    try:
        model = factories[model_name]()
        model.fit(x, y)
        return model, len(x)
    except Exception:
        return None, len(x)


def _predict_surrogate_score(ml_model, sample_features: dict) -> float:
    feat = [_safe_float(sample_features.get(c)) for c in _SURROGATE_FEATURE_COLUMNS]
    if any(v != v for v in feat):
        return float("nan")
    try:
        if ml_model is not None:
            return float(ml_model.predict([feat])[0])
    except Exception:
        return float("nan")
    return float("nan")


def _score_from_case_subset(case_map: dict, pairs: list[tuple[str, int]]) -> dict | None:
    if not case_map or not pairs:
        return None
    selected = [case_map[p] for p in pairs if p in case_map]
    if not selected:
        return None
    depth = sum(_safe_float(v.get("depth_ratio")) for v in selected) / len(selected)
    cnot = sum(_safe_float(v.get("cnot_ratio")) for v in selected) / len(selected)
    over = sum(_safe_float(v.get("overhead_ratio")) for v in selected) / len(selected)
    run = sum(_safe_float(v.get("time_ratio")) for v in selected) / len(selected)
    combined, _ = _combined_score_from_ratios(depth, cnot, run, over)
    return {
        "sample_qose_depth": depth,
        "sample_qose_cnot": cnot,
        "sample_qose_overhead": over,
        "sample_avg_run_time": run,
        "sample_combined_score_raw": combined,
    }


def _select_random_pairs_from_case_map(
    case_map: dict,
    benches: list[str],
    samples_per_bench: int,
) -> list[tuple[str, int]]:
    if not case_map:
        return []
    size_min = int(os.getenv("QOSE_SIZE_MIN", "12"))
    stratified = os.getenv("QOSE_STRATIFIED_SIZES", "1").strip().lower() in {"1", "true", "yes", "y"}
    distinct_sizes = os.getenv("QOSE_DISTINCT_SIZES_PER_BENCH", "1").strip().lower() in {"1", "true", "yes", "y"}
    sample_seed_raw = os.getenv("QOSE_SAMPLE_SEED", "").strip()
    sample_seed = int(sample_seed_raw) if sample_seed_raw else 123
    rng = random.Random(sample_seed)

    bench_to_sizes: dict[str, set[int]] = {}
    for bench, size in case_map:
        bench_to_sizes.setdefault(str(bench), set()).add(int(size))

    picked: list[tuple[str, int]] = []
    for bench in benches:
        all_sizes = sorted(bench_to_sizes.get(bench, set()))
        if not all_sizes:
            continue
        sample_pool = all_sizes
        if stratified:
            parity_sizes = [s for s in all_sizes if (s - size_min) % 2 == 0]
            if parity_sizes:
                sample_pool = parity_sizes
        if distinct_sizes and len(sample_pool) < samples_per_bench and len(all_sizes) >= samples_per_bench:
            sample_pool = all_sizes
        if distinct_sizes and len(sample_pool) >= samples_per_bench:
            chosen = rng.sample(sample_pool, samples_per_bench)
        else:
            chosen = [rng.choice(sample_pool) for _ in range(max(1, samples_per_bench))]
        picked.extend((bench, int(s)) for s in chosen)
    return picked


def _corr_selection_for_target(
    target_case_map: dict,
    history_events: list[dict],
    all_case_by_pid: dict,
    target_k: int,
) -> tuple[list[tuple[str, int]], list[dict]]:
    if not target_case_map or target_k <= 0:
        return [], []
    latest_history: dict[str, dict] = {}
    for ev in history_events:
        pid = str(ev.get("program_hash", "")).strip()
        if pid:
            latest_history[pid] = ev
    if len(latest_history) < _surrogate_min_train_rows():
        return [], []

    y_by_pid = {
        pid: _safe_float(ev.get("global_combined_score"))
        for pid, ev in latest_history.items()
        if _safe_float(ev.get("global_combined_score")) == _safe_float(ev.get("global_combined_score"))
    }
    if len(y_by_pid) < _surrogate_min_train_rows():
        return [], []

    ranking = []
    for pair in sorted(target_case_map.keys(), key=lambda p: (str(p[0]), int(p[1]))):
        xs = []
        ys = []
        for pid, y in y_by_pid.items():
            case_map = all_case_by_pid.get(pid, {})
            case = case_map.get(pair)
            if not case:
                continue
            score = _safe_float(case.get("case_score"))
            if score != score:
                continue
            xs.append(score)
            ys.append(y)
        corr = _pearson_corr(xs, ys)
        ranking.append(
            {
                "bench": pair[0],
                "size": pair[1],
                "corr": corr,
                "abs_corr": abs(corr) if corr == corr else float("nan"),
                "support": len(xs),
            }
        )

    ranking.sort(
        key=lambda r: (
            -_safe_float(r.get("abs_corr"), default=-1.0),
            -int(_safe_float(r.get("support"), default=0)),
            str(r.get("bench", "")),
            int(_safe_float(r.get("size"), default=0)),
        )
    )
    selected = []
    for row in ranking:
        selected.append((str(row["bench"]), int(row["size"])))
        if len(selected) >= target_k:
            break
    return selected, ranking


def _extract_eval_events(rows: list[dict]) -> list[dict]:
    latest_by_eval: dict[int, dict] = {}
    latest_ts: dict[int, float] = {}
    for i, r in enumerate(rows):
        ts = _safe_float(r.get("timestamp_sec"), default=float(i))
        try:
            eval_idx = int(float(r.get("eval_index")))
        except Exception:
            eval_idx = i + 1
        pid = str(r.get("program_hash", "")).strip()
        if not pid:
            continue
        global_score = _safe_float(r.get("global_combined_score"))
        event = {
            "eval_index": eval_idx,
            "timestamp_sec": ts,
            "program_hash": pid,
            "global_combined_score": global_score,
            "sample_qose_depth": _safe_float(r.get("sample_qose_depth")),
            "sample_qose_cnot": _safe_float(r.get("sample_qose_cnot")),
            "sample_qose_overhead": _safe_float(r.get("sample_qose_overhead")),
            "sample_avg_run_time": _safe_float(r.get("sample_avg_run_time")),
            "sample_combined_score_raw": _safe_float(r.get("sample_combined_score_raw")),
        }
        if eval_idx not in latest_by_eval or ts >= latest_ts.get(eval_idx, float("-inf")):
            latest_by_eval[eval_idx] = event
            latest_ts[eval_idx] = ts
    ordered = sorted(latest_by_eval.values(), key=lambda r: (int(r["eval_index"]), float(r["timestamp_sec"])))
    return ordered


def _build_case_maps(case_rows: list[dict]) -> dict[str, dict]:
    by_pid: dict[str, dict] = {}
    score_col = "case_score_piecewise" if _score_mode() == _SCORE_MODE_PIECEWISE else "case_score_legacy"
    for r in _dedupe_case_rows_latest(case_rows):
        pid = str(r.get("program_hash", "")).strip()
        bench = str(r.get("bench", "")).strip()
        try:
            size = int(float(r.get("size")))
        except Exception:
            continue
        if not pid or not bench:
            continue
        by_pid.setdefault(pid, {})[(bench, size)] = {
            "depth_ratio": _safe_float(r.get("depth_ratio")),
            "cnot_ratio": _safe_float(r.get("cnot_ratio")),
            "time_ratio": _safe_float(r.get("time_ratio")),
            "overhead_ratio": _safe_float(r.get("overhead_ratio")),
            "case_score": _safe_float(r.get(score_col)),
        }
    return by_pid


def _compute_surrogate_accuracy_trace(
    state_rows: list[dict],
    case_rows: list[dict],
    freeze_after_eval: int,
    retrain_every: int,
) -> list[dict]:
    events = _extract_eval_events(state_rows)
    if not events:
        return []
    case_by_pid = _build_case_maps(case_rows)
    samples_per_bench = max(1, int(os.getenv("QOSE_SAMPLES_PER_BENCH", "1")))
    model_factories = _surrogate_model_factories()
    min_train = _surrogate_min_train_rows()

    replay = []
    for idx, ev in enumerate(events):
        pid = ev["program_hash"]
        target_case_map = case_by_pid.get(pid, {})
        if not target_case_map:
            continue
        benches = [b for b, _ in BENCHES if any(pair[0] == b for pair in target_case_map.keys())]
        target_k = max(1, len(benches) * samples_per_bench) if benches else max(1, samples_per_bench)
        history = replay[:idx]
        corr_pairs, corr_rank = _corr_selection_for_target(
            target_case_map=target_case_map,
            history_events=history,
            all_case_by_pid=case_by_pid,
            target_k=target_k,
        )
        random_pairs = _select_random_pairs_from_case_map(
            case_map=target_case_map,
            benches=benches or sorted({b for b, _ in target_case_map.keys()}),
            samples_per_bench=samples_per_bench,
        )
        replay.append(
            {
                **ev,
                "corr_pairs": corr_pairs,
                "corr_summary": _score_from_case_subset(target_case_map, corr_pairs),
                "random_summary": _score_from_case_subset(target_case_map, random_pairs),
                "corr_support": int(_safe_float(corr_rank[0].get("support"), default=0)) if corr_rank else 0,
                "corr_top_abs_corr": _safe_float(corr_rank[0].get("abs_corr")) if corr_rank else float("nan"),
            }
        )

    if not replay:
        return []

    trace_rows = []
    running = {}
    baseline_names = ["baseline_random_no_pred", "baseline_corr_no_pred"]
    for name in baseline_names + sorted(model_factories.keys()):
        running[name] = {"sum": 0.0, "count": 0, "model": None, "last_fit_eval": None}

    for i, ev in enumerate(replay):
        eval_idx = int(ev["eval_index"])
        global_score = _safe_float(ev["global_combined_score"])
        is_rolling_refit = (
            eval_idx > freeze_after_eval
            and retrain_every > 0
            and ((eval_idx - freeze_after_eval) % retrain_every == 0)
        )
        phase = "rolling" if (eval_idx <= freeze_after_eval or is_rolling_refit) else "frozen"

        for bname, summary_key in [
            ("baseline_random_no_pred", "random_summary"),
            ("baseline_corr_no_pred", "corr_summary"),
        ]:
            pred = float("nan")
            summ = ev.get(summary_key)
            if summ:
                pred = _safe_float(summ.get("sample_combined_score_raw"))
            abs_err = abs(pred - global_score) if pred == pred and global_score == global_score else float("nan")
            if abs_err == abs_err:
                running[bname]["sum"] += abs_err
                running[bname]["count"] += 1
            running_mae = (
                running[bname]["sum"] / running[bname]["count"] if running[bname]["count"] > 0 else float("nan")
            )
            trace_rows.append(
                {
                    "eval_index": eval_idx,
                    "program_hash": ev["program_hash"],
                    "method": bname,
                    "phase": phase,
                    "global_combined_score": global_score,
                    "predicted_score": pred,
                    "abs_err": abs_err,
                    "running_mae": running_mae,
                    "train_rows": 0,
                }
            )

        for mname, factory in sorted(model_factories.items()):
            current_summary = ev.get("corr_summary")
            pred = float("nan")
            train_rows = 0
            if current_summary:
                latest_train: dict[str, dict] = {}
                for prev in replay[:i]:
                    if not prev.get("corr_summary"):
                        continue
                    latest_train[prev["program_hash"]] = prev
                train_events = sorted(
                    latest_train.values(),
                    key=lambda r: (int(r["eval_index"]), float(r["timestamp_sec"])),
                )
                train_rows = len(train_events)
                if train_rows >= min_train:
                    x_train = [
                        [_safe_float(te["corr_summary"].get(c)) for c in _SURROGATE_FEATURE_COLUMNS]
                        for te in train_events
                    ]
                    y_train = [_safe_float(te.get("global_combined_score")) for te in train_events]
                    x_cur = [[_safe_float(current_summary.get(c)) for c in _SURROGATE_FEATURE_COLUMNS]]
                    if all(v == v for row_vals in x_train for v in row_vals) and all(v == v for v in y_train) and all(
                        v == v for v in x_cur[0]
                    ):
                        state = running[mname]
                        if eval_idx <= freeze_after_eval:
                            model = factory()
                            model.fit(x_train, y_train)
                            state["model"] = model
                            state["last_fit_eval"] = eval_idx
                        else:
                            model = state.get("model")
                            last_fit_eval = state.get("last_fit_eval")
                            need_refit = model is None
                            if (not need_refit) and retrain_every > 0 and last_fit_eval is not None:
                                if (eval_idx - int(last_fit_eval)) >= retrain_every:
                                    need_refit = True
                            if need_refit:
                                model = factory()
                                model.fit(x_train, y_train)
                                state["model"] = model
                                state["last_fit_eval"] = eval_idx
                        try:
                            pred = float(running[mname]["model"].predict(x_cur)[0])
                        except Exception:
                            pred = float("nan")

            abs_err = abs(pred - global_score) if pred == pred and global_score == global_score else float("nan")
            if abs_err == abs_err:
                running[mname]["sum"] += abs_err
                running[mname]["count"] += 1
            running_mae = (
                running[mname]["sum"] / running[mname]["count"] if running[mname]["count"] > 0 else float("nan")
            )
            trace_rows.append(
                {
                    "eval_index": eval_idx,
                    "program_hash": ev["program_hash"],
                    "method": mname,
                    "phase": phase,
                    "global_combined_score": global_score,
                    "predicted_score": pred,
                    "abs_err": abs_err,
                    "running_mae": running_mae,
                    "train_rows": train_rows,
                }
            )

    return trace_rows


def _write_model_compare_csv(
    path: Path,
    trace_rows: list[dict],
    eval_index: int,
    warmup_iters: int,
    freeze_after_eval: int,
    retrain_every: int,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "eval_index",
        "program_hash",
        "method",
        "phase",
        "global_combined_score",
        "predicted_score",
        "abs_err",
        "running_mae",
        "train_rows",
        "snapshot_eval_index",
        "warmup_iters",
        "freeze_after_eval",
        "retrain_every",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in trace_rows:
            out = dict(row)
            out["snapshot_eval_index"] = eval_index
            out["warmup_iters"] = warmup_iters
            out["freeze_after_eval"] = freeze_after_eval
            out["retrain_every"] = retrain_every
            w.writerow({k: out.get(k, "") for k in fields})


def _write_model_compare_pdf(
    path: Path,
    trace_rows: list[dict],
    selected_model: str,
    eval_index: int,
    freeze_after_eval: int,
    retrain_every: int,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not trace_rows:
        return

    by_method: dict[str, list[dict]] = {}
    for row in trace_rows:
        method = str(row.get("method", ""))
        by_method.setdefault(method, []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True, sharey=True)
    ax_roll, ax_frozen = axes
    method_order = [
        "baseline_random_no_pred",
        "baseline_corr_no_pred",
    ] + sorted(
        [m for m in by_method.keys() if m not in {"baseline_random_no_pred", "baseline_corr_no_pred"}]
    )
    colors = {
        "baseline_random_no_pred": "tab:gray",
        "baseline_corr_no_pred": "tab:olive",
    }

    def _plot_phase(ax, phase_name: str, title: str):
        for method in method_order:
            rows = [
                r
                for r in sorted(
                    by_method.get(method, []),
                    key=lambda r: int(_safe_float(r.get("eval_index"), default=0)),
                )
                if str(r.get("phase", "")) == phase_name
            ]
            if not rows:
                continue
            xs = [int(_safe_float(r.get("eval_index"), default=0)) for r in rows]
            ys = [_safe_float(r.get("running_mae")) for r in rows]
            valid = [(x, y) for x, y in zip(xs, ys) if y == y]
            if not valid:
                continue
            xs = [x for x, _ in valid]
            ys = [y for _, y in valid]
            style = "--" if method.startswith("baseline_") else "-"
            lw = 2.4 if method == selected_model else 1.8
            marker = "s" if method.startswith("baseline_") else "o"
            color = colors.get(method, None)
            label = method
            if method == selected_model:
                label = f"{method} (selected)"
            ax.plot(xs, ys, linestyle=style, marker=marker, linewidth=lw, color=color, label=label)

        ax.set_xlabel("Evaluation index (k)")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    _plot_phase(
        ax_roll,
        "rolling",
        f"Rolling Points (<= {freeze_after_eval} and every {retrain_every})",
    )
    _plot_phase(
        ax_frozen,
        "frozen",
        "Frozen Between Refreshes",
    )

    ax_roll.set_ylabel("Running mean absolute error")
    handles, labels = ax_roll.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle(f"Surrogate Accuracy Trend @ eval {eval_index}", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _dedupe_surrogate_rows_latest(rows: list[dict]) -> list[dict]:
    latest_labeled: dict[str, dict] = {}
    latest_labeled_ts: dict[str, float] = {}
    latest_any: dict[str, dict] = {}
    latest_any_ts: dict[str, float] = {}
    for r in rows:
        pid = str(r.get("program_hash", "")).strip()
        if not pid:
            continue
        ts = _safe_float(r.get("timestamp_sec"), default=0.0)
        if pid not in latest_any or ts >= latest_any_ts.get(pid, float("-inf")):
            latest_any[pid] = r
            latest_any_ts[pid] = ts
        labeled = not str(r.get("global_combined_score", "")).strip() in {"", "nan"}
        if labeled and (pid not in latest_labeled or ts >= latest_labeled_ts.get(pid, float("-inf"))):
            latest_labeled[pid] = r
            latest_labeled_ts[pid] = ts
    out = []
    for pid in latest_any:
        out.append(latest_labeled.get(pid, latest_any[pid]))
    return out


def _dedupe_case_rows_latest(rows: list[dict]) -> list[dict]:
    latest: dict[tuple[str, str, int], dict] = {}
    latest_ts: dict[tuple[str, str, int], float] = {}
    for r in rows:
        pid = str(r.get("program_hash", "")).strip()
        bench = str(r.get("bench", "")).strip()
        if not pid or not bench:
            continue
        try:
            size = int(float(r.get("size", "nan")))
        except Exception:
            continue
        key = (pid, bench, size)
        ts = _safe_float(r.get("timestamp_sec"), default=0.0)
        if key not in latest or ts >= latest_ts.get(key, float("-inf")):
            latest[key] = r
            latest_ts[key] = ts
    return list(latest.values())


def _refresh_surrogate_model_selection(
    state_csv: Path,
    meta_json: Path,
    cases_csv: Path,
    eval_index: int,
    warmup_iters: int,
    write_plot: bool,
) -> tuple[str, dict[str, dict], int]:
    rows = _dedupe_surrogate_rows_latest(_load_surrogate_rows(state_csv))
    stats, labeled_count = _compute_surrogate_model_stats(rows)
    fallback = _surrogate_ml_model_name()
    selected = _choose_best_surrogate_model(stats, fallback=fallback)

    def _update(meta: dict) -> dict:
        meta["selected_ml_model"] = selected
        meta["selected_ml_model_updated_eval"] = int(eval_index)
        meta["labeled_rows"] = int(labeled_count)
        meta["model_compare_stats"] = stats
        meta["model_compare_warmup_iters"] = int(warmup_iters)
        return meta

    _update_meta_json(meta_json, _update)

    if write_plot:
        csv_path = _surrogate_model_compare_csv(state_csv)
        pdf_path = _surrogate_model_compare_pdf(state_csv)
        freeze_after_eval = max(1, int(warmup_iters))
        retrain_every = _surrogate_refit_every()
        trace_rows = _compute_surrogate_accuracy_trace(
            state_rows=_load_surrogate_rows(state_csv),
            case_rows=_load_surrogate_case_rows(cases_csv),
            freeze_after_eval=freeze_after_eval,
            retrain_every=retrain_every,
        )
        _write_model_compare_csv(
            csv_path,
            trace_rows=trace_rows,
            eval_index=eval_index,
            warmup_iters=warmup_iters,
            freeze_after_eval=freeze_after_eval,
            retrain_every=retrain_every,
        )
        _write_model_compare_pdf(
            pdf_path,
            trace_rows=trace_rows,
            selected_model=selected,
            eval_index=eval_index,
            freeze_after_eval=freeze_after_eval,
            retrain_every=retrain_every,
        )
    return selected, stats, labeled_count


def _sample_pairs_target_count(benches: list[str]) -> int:
    samples_per_bench = int(os.getenv("QOSE_SAMPLES_PER_BENCH", "1"))
    if samples_per_bench <= 0:
        raise ValueError("QOSE_SAMPLES_PER_BENCH must be >= 1")
    return max(1, len(benches) * samples_per_bench)


def _select_bench_size_pairs_by_correlation(
    benches: list[str],
    candidate_pairs: list[tuple[str, int]],
    state_rows: list[dict],
    case_rows: list[dict],
) -> tuple[list[tuple[str, int]], list[dict]]:
    target_k = _sample_pairs_target_count(benches)
    labeled = [
        r
        for r in _dedupe_surrogate_rows_latest(state_rows)
        if not str(r.get("global_combined_score", "")).strip() in {"", "nan"}
    ]
    if len(labeled) < _surrogate_min_train_rows():
        return [], []

    y_by_program: dict[str, float] = {}
    for r in labeled:
        pid = str(r.get("program_hash", "")).strip()
        y = _safe_float(r.get("global_combined_score"))
        if pid and y == y:
            y_by_program[pid] = y
    if len(y_by_program) < _surrogate_min_train_rows():
        return [], []

    pair_pid_score: dict[tuple[str, int], dict[str, float]] = {}
    score_col = "case_score_piecewise" if _score_mode() == _SCORE_MODE_PIECEWISE else "case_score_legacy"
    for r in _dedupe_case_rows_latest(case_rows):
        pid = str(r.get("program_hash", "")).strip()
        bench = str(r.get("bench", "")).strip()
        try:
            size = int(float(r.get("size", "nan")))
        except Exception:
            continue
        if pid not in y_by_program:
            continue
        x = _safe_float(r.get(score_col))
        if x != x:
            continue
        pair_pid_score.setdefault((bench, size), {})[pid] = x

    ranking = []
    candidate_set = set(candidate_pairs)
    for pair in candidate_pairs:
        pid_to_score = pair_pid_score.get(pair, {})
        xs = []
        ys = []
        for pid, y in y_by_program.items():
            x = pid_to_score.get(pid)
            if x is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        corr = _pearson_corr(xs, ys)
        ranking.append(
            {
                "bench": pair[0],
                "size": pair[1],
                "corr": corr,
                "abs_corr": abs(corr) if corr == corr else float("nan"),
                "support": len(xs),
            }
        )

    ranking.sort(
        key=lambda r: (
            -_safe_float(r.get("abs_corr"), default=-1.0),
            -int(_safe_float(r.get("support"), default=0)),
            str(r.get("bench", "")),
            int(_safe_float(r.get("size"), default=0)),
        )
    )
    selected = []
    for row in ranking:
        pair = (str(row["bench"]), int(row["size"]))
        if pair in candidate_set:
            selected.append(pair)
        if len(selected) >= target_k:
            break
    return selected, ranking


def _build_surrogate_case_rows(
    program_hash: str,
    timestamp_sec: float,
    cases: list[dict],
    global_combined_score,
    score_mode: str,
) -> list[dict]:
    out = []
    for case in cases:
        qos_depth = _safe_float(case.get("qos_depth"))
        qose_depth = _safe_float(case.get("qose_depth"))
        qos_cnot = _safe_float(case.get("qos_cnot"))
        qose_cnot = _safe_float(case.get("qose_cnot"))
        qos_time = _safe_float(case.get("qos_run_sec"))
        qose_time = _safe_float(case.get("qose_run_sec"))
        qos_over = _safe_float(case.get("qos_num_circuits"))
        qose_over = _safe_float(case.get("qose_num_circuits"))
        if any(v != v for v in [qos_depth, qose_depth, qos_cnot, qose_cnot, qos_time, qose_time, qos_over, qose_over]):
            continue
        depth_ratio = _safe_ratio(qose_depth, qos_depth)
        cnot_ratio = _safe_ratio(qose_cnot, qos_cnot)
        time_ratio = _safe_ratio(qose_time, qos_time)
        overhead_ratio = _safe_ratio(qose_over, qos_over)
        struct_delta = 1.0 - ((depth_ratio + cnot_ratio) / 2.0)
        time_delta = 1.0 - time_ratio
        struct_term = struct_delta if struct_delta >= 0 else 8.0 * struct_delta
        case_piecewise = struct_term + time_delta
        case_legacy = -(depth_ratio + cnot_ratio + overhead_ratio + time_ratio)
        out.append(
            {
                "program_hash": program_hash,
                "timestamp_sec": timestamp_sec,
                "bench": str(case.get("bench", "")),
                "size": int(_safe_float(case.get("size"), default=0)),
                "depth_ratio": depth_ratio,
                "cnot_ratio": cnot_ratio,
                "time_ratio": time_ratio,
                "overhead_ratio": overhead_ratio,
                "case_score_piecewise": case_piecewise,
                "case_score_legacy": case_legacy,
                "global_combined_score": global_combined_score,
                "score_mode": score_mode,
            }
        )
    return out

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

    surrogate_enabled = _surrogate_enabled()
    surrogate_prediction_enabled = _surrogate_prediction_enabled()
    surrogate_warmup_iters = _surrogate_warmup_iters()
    surrogate_refit_every = _surrogate_refit_every()
    surrogate_model_name = _surrogate_ml_model_name()
    state_csv = _surrogate_state_csv()
    meta_json = _surrogate_meta_json(state_csv)
    cases_csv = _surrogate_cases_csv(state_csv)
    program_hash = _program_hash(program_path)

    full_eval_claimed = False
    full_eval_claim_index = 0
    eval_index = 0
    full_eval_reason = ""
    if surrogate_enabled:
        (
            full_eval_claimed,
            full_eval_claim_index,
            eval_index,
            full_eval_reason,
        ) = _claim_full_eval_slot(
            meta_json=meta_json,
            warmup_iters=surrogate_warmup_iters,
            refit_every=surrogate_refit_every,
        )

    size_min = int(os.getenv("QOSE_SIZE_MIN", "12"))
    size_max = int(os.getenv("QOSE_SIZE_MAX", "24"))
    candidate_pairs = _collect_candidate_pairs(benches, size_min, size_max)
    if not candidate_pairs:
        return {"combined_score": -1000.0}, {"info": "No valid (bench,size) pairs found"}

    corr_selection_used = False
    corr_selection_support = 0
    corr_selection_top_abs_corr = float("nan")
    corr_ranking = []

    try:
        sample_pairs: list[tuple[str, int]] = []
        if surrogate_enabled:
            state_rows = _load_surrogate_rows(state_csv)
            case_rows = _load_surrogate_case_rows(cases_csv)
            sample_pairs, corr_ranking = _select_bench_size_pairs_by_correlation(
                benches=benches,
                candidate_pairs=candidate_pairs,
                state_rows=state_rows,
                case_rows=case_rows,
            )
            if sample_pairs:
                corr_selection_used = True
                if corr_ranking:
                    corr_selection_support = int(
                        _safe_float(corr_ranking[0].get("support"), default=0)
                    )
                    corr_selection_top_abs_corr = _safe_float(
                        corr_ranking[0].get("abs_corr")
                    )
        if not sample_pairs:
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

    global_combined_score = float("nan")
    global_num_cases = 0
    surrogate_source = "sample_raw"
    predicted_ml = float("nan")
    surrogate_train_rows = 0
    full_eval_error = ""
    timestamp_sec = time.time()
    selected_model_name = surrogate_model_name
    model_compare_stats = {}
    model_compare_labeled_rows = 0

    if surrogate_enabled:
        if full_eval_claimed:
            if candidate_pairs:
                full_res = _evaluate_bench_size_pairs(
                    evolved_cost_search=evolved_cost_search,
                    args=args,
                    bench_size_pairs=candidate_pairs,
                    include_cases=True,
                )
                if full_res.get("ok", False):
                    global_combined_score = float(full_res["combined_score_raw"])
                    global_num_cases = int(full_res.get("num_cases", 0))
                    case_rows_to_save = _build_surrogate_case_rows(
                        program_hash=program_hash,
                        timestamp_sec=timestamp_sec,
                        cases=full_res.get("cases", []),
                        global_combined_score=global_combined_score,
                        score_mode=score_mode,
                    )
                    _append_surrogate_case_rows(cases_csv, case_rows_to_save)
                else:
                    full_eval_error = str(full_res.get("error", "Full evaluation failed"))
            else:
                full_eval_error = "No full-eval (bench,size) pairs found."

        _append_surrogate_row(
            state_csv,
            {
                "eval_index": eval_index,
                "program_hash": program_hash,
                "timestamp_sec": timestamp_sec,
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
        if surrogate_prediction_enabled:
            should_write_model_plot = True
            selected_model_name, model_compare_stats, model_compare_labeled_rows = _refresh_surrogate_model_selection(
                state_csv=state_csv,
                meta_json=meta_json,
                cases_csv=cases_csv,
                eval_index=eval_index,
                warmup_iters=surrogate_warmup_iters,
                write_plot=should_write_model_plot,
            )
            train_rows = _dedupe_surrogate_rows_latest(_load_surrogate_rows(state_csv))
            ml_model, surrogate_train_rows = _fit_surrogate_model(
                train_rows, ml_model_name=selected_model_name
            )
            predicted_ml = _predict_surrogate_score(
                ml_model=ml_model,
                sample_features=sample_features,
            )

    if global_combined_score == global_combined_score:
        combined_score_final = global_combined_score
        surrogate_source = "global_full_eval"
    elif surrogate_prediction_enabled and predicted_ml == predicted_ml:
        combined_score_final = predicted_ml
        surrogate_source = "ml_predict"
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
    }

    score_source_mode = "prediction" if surrogate_source == "ml_predict" else "raw"
    selection_mode = "correlation" if corr_selection_used else "random"

    if surrogate_enabled:
        try:
            _append_surrogate_selection_row(
                _surrogate_selection_csv(state_csv),
                {
                    "eval_index": eval_index,
                    "timestamp_sec": timestamp_sec,
                    "program_hash": program_hash,
                    "selection_mode": selection_mode,
                    "score_source": score_source_mode,
                    "sample_count": len(sample_pairs),
                    "sample_pairs_json": json.dumps(sample_pairs),
                    "corr_support": corr_selection_support,
                    "corr_top_abs_corr": corr_selection_top_abs_corr,
                    "full_eval_claimed": int(bool(full_eval_claimed)),
                    "full_eval_reason": full_eval_reason,
                },
            )
        except Exception:
            pass

    summary = {
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
        # Minimal surrogate transparency for evolution prompt.
        "qose_selection_mode": selection_mode,
        "qose_score_source": score_source_mode,
    }
    summary.update(score_meta)
    artifacts = {
        "summary": summary,
        "cases": sample_res["cases"],
    }
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
