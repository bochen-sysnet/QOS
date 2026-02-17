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
import ast
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

logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("qiskit.transpiler").setLevel(logging.WARNING)
logging.getLogger("qiskit.compiler").setLevel(logging.WARNING)
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_SCORE_MODE_LEGACY = "legacy"
_SCORE_MODE_PIECEWISE = "piecewise"
_SURROGATE_MIN_TRAIN_ROWS_DEFAULT = 5


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
    return _env_bool("QOSE_SURROGATE_ENABLE", False)


def _surrogate_warmup_iters() -> int:
    return max(0, int(os.getenv("QOSE_SURROGATE_WARMUP_ITERS", "20")))


def _surrogate_selection_mode() -> str:
    raw = os.getenv("QOSE_SURROGATE_SELECTION_MODE", "correlation").strip().lower()
    if raw in {"worst", "worse", "worst_case", "worst-performance", "worst_performance"}:
        return "worst"
    if raw in {"random"}:
        return "random"
    return "correlation"


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


def _surrogate_progress_csv(state_csv: Path) -> Path:
    raw = os.getenv("QOSE_SURROGATE_PROGRESS_CSV", "").strip()
    if not raw:
        # Backward compatible env key.
        raw = os.getenv("QOSE_SURROGATE_MODEL_COMPARE_CSV", "").strip()
    if raw:
        return Path(raw)
    return state_csv.with_suffix(state_csv.suffix + ".progress.csv")


def _surrogate_progress_pdf(state_csv: Path) -> Path:
    raw = os.getenv("QOSE_SURROGATE_PROGRESS_PDF", "").strip()
    if not raw:
        # Backward compatible env key.
        raw = os.getenv("QOSE_SURROGATE_MODEL_COMPARE_PDF", "").strip()
    if raw:
        return Path(raw)
    return state_csv.with_suffix(state_csv.suffix + ".progress.pdf")


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
            "selection_support",
            "selection_signal",
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


def _compute_surrogate_progress_trace(
    state_rows: list[dict],
) -> list[dict]:
    events = _extract_eval_events(state_rows)
    if not events:
        return []
    full_only = [ev for ev in events if _safe_float(ev.get("global_combined_score")) == _safe_float(ev.get("global_combined_score"))]
    if not full_only:
        return []

    out = []
    running_full = 0.0
    running_selected = 0.0
    for i, ev in enumerate(full_only, start=1):
        full_score = _safe_float(ev.get("global_combined_score"))
        selected_score = _safe_float(ev.get("sample_combined_score_raw"))
        running_full += float(full_score)
        running_selected += float(selected_score)
        out.append(
            {
                "eval_index": int(ev.get("eval_index", 0)),
                "program_hash": str(ev.get("program_hash", "")),
                "full_score": full_score,
                "selected_score": selected_score,
                "running_avg_full_score": running_full / i,
                "running_avg_selected_score": running_selected / i,
                "num_full_points": i,
            }
        )
    return out


def _write_surrogate_progress_csv(
    path: Path,
    trace_rows: list[dict],
    eval_index: int,
    warmup_iters: int,
    refit_every: int,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "eval_index",
        "program_hash",
        "full_score",
        "selected_score",
        "running_avg_full_score",
        "running_avg_selected_score",
        "num_full_points",
        "snapshot_eval_index",
        "warmup_iters",
        "refit_every",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in trace_rows:
            out = dict(row)
            out["snapshot_eval_index"] = eval_index
            out["warmup_iters"] = warmup_iters
            out["refit_every"] = refit_every
            w.writerow({k: out.get(k, "") for k in fields})


def _write_surrogate_progress_pdf(
    path: Path,
    trace_rows: list[dict],
    eval_index: int,
    warmup_iters: int,
    refit_every: int,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not trace_rows:
        return

    xs = [int(_safe_float(r.get("eval_index"), default=0)) for r in trace_rows]
    y_sel = [_safe_float(r.get("running_avg_selected_score")) for r in trace_rows]
    y_full = [_safe_float(r.get("running_avg_full_score")) for r in trace_rows]
    raw_sel = [_safe_float(r.get("selected_score")) for r in trace_rows]
    raw_full = [_safe_float(r.get("full_score")) for r in trace_rows]

    fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.0), constrained_layout=True)
    ax.plot(xs, y_sel, marker="o", linewidth=2.0, color="tab:blue", label="Selected score (running avg)")
    ax.plot(xs, y_full, marker="s", linewidth=2.0, color="tab:orange", label="Full score (running avg)")
    ax.scatter(xs, raw_sel, s=22, alpha=0.35, color="tab:blue", label="Selected score (point)")
    ax.scatter(xs, raw_full, s=22, alpha=0.35, color="tab:orange", label="Full score (point)")
    ax.set_xlabel("Evaluation index (rolling/full-eval points)")
    ax.set_ylabel("Combined score")
    ax.set_title(
        f"Surrogate Progress @ eval {eval_index} (warmup={warmup_iters}, refit={refit_every})"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
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


def _refresh_surrogate_progress(
    state_csv: Path,
    meta_json: Path,
    eval_index: int,
    warmup_iters: int,
    refit_every: int,
    write_plot: bool,
) -> tuple[int, float, float]:
    rows = _extract_eval_events(_load_surrogate_rows(state_csv))
    full_rows = [
        r for r in rows
        if _safe_float(r.get("global_combined_score")) == _safe_float(r.get("global_combined_score"))
    ]
    num_full = len(full_rows)
    avg_full = (
        sum(_safe_float(r.get("global_combined_score")) for r in full_rows) / num_full
        if num_full > 0
        else float("nan")
    )
    avg_selected = (
        sum(_safe_float(r.get("sample_combined_score_raw")) for r in full_rows) / num_full
        if num_full > 0
        else float("nan")
    )

    def _update(meta: dict) -> dict:
        meta["full_eval_points"] = int(num_full)
        meta["avg_full_score"] = float(avg_full) if avg_full == avg_full else None
        meta["avg_selected_score"] = float(avg_selected) if avg_selected == avg_selected else None
        meta["selection_strategy"] = _surrogate_selection_mode()
        meta["progress_warmup_iters"] = int(warmup_iters)
        meta["progress_refit_every"] = int(refit_every)
        meta["progress_updated_eval"] = int(eval_index)
        return meta

    _update_meta_json(meta_json, _update)

    if write_plot:
        csv_path = _surrogate_progress_csv(state_csv)
        pdf_path = _surrogate_progress_pdf(state_csv)
        trace_rows = _compute_surrogate_progress_trace(
            state_rows=_load_surrogate_rows(state_csv),
        )
        _write_surrogate_progress_csv(
            csv_path,
            trace_rows=trace_rows,
            eval_index=eval_index,
            warmup_iters=warmup_iters,
            refit_every=refit_every,
        )
        _write_surrogate_progress_pdf(
            pdf_path,
            trace_rows=trace_rows,
            eval_index=eval_index,
            warmup_iters=warmup_iters,
            refit_every=refit_every,
        )
    return num_full, avg_selected, avg_full


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


def _select_bench_size_pairs_by_worst_case(
    benches: list[str],
    candidate_pairs: list[tuple[str, int]],
    case_rows: list[dict],
) -> tuple[list[tuple[str, int]], list[dict]]:
    target_k = _sample_pairs_target_count(benches)
    score_col = "case_score_piecewise" if _score_mode() == _SCORE_MODE_PIECEWISE else "case_score_legacy"
    sums: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}
    for row in _dedupe_case_rows_latest(case_rows):
        try:
            pair = (str(row.get("bench", "")), int(float(row.get("size", "nan"))))
        except Exception:
            continue
        score = _safe_float(row.get(score_col))
        if score != score:
            continue
        sums[pair] = sums.get(pair, 0.0) + float(score)
        counts[pair] = counts.get(pair, 0) + 1

    ranking = []
    candidate_set = set(candidate_pairs)
    for pair in candidate_pairs:
        support = counts.get(pair, 0)
        mean_score = (sums[pair] / support) if support > 0 else float("nan")
        ranking.append(
            {
                "bench": pair[0],
                "size": pair[1],
                "mean_case_score": mean_score,
                "support": support,
            }
        )

    ranking.sort(
        key=lambda r: (
            _safe_float(r.get("mean_case_score"), default=float("inf")),
            -int(_safe_float(r.get("support"), default=0)),
            str(r.get("bench", "")),
            int(_safe_float(r.get("size"), default=0)),
        )
    )

    selected = []
    for row in ranking:
        pair = (str(row["bench"]), int(row["size"]))
        if pair not in candidate_set:
            continue
        if int(_safe_float(row.get("support"), default=0)) <= 0:
            continue
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
    surrogate_warmup_iters = _surrogate_warmup_iters()
    surrogate_refit_every = _surrogate_refit_every()
    surrogate_selection_strategy = _surrogate_selection_mode()
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
    try:
        fixed_pairs = _fixed_bench_size_pairs_from_env(benches)
    except ValueError as exc:
        return {"combined_score": -1000.0}, {"info": str(exc)}

    candidate_pairs = _collect_candidate_pairs(benches, size_min, size_max)
    if not candidate_pairs and (surrogate_enabled or not fixed_pairs):
        return {"combined_score": -1000.0}, {"info": "No valid (bench,size) pairs found"}

    selection_mode = "random"
    selection_support = 0
    selection_signal = float("nan")

    try:
        sample_pairs: list[tuple[str, int]] = []
        if fixed_pairs:
            sample_pairs = fixed_pairs
            selection_mode = "fixed"
            selection_support = len(sample_pairs)
            logger.warning(
                "Using fixed bench/size pairs from QOSE_FIXED_BENCH_SIZE_PAIRS: %s",
                ", ".join(f"({b},{s})" for b, s in sample_pairs),
            )
        elif surrogate_enabled:
            state_rows = _load_surrogate_rows(state_csv)
            case_rows = _load_surrogate_case_rows(cases_csv)
            ranking = []
            if surrogate_selection_strategy == "correlation":
                sample_pairs, ranking = _select_bench_size_pairs_by_correlation(
                    benches=benches,
                    candidate_pairs=candidate_pairs,
                    state_rows=state_rows,
                    case_rows=case_rows,
                )
                if sample_pairs:
                    selection_mode = "correlation"
                    if ranking:
                        selection_support = int(
                            _safe_float(ranking[0].get("support"), default=0)
                        )
                        selection_signal = _safe_float(ranking[0].get("abs_corr"))
            elif surrogate_selection_strategy == "worst":
                sample_pairs, ranking = _select_bench_size_pairs_by_worst_case(
                    benches=benches,
                    candidate_pairs=candidate_pairs,
                    case_rows=case_rows,
                )
                if sample_pairs:
                    selection_mode = "worst"
                    if ranking:
                        selection_support = int(
                            _safe_float(ranking[0].get("support"), default=0)
                        )
                        selection_signal = _safe_float(ranking[0].get("mean_case_score"))
        if not sample_pairs:
            sample_pairs = _select_bench_size_pairs(args, benches)
            selection_mode = "random"
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

    global_combined_score = float("nan")
    surrogate_source = "sample_raw"
    full_eval_error = ""
    timestamp_sec = time.time()

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
                "sample_qose_depth": float(sample_res["avg_depth"]),
                "sample_qose_cnot": float(sample_res["avg_cnot"]),
                "sample_qose_overhead": float(sample_res["avg_overhead"]),
                "sample_avg_run_time": float(sample_res["avg_run_time"]),
                "sample_combined_score_raw": sample_combined_raw,
                "global_combined_score": (
                    global_combined_score if global_combined_score == global_combined_score else ""
                ),
                "score_mode": score_mode,
            },
        )
        _refresh_surrogate_progress(
            state_csv=state_csv,
            meta_json=meta_json,
            eval_index=eval_index,
            warmup_iters=surrogate_warmup_iters,
            refit_every=surrogate_refit_every,
            write_plot=True,
        )

    if global_combined_score == global_combined_score:
        combined_score_final = global_combined_score
        surrogate_source = "global_full_eval"
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

    if surrogate_enabled:
        try:
            _append_surrogate_selection_row(
                _surrogate_selection_csv(state_csv),
                {
                    "eval_index": eval_index,
                    "timestamp_sec": timestamp_sec,
                    "program_hash": program_hash,
                    "selection_mode": selection_mode,
                    "score_source": "raw",
                    "sample_count": len(sample_pairs),
                    "sample_pairs_json": json.dumps(sample_pairs),
                    "selection_support": selection_support,
                    "selection_signal": selection_signal,
                    "corr_support": selection_support,
                    "corr_top_abs_corr": selection_signal,
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
