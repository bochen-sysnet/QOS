import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except Exception:
            continue
    return sorted(set(values))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _combined_piecewise(depth_ratio: float, cnot_ratio: float, time_ratio: float) -> float:
    struct_delta = 1.0 - ((depth_ratio + cnot_ratio) / 2.0)
    time_delta = 1.0 - time_ratio
    struct_term = struct_delta if struct_delta >= 0 else 8.0 * struct_delta
    return struct_term + time_delta


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
    if int(np.sum(valid)) < 2:
        return float("nan")
    xv = x_arr[valid]
    yv = y_arr[valid]
    if np.std(xv) == 0.0 or np.std(yv) == 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _order_key(row: dict[str, Any]) -> tuple[float, float, str]:
    return (
        _safe_float(row.get("generation"), default=-1.0),
        _safe_float(row.get("iteration_found"), default=-1.0),
        str(row.get("program_id", "")),
    )


def _build_pipeline(estimator: Any, use_scaler: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def _models(random_state: int) -> list[tuple[str, Pipeline]]:
    return [
        ("ridge", _build_pipeline(Ridge(alpha=1.0, random_state=random_state), use_scaler=True)),
        (
            "elasticnet",
            _build_pipeline(
                ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=20000, random_state=random_state),
                use_scaler=True,
            ),
        ),
        (
            "random_forest",
            _build_pipeline(
                RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1),
                use_scaler=False,
            ),
        ),
        (
            "extra_trees",
            _build_pipeline(
                ExtraTreesRegressor(n_estimators=500, random_state=random_state, n_jobs=-1),
                use_scaler=False,
            ),
        ),
        (
            "gradient_boosting",
            _build_pipeline(GradientBoostingRegressor(random_state=random_state), use_scaler=False),
        ),
    ]


def _subset_cv_mae(
    X: np.ndarray,
    y: np.ndarray,
    cols: list[int],
    random_state: int,
) -> float:
    if X.ndim != 2 or y.ndim != 1 or not cols:
        return float("inf")
    if X.shape[0] != y.shape[0] or X.shape[0] < 2:
        return float("inf")

    n = int(X.shape[0])
    n_splits = min(5, n)
    if n_splits < 2:
        return float("inf")
    kf = KFold(n_splits=n_splits, shuffle=False)
    errs: list[float] = []
    model = _build_pipeline(Ridge(alpha=1.0, random_state=random_state), use_scaler=True)
    Xs = X[:, cols]
    for train_idx, test_idx in kf.split(Xs):
        try:
            model.fit(Xs[train_idx], y[train_idx])
            pred = model.predict(Xs[test_idx])
        except Exception:
            return float("inf")
        y_true = y[test_idx]
        for p, t in zip(pred, y_true):
            if math.isnan(float(p)) or math.isnan(float(t)):
                continue
            errs.append(abs(float(p) - float(t)))
    if not errs:
        return float("inf")
    return float(np.mean(errs))


def _load_merged(output_dir: Path, merged_csv: Path | None) -> tuple[list[dict[str, Any]], str]:
    # If explicitly provided, trust that file.
    if merged_csv is not None:
        rows = _read_csv(merged_csv)
        if not rows:
            raise RuntimeError(f"No merged rows in {merged_csv}")
        return rows, f"merged_csv={merged_csv}"

    # Prefer rebuilding from canonical cache tables to avoid stale merged snapshots.
    full_csv = output_dir / "gen_seed323_full63_program_scores.csv"
    selected_summary_csv = output_dir / "gen_seed323_selected_summary.csv"
    full_rows = _read_csv(full_csv)
    if full_rows:
        selected_rows = _read_csv(selected_summary_csv)
        selected_by_id = {r.get("program_id", ""): r for r in selected_rows if r.get("program_id")}
        merged_rows: list[dict[str, Any]] = []
        for fr in full_rows:
            pid = fr.get("program_id", "")
            if not pid:
                continue
            row = dict(fr)
            row.update(selected_by_id.get(pid, {}))
            merged_rows.append(row)
        if merged_rows:
            return merged_rows, f"full_scores={full_csv}, selected_summary={selected_summary_csv}"

    # Fallback to historical merged output.
    merged_fallback = output_dir / "gen_seed323_program_merged.csv"
    rows = _read_csv(merged_fallback)
    if not rows:
        raise RuntimeError(
            "No usable cache rows found. Expected either "
            f"{full_csv} or {merged_fallback}"
        )
    return rows, f"merged_csv={merged_fallback}"


def _parse_artifacts_cases(
    program_json: Path,
) -> tuple[list[tuple[str, int]], bool]:
    try:
        obj = json.loads(program_json.read_text(encoding="utf-8"))
    except Exception:
        return [], False
    metadata = obj.get("metadata")
    is_migrant = bool(metadata.get("migrant")) if isinstance(metadata, dict) else False
    artifacts = obj.get("artifacts_json")
    has_artifacts_payload = False
    if isinstance(artifacts, str):
        if artifacts.strip():
            has_artifacts_payload = True
        try:
            artifacts = json.loads(artifacts)
        except Exception:
            artifacts = {}
    elif isinstance(artifacts, dict):
        has_artifacts_payload = True
    if not isinstance(artifacts, dict):
        artifacts = {}
    has_artifacts = has_artifacts_payload or bool(obj.get("artifact_dir"))
    migrant_without_artifacts = is_migrant and not has_artifacts

    cases = artifacts.get("cases")
    if not isinstance(cases, list):
        return [], migrant_without_artifacts
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for case in cases:
        if not isinstance(case, dict):
            continue
        bench = str(case.get("bench", "")).strip()
        if not bench:
            continue
        try:
            size = int(case.get("size"))
        except Exception:
            continue
        key = (bench, size)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out, migrant_without_artifacts


def _guess_program_dir(output_dir: Path) -> Path | None:
    name = output_dir.name
    if name.endswith("_analysis"):
        run_name = name[: -len("_analysis")]
        cand = Path("openevolve_output") / run_name / "checkpoints" / "checkpoint_100" / "programs"
        if cand.exists():
            return cand
    return None


def _numeric_case_columns(case_rows: list[dict[str, str]]) -> list[str]:
    if not case_rows:
        return []
    excluded = {
        "program_id",
        "generation",
        "iteration_found",
        "case_index",
        "bench",
        "qose_method",
    }
    cols: set[str] = set()
    for row in case_rows:
        cols.update(row.keys())
    numeric_cols: list[str] = []
    for col in sorted(cols):
        if col in excluded:
            continue
        has_numeric = False
        for row in case_rows:
            val = _safe_float(row.get(col))
            if not math.isnan(val):
                has_numeric = True
                break
        if has_numeric:
            numeric_cols.append(col)
    return numeric_cols


def _build_feature_rows_from_selected_pairs(
    rows: list[dict[str, Any]],
    case_rows: list[dict[str, str]],
    pid_pair_map: dict[str, dict[tuple[str, int], dict[str, str]]],
    selected_pairs: list[tuple[str, int]],
) -> tuple[list[dict[str, Any]], int, int]:
    numeric_cols = _numeric_case_columns(case_rows)
    out_rows: list[dict[str, Any]] = []
    filled = 0
    unfilled = 0

    for base in rows:
        pid = str(base.get("program_id", "")).strip()
        by_pair = pid_pair_map.get(pid, {})
        selected_case_rows = [by_pair[p] for p in selected_pairs if p in by_pair]

        row: dict[str, Any] = {
            "program_id": pid,
            "generation": base.get("generation", ""),
            "iteration_found": base.get("iteration_found", ""),
            "full_score_piecewise": _safe_float(base.get("full_score_piecewise")),
            "failure_reason": base.get("failure_reason", ""),
            "full_cases_count": int(_safe_float(base.get("full_cases_count"), default=0)),
            "selected_pair_count": len(selected_pairs),
            "selected_matched_count": len(selected_case_rows),
        }

        if not selected_case_rows:
            row["selected_score_piecewise"] = float("nan")
            for col in numeric_cols:
                row[f"sel_{col}_mean"] = float("nan")
                row[f"sel_{col}_std"] = float("nan")
            row["sel_wc_fraction"] = float("nan")
            unfilled += 1
            out_rows.append(row)
            continue

        depth_vals: list[float] = []
        cnot_vals: list[float] = []
        time_vals: list[float] = []
        method_wc_vals: list[float] = []
        for case in selected_case_rows:
            d = _safe_float(case.get("depth_ratio"))
            c = _safe_float(case.get("cnot_ratio"))
            t = _safe_float(case.get("time_ratio"))
            if not (math.isnan(d) or math.isnan(c) or math.isnan(t)):
                depth_vals.append(d)
                cnot_vals.append(c)
                time_vals.append(t)
            method_wc_vals.append(1.0 if str(case.get("qose_method", "")).strip() == "WC" else 0.0)

        if depth_vals:
            row["selected_score_piecewise"] = _combined_piecewise(
                float(np.mean(depth_vals)),
                float(np.mean(cnot_vals)),
                float(np.mean(time_vals)),
            )
        else:
            row["selected_score_piecewise"] = float("nan")

        for col in numeric_cols:
            vals = [_safe_float(case.get(col)) for case in selected_case_rows]
            vals = [v for v in vals if not math.isnan(v)]
            row[f"sel_{col}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"sel_{col}_std"] = float(np.std(vals)) if vals else float("nan")
        row["sel_wc_fraction"] = float(np.mean(method_wc_vals)) if method_wc_vals else float("nan")
        filled += 1
        out_rows.append(row)

    return out_rows, filled, unfilled


def _build_selected_feature_rows(
    rows: list[dict[str, Any]],
    output_dir: Path,
    program_dir: Path | None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    case_csv = output_dir / "gen_seed323_full63_case_metrics.csv"
    case_rows = _read_csv(case_csv)
    if not case_rows:
        return [], 0, len(rows), 0

    if program_dir is None:
        program_dir = _guess_program_dir(output_dir)
    if program_dir is None or not program_dir.exists():
        return [], 0, len(rows), 0

    case_map: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    for r in case_rows:
        pid = str(r.get("program_id", "")).strip()
        bench = str(r.get("bench", "")).strip()
        if not pid or not bench:
            continue
        try:
            size = int(float(r.get("size", "nan")))
        except Exception:
            continue
        case_map.setdefault(pid, {})[(bench, size)] = r

    numeric_cols = _numeric_case_columns(case_rows)
    out_rows: list[dict[str, Any]] = []
    filled = 0
    unfilled = 0
    skipped_migrant_no_artifacts = 0

    for base in rows:
        pid = str(base.get("program_id", "")).strip()
        selected_pairs, is_migrant_without_artifacts = _parse_artifacts_cases(
            program_dir / f"{pid}.json"
        )
        if is_migrant_without_artifacts:
            skipped_migrant_no_artifacts += 1
            continue
        by_pair = case_map.get(pid, {})
        selected_case_rows = [by_pair[p] for p in selected_pairs if p in by_pair]

        row: dict[str, Any] = {
            "program_id": pid,
            "generation": base.get("generation", ""),
            "iteration_found": base.get("iteration_found", ""),
            "full_score_piecewise": _safe_float(base.get("full_score_piecewise")),
            "failure_reason": base.get("failure_reason", ""),
            "full_cases_count": int(_safe_float(base.get("full_cases_count"), default=0)),
            "selected_pair_count": len(selected_pairs),
            "selected_matched_count": len(selected_case_rows),
        }

        if not selected_case_rows:
            row["selected_score_piecewise"] = float("nan")
            for col in numeric_cols:
                row[f"sel_{col}_mean"] = float("nan")
                row[f"sel_{col}_std"] = float("nan")
            row["sel_wc_fraction"] = float("nan")
            unfilled += 1
            out_rows.append(row)
            continue

        depth_vals: list[float] = []
        cnot_vals: list[float] = []
        time_vals: list[float] = []
        method_wc_vals: list[float] = []
        for case in selected_case_rows:
            d = _safe_float(case.get("depth_ratio"))
            c = _safe_float(case.get("cnot_ratio"))
            t = _safe_float(case.get("time_ratio"))
            if not (math.isnan(d) or math.isnan(c) or math.isnan(t)):
                depth_vals.append(d)
                cnot_vals.append(c)
                time_vals.append(t)
            method_wc_vals.append(1.0 if str(case.get("qose_method", "")).strip() == "WC" else 0.0)

        if depth_vals:
            row["selected_score_piecewise"] = _combined_piecewise(
                float(np.mean(depth_vals)),
                float(np.mean(cnot_vals)),
                float(np.mean(time_vals)),
            )
        else:
            row["selected_score_piecewise"] = float("nan")

        for col in numeric_cols:
            vals = [_safe_float(case.get(col)) for case in selected_case_rows]
            vals = [v for v in vals if not math.isnan(v)]
            row[f"sel_{col}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"sel_{col}_std"] = float(np.std(vals)) if vals else float("nan")
        row["sel_wc_fraction"] = float(np.mean(method_wc_vals)) if method_wc_vals else float("nan")
        filled += 1
        out_rows.append(row)

    return out_rows, filled, unfilled, skipped_migrant_no_artifacts


def _build_corr_selected_feature_rows(
    rows: list[dict[str, Any]],
    output_dir: Path,
    selection_rows: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], int, int, list[dict[str, Any]]]:
    case_csv = output_dir / "gen_seed323_full63_case_metrics.csv"
    case_rows = _read_csv(case_csv)
    if not case_rows:
        return [], 0, len(rows), []

    pid_pair_map: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    pair_pid_map: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
    for r in case_rows:
        pid = str(r.get("program_id", "")).strip()
        bench = str(r.get("bench", "")).strip()
        if not pid or not bench:
            continue
        try:
            size = int(float(r.get("size", "nan")))
        except Exception:
            continue
        pair = (bench, size)
        pid_pair_map.setdefault(pid, {})[pair] = r
        pair_pid_map.setdefault(pair, {})[pid] = r

    y_by_pid: dict[str, float] = {}
    for r in selection_rows:
        pid = str(r.get("program_id", "")).strip()
        y = _safe_float(r.get("full_score_piecewise"))
        if pid and not math.isnan(y):
            y_by_pid[pid] = y

    ranking: list[dict[str, Any]] = []
    for pair, pid_rows in pair_pid_map.items():
        xs: list[float] = []
        ys: list[float] = []
        for pid, y in y_by_pid.items():
            row = pid_rows.get(pid)
            if row is None:
                continue
            x = _safe_float(row.get("full_case_piecewise_score"))
            if math.isnan(x):
                d = _safe_float(row.get("depth_ratio"))
                c = _safe_float(row.get("cnot_ratio"))
                t = _safe_float(row.get("time_ratio"))
                if not (math.isnan(d) or math.isnan(c) or math.isnan(t)):
                    x = _combined_piecewise(d, c, t)
            if math.isnan(x):
                continue
            xs.append(x)
            ys.append(y)
        corr = _pearson_corr(xs, ys)
        ranking.append(
            {
                "bench": pair[0],
                "size": pair[1],
                "abs_corr": abs(corr) if not math.isnan(corr) else float("nan"),
                "corr": corr,
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

    top_k = max(1, int(top_k))
    selected_ranked = [r for r in ranking if not math.isnan(_safe_float(r.get("abs_corr")))]
    if not selected_ranked:
        selected_ranked = ranking
    selected_ranked = selected_ranked[:top_k]
    selected_pairs = [(str(r["bench"]), int(r["size"])) for r in selected_ranked]

    out_rows, filled, unfilled = _build_feature_rows_from_selected_pairs(
        rows=rows,
        case_rows=case_rows,
        pid_pair_map=pid_pair_map,
        selected_pairs=selected_pairs,
    )

    selection_table: list[dict[str, Any]] = []
    selected_key = {(str(r["bench"]), int(r["size"])) for r in selected_ranked}
    rank_lookup = {k: i + 1 for i, k in enumerate(selected_pairs)}
    for r in ranking:
        key = (str(r["bench"]), int(r["size"]))
        selection_table.append(
            {
                "bench": key[0],
                "size": key[1],
                "selected": 1 if key in selected_key else 0,
                "selected_rank": rank_lookup.get(key, ""),
                "abs_corr": _safe_float(r.get("abs_corr")),
                "corr": _safe_float(r.get("corr")),
                "support": int(_safe_float(r.get("support"), default=0)),
            }
        )

    return out_rows, filled, unfilled, selection_table


def _build_ml_selected_feature_rows(
    rows: list[dict[str, Any]],
    output_dir: Path,
    selection_rows: list[dict[str, Any]],
    top_k: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], int, int, list[dict[str, Any]]]:
    case_csv = output_dir / "gen_seed323_full63_case_metrics.csv"
    case_rows = _read_csv(case_csv)
    if not case_rows:
        return [], 0, len(rows), []

    pid_pair_map: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    pair_pid_map: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
    for r in case_rows:
        pid = str(r.get("program_id", "")).strip()
        bench = str(r.get("bench", "")).strip()
        if not pid or not bench:
            continue
        try:
            size = int(float(r.get("size", "nan")))
        except Exception:
            continue
        pair = (bench, size)
        pid_pair_map.setdefault(pid, {})[pair] = r
        pair_pid_map.setdefault(pair, {})[pid] = r

    pairs = sorted(pair_pid_map.keys(), key=lambda x: (x[0], x[1]))
    train_rows: list[list[float]] = []
    y_vals: list[float] = []
    pair_stats: list[dict[str, Any]] = []
    for pair in pairs:
        xs: list[float] = []
        ys: list[float] = []
        pid_rows = pair_pid_map.get(pair, {})
        for sr in selection_rows:
            pid = str(sr.get("program_id", "")).strip()
            y = _safe_float(sr.get("full_score_piecewise"))
            if not pid or math.isnan(y):
                continue
            case = pid_rows.get(pid)
            if case is None:
                continue
            x = _safe_float(case.get("full_case_piecewise_score"))
            if math.isnan(x):
                d = _safe_float(case.get("depth_ratio"))
                c = _safe_float(case.get("cnot_ratio"))
                t = _safe_float(case.get("time_ratio"))
                if not (math.isnan(d) or math.isnan(c) or math.isnan(t)):
                    x = _combined_piecewise(d, c, t)
            if math.isnan(x):
                continue
            xs.append(x)
            ys.append(y)
        pair_stats.append(
            {
                "pair": pair,
                "abs_corr": abs(_pearson_corr(xs, ys)) if xs and ys else float("nan"),
                "support": len(xs),
            }
        )

    for r in selection_rows:
        pid = str(r.get("program_id", "")).strip()
        y = _safe_float(r.get("full_score_piecewise"))
        if not pid or math.isnan(y):
            continue
        by_pair = pid_pair_map.get(pid, {})
        x_row: list[float] = []
        for p in pairs:
            case = by_pair.get(p)
            if case is None:
                x_row.append(float("nan"))
                continue
            x = _safe_float(case.get("full_case_piecewise_score"))
            if math.isnan(x):
                d = _safe_float(case.get("depth_ratio"))
                c = _safe_float(case.get("cnot_ratio"))
                t = _safe_float(case.get("time_ratio"))
                if not (math.isnan(d) or math.isnan(c) or math.isnan(t)):
                    x = _combined_piecewise(d, c, t)
            x_row.append(x)
        train_rows.append(x_row)
        y_vals.append(y)

    top_k = max(1, int(top_k))
    if len(train_rows) < 3 or len(pairs) < 1:
        return _build_corr_selected_feature_rows(
            rows=rows,
            output_dir=output_dir,
            selection_rows=selection_rows,
            top_k=top_k,
        )

    X = np.array(train_rows, dtype=float)
    y = np.array(y_vals, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] < 3 or X.shape[1] < 1:
        return _build_corr_selected_feature_rows(
            rows=rows,
            output_dir=output_dir,
            selection_rows=selection_rows,
            top_k=top_k,
        )

    idx_stats = []
    pair_to_idx = {pair: i for i, pair in enumerate(pairs)}
    for ps in pair_stats:
        idx_stats.append(
            {
                "idx": int(pair_to_idx[ps["pair"]]),
                "pair": ps["pair"],
                "abs_corr": _safe_float(ps.get("abs_corr")),
                "support": int(_safe_float(ps.get("support"), default=0)),
            }
        )
    idx_stats.sort(
        key=lambda r: (
            -_safe_float(r.get("abs_corr"), default=-1.0),
            -int(_safe_float(r.get("support"), default=0)),
            str(r.get("pair", ("", 0))[0]),
            int(r.get("pair", ("", 0))[1]),
        )
    )

    pool_size = min(len(idx_stats), max(top_k * 4, 20))
    candidate_indices = [int(r["idx"]) for r in idx_stats[:pool_size]]
    if not candidate_indices:
        candidate_indices = list(range(len(pairs)))

    cache_loss: dict[tuple[int, ...], float] = {}
    selected_indices: list[int] = []
    selected_loss: list[float] = []
    max_select = min(top_k, len(candidate_indices))
    for _ in range(max_select):
        best_idx = None
        best_loss = float("inf")
        for idx in candidate_indices:
            if idx in selected_indices:
                continue
            cols = tuple(sorted(selected_indices + [idx]))
            if cols not in cache_loss:
                cache_loss[cols] = _subset_cv_mae(X, y, list(cols), random_state=random_state)
            loss = cache_loss[cols]
            if loss < best_loss:
                best_loss = loss
                best_idx = idx
            elif loss == best_loss and best_idx is not None:
                # tie-break toward stronger univariate correlation
                corr_idx = next(
                    (_safe_float(r.get("abs_corr")) for r in idx_stats if int(r.get("idx", -1)) == idx),
                    float("nan"),
                )
                corr_best = next(
                    (_safe_float(r.get("abs_corr")) for r in idx_stats if int(r.get("idx", -1)) == best_idx),
                    float("nan"),
                )
                if not math.isnan(corr_idx) and (math.isnan(corr_best) or corr_idx > corr_best):
                    best_idx = idx
        if best_idx is None:
            break
        selected_indices.append(int(best_idx))
        selected_loss.append(float(best_loss))

    if not selected_indices:
        return _build_corr_selected_feature_rows(
            rows=rows,
            output_dir=output_dir,
            selection_rows=selection_rows,
            top_k=top_k,
        )

    selected_pairs = [pairs[i] for i in selected_indices]

    out_rows, filled, unfilled = _build_feature_rows_from_selected_pairs(
        rows=rows,
        case_rows=case_rows,
        pid_pair_map=pid_pair_map,
        selected_pairs=selected_pairs,
    )

    selected_key = {(str(p[0]), int(p[1])) for p in selected_pairs}
    rank_lookup = {k: i + 1 for i, k in enumerate(selected_pairs)}
    loss_lookup = {selected_pairs[i]: selected_loss[i] for i in range(len(selected_pairs))}
    selection_table: list[dict[str, Any]] = []
    for r in idx_stats:
        pair = r.get("pair", ("", 0))
        key = (str(pair[0]), int(pair[1]))
        selection_table.append(
            {
                "bench": key[0],
                "size": key[1],
                "selected": 1 if key in selected_key else 0,
                "selected_rank": rank_lookup.get(key, ""),
                "abs_corr": _safe_float(r.get("abs_corr")),
                "cv_mae_if_selected": _safe_float(loss_lookup.get(key)),
                "support": int(_safe_float(r.get("support"), default=0)),
            }
        )

    return out_rows, filled, unfilled, selection_table


def _build_ml_importance_selected_feature_rows(
    rows: list[dict[str, Any]],
    output_dir: Path,
    selection_rows: list[dict[str, Any]],
    top_k: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], int, int, list[dict[str, Any]]]:
    case_csv = output_dir / "gen_seed323_full63_case_metrics.csv"
    case_rows = _read_csv(case_csv)
    if not case_rows:
        return [], 0, len(rows), []

    pid_pair_map: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    pair_pid_map: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
    for r in case_rows:
        pid = str(r.get("program_id", "")).strip()
        bench = str(r.get("bench", "")).strip()
        if not pid or not bench:
            continue
        try:
            size = int(float(r.get("size", "nan")))
        except Exception:
            continue
        pair = (bench, size)
        pid_pair_map.setdefault(pid, {})[pair] = r
        pair_pid_map.setdefault(pair, {})[pid] = r

    pairs = sorted(pair_pid_map.keys(), key=lambda x: (x[0], x[1]))
    train_rows: list[list[float]] = []
    y_vals: list[float] = []
    for r in selection_rows:
        pid = str(r.get("program_id", "")).strip()
        y = _safe_float(r.get("full_score_piecewise"))
        if not pid or math.isnan(y):
            continue
        by_pair = pid_pair_map.get(pid, {})
        x_row: list[float] = []
        for p in pairs:
            case = by_pair.get(p)
            if case is None:
                x_row.append(float("nan"))
                continue
            x = _safe_float(case.get("full_case_piecewise_score"))
            if math.isnan(x):
                d = _safe_float(case.get("depth_ratio"))
                c = _safe_float(case.get("cnot_ratio"))
                t = _safe_float(case.get("time_ratio"))
                if not (math.isnan(d) or math.isnan(c) or math.isnan(t)):
                    x = _combined_piecewise(d, c, t)
            x_row.append(x)
        train_rows.append(x_row)
        y_vals.append(y)

    top_k = max(1, int(top_k))
    if len(train_rows) < 3 or len(pairs) < 1:
        return _build_corr_selected_feature_rows(
            rows=rows,
            output_dir=output_dir,
            selection_rows=selection_rows,
            top_k=top_k,
        )

    importances = np.full(len(pairs), np.nan, dtype=float)
    try:
        X = np.array(train_rows, dtype=float)
        y = np.array(y_vals, dtype=float)
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(X)
        if X.shape[0] >= 3 and X.shape[1] >= 1:
            model = ExtraTreesRegressor(
                n_estimators=400,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X, y)
            importances = np.array(model.feature_importances_, dtype=float)
    except Exception:
        importances = np.full(len(pairs), np.nan, dtype=float)

    if np.all(np.isnan(importances)):
        return _build_corr_selected_feature_rows(
            rows=rows,
            output_dir=output_dir,
            selection_rows=selection_rows,
            top_k=top_k,
        )

    ranking: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        pid_rows = pair_pid_map.get(pair, {})
        support = 0
        for sr in selection_rows:
            pid = str(sr.get("program_id", "")).strip()
            if pid in pid_rows:
                support += 1
        imp = float(importances[idx]) if idx < len(importances) else float("nan")
        ranking.append(
            {
                "bench": pair[0],
                "size": pair[1],
                "importance": imp,
                "support": support,
            }
        )
    ranking.sort(
        key=lambda r: (
            -_safe_float(r.get("importance"), default=-1.0),
            -int(_safe_float(r.get("support"), default=0)),
            str(r.get("bench", "")),
            int(_safe_float(r.get("size"), default=0)),
        )
    )
    selected_ranked = ranking[:top_k]
    selected_pairs = [(str(r["bench"]), int(r["size"])) for r in selected_ranked]

    out_rows, filled, unfilled = _build_feature_rows_from_selected_pairs(
        rows=rows,
        case_rows=case_rows,
        pid_pair_map=pid_pair_map,
        selected_pairs=selected_pairs,
    )

    selected_key = {(str(r["bench"]), int(r["size"])) for r in selected_ranked}
    rank_lookup = {k: i + 1 for i, k in enumerate(selected_pairs)}
    selection_table: list[dict[str, Any]] = []
    for r in ranking:
        key = (str(r["bench"]), int(r["size"]))
        selection_table.append(
            {
                "bench": key[0],
                "size": key[1],
                "selected": 1 if key in selected_key else 0,
                "selected_rank": rank_lookup.get(key, ""),
                "importance": _safe_float(r.get("importance")),
                "support": int(_safe_float(r.get("support"), default=0)),
            }
        )
    return out_rows, filled, unfilled, selection_table


def _selector_cv_score_from_feature_rows(
    feature_rows: list[dict[str, Any]],
    train_base: list[dict[str, Any]],
    random_state: int,
) -> float:
    feature_map = {str(r.get("program_id", "")): r for r in feature_rows}
    train_rows = [
        feature_map.get(str(r.get("program_id", "")))
        for r in train_base
        if feature_map.get(str(r.get("program_id", ""))) is not None
    ]
    if len(train_rows) < 4:
        return float("inf")
    feature_cols = _feature_columns(train_rows)
    if not feature_cols:
        return float("inf")
    X = np.array([[_safe_float(r.get(c)) for c in feature_cols] for r in train_rows], dtype=float)
    y = np.array([_safe_float(r.get("full_score_piecewise")) for r in train_rows], dtype=float)
    valid = ~np.isnan(y)
    if int(np.sum(valid)) < 4:
        return float("inf")
    X = X[valid]
    y = y[valid]
    n = int(X.shape[0])
    n_splits = min(5, n)
    if n_splits < 2:
        return float("inf")
    kf = KFold(n_splits=n_splits, shuffle=False)

    model_builders = [
        lambda: _build_pipeline(Ridge(alpha=1.0, random_state=random_state), use_scaler=True),
        lambda: _build_pipeline(
            ExtraTreesRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
            use_scaler=False,
        ),
        lambda: _build_pipeline(
            GradientBoostingRegressor(random_state=random_state),
            use_scaler=False,
        ),
    ]
    errs: list[float] = []
    for tr_idx, te_idx in kf.split(X):
        for build in model_builders:
            try:
                m = build()
                m.fit(X[tr_idx], y[tr_idx])
                pred = m.predict(X[te_idx])
            except Exception:
                return float("inf")
            for p, t in zip(pred, y[te_idx]):
                if not (math.isnan(float(p)) or math.isnan(float(t))):
                    errs.append(abs(float(p) - float(t)))
    if not errs:
        return float("inf")
    return float(np.mean(errs))


def _feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    all_cols = set()
    for r in rows:
        all_cols.update(r.keys())
    return sorted(c for c in all_cols if c.startswith("sel_"))


def _iterative_eval(rows: list[dict[str, Any]], random_state: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=_order_key)
    feature_cols = _feature_columns(rows)
    model_defs = _models(random_state)
    out_rows: list[dict[str, Any]] = []

    for k in range(1, len(rows)):
        train_rows = rows[:k]
        test_row = rows[k]
        y_actual = _safe_float(test_row.get("full_score_piecewise"))
        y_selected = _safe_float(test_row.get("selected_score_piecewise"))
        selected_gap = abs(y_selected - y_actual) if not (math.isnan(y_selected) or math.isnan(y_actual)) else float("nan")

        x_train = np.array(
            [[_safe_float(r.get(c)) for c in feature_cols] for r in train_rows], dtype=float
        )
        y_train = np.array([_safe_float(r.get("full_score_piecewise")) for r in train_rows], dtype=float)
        x_test = np.array([[_safe_float(test_row.get(c)) for c in feature_cols]], dtype=float)

        for model_name, model in model_defs:
            pred = float("nan")
            err_msg = ""
            try:
                model.fit(x_train, y_train)
                pred = float(model.predict(x_test)[0])
            except Exception as exc:
                err_msg = str(exc)
            model_gap = abs(pred - y_actual) if not (math.isnan(pred) or math.isnan(y_actual)) else float("nan")
            gap_reduction = (
                selected_gap - model_gap
                if not (math.isnan(selected_gap) or math.isnan(model_gap))
                else float("nan")
            )
            out_rows.append(
                {
                    "k_index": k + 1,
                    "train_size": k,
                    "program_id": test_row.get("program_id", ""),
                    "generation": test_row.get("generation", ""),
                    "iteration_found": test_row.get("iteration_found", ""),
                    "model": model_name,
                    "actual_full_score_piecewise": y_actual,
                    "selected_score_piecewise": y_selected,
                    "selected_gap_abs": selected_gap,
                    "predicted_full_score_piecewise": pred,
                    "model_gap_abs": model_gap,
                    "gap_reduction_abs": gap_reduction,
                    "error": err_msg,
                }
            )
    return out_rows


def _iterative_eval_ml_rolling(
    rows: list[dict[str, Any]],
    output_dir: Path,
    top_k: int,
    random_state: int,
    selector_mode: str = "corr",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = sorted(rows, key=_order_key)
    model_defs = _models(random_state)
    out_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for k in range(1, len(rows)):
        train_base = rows[:k]
        test_base = rows[k]

        mode = selector_mode.strip().lower()
        if mode == "ml_greedy_cv":
            mode = "ml"

        def _build_by_mode(selector: str):
            if selector == "ml":
                return _build_ml_selected_feature_rows(
                    rows=train_base + [test_base],
                    output_dir=output_dir,
                    selection_rows=train_base,
                    top_k=top_k,
                    random_state=random_state,
                )
            if selector == "ml_importance":
                return _build_ml_importance_selected_feature_rows(
                    rows=train_base + [test_base],
                    output_dir=output_dir,
                    selection_rows=train_base,
                    top_k=top_k,
                    random_state=random_state,
                )
            return _build_corr_selected_feature_rows(
                rows=train_base + [test_base],
                output_dir=output_dir,
                selection_rows=train_base,
                top_k=top_k,
            )

        chosen_mode = mode
        if mode == "auto":
            candidates = ["corr", "ml_importance", "ml"]
            scored: list[tuple[float, str, list[dict[str, Any]], list[dict[str, Any]]]] = []
            for cand in candidates:
                rows_c, _, _, sel_c = _build_by_mode(cand)
                score_c = _selector_cv_score_from_feature_rows(
                    feature_rows=rows_c,
                    train_base=train_base,
                    random_state=random_state,
                )
                scored.append((score_c, cand, rows_c, sel_c))
            scored.sort(key=lambda t: (t[0], t[1]))
            _, chosen_mode, feature_rows_k, selection_table_k = scored[0]
        else:
            feature_rows_k, _, _, selection_table_k = _build_by_mode(chosen_mode)

        feature_map = {str(r.get("program_id", "")): r for r in feature_rows_k}
        train_rows = [
            feature_map.get(str(r.get("program_id", "")))
            for r in train_base
            if feature_map.get(str(r.get("program_id", ""))) is not None
        ]
        test_row = feature_map.get(str(test_base.get("program_id", "")))
        if not train_rows or test_row is None:
            continue

        feature_cols = _feature_columns(train_rows + [test_row])
        y_actual = _safe_float(test_row.get("full_score_piecewise"))
        y_selected = _safe_float(test_row.get("selected_score_piecewise"))
        selected_gap = (
            abs(y_selected - y_actual)
            if not (math.isnan(y_selected) or math.isnan(y_actual))
            else float("nan")
        )

        x_train = np.array(
            [[_safe_float(r.get(c)) for c in feature_cols] for r in train_rows], dtype=float
        )
        y_train = np.array(
            [_safe_float(r.get("full_score_piecewise")) for r in train_rows], dtype=float
        )
        x_test = np.array([[_safe_float(test_row.get(c)) for c in feature_cols]], dtype=float)

        for model_name, model in model_defs:
            pred = float("nan")
            err_msg = ""
            try:
                model.fit(x_train, y_train)
                pred = float(model.predict(x_test)[0])
            except Exception as exc:
                err_msg = str(exc)
            model_gap = (
                abs(pred - y_actual)
                if not (math.isnan(pred) or math.isnan(y_actual))
                else float("nan")
            )
            gap_reduction = (
                selected_gap - model_gap
                if not (math.isnan(selected_gap) or math.isnan(model_gap))
                else float("nan")
            )
            out_rows.append(
                {
                    "k_index": k + 1,
                    "train_size": k,
                    "program_id": test_row.get("program_id", ""),
                    "generation": test_row.get("generation", ""),
                    "iteration_found": test_row.get("iteration_found", ""),
                    "model": model_name,
                    "actual_full_score_piecewise": y_actual,
                    "selected_score_piecewise": y_selected,
                    "selected_gap_abs": selected_gap,
                    "predicted_full_score_piecewise": pred,
                    "model_gap_abs": model_gap,
                    "gap_reduction_abs": gap_reduction,
                    "selector_mode": chosen_mode,
                    "error": err_msg,
                }
            )

        selected_k = [
            s
            for s in selection_table_k
            if int(_safe_float(s.get("selected"), default=0)) == 1
        ]
        selected_k.sort(key=lambda s: int(_safe_float(s.get("selected_rank"), default=9999)))
        for s in selected_k:
            criterion_value = _safe_float(s.get("abs_corr"))
            criterion_name = "abs_corr"
            if math.isnan(criterion_value):
                criterion_value = _safe_float(s.get("cv_mae_if_selected"))
                criterion_name = "cv_mae_if_selected"
            selected_rows.append(
                {
                    "k_index": k + 1,
                    "train_size": k,
                    "bench": s.get("bench", ""),
                    "size": int(_safe_float(s.get("size"), default=0)),
                    "selected_rank": int(_safe_float(s.get("selected_rank"), default=0)),
                    "criterion_name": criterion_name,
                    "criterion_value": criterion_value,
                    "support": int(_safe_float(s.get("support"), default=0)),
                    "selector_mode": chosen_mode,
                }
            )

    return out_rows, selected_rows


def _plot(records: list[dict[str, Any]], out_pdf: Path, warmups: list[int]) -> None:
    model_names = sorted({str(r["model"]) for r in records})
    k_vals = sorted({int(r["k_index"]) for r in records})

    selected_score_by_k: list[float] = []
    actual_score_by_k: list[float] = []
    gap_by_model: dict[str, list[float]] = {name: [] for name in model_names}
    ratio_by_model: dict[str, list[float]] = {name: [] for name in model_names}
    pred_score_by_model: dict[str, list[float]] = {name: [] for name in model_names}

    for k in k_vals:
        k_rows = [r for r in records if int(r["k_index"]) == k]
        selected_score = next(
            (_safe_float(r.get("selected_score_piecewise")) for r in k_rows),
            float("nan"),
        )
        actual_score = next(
            (_safe_float(r.get("actual_full_score_piecewise")) for r in k_rows),
            float("nan"),
        )
        selected_score_by_k.append(selected_score)
        actual_score_by_k.append(actual_score)

        selected_gap = next(
            (_safe_float(r.get("selected_gap_abs")) for r in k_rows),
            float("nan"),
        )
        for name in model_names:
            row = next((r for r in k_rows if str(r["model"]) == name), None)
            gap = _safe_float(row.get("model_gap_abs")) if row else float("nan")
            pred_score = _safe_float(row.get("predicted_full_score_piecewise")) if row else float("nan")
            gap_by_model[name].append(gap)
            pred_score_by_model[name].append(pred_score)
            if math.isnan(gap) or math.isnan(selected_gap) or selected_gap <= 0.0:
                ratio = float("nan")
            else:
                ratio = gap / selected_gap
            ratio_by_model[name].append(ratio)

    fig, axes = plt.subplots(2, 3, figsize=(26.0, 10.8))
    ax_score, ax_gap, ax_ratio, ax_corr, ax_stats, ax_warmup = axes.ravel()

    ax_score.plot(
        k_vals,
        selected_score_by_k,
        marker="o",
        linewidth=2.0,
        color="black",
        label="selected score",
    )
    ax_score.plot(
        k_vals,
        actual_score_by_k,
        marker="o",
        linewidth=2.0,
        color="tab:red",
        label="all-circuits avg score",
    )
    for name in model_names:
        ax_score.plot(
            k_vals,
            pred_score_by_model[name],
            marker="o",
            linewidth=1.6,
            linestyle="--",
            label=f"pred:{name}",
        )
    ax_score.set_title("Score Trajectory")
    ax_score.set_xlabel("k (predict k using first k-1 programs)")
    ax_score.set_ylabel("Piecewise score")
    ax_score.grid(alpha=0.2)
    ax_score.legend(loc="best", frameon=False)

    for name in model_names:
        ax_gap.plot(
            k_vals,
            gap_by_model[name],
            marker="o",
            linewidth=1.6,
            linestyle="--",
            label=name,
        )
    ax_gap.set_title("Absolute Prediction Gap")
    ax_gap.set_xlabel("k (predict k using first k-1 programs)")
    ax_gap.set_ylabel("|predicted - all-circuits|")
    ax_gap.grid(alpha=0.2)
    ax_gap.legend(loc="best", frameon=False)

    for name in model_names:
        ax_ratio.plot(
            k_vals,
            ratio_by_model[name],
            marker="o",
            linewidth=1.8,
            linestyle="--",
            label=name,
        )
    ax_ratio.set_title("Normalized Gap vs Selected Baseline")
    ax_ratio.set_xlabel("k (predict k using first k-1 programs)")
    ax_ratio.set_ylabel("pred_diff / selected_diff")
    ax_ratio.grid(alpha=0.2)
    ax_ratio.legend(loc="best", frameon=False)

    def _corr(xs: list[float], ys: list[float]) -> float:
        x_arr = np.array(xs, dtype=float)
        y_arr = np.array(ys, dtype=float)
        valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
        if int(np.sum(valid)) < 2:
            return float("nan")
        xv = x_arr[valid]
        yv = y_arr[valid]
        if np.std(xv) == 0.0 or np.std(yv) == 0.0:
            return float("nan")
        return float(np.corrcoef(xv, yv)[0, 1])

    corr_labels = ["baseline(selected)"] + model_names
    corr_vals = [_corr(actual_score_by_k, selected_score_by_k)]
    corr_vals.extend(_corr(actual_score_by_k, pred_score_by_model[name]) for name in model_names)
    corr_plot_vals = [0.0 if math.isnan(v) else v for v in corr_vals]
    bar_colors = ["tab:gray"] + ["tab:blue"] * len(model_names)
    bars = ax_corr.bar(corr_labels, corr_plot_vals, color=bar_colors, alpha=0.9)
    for b, v in zip(bars, corr_vals):
        label = "nan" if math.isnan(v) else f"{v:.2f}"
        ax_corr.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + (0.02 if b.get_height() >= 0 else -0.06),
            label,
            ha="center",
            va="bottom" if b.get_height() >= 0 else "top",
            fontsize=9,
        )
    ax_corr.set_title("Correlation with Global Score")
    ax_corr.set_ylabel("Pearson r")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.grid(axis="y", alpha=0.2)
    ax_corr.tick_params(axis="x", labelrotation=25)

    selected_gap_by_k: list[float] = []
    for s, a in zip(selected_score_by_k, actual_score_by_k):
        if math.isnan(s) or math.isnan(a):
            selected_gap_by_k.append(float("nan"))
        else:
            selected_gap_by_k.append(abs(s - a))

    stat_labels = ["baseline(selected)"] + model_names
    stat_series = [selected_gap_by_k] + [gap_by_model[name] for name in model_names]
    stat_means: list[float] = []
    stat_stds: list[float] = []
    for vals in stat_series:
        arr = np.array(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            stat_means.append(float("nan"))
            stat_stds.append(float("nan"))
        else:
            stat_means.append(float(np.mean(arr)))
            stat_stds.append(float(np.std(arr)))

    bar_x = np.arange(len(stat_labels))
    bar_heights = [0.0 if math.isnan(v) else v for v in stat_means]
    bar_err = [0.0 if math.isnan(v) else v for v in stat_stds]
    stats_colors = ["tab:gray"] + ["tab:blue"] * len(model_names)
    bars = ax_stats.bar(
        bar_x,
        bar_heights,
        yerr=bar_err,
        capsize=4,
        color=stats_colors,
        alpha=0.9,
    )
    for i, (b, m, s) in enumerate(zip(bars, stat_means, stat_stds)):
        label = "nan" if math.isnan(m) else (f"{m:.3f}" if math.isnan(s) else f"{m:.3f}\u00b1{s:.3f}")
        ax_stats.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + (bar_err[i] if i < len(bar_err) else 0.0) + 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_stats.set_xticks(bar_x)
    ax_stats.set_xticklabels(stat_labels, rotation=25)
    ax_stats.set_title("Mean\u00b1Std of |method-all| Across k")
    ax_stats.set_ylabel("Absolute diff")
    ax_stats.grid(axis="y", alpha=0.2)

    if not warmups:
        warmups = [0, 2, 6, 10]
    warmups = sorted(set(warmups))
    if warmups[0] != 0:
        warmups = [0] + warmups

    def _mean_after_warmup(values: list[float], kmin: int) -> float:
        picked: list[float] = []
        for i, v in enumerate(values):
            if i >= len(k_vals):
                break
            if k_vals[i] < kmin or math.isnan(v):
                continue
            picked.append(v)
        if not picked:
            return float("nan")
        return float(np.mean(picked))

    warm_labels = ["baseline(selected)"] + model_names
    warm_series = [selected_gap_by_k] + [gap_by_model[m] for m in model_names]
    for label, series in zip(warm_labels, warm_series):
        ys = [_mean_after_warmup(series, w) for w in warmups]
        ax_warmup.plot(warmups, ys, marker="o", linewidth=2.0 if label == "baseline(selected)" else 1.8, label=label)
    ax_warmup.set_title("Mean |method-all| After Warmup")
    ax_warmup.set_xlabel("Warmup k_min (use k >= k_min)")
    ax_warmup.set_ylabel("Mean absolute diff")
    ax_warmup.grid(alpha=0.2)
    ax_warmup.legend(loc="best", frameon=False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def _warmup_summary(records: list[dict[str, Any]], warmups: list[int]) -> list[dict[str, Any]]:
    if not warmups:
        warmups = [0, 2, 6, 10]
    warmups = sorted(set(warmups))
    if warmups[0] != 0:
        warmups = [0] + warmups

    model_names = sorted({str(r["model"]) for r in records})
    k_vals = sorted({int(r["k_index"]) for r in records})

    by_k: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        by_k.setdefault(int(r["k_index"]), []).append(r)

    baseline_by_k: list[float] = []
    model_gap_by_k: dict[str, list[float]] = {m: [] for m in model_names}
    for k in k_vals:
        rows_k = by_k.get(k, [])
        if rows_k:
            baseline_by_k.append(_safe_float(rows_k[0].get("selected_gap_abs")))
        else:
            baseline_by_k.append(float("nan"))
        for m in model_names:
            row = next((x for x in rows_k if str(x.get("model", "")) == m), None)
            model_gap_by_k[m].append(_safe_float(row.get("model_gap_abs")) if row else float("nan"))

    def _stats_after(values: list[float], kmin: int) -> tuple[float, float, int]:
        picked: list[float] = []
        for idx, k in enumerate(k_vals):
            if k < kmin or idx >= len(values):
                continue
            v = values[idx]
            if not math.isnan(v):
                picked.append(v)
        if not picked:
            return float("nan"), float("nan"), 0
        arr = np.array(picked, dtype=float)
        return float(np.mean(arr)), float(np.std(arr)), int(arr.size)

    rows_out: list[dict[str, Any]] = []
    for w in warmups:
        mean_b, std_b, n_b = _stats_after(baseline_by_k, w)
        rows_out.append(
            {
                "warmup_kmin": w,
                "method": "baseline_selected",
                "mean_abs_diff": mean_b,
                "std_abs_diff": std_b,
                "count": n_b,
            }
        )
        for m in model_names:
            mean_m, std_m, n_m = _stats_after(model_gap_by_k[m], w)
            rows_out.append(
                {
                    "warmup_kmin": w,
                    "method": m,
                    "mean_abs_diff": mean_m,
                    "std_abs_diff": std_m,
                    "count": n_m,
                }
            )
    return rows_out


def _selector_compare_summary(
    selector_records: dict[str, list[dict[str, Any]]],
    warmups: list[int],
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for selector_name, recs in selector_records.items():
        warm = _warmup_summary(recs, warmups)
        for r in warm:
            rows_out.append(
                {
                    "selector": selector_name,
                    "warmup_kmin": int(_safe_float(r.get("warmup_kmin"), default=0)),
                    "method": str(r.get("method", "")),
                    "mean_abs_diff": _safe_float(r.get("mean_abs_diff")),
                    "std_abs_diff": _safe_float(r.get("std_abs_diff")),
                    "count": int(_safe_float(r.get("count"), default=0)),
                }
            )
    return rows_out


def _selector_metrics_by_k(records: list[dict[str, Any]]) -> dict[str, Any]:
    model_names = sorted({str(r["model"]) for r in records})
    k_vals = sorted({int(r["k_index"]) for r in records})
    selected_score_by_k: list[float] = []
    actual_score_by_k: list[float] = []
    gap_by_model: dict[str, list[float]] = {name: [] for name in model_names}
    pred_by_model: dict[str, list[float]] = {name: [] for name in model_names}
    for k in k_vals:
        k_rows = [r for r in records if int(r["k_index"]) == k]
        selected_score_by_k.append(
            next((_safe_float(r.get("selected_score_piecewise")) for r in k_rows), float("nan"))
        )
        actual_score_by_k.append(
            next((_safe_float(r.get("actual_full_score_piecewise")) for r in k_rows), float("nan"))
        )
        for name in model_names:
            row = next((r for r in k_rows if str(r["model"]) == name), None)
            gap_by_model[name].append(_safe_float(row.get("model_gap_abs")) if row else float("nan"))
            pred_by_model[name].append(
                _safe_float(row.get("predicted_full_score_piecewise")) if row else float("nan")
            )
    return {
        "models": model_names,
        "k_vals": k_vals,
        "selected_score_by_k": selected_score_by_k,
        "actual_score_by_k": actual_score_by_k,
        "gap_by_model": gap_by_model,
        "pred_by_model": pred_by_model,
    }


def _plot_selector_triplet(
    selector_records: dict[str, list[dict[str, Any]]],
    out_pdf: Path,
    warmups: list[int],
) -> None:
    selector_order = ["artifacts_selection", "corr_selection"]
    selector_titles = {
        "artifacts_selection": "Artifacts Selection",
        "corr_selection": "Correlation Selection",
    }
    available = [s for s in selector_order if s in selector_records and selector_records[s]]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 5, figsize=(28.0, 4.4 * len(available)), squeeze=False)
    if not warmups:
        warmups = [0, 2, 6, 10]
    warmups = sorted(set(warmups))
    if warmups[0] != 0:
        warmups = [0] + warmups

    for row_idx, selector_name in enumerate(available):
        recs = selector_records[selector_name]
        m = _selector_metrics_by_k(recs)
        model_names = m["models"]
        k_vals = m["k_vals"]
        selected_scores = m["selected_score_by_k"]
        actual_scores = m["actual_score_by_k"]
        gap_by_model = m["gap_by_model"]
        pred_by_model = m["pred_by_model"]

        warm = _warmup_summary(recs, warmups)
        mean_by_method = {
            str(r["method"]): _safe_float(r["mean_abs_diff"])
            for r in warm
            if int(_safe_float(r.get("warmup_kmin"), default=-1)) == 0
        }
        best_model = min(
            (
                name
                for name in model_names
                if not math.isnan(_safe_float(mean_by_method.get(name)))
            ),
            key=lambda name: _safe_float(mean_by_method.get(name)),
            default=model_names[0] if model_names else "",
        )

        selected_gap_by_k: list[float] = []
        for idx in range(len(k_vals)):
            s = selected_scores[idx] if idx < len(selected_scores) else float("nan")
            a = actual_scores[idx] if idx < len(actual_scores) else float("nan")
            if math.isnan(s) or math.isnan(a):
                selected_gap = float("nan")
            else:
                selected_gap = abs(s - a)
            selected_gap_by_k.append(selected_gap)

        ax_traj = axes[row_idx, 0]
        ax_traj.plot(k_vals, selected_scores, marker="o", linewidth=1.8, color="black", label="selected")
        ax_traj.plot(k_vals, actual_scores, marker="o", linewidth=1.8, color="tab:red", label="all-circuits")
        if best_model:
            ax_traj.plot(
                k_vals,
                pred_by_model.get(best_model, []),
                marker="o",
                linewidth=1.6,
                linestyle="--",
                color="tab:blue",
                label=f"pred:{best_model}",
            )
        ax_traj.set_title(f"{selector_titles.get(selector_name, selector_name)}: Score Trajectory")
        ax_traj.set_xlabel("k")
        ax_traj.set_ylabel("Piecewise score")
        ax_traj.grid(alpha=0.2)
        ax_traj.legend(frameon=False, fontsize=8)

        ax_gap = axes[row_idx, 1]
        for name in model_names:
            ax_gap.plot(k_vals, gap_by_model.get(name, []), marker="o", linewidth=1.4, linestyle="--", label=name)
        ax_gap.set_title(f"{selector_titles.get(selector_name, selector_name)}: |pred-all|")
        ax_gap.set_xlabel("k")
        ax_gap.set_ylabel("Absolute diff")
        ax_gap.grid(alpha=0.2)
        ax_gap.legend(frameon=False, fontsize=8, ncol=2)

        ax_corr = axes[row_idx, 2]
        corr_labels = ["baseline"] + model_names
        corr_vals = [_pearson_corr(actual_scores, selected_scores)]
        corr_vals.extend(_pearson_corr(actual_scores, pred_by_model.get(name, [])) for name in model_names)
        corr_plot_vals = [0.0 if math.isnan(v) else v for v in corr_vals]
        bars = ax_corr.bar(
            corr_labels,
            corr_plot_vals,
            color=["tab:gray"] + ["tab:blue"] * len(model_names),
            alpha=0.9,
        )
        for b, v in zip(bars, corr_vals):
            ax_corr.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height() + (0.02 if b.get_height() >= 0 else -0.06),
                "nan" if math.isnan(v) else f"{v:.2f}",
                ha="center",
                va="bottom" if b.get_height() >= 0 else "top",
                fontsize=7,
            )
        ax_corr.set_title(f"{selector_titles.get(selector_name, selector_name)}: Corr")
        ax_corr.set_ylabel("Pearson r")
        ax_corr.set_ylim(-1.05, 1.05)
        ax_corr.grid(axis="y", alpha=0.2)
        ax_corr.tick_params(axis="x", labelrotation=25, labelsize=7)

        ax_stats = axes[row_idx, 3]
        stat_labels = ["baseline"] + model_names
        stat_series = [selected_gap_by_k] + [gap_by_model.get(name, []) for name in model_names]
        stat_means: list[float] = []
        stat_stds: list[float] = []
        for vals in stat_series:
            arr = np.array(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                stat_means.append(float("nan"))
                stat_stds.append(float("nan"))
            else:
                stat_means.append(float(np.mean(arr)))
                stat_stds.append(float(np.std(arr)))
        bar_x = np.arange(len(stat_labels))
        ax_stats.bar(
            bar_x,
            [0.0 if math.isnan(v) else v for v in stat_means],
            yerr=[0.0 if math.isnan(v) else v for v in stat_stds],
            capsize=3,
            color=["tab:gray"] + ["tab:blue"] * len(model_names),
            alpha=0.9,
        )
        ax_stats.set_xticks(bar_x)
        ax_stats.set_xticklabels(stat_labels, rotation=25, fontsize=7)
        ax_stats.set_title(f"{selector_titles.get(selector_name, selector_name)}: Mean±Std")
        ax_stats.set_ylabel("|method-all|")
        ax_stats.grid(axis="y", alpha=0.2)

        ax_warm = axes[row_idx, 4]
        for method in ["baseline_selected"] + model_names:
            ys = [
                _safe_float(
                    next(
                        (
                            r.get("mean_abs_diff")
                            for r in warm
                            if str(r.get("method", "")) == method
                            and int(_safe_float(r.get("warmup_kmin"), default=-1)) == w
                        ),
                        float("nan"),
                    )
                )
                for w in warmups
            ]
            ax_warm.plot(
                warmups,
                ys,
                marker="o",
                linewidth=1.8 if method == "baseline_selected" else 1.4,
                label=method,
            )
        ax_warm.set_title(f"{selector_titles.get(selector_name, selector_name)}: Warmup")
        ax_warm.set_xlabel("k_min")
        ax_warm.set_ylabel("Mean |method-all|")
        ax_warm.grid(alpha=0.2)
        ax_warm.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_selector_compare(
    corr_records: list[dict[str, Any]],
    ml_records: list[dict[str, Any]],
    out_pdf: Path,
    warmups: list[int],
) -> None:
    corr_warm = _warmup_summary(corr_records, warmups)
    ml_warm = _warmup_summary(ml_records, warmups)
    methods = sorted({str(r["method"]) for r in corr_warm if int(_safe_float(r.get("warmup_kmin"), 0)) == 0})
    if "baseline_selected" in methods:
        methods = ["baseline_selected"] + [m for m in methods if m != "baseline_selected"]

    def _mean_for(rows: list[dict[str, Any]], method: str, kmin: int) -> float:
        row = next(
            (r for r in rows if str(r.get("method", "")) == method and int(_safe_float(r.get("warmup_kmin"), default=-1)) == kmin),
            None,
        )
        return _safe_float(row.get("mean_abs_diff")) if row is not None else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    ax_bar, ax_warm = axes

    x = np.arange(len(methods))
    width = 0.38
    corr_vals = [_mean_for(corr_warm, m, 0) for m in methods]
    ml_vals = [_mean_for(ml_warm, m, 0) for m in methods]
    ax_bar.bar(
        x - width / 2.0,
        [0.0 if math.isnan(v) else v for v in corr_vals],
        width=width,
        label="correlation selection",
        color="tab:blue",
        alpha=0.9,
    )
    ax_bar.bar(
        x + width / 2.0,
        [0.0 if math.isnan(v) else v for v in ml_vals],
        width=width,
        label="ml selection",
        color="tab:orange",
        alpha=0.9,
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(methods, rotation=25)
    ax_bar.set_ylabel("Mean |method-all| (k>=0)")
    ax_bar.set_title("Selector Comparison at Full Range")
    ax_bar.grid(axis="y", alpha=0.2)
    ax_bar.legend(frameon=False)

    if not warmups:
        warmups = [0, 2, 6, 10]
    warmups = sorted(set(warmups))
    if warmups[0] != 0:
        warmups = [0] + warmups

    # Show best method under each selector across warmups.
    corr_best = []
    ml_best = []
    for w in warmups:
        corr_candidates = [
            _mean_for(corr_warm, m, w) for m in methods if not math.isnan(_mean_for(corr_warm, m, w))
        ]
        ml_candidates = [
            _mean_for(ml_warm, m, w) for m in methods if not math.isnan(_mean_for(ml_warm, m, w))
        ]
        corr_best.append(min(corr_candidates) if corr_candidates else float("nan"))
        ml_best.append(min(ml_candidates) if ml_candidates else float("nan"))
    ax_warm.plot(warmups, corr_best, marker="o", linewidth=2.0, label="correlation selection", color="tab:blue")
    ax_warm.plot(warmups, ml_best, marker="o", linewidth=2.0, label="ml selection", color="tab:orange")
    ax_warm.set_title("Best Achievable Mean |method-all| vs Warmup")
    ax_warm.set_xlabel("Warmup k_min (use k >= k_min)")
    ax_warm.set_ylabel("Best mean absolute diff")
    ax_warm.grid(alpha=0.2)
    ax_warm.legend(frameon=False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rolling k-1 -> k prediction and gap-reduction plot from cached merged dataset."
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/plots/gpt5mini_psw1_all_analysis",
        help="Directory containing gen_seed323_program_merged.csv",
    )
    parser.add_argument(
        "--merged-csv",
        default="",
        help="Optional explicit merged csv path; defaults to output-dir/gen_seed323_program_merged.csv",
    )
    parser.add_argument(
        "--program-dir",
        default="",
        help=(
            "Optional program JSON directory (checkpoint_*/programs) used to reconstruct "
            "selected-score from artifacts_json.cases when selected summary is missing."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        default="artifacts",
        choices=["artifacts", "ml_rolling"],
        help=(
            "How selected circuits are chosen. "
            "'artifacts': use per-program artifacts_json.cases. "
            "'ml_rolling': choose circuits per k using only first k-1 programs (no future leakage)."
        ),
    )
    parser.add_argument(
        "--ml-selected-cases",
        type=int,
        default=9,
        help="When --selection-mode=ml_rolling, number of circuits selected per k step.",
    )
    parser.add_argument(
        "--ml-selector-variant",
        default="auto",
        choices=["auto", "corr", "ml_greedy_cv", "ml_importance"],
        help=(
            "When --selection-mode=ml_rolling: selector implementation. "
            "'auto' tries corr/ml_importance/ml_greedy_cv on train-only data and picks best CV score."
        ),
    )
    parser.add_argument(
        "--min-full-cases",
        type=int,
        default=63,
        help="Filter rows by full_cases_count >= this threshold.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include rows with failure_reason (default excludes them).",
    )
    parser.add_argument(
        "--max-programs",
        type=int,
        default=0,
        help="Optional cap after sorting; 0 means all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stochastic models.",
    )
    parser.add_argument(
        "--tag",
        default="cache_rolling",
        help="Tag for output file names.",
    )
    parser.add_argument(
        "--warmups",
        default="0,2,6,10",
        help="Comma-separated warmup thresholds k_min for warmup analysis (k >= k_min).",
    )
    parser.add_argument(
        "--compare-selectors",
        action="store_true",
        help=(
            "Only for --selection-mode=ml_rolling: compare artifact / correlation selection "
            "in one PDF with subfigs and write selector comparison CSV."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    merged_csv = Path(args.merged_csv) if args.merged_csv else None
    rows, data_source = _load_merged(output_dir, merged_csv)
    program_dir = Path(args.program_dir) if args.program_dir else None
    n_skipped_migrant_no_artifacts = 0
    rolling_selection_rows: list[dict[str, Any]] = []
    if args.selection_mode == "artifacts":
        feature_rows, n_filled, n_failed, n_skipped_migrant_no_artifacts = _build_selected_feature_rows(
            rows, output_dir, program_dir
        )
    else:
        feature_rows = list(rows)
        n_filled = len(feature_rows)
        n_failed = 0
    if not feature_rows:
        raise RuntimeError(
            "Could not build selected-only feature rows from cases. "
            "Check program-dir and gen_seed323_full63_case_metrics.csv."
        )

    filtered: list[dict[str, Any]] = []
    skipped_no_full_score = 0
    skipped_failed_rows = 0
    skipped_min_cases = 0
    for row in feature_rows:
        if math.isnan(_safe_float(row.get("full_score_piecewise"))):
            skipped_no_full_score += 1
            continue
        if not args.include_failed and str(row.get("failure_reason", "")).strip():
            skipped_failed_rows += 1
            continue
        if int(_safe_float(row.get("full_cases_count"), default=0)) < args.min_full_cases:
            skipped_min_cases += 1
            continue
        filtered.append(row)

    filtered = sorted(filtered, key=_order_key)
    if args.max_programs > 0:
        filtered = filtered[: args.max_programs]

    print(
        f"[start] source={data_source} merged_rows={len(rows)} selected_feature_rows={len(feature_rows)} "
        f"filtered_rows={len(filtered)} "
        f"(min_full_cases={args.min_full_cases}, include_failed={args.include_failed}, "
        f"selection_mode={args.selection_mode}, "
        f"ml_selector_variant={args.ml_selector_variant}, "
        f"selected_filled={n_filled}, selected_unfilled={n_failed}, "
        f"skipped_migrant_no_artifacts={n_skipped_migrant_no_artifacts})"
    )
    print(
        "[rows] "
        f"extracted={len(feature_rows)} "
        f"skipped_total={len(rows) - len(feature_rows)} "
        f"skipped_no_full_score={skipped_no_full_score} "
        f"skipped_failed={skipped_failed_rows} "
        f"skipped_min_full_cases={skipped_min_cases}"
    )
    if len(filtered) < 2:
        raise RuntimeError("Need at least 2 filtered rows for k-1 -> k evaluation.")

    compare_summary_rows: list[dict[str, Any]] = []
    compare_csv: Path | None = None
    out_pdf = output_dir / f"iterative_gap_reduction_{args.tag}.pdf"
    warmups = _parse_int_list(args.warmups)
    used_selector_compare_figure = False
    if args.selection_mode == "ml_rolling":
        recs_main, rolling_selection_rows = _iterative_eval_ml_rolling(
            filtered,
            output_dir=output_dir,
            top_k=args.ml_selected_cases,
            random_state=args.seed,
            selector_mode=args.ml_selector_variant,
        )
        recs = recs_main
        if args.compare_selectors:
            recs_corr, _ = _iterative_eval_ml_rolling(
                filtered,
                output_dir=output_dir,
                top_k=args.ml_selected_cases,
                random_state=args.seed,
                selector_mode="corr",
            )
            warmups = _parse_int_list(args.warmups)
            compare_records: dict[str, list[dict[str, Any]]] = {
                "corr_selection": recs_corr,
            }
            # Artifacts selection can be unavailable if program artifacts are missing.
            artifacts_feature_rows, _, _, _ = _build_selected_feature_rows(rows, output_dir, program_dir)
            artifacts_filtered = []
            for row in sorted(artifacts_feature_rows, key=_order_key):
                if math.isnan(_safe_float(row.get("full_score_piecewise"))):
                    continue
                if not args.include_failed and str(row.get("failure_reason", "")).strip():
                    continue
                if int(_safe_float(row.get("full_cases_count"), default=0)) < args.min_full_cases:
                    continue
                artifacts_filtered.append(row)
            if len(artifacts_filtered) >= 2:
                recs_artifacts = _iterative_eval(artifacts_filtered, random_state=args.seed)
                compare_records["artifacts_selection"] = recs_artifacts

            compare_summary_rows = _selector_compare_summary(compare_records, warmups)
            compare_csv = output_dir / f"iterative_gap_reduction_{args.tag}_selector_compare.csv"
            _plot_selector_triplet(compare_records, out_pdf, warmups)
            used_selector_compare_figure = True
            _write_csv(
                compare_csv,
                compare_summary_rows,
                ["selector", "warmup_kmin", "method", "mean_abs_diff", "std_abs_diff", "count"],
            )
    else:
        recs = _iterative_eval(filtered, random_state=args.seed)
    out_csv = output_dir / f"iterative_gap_reduction_{args.tag}.csv"
    _write_csv(
        out_csv,
        recs,
        [
            "k_index",
            "train_size",
            "program_id",
            "generation",
            "iteration_found",
            "model",
            "actual_full_score_piecewise",
            "selected_score_piecewise",
            "selected_gap_abs",
            "predicted_full_score_piecewise",
            "model_gap_abs",
            "gap_reduction_abs",
            "selector_mode",
            "error",
        ],
    )
    if not used_selector_compare_figure:
        _plot(recs, out_pdf, warmups)
    warmup_rows = _warmup_summary(recs, warmups)
    out_warmup_csv = output_dir / f"iterative_gap_reduction_{args.tag}_warmup_summary.csv"
    _write_csv(
        out_warmup_csv,
        warmup_rows,
        ["warmup_kmin", "method", "mean_abs_diff", "std_abs_diff", "count"],
    )

    model_names = sorted({str(r["model"]) for r in recs})
    prediction_targets = len(filtered) - 1
    print(
        "[predictions] "
        f"targets={prediction_targets} "
        f"models={len(model_names)} "
        f"prediction_rows={len(recs)}"
    )
    print(f"[done] records={len(recs)} steps={len(filtered)-1} models={len(model_names)}")
    print(f"[done] wrote csv: {out_csv}")
    if args.selection_mode == "ml_rolling":
        unique_k = sorted({int(_safe_float(r.get("k_index"), default=0)) for r in rolling_selection_rows})
        if unique_k:
            print(
                f"[selection] mode=ml_rolling variant={args.ml_selector_variant} top_k={args.ml_selected_cases} "
                f"steps_with_selection={len(unique_k)}"
            )
        if args.compare_selectors and compare_csv is not None:
            selectors_present = sorted({str(r.get("selector", "")) for r in compare_summary_rows if r.get("selector")})
            print(f"[compare] selectors={','.join(selectors_present)}")
            print(f"[done] wrote selector-compare csv: {compare_csv}")
    print(f"[done] wrote warmup csv: {out_warmup_csv}")
    for w in sorted({int(r["warmup_kmin"]) for r in warmup_rows}):
        subset = [r for r in warmup_rows if int(r["warmup_kmin"]) == w]
        best = min(
            (r for r in subset if not math.isnan(_safe_float(r.get("mean_abs_diff")))),
            key=lambda r: _safe_float(r.get("mean_abs_diff")),
            default=None,
        )
        if best is not None:
            print(
                f"[warmup] k>={w} best={best['method']} "
                f"mean_abs_diff={_safe_float(best.get('mean_abs_diff')):.4f} "
                f"n={int(_safe_float(best.get('count'), default=0))}"
            )
    print(f"[done] wrote figure: {out_pdf}")


if __name__ == "__main__":
    main()
