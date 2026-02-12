import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qos.error_mitigator import evaluator as evaluator_module


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float(numerator)
    return float(numerator) / float(denominator)


def _combined_piecewise(depth_ratio: float, cnot_ratio: float, time_ratio: float) -> float:
    struct_delta = 1.0 - ((depth_ratio + cnot_ratio) / 2.0)
    time_delta = 1.0 - time_ratio
    struct_term = struct_delta if struct_delta >= 0 else 8.0 * struct_delta
    return struct_term + time_delta


def _parse_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _set_env(key: str, value: str | None) -> tuple[str, str | None]:
    prev = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    return key, prev


def _restore_env(items: list[tuple[str, str | None]]) -> None:
    for key, prev in reversed(items):
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _load_program_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _load_selected_pairs(program_obj: dict[str, Any]) -> list[tuple[str, int]]:
    artifacts_json = program_obj.get("artifacts_json")
    if artifacts_json is None:
        return []
    if isinstance(artifacts_json, str):
        try:
            artifacts = json.loads(artifacts_json)
        except Exception:
            return []
    elif isinstance(artifacts_json, dict):
        artifacts = artifacts_json
    else:
        return []
    cases = artifacts.get("cases", [])
    if not isinstance(cases, list):
        return []
    pairs: list[tuple[str, int]] = []
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
        pairs.append((bench, size))
    return pairs


def _selected_case_rows_from_full(
    program_id: str,
    generation: int | None,
    iteration_found: int | None,
    selected_pairs: list[tuple[str, int]],
    full_case_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    pair_to_case: dict[tuple[str, int], dict[str, Any]] = {}
    for r in full_case_rows:
        bench = str(r.get("bench", "")).strip()
        try:
            size = int(float(r.get("size", "nan")))
        except Exception:
            continue
        pair_to_case[(bench, size)] = r

    rows: list[dict[str, Any]] = []
    missing = 0
    for idx, (bench, size) in enumerate(selected_pairs):
        case = pair_to_case.get((bench, size))
        if case is None:
            missing += 1
            continue
        qose_depth = _safe_float(case.get("qose_depth"))
        qos_depth = _safe_float(case.get("qos_depth"))
        qose_cnot = _safe_float(case.get("qose_cnot"))
        qos_cnot = _safe_float(case.get("qos_cnot"))
        qose_run_sec = _safe_float(case.get("qose_run_sec"))
        qos_run_sec = _safe_float(case.get("qos_run_sec"))
        qose_num_circuits = _safe_float(case.get("qose_num_circuits"))
        qos_num_circuits = _safe_float(case.get("qos_num_circuits"))
        depth_ratio = _safe_ratio(qose_depth, qos_depth)
        cnot_ratio = _safe_ratio(qose_cnot, qos_cnot)
        time_ratio = _safe_ratio(qose_run_sec, qos_run_sec)
        overhead_ratio = _safe_ratio(qose_num_circuits, qos_num_circuits)
        row: dict[str, Any] = {
            "program_id": program_id,
            "generation": generation,
            "iteration_found": iteration_found,
            "selected_index": idx,
            "bench": bench,
            "size": size,
            "qose_depth": qose_depth,
            "qos_depth": qos_depth,
            "qose_cnot": qose_cnot,
            "qos_cnot": qos_cnot,
            "qose_run_sec": qose_run_sec,
            "qos_run_sec": qos_run_sec,
            "qose_num_circuits": qose_num_circuits,
            "qos_num_circuits": qos_num_circuits,
            "depth_ratio": depth_ratio,
            "cnot_ratio": cnot_ratio,
            "time_ratio": time_ratio,
            "overhead_ratio": overhead_ratio,
            "selected_case_piecewise_score": _combined_piecewise(
                depth_ratio, cnot_ratio, time_ratio
            ),
            "qose_output_size": case.get("qose_output_size", ""),
            "qose_method": case.get("qose_method", ""),
        }
        input_features = case.get("input_features", {})
        if isinstance(input_features, dict):
            for k, v in input_features.items():
                row[f"feat_{k}"] = _safe_float(v, default=float("nan"))
        rows.append(row)
    return rows, missing


def _full_case_rows(
    program_id: str,
    generation: int | None,
    iteration_found: int | None,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        qose_depth = _safe_float(case.get("qose_depth"))
        qos_depth = _safe_float(case.get("qos_depth"))
        qose_cnot = _safe_float(case.get("qose_cnot"))
        qos_cnot = _safe_float(case.get("qos_cnot"))
        qose_run_sec = _safe_float(case.get("qose_run_sec"))
        qos_run_sec = _safe_float(case.get("qos_run_sec"))
        qose_num_circuits = _safe_float(case.get("qose_num_circuits"))
        qos_num_circuits = _safe_float(case.get("qos_num_circuits"))
        depth_ratio = _safe_ratio(qose_depth, qos_depth)
        cnot_ratio = _safe_ratio(qose_cnot, qos_cnot)
        time_ratio = _safe_ratio(qose_run_sec, qos_run_sec)
        overhead_ratio = _safe_ratio(qose_num_circuits, qos_num_circuits)
        row: dict[str, Any] = {
            "program_id": program_id,
            "generation": generation,
            "iteration_found": iteration_found,
            "case_index": idx,
            "bench": case.get("bench", ""),
            "size": case.get("size", ""),
            "qose_depth": qose_depth,
            "qos_depth": qos_depth,
            "qose_cnot": qose_cnot,
            "qos_cnot": qos_cnot,
            "qose_run_sec": qose_run_sec,
            "qos_run_sec": qos_run_sec,
            "qose_num_circuits": qose_num_circuits,
            "qos_num_circuits": qos_num_circuits,
            "depth_ratio": depth_ratio,
            "cnot_ratio": cnot_ratio,
            "time_ratio": time_ratio,
            "overhead_ratio": overhead_ratio,
            "full_case_piecewise_score": _combined_piecewise(depth_ratio, cnot_ratio, time_ratio),
            "qose_output_size": case.get("qose_output_size", ""),
            "qose_method": case.get("qose_method", ""),
        }
        input_features = case.get("input_features", {})
        if isinstance(input_features, dict):
            for k, v in input_features.items():
                row[f"feat_{k}"] = _safe_float(v, default=float("nan"))
        rows.append(row)
    return rows


def _row_matches_pair(row: dict[str, Any], bench: str, size: int) -> bool:
    if str(row.get("bench", "")).strip() != bench:
        return False
    try:
        row_size = int(float(row.get("size", "nan")))
    except Exception:
        return False
    return row_size == size


def _aggregate_selected_summary(program_id: str, case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"program_id": program_id}
    if not case_rows:
        out.update(
            {
                "selected_case_count": 0,
                "selected_depth_ratio_mean": float("nan"),
                "selected_cnot_ratio_mean": float("nan"),
                "selected_time_ratio_mean": float("nan"),
                "selected_overhead_ratio_mean": float("nan"),
                "selected_score_piecewise": float("nan"),
                "selected_size_mean": float("nan"),
                "selected_size_std": float("nan"),
                "selected_wc_fraction": float("nan"),
            }
        )
        return out

    def _mean(col: str) -> float:
        vals = [float(r[col]) for r in case_rows if not math.isnan(_safe_float(r.get(col)))]
        return float(np.mean(vals)) if vals else float("nan")

    def _std(col: str) -> float:
        vals = [float(r[col]) for r in case_rows if not math.isnan(_safe_float(r.get(col)))]
        return float(np.std(vals)) if vals else float("nan")

    depth_m = _mean("depth_ratio")
    cnot_m = _mean("cnot_ratio")
    time_m = _mean("time_ratio")
    out.update(
        {
            "selected_case_count": len(case_rows),
            "selected_depth_ratio_mean": depth_m,
            "selected_cnot_ratio_mean": cnot_m,
            "selected_time_ratio_mean": time_m,
            "selected_overhead_ratio_mean": _mean("overhead_ratio"),
            "selected_score_piecewise": _combined_piecewise(depth_m, cnot_m, time_m),
            "selected_size_mean": _mean("size"),
            "selected_size_std": _std("size"),
            "selected_wc_fraction": float(
                np.mean([1.0 if str(r.get("qose_method", "")) == "WC" else 0.0 for r in case_rows])
            ),
        }
    )

    feature_cols = sorted({k for r in case_rows for k in r if k.startswith("feat_")})
    for col in feature_cols:
        out[f"{col}_mean"] = _mean(col)
        out[f"{col}_std"] = _std(col)
    return out


def _evaluate_program_full63(
    code: str,
    benches: list[str],
    size_min: int,
    size_max: int,
    samples_per_bench: int,
    sample_seed: int,
    eval_timeout_sec: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(code)
        temp_program = Path(tf.name)
    env_edits = [
        _set_env("QOSE_SCORE_MODE", "piecewise"),
        _set_env("QOSE_SIZE_MIN", str(size_min)),
        _set_env("QOSE_SIZE_MAX", str(size_max)),
        _set_env("QOSE_STRATIFIED_SIZES", "1"),
        _set_env("QOSE_SAMPLES_PER_BENCH", str(samples_per_bench)),
        _set_env("QOSE_DISTINCT_SIZES_PER_BENCH", "1"),
        _set_env("QOSE_SAMPLE_SEED", str(sample_seed)),
        _set_env("QOSE_BENCHES", ",".join(benches)),
        _set_env("QOSE_EVAL_TIMEOUT_SEC", str(eval_timeout_sec)),
    ]
    try:
        result = evaluator_module.evaluate(str(temp_program))
        return dict(result.metrics), dict(result.artifacts)
    finally:
        _restore_env(env_edits)
        try:
            temp_program.unlink(missing_ok=True)
        except Exception:
            pass


def _scatter_selected_vs_full(rows: list[dict[str, Any]], out_pdf: Path) -> None:
    xs = np.array([_safe_float(r.get("selected_score_piecewise")) for r in rows], dtype=float)
    ys = np.array([_safe_float(r.get("full_score_piecewise")) for r in rows], dtype=float)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs = xs[mask]
    ys = ys[mask]
    if xs.size == 0:
        print("[warn] no valid selected/full scores for scatter plot")
        return
    plt.figure(figsize=(7.2, 6.0))
    plt.scatter(xs, ys, alpha=0.75, s=28, edgecolors="none")
    lo = min(float(xs.min()), float(ys.min()))
    hi = max(float(xs.max()), float(ys.max()))
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1.2, color="gray", label="y=x")
    if xs.size >= 2:
        coeffs = np.polyfit(xs, ys, 1)
        fit_x = np.linspace(lo, hi, 100)
        fit_y = coeffs[0] * fit_x + coeffs[1]
        plt.plot(fit_x, fit_y, linewidth=1.4, color="tab:red", label="linear fit")
    plt.xlabel("Selected-subset score (piecewise)")
    plt.ylabel("All-63 score (piecewise)")
    plt.title("Program score: selected subset vs all 63 circuits")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf)
    plt.close()
    print(f"[done] wrote figure: {out_pdf}")


def _build_model_pipeline(estimator: Any, use_scaler: bool) -> Pipeline:
    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        num_steps.append(("scaler", StandardScaler()))
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), slice(0, None)),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("pre", pre), ("model", estimator)])


def _train_predict_models(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        print("[warn] no rows for ML model")
        return
    target_col = "full_score_piecewise"
    all_cols = set()
    for r in rows:
        all_cols.update(r.keys())
    excluded = {
        "program_id",
        "full_cases_count",
        "full_score_mode",
        "full_score_piecewise",
        "full_qose_depth",
        "full_qose_cnot",
        "full_avg_run_time",
        "full_combined_score_raw",
        "failure_reason",
    }
    feature_cols = sorted(c for c in all_cols if c not in excluded)
    valid_rows: list[dict[str, Any]] = []
    for r in rows:
        y = _safe_float(r.get(target_col))
        if math.isnan(y):
            continue
        valid_rows.append(r)
    if len(valid_rows) < 12:
        print(f"[warn] too few valid rows for ML model: {len(valid_rows)}")
        return

    X = np.array([[_safe_float(r.get(c)) for c in feature_cols] for r in valid_rows], dtype=float)
    y = np.array([_safe_float(r.get(target_col)) for r in valid_rows], dtype=float)
    dataset_csv = output_dir / "gen_seed323_ml_dataset.csv"
    dataset_cols = sorted({k for r in valid_rows for k in r.keys()})
    _write_csv(dataset_csv, valid_rows, dataset_cols)
    feature_json = output_dir / "gen_seed323_ml_feature_columns.json"
    with feature_json.open("w") as f:
        json.dump({"target": target_col, "features": feature_cols}, f, indent=2)
    print(f"[done] wrote ML dataset: {dataset_csv}")
    print(f"[done] wrote ML feature config: {feature_json}")

    X_train, X_test, y_train, y_test, train_rows, test_rows = train_test_split(
        X,
        y,
        valid_rows,
        test_size=0.2,
        random_state=42,
    )

    baseline_pred = np.array(
        [_safe_float(r.get("selected_score_piecewise")) for r in test_rows], dtype=float
    )
    baseline_mask = ~np.isnan(baseline_pred)

    models: list[tuple[str, Pipeline]] = [
        ("ridge", _build_model_pipeline(Ridge(alpha=1.0, random_state=42), use_scaler=True)),
        ("lasso", _build_model_pipeline(Lasso(alpha=0.005, max_iter=20000, random_state=42), use_scaler=True)),
        ("elasticnet", _build_model_pipeline(ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=20000, random_state=42), use_scaler=True)),
        ("random_forest", _build_model_pipeline(RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1), use_scaler=False)),
        ("extra_trees", _build_model_pipeline(ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1), use_scaler=False)),
        ("gradient_boosting", _build_model_pipeline(GradientBoostingRegressor(random_state=42), use_scaler=False)),
    ]

    cv = KFold(n_splits=min(5, len(y_train)), shuffle=True, random_state=42)
    summary_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    best_name = ""
    best_r2 = -1e18
    best_preds: np.ndarray | None = None

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        holdout_r2 = float(r2_score(y_test, y_pred))
        holdout_mae = float(mean_absolute_error(y_test, y_pred))
        holdout_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)
        cv_mae = -cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1
        )
        row = {
            "model": name,
            "n_total": int(len(valid_rows)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "holdout_r2": holdout_r2,
            "holdout_mae": holdout_mae,
            "holdout_rmse": holdout_rmse,
            "cv_r2_mean": float(np.mean(cv_r2)),
            "cv_r2_std": float(np.std(cv_r2)),
            "cv_mae_mean": float(np.mean(cv_mae)),
            "cv_mae_std": float(np.std(cv_mae)),
        }
        summary_rows.append(row)
        for r, yt, yp in zip(test_rows, y_test, y_pred):
            pred_rows.append(
                {
                    "model": name,
                    "program_id": r.get("program_id", ""),
                    "actual_full_score_piecewise": float(yt),
                    "predicted_full_score_piecewise": float(yp),
                    "selected_score_piecewise": _safe_float(r.get("selected_score_piecewise")),
                }
            )
        if holdout_r2 > best_r2:
            best_r2 = holdout_r2
            best_name = name
            best_preds = y_pred.copy()

    baseline_row = {
        "model": "baseline_selected_score",
        "n_total": int(len(valid_rows)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "holdout_r2": float("nan"),
        "holdout_mae": float("nan"),
        "holdout_rmse": float("nan"),
        "cv_r2_mean": float("nan"),
        "cv_r2_std": float("nan"),
        "cv_mae_mean": float("nan"),
        "cv_mae_std": float("nan"),
    }
    if baseline_mask.any():
        baseline_row["holdout_r2"] = float(r2_score(y_test[baseline_mask], baseline_pred[baseline_mask]))
        baseline_row["holdout_mae"] = float(mean_absolute_error(y_test[baseline_mask], baseline_pred[baseline_mask]))
        baseline_row["holdout_rmse"] = float(
            np.sqrt(mean_squared_error(y_test[baseline_mask], baseline_pred[baseline_mask]))
        )
    summary_rows.append(baseline_row)

    summary_csv = output_dir / "gen_seed323_model_comparison.csv"
    _write_csv(
        summary_csv,
        summary_rows,
        [
            "model",
            "n_total",
            "n_train",
            "n_test",
            "holdout_r2",
            "holdout_mae",
            "holdout_rmse",
            "cv_r2_mean",
            "cv_r2_std",
            "cv_mae_mean",
            "cv_mae_std",
        ],
    )
    summary_json = output_dir / "gen_seed323_model_comparison.json"
    with summary_json.open("w") as f:
        json.dump(summary_rows, f, indent=2)
    pred_csv = output_dir / "gen_seed323_model_predictions.csv"
    _write_csv(
        pred_csv,
        pred_rows,
        [
            "model",
            "program_id",
            "actual_full_score_piecewise",
            "predicted_full_score_piecewise",
            "selected_score_piecewise",
        ],
    )
    print(f"[done] wrote model comparison: {summary_csv}")
    print(f"[done] wrote model comparison: {summary_json}")
    print(f"[done] wrote model predictions: {pred_csv}")

    if best_preds is not None:
        plt.figure(figsize=(6.8, 5.8))
        plt.scatter(y_test, best_preds, alpha=0.8, s=30, edgecolors="none", label=best_name)
        lo = min(float(np.min(y_test)), float(np.min(best_preds)))
        hi = max(float(np.max(y_test)), float(np.max(best_preds)))
        plt.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1.2, label="y=x")
        plt.xlabel("Actual full score (piecewise)")
        plt.ylabel("Predicted full score")
        plt.title(f"Best model on holdout: {best_name}")
        plt.legend(frameon=False)
        plt.tight_layout()
        fig_path = output_dir / "gen_seed323_model_pred_vs_actual.pdf"
        plt.savefig(fig_path)
        plt.close()
        print(f"[done] wrote figure: {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze generated programs: recompute full-63 piecewise score, "
            "compare vs selected-subset score, and train a lightweight predictor."
        )
    )
    parser.add_argument(
        "--program-dir",
        default="openevolve_output/gpt5mini_gen_seed323/checkpoints/checkpoint_100/programs",
        help="Directory of program JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/plots/gen_seed323_analysis",
        help="Output directory.",
    )
    parser.add_argument(
        "--size-min",
        type=int,
        default=12,
        help="Minimum size for full-eval recomputation.",
    )
    parser.add_argument(
        "--size-max",
        type=int,
        default=24,
        help="Maximum size for full-eval recomputation.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=323,
        help="Sampling seed for evaluator pair selection.",
    )
    parser.add_argument(
        "--samples-per-bench",
        type=int,
        default=0,
        help=(
            "Samples per bench for full eval. "
            "Default 0 => auto use all parity sizes in [size-min,size-max] "
            "(e.g., 7 for 12..24)."
        ),
    )
    parser.add_argument(
        "--eval-timeout-sec",
        type=int,
        default=0,
        help="Evaluator timeout per program in seconds (0 disables timeout process wrapper).",
    )
    parser.add_argument(
        "--max-programs",
        type=int,
        default=0,
        help="Optional max number of programs to process (0 means all).",
    )
    parser.add_argument(
        "--benches",
        default="",
        help="Optional comma-separated benches; default uses evaluator BENCHES.",
    )
    parser.add_argument(
        "--no-reuse-full-cache",
        action="store_true",
        help="Recompute full scores even if cached csv already has rows.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "Write incremental full-score/full-case checkpoints every N processed programs. "
            "Set 0 to disable incremental checkpointing."
        ),
    )
    args = parser.parse_args()

    program_dir = Path(args.program_dir)
    if not program_dir.exists():
        raise FileNotFoundError(f"Program directory not found: {program_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    program_paths = sorted(program_dir.glob("*.json"))
    if args.max_programs > 0:
        program_paths = program_paths[: args.max_programs]
    if not program_paths:
        raise RuntimeError(f"No program JSON files in: {program_dir}")

    if args.benches.strip():
        benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    else:
        benches = [b for b, _ in evaluator_module.BENCHES]
    print(f"[start] programs={len(program_paths)} benches={len(benches)}")
    print(f"[start] full-eval sizes={args.size_min}..{args.size_max} piecewise mode")
    print(
        "[start] ignoring checkpoint JSON metrics (including obsolete combined_score); "
        "full score is recomputed via evaluator in piecewise mode or loaded from this script's cache."
    )
    if args.samples_per_bench > 0:
        samples_per_bench = args.samples_per_bench
    else:
        samples_per_bench = max(1, len(list(range(args.size_min, args.size_max + 1, 2))))
    print(f"[start] samples_per_bench={samples_per_bench} distinct_sizes_per_bench=1")
    expected_case_count = len(benches) * samples_per_bench
    print(f"[start] expected_full_cases_per_program={expected_case_count}")

    full_csv = output_dir / "gen_seed323_full63_program_scores.csv"
    full_cases_csv = output_dir / "gen_seed323_full63_case_metrics.csv"
    selected_pairs_csv = output_dir / "gen_seed323_selected_pairs.csv"
    selected_cases_csv = output_dir / "gen_seed323_selected_case_features.csv"
    selected_summary_csv = output_dir / "gen_seed323_selected_summary.csv"

    full_cached: dict[str, dict[str, Any]] = {}
    if full_csv.exists() and not args.no_reuse_full_cache:
        for r in _parse_csv_rows(full_csv):
            pid = r.get("program_id", "")
            if not pid:
                continue
            full_cached[pid] = {
                "program_id": pid,
                "full_score_piecewise": _safe_float(r.get("full_score_piecewise")),
                "full_qose_depth": _safe_float(r.get("full_qose_depth")),
                "full_qose_cnot": _safe_float(r.get("full_qose_cnot")),
                "full_avg_run_time": _safe_float(r.get("full_avg_run_time")),
                "full_combined_score_raw": _safe_float(r.get("full_combined_score_raw")),
                "full_cases_count": int(float(r.get("full_cases_count", 0) or 0)),
                "full_score_mode": r.get("full_score_mode", "piecewise"),
                "failure_reason": r.get("failure_reason", ""),
            }
        print(f"[start] loaded full-score cache rows={len(full_cached)} from {full_csv}")

    full_cases_cached_by_program: dict[str, list[dict[str, Any]]] = {}
    if full_cases_csv.exists() and not args.no_reuse_full_cache:
        for r in _parse_csv_rows(full_cases_csv):
            pid = r.get("program_id", "")
            if not pid:
                continue
            full_cases_cached_by_program.setdefault(pid, []).append(r)
        print(
            f"[start] loaded full-case cache rows={sum(len(v) for v in full_cases_cached_by_program.values())} "
            f"from {full_cases_csv}"
        )

    all_full_rows: list[dict[str, Any]] = []
    all_full_case_rows: list[dict[str, Any]] = []
    all_selected_pair_rows: list[dict[str, Any]] = []
    all_selected_case_rows: list[dict[str, Any]] = []
    all_selected_summary_rows: list[dict[str, Any]] = []
    reused_count = 0
    computed_count = 0
    skipped_missing_code_count = 0
    skipped_missing_file_count = 0

    full_fields = [
        "program_id",
        "generation",
        "iteration_found",
        "full_score_piecewise",
        "full_qose_depth",
        "full_qose_cnot",
        "full_avg_run_time",
        "full_combined_score_raw",
        "full_cases_count",
        "full_score_mode",
        "failure_reason",
    ]

    def _checkpoint_raw_tables(processed_idx: int) -> None:
        _write_csv(full_csv, all_full_rows, full_fields)
        full_case_feature_cols = sorted({k for r in all_full_case_rows for k in r if k.startswith("feat_")})
        full_case_fields = [
            "program_id",
            "generation",
            "iteration_found",
            "case_index",
            "bench",
            "size",
            "qose_depth",
            "qos_depth",
            "qose_cnot",
            "qos_cnot",
            "qose_run_sec",
            "qos_run_sec",
            "qose_num_circuits",
            "qos_num_circuits",
            "depth_ratio",
            "cnot_ratio",
            "time_ratio",
            "overhead_ratio",
            "full_case_piecewise_score",
            "qose_output_size",
            "qose_method",
        ] + full_case_feature_cols
        _write_csv(full_cases_csv, all_full_case_rows, full_case_fields)
        print(
            f"[checkpoint] processed={processed_idx} full_rows={len(all_full_rows)} "
            f"full_case_rows={len(all_full_case_rows)}",
            flush=True,
        )

    for idx, path in enumerate(program_paths, start=1):
        try:
            program_obj = _load_program_json(path)
        except FileNotFoundError:
            skipped_missing_file_count += 1
            print(f"[warn] {idx}/{len(program_paths)} missing file, skipped: {path}", flush=True)
            continue
        except Exception as exc:
            skipped_missing_file_count += 1
            print(f"[warn] {idx}/{len(program_paths)} failed to read {path}: {exc}", flush=True)
            continue
        program_id = str(program_obj.get("id", path.stem))
        generation = program_obj.get("generation")
        iteration_found = program_obj.get("iteration_found")
        # Explicitly ignore precomputed metrics in checkpoint JSON; they may be stale.
        _ = program_obj.get("metrics")
        code = str(program_obj.get("code", ""))
        selected_pairs = _load_selected_pairs(program_obj)

        cached_case_rows = full_cases_cached_by_program.get(program_id, [])
        has_complete_case_cache = len(cached_case_rows) >= expected_case_count
        if (
            program_id in full_cached
            and not args.no_reuse_full_cache
            and has_complete_case_cache
        ):
            full_row = dict(full_cached[program_id])
            full_row["generation"] = generation
            full_row["iteration_found"] = iteration_found
            all_full_rows.append(full_row)
            all_full_case_rows.extend(cached_case_rows)
            selected_case_rows, selected_missing = _selected_case_rows_from_full(
                program_id=program_id,
                generation=generation,
                iteration_found=iteration_found,
                selected_pairs=selected_pairs,
                full_case_rows=cached_case_rows,
            )
            selected_summary = _aggregate_selected_summary(program_id, selected_case_rows)
            selected_summary["generation"] = generation
            selected_summary["iteration_found"] = iteration_found
            selected_summary["selected_pair_count"] = len(selected_pairs)
            selected_summary["selected_missing_in_full63"] = selected_missing
            all_selected_case_rows.extend(selected_case_rows)
            all_selected_summary_rows.append(selected_summary)
            for i, (bench, size) in enumerate(selected_pairs):
                all_selected_pair_rows.append(
                    {
                        "program_id": program_id,
                        "generation": generation,
                        "iteration_found": iteration_found,
                        "selected_index": i,
                        "bench": bench,
                        "size": size,
                        "matched_in_full63": int(
                            any(_row_matches_pair(r, bench, size) for r in cached_case_rows)
                        ),
                    }
                )
            reused_count += 1
            print(
                f"[progress] {idx}/{len(program_paths)} program={program_id} full63=reuse "
                f"selected_pairs={len(selected_pairs)} selected_from_full={len(selected_case_rows)}"
            )
            if args.checkpoint_every > 0 and (idx % args.checkpoint_every == 0):
                _checkpoint_raw_tables(idx)
            continue
        if program_id in full_cached and not args.no_reuse_full_cache and not has_complete_case_cache:
            print(
                f"[progress] {idx}/{len(program_paths)} program={program_id} "
                f"full63=cache-incomplete(cases={len(cached_case_rows)}/{expected_case_count}) -> recompute"
            )

        if not code.strip():
            all_full_rows.append(
                {
                    "program_id": program_id,
                    "generation": generation,
                    "iteration_found": iteration_found,
                    "full_score_piecewise": float("nan"),
                    "full_qose_depth": float("nan"),
                    "full_qose_cnot": float("nan"),
                    "full_avg_run_time": float("nan"),
                    "full_combined_score_raw": float("nan"),
                    "full_cases_count": 0,
                    "full_score_mode": "piecewise",
                    "failure_reason": "missing code in program json",
                }
            )
            selected_summary = _aggregate_selected_summary(program_id, [])
            selected_summary["generation"] = generation
            selected_summary["iteration_found"] = iteration_found
            selected_summary["selected_pair_count"] = len(selected_pairs)
            selected_summary["selected_missing_in_full63"] = len(selected_pairs)
            all_selected_summary_rows.append(selected_summary)
            for i, (bench, size) in enumerate(selected_pairs):
                all_selected_pair_rows.append(
                    {
                        "program_id": program_id,
                        "generation": generation,
                        "iteration_found": iteration_found,
                        "selected_index": i,
                        "bench": bench,
                        "size": size,
                        "matched_in_full63": 0,
                    }
                )
            skipped_missing_code_count += 1
            print(f"[progress] {idx}/{len(program_paths)} program={program_id} skipped (missing code)")
            if args.checkpoint_every > 0 and (idx % args.checkpoint_every == 0):
                _checkpoint_raw_tables(idx)
            continue

        print(f"[progress] {idx}/{len(program_paths)} program={program_id} full63=compute ...")
        metrics, artifacts = _evaluate_program_full63(
            code=code,
            benches=benches,
            size_min=args.size_min,
            size_max=args.size_max,
            samples_per_bench=samples_per_bench,
            sample_seed=args.sample_seed,
            eval_timeout_sec=args.eval_timeout_sec,
        )
        full_row = {
            "program_id": program_id,
            "generation": generation,
            "iteration_found": iteration_found,
            "full_score_piecewise": _safe_float(metrics.get("combined_score")),
            "full_qose_depth": _safe_float(metrics.get("qose_depth")),
            "full_qose_cnot": _safe_float(metrics.get("qose_cnot")),
            "full_avg_run_time": _safe_float(metrics.get("avg_run_time")),
            "full_combined_score_raw": _safe_float(metrics.get("combined_score")),
            "full_cases_count": len(artifacts.get("cases", []))
            if isinstance(artifacts.get("cases"), list)
            else 0,
            "full_score_mode": "piecewise",
            "failure_reason": metrics.get("failure_reason", ""),
        }
        all_full_rows.append(full_row)
        full_cases = artifacts.get("cases", []) if isinstance(artifacts, dict) else []
        if isinstance(full_cases, list) and full_cases:
            program_full_case_rows = _full_case_rows(program_id, generation, iteration_found, full_cases)
            all_full_case_rows.extend(program_full_case_rows)
        else:
            program_full_case_rows = []
        selected_case_rows, selected_missing = _selected_case_rows_from_full(
            program_id=program_id,
            generation=generation,
            iteration_found=iteration_found,
            selected_pairs=selected_pairs,
            full_case_rows=program_full_case_rows,
        )
        selected_summary = _aggregate_selected_summary(program_id, selected_case_rows)
        selected_summary["generation"] = generation
        selected_summary["iteration_found"] = iteration_found
        selected_summary["selected_pair_count"] = len(selected_pairs)
        selected_summary["selected_missing_in_full63"] = selected_missing
        all_selected_case_rows.extend(selected_case_rows)
        all_selected_summary_rows.append(selected_summary)
        for i, (bench, size) in enumerate(selected_pairs):
            all_selected_pair_rows.append(
                {
                    "program_id": program_id,
                    "generation": generation,
                    "iteration_found": iteration_found,
                    "selected_index": i,
                    "bench": bench,
                    "size": size,
                    "matched_in_full63": int(
                        any(_row_matches_pair(r, bench, size) for r in program_full_case_rows)
                    ),
                }
            )
        computed_count += 1
        print(
            f"[progress] {idx}/{len(program_paths)} program={program_id} "
            f"score={full_row['full_score_piecewise']:.4f} cases={full_row['full_cases_count']}"
        )
        if args.checkpoint_every > 0 and (idx % args.checkpoint_every == 0):
            _checkpoint_raw_tables(idx)

    # Persist raw tables.
    _checkpoint_raw_tables(len(program_paths))
    print(f"[done] wrote full scores: {full_csv}")
    print(f"[done] wrote full case metrics: {full_cases_csv}")

    _write_csv(
        selected_pairs_csv,
        all_selected_pair_rows,
        [
            "program_id",
            "generation",
            "iteration_found",
            "selected_index",
            "bench",
            "size",
            "matched_in_full63",
        ],
    )
    print(f"[done] wrote selected pair ids: {selected_pairs_csv}")

    case_feature_cols = sorted({k for r in all_selected_case_rows for k in r if k.startswith("feat_")})
    selected_case_fields = [
        "program_id",
        "generation",
        "iteration_found",
        "selected_index",
        "bench",
        "size",
        "qose_depth",
        "qos_depth",
        "qose_cnot",
        "qos_cnot",
        "qose_run_sec",
        "qos_run_sec",
        "qose_num_circuits",
        "qos_num_circuits",
        "depth_ratio",
        "cnot_ratio",
        "time_ratio",
        "overhead_ratio",
        "selected_case_piecewise_score",
        "qose_output_size",
        "qose_method",
    ] + case_feature_cols
    _write_csv(selected_cases_csv, all_selected_case_rows, selected_case_fields)
    print(f"[done] wrote selected case features: {selected_cases_csv}")

    summary_cols = sorted({k for r in all_selected_summary_rows for k in r})
    _write_csv(selected_summary_csv, all_selected_summary_rows, summary_cols)
    print(f"[done] wrote selected summary: {selected_summary_csv}")

    # Merge for plotting/modeling.
    summary_by_id = {r["program_id"]: r for r in all_selected_summary_rows}
    merged_rows: list[dict[str, Any]] = []
    for fr in all_full_rows:
        pid = fr["program_id"]
        row = dict(fr)
        row.update(summary_by_id.get(pid, {}))
        merged_rows.append(row)
    merged_csv = output_dir / "gen_seed323_program_merged.csv"
    merged_cols = sorted({k for r in merged_rows for k in r})
    _write_csv(merged_csv, merged_rows, merged_cols)
    print(f"[done] wrote merged dataset: {merged_csv}")

    # Scatter + lightweight ML.
    _scatter_selected_vs_full(
        merged_rows,
        output_dir / "gen_seed323_selected_vs_all63_scatter.pdf",
    )
    _train_predict_models(merged_rows, output_dir)
    failed_full = sum(
        1
        for r in all_full_rows
        if str(r.get("failure_reason", "")).strip() or math.isnan(_safe_float(r.get("full_score_piecewise")))
    )
    print(
        "[summary] programs_total=%d processed=%d reused=%d computed=%d "
        "skipped_missing_code=%d skipped_missing_file=%d full_failed=%d"
        % (
            len(program_paths),
            len(all_full_rows),
            reused_count,
            computed_count,
            skipped_missing_code_count,
            skipped_missing_file_count,
            failed_full,
        )
    )
    print(
        "[summary] dataset_rows full_program=%d full_cases=%d selected_pairs=%d selected_cases=%d selected_summary=%d merged=%d"
        % (
            len(all_full_rows),
            len(all_full_case_rows),
            len(all_selected_pair_rows),
            len(all_selected_case_rows),
            len(all_selected_summary_rows),
            len(merged_rows),
        )
    )
    selected_missing_total = sum(
        int(float(r.get("selected_missing_in_full63", 0) or 0)) for r in all_selected_summary_rows
    )
    print(
        "[summary] selected_pair_matching matched=%d missing=%d"
        % (len(all_selected_pair_rows) - selected_missing_total, selected_missing_total)
    )


if __name__ == "__main__":
    main()
