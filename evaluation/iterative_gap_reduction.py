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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


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


def _plot(records: list[dict[str, Any]], out_pdf: Path) -> None:
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

    fig, axes = plt.subplots(1, 5, figsize=(34.0, 5.6))
    ax_score, ax_gap, ax_ratio, ax_corr, ax_stats = axes

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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    merged_csv = Path(args.merged_csv) if args.merged_csv else None
    rows, data_source = _load_merged(output_dir, merged_csv)
    program_dir = Path(args.program_dir) if args.program_dir else None
    feature_rows, n_filled, n_failed, n_skipped_migrant_no_artifacts = _build_selected_feature_rows(
        rows, output_dir, program_dir
    )
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
        score = _safe_float(row.get("full_score_piecewise"))
        if math.isnan(score):
            skipped_no_full_score += 1
            continue
        if not args.include_failed and str(row.get("failure_reason", "")).strip():
            skipped_failed_rows += 1
            continue
        case_count = int(_safe_float(row.get("full_cases_count"), default=0))
        if case_count < args.min_full_cases:
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
            "error",
        ],
    )
    out_pdf = output_dir / f"iterative_gap_reduction_{args.tag}.pdf"
    _plot(recs, out_pdf)

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
    print(f"[done] wrote figure: {out_pdf}")


if __name__ == "__main__":
    main()
