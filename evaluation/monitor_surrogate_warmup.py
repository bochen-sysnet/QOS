#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


_FEATURE_COLS = [
    "sample_qose_depth",
    "sample_qose_cnot",
    "sample_qose_overhead",
    "sample_avg_run_time",
    "sample_combined_score_raw",
]


def _safe_float(v, default=float("nan")):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / den)


def _dedupe_state_rows(rows: list[dict]) -> list[dict]:
    latest_labeled = {}
    latest_labeled_ts = {}
    latest_any = {}
    latest_any_ts = {}
    for r in rows:
        pid = str(r.get("program_hash", "")).strip()
        if not pid:
            continue
        ts = _safe_float(r.get("timestamp_sec"), default=0.0)
        if pid not in latest_any or ts >= latest_any_ts.get(pid, float("-inf")):
            latest_any[pid] = r
            latest_any_ts[pid] = ts
        labeled = str(r.get("global_combined_score", "")).strip() not in {"", "nan"}
        if labeled and (pid not in latest_labeled or ts >= latest_labeled_ts.get(pid, float("-inf"))):
            latest_labeled[pid] = r
            latest_labeled_ts[pid] = ts
    out = []
    for pid in latest_any:
        out.append(latest_labeled.get(pid, latest_any[pid]))
    return out


def _dedupe_case_rows(rows: list[dict]) -> list[dict]:
    latest = {}
    latest_ts = {}
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


def _get_case_score(row: dict, score_mode: str) -> float:
    if score_mode == "legacy":
        return _safe_float(row.get("case_score_legacy"))
    return _safe_float(row.get("case_score_piecewise"))


def _build_maps(state_rows: list[dict], case_rows: list[dict], score_mode: str):
    state_rows = _dedupe_state_rows(state_rows)
    labeled = []
    for r in state_rows:
        g = _safe_float(r.get("global_combined_score"))
        if g == g:
            labeled.append(r)
    labeled.sort(key=lambda r: _safe_float(r.get("timestamp_sec"), default=0.0))

    global_by_pid = {
        str(r.get("program_hash", "")).strip(): _safe_float(r.get("global_combined_score"))
        for r in labeled
    }
    state_by_pid = {str(r.get("program_hash", "")).strip(): r for r in labeled}

    pair_pid_score = {}
    pid_pair_score = {}
    for r in _dedupe_case_rows(case_rows):
        pid = str(r.get("program_hash", "")).strip()
        bench = str(r.get("bench", "")).strip()
        try:
            size = int(float(r.get("size", "nan")))
        except Exception:
            continue
        if pid not in global_by_pid:
            continue
        s = _get_case_score(r, score_mode=score_mode)
        if s != s:
            continue
        pair = (bench, size)
        pair_pid_score.setdefault(pair, {})[pid] = s
        pid_pair_score.setdefault(pid, {})[pair] = s

    ordered_ids = [str(r.get("program_hash", "")).strip() for r in labeled]
    return ordered_ids, global_by_pid, state_by_pid, pair_pid_score, pid_pair_score


def _select_top_corr_pairs(train_ids: list[str], global_by_pid: dict, pair_pid_score: dict, top_k: int):
    ranking = []
    for pair, pid_scores in pair_pid_score.items():
        xs = []
        ys = []
        for pid in train_ids:
            if pid not in pid_scores:
                continue
            x = _safe_float(pid_scores.get(pid))
            y = _safe_float(global_by_pid.get(pid))
            if x != x or y != y:
                continue
            xs.append(x)
            ys.append(y)
        corr = _pearson(xs, ys)
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
    ranked = [r for r in ranking if _safe_float(r.get("abs_corr")) == _safe_float(r.get("abs_corr"))]
    if not ranked:
        ranked = ranking
    selected = ranked[: max(1, int(top_k))]
    return selected, ranking


def _train_predict_rf(train_state_rows: list[dict], test_state_row: dict):
    x = []
    y = []
    for r in train_state_rows:
        feat = [_safe_float(r.get(c)) for c in _FEATURE_COLS]
        target = _safe_float(r.get("global_combined_score"))
        if any(v != v for v in feat) or target != target:
            continue
        x.append(feat)
        y.append(target)
    if len(x) < 2:
        return float("nan")
    tfeat = [_safe_float(test_state_row.get(c)) for c in _FEATURE_COLS]
    if any(v != v for v in tfeat):
        return float("nan")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
            ),
        ]
    )
    model.fit(x, y)
    return float(model.predict([tfeat])[0])


def compute_monitor_rows(state_csv: Path, case_csv: Path, score_mode: str, min_train: int, top_k: int):
    state_rows = _read_csv(state_csv)
    case_rows = _read_csv(case_csv)
    (
        ordered_ids,
        global_by_pid,
        state_by_pid,
        pair_pid_score,
        pid_pair_score,
    ) = _build_maps(state_rows, case_rows, score_mode=score_mode)

    out = []
    n = len(ordered_ids)
    for k in range(max(1, min_train), n):
        train_ids = ordered_ids[:k]
        test_id = ordered_ids[k]
        actual = _safe_float(global_by_pid.get(test_id))
        if actual != actual:
            continue

        selected, ranking = _select_top_corr_pairs(
            train_ids=train_ids,
            global_by_pid=global_by_pid,
            pair_pid_score=pair_pid_score,
            top_k=top_k,
        )
        selected_pairs = [(str(r["bench"]), int(r["size"])) for r in selected]
        test_pair_scores = pid_pair_score.get(test_id, {})
        sel_scores = [
            _safe_float(test_pair_scores.get(p))
            for p in selected_pairs
            if _safe_float(test_pair_scores.get(p)) == _safe_float(test_pair_scores.get(p))
        ]
        selected_score = float(np.mean(sel_scores)) if sel_scores else float("nan")
        selected_gap = (
            abs(selected_score - actual)
            if selected_score == selected_score
            else float("nan")
        )

        pred = _train_predict_rf(
            train_state_rows=[state_by_pid[pid] for pid in train_ids if pid in state_by_pid],
            test_state_row=state_by_pid[test_id],
        )
        pred_gap = abs(pred - actual) if pred == pred else float("nan")

        top_abs_corr = _safe_float(selected[0].get("abs_corr")) if selected else float("nan")
        top_support = int(_safe_float(selected[0].get("support"), default=0)) if selected else 0

        out.append(
            {
                "k_train": k,
                "test_index": k + 1,
                "program_hash": test_id,
                "actual_global_score": actual,
                "selected_score": selected_score,
                "predicted_score_rf": pred,
                "selected_gap_abs": selected_gap,
                "pred_gap_abs": pred_gap,
                "pred_better_than_selected": (
                    int(pred_gap < selected_gap)
                    if not (math.isnan(pred_gap) or math.isnan(selected_gap))
                    else ""
                ),
                "selected_pairs": ",".join(f"{b}-{s}" for b, s in selected_pairs),
                "selected_count": len(selected_pairs),
                "selected_matched_count": len(sel_scores),
                "top_abs_corr": top_abs_corr,
                "top_support": top_support,
                "candidate_pair_count": len(ranking),
            }
        )
    return out


def _plot(rows: list[dict], out_pdf: Path, title: str) -> bool:
    if not rows:
        return False
    ks = [int(_safe_float(r.get("k_train"), default=0)) for r in rows]
    sel_gap = [_safe_float(r.get("selected_gap_abs")) for r in rows]
    pred_gap = [_safe_float(r.get("pred_gap_abs")) for r in rows]

    def _cummean(vals):
        out = []
        seen = []
        for v in vals:
            if v == v:
                seen.append(v)
            out.append(float(np.mean(seen)) if seen else float("nan"))
        return out

    sel_cum = _cummean(sel_gap)
    pred_cum = _cummean(pred_gap)

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    axes[0].plot(ks, sel_gap, marker="o", linewidth=2.0, label="selection gap |selected-global|")
    axes[0].plot(ks, pred_gap, marker="o", linewidth=2.0, label="prediction gap |pred-global|")
    axes[0].set_xlabel("Warmup train size k")
    axes[0].set_ylabel("Abs gap")
    axes[0].set_title("Per-step Performance")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(ks, sel_cum, marker="o", linewidth=2.0, label="cummean selection gap")
    axes[1].plot(ks, pred_cum, marker="o", linewidth=2.0, label="cummean prediction gap")
    axes[1].set_xlabel("Warmup train size k")
    axes[1].set_ylabel("Cumulative mean abs gap")
    axes[1].set_title("Cumulative Performance")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)

    fig.suptitle(title)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    return True


def _summary(rows: list[dict]) -> str:
    if not rows:
        return "rows=0"
    sel = [_safe_float(r.get("selected_gap_abs")) for r in rows if _safe_float(r.get("selected_gap_abs")) == _safe_float(r.get("selected_gap_abs"))]
    pred = [_safe_float(r.get("pred_gap_abs")) for r in rows if _safe_float(r.get("pred_gap_abs")) == _safe_float(r.get("pred_gap_abs"))]
    wins = [int(_safe_float(r.get("pred_better_than_selected"), default=0)) for r in rows if str(r.get("pred_better_than_selected", "")).strip() != ""]
    return (
        f"rows={len(rows)} "
        f"mean_selected_gap={float(np.mean(sel)) if sel else float('nan'):.4f} "
        f"mean_pred_gap={float(np.mean(pred)) if pred else float('nan'):.4f} "
        f"pred_win_rate={float(np.mean(wins)) if wins else float('nan'):.3f}"
    )


def run_once(args):
    rows = compute_monitor_rows(
        state_csv=args.state_csv,
        case_csv=args.case_csv,
        score_mode=args.score_mode,
        min_train=args.min_train,
        top_k=args.top_k,
    )
    fields = [
        "k_train",
        "test_index",
        "program_hash",
        "actual_global_score",
        "selected_score",
        "predicted_score_rf",
        "selected_gap_abs",
        "pred_gap_abs",
        "pred_better_than_selected",
        "selected_pairs",
        "selected_count",
        "selected_matched_count",
        "top_abs_corr",
        "top_support",
        "candidate_pair_count",
    ]
    _write_csv(args.out_csv, rows, fields)
    wrote_pdf = _plot(rows, args.out_pdf, title=f"Warmup Monitor ({args.score_mode})")
    print(f"[monitor] { _summary(rows) }")
    print(f"[monitor] wrote csv: {args.out_csv}")
    if wrote_pdf:
        print(f"[monitor] wrote pdf: {args.out_pdf}")
    else:
        print("[monitor] skipped pdf (no monitor rows yet)")


def main():
    parser = argparse.ArgumentParser(description="Monitor surrogate warmup selection/prediction performance.")
    parser.add_argument(
        "--state-csv",
        type=Path,
        default=Path("openevolve_output/baselines/qose_surrogate_state.csv"),
    )
    parser.add_argument(
        "--case-csv",
        type=Path,
        default=None,
        help="Defaults to <state_csv>.cases.csv",
    )
    parser.add_argument("--score-mode", choices=["piecewise", "legacy"], default="piecewise")
    parser.add_argument("--min-train", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--out-csv", type=Path, default=Path("evaluation/plots/surrogate_warmup_monitor.csv"))
    parser.add_argument("--out-pdf", type=Path, default=Path("evaluation/plots/surrogate_warmup_monitor.pdf"))
    parser.add_argument("--watch-sec", type=float, default=0.0, help="If >0, refresh continuously.")
    args = parser.parse_args()

    if args.case_csv is None:
        args.case_csv = args.state_csv.with_suffix(args.state_csv.suffix + ".cases.csv")

    if args.watch_sec > 0:
        print("[monitor] watch mode enabled; Ctrl+C to stop")
        last_sig = None
        while True:
            sig = (
                args.state_csv.stat().st_mtime if args.state_csv.exists() else -1,
                args.case_csv.stat().st_mtime if args.case_csv.exists() else -1,
            )
            if sig != last_sig:
                run_once(args)
                last_sig = sig
            time.sleep(args.watch_sec)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
