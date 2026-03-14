#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path
import sys
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.methods_combo_eval import (
    BENCHES,
    METHODS,
    _combo_label,
    _run_combo_eval,
    _run_permutation_eval,
)


OUT_ROOT = Path(__file__).resolve().parent
DATA_DIR = OUT_ROOT / "data"
FIG_DIR = OUT_ROOT / "figures"


def _to_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_raw_rows(size: int, budget: int, size_to_reach: int, ideal_size_to_reach: int, timeout_sec: int) -> List[dict]:
    eval_args = SimpleNamespace(
        size_to_reach=size_to_reach,
        ideal_size_to_reach=ideal_size_to_reach,
        budget=budget,
        qos_cost_search=False,
        collect_timing=False,
        timeout_sec=timeout_sec,
        metric_mode="fragment",
        metrics_baseline="kolkata",
        metrics_optimization_level=3,
    )

    rows: List[dict] = []
    for bench, _ in BENCHES:
        combo_rows = _run_combo_eval(bench, size, eval_args)
        for r in combo_rows:
            rows.append(
                {
                    "kind": "combination",
                    "bench": bench,
                    "size": size,
                    "methods": r["methods"],
                    "depth": r["depth"],
                    "cnot": r["cnot"],
                    "error": "",
                }
            )
        perm_rows = _run_permutation_eval(bench, size, eval_args)
        for r in perm_rows:
            rows.append(
                {
                    "kind": "permutation",
                    "bench": bench,
                    "size": size,
                    "methods": r["methods"],
                    "depth": r["depth"],
                    "cnot": r["cnot"],
                    "error": r.get("error", "") or "",
                }
            )
    return rows


def _aggregate(rows: List[dict], labels_combo: List[str], labels_perm: List[str]) -> List[dict]:
    grouped: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: {"depth": [], "cnot": []})
    for r in rows:
        kind = str(r.get("kind", "")).strip()
        methods = str(r.get("methods", "")).strip()
        d = _to_float(r.get("depth"))
        c = _to_float(r.get("cnot"))
        if np.isfinite(d):
            grouped[(kind, methods)]["depth"].append(d)
        if np.isfinite(c):
            grouped[(kind, methods)]["cnot"].append(c)

    out: List[dict] = []
    for kind, labels in (("combination", labels_combo), ("permutation", labels_perm)):
        for m in labels:
            dvals = grouped[(kind, m)]["depth"]
            cvals = grouped[(kind, m)]["cnot"]
            out.append(
                {
                    "kind": kind,
                    "methods": m,
                    "depth_mean": mean(dvals) if dvals else float("nan"),
                    "depth_std": pstdev(dvals) if len(dvals) > 1 else 0.0,
                    "cnot_mean": mean(cvals) if cvals else float("nan"),
                    "cnot_std": pstdev(cvals) if len(cvals) > 1 else 0.0,
                    "n_depth": len(dvals),
                    "n_cnot": len(cvals),
                }
            )
    return out


def _plot_metric_two_panels(
    agg_rows: List[dict],
    labels_combo: List[str],
    labels_perm: List[str],
    metric_mean: str,
    metric_std: str,
    y_label: str,
    out_pdf: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    amap = {(r["kind"], r["methods"]): r for r in agg_rows}
    combo_means = np.array([amap[("combination", m)][metric_mean] for m in labels_combo], dtype=float)
    combo_stds = np.array([amap[("combination", m)][metric_std] for m in labels_combo], dtype=float)
    perm_means = np.array([amap[("permutation", m)][metric_mean] for m in labels_perm], dtype=float)
    perm_stds = np.array([amap[("permutation", m)][metric_std] for m in labels_perm], dtype=float)

    x_combo = np.arange(1, len(labels_combo) + 1, dtype=float)
    x_perm = np.arange(1, len(labels_perm) + 1, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.4), constrained_layout=False)

    ax_cmb, ax_prm = axes
    ours_combo = _combo_label(tuple(METHODS))
    ours_perm = " > ".join(METHODS)

    # Combination panel: line + band, with a red horizontal reference line for ours.
    ax_cmb.plot(x_combo, combo_means, color="#6C757D", linewidth=2.0, zorder=3)
    ax_cmb.fill_between(
        x_combo,
        combo_means - combo_stds,
        combo_means + combo_stds,
        color="#6C757D",
        alpha=0.22,
        zorder=2,
    )
    ax_cmb.set_ylabel(y_label)
    ax_cmb.set_xlabel("Combination ID")
    if len(x_combo) <= 12:
        ax_cmb.set_xticks(x_combo)
    else:
        ticks = np.unique(np.rint(np.linspace(1, len(x_combo), 6)).astype(int))
        ax_cmb.set_xticks(ticks)
    ax_cmb.set_xlim(1, len(x_combo))
    ax_cmb.grid(axis="y", linestyle="--", alpha=0.3)

    # Permutation panel: line + band, with a red horizontal reference line for ours.
    ax_prm.plot(x_perm, perm_means, color="#6C757D", linewidth=2.0, zorder=3)
    ax_prm.fill_between(
        x_perm,
        perm_means - perm_stds,
        perm_means + perm_stds,
        color="#6C757D",
        alpha=0.22,
        zorder=2,
    )
    ax_prm.set_ylabel(y_label)
    ax_prm.set_xlabel("Permutation ID")
    if len(x_perm) <= 12:
        ax_prm.set_xticks(x_perm)
    else:
        ticks = np.unique(np.rint(np.linspace(1, len(x_perm), 6)).astype(int))
        ax_prm.set_xticks(ticks)
    ax_prm.set_xlim(1, len(x_perm))
    ax_prm.grid(axis="y", linestyle="--", alpha=0.3)

    if ours_combo in labels_combo:
        i = labels_combo.index(ours_combo)
        ours_val = float(combo_means[i])
        ax_cmb.axhline(ours_val, color="#E45756", linestyle="--", linewidth=2.0, alpha=0.95, label="Ours")
    if ours_perm in labels_perm:
        i = labels_perm.index(ours_perm)
        ours_val = float(perm_means[i])
        ax_prm.axhline(ours_val, color="#E45756", linestyle="--", linewidth=2.0, alpha=0.95, label="Ours")

    legend_handles = [
        Line2D([0], [0], color="#6C757D", linewidth=2.0, label="Others"),
        Line2D([0], [0], color="#E45756", linestyle="--", linewidth=2.0, label="Ours"),
    ]
    ax_cmb.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=22)
    ax_prm.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=22)

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.15, top=0.98, wspace=0.12)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate methods_combo_perm_all_12 results across benches with error bars."
    )
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--size-to-reach", type=int, default=7)
    parser.add_argument("--ideal-size-to-reach", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    combo_labels = [
        _combo_label(combo)
        for r in range(1, len(METHODS) + 1)
        for combo in combinations(METHODS, r)
    ]
    combo_labels = list(reversed(combo_labels))
    perm_labels = [" > ".join(p) for p in permutations(METHODS)]

    raw_csv = DATA_DIR / f"methods_combo_perm_all_{args.size}_raw.csv"
    agg_csv = DATA_DIR / f"methods_combo_perm_all_{args.size}_aggregate.csv"
    out_pdf_depth = FIG_DIR / f"methods_combo_perm_all_{args.size}_aggregate_depth.pdf"
    out_pdf_cnot = FIG_DIR / f"methods_combo_perm_all_{args.size}_aggregate_cnot.pdf"

    if raw_csv.exists() and not args.recompute:
        rows = _read_csv(raw_csv)
    else:
        rows = _build_raw_rows(
            size=args.size,
            budget=args.budget,
            size_to_reach=args.size_to_reach,
            ideal_size_to_reach=args.ideal_size_to_reach,
            timeout_sec=args.timeout_sec,
        )
        _write_csv(
            raw_csv,
            rows,
            ["kind", "bench", "size", "methods", "depth", "cnot", "error"],
        )

    agg_rows = _aggregate(rows, combo_labels, perm_labels)
    _write_csv(
        agg_csv,
        agg_rows,
        ["kind", "methods", "depth_mean", "depth_std", "cnot_mean", "cnot_std", "n_depth", "n_cnot"],
    )
    _plot_metric_two_panels(
        agg_rows,
        combo_labels,
        perm_labels,
        metric_mean="depth_mean",
        metric_std="depth_std",
        y_label="Depth",
        out_pdf=out_pdf_depth,
    )
    _plot_metric_two_panels(
        agg_rows,
        combo_labels,
        perm_labels,
        metric_mean="cnot_mean",
        metric_std="cnot_std",
        y_label="CNOT",
        out_pdf=out_pdf_cnot,
    )

    print(f"Wrote data: {raw_csv}")
    print(f"Wrote data: {agg_csv}")
    print(f"Wrote figure: {out_pdf_depth}")
    print(f"Wrote figure: {out_pdf_cnot}")


if __name__ == "__main__":
    main()
