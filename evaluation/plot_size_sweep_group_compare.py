#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_sizes(raw: str) -> List[int]:
    raw = raw.strip()
    if not raw:
        return []
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _parse_group(raw: str) -> Tuple[str, List[str]]:
    if ":" in raw:
        title, names = raw.split(":", 1)
        title = title.strip() or "Group"
    else:
        title = "Group"
        names = raw
    items = [x.strip() for x in names.split(",") if x.strip()]
    if not items:
        raise ValueError(f"Empty group definition: {raw}")
    return title, items


def _to_float(value: str) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _resolve_metrics_csv(item: str, metrics_dir: Path) -> Path:
    path = Path(item)
    if path.exists():
        return path
    cand = metrics_dir / f"size_sweep_{item}_metrics.csv"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"Could not resolve metrics CSV for '{item}'. "
        f"Tried '{item}' and '{cand}'."
    )


def _load_means(csv_path: Path, sizes_filter: List[int]) -> Dict[str, float]:
    rows: List[Dict[str, str]] = []
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if sizes_filter:
        size_set = set(sizes_filter)
        rows = [r for r in rows if str(r.get("size", "")).strip() and int(r["size"]) in size_set]
    if not rows:
        raise ValueError(f"No rows after size filter in {csv_path}")

    def _mean(key: str) -> float:
        vals = [_to_float(r.get(key, "")) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    return {
        "depth": _mean("qose_depth"),
        "cnot": _mean("qose_cnot"),
        "time_mean": _mean("avg_run_time"),
        "time_sum": _mean("qose_over_qos_run_time_sum_ratio"),
    }


def _plot_group(
    ax,
    title: str,
    items: List[str],
    means_by_item: Dict[str, Dict[str, float]],
) -> None:
    x = np.arange(len(items))
    width = 0.19
    metrics = [
        ("Depth Ratio (QOSE/QOS)", "depth", "#4C72B0"),
        ("CNOT Ratio (QOSE/QOS)", "cnot", "#DD8452"),
        ("Time Ratio (QOSE/QOS, mean)", "time_mean", "#55A868"),
        ("Time Ratio (QOSE/QOS, sum)", "time_sum", "#C44E52"),
    ]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for (label, key, color), off in zip(metrics, offsets):
        vals = [means_by_item[item].get(key, float("nan")) for item in items]
        bars = ax.bar(x + off, vals, width, label=label, color=color, edgecolor="black", linewidth=0.4)
        for bar, val in zip(bars, vals):
            if not math.isfinite(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Average Ratio Across Sizes", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(items, rotation=25, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare averaged size-sweep ratios across multiple evolved runs. "
            "Each group becomes one subplot."
        )
    )
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help=(
            "Group definition: 'Title:item1,item2,...'. "
            "Each item is either a metrics CSV path or a run name resolved as "
            "evaluation/plots/size_sweep/size_sweep_<name>_metrics.csv"
        ),
    )
    parser.add_argument(
        "--sizes",
        default="12,14,16,18,20,22,24",
        help="Comma-separated sizes to average over (default: 12,14,...,24).",
    )
    parser.add_argument(
        "--metrics-dir",
        default="evaluation/plots/size_sweep",
        help="Directory containing size_sweep_*_metrics.csv files.",
    )
    parser.add_argument(
        "--out",
        default="evaluation/plots/size_sweep_group_compare.pdf",
        help="Output PDF path.",
    )
    args = parser.parse_args()

    groups = [_parse_group(g) for g in args.group]
    sizes = _parse_sizes(args.sizes)
    metrics_dir = Path(args.metrics_dir)

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(11.5, 4.6 * len(groups)),
        constrained_layout=True,
    )
    if len(groups) == 1:
        axes = [axes]

    for ax, (title, items) in zip(axes, groups):
        means_by_item: Dict[str, Dict[str, float]] = {}
        for item in items:
            csv_path = _resolve_metrics_csv(item, metrics_dir)
            means_by_item[item] = _load_means(csv_path, sizes)
        _plot_group(ax, title, items, means_by_item)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", fontsize=9, frameon=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
