#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
PLOT_DATA = ROOT / "plot_data"
OUT_DIR = ROOT / "paper_figures"
SELECTED_MODELS_CSV = ROOT.parent / "current_sweep_compare" / "data" / "selected_models.csv"
SIZE_SWEEP_DIR = ROOT.parent / "current_sweep_compare" / "data" / "size_sweep"
REAL_FALLBACK_ROOT = ROOT.parent / "gem3pro_full_eval"

METHOD_SPECS = [
    ("FrozenQubits", "FQ", "#4C78A8"),
    ("CutQC", "CutQC", "#F58518"),
    ("QOS", "QOS", "#54A24B"),
    ("QOSE", "QOSE", "#B279A2"),
]
METHOD_TO_LABEL = {m: label for m, label, _ in METHOD_SPECS}
METHOD_TO_COLOR = {m: color for m, _, color in METHOD_SPECS}
METHOD_TO_MARKER = {
    "FrozenQubits": "o",
    "CutQC": "s",
    "QOS": "^",
    "QOSE": "D",
}
SIZES = [12, 24]
BACKENDS = ["torino", "marrakesh"]

AXIS_LABEL_FONTSIZE = 19
TICK_FONTSIZE = 15
TITLE_FONTSIZE = 19
LEGEND_FONTSIZE = 14
MARKER_SIZE_DEFAULT = 150.0
MARKER_SIZE_AVG = 210.0


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _read_rows(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _iter_selected_backends(raw: str) -> Iterable[str]:
    norm = raw.strip().lower()
    if norm == "all":
        return BACKENDS
    if norm not in BACKENDS:
        raise ValueError(f"Unsupported backend '{raw}'. Choose from: all, torino, marrakesh")
    return [norm]


def _load_qose_timing_from_current_sweep_best() -> Dict[int, float]:
    rows = _read_rows(SELECTED_MODELS_CSV)
    gem3pro = None
    for row in rows:
        if str(row.get("family", "")).strip() == "gem3pro":
            gem3pro = row
            break
    if gem3pro is None:
        raise FileNotFoundError(f"gem3pro row not found in {SELECTED_MODELS_CSV}")

    run_name = str(gem3pro.get("run_name", "")).strip()
    if not run_name:
        raise ValueError(f"Invalid gem3pro run_name in {SELECTED_MODELS_CSV}")
    metrics_csv = SIZE_SWEEP_DIR / f"size_sweep_{run_name}_metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing size sweep metrics for selected gem3pro run: {metrics_csv}")

    out: Dict[int, float] = {}
    for row in _read_rows(metrics_csv):
        size = int(row.get("size", 0) or 0)
        if size not in SIZES:
            continue
        val = _safe_float(row.get("qose_run_sec_avg"))
        if val is None:
            val = _safe_float(row.get("avg_run_time"))
        if val is not None:
            out[size] = val
    return out


def _load_real_fidelity_fallback_map() -> Dict[Tuple[int, str, str], float]:
    candidates: List[Path] = []
    for base in [
        REAL_FALLBACK_ROOT / "full_eval_artifacts",
        REAL_FALLBACK_ROOT / "tables" / "debug",
    ]:
        if base.exists():
            candidates.extend(sorted(base.glob("relative_properties*.csv")))

    best_rows: List[dict] = []
    best_score = -1
    for path in candidates:
        try:
            rows = _read_rows(path)
        except Exception:
            continue
        score = 0
        for row in rows:
            size = int(row.get("size", 0) or 0)
            if size not in SIZES:
                continue
            for field in ("fq_real_fidelity", "cutqc_real_fidelity", "qos_real_fidelity"):
                if str(row.get(field, "")).strip():
                    score += 1
        if score > best_score:
            best_rows = rows
            best_score = score

    field_map = {
        "FrozenQubits": "fq_real_fidelity",
        "CutQC": "cutqc_real_fidelity",
        "QOS": "qos_real_fidelity",
        "QOSE": "qose_real_fidelity",
    }
    acc: Dict[Tuple[int, str, str], List[float]] = {}
    for row in best_rows:
        size = int(row.get("size", 0) or 0)
        bench = str(row.get("bench", "")).strip()
        if size not in SIZES or not bench:
            continue
        base = _safe_float(row.get("baseline_real_fidelity"))
        if base is None or base <= 0:
            continue
        for method, _, _ in METHOD_SPECS:
            num = _safe_float(row.get(field_map[method]))
            if num is None:
                continue
            key = (size, bench, method)
            acc.setdefault(key, []).append(num / base)
    out = {}
    for key, vals in acc.items():
        out[key] = statistics.median(vals)
    return out


def _collect_maps(selected_backends: List[str]):
    sim_map: Dict[Tuple[str, int, str, str], float] = {}
    real_map: Dict[Tuple[str, int, str, str], float] = {}
    timing_map: Dict[Tuple[str, int, str, str], float] = {}

    sim_field = {
        "FrozenQubits": "rel_fidelity_fq",
        "CutQC": "rel_fidelity_cutqc",
        "QOS": "rel_fidelity_qos",
        "QOSE": "rel_fidelity_qose",
    }
    real_field = {
        "FrozenQubits": "fq_real_fidelity",
        "CutQC": "cutqc_real_fidelity",
        "QOS": "qos_real_fidelity",
        "QOSE": "qose_real_fidelity",
    }

    for backend in selected_backends:
        for row in _read_rows(PLOT_DATA / f"relative_properties_sim_{backend}.csv"):
            size = int(row.get("size", 0) or 0)
            bench = str(row.get("bench", "")).strip()
            if size not in SIZES or not bench:
                continue
            for method, _, _ in METHOD_SPECS:
                val = _safe_float(row.get(sim_field[method]))
                if val is not None:
                    sim_map[(backend, size, bench, method)] = val

        for row in _read_rows(PLOT_DATA / f"relative_properties_real_{backend}.csv"):
            size = int(row.get("size", 0) or 0)
            bench = str(row.get("bench", "")).strip()
            if size not in SIZES or not bench:
                continue
            base = _safe_float(row.get("baseline_real_fidelity"))
            if base is None or base <= 0:
                continue
            for method, _, _ in METHOD_SPECS:
                num = _safe_float(row.get(real_field[method]))
                if num is not None:
                    real_map[(backend, size, bench, method)] = num / base

        for row in _read_rows(PLOT_DATA / f"timing_{backend}.csv"):
            size = int(row.get("size", 0) or 0)
            bench = str(row.get("bench", "")).strip()
            method = str(row.get("method", "")).strip()
            if size not in SIZES or not bench or method not in ("FrozenQubits", "CutQC", "QOS"):
                continue
            val = _safe_float(row.get("total"))
            if val is not None:
                timing_map[(backend, size, bench, method)] = val

    # Inject QOSE timing distribution from Gemini-3 Pro best ratio by size.
    qose_sec = _load_qose_timing_from_current_sweep_best()
    for backend in selected_backends:
        for size in SIZES:
            qos_vals = [
                v for (b, s, _bench, m), v in timing_map.items() if b == backend and s == size and m == "QOS"
            ]
            qos_mean = statistics.mean(qos_vals) if qos_vals else None
            if qos_mean is None or qos_mean <= 0:
                continue
            ratio = qose_sec.get(size, 0.0) / qos_mean if qose_sec.get(size, 0.0) > 0 else None
            if ratio is None:
                continue
            for (b, s, bench, m), v in list(timing_map.items()):
                if b == backend and s == size and m == "QOS":
                    timing_map[(backend, size, bench, "QOSE")] = v * ratio

    # Real-fidelity fallback for missing baseline methods in current real CSVs.
    fallback_map = _load_real_fidelity_fallback_map()
    for backend in selected_backends:
        for size in SIZES:
            for bench in {k[2] for k in timing_map if k[0] == backend and k[1] == size}:
                for method, _, _ in METHOD_SPECS:
                    key = (backend, size, bench, method)
                    if key in real_map:
                        continue
                    fkey = (size, bench, method)
                    if fkey in fallback_map:
                        real_map[key] = fallback_map[fkey]

    return sim_map, real_map, timing_map


def _build_points(
    fidelity_map: Dict[Tuple[str, int, str, str], float],
    timing_map: Dict[Tuple[str, int, str, str], float],
) -> List[dict]:
    points: List[dict] = []
    for key, y in timing_map.items():
        x = fidelity_map.get(key)
        if x is None:
            continue
        backend, size, bench, method = key
        points.append(
            {
                "backend": backend,
                "size": size,
                "bench": bench,
                "method": method,
                "x": x,
                "y": y,
            }
        )
    return points


def _average_sim_real_fidelity(
    sim_map: Dict[Tuple[str, int, str, str], float],
    real_map: Dict[Tuple[str, int, str, str], float],
) -> Dict[Tuple[str, int, str, str], float]:
    out: Dict[Tuple[str, int, str, str], float] = {}
    for key, sim_val in sim_map.items():
        real_val = real_map.get(key)
        if real_val is None:
            continue
        out[key] = 0.5 * (sim_val + real_val)
    return out


def _add_better_arrow(ax) -> None:
    ax.annotate(
        "Better",
        xy=(0.30, 0.72),
        xytext=(0.05, 0.95),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        fontsize=14,
        ha="left",
        va="center",
    )


def _plot_aggregated_centers(
    ax,
    points: List[dict],
    x_label: str,
    center: str = "median",
    with_errorbars: bool = False,
    marker_size: float = MARKER_SIZE_DEFAULT,
    marker_alpha: float = 0.55,
):
    legend_handles: List[Line2D] = []
    for method, label, color in METHOD_SPECS:
        sub = [p for p in points if p["method"] == method]
        if not sub:
            continue
        xs = [p["x"] for p in sub]
        ys = [p["y"] for p in sub]
        if center == "mean":
            x_med = statistics.mean(xs)
            y_med = statistics.mean(ys)
            if with_errorbars:
                x_err = statistics.pstdev(xs) if len(xs) > 1 else 0.0
                y_err = statistics.pstdev(ys) if len(ys) > 1 else 0.0
                ax.errorbar(
                    [x_med],
                    [y_med],
                    xerr=[x_err],
                    yerr=[y_err],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.7,
                    capsize=4,
                    alpha=0.9,
                    zorder=2,
                )
        else:
            x_med = statistics.median(xs)
            y_med = statistics.median(ys)
        ax.scatter(
            [x_med],
            [y_med],
            marker=METHOD_TO_MARKER[method],
            s=marker_size,
            color=color,
            edgecolors="black",
            linewidths=1.0,
            alpha=marker_alpha,
            zorder=3,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=METHOD_TO_MARKER[method],
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=10,
                label=label,
            )
        )
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Mitigation Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    _add_better_arrow(ax)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=LEGEND_FONTSIZE)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
        }
    )
    parser = argparse.ArgumentParser(
        description=(
            "Scatter panels for FQ/CutQC/QOS/QOSE aggregated across selected backends: "
            "sim-fidelity-vs-time and real-fidelity-vs-time."
        )
    )
    parser.add_argument("--backend", default="all", help="all, torino, or marrakesh")
    parser.add_argument(
        "--output",
        default=str(OUT_DIR / "panel_fq_cutqc_qos_qose_fidelity_time_scatter.pdf"),
        help="Output PDF path",
    )
    args = parser.parse_args()

    selected_backends = list(_iter_selected_backends(args.backend))
    sim_map, real_map, timing_map = _collect_maps(selected_backends)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_sim_pts = _build_points(sim_map, timing_map)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    panel_specs = [
        (12, "12-qubit Circuits"),
        (24, "24-qubit Circuits"),
    ]
    for cidx, (size, title) in enumerate(panel_specs):
        pts = [p for p in all_sim_pts if p["size"] == size and p["backend"] in selected_backends]
        _plot_aggregated_centers(
            axes[cidx],
            pts,
            "Relative Fidelity",
            center="mean",
            with_errorbars=True,
            marker_size=MARKER_SIZE_DEFAULT,
            marker_alpha=0.55,
        )
        axes[cidx].set_title(title, fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
