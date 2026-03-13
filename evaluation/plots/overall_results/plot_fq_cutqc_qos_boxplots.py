#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
PLOT_DATA = ROOT / "plot_data"
OUT_DIR = ROOT / "paper_figures"
JOB_COUNTS_CSV = PLOT_DATA / "job_counts_strictdry.csv"

METHOD_SPECS = [
    ("FrozenQubits", "FQ", "#4C78A8"),
    ("CutQC", "CutQC", "#F58518"),
    ("QOS", "QOS", "#54A24B"),
]
SIZES = [12, 24]
BACKENDS = ["torino", "marrakesh"]
GROUP_CENTERS = [1.0, 2.0]
BOX_WIDTH = 0.18
METHOD_OFFSETS = {
    "FrozenQubits": -0.24,
    "CutQC": 0.0,
    "QOS": 0.24,
}


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


def _quantile(vals: List[float], q: float) -> float:
    data = sorted(vals)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    frac = pos - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def _fmt_value(v: float) -> str:
    if v >= 10:
        return f"{v:.1f}"
    if v >= 0.1:
        return f"{v:.2f}"
    if v >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}"


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


def _collect_relative_fidelity(rows_by_backend: Dict[str, List[dict]]) -> Dict[int, Dict[str, List[float]]]:
    out: Dict[int, Dict[str, List[float]]] = {size: {method: [] for method, _, _ in METHOD_SPECS} for size in SIZES}
    field_map = {
        "FrozenQubits": "rel_fidelity_fq",
        "CutQC": "rel_fidelity_cutqc",
        "QOS": "rel_fidelity_qos",
    }
    for _, rows in rows_by_backend.items():
        for row in rows:
            size = int(row.get("size", 0) or 0)
            if size not in out:
                continue
            for method, _, _ in METHOD_SPECS:
                val = _safe_float(row.get(field_map[method]))
                if val is not None:
                    out[size][method].append(val)
    return out


def _collect_relative_quantum_overhead_from_jobs(job_rows: List[dict], selected_backends: Iterable[str]) -> Dict[int, Dict[str, List[float]]]:
    out: Dict[int, Dict[str, List[float]]] = {size: {method: [] for method, _, _ in METHOD_SPECS} for size in SIZES}
    backend_name_map = {
        "torino": "Torino",
        "marrakesh": "Marrakesh",
    }
    active_backends = {backend_name_map[b] for b in selected_backends}

    for size in SIZES:
        for backend in active_backends:
            qiskit_by_bench: Dict[str, float] = {}
            for row in job_rows:
                if (
                    row.get("backend") == backend
                    and int(row.get("size", 0) or 0) == size
                    and str(row.get("method", "")).strip() == "Qiskit"
                ):
                    qiskit_jobs = _safe_float(row.get("jobs"))
                    if qiskit_jobs is not None and qiskit_jobs > 0:
                        qiskit_by_bench[str(row.get("bench", "")).strip()] = qiskit_jobs
            for method, _, _ in METHOD_SPECS:
                for row in job_rows:
                    if (
                        row.get("backend") != backend
                        or int(row.get("size", 0) or 0) != size
                        or str(row.get("method", "")).strip() != method
                    ):
                        continue
                    bench = str(row.get("bench", "")).strip()
                    qiskit_jobs = qiskit_by_bench.get(bench)
                    method_jobs = _safe_float(row.get("jobs"))
                    if qiskit_jobs is not None and qiskit_jobs > 0 and method_jobs is not None:
                        out[size][method].append(method_jobs / qiskit_jobs)
    return out


def _collect_timing(rows_by_backend: Dict[str, List[dict]]) -> Dict[int, Dict[str, List[float]]]:
    out: Dict[int, Dict[str, List[float]]] = {size: {method: [] for method, _, _ in METHOD_SPECS} for size in SIZES}
    for _, rows in rows_by_backend.items():
        for row in rows:
            method = str(row.get("method", "")).strip()
            size = int(row.get("size", 0) or 0)
            if size not in out or method not in out[size]:
                continue
            val = _safe_float(row.get("total"))
            if val is not None:
                out[size][method].append(val)
    return out


def _draw_grouped_boxplot(
    ax,
    data: Dict[int, Dict[str, List[float]]],
    ylabel: str,
    title: str,
    use_log: bool = False,
    label_mode: str = "side",
    annotate_medians: bool = True,
) -> None:
    for method, label, color in METHOD_SPECS:
        offset = METHOD_OFFSETS[method]
        series = [data[size][method] for size in SIZES]
        positions = [c + offset for c in GROUP_CENTERS]
        bp = ax.boxplot(
            series,
            positions=positions,
            widths=BOX_WIDTH,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            boxprops={"linewidth": 1.0},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
            patch.set_edgecolor("black")
        for key in ("whiskers", "caps"):
            for artist in bp[key]:
                artist.set_color("black")
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=3.5, markerfacecolor=color, markeredgecolor="black", alpha=0.65)
        if annotate_medians:
            for pos, vals in zip(positions, series):
                if not vals:
                    continue
                med = statistics.median(vals)
                if label_mode == "top":
                    x_text = pos
                    y_text = _quantile(vals, 0.75) * 1.5
                    ha = "center"
                    va = "bottom"
                else:
                    x_text = pos + BOX_WIDTH * 0.72
                    y_text = med * (1.02 if med > 0 else 1.0)
                    ha = "left"
                    va = "center"
                ax.text(
                    x_text,
                    y_text,
                    _fmt_value(med),
                    ha=ha,
                    va=va,
                    fontsize=20,
                    color="black",
                )

    ax.set_xticks(GROUP_CENTERS)
    ax.set_xticklabels([str(size) for size in SIZES])
    ax.set_xlabel("Qubits")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    if use_log:
        ax.set_yscale("log")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FQ, CutQC, and QOS using relative simulated fidelity (vs Qiskit) "
            "and absolute mitigation time with grouped box plots."
        )
    )
    parser.add_argument("--backend", default="all", help="all, torino, or marrakesh")
    parser.add_argument(
        "--output",
        default=str(OUT_DIR / "panel_fq_cutqc_qos_sim_fidelity_time_boxplot.pdf"),
        help="Output PDF path",
    )
    args = parser.parse_args()

    selected_backends = list(_iter_selected_backends(args.backend))

    sim_rows = {
        backend: _read_rows(PLOT_DATA / f"relative_properties_sim_{backend}.csv")
        for backend in selected_backends
    }
    timing_rows = {
        backend: _read_rows(PLOT_DATA / f"timing_{backend}.csv")
        for backend in selected_backends
    }

    fidelity = _collect_relative_fidelity(sim_rows)
    timing = _collect_timing(timing_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    _draw_grouped_boxplot(
        axes[0],
        fidelity,
        ylabel="Relative Fidelity",
        title="",
        use_log=True,
        label_mode="top",
        annotate_medians=False,
    )
    _draw_grouped_boxplot(
        axes[1],
        timing,
        ylabel="Mitigation Time (s)",
        title="",
        use_log=True,
        label_mode="side",
        annotate_medians=False,
    )

    # Annotate only the largest QOS fidelity value at 24 qubits.
    qos_vals_24 = fidelity.get(24, {}).get("QOS", [])
    if qos_vals_24:
        y_max = max(qos_vals_24)
        x_qos_24 = GROUP_CENTERS[1] + METHOD_OFFSETS["QOS"]
        y_min_fid, y_max_fid = axes[0].get_ylim()
        y_text_fid = min(y_max * 1.03, y_max_fid * 0.95)
        axes[0].annotate(
            _fmt_value(y_max),
            xy=(x_qos_24, y_max),
            xytext=(x_qos_24 + 0.03, 100),
            textcoords="data",
            ha="left",
            va="bottom",
            fontsize=20,
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "black"},
        )

    # Annotate 24q median mitigation time for FQ/CutQC/QOS as text near each point (no arrow).
    y_min_t, y_max_t = axes[1].get_ylim()
    for method, _label, _color in METHOD_SPECS:
        vals_24 = timing.get(24, {}).get(method, [])
        if not vals_24:
            continue
        y_med = statistics.median(vals_24)
        x_pos = GROUP_CENTERS[1] + METHOD_OFFSETS[method]
        y_text_t = y_med
        if method == "FrozenQubits":
            # Move FQ label slightly lower to reduce overlap with nearby text.
            y_text_t = max(y_med * 0.70, y_min_t * 1.05)
        axes[1].text(
            x_pos + 0.12,
            y_text_t,
            _fmt_value(y_med),
            ha="left",
            va="center",
            fontsize=20,
            color="black",
        )

    legend_handles = [Patch(facecolor=color, edgecolor="black", alpha=0.78, label=label) for _, label, color in METHOD_SPECS]
    legend_labels = [label for _, label, _ in METHOD_SPECS]
    axes[0].legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    axes[1].legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        bbox_to_anchor=(-0.05, 1.0),
        ncol=1,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
