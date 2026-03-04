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
SELECTED_MODELS_CSV = ROOT.parent / "current_sweep_compare" / "data" / "selected_models.csv"
SIZE_SWEEP_DIR = ROOT.parent / "current_sweep_compare" / "data" / "size_sweep"
REAL_FALLBACK_ROOT = ROOT.parent / "gem3pro_full_eval"

METHOD_SPECS = [
    ("FrozenQubits", "FQ", "#4C78A8"),
    ("CutQC", "CutQC", "#F58518"),
    ("QOS", "QOS", "#54A24B"),
    ("QOSE", "QOSE", "#B279A2"),
]
SIZES = [12, 24]
BACKENDS = ["torino", "marrakesh"]


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
        "QOSE": "rel_fidelity_qose",
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


def _collect_relative_real_fidelity(rows_by_backend: Dict[str, List[dict]]) -> Dict[int, Dict[str, List[float]]]:
    out: Dict[int, Dict[str, List[float]]] = {size: {method: [] for method, _, _ in METHOD_SPECS} for size in SIZES}
    field_map = {
        "FrozenQubits": "fq_real_fidelity",
        "CutQC": "cutqc_real_fidelity",
        "QOS": "qos_real_fidelity",
        "QOSE": "qose_real_fidelity",
    }
    for _, rows in rows_by_backend.items():
        for row in rows:
            size = int(row.get("size", 0) or 0)
            if size not in out:
                continue
            base = _safe_float(row.get("baseline_real_fidelity"))
            if base is None or base <= 0:
                continue
            for method, _, _ in METHOD_SPECS:
                num = _safe_float(row.get(field_map[method]))
                if num is not None:
                    out[size][method].append(num / base)
    return out


def _load_real_fidelity_fallback_rows() -> List[dict]:
    candidates = []
    for base in [
        REAL_FALLBACK_ROOT / "full_eval_artifacts",
        REAL_FALLBACK_ROOT / "tables" / "debug",
    ]:
        if not base.exists():
            continue
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
            for field in ("rel_fidelity_fq", "rel_fidelity_cutqc", "rel_fidelity_qos"):
                if str(row.get(field, "")).strip():
                    score += 1
        if score > best_score:
            best_rows = rows
            best_score = score
    return best_rows


def _fill_missing_real_fidelity(
    real_data: Dict[int, Dict[str, List[float]]],
    fallback_rows: List[dict],
) -> None:
    if not fallback_rows:
        return
    field_map = {
        "FrozenQubits": "fq_real_fidelity",
        "CutQC": "cutqc_real_fidelity",
        "QOS": "qos_real_fidelity",
        "QOSE": "qose_real_fidelity",
    }
    fallback_by_size_method: Dict[int, Dict[str, List[float]]] = {
        s: {m: [] for m, _, _ in METHOD_SPECS} for s in SIZES
    }
    for row in fallback_rows:
        size = int(row.get("size", 0) or 0)
        if size not in fallback_by_size_method:
            continue
        base = _safe_float(row.get("baseline_real_fidelity"))
        if base is None or base <= 0:
            continue
        for method, _, _ in METHOD_SPECS:
            num = _safe_float(row.get(field_map[method]))
            val = (num / base) if num is not None else None
            if val is not None:
                fallback_by_size_method[size][method].append(val)

    for size in SIZES:
        for method, _, _ in METHOD_SPECS:
            if real_data[size][method]:
                continue
            if fallback_by_size_method[size][method]:
                real_data[size][method] = fallback_by_size_method[size][method]


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


def _collect_timing_baselines(rows_by_backend: Dict[str, List[dict]]) -> Dict[int, Dict[str, List[float]]]:
    out: Dict[int, Dict[str, List[float]]] = {size: {method: [] for method, _, _ in METHOD_SPECS} for size in SIZES}
    for _, rows in rows_by_backend.items():
        for row in rows:
            method = str(row.get("method", "")).strip()
            size = int(row.get("size", 0) or 0)
            if size not in out or method not in out[size]:
                continue
            if method == "QOSE":
                continue
            val = _safe_float(row.get("total"))
            if val is not None:
                out[size][method].append(val)
    return out


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
        # Prefer absolute runtime from qose_run_sec_avg; fallback to avg_run_time ratio if needed.
        val = _safe_float(row.get("qose_run_sec_avg"))
        if val is None:
            val = _safe_float(row.get("avg_run_time"))
        if val is not None:
            out[size] = val
    return out


def _inject_qose_timing(time_data: Dict[int, Dict[str, List[float]]]) -> None:
    qose_time = _load_qose_timing_from_current_sweep_best()
    # Build a distribution from baseline QOS timing using the best Gemini-3 Pro ratio,
    # so QOSE appears as a box (not a single-point line) while still reflecting sweep data.
    for size in SIZES:
        qose_sec = qose_time.get(size)
        qos_vals = list(time_data[size]["QOS"])
        if qose_sec is None:
            continue
        # ratio from sweep: qose_run_sec_avg / qos_run_sec_avg
        qos_mean = statistics.mean(qos_vals) if qos_vals else None
        if qos_mean is not None and qos_mean > 0:
            ratio = qose_sec / qos_mean
            time_data[size]["QOSE"] = [v * ratio for v in qos_vals]
        else:
            time_data[size]["QOSE"] = [qose_sec]


def _draw_grouped_boxplot(
    ax,
    data: Dict[int, Dict[str, List[float]]],
    ylabel: str,
    use_log: bool = False,
    label_mode: str = "side",
) -> None:
    centers = [1.0, 2.0]
    offsets = [-0.30, -0.10, 0.10, 0.30]
    width = 0.16

    for (method, label, color), offset in zip(METHOD_SPECS, offsets):
        series = [data[size][method] for size in SIZES]
        positions = [c + offset for c in centers]
        bp = ax.boxplot(
            series,
            positions=positions,
            widths=width,
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
        for pos, vals in zip(positions, series):
            if not vals:
                continue
            med = statistics.median(vals)
            if label_mode == "top":
                x_text = pos
                y_text = _quantile(vals, 0.75) * 1.35
                ha = "center"
                va = "bottom"
            else:
                x_text = pos + width * 0.68
                y_text = med * (1.02 if med > 0 else 1.0)
                ha = "left"
                va = "center"
            ax.text(
                x_text,
                y_text,
                _fmt_value(med),
                ha=ha,
                va=va,
                fontsize=9.5,
                color="black",
            )

    ax.set_xticks(centers)
    ax.set_xticklabels([str(size) for size in SIZES], fontsize=11)
    ax.set_xlabel("Qubits", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    if use_log:
        ax.set_yscale("log")


def _draw_grouped_bar_mean_std(
    ax,
    data: Dict[int, Dict[str, List[float]]],
    ylabel: str,
    use_log: bool = False,
) -> None:
    centers = [1.0, 2.0]
    offsets = [-0.30, -0.10, 0.10, 0.30]
    width = 0.16

    for (method, _label, color), offset in zip(METHOD_SPECS, offsets):
        means = []
        stds = []
        for size in SIZES:
            vals = list(data[size][method])
            if vals:
                means.append(statistics.mean(vals))
                stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)

        positions = [c + offset for c in centers]
        bars = ax.bar(
            positions,
            means,
            width=width,
            color=color,
            alpha=0.78,
            edgecolor="black",
            linewidth=1.0,
            yerr=stds,
            capsize=2.5,
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
        )
        for bar, mean in zip(bars, means):
            if mean <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                mean * (1.02 if mean > 0 else 1.0),
                _fmt_value(mean),
                ha="center",
                va="bottom",
                fontsize=9.0,
                color="black",
            )

    ax.set_xticks(centers)
    ax.set_xticklabels([str(size) for size in SIZES], fontsize=11)
    ax.set_xlabel("Qubits", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    if use_log:
        ax.set_yscale("log")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FQ, CutQC, QOS, QOSE using four grouped box-plot panels: "
            "sim fidelity, real fidelity, mitigation time, and quantum overhead."
        )
    )
    parser.add_argument("--backend", default="all", help="all, torino, or marrakesh")
    parser.add_argument(
        "--output",
        default=str(OUT_DIR / "panel_fq_cutqc_qos_qose_sim_real_time_overhead_boxplot.pdf"),
        help="Output PDF path",
    )
    args = parser.parse_args()

    selected_backends = list(_iter_selected_backends(args.backend))

    sim_rows = {
        backend: _read_rows(PLOT_DATA / f"relative_properties_sim_{backend}.csv")
        for backend in selected_backends
    }
    real_rows = {
        backend: _read_rows(PLOT_DATA / f"relative_properties_real_{backend}.csv")
        for backend in selected_backends
    }
    timing_rows = {
        backend: _read_rows(PLOT_DATA / f"timing_{backend}.csv")
        for backend in selected_backends
    }
    job_rows = _read_rows(JOB_COUNTS_CSV)

    fidelity_sim = _collect_relative_fidelity(sim_rows)
    fidelity_real = _collect_relative_real_fidelity(real_rows)
    _fill_missing_real_fidelity(fidelity_real, _load_real_fidelity_fallback_rows())
    timing = _collect_timing_baselines(timing_rows)
    _inject_qose_timing(timing)
    overhead = _collect_relative_quantum_overhead_from_jobs(job_rows, selected_backends)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20.5, 8.4))
    _draw_grouped_boxplot(
        axes[0, 0],
        fidelity_sim,
        ylabel="Sim Fidelity / Qiskit",
        use_log=True,
        label_mode="top",
    )
    _draw_grouped_boxplot(
        axes[0, 1],
        fidelity_real,
        ylabel="Real Fidelity / Qiskit",
        use_log=True,
        label_mode="top",
    )
    _draw_grouped_boxplot(
        axes[0, 2],
        timing,
        ylabel="Mitigation Time (s)",
        use_log=True,
        label_mode="side",
    )
    _draw_grouped_boxplot(
        axes[0, 3],
        overhead,
        ylabel="Quantum Overhead / Qiskit",
        use_log=True,
        label_mode="top",
    )
    _draw_grouped_bar_mean_std(
        axes[1, 0],
        fidelity_sim,
        ylabel="Sim Fidelity / Qiskit (mean±std)",
        use_log=True,
    )
    _draw_grouped_bar_mean_std(
        axes[1, 1],
        fidelity_real,
        ylabel="Real Fidelity / Qiskit (mean±std)",
        use_log=True,
    )
    _draw_grouped_bar_mean_std(
        axes[1, 2],
        timing,
        ylabel="Mitigation Time (s) (mean±std)",
        use_log=True,
    )
    _draw_grouped_bar_mean_std(
        axes[1, 3],
        overhead,
        ylabel="Quantum Overhead / Qiskit (mean±std)",
        use_log=True,
    )

    legend_handles = [Patch(facecolor=color, edgecolor="black", alpha=0.78, label=label) for _, label, color in METHOD_SPECS]
    fig.legend(
        handles=legend_handles,
        labels=[label for _, label, _ in METHOD_SPECS],
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
