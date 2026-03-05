#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import full_eval as fe


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def _pick_pdf(paths: Iterable[Path], token: str) -> Path:
    for p in paths:
        if p.suffix.lower() == ".pdf" and token in p.name:
            return p
    raise RuntimeError(f"Could not find generated PDF containing token '{token}'.")


def _replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    src.replace(dst)


def _unlink_if_exists(path: Path) -> None:
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
    except Exception:
        pass


def _plot_avg_jobs_from_strict_csv(csv_path: Path, out_pdf: Path, sizes: List[int]) -> None:
    plt = fe._import_matplotlib()
    np = fe.np
    rows = fe._read_rows_csv(csv_path)

    method_priority = ["Qiskit", "FrozenQubits", "CutQC", "QOS", "QOSE"]
    hatch_patterns = ["///", "\\\\\\", "...", "xxx", "+++", "---", "|||", "ooo", "***"]
    cmap_jobs = plt.get_cmap("tab10")
    y_eps = 1e-2

    panel_specs = [int(s) for s in sizes]
    with plt.rc_context(
        {
            "font.size": 15,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, len(panel_specs), figsize=(5.9 * len(panel_specs), 4.8), squeeze=False)

        for panel_idx, size in enumerate(panel_specs):
            ax = axes[0, panel_idx]
            sub = [r for r in rows if int(r.get("size", 0) or 0) == size]
            methods = [m for m in method_priority if any(str(r.get("method", "")) == m for r in sub)]
            x = np.arange(len(methods))
            vals = []
            for m in methods:
                m_rows = [r for r in sub if str(r.get("method", "")) == m]
                jobs_vals = [
                    fe._safe_float(r.get("jobs"), 0.0)
                    for r in m_rows
                    if fe._safe_float(r.get("jobs"), 0.0) > 0.0
                ]
                avg = float(sum(jobs_vals) / len(jobs_vals)) if jobs_vals else 0.0
                vals.append(max(avg, y_eps))
            colors = [cmap_jobs(i % 10) for i in range(len(methods))]
            bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.7)
            for i, bar in enumerate(bars):
                bar.set_hatch(hatch_patterns[i % len(hatch_patterns)])
                m_rows = [r for r in sub if str(r.get("method", "")) == methods[i]]
                jobs_vals = [
                    fe._safe_float(r.get("jobs"), 0.0)
                    for r in m_rows
                    if fe._safe_float(r.get("jobs"), 0.0) > 0.0
                ]
                raw = float(sum(jobs_vals) / len(jobs_vals)) if jobs_vals else 0.0
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    max(vals[i], y_eps) * 1.08,
                    f"{raw:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
            ax.set_title(f"{size} qubits", fontsize=17)
            ax.set_ylabel("#Jobs Per Circuit")
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=32, ha="right")
            ax.set_yscale("log")
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_pdf)
        plt.close(fig)


def _plot_qose_time_breakdown_aggregated(
    timing_torino_csv: Path,
    timing_marrakesh_csv: Path,
    job_torino_jsonl: Path,
    job_marrakesh_jsonl: Path,
    out_pdf: Path,
    sizes: List[int],
) -> None:
    """
    Aggregated (Torino + Marrakesh) QOSE time breakdown in one bar figure.
    Each qubit size has separate bars for:
    - Mitigation (avg per bench)
    - Queue/Network (avg per job)
    - QPU (avg per job)
    """
    plt = fe._import_matplotlib()
    np = fe.np

    backend_timing = {
        "torino": fe._read_rows_csv(timing_torino_csv),
        "marrakesh": fe._read_rows_csv(timing_marrakesh_csv),
    }
    backend_jobs = {
        "torino": job_torino_jsonl,
        "marrakesh": job_marrakesh_jsonl,
    }

    per_bench_mit: list[float] = []
    per_job_wait: list[float] = []
    per_job_qpu: list[float] = []
    per_bench_mit_err: list[float] = []
    per_job_wait_err: list[float] = []
    per_job_qpu_err: list[float] = []
    size_labels = [f"{int(s)} qubit" for s in sizes]

    for size in sizes:
        size = int(size)
        mitigation_sum = 0.0
        benches_total = 0.0
        jobs = 0.0
        wait_sum = 0.0
        qpu_sum = 0.0
        backend_m_vals: list[float] = []
        backend_w_vals: list[float] = []
        backend_q_vals: list[float] = []

        for backend, rows in backend_timing.items():
            mitigation_sum_b = 0.0
            bench_seen_b: set[str] = set()
            for r in rows:
                if str(r.get("method", "")).strip() != "QOSE":
                    continue
                if int(r.get("size", 0) or 0) != size:
                    continue
                total = fe._safe_float(r.get("total"))
                if total is not None:
                    mitigation_sum += float(total)
                    mitigation_sum_b += float(total)
                bench = str(r.get("bench", "")).strip()
                if bench:
                    bench_seen_b.add(bench)

            jobs_b = 0.0
            wait_sum_b = 0.0
            qpu_sum_b = 0.0
            with open(backend_jobs[backend], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(rec.get("method", "")).strip() != "QOSE":
                        continue
                    if int(rec.get("size", 0) or 0) != size:
                        continue
                    elapsed = fe._safe_float(rec.get("elapsed_sec"), 0.0)
                    qpu = fe._safe_float(rec.get("qpu_sec"), 0.0)
                    wait = max(float(elapsed) - float(qpu), 0.0)
                    jobs += 1.0
                    jobs_b += 1.0
                    wait_sum += wait
                    wait_sum_b += wait
                    qpu_sum += float(qpu)
                    qpu_sum_b += float(qpu)

            if jobs_b > 0:
                bench_count_b = float(len(bench_seen_b))
                benches_total += bench_count_b
                backend_m_vals.append(mitigation_sum_b / bench_count_b if bench_count_b > 0 else 0.0)
                backend_w_vals.append(wait_sum_b / jobs_b)
                backend_q_vals.append(qpu_sum_b / jobs_b)

        if benches_total > 0:
            per_bench_mit.append(mitigation_sum / benches_total)
        else:
            per_bench_mit.append(0.0)

        if jobs > 0:
            per_job_wait.append(wait_sum / jobs)
            per_job_qpu.append(qpu_sum / jobs)
        else:
            per_job_wait.append(0.0)
            per_job_qpu.append(0.0)

        per_bench_mit_err.append(float(np.std(backend_m_vals)) if len(backend_m_vals) >= 2 else 0.0)
        per_job_wait_err.append(float(np.std(backend_w_vals)) if len(backend_w_vals) >= 2 else 0.0)
        per_job_qpu_err.append(float(np.std(backend_q_vals)) if len(backend_q_vals) >= 2 else 0.0)

    colors = {
        "Mitigation": "#4C78A8",
        "Queue/Network": "#F58518",
        "QPU": "#54A24B",
    }
    x = np.arange(len(sizes))
    width = 0.24
    y_eps = 1e-3

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2))

    vals_m = [max(v, y_eps) for v in per_bench_mit]
    vals_w = [max(v, y_eps) for v in per_job_wait]
    vals_q = [max(v, y_eps) for v in per_job_qpu]
    err_m = [min(max(e, 0.0), max(v * 0.8, 0.0)) for v, e in zip(vals_m, per_bench_mit_err)]
    err_w = [min(max(e, 0.0), max(v * 0.8, 0.0)) for v, e in zip(vals_w, per_job_wait_err)]
    err_q = [min(max(e, 0.0), max(v * 0.8, 0.0)) for v, e in zip(vals_q, per_job_qpu_err)]

    b1 = ax.bar(
        x - width,
        vals_m,
        width=width,
        yerr=err_m,
        capsize=4,
        color=colors["Mitigation"],
        edgecolor="black",
        linewidth=0.7,
    )
    b2 = ax.bar(
        x,
        vals_w,
        width=width,
        yerr=err_w,
        capsize=4,
        color=colors["Queue/Network"],
        edgecolor="black",
        linewidth=0.7,
    )
    b3 = ax.bar(
        x + width,
        vals_q,
        width=width,
        yerr=err_q,
        capsize=4,
        color=colors["QPU"],
        edgecolor="black",
        linewidth=0.7,
    )

    for bar in b1:
        bar.set_hatch("///")
    for bar in b2:
        bar.set_hatch("\\\\\\")
    for bar in b3:
        bar.set_hatch("xx")

    def _annotate(container, raw_vals, err_vals):
        for bar, raw, err in zip(container, raw_vals, err_vals):
            shown = max(float(raw), y_eps)
            label = f"{float(raw):.2f}" if float(raw) > 0 else "0"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(shown + float(err), shown) * 1.08,
                label,
                ha="center",
                va="bottom",
                fontsize=11,
            )

    _annotate(b1, per_bench_mit, err_m)
    _annotate(b2, per_job_wait, err_w)
    _annotate(b3, per_job_qpu, err_q)

    ax.set_xticks(x, size_labels)
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.legend(
        [b1[0], b2[0], b3[0]],
        ["Mitigation", "Queue/Network", "QPU"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_qos_component_breakdown_12_24(
    timing_torino_csv: Path,
    timing_marrakesh_csv: Path,
    out_pdf: Path,
    sizes: List[int],
) -> None:
    """
    QOS component breakdown (different components as different bars),
    grouped by size (12q/24q), with log-scale y-axis, error bars, and value labels.
    """
    plt = fe._import_matplotlib()
    np = fe.np

    stage_skip = {"bench", "size", "method", "total", "overall", "simulation", "cost_search_calls"}
    stage_alias = {"qaoa_analysis": "analysis"}
    stage_order_pref = ["analysis", "qf", "gv", "wc", "qr", "cost_search"]
    stage_label_map = {
        "analysis": "Analysis",
        "qf": "QF",
        "gv": "GV",
        "wc": "WC",
        "qr": "QR",
        "cost_search": "Search",
    }

    def _stage_values_merged(row: Dict[str, object]) -> Dict[str, float]:
        merged_vals: Dict[str, float] = {}
        for key, val in row.items():
            if key in stage_skip:
                continue
            stage = stage_alias.get(key, key)
            sec = max(fe._safe_float(val, 0.0), 0.0)
            if sec <= 0.0:
                continue
            merged_vals[stage] = merged_vals.get(stage, 0.0) + sec
        return merged_vals

    timing_rows = fe._read_rows_csv(timing_torino_csv) + fe._read_rows_csv(timing_marrakesh_csv)
    size_set = {int(s) for s in sizes}
    stage_vals_by_size: Dict[int, Dict[str, List[float]]] = {int(s): {} for s in sizes}

    for row in timing_rows:
        if str(row.get("method", "")).strip() != "QOS":
            continue
        size = fe._safe_int(row.get("size", ""), -1)
        if size not in size_set:
            continue
        merged = _stage_values_merged(row)
        if not merged:
            continue
        for stage, sec in merged.items():
            stage_vals_by_size[size].setdefault(stage, []).append(float(sec))

    stage_set = set()
    for s in size_set:
        stage_set.update(stage_vals_by_size[s].keys())
    stage_order = [s for s in stage_order_pref if s in stage_set] + sorted(stage_set - set(stage_order_pref))
    if not stage_order:
        stage_order = ["analysis"]

    # Keep components with any signal across requested sizes.
    keep_stages = []
    for stage in stage_order:
        has_signal = any(sum(stage_vals_by_size[int(sz)].get(stage, [])) > 0.0 for sz in sizes)
        if has_signal:
            keep_stages.append(stage)
    stage_order = keep_stages if keep_stages else stage_order

    means: Dict[int, List[float]] = {int(s): [] for s in sizes}
    stds: Dict[int, List[float]] = {int(s): [] for s in sizes}
    for sz in sizes:
        sz = int(sz)
        for stage in stage_order:
            vals = stage_vals_by_size[sz].get(stage, [])
            if vals:
                means[sz].append(float(np.mean(vals)))
                stds[sz].append(float(np.std(vals)))
            else:
                means[sz].append(0.0)
                stds[sz].append(0.0)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(stage_order))
    width = 0.34
    y_eps = 1e-4
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.4))

    colors = {int(sizes[0]): "#4C78A8", int(sizes[1]): "#F58518"} if len(sizes) >= 2 else {int(sizes[0]): "#4C78A8"}
    hatches = {int(sizes[0]): "///", int(sizes[1]): "\\\\\\",}

    size_a = int(sizes[0])
    vals_a = [max(v, y_eps) for v in means[size_a]]
    err_a = [min(max(e, 0.0), max(v * 0.8, 0.0)) for v, e in zip(vals_a, stds[size_a])]
    bars_a = ax.bar(
        x - width / 2.0,
        vals_a,
        width=width,
        yerr=err_a,
        capsize=4,
        color=colors[size_a],
        edgecolor="black",
        linewidth=0.7,
        label=f"{size_a} qubit",
    )
    for b in bars_a:
        b.set_hatch(hatches[size_a])

    bars_b = []
    if len(sizes) >= 2:
        size_b = int(sizes[1])
        vals_b = [max(v, y_eps) for v in means[size_b]]
        err_b = [min(max(e, 0.0), max(v * 0.8, 0.0)) for v, e in zip(vals_b, stds[size_b])]
        bars_b = ax.bar(
            x + width / 2.0,
            vals_b,
            width=width,
            yerr=err_b,
            capsize=4,
            color=colors[size_b],
            edgecolor="black",
            linewidth=0.7,
            label=f"{size_b} qubit",
        )
        for b in bars_b:
            b.set_hatch(hatches[size_b])

    # Annotate mean value on top of each bar.
    def _annotate(bar_container, raw_vals: List[float], err_vals: List[float]) -> None:
        for bar, raw, err in zip(bar_container, raw_vals, err_vals):
            shown = max(float(raw), y_eps)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(shown + float(err), shown) * 1.08,
                f"{float(raw):.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    _annotate(bars_a, means[size_a], err_a)
    if len(sizes) >= 2:
        _annotate(bars_b, means[size_b], err_b)

    stage_labels = [stage_label_map.get(s, s) for s in stage_order]
    ax.set_xticks(x, stage_labels)
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_mitigation_stage_breakdown_qos_qsa_24(
    timing_torino_csv: Path,
    timing_marrakesh_csv: Path,
    out_pdf: Path,
    size: int = 24,
) -> None:
    """
    Single-bar mitigation breakdown per benchmark for QOS (left) and QSA (right),
    where QSA is QOSE renamed in the plot. Only uses one qubit size (default: 24).
    """
    plt = fe._import_matplotlib()
    np = fe.np

    rows = fe._read_rows_csv(timing_torino_csv) + fe._read_rows_csv(timing_marrakesh_csv)
    bench_order = [b for b, _ in fe.BENCHES]
    method_map = {"QOS": "QOS", "QOSE": "QSA"}

    stage_skip = {"bench", "size", "method", "overall", "simulation", "cost_search_calls"}

    def _row_total(row: dict) -> float:
        total = fe._safe_float(row.get("total"))
        if total is not None and total > 0:
            return float(total)
        s = 0.0
        for k, v in row.items():
            if k in stage_skip:
                continue
            s += max(fe._safe_float(v, 0.0), 0.0)
        return s

    values: dict[str, dict[str, list[float]]] = {
        "QOS": {b: [] for b in bench_order},
        "QOSE": {b: [] for b in bench_order},
    }

    for r in rows:
        method = str(r.get("method", "")).strip()
        if method not in values:
            continue
        if fe._safe_int(r.get("size", -1), -1) != int(size):
            continue
        bench = str(r.get("bench", "")).strip()
        if bench not in values[method]:
            continue
        t = _row_total(r)
        if t > 0:
            values[method][bench].append(t)

    benches = [
        b
        for b in bench_order
        if values["QOS"].get(b) or values["QOSE"].get(b)
    ]
    if not benches:
        benches = bench_order

    means = {
        method: [
            float(np.mean(values[method][b])) if values[method][b] else 0.0
            for b in benches
        ]
        for method in ("QOS", "QOSE")
    }

    y_eps = 1e-4
    x = np.arange(len(benches))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(benches))]
    hatches = ["///", "\\\\\\", "xxx", "...", "++", "--", "||", "oo", "**", "///"]
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.3), sharey=True)
        for ax, method in zip(axes, ("QOS", "QOSE")):
            vals = [max(v, y_eps) for v in means[method]]
            bars = ax.bar(
                x,
                vals,
                color=colors,
                edgecolor="black",
                linewidth=0.7,
            )
            for i, b in enumerate(bars):
                b.set_hatch(hatches[i % len(hatches)])
                raw = means[method][i]
                y = max(raw, y_eps) * 1.08
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    y,
                    f"{raw:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

            ax.set_title(method_map[method], fontsize=18)
            ax.set_xticks(x, benches, rotation=32, ha="right")
            ax.set_yscale("log")
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        axes[0].set_ylabel("Time (s)")
        fig.tight_layout()
        fig.savefig(out_pdf)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all overall paper figures from local plot_data only. "
            "No temporary directories and no extra output artifacts are kept."
        )
    )
    default_root = Path(__file__).resolve().parent
    parser.add_argument("--root", default=str(default_root))
    parser.add_argument("--sizes", default="12,24")
    parser.add_argument("--panel-methods", default="FrozenQubits,CutQC,QOS,QOSE")
    parser.add_argument("--timebreak-methods", default="QOSE")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_dir = root / "plot_data"
    out_dir = root / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = _parse_int_list(args.sizes)
    panel_methods = _parse_csv_list(args.panel_methods)
    timebreak_methods = _parse_csv_list(args.timebreak_methods)
    benches = [b for b, _ in fe.BENCHES]

    sim_torino = _require(data_dir / "relative_properties_sim_torino.csv")
    sim_marrakesh = _require(data_dir / "relative_properties_sim_marrakesh.csv")
    real_torino = _require(data_dir / "relative_properties_real_torino.csv")
    real_marrakesh = _require(data_dir / "relative_properties_real_marrakesh.csv")
    timing_torino = _require(data_dir / "timing_torino.csv")
    timing_marrakesh = _require(data_dir / "timing_marrakesh.csv")
    job_torino = _require(data_dir / "job_cache_torino.jsonl")
    job_marrakesh = _require(data_dir / "job_cache_marrakesh.jsonl")
    job_counts_strictdry = _require(data_dir / "job_counts_strictdry.csv")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Enforce self-contained generation: disable auto-fill fallbacks to external CSVs.
    original_auto_fill = fe._auto_fill_rows_from_csv_fallbacks
    fe._auto_fill_rows_from_csv_fallbacks = (
        lambda primary_rows, primary_csv, **kwargs: (list(primary_rows), 0, 0)
    )
    try:
        # Torino panels (keep all three).
        tor_paths = fe._plot_cached_panels(
            simtiming_csv=sim_torino,
            real_torino_csv=real_torino,
            real_marrakesh_csv=real_marrakesh,
            timing_csv=timing_torino,
            out_dir=out_dir,
            timestamp=ts,
            tag="overall_torino",
            sizes=sizes,
            methods=panel_methods,
        )
        _replace(_pick_pdf(tor_paths, "depth_cnot"), out_dir / "panel_depth_cnot.pdf")
        _replace(_pick_pdf(tor_paths, "real_fidelity"), out_dir / "panel_real_fidelity.pdf")
        _replace(
            _pick_pdf(tor_paths, "sim_fidelity_timing"),
            out_dir / "panel_sim_fidelity_timing_torino.pdf",
        )

        # Marrakesh panels (keep only sim+timing panel; depth/real are duplicates).
        mar_paths = fe._plot_cached_panels(
            simtiming_csv=sim_marrakesh,
            real_torino_csv=real_torino,
            real_marrakesh_csv=real_marrakesh,
            timing_csv=timing_marrakesh,
            out_dir=out_dir,
            timestamp=ts,
            tag="overall_marrakesh",
            sizes=sizes,
            methods=panel_methods,
        )
        _replace(
            _pick_pdf(mar_paths, "sim_fidelity_timing"),
            out_dir / "panel_sim_fidelity_timing_marrakesh.pdf",
        )
        _unlink_if_exists(_pick_pdf(mar_paths, "depth_cnot"))
        _unlink_if_exists(_pick_pdf(mar_paths, "real_fidelity"))

        # Breakdown figures.
        brk_paths = fe._plot_time_breakdowns(
            timing_csv=timing_torino,
            job_cache_jsonl=job_torino,
            out_dir=out_dir,
            timestamp=ts,
            tag="overall_torino_marrakesh",
            sizes=sizes,
            methods=timebreak_methods,
            benches=benches,
            secondary_timing_csv=timing_marrakesh,
            secondary_job_cache_jsonl=job_marrakesh,
            primary_label="torino",
            secondary_label="marrakesh",
        )
        _plot_qose_time_breakdown_aggregated(
            timing_torino_csv=timing_torino,
            timing_marrakesh_csv=timing_marrakesh,
            job_torino_jsonl=job_torino,
            job_marrakesh_jsonl=job_marrakesh,
            out_pdf=out_dir / "qose_time_breakdown_torino_marrakesh.pdf",
            sizes=sizes,
        )
        _plot_mitigation_stage_breakdown_qos_qsa_24(
            timing_torino_csv=timing_torino,
            timing_marrakesh_csv=timing_marrakesh,
            out_pdf=out_dir / "mitigation_stage_breakdown_qos_qose.pdf",
            size=24,
        )
        _plot_avg_jobs_from_strict_csv(
            job_counts_strictdry,
            out_dir / "panel_avg_jobs_per_bench.pdf",
            sizes=sizes,
        )
        _plot_qos_component_breakdown_12_24(
            timing_torino_csv=timing_torino,
            timing_marrakesh_csv=timing_marrakesh,
            out_pdf=out_dir / "qos_component_breakdown_12_24.pdf",
            sizes=sizes,
        )
        for p in brk_paths:
            if p.suffix.lower() == ".pdf" and "qose_time_breakdown" in p.name:
                _unlink_if_exists(p)
            if p.suffix.lower() == ".pdf" and "mitigation_stage_breakdown_qos_qose" in p.name:
                _unlink_if_exists(p)
            if p.suffix.lower() == ".pdf" and "real_jobs_avg_per_bench" in p.name:
                _unlink_if_exists(p)
            if p.suffix.lower() == ".csv":
                _unlink_if_exists(p)
    finally:
        fe._auto_fill_rows_from_csv_fallbacks = original_auto_fill

    print(f"Generated figures in: {out_dir}")
    for name in [
        "panel_depth_cnot.pdf",
        "panel_real_fidelity.pdf",
        "panel_sim_fidelity_timing_torino.pdf",
        "panel_sim_fidelity_timing_marrakesh.pdf",
        "qose_time_breakdown_torino_marrakesh.pdf",
        "mitigation_stage_breakdown_qos_qose.pdf",
        "panel_avg_jobs_per_bench.pdf",
        "qos_component_breakdown_12_24.pdf",
    ]:
        path = out_dir / name
        print(path)


if __name__ == "__main__":
    main()
