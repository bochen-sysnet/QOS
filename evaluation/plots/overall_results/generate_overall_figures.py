#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
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

    panel_specs = [("Torino", int(s)) for s in sizes] + [("Marrakesh", int(s)) for s in sizes]
    fig, axes = plt.subplots(1, len(panel_specs), figsize=(5.5 * len(panel_specs), 4.6), squeeze=False)

    legend_handles = []
    legend_labels: List[str] = []
    for panel_idx, (backend, size) in enumerate(panel_specs):
        ax = axes[0, panel_idx]
        sub = [r for r in rows if str(r.get("backend", "")) == backend and int(r.get("size", 0)) == size]
        methods = [m for m in method_priority if any(str(r.get("method", "")) == m for r in sub)]
        x = np.arange(len(methods))
        vals = []
        for m in methods:
            m_rows = [r for r in sub if str(r.get("method", "")) == m]
            avg = fe._safe_float(m_rows[0].get("avg_jobs_per_bench_active"), 0.0) if m_rows else 0.0
            vals.append(max(avg, y_eps))
        colors = [cmap_jobs(i % 10) for i in range(len(methods))]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.7)
        for i, bar in enumerate(bars):
            bar.set_hatch(hatch_patterns[i % len(hatch_patterns)])
            if panel_idx == 0:
                legend_handles.append(bar)
                legend_labels.append(methods[i])
            m_rows = [r for r in sub if str(r.get("method", "")) == methods[i]]
            raw = fe._safe_float(m_rows[0].get("avg_jobs_per_bench_active"), 0.0) if m_rows else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(vals[i], y_eps) * 1.08,
                f"{raw:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"{backend} - {size} qubits", fontsize=13)
        ax.set_ylabel("Avg jobs per bench", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=32, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            ncol=max(1, len(legend_labels)),
            fontsize=10,
            loc="upper center",
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
    else:
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
    parser.add_argument("--timebreak-methods", default="QOS,QOSE")
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
        _replace(
            _pick_pdf(brk_paths, "qose_time_breakdown"),
            out_dir / "qose_time_breakdown_torino_marrakesh.pdf",
        )
        _replace(
            _pick_pdf(brk_paths, "mitigation_stage_breakdown_qos_qose"),
            out_dir / "mitigation_stage_breakdown_qos_qose.pdf",
        )
        _plot_avg_jobs_from_strict_csv(
            job_counts_strictdry,
            out_dir / "panel_avg_jobs_per_bench.pdf",
            sizes=sizes,
        )
        for p in brk_paths:
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
    ]:
        path = out_dir / name
        print(path)


if __name__ == "__main__":
    main()
