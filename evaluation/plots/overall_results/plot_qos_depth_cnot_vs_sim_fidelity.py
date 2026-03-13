#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SIM_TORINO = ROOT / "plot_data" / "relative_properties_sim_torino.csv"
DEFAULT_SIM_MARRAKESH = ROOT / "plot_data" / "relative_properties_sim_marrakesh.csv"
DEFAULT_OUT = ROOT / "paper_figures" / "qiskit_qos_cutqc_fq_24q_depth_cnot_vs_sim_fidelity_scatter.pdf"

METHOD_SPECS = {
    "qiskit": {
        "label": "Qiskit",
        "depth_col": "baseline_depth",
        "cnot_col": "baseline_nonlocal",
        "fid_col": "baseline_fidelity",
        "marker": "o",
        "color": "#4C72B0",
    },
    "qos": {
        "label": "QOS",
        "depth_col": "qos_depth",
        "cnot_col": "qos_nonlocal",
        "fid_col": "qos_fidelity",
        "marker": "s",
        "color": "#DD8452",
    },
    "cutqc": {
        "label": "CutQC",
        "depth_col": "cutqc_depth",
        "cnot_col": "cutqc_nonlocal",
        "fid_col": "cutqc_fidelity",
        "marker": "^",
        "color": "#55A868",
    },
    "fq": {
        "label": "FrozenQubits",
        "depth_col": "fq_depth",
        "cnot_col": "fq_nonlocal",
        "fid_col": "fq_fidelity",
        "marker": "D",
        "color": "#C44E52",
    },
}


def _to_float(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _append_method_point(
    out: dict[str, list[dict]],
    row: dict,
    backend: str,
    method_key: str,
    depth_col: str,
    cnot_col: str,
    fid_col: str,
) -> None:
    size_v = _to_float(row.get("size"))
    depth_v = _to_float(row.get(depth_col))
    cnot_v = _to_float(row.get(cnot_col))  # nonlocal ~= CNOT count in this data
    fid_v = _to_float(row.get(fid_col))
    if None in (size_v, depth_v, cnot_v, fid_v):
        return
    out[method_key].append(
        {
            "backend": backend,
            "bench": (row.get("bench") or "").strip(),
            "size": int(size_v),
            "depth": float(depth_v),
            "cnot": float(cnot_v),
            "sim_fidelity": float(fid_v),
        }
    )


def _load_rows(path: Path, backend: str, out: dict[str, list[dict]]) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for method_key, spec in METHOD_SPECS.items():
                _append_method_point(
                    out,
                    row,
                    backend=backend,
                    method_key=method_key,
                    depth_col=spec["depth_col"],
                    cnot_col=spec["cnot_col"],
                    fid_col=spec["fid_col"],
                )


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _plot_scatter(
    ax: plt.Axes,
    rows_by_method: dict[str, list[dict]],
    x_key: str,
    x_label: str,
) -> None:
    x_all: list[float] = []
    y_all: list[float] = []

    for method_key, spec in METHOD_SPECS.items():
        pts = [r for r in rows_by_method[method_key] if r["size"] == 24]
        if not pts:
            continue
        x = np.asarray([r[x_key] for r in pts], dtype=float)
        y = np.asarray([r["sim_fidelity"] for r in pts], dtype=float)
        ax.scatter(
            x,
            y,
            s=62,
            alpha=0.85,
            marker=spec["marker"],
            color=spec["color"],
            edgecolor="black",
            linewidth=0.6,
            label=spec["label"],
        )
        x_all.extend(x.tolist())
        y_all.extend(y.tolist())

    x_all_arr = np.asarray(x_all, dtype=float)
    y_all_arr = np.asarray(y_all, dtype=float)
    if x_all_arr.size >= 2:
        fit = np.polyfit(x_all_arr, y_all_arr, 1)
        x_line = np.linspace(np.nanmin(x_all_arr), np.nanmax(x_all_arr), 160)
        y_line = fit[0] * x_line + fit[1]
        ax.plot(x_line, y_line, color="black", linewidth=1.8, linestyle="--")

    r = _pearson_r(x_all_arr, y_all_arr)
    if np.isfinite(r):
        ax.text(
            0.97,
            0.03,
            f"Pearson r = {r:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=20,
            bbox={"facecolor": "white", "edgecolor": "#666666", "alpha": 0.8},
        )

    ax.set_xlabel(x_label)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter correlation between raw depth and simulated fidelity for 24q, "
            "combining Qiskit, QOS, CutQC, and FrozenQubits."
        )
    )
    parser.add_argument("--sim-torino", type=Path, default=DEFAULT_SIM_TORINO)
    parser.add_argument("--sim-marrakesh", type=Path, default=DEFAULT_SIM_MARRAKESH)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.sim_torino.exists():
        raise FileNotFoundError(f"Missing CSV: {args.sim_torino}")
    if not args.sim_marrakesh.exists():
        raise FileNotFoundError(f"Missing CSV: {args.sim_marrakesh}")

    rows_by_method: dict[str, list[dict]] = {k: [] for k in METHOD_SPECS}
    _load_rows(args.sim_torino, "torino", rows_by_method)
    _load_rows(args.sim_marrakesh, "marrakesh", rows_by_method)

    if any(not rows_by_method[k] for k in METHOD_SPECS):
        missing = [k for k in METHOD_SPECS if not rows_by_method[k]]
        raise RuntimeError(f"No valid rows found for methods: {missing}")

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2), constrained_layout=True)
    _plot_scatter(
        ax,
        rows_by_method,
        x_key="depth",
        x_label="Depth",
    )
    ax.set_ylabel("Fidelity")

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure: {args.out_pdf}")


if __name__ == "__main__":
    main()
