#!/usr/bin/env python3
"""
ibm_qpu_logger.py

Log IBM Quantum backend status + calibration-derived metrics every N seconds (default 10 min)
until Ctrl+C, appending all records to a CSV.

Usage examples:
  python ibm_qpu_logger.py --out ibm_qpu_metrics.csv
  python ibm_qpu_logger.py --backends ibm_marrakesh ibm_torino ibm_fez --out mylog.csv
  python ibm_qpu_logger.py --interval 600 --out ibm.csv

Notes:
- Requires: pip install qiskit-ibm-runtime
- Assumes your IBM Quantum account is configured for QiskitRuntimeService().
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from qiskit_ibm_runtime import QiskitRuntimeService


STOP_REQUESTED = False


def _handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_utc_iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat(timespec="seconds")
    return str(value)


def safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def median(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def try_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def gate_is_2q(gate_name: str) -> bool:
    # IBM devices may use 'cx' or 'ecr' as native 2q gate; others possible.
    return gate_name in {"cx", "ecr", "cz", "iswap", "rzz"}


def gate_is_1q(gate_name: str) -> bool:
    # Common 1q gates; there are more, but this covers typical calibration props.
    return gate_name in {"x", "sx", "rz", "id", "u1", "u2", "u3"}


def extract_gate_errors(properties) -> Dict[str, Optional[float]]:
    """
    From backend.properties(), compute median gate error for:
      - 2Q gates (cx/ecr/cz/...)
      - 1Q gates (x/sx/...)
    Also report counts to know sample size.
    """
    twoq_errs: List[float] = []
    oneq_errs: List[float] = []
    twoq_count = 0
    oneq_count = 0

    # properties.gates is a list of Gate objects, each has .gate, .qubits, .parameters
    gates = safe_getattr(properties, "gates", []) or []
    for g in gates:
        gname = safe_getattr(g, "gate", None)
        qubits = safe_getattr(g, "qubits", None) or []
        params = safe_getattr(g, "parameters", None) or []

        # Find parameter named 'gate_error' if present
        gate_error = None
        for p in params:
            if safe_getattr(p, "name", None) == "gate_error":
                gate_error = try_float(safe_getattr(p, "value", None))
                break

        if gate_error is None:
            continue

        if len(qubits) == 2 and gname and gate_is_2q(str(gname)):
            twoq_errs.append(gate_error)
            twoq_count += 1
        elif len(qubits) == 1 and gname and gate_is_1q(str(gname)):
            oneq_errs.append(gate_error)
            oneq_count += 1
        elif len(qubits) == 2 and gname:
            # If it's 2Q but not in our whitelist, still count toward 2Q bucket
            twoq_errs.append(gate_error)
            twoq_count += 1
        elif len(qubits) == 1 and gname:
            # If it's 1Q but not in our whitelist, still count toward 1Q bucket
            oneq_errs.append(gate_error)
            oneq_count += 1

    return {
        "median_2q_gate_error": median(twoq_errs),
        "median_1q_gate_error": median(oneq_errs),
        "count_2q_gates_with_error": twoq_count,
        "count_1q_gates_with_error": oneq_count,
    }


def extract_qubit_metrics(properties) -> Dict[str, Optional[float]]:
    """
    From properties.qubits (per-qubit parameter lists), compute:
      - median readout error (if available)
      - median T1, median T2 (if available)
    We also keep counts.
    """
    readout_errs: List[float] = []
    t1s: List[float] = []
    t2s: List[float] = []

    qubits = safe_getattr(properties, "qubits", []) or []
    for q_params in qubits:
        # q_params is a list of Nduv-like parameter objects with .name, .value, .unit
        d = {}
        for p in q_params:
            name = safe_getattr(p, "name", None)
            val = safe_getattr(p, "value", None)
            if name is not None:
                d[str(name)] = val

        # readout error naming can vary; try a few common ones
        ro = None
        for k in ("readout_error", "prob_meas0_prep1", "prob_meas1_prep0"):
            if k in d:
                ro = try_float(d.get(k))
                # If using prob_meas*, that's not exactly readout_error; but it's still useful.
                break
        if ro is not None:
            readout_errs.append(ro)

        t1 = try_float(d.get("T1"))
        t2 = try_float(d.get("T2"))
        if t1 is not None:
            t1s.append(t1)
        if t2 is not None:
            t2s.append(t2)

    return {
        "median_readout_metric": median(readout_errs),  # may be readout_error or prob_meas*
        "count_readout_metric": len(readout_errs),
        "median_T1": median(t1s),
        "count_T1": len(t1s),
        "median_T2": median(t2s),
        "count_T2": len(t2s),
    }


def extract_backend_config(backend) -> Dict[str, Any]:
    """
    Pull what we can from backend.configuration() and backend.options().
    CLOPS may or may not exist depending on backend object.
    """
    cfg = safe_call(backend.configuration, default=None)
    opts = safe_getattr(backend, "options", None)

    out: Dict[str, Any] = {}

    if cfg is not None:
        out["backend_version"] = safe_getattr(cfg, "backend_version", None)
        out["num_qubits"] = safe_getattr(cfg, "n_qubits", None) or safe_getattr(cfg, "num_qubits", None) or safe_getattr(backend, "num_qubits", None)
        out["basis_gates"] = ";".join(list(safe_getattr(cfg, "basis_gates", []) or [])) or None
        out["coupling_map_len"] = len(safe_getattr(cfg, "coupling_map", []) or [])
        out["max_shots"] = safe_getattr(cfg, "max_shots", None)
        # CLOPS sometimes appears as an attribute on configuration
        out["clops"] = safe_getattr(cfg, "clops", None)
        out["processor_type"] = safe_getattr(cfg, "processor_type", None)
        out["simulator"] = safe_getattr(cfg, "simulator", None)

    if out.get("num_qubits") is None:
        out["num_qubits"] = safe_getattr(backend, "num_qubits", None)

    # Options are dynamic / runtime; just store a small useful subset
    if opts is not None:
        out["default_shots"] = safe_getattr(opts, "shots", None)

    return out


def rowify(value: Any) -> Any:
    """Convert non-primitive to stable string for CSV."""
    if value is None:
        return ""
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def collect_one_backend(service: QiskitRuntimeService, backend_name: str) -> Dict[str, Any]:
    backend = service.backend(backend_name)

    ts = utc_now_iso()

    status = safe_call(backend.status, default=None)
    props = safe_call(backend.properties, default=None)

    row: Dict[str, Any] = {
        "timestamp_utc": ts,
        "backend_name": backend_name,
    }

    # Status / queue
    if status is not None:
        row["operational"] = safe_getattr(status, "operational", None)
        row["pending_jobs"] = safe_getattr(status, "pending_jobs", None)
        row["status_msg"] = safe_getattr(status, "status_msg", None)

    # Config / static-ish
    row.update(extract_backend_config(backend))

    # Calibration timestamp and derived stats
    if props is not None:
        row["last_update_date_utc"] = to_utc_iso(safe_getattr(props, "last_update_date", None))
        row.update(extract_gate_errors(props))
        row.update(extract_qubit_metrics(props))

    return row


def ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    # If file doesn't exist or is empty, write header.
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


def append_rows(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    ensure_csv_header(path, fieldnames)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            w.writerow({k: rowify(r.get(k)) for k in fieldnames})


def list_backends(service: QiskitRuntimeService) -> List[str]:
    # Get backends visible to the account; filter to real devices by default.
    bks = safe_call(service.backends, default=[]) or []
    names: List[str] = []
    for b in bks:
        name = safe_getattr(b, "name", None)
        cfg = safe_call(b.configuration, default=None)
        is_sim = bool(safe_getattr(cfg, "simulator", False)) if cfg is not None else bool(safe_getattr(b, "simulator", False))
        if name and not is_sim:
            names.append(str(name))
    names.sort()
    return names


def build_fieldnames() -> List[str]:
    # Stable column order
    return [
        "timestamp_utc",
        "backend_name",
        "operational",
        "status_msg",
        "pending_jobs",
        "backend_version",
        "num_qubits",
        "basis_gates",
        "coupling_map_len",
        "max_shots",
        "default_shots",
        "clops",
        "processor_type",
        "simulator",
        "last_update_date_utc",
        "median_2q_gate_error",
        "count_2q_gates_with_error",
        "median_1q_gate_error",
        "count_1q_gates_with_error",
        "median_readout_metric",
        "count_readout_metric",
        "median_T1",
        "count_T1",
        "median_T2",
        "count_T2",
    ]


def build_paths(out_dir: str, out_csv: str | None) -> tuple[Path, Path]:
    root = Path(out_dir)
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(out_csv) if out_csv else data_dir / "ibm_qpu_metrics.csv"
    figure_path = fig_dir / "ibm_qpu_metrics_over_time.pdf"
    return csv_path, figure_path


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt, mdates


def plot_metrics_panels(csv_path: Path, out_pdf: Path) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    metric_keys = [
        ("pending_jobs", "Pending Jobs"),
        ("median_2q_gate_error", "Median 2Q Gate Error"),
        ("median_1q_gate_error", "Median 1Q Gate Error"),
        ("median_readout_metric", "Median Readout Metric"),
    ]
    rows_by_backend: Dict[str, Dict[str, list[tuple[datetime, float]]]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            backend = str(row.get("backend_name", "")).strip()
            ts_raw = str(row.get("timestamp_utc", "")).strip()
            if not backend or not ts_raw:
                continue
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                continue
            per_backend = rows_by_backend.setdefault(backend, {key: [] for key, _label in metric_keys})
            for key, _label in metric_keys:
                raw = str(row.get(key, "")).strip()
                if not raw:
                    continue
                try:
                    value = float(raw)
                except Exception:
                    continue
                per_backend[key].append((ts, value))

    if not rows_by_backend:
        return

    plt, mdates = _import_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), constrained_layout=True)
    axes = axes.flatten()

    backend_names = sorted(rows_by_backend)
    cmap = plt.get_cmap("tab10")
    colors = {backend: cmap(idx % 10) for idx, backend in enumerate(backend_names)}

    for ax, (key, ylabel) in zip(axes, metric_keys):
        for backend in backend_names:
            points = rows_by_backend[backend].get(key, [])
            if not points:
                continue
            points.sort(key=lambda item: item[0])
            xs = [ts for ts, _value in points]
            ys = [value for _ts, value in points]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=backend,
                color=colors[backend],
            )
        ax.set_xlabel("UTC Time")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log IBM Quantum backend metrics to CSV periodically.")
    p.add_argument("--out-dir", type=str, default="evaluation/plots/ibm_qpu_logger", help="Output root directory containing data/ and figures/.")
    p.add_argument("--out", type=str, default="", help="Optional explicit CSV path (appended). Overrides default data path under --out-dir.")
    p.add_argument("--interval", type=int, default=600, help="Logging interval in seconds (default 600 = 10 min).")
    p.add_argument("--max-iterations", type=int, default=0, help="Stop after this many logging iterations; 0 means run until interrupted.")
    p.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help="Backend names to track. If omitted, logs all visible backends.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    service = QiskitRuntimeService()
    csv_path, figure_pdf = build_paths(args.out_dir, args.out or None)

    if args.backends and len(args.backends) > 0:
        backend_names = list(args.backends)
    else:
        backend_names = list_backends(service)

    fieldnames = build_fieldnames()

    print(f"[ibm_qpu_logger] logging {len(backend_names)} backends every {args.interval}s to: {csv_path}")
    print(f"[ibm_qpu_logger] backends: {', '.join(backend_names)}")
    print("[ibm_qpu_logger] Press Ctrl+C to stop.")
    if csv_path.exists():
        plot_metrics_panels(csv_path, figure_pdf)
        print(f"[ibm_qpu_logger] refreshed existing figure from: {csv_path}")

    iteration = 0
    while not STOP_REQUESTED:
        if args.max_iterations > 0 and iteration >= args.max_iterations:
            break
        iteration += 1
        rows: List[Dict[str, Any]] = []
        started = time.time()

        for name in backend_names:
            if STOP_REQUESTED:
                break
            try:
                row = collect_one_backend(service, name)
                rows.append(row)
            except Exception as e:
                # Keep going even if one backend fails
                rows.append(
                    {
                        "timestamp_utc": utc_now_iso(),
                        "backend_name": name,
                        "status_msg": f"ERROR: {type(e).__name__}: {e}",
                    }
                )

        append_rows(str(csv_path), fieldnames, rows)
        plot_metrics_panels(csv_path, figure_pdf)

        elapsed = time.time() - started
        print(f"[ibm_qpu_logger] wrote {len(rows)} rows (iter={iteration}) in {elapsed:.1f}s @ {utc_now_iso()}")

        # sleep remaining time
        remaining = args.interval - elapsed
        while remaining > 0 and not STOP_REQUESTED:
            step = min(1.0, remaining)
            time.sleep(step)
            remaining -= step

    if csv_path.exists():
        plot_metrics_panels(csv_path, figure_pdf)
    print(f"[ibm_qpu_logger] stopping; CSV saved at: {csv_path}")
    print(f"[ibm_qpu_logger] figure saved at: {figure_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
