import argparse
import csv
import os
import sys
import time
from multiprocessing import Process, Value
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.full_eval import _load_qasm_circuit
from qos.error_mitigator.optimiser import GVOptimalDecompositionPass, OptimalWireCuttingPass
from qos.types.types import Qernel


PLOT_DIR = Path(__file__).resolve().parent
DATA_DIR = PLOT_DIR / "data"
FIGURES_DIR = PLOT_DIR / "figures"


def _import_matplotlib():
    import matplotlib.pyplot as plt  # type: ignore

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _measure_cost(pass_cls, qernel, size_to_reach, repeats, timeout_sec):
    total = 0.0
    for _ in range(repeats):
        box = Value("i", 1000)
        t0 = time.perf_counter()
        proc = Process(target=pass_cls(size_to_reach).cost, args=(qernel, box))
        proc.start()
        proc.join(timeout_sec)
        if proc.is_alive():
            proc.terminate()
            proc.join()
        total += time.perf_counter() - t0
    return total / max(1, repeats)


def _record_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["sweep"],
        row["bench"],
        int(row["qubits"]),
        int(row["size_to_reach"]),
        row["method"],
        int(row["repeats"]),
        int(row["cost_timeout_sec"]),
        int(row["fixed_size_to_reach"]),
        int(row["fixed_qubits"]),
    )


def _parse_sizes(args):
    if args.sizes:
        return [int(s) for s in args.sizes.split(",") if s.strip()]
    return list(range(args.size_min, args.size_max + 1, args.size_step))


def _parse_sizes_to_reach(args):
    if args.size_to_reach_values:
        return [int(s) for s in args.size_to_reach_values.split(",") if s.strip()]
    return list(range(args.size_to_reach_min, args.size_to_reach_max + 1))


def _write_records(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sweep",
                "bench",
                "qubits",
                "size_to_reach",
                "method",
                "elapsed_sec",
                "repeats",
                "cost_timeout_sec",
                "fixed_size_to_reach",
                "fixed_qubits",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _plot(records, fixed_str, fixed_qubits, out_path: Path) -> None:
    plt = _import_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4), sharey=False)

    colors = {"GV": "#1f77b4", "WC": "#d55e00"}
    labels = {"GV": "Gate Virtualization", "WC": "Wire Cutting"}

    fixed_rows = [row for row in records if row["sweep"] == "fixed_size_to_reach"]
    sweep_rows = [row for row in records if row["sweep"] == "fixed_qubits"]

    sizes = sorted({int(row["qubits"]) for row in fixed_rows})
    sizes_to_reach = sorted({int(row["size_to_reach"]) for row in sweep_rows})

    for method in ("GV", "WC"):
        y_vals = [
            float(
                next(
                    row["elapsed_sec"]
                    for row in fixed_rows
                    if row["method"] == method and int(row["qubits"]) == size
                )
            )
            for size in sizes
        ]
        axes[0].plot(
            sizes,
            y_vals,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=colors[method],
            label=labels[method],
        )

    axes[0].set_title(f"Fixed target partition size = {fixed_str}")
    axes[0].set_xlabel("Circuit Size (qubits)")
    axes[0].set_ylabel("Mitigation Time (s)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    for method in ("GV", "WC"):
        y_vals = [
            float(
                next(
                    row["elapsed_sec"]
                    for row in sweep_rows
                    if row["method"] == method and int(row["size_to_reach"]) == size_to_reach
                )
            )
            for size_to_reach in sizes_to_reach
        ]
        axes[1].plot(
            sizes_to_reach,
            y_vals,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=colors[method],
            label=labels[method],
        )

    axes[1].set_title(f"Fixed circuit size = {fixed_qubits} qubits")
    axes[1].set_xlabel("Target Partition Size")
    axes[1].set_ylabel("Mitigation Time (s)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        handlelength=2.2,
        columnspacing=1.6,
        borderaxespad=0.2,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="qaoa_r3")
    parser.add_argument("--sizes", default="12,14,16,18,20,22,24", help="Comma-separated circuit sizes.")
    parser.add_argument("--size-min", type=int, default=12)
    parser.add_argument("--size-max", type=int, default=24)
    parser.add_argument("--size-step", type=int, default=2)
    parser.add_argument(
        "--size-to-reach-values",
        default="7,8,9,10,11,12,13,14,15",
        help="Comma-separated size-to-reach values.",
    )
    parser.add_argument("--size-to-reach-min", type=int, default=3)
    parser.add_argument("--size-to-reach-max", type=int, default=12)
    parser.add_argument("--fixed-size-to-reach", type=int, default=7)
    parser.add_argument("--fixed-qubits", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cost-timeout-sec", type=int, default=60)
    parser.add_argument("--clingo-timeout-sec", type=int, default=0)
    parser.add_argument("--max-partition-tries", type=int, default=0)
    parser.add_argument("--out-dir", default=str(PLOT_DIR))
    parser.add_argument("--tag", default="")
    parser.add_argument("--input-csv", default="", help="Replot from an existing CSV.")
    parser.add_argument("--out-pdf", default="", help="Optional explicit PDF output path.")
    parser.add_argument("--out-csv", default="", help="Optional explicit CSV output path.")
    args = parser.parse_args()

    if args.input_csv:
        input_csv = Path(args.input_csv)
        rows = _read_records(input_csv)
        fixed_str = int(next(row["size_to_reach"] for row in rows if row["sweep"] == "fixed_size_to_reach"))
        fixed_qubits = int(next(row["qubits"] for row in rows if row["sweep"] == "fixed_qubits"))
        out_path = Path(args.out_pdf) if args.out_pdf else FIGURES_DIR / input_csv.with_suffix(".pdf").name
        _plot(rows, fixed_str, fixed_qubits, out_path)
        print(f"Wrote timing figure: {out_path}")
        return

    os.environ["QVM_CLINGO_TIMEOUT_SEC"] = str(args.clingo_timeout_sec)
    os.environ["QVM_MAX_PARTITION_TRIES"] = str(args.max_partition_tries)

    sizes = _parse_sizes(args)
    sizes_to_reach = _parse_sizes_to_reach(args)
    fixed_str = args.fixed_size_to_reach
    fixed_qubits = args.fixed_qubits
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag.strip() else ""
    data_dir = out_dir / "data"
    figures_dir = out_dir / "figures"
    out_pdf = Path(args.out_pdf) if args.out_pdf else figures_dir / f"cut_pass_timing_{args.bench}{tag}.pdf"
    out_csv = Path(args.out_csv) if args.out_csv else data_dir / f"cut_pass_timing_{args.bench}{tag}.csv"

    rows: list[dict[str, object]] = []
    if out_csv.exists():
        rows = [dict(row) for row in _read_records(out_csv)]
    existing_keys = {_record_key(row) for row in rows}

    for size in sizes:
        print(f"[timing] size={size} fixed_size_to_reach={fixed_str}", flush=True)
        qc = None
        q = None
        for method, pass_cls in (("GV", GVOptimalDecompositionPass), ("WC", OptimalWireCuttingPass)):
            row = {
                "sweep": "fixed_size_to_reach",
                "bench": args.bench,
                "qubits": size,
                "size_to_reach": fixed_str,
                "method": method,
                "elapsed_sec": 0.0,
                "repeats": args.repeats,
                "cost_timeout_sec": args.cost_timeout_sec,
                "fixed_size_to_reach": fixed_str,
                "fixed_qubits": fixed_qubits,
            }
            key = _record_key(row)
            if key in existing_keys:
                print(f"[resume] skip size={size} method={method}", flush=True)
                continue
            if q is None:
                qc = _load_qasm_circuit(args.bench, size)
                q = Qernel(qc.copy())
            t0 = time.perf_counter()
            value = _measure_cost(pass_cls, q, fixed_str, args.repeats, args.cost_timeout_sec)
            print(f"[timing]  {method} sec={value:.2f} total={time.perf_counter() - t0:.2f}", flush=True)
            row["elapsed_sec"] = value
            rows.append(row)
            existing_keys.add(key)
            _write_records(out_csv, rows)

    qc = None
    q = None
    for size_to_reach in sizes_to_reach:
        print(f"[timing] fixed_qubits={fixed_qubits} size_to_reach={size_to_reach}", flush=True)
        for method, pass_cls in (("GV", GVOptimalDecompositionPass), ("WC", OptimalWireCuttingPass)):
            row = {
                "sweep": "fixed_qubits",
                "bench": args.bench,
                "qubits": fixed_qubits,
                "size_to_reach": size_to_reach,
                "method": method,
                "elapsed_sec": 0.0,
                "repeats": args.repeats,
                "cost_timeout_sec": args.cost_timeout_sec,
                "fixed_size_to_reach": fixed_str,
                "fixed_qubits": fixed_qubits,
            }
            key = _record_key(row)
            if key in existing_keys:
                print(f"[resume] skip size_to_reach={size_to_reach} method={method}", flush=True)
                continue
            if q is None:
                qc = _load_qasm_circuit(args.bench, fixed_qubits)
                q = Qernel(qc.copy())
            t0 = time.perf_counter()
            value = _measure_cost(pass_cls, q, size_to_reach, args.repeats, args.cost_timeout_sec)
            print(f"[timing]  {method} sec={value:.2f} total={time.perf_counter() - t0:.2f}", flush=True)
            row["elapsed_sec"] = value
            rows.append(row)
            existing_keys.add(key)
            _write_records(out_csv, rows)

    _write_records(out_csv, rows)
    _plot(rows, fixed_str, fixed_qubits, out_pdf)
    print(f"Wrote timing figure: {out_pdf}")
    print(f"Wrote timing data: {out_csv}")


if __name__ == "__main__":
    main()
