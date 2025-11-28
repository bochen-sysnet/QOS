#!/usr/bin/env python3
"""
Simulation-only experiment runner for QOS.

What it does:
  - Loads benchmark circuits (evaluation/benchmarks) for a given qubit range.
  - Uses fake IBM backends (evaluation/qpus_available.yml) to:
      * Estimate baseline fidelity (gate/readout error heuristic).
      * Apply the QOS error mitigator (gate/wire cutting, qubit freezing/reuse)
        and re-estimate fidelity.
      * Compute multiprogramming compatibility scores between circuits.
      * Estimate simple scheduling trade-offs (fidelity vs. ETA) across backends.
  - Writes fresh CSVs and PDFs under evaluation/out/sim_experiments.

No Redis or real QPUs are used.

Usage:
  python3 evaluation/run_sim_experiments.py --benchmarks qaoa_r3 ghz --min-qubits 8 --max-qubits 14 --plots
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Ensure repo root and evaluation on sys.path when run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from evaluation.benchmarks import circuits as bench_circuits
from qos.error_mitigator.analyser import BasicAnalysisPass, SupermarqFeaturesAnalysisPass
from qos.error_mitigator.run import ErrorMitigator
from qos.types.types import Qernel
from qiskit import QuantumCircuit, transpile
from FrozenQubits import helper_FrozenQubits as fq

OUT_DIR = Path(__file__).resolve().parent / "out" / "sim_experiments"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_fake_qpus(cfg: Path) -> list:
    cfg_data = yaml.safe_load(cfg.read_text())
    out = []
    for qpu in cfg_data.get("qpus", []):
        if qpu.get("type") != "fake":
            continue
        name = qpu["name"]
        backend_cls = None
        for mod in ("qiskit_ibm_runtime.fake_provider", "qiskit.providers.fake_provider"):
            try:
                fp = importlib.import_module(mod)
            except Exception:
                continue
            backend_cls = getattr(fp, name, None)
            if backend_cls is None and name.endswith("V2"):
                backend_cls = getattr(fp, name.replace("V2", ""), None)
            if backend_cls is not None:
                break
        if backend_cls is None:
            raise AttributeError(f"Fake backend {name} not found in available fake providers")
        out.append(backend_cls())
    if not out:
        raise SystemExit("No fake QPUs found in evaluation/qpus_available.yml")
    return out


def heuristic_fidelity(circ: QuantumCircuit, backend) -> float:
    # Transpile to map onto the backend topology and basis.
    tcirc = transpile(
        circ,
        backend=backend,
        optimization_level=1,
        scheduling_method="alap",
    )
    props = backend.properties()
    fid = 1.0
    for inst, qargs, _ in tcirc.data:
        name = inst.name
        if name in ("measure", "barrier", "delay"):
            if name == "measure":
                qubit = tcirc.find_bit(qargs[0]).index
                fid *= 1 - props.readout_error(qubit)
            continue
        try:
            qubits = [tcirc.find_bit(q).index for q in qargs]
            err = props.gate_error(name, qubits)
            fid *= 1 - err
        except Exception:
            fid *= 0.999
    return max(0.0, fid)


def estimate_eta(circ: QuantumCircuit, backend) -> float:
    props = backend.properties()
    tcirc = transpile(circ, backend=backend, optimization_level=1)
    total = 0.0
    for inst, qargs, _ in tcirc.data:
        if inst.name == "measure":
            total += props.readout_length(tcirc.find_bit(qargs[0]).index)
        else:
            try:
                qubits = [tcirc.find_bit(q).index for q in qargs]
                total += props.gate_length(inst.name, qubits)
            except Exception:
                total += 1e-7
    shots = min(getattr(backend, "max_shots", 8192), 8192)
    return total * shots


def analyse_metadata(qc: QuantumCircuit) -> dict:
    q = Qernel(qc.copy())
    BasicAnalysisPass().run(q)
    SupermarqFeaturesAnalysisPass().run(q)
    return q.get_metadata()


def effective_util(q1: QuantumCircuit, q2: QuantumCircuit, backend) -> float:
    depth1, depth2 = q1.depth(), q2.depth()
    dq = max(depth1, depth2) or 1
    nb = backend.configuration().num_qubits
    spatial = max(q1.num_qubits, q2.num_qubits) / nb
    temporal = (depth1 / dq) * (q1.num_qubits / nb) + (depth2 / dq) * (
        q2.num_qubits / nb
    )
    return spatial + temporal


def compatibility_score(m1: dict, m2: dict, util: float) -> float:
    ent = (1 - m1.get("entanglement_ratio", 0)) * (1 - m2.get("entanglement_ratio", 0))
    meas = (1 - m1.get("measurement", 0)) * (1 - m2.get("measurement", 0))
    par = (1 - m1.get("parallelism", 0)) * (1 - m2.get("parallelism", 0))
    return (util + ent + meas + par) / 4


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    bench: str
    qubits: int
    backend: str
    fidelity: float
    fidelity_mitigated: float
    eta: float
    eta_mitigated: float


@dataclass
class CompatResult:
    bench_a: str
    bench_b: str
    backend: str
    util: float
    score: float


@dataclass
class ScheduleResult:
    bench: str
    backend_pick: str
    backend_alt: str
    fid_pick: float
    fid_alt: float
    eta_pick: float
    eta_alt: float
    score: float


# --------------------------------------------------------------------------- #
# Core experiment
# --------------------------------------------------------------------------- #


def run_experiments(
    circuits: list[QuantumCircuit], backends: list, methods: list[str]
) -> tuple[list[RunResult], list[CompatResult], list[ScheduleResult]]:
    # Drop QF if dimod is unavailable
    if fq.dimod is None and "QF" in methods:
        methods = [m for m in methods if m != "QF"]
    mitigator = ErrorMitigator(methods=methods)
    run_results: list[RunResult] = []
    compat_results: list[CompatResult] = []
    sched_results: list[ScheduleResult] = []

    # Baseline + mitigated per backend
    for qc in circuits:
        qname = qc.name or "bench"
        for backend in backends:
            fid_base = heuristic_fidelity(qc, backend)
            eta_base = estimate_eta(qc, backend)

            if methods:
                try:
                    mitigated_q = mitigator.run(Qernel(qc.copy()))
                    mqc = mitigated_q.get_circuit()
                    fid_mit = heuristic_fidelity(mqc, backend)
                    eta_mit = estimate_eta(mqc, backend)
                except Exception:
                    fid_mit, eta_mit = fid_base, eta_base
            else:
                fid_mit, eta_mit = fid_base, eta_base
            run_results.append(
                RunResult(
                    bench=qname,
                    qubits=qc.num_qubits,
                    backend=backend.name,
                    fidelity=fid_base,
                    fidelity_mitigated=fid_mit,
                    eta=eta_base,
                    eta_mitigated=eta_mit,
                )
            )

    # Compatibility (single backend for simplicity)
    backend = backends[0]
    metadata = [analyse_metadata(qc) for qc in circuits]
    for i, qa in enumerate(circuits):
        for j, qb in enumerate(circuits[i + 1 :], start=i + 1):
            util = effective_util(qa, qb, backend)
            score = compatibility_score(metadata[i], metadata[j], util)
            compat_results.append(
                CompatResult(
                    bench_a=qa.name or f"a{i}",
                    bench_b=qb.name or f"b{j}",
                    backend=backend.name,
                    util=util,
                    score=score,
                )
            )

    # Scheduling trade-off (pick best vs second-best backend)
    for qc in circuits:
        preds = []
        for backend in backends:
            fid = heuristic_fidelity(qc, backend)
            eta = estimate_eta(qc, backend)
            preds.append((backend, fid, eta))
        preds.sort(key=lambda x: x[1], reverse=True)
        if len(preds) < 2:
            continue
        pick, alt = preds[0], preds[1]
        fid_weight = 0.7
        try:
            score = fid_weight * (alt[1] / pick[1] - 1) - (1 - fid_weight) * (
                alt[2] / pick[2] - 1
            )
        except ZeroDivisionError:
            score = math.nan
        sched_results.append(
            ScheduleResult(
                bench=qc.name or "bench",
                backend_pick=pick[0].name,
                backend_alt=alt[0].name,
                fid_pick=pick[1],
                fid_alt=alt[1],
                eta_pick=pick[2],
                eta_alt=alt[2],
                score=score,
            )
        )

    return run_results, compat_results, sched_results


# --------------------------------------------------------------------------- #
# CSV and plotting
# --------------------------------------------------------------------------- #


def write_csv(path: Path, rows: Iterable[dataclass]) -> None:
    data = [asdict(r) for r in rows]
    if not data:
        return
    keys = list(data[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write(",".join(keys) + "\n")
        for row in data:
            f.write(",".join(str(row[k]) for k in keys) + "\n")


def plot_fidelity(run_results: list[RunResult], out_dir: Path) -> None:
    if not run_results:
        return
    plt.figure(figsize=(8, 4))
    benches = sorted(set(r.bench for r in run_results))
    for b in benches:
        subset = [r for r in run_results if r.bench == b]
        xs = [r.backend for r in subset]
        base = [r.fidelity for r in subset]
        mit = [r.fidelity_mitigated for r in subset]
        x = np.arange(len(xs))
        width = 0.35
        plt.bar(x - width / 2, base, width=width, label=f"{b} baseline")
        plt.bar(x + width / 2, mit, width=width, label=f"{b} mitigated")
    plt.xticks(x, xs, rotation=45, ha="right")
    plt.ylabel("Predicted fidelity")
    plt.title("Fidelity before/after mitigation")
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "fidelity_mitigation.pdf")
    plt.close()


def plot_compat(compat_results: list[CompatResult], out_dir: Path) -> None:
    if not compat_results:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(
        [r.util for r in compat_results],
        [r.score for r in compat_results],
        alpha=0.7,
    )
    plt.xlabel("Effective utilization")
    plt.ylabel("Compatibility score")
    plt.title("Multiprogramming compatibility")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "compatibility.pdf")
    plt.close()


def plot_scheduler(sched_results: list[ScheduleResult], out_dir: Path) -> None:
    if not sched_results:
        return
    plt.figure(figsize=(6, 4))
    deltas = [
        math.log10(r.eta_alt / r.eta_pick) if r.eta_pick and r.eta_alt else 0
        for r in sched_results
    ]
    fid_diffs = [r.fid_alt - r.fid_pick for r in sched_results]
    plt.scatter(deltas, fid_diffs, alpha=0.7)
    plt.xlabel("log10(ETA_alt / ETA_pick)")
    plt.ylabel("Fidelity alt - Fidelity pick")
    plt.title("Scheduling trade-off")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "scheduler_tradeoff.pdf")
    plt.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qpu-cfg", type=Path, default=Path(__file__).resolve().parent / "qpus_available.yml")
    p.add_argument("--benchmarks", nargs="+", default=["qaoa_r3", "ghz", "bv"])
    p.add_argument("--min-qubits", type=int, default=8)
    p.add_argument("--max-qubits", type=int, default=14)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument(
        "--mitigation-methods",
        nargs="+",
        default=[],
        help="Subset of mitigation passes to enable. Default disables mitigation to avoid optional deps.",
    )
    p.add_argument("--plots", action="store_true", help="Also generate PDFs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    backends = load_fake_qpus(args.qpu_cfg)

    circuits: list[QuantumCircuit] = []
    for bench in args.benchmarks:
        qcs = bench_circuits.get_circuits(bench, (args.min_qubits, args.max_qubits + 1))
        for qc in qcs:
            qc.name = qc.name or bench
        circuits.extend(qcs)
    if not circuits:
        raise SystemExit("No circuits loaded; check benchmark names/qubit range.")

    run_results, compat_results, sched_results = run_experiments(
        circuits, backends, args.mitigation_methods
    )

    out_dir = args.out
    write_csv(out_dir / "run_results.csv", run_results)
    write_csv(out_dir / "compatibility.csv", compat_results)
    write_csv(out_dir / "scheduler.csv", sched_results)

    if args.plots:
        plot_fidelity(run_results, out_dir)
        plot_compat(compat_results, out_dir)
        plot_scheduler(sched_results, out_dir)

    print(f"Simulation outputs written to {out_dir}")


if __name__ == "__main__":
    main()
