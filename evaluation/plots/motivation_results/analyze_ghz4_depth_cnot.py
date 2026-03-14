#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


QASM_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "ghz" / "4.qasm"
FIG_DIR = Path(__file__).resolve().parent / "figures"
OUT_CIRCUIT_PDF = FIG_DIR / "ghz4_circuit_example.pdf"


def _save_circuit_plot(qc_nom: QuantumCircuit) -> None:
    """Save GHZ-4 circuit figure with on-figure depth/CNOT derivation."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    circuit_text = str(qc_nom.draw("text", fold=100))
    for i in range(qc_nom.num_qubits):
        circuit_text = circuit_text.replace(f"q_{i}:", "|0>:")
    depth = int(qc_nom.depth())
    cnot_count = int(qc_nom.count_ops().get("cx", 0))

    plt.rcParams.update(
        {
            "font.size": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_circuit, ax_note) = plt.subplots(
        1,
        2,
        figsize=(12.6, 3.2),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    ax_circuit.axis("off")
    ax_circuit.text(
        0.01,
        0.95,
        circuit_text,
        transform=ax_circuit.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=14,
    )

    ax_note.axis("off")
    box = FancyBboxPatch(
        (0.02, 0.52),
        0.96,
        0.42,
        boxstyle="round,pad=0.03",
        linewidth=1.0,
        edgecolor="#7a7a7a",
        facecolor="#f5f5f5",
        transform=ax_note.transAxes,
    )
    ax_note.add_patch(box)
    ax_note.text(
        0.06,
        0.86,
        "Depth derivation",
        transform=ax_note.transAxes,
        ha="left",
        va="center",
        fontsize=15,
        weight="bold",
    )
    ax_note.text(
        0.06,
        0.74,
        f"Depth = # sequential layers = {depth}",
        transform=ax_note.transAxes,
        ha="left",
        va="center",
        fontsize=14,
    )
    ax_note.text(
        0.06,
        0.64,
        "  = 1 (H layer) + 3 (CX chain)",
        transform=ax_note.transAxes,
        ha="left",
        va="center",
        fontsize=14,
    )
    ax_note.text(
        0.06,
        0.35,
        f"#CNOT derivation\n#CNOT = count(CX) = {cnot_count}",
        transform=ax_note.transAxes,
        ha="left",
        va="center",
        fontsize=14,
        weight="bold",
    )

    fig.tight_layout()
    fig.savefig(OUT_CIRCUIT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    qc = QuantumCircuit.from_qasm_file(str(QASM_PATH))
    qc_nom = qc.remove_final_measurements(inplace=False)
    _save_circuit_plot(qc_nom)

    ops_with_measure = qc.count_ops()
    ops_no_measure = qc_nom.count_ops()
    depth_with_measure = qc.depth()
    depth_no_measure = qc_nom.depth()
    cnot_count = int(ops_no_measure.get("cx", 0))

    print(f"Benchmark circuit: {QASM_PATH}")
    print()
    print("GHZ-4 depth/CNOT analysis (excluding final measurements):")
    print(f"- Depth = {depth_no_measure}")
    print(f"- #CNOT = {cnot_count}")
    print(f"- Ops (no measure) = {dict(ops_no_measure)}")
    print()
    print("Reference (including final measurements):")
    print(f"- Depth = {depth_with_measure}")
    print(f"- Ops (with measure) = {dict(ops_with_measure)}")
    print()

    dag = circuit_to_dag(qc_nom)
    print("Layer-by-layer derivation:")
    for i, layer in enumerate(dag.layers(), start=1):
        parts = []
        for node in layer["graph"].op_nodes():
            qids = [q._index for q in node.qargs]
            parts.append(f"{node.name}({','.join(map(str, qids))})")
        print(f"- Layer {i}: " + " || ".join(parts))

    print()
    print("Derivation summary:")
    print("- Layer 1 is H on q0.")
    print("- Layers 2-4 are serial CNOT chain: cx(0,1), cx(1,2), cx(2,3).")
    print("- Therefore depth = 1 + 3 = 4, and #CNOT = 3.")
    print()
    print(f"Circuit figure saved to: {OUT_CIRCUIT_PDF}")


if __name__ == "__main__":
    main()
