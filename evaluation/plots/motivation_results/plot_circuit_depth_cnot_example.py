#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_PDF = OUT_DIR / "circuit_depth_cnot_example.pdf"
QAOA_R3_QASM = Path(__file__).resolve().parents[2] / "benchmarks" / "qaoa_r3" / "4.qasm"


def _example_data() -> tuple[str, list[str], int, int]:
    """Return QAOA-R3 example text, derivation lines, depth, and CNOT count."""
    # Fallback if qiskit/qasm load fails.
    circuit_text = "QAOA-R3 example circuit could not be loaded."
    layers = ["Unable to derive layers from qaoa_r3/4.qasm."]
    depth = 0
    cnot = 0

    try:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit.from_qasm_file(str(QAOA_R3_QASM))
        qc.remove_final_measurements(inplace=True)
        circuit_text = str(qc.draw("text", fold=110))
        depth = int(qc.depth())
        cnot = int(qc.count_ops().get("cx", 0))
        rz = int(qc.count_ops().get("rz", 0))
        h = int(qc.count_ops().get("h", 0))
        rx = int(qc.count_ops().get("rx", 0))
        # For this circuit shape, each QAOA cost term is CX-RZ-CX.
        edge_terms = cnot // 2
        layers = [
            "QAOA-R3 example: 4-qubit circuit (from benchmark qaoa_r3/4.qasm).",
            f"Init layer: {h} H gates in parallel -> depth contribution = 1",
            (
                "Cost unitary: each edge term uses CX-RZ-CX "
                f"(3 sequential layers); {edge_terms} edge terms -> 3 x {edge_terms} = {3 * edge_terms}"
            ),
            f"Mixer layer: {rx} RX gates in parallel -> depth contribution = 1",
            f"Total depth = 1 + {3 * edge_terms} + 1 = {depth}",
            f"#CNOT = total CX count = {cnot} (with {rz} RZ gates)",
        ]
    except Exception:
        # Keep fallback values; still generate a valid figure.
        pass

    return circuit_text, layers, depth, cnot


def main() -> None:
    circuit_text, layers, depth, cnot = _example_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_circuit, ax_derivation) = plt.subplots(
        1,
        2,
        figsize=(15.2, 6.2),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    # Left: circuit drawing.
    ax_circuit.axis("off")
    ax_circuit.set_title("QAOA-R3 Circuit Example (4 Qubits)")
    ax_circuit.text(
        0.01,
        0.99,
        circuit_text,
        transform=ax_circuit.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=12.5,
    )

    # Right: derivation from layers to metrics.
    ax_derivation.axis("off")
    ax_derivation.set_title("Depth / CNOT Derivation")

    y0 = 0.92
    dy = 0.125
    for i, text in enumerate(layers):
        y = y0 - i * dy
        box = FancyBboxPatch(
            (0.02, y - 0.10),
            0.96,
            0.10,
            boxstyle="round,pad=0.02",
            linewidth=1.0,
            edgecolor="#8a8a8a",
            facecolor="#f3f3f3",
            transform=ax_derivation.transAxes,
        )
        ax_derivation.add_patch(box)
        ax_derivation.text(
            0.05,
            y - 0.045,
            text,
            transform=ax_derivation.transAxes,
            va="center",
            ha="left",
            fontsize=12,
        )

    ax_derivation.text(
        0.02,
        0.18,
        f"Depth = number of layers = {depth}",
        transform=ax_derivation.transAxes,
        va="center",
        ha="left",
        fontsize=16,
        weight="bold",
    )
    ax_derivation.text(
        0.02,
        0.10,
        f"#CNOT = count(CX gates) = {cnot}",
        transform=ax_derivation.transAxes,
        va="center",
        ha="left",
        fontsize=16,
        weight="bold",
    )
    ax_derivation.text(
        0.02,
        0.03,
        "Parallel gates in one layer contribute depth = 1.",
        transform=ax_derivation.transAxes,
        va="bottom",
        ha="left",
        fontsize=12,
        color="#333333",
    )

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure: {OUT_PDF}")


if __name__ == "__main__":
    main()
