from qos.error_mitigator.run import compute_gv_cost, compute_wc_cost
from qos.types.types import Qernel


def evolved_cost_search(self, q: Qernel, size_to_reach: int, budget: int):
    metadata = q.get_metadata()
    depth = metadata.get("depth", 0)
    num_qubits = metadata.get("num_qubits", 0)
    num_clbits = metadata.get("num_clbits", 0)
    num_nonlocal_gates = metadata.get("num_nonlocal_gates", 0)
    num_connected_components = metadata.get("num_connected_components", 0)
    number_instructions = metadata.get("number_instructions", 0)
    num_measurements = metadata.get("num_measurements", 0)
    num_cnot_gates = metadata.get("num_cnot_gates", 0)
    program_communication = metadata.get("program_communication", 0.0)
    liveness = metadata.get("liveness", 0.0)
    parallelism = metadata.get("parallelism", 0.0)
    measurement = metadata.get("measurement", 0.0)
    entanglement_ratio = metadata.get("entanglement_ratio", 0.0)
    critical_depth = metadata.get("critical_depth", 0.0)

    # OE_BEGIN
    gv_cost = compute_gv_cost(q, size_to_reach)
    wc_cost = compute_wc_cost(q, size_to_reach)
    while (gv_cost <= budget or wc_cost <= budget) and size_to_reach > 2:
        size_to_reach = size_to_reach - 1
        gv_cost = compute_gv_cost(q, size_to_reach)
        wc_cost = compute_wc_cost(q, size_to_reach)

    while gv_cost > budget and wc_cost > budget:
        size_to_reach = size_to_reach + 1
        gv_cost = compute_gv_cost(q, size_to_reach)
        wc_cost = compute_wc_cost(q, size_to_reach)

    method = "GV" if gv_cost <= wc_cost else "WC"
    # OE_END
    return size_to_reach, method
