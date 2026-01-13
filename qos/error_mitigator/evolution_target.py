from multiprocessing import Value

from qos.error_mitigator.optimiser import (
    GVOptimalDecompositionPass,
    OptimalWireCuttingPass,
)
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

    input_size = size_to_reach
    
    # OE_BEGIN
    def compute_gv_cost(q, size_to_reach):
        gv_pass = GVOptimalDecompositionPass(size_to_reach)
        gv_cost_value = Value("i", 0)
        gv_pass.cost(q, gv_cost_value)
        return gv_cost_value.value

    def compute_wc_cost(q, size_to_reach):
        wc_pass = OptimalWireCuttingPass(size_to_reach)
        wc_cost_value = Value("i", 0)
        wc_pass.cost(q, wc_cost_value)
        return wc_cost_value.value

    gv_cost = compute_gv_cost(q, size_to_reach)
    wc_cost = compute_wc_cost(q, size_to_reach)
    gv_cost_trace = [gv_cost]
    wc_cost_trace = [wc_cost]
    while (gv_cost <= budget or wc_cost <= budget) and size_to_reach > 2:
        size_to_reach = size_to_reach - 1
        gv_cost = compute_gv_cost(q, size_to_reach)
        wc_cost = compute_wc_cost(q, size_to_reach)
        gv_cost_trace.append(gv_cost)
        wc_cost_trace.append(wc_cost)

    while gv_cost > budget and wc_cost > budget:
        size_to_reach = size_to_reach + 1
        gv_cost = compute_gv_cost(q, size_to_reach)
        wc_cost = compute_wc_cost(q, size_to_reach)
        gv_cost_trace.append(gv_cost)
        wc_cost_trace.append(wc_cost)

    method = "GV" if gv_cost <= wc_cost else "WC"
    # OE_END
    self._qose_cost_search_input_size = input_size
    self._qose_cost_search_budget = budget
    self._qose_cost_search_output_size = size_to_reach
    self._qose_cost_search_method = method
    self._qose_gv_cost_trace = gv_cost_trace
    self._qose_wc_cost_trace = wc_cost_trace

    return size_to_reach, method
