from multiprocessing import Value

from qos.error_mitigator.optimiser import (
    GVOptimalDecompositionPass,
    OptimalWireCuttingPass,
)
from qos.types.types import Qernel


def evolved_cost_search(self, q: Qernel, size_to_reach: int, budget: int):
    # OE_BEGIN
    def computeCuttingCosts(q, size_to_reach):
        gv_pass = GVOptimalDecompositionPass(size_to_reach)
        gv_cost_value = Value("i", 0)
        gv_pass.cost(q, gv_cost_value)
        gv_cost = gv_cost_value.value

        wc_pass = OptimalWireCuttingPass(size_to_reach)
        wc_cost_value = Value("i", 0)
        wc_pass.cost(q, wc_cost_value)
        wc_cost = wc_cost_value.value

        return {"GV": gv_cost, "WC": wc_cost}

    costs = computeCuttingCosts(q, size_to_reach)
    while (costs["GV"] <= budget or costs["WC"] <= budget) and size_to_reach > 2:
        size_to_reach = size_to_reach - 1
        costs = computeCuttingCosts(q, size_to_reach)

    while costs["GV"] > budget and costs["WC"] > budget:
        size_to_reach = size_to_reach + 1
        costs = computeCuttingCosts(q, size_to_reach)
    # OE_END

    return size_to_reach, costs, 0.0
