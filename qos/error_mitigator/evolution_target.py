import os
from multiprocessing import Process, Value

from qos.error_mitigator.optimiser import (
    GVOptimalDecompositionPass,
    OptimalWireCuttingPass,
)
from qos.types.types import Qernel


def evolved_cost_search(self, q: Qernel, size_to_reach: int, budget: int):
    # OE_BEGIN
    def computeCuttingCosts(q, size_to_reach):
        gv_cost = 1000
        wc_cost = 1000

        if self.methods.get("GV", True):
            self._gv_cost_calls = getattr(self, "_gv_cost_calls", 0) + 1
            gv_pass = GVOptimalDecompositionPass(size_to_reach)
            gv_cost_value = Value("i", 1000)
            p = Process(target=gv_pass.cost, args=(q, gv_cost_value))
            p.start()
            p.join(600)
            if p.is_alive():
                p.terminate()
                p.join()
            gv_cost = gv_cost_value.value

        if self.methods.get("WC", True):
            self._wc_cost_calls = getattr(self, "_wc_cost_calls", 0) + 1
            wc_pass = OptimalWireCuttingPass(size_to_reach)
            wc_cost_value = Value("i", 1000)
            p = Process(target=wc_pass.cost, args=(q, wc_cost_value))
            p.start()
            p.join(600)
            if p.is_alive():
                p.terminate()
                p.join()
            wc_cost = wc_cost_value.value

        return {"GV": gv_cost, "WC": wc_cost}

    costs = computeCuttingCosts(q, size_to_reach)
    iter_ctr = 0
    while (costs["GV"] <= budget or costs["WC"] <= budget) and size_to_reach > 2:
        iter_ctr += 1
        size_to_reach = size_to_reach - 1
        costs = computeCuttingCosts(q, size_to_reach)

    iter_ctr = 0
    while costs["GV"] > budget and costs["WC"] > budget:
        iter_ctr += 1
        size_to_reach = size_to_reach + 1
        costs = computeCuttingCosts(q, size_to_reach)
    # OE_END

    return size_to_reach, costs, 0.0
