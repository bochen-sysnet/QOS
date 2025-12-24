import os

from multiprocessing import Process, Value
from qos.error_mitigator.analyser import *
from qos.error_mitigator.optimiser import *
from qos.types.types import Qernel


def evolved_run(self, q: Qernel):
    # OE_BEGIN
    def computeCuttingCosts(q: Qernel, size_to_reach: int):
        gv_pass = GVOptimalDecompositionPass(size_to_reach)
        wc_pass = OptimalWireCuttingPass(size_to_reach)
        try:
            gv_cost = Value("i", 1000)
            wc_cost = Value("i", 1000)

            p = Process(target=gv_pass.cost, args=(q, gv_cost))
            p.start()
            p.join(600)
            if p.is_alive():
                p.terminate()
                p.join()
            gv_cost = gv_cost.value
            p = Process(target=wc_pass.cost, args=(q, wc_cost))
            p.start()
            p.join(600)
            if p.is_alive():
                p.terminate()
                p.join()
            wc_cost = wc_cost.value
        except Exception:
            class _CostBox:
                def __init__(self, value: int):
                    self.value = value

            gv_box = _CostBox(1000)
            wc_box = _CostBox(1000)
            try:
                gv_pass.cost(q, gv_box)
            except ValueError:
                gv_box.value = 1000
            try:
                wc_pass.cost(q, wc_box)
            except ValueError:
                wc_box.value = 1000
            gv_cost = gv_box.value
            wc_cost = wc_box.value

        return {"GV": gv_cost, "WC": wc_cost}

    analysis_pass = BasicAnalysisPass()
    supermarq_features_pass = SupermarqFeaturesAnalysisPass()

    analysis_pass.run(q)
    supermarq_features_pass.run(q)

    is_qaoa_pass = IsQAOACircuitPass()
    budget = self.budget
    if is_qaoa_pass.run(q):
        qaoa_analysis_pass = QAOAAnalysisPass()
        qaoa_analysis_pass.run(q)
        metadata = q.get_metadata()
        num_cnots = metadata["num_nonlocal_gates"]
        hotspots = list(metadata["hotspot_nodes"].values())
        qubits_to_freeze = 0

        for i in range(2):
            if hotspots[i] / num_cnots >= 0.07:
                qubits_to_freeze = qubits_to_freeze + 1

        qubits_to_freeze = min(qubits_to_freeze, budget)

        if qubits_to_freeze > 0:
            QF_pass = FrozenQubitsPass(qubits_to_freeze)
            q = QF_pass.run(q)
            budget = budget - qubits_to_freeze

    size_to_reach = self.size_to_reach
    costs = computeCuttingCosts(q, size_to_reach)
    max_iters = int(os.getenv("QOS_COST_SEARCH_MAX_ITERS", "0"))
    iter_ctr = 0
    while (costs["GV"] <= budget or costs["WC"] <= budget) and size_to_reach > 2:
        iter_ctr += 1
        if max_iters > 0 and iter_ctr > max_iters:
            break
        size_to_reach = size_to_reach - 1
        costs = computeCuttingCosts(q, size_to_reach)

    iter_ctr = 0
    while costs["GV"] > budget and costs["WC"] > budget:
        iter_ctr += 1
        if max_iters > 0 and iter_ctr > max_iters:
            break
        size_to_reach = size_to_reach + 1
        costs = computeCuttingCosts(q, size_to_reach)

    if costs["GV"] <= budget or costs["WC"] <= budget:
        if costs["GV"] <= costs["WC"] or (costs["GV"] == 0 and costs["WC"] == 0):
            gv_pass = GVOptimalDecompositionPass(size_to_reach)
            q = gv_pass.run(q, self.budget)
        else:
            wc_pass = OptimalWireCuttingPass(size_to_reach)
            q = wc_pass.run(q, self.budget)

    qr_pass = RandomQubitReusePass(self.size_to_reach)
    q = qr_pass.run(q)

    # OE_END
    return q
