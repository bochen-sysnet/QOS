import os

from qos.error_mitigator.analyser import IsQAOACircuitPass, QAOAAnalysisPass
from qos.error_mitigator.optimiser import FrozenQubitsPass
from qos.types.types import Qernel


def evolved_run(self, q: Qernel):
    # OE_BEGIN
    for key in self.methods:
        self.methods[key] = True

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
    if not getattr(self, "use_cost_search", True):
        if self.methods["GV"] and self.methods["WC"]:
            q = self.applyBestCut(q, size_to_reach)
        elif self.methods["GV"]:
            q = self.applyGV(q, size_to_reach)
        elif self.methods["WC"]:
            q = self.applyWC(q, size_to_reach)
    else:
        costs = self.computeCuttingCosts(q, size_to_reach)
        gv_enabled = self.methods.get("GV", False)
        wc_enabled = self.methods.get("WC", False)
        max_iters = int(os.getenv("QOS_COST_SEARCH_MAX_ITERS", "5"))
        iter_ctr = 0
        while (
            (gv_enabled and costs["GV"] <= budget)
            or (wc_enabled and costs["WC"] <= budget)
        ) and size_to_reach > 2:
            iter_ctr += 1
            if max_iters > 0 and iter_ctr > max_iters:
                break
            size_to_reach = size_to_reach - 1
            costs = self.computeCuttingCosts(q, size_to_reach)

        iter_ctr = 0
        while (
            (gv_enabled and costs["GV"] > budget)
            and (wc_enabled and costs["WC"] > budget)
        ):
            iter_ctr += 1
            if max_iters > 0 and iter_ctr > max_iters:
                break
            size_to_reach = size_to_reach + 1
            costs = self.computeCuttingCosts(q, size_to_reach)

        if costs["GV"] <= budget or costs["WC"] <= budget:
            if costs["GV"] <= costs["WC"] or (costs["GV"] == 0 and costs["WC"] == 0):
                q = self.applyGV(q, size_to_reach)
            else:
                q = self.applyWC(q, size_to_reach)

    q = self.applyQR(q, size_to_reach)

    # OE_END
    return q
