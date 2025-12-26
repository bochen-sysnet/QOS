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
        qubits_to_freeze = sum(1 for hs in hotspots if hs / num_cnots >= 0.07)
        qubits_to_freeze = min(qubits_to_freeze, budget)

        if qubits_to_freeze > 0:
            QF_pass = FrozenQubitsPass(qubits_to_freeze)
            q = QF_pass.run(q)
            budget -= qubits_to_freeze

    size_to_reach = self.size_to_reach
    costs = self.computeCuttingCosts(q, size_to_reach)
    max_iters = int(os.getenv("QOS_COST_SEARCH_MAX_ITERS", "5"))
    iter_ctr = 0

    # Improved ordering and condition for applying techniques
    if costs["GV"] <= budget or costs["WC"] <= budget:
        technique = "GV" if costs["GV"] <= costs["WC"] else "WC"
        q = getattr(self, f"apply{technique}")(q, size_to_reach)

    # Apply QR before final cost check
    q = self.applyQR(q, self.size_to_reach)

    # Re-evaluate costs after QR
    costs = self.computeCuttingCosts(q, size_to_reach)

    while (costs["GV"] > budget and costs["WC"] > budget) and size_to_reach > 2:
        iter_ctr += 1
        if max_iters > 0 and iter_ctr > max_iters:
            break
        size_to_reach -= 1
        costs = self.computeCuttingCosts(q, size_to_reach)

    while costs["GV"] > budget and costs["WC"] > budget:
        iter_ctr += 1
        if max_iters > 0 and iter_ctr > max_iters:
            break
        size_to_reach += 1
        costs = self.computeCuttingCosts(q, size_to_reach)

    if costs["GV"] <= budget or costs["WC"] <= budget:
        technique = "GV" if costs["GV"] <= costs["WC"] else "WC"
        q = getattr(self, f"apply{technique}")(q, size_to_reach)

    # OE_END
    return q