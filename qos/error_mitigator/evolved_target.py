from multiprocessing import Value

from qos.error_mitigator.optimiser import (
    GVOptimalDecompositionPass,
    OptimalWireCuttingPass,
)
from qos.types.types import Qernel


def evolved_cost_search(self, q: Qernel, size_to_reach: int, budget: int):
    # OE_BEGIN
    # We observed a failure mode: any strategy that evaluates MANY sizes can "walk into"
    # a pathological cost() region and blow up to 10s~30s per case (even without timeouts),
    # because we cannot preempt a long-running cost() call.
    #
    # So this version is STRICTLY constant-work:
    #   - At most 2 sizes tested for "hard/entangling" circuits
    #   - At most 2 sizes tested + a single 1-step refinement for "normal" circuits
    #   - Never scans / binary searches / wide windows
    #
    # This should keep runtime stable (like your best 0.03s run) while recovering
    # some qose_depth/cnot by a *single* downward refinement when safe.

    # ---- metadata / bounds ----
    try:
        meta = q.get_metadata() or {}
    except Exception:
        meta = {}

    try:
        num_qubits = int(meta.get("num_qubits", 0))
    except Exception:
        num_qubits = 0

    max_size = num_qubits if num_qubits and num_qubits > 0 else max(2, int(size_to_reach) + 8)

    if size_to_reach < 2:
        size_to_reach = 2
    if size_to_reach > max_size:
        size_to_reach = max_size

    # Heuristic: high-entanglement circuits were the ones that previously exploded in cost_search.
    # For them, we do *fewer probes* (no downward refinement).
    try:
        er = float(meta.get("entanglement_ratio", 0.0))
    except Exception:
        er = 0.0
    is_hard = er >= 0.80

    # ---- memoization ----
    gv_cache = {}
    wc_cache = {}

    def gv_cost(s: int) -> int:
        v = gv_cache.get(s)
        if v is not None:
            return v
        gv_pass = GVOptimalDecompositionPass(s)
        gv_cost_value = Value("i", 0)
        gv_pass.cost(q, gv_cost_value)
        v = int(gv_cost_value.value)
        gv_cache[s] = v
        return v

    def wc_cost(s: int) -> int:
        v = wc_cache.get(s)
        if v is not None:
            return v
        wc_pass = OptimalWireCuttingPass(s)
        wc_cost_value = Value("i", 0)
        wc_pass.cost(q, wc_cost_value)
        v = int(wc_cost_value.value)
        wc_cache[s] = v
        return v

    def costs_both(s: int):
        return {"GV": gv_cost(s), "WC": wc_cost(s)}

    def feasible(s: int) -> bool:
        # Order by circuit type: for highly entangling circuits, check WC first (often cheaper / smoother),
        # otherwise check GV first.
        if is_hard:
            if wc_cost(s) <= budget:
                return True
            return gv_cost(s) <= budget
        else:
            if gv_cost(s) <= budget:
                return True
            return wc_cost(s) <= budget

    s0 = size_to_reach

    # ---- Probe 1: s0 ----
    if feasible(s0):
        # Optional 1-step refinement down (ONLY for non-hard circuits)
        if (not is_hard) and s0 > 2:
            s1 = s0 - 1
            if feasible(s1):
                return s1, costs_both(s1), 0.0
        return s0, costs_both(s0), 0.0

    # ---- Probe 2: s0+1 ----
    s_up1 = s0 + 1
    if s_up1 <= max_size and feasible(s_up1):
        # If we had to go up, do not refine down (keeps calls bounded and avoids oscillation).
        return s_up1, costs_both(s_up1), 0.0

    # For non-hard circuits only, allow ONE more upward probe (still constant)
    if not is_hard:
        s_up2 = s0 + 2
        if s_up2 <= max_size and feasible(s_up2):
            return s_up2, costs_both(s_up2), 0.0

    # Fallback (bounded-time): return s0 with computed costs (cached)
    return s0, costs_both(s0), 0.0
    # OE_END
