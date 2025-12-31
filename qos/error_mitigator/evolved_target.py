from multiprocessing import Value

from qos.error_mitigator.optimiser import (
    GVOptimalDecompositionPass,
    OptimalWireCuttingPass,
)
from qos.types.types import Qernel


def evolved_cost_search(self, q: Qernel, size_to_reach: int, budget: int):
    # OE_BEGIN
    # Your latest run is *much* faster (avg_run_time ~0.096s, cost_search ~0.078s),
    # meaning the cost() calls are now cheap enough that we can afford a bit more
    # searching to reduce qose_depth/cnot without regressing runtime.
    #
    # Strategy:
    #   1) Use the previous bounded-window search to find a feasible "anchor" size s*.
    #   2) Then perform a small, cheap LOCAL refinement around s* to find the smallest
    #      feasible size within a wider window (this tends to reduce depth/CNOT).
    #
    # Still: never force s=2 unless we actually search down to it.

    # ---- metadata / bounds ----
    try:
        meta = q.get_metadata() or {}
    except Exception:
        meta = {}

    try:
        num_qubits = int(meta.get("num_qubits", 0))
    except Exception:
        num_qubits = 0

    max_size = num_qubits if num_qubits and num_qubits > 0 else max(2, int(size_to_reach) + 32)

    if size_to_reach < 2:
        size_to_reach = 2
    if size_to_reach > max_size:
        size_to_reach = max_size

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

    def predicate(s: int) -> bool:
        # short-circuit: compute WC only if needed
        if gv_cost(s) <= budget:
            return True
        return wc_cost(s) <= budget

    def full_costs(s: int):
        return {"GV": gv_cost(s), "WC": wc_cost(s)}

    # ---- Phase 1: bounded anchor search (same as previous, slightly larger window) ----
    WINDOW1 = 10  # a bit wider now that costs are cheap
    lo1 = max(2, size_to_reach - WINDOW1)
    hi1 = min(max_size, size_to_reach + WINDOW1)

    s0 = size_to_reach
    p0 = predicate(s0)

    anchor = s0

    if p0:
        hi_true = s0
        step = 1
        lo_false = None
        while True:
            cand = hi_true - step
            if cand < lo1:
                break
            if not predicate(cand):
                lo_false = cand
                break
            hi_true = cand
            step *= 2

        if lo_false is None:
            anchor = hi_true
        else:
            left = lo_false + 1
            right = hi_true
            while left < right:
                mid = (left + right) // 2
                if predicate(mid):
                    right = mid
                else:
                    left = mid + 1
            anchor = left
    else:
        lo_false = s0
        step = 1
        hi_true = None
        while True:
            cand = lo_false + step
            if cand > hi1:
                break
            if predicate(cand):
                hi_true = cand
                break
            lo_false = cand
            step *= 2

        if hi_true is None:
            anchor = s0
        else:
            left = lo_false + 1
            right = hi_true
            while left < right:
                mid = (left + right) // 2
                if predicate(mid):
                    right = mid
                else:
                    left = mid + 1
            anchor = left

    # ---- Phase 2: local refinement around anchor to reduce depth/CNOT ----
    # Since the objective is "min feasible size", we try to walk down greedily
    # inside a small refinement window. This is cheap now and helps qose_*.
    WINDOW2 = 6
    low_ref = max(2, anchor - WINDOW2)

    chosen = anchor
    s = anchor - 1
    steps = 0
    # Cap refinement steps so we never blow up.
    while s >= low_ref and steps < (WINDOW2 + 2):
        if predicate(s):
            chosen = s
            s -= 1
            steps += 1
        else:
            break

    costs = full_costs(chosen)
    # OE_END
    return chosen, costs, 0.0
