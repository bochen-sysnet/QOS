from multiprocessing import Value

from qos.error_mitigator.optimiser import (
    GVOptimalDecompositionPass,
    OptimalWireCuttingPass,
)
from qos.types.types import Qernel


def evolved_cost_search(self, q: Qernel, size_to_reach: int, budget: int):
    # OE_BEGIN
    # New policy (based on observed metric regression):
    # - The caller-provided size_to_reach can be "feasible" but overly aggressive (too small),
    #   causing excessive cutting and blowing up depth/CNOT.
    # - Therefore we prefer the LARGEST feasible size (least cutting), starting at max_size_cap.
    #
    # Feasible(size) := (GV_cost <= budget) OR (WC_cost <= budget)

    if size_to_reach is None:
        size_to_reach = 2
    if size_to_reach < 2:
        size_to_reach = 2

    # Best-effort upper cap using metadata (usually num_qubits)
    try:
        input_meta = q.get_metadata() or {}
        max_size_cap = int(input_meta.get("num_qubits", size_to_reach))
        if max_size_cap < 2:
            max_size_cap = max(2, size_to_reach)
    except Exception:
        max_size_cap = max(2, size_to_reach)

    if size_to_reach > max_size_cap:
        size_to_reach = max_size_cap

    # Cache per size: {"GV": int, "WC": int or None}
    cache = {}

    # Reuse Value objects to avoid repeated allocations
    gv_cost_value = Value("i", 0)
    wc_cost_value = Value("i", 0)

    def _compute_gv(size: int) -> int:
        gv_cost_value.value = 0
        gv_pass = GVOptimalDecompositionPass(size)
        gv_pass.cost(q, gv_cost_value)
        return int(gv_cost_value.value)

    def _compute_wc(size: int) -> int:
        wc_cost_value.value = 0
        wc_pass = OptimalWireCuttingPass(size)
        wc_pass.cost(q, wc_cost_value)
        return int(wc_cost_value.value)

    def get_costs(size: int, need_wc: bool) -> dict:
        if size in cache:
            entry = cache[size]
            if (not need_wc) or (entry.get("WC", None) is not None):
                return entry

        entry = cache.get(size, {"GV": None, "WC": None})

        if entry.get("GV", None) is None:
            entry["GV"] = _compute_gv(size)

        # Only compute WC if needed (usually only when GV fails)
        if need_wc and entry.get("WC", None) is None:
            entry["WC"] = _compute_wc(size)

        cache[size] = entry
        return entry

    def feasible(size: int) -> bool:
        c = get_costs(size, need_wc=False)
        if c["GV"] <= budget:
            return True
        c = get_costs(size, need_wc=True)
        return c["WC"] <= budget

    def full_costs(size: int) -> dict:
        c = get_costs(size, need_wc=True)
        return {"GV": c["GV"], "WC": c["WC"]}

    # 1) Prefer the cap (least cutting)
    if feasible(max_size_cap):
        size_to_reach = max_size_cap
        costs = full_costs(size_to_reach)
        # OE_END
        return size_to_reach, costs, 0.0

    # 2) If cap infeasible, search downward for the largest feasible size.
    #    Exponential bracketing downward, then binary search.
    hi = max_size_cap  # infeasible
    step = 1
    lo = max(2, hi - step)

    # Try to find a feasible lo by moving down
    while lo > 2 and (not feasible(lo)):
        hi = lo
        step *= 2
        lo = max(2, hi - step)
        if lo == 2:
            break

    # If even at size 2 infeasible, return 2 (best we can do)
    if not feasible(lo):
        size_to_reach = lo
        costs = full_costs(size_to_reach)
        # OE_END
        return size_to_reach, costs, 0.0

    # Now: lo feasible, hi infeasible (or hi == lo+something). We want the LARGEST feasible.
    L, R = lo, hi - 1
    # Binary search for last feasible in [L, R]
    ans = L
    while L <= R:
        mid = (L + R) // 2
        if feasible(mid):
            ans = mid
            L = mid + 1
        else:
            R = mid - 1

    size_to_reach = ans
    costs = full_costs(size_to_reach)
    # OE_END

    return size_to_reach, costs, 0.0
