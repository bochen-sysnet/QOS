from multiprocessing import Value

from qos.error_mitigator.optimiser import (
    GVOptimalDecompositionPass,
    OptimalWireCuttingPass,
)
from qos.types.types import Qernel


def evolved_cost_search(self, q: Qernel, size_to_reach: int, budget: int):
    # OE_BEGIN
    # Preserve baseline semantics while reducing cost() calls:
    # Baseline behavior:
    #   1) While feasible (GV<=B or WC<=B), decrease size
    #   2) Then while infeasible (GV>B and WC>B), increase size
    # This yields the smallest feasible size (boundary).
    #
    # Optimization:
    #   - Cache costs per size
    #   - If s0 is feasible (common), only search DOWN to find the boundary:
    #       * exponential bracketing down to find an infeasible point
    #       * binary search to find the smallest feasible size
    #   - If s0 is infeasible, search UP with exponential bracketing + binary search, but
    #     with strict caps to avoid 300s timeout.
    #   - Always return fully-computed {"GV":..., "WC":...} (no dummy costs).

    lower = 2
    s0 = int(size_to_reach)
    if s0 < lower:
        s0 = lower

    # A tight upward safety cap (prevents runaway if budget is tiny or metadata is weird).
    # We intentionally do NOT use num_qubits as an upper bound because it caused timeouts before.
    upper_cap = s0 + 16

    # Hard cap on how many expensive evaluations we allow (across all sizes).
    # Keeps worst-case bounded even if a single cost() call is slow.
    MAX_EVALS = 16
    evals_used = 0

    # Cache: size -> {"GV": int, "WC": int}
    cache = {}

    def compute_costs(size: int):
        nonlocal evals_used
        if size in cache:
            return cache[size]

        # Enforce evaluation budget
        if evals_used >= MAX_EVALS:
            # Fallback: return a clearly infeasible cost to stop searching.
            # (Should be rare; prevents timeouts.)
            costs = {"GV": budget + 1, "WC": budget + 1}
            cache[size] = costs
            return costs

        evals_used += 1

        gv_pass = GVOptimalDecompositionPass(size)
        gv_cost_value = Value("i", 0)
        gv_pass.cost(q, gv_cost_value)
        gv = gv_cost_value.value

        wc_pass = OptimalWireCuttingPass(size)
        wc_cost_value = Value("i", 0)
        wc_pass.cost(q, wc_cost_value)
        wc = wc_cost_value.value

        costs = {"GV": gv, "WC": wc}
        cache[size] = costs
        return costs

    def is_feasible(size: int) -> bool:
        c = compute_costs(size)
        return (c["GV"] <= budget) or (c["WC"] <= budget)

    # --- Case 1: s0 feasible -> search downward to find smallest feasible size ---
    if is_feasible(s0):
        # Exponential step down to find an infeasible "lo"
        hi = s0  # feasible
        step = 1
        lo = None  # infeasible

        while True:
            cand = hi - step
            if cand <= lower:
                cand = lower
            if not is_feasible(cand):
                lo = cand
                break
            # still feasible
            hi = cand
            if cand == lower:
                # Everything down to lower is feasible; boundary is lower.
                return lower, compute_costs(lower), 0.0
            step *= 2
            # Also stop if we run out of eval budget
            if evals_used >= MAX_EVALS:
                return hi, compute_costs(hi), 0.0

        # Now we have: lo infeasible, and some feasible size > lo.
        # Binary search smallest feasible in (lo, original_s0]
        left = lo + 1
        right = s0
        best = s0
        while left <= right and evals_used < MAX_EVALS:
            mid = (left + right) // 2
            if is_feasible(mid):
                best = mid
                right = mid - 1
            else:
                left = mid + 1

        return best, compute_costs(best), 0.0

    # --- Case 2: s0 infeasible -> search upward to find feasibility (bounded) ---
    lo = s0  # infeasible
    step = 1
    hi = None  # feasible
    cur = lo

    while True:
        cand = cur + step
        if cand >= upper_cap:
            cand = upper_cap
        if is_feasible(cand):
            hi = cand
            break
        cur = cand
        if cand == upper_cap or evals_used >= MAX_EVALS:
            # Could not find feasible within cap/budget; return best effort.
            return cand, compute_costs(cand), 0.0
        step *= 2

    # Binary search smallest feasible in (lo, hi]
    left = lo + 1
    right = hi
    best = hi
    while left <= right and evals_used < MAX_EVALS:
        mid = (left + right) // 2
        if is_feasible(mid):
            best = mid
            right = mid - 1
        else:
            left = mid + 1

    return best, compute_costs(best), 0.0
    # OE_END
