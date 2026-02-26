from __future__ import annotations

import math
from typing import Callable, Dict, Tuple, Union, Any

from qos.error_mitigator.run import compute_gv_cost, compute_wc_cost
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

    # OE_BEGIN
    def _clamp_int(x: int, lo: int, hi: int) -> int:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _clamp_float(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _safe_cost_call(
        fn: Callable[..., Any], s: int, timeout_sec: float
    ) -> Tuple[float, bool]:
        """
        Returns (cost, timed_out). If the underlying helper doesn't support timeout
        or doesn't return a timed_out flag, we fall back gracefully.
        """
        try:
            out = fn(q, s, timeout_sec=timeout_sec)
        except TypeError:
            out = fn(q, s)

        # Common expected: (cost, timed_out)
        if isinstance(out, tuple) and len(out) >= 2:
            cost = float(out[0])
            timed_out = bool(out[1])
        else:
            cost = float(out)
            timed_out = False

        # Defensive normalization
        if math.isnan(cost) or cost < 0:
            cost = float("inf")
        if timed_out:
            cost = float("inf")
        return cost, timed_out

    # Normalize metadata for scale-free heuristics (generalize across qubit counts).
    n = int(num_qubits) if int(num_qubits) > 0 else max(2, int(size_to_reach) if int(size_to_reach) > 0 else 2)
    n = max(2, n)
    denom = float(max(1, n))

    nonlocal_density = float(num_nonlocal_gates) / denom
    instr_density = float(number_instructions) / denom
    depth_density = float(depth) / denom
    comp_penalty = math.log1p(max(0, int(num_connected_components) - 1))

    # A smooth "complexity" proxy (bounded), only used to set timeouts / probing.
    complexity = (
        0.35 * nonlocal_density
        + 0.25 * depth_density
        + 0.20 * instr_density
        + 0.10 * float(entanglement_ratio)
        + 0.10 * comp_penalty
    )
    complexity = _clamp_float(complexity, 0.0, 6.0)

    # Dynamic timeouts to avoid pathological cases (e.g., mid-size subgraphs) while
    # remaining permissive on easy instances. Keep bounded for stability.
    gv_timeout_sec = _clamp_float(0.15 + 0.65 * math.tanh(complexity / 2.0), 0.08, 0.90)
    wc_timeout_sec = _clamp_float(0.20 + 0.90 * math.tanh(complexity / 2.0), 0.10, 1.20)

    # Memoize cost evaluations to reduce repeated calls.
    # cache[s] = (gv_cost, wc_cost)
    cache: Dict[int, Tuple[float, float]] = {}

    def _eval_size(s: int) -> Tuple[float, str, float, float]:
        """
        Evaluate costs at size s and return:
        (best_cost, best_method, gv_cost, wc_cost)
        """
        s = max(2, int(s))
        if s in cache:
            gv_cost, wc_cost = cache[s]
        else:
            gv_cost, _gv_to = _safe_cost_call(compute_gv_cost, s, gv_timeout_sec)

            # Compute WC only when it can change the decision:
            # - GV infeasible (must try WC)
            # - or near boundary / structurally "split" circuits where WC can win
            #   (helps generalization; avoids overfitting to GV-only behavior).
            want_wc = False
            if gv_cost > float(budget):
                want_wc = True
            else:
                # Boundary or structure-triggered exploration
                near_budget = gv_cost >= float(budget) - 1.0
                split_like = (int(num_connected_components) > 1) or (float(program_communication) > 0.15)
                low_ent = float(entanglement_ratio) < 0.25
                want_wc = near_budget or split_like or low_ent

            if want_wc:
                wc_cost, _wc_to = _safe_cost_call(compute_wc_cost, s, wc_timeout_sec)
            else:
                wc_cost = float("inf")

            cache[s] = (gv_cost, wc_cost)

        best_cost = gv_cost
        best_method = "GV"
        if wc_cost < best_cost:
            best_cost = wc_cost
            best_method = "WC"

        return best_cost, best_method, gv_cost, wc_cost

    def _feasible(best_cost: float) -> bool:
        return best_cost <= float(budget)

    # Choose a robust starting point:
    # - Respect caller-provided size_to_reach
    # - Avoid starting too small on complex circuits (reduces oscillation/over-search)
    base_start = int(size_to_reach) if int(size_to_reach) > 0 else n
    # Complexity-adaptive floor (scale-free).
    # For harder circuits, start closer to n; for easier ones, allow smaller start.
    start_floor = int(round(n * (0.55 + 0.15 * math.tanh(complexity / 2.0))))
    start = max(base_start, start_floor)
    start = _clamp_int(start, 2, n)

    # Ensure we have a feasible upper point (may need to expand up to n).
    hi = start
    hi_best_cost, hi_best_method, _, _ = _eval_size(hi)
    if not _feasible(hi_best_cost):
        # Expand toward n with doubling steps (few evaluations).
        step = 1
        while hi < n and not _feasible(hi_best_cost):
            hi = min(n, hi + step)
            hi_best_cost, hi_best_method, _, _ = _eval_size(hi)
            step = min(step * 2, max(1, n))

        # If still infeasible at n, return the best effort at n (avoids infinite loops).
        if not _feasible(hi_best_cost):
            # Force WC evaluation at final size to avoid missing a feasible alternative.
            _, _, gv_c, wc_c = _eval_size(hi)
            if wc_c < gv_c:
                return hi, "WC"
            return hi, "GV"

    # Step-down search (coarse-to-fine) for the smallest feasible size.
    candidate = hi
    step = max(1, (candidate - 2) // 2)
    while step >= 1:
        trial = candidate - step
        if trial < 2:
            step //= 2
            continue
        t_cost, _t_method, _, _ = _eval_size(trial)
        if _feasible(t_cost):
            candidate = trial
        else:
            step //= 2

    # Robustness margin: avoid picking a razor-thin boundary point that can fail
    # on nearby sizes/qubit-subsets. If tight, allow bumping size slightly.
    final_size = candidate
    final_cost, final_method, _, _ = _eval_size(final_size)
    slack = float(budget) - float(final_cost)

    if slack < 1.0:
        for bump in (1, 2):
            s2 = final_size + bump
            if s2 > n:
                break
            c2, m2, _, _ = _eval_size(s2)
            if _feasible(c2) and (float(budget) - float(c2)) >= 1.0:
                final_size, final_cost, final_method = s2, c2, m2
                break

    # Tie-breaker for stability if WC was skipped (inf) but could be competitive:
    # if we are near budget, evaluate WC once at the final size.
    if float(budget) - float(final_cost) <= 1.0:
        gv_c, _ = _safe_cost_call(compute_gv_cost, final_size, gv_timeout_sec)
        wc_c, _ = _safe_cost_call(compute_wc_cost, final_size, wc_timeout_sec)
        if wc_c < gv_c and wc_c <= float(budget):
            final_method = "WC"
        else:
            final_method = "GV"

    # OE_END
    return final_size, final_method