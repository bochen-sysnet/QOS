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
    # Clamp into a meaningful active range; allow >= num_qubits only as explicit fallback.
    nq = int(num_qubits) if num_qubits else 0
    if nq <= 2:
        s_fallback = int(size_to_reach) if size_to_reach is not None else 2
        if s_fallback < 2:
            s_fallback = 2
        return s_fallback, "GV"

    s_min = 2
    s_max = nq - 1
    s_in = int(size_to_reach) if size_to_reach is not None else s_max
    if s_in < s_min:
        s_in = s_min
    elif s_in > s_max:
        s_in = s_max

    # Cheap heuristic to pick a good starting probe size (reduces search steps).
    # Higher "complexity" -> bias to larger target sizes (less aggressive cutting).
    denom_inst = float(number_instructions) if number_instructions else 1.0
    cnot_ratio = float(num_cnot_gates) / denom_inst
    depth_norm = float(depth) / max(1.0, float(nq))
    crit_norm = float(critical_depth) / max(1.0, float(depth) if depth else 1.0)

    complexity = 0.0
    complexity += 0.35 * min(1.0, depth_norm / 5.0)
    complexity += 0.30 * min(1.0, float(entanglement_ratio))
    complexity += 0.20 * min(1.0, cnot_ratio)
    complexity += 0.15 * min(1.0, crit_norm)

    # Map complexity to a fraction of nq, then blend with the incoming size_to_reach.
    frac = 0.40 + 0.40 * complexity  # in [0.40, 0.80]
    s_guess = int(round(frac * nq))
    if s_guess < s_min:
        s_guess = s_min
    elif s_guess > s_max:
        s_guess = s_max

    s0 = int(round(0.65 * s_in + 0.35 * s_guess))
    if s0 < s_min:
        s0 = s_min
    elif s0 > s_max:
        s0 = s_max

    b = int(budget) if budget is not None else 0

    cache = {}

    def _get_cost(m: str, s: int) -> int:
        key = (m, s)
        if key in cache:
            return cache[key]
        if m == "GV":
            v = compute_gv_cost(q, s)
        else:
            v = compute_wc_cost(q, s)
        cache[key] = v
        return v

    def _find_smallest_valid(method: str, start_s: int) -> (int, int):
        """
        Assume cost is monotone non-increasing w.r.t. size_to_reach (larger size => lower/equal cost).
        Return (best_size, best_cost) where best_cost <= budget if feasible, else (nq, cost_at_s_max).
        """
        c_start = _get_cost(method, start_s)

        # If budget is non-positive, avoid search churn; pick a conservative no-op fallback
        # only when even the loosest meaningful size is still expensive.
        if b <= 0:
            # Keep within action range (avoid no-op) unless clearly impossible (checked below).
            c_loose = _get_cost(method, s_max)
            if c_loose > b:
                return nq, c_loose
            return s_max, c_loose

        if c_start <= b:
            # Search downward to find an invalid lower bound quickly (exponential), then binary.
            hi = start_s
            c_hi = c_start
            lo = s_min - 1  # conceptual invalid (below domain)
            step = 1
            # Exponential step-down (bounded).
            for _ in range(5):
                if hi <= s_min:
                    break
                cand = hi - step
                if cand < s_min:
                    cand = s_min
                c_cand = _get_cost(method, cand)
                if c_cand <= b:
                    hi = cand
                    c_hi = c_cand
                    if hi == s_min:
                        break
                    step <<= 1
                else:
                    lo = cand
                    break

            if hi == s_min:
                return hi, c_hi
            if lo == s_min - 1:
                # Even s_min is valid (or we never found invalid); return smallest meaningful size.
                c_min = _get_cost(method, s_min)
                return s_min, c_min

            # Binary search in (lo, hi] for smallest valid.
            for _ in range(6):
                if hi - lo <= 1:
                    break
                mid = (lo + hi) // 2
                c_mid = _get_cost(method, mid)
                if c_mid <= b:
                    hi = mid
                    c_hi = c_mid
                else:
                    lo = mid
            return hi, c_hi

        else:
            # Search upward to find a valid upper bound quickly (exponential), then binary.
            lo = start_s  # known invalid
            hi = start_s
            c_hi = c_start
            step = 1
            for _ in range(6):
                if hi >= s_max:
                    break
                cand = hi + step
                if cand > s_max:
                    cand = s_max
                c_cand = _get_cost(method, cand)
                lo = hi
                hi = cand
                c_hi = c_cand
                if c_hi <= b:
                    break
                step <<= 1

            if c_hi > b:
                # Too expensive even at the loosest meaningful size; explicit no-op fallback.
                c_loose = _get_cost(method, s_max)
                return nq, c_loose

            # Binary search in (lo, hi] for smallest valid.
            for _ in range(6):
                if hi - lo <= 1:
                    break
                mid = (lo + hi) // 2
                c_mid = _get_cost(method, mid)
                if c_mid <= b:
                    hi = mid
                    c_hi = c_mid
                else:
                    lo = mid
            return hi, c_hi

    # Method prior: if circuit looks more entangled/communicating, prefer WC; else GV.
    # (Only acts as a tie-breaker; cost feasibility still dominates.)
    nonlocal_ratio = float(num_nonlocal_gates) / (float(number_instructions) if number_instructions else 1.0)
    prefer_wc = (float(entanglement_ratio) >= 0.45) or (float(program_communication) >= 0.25) or (nonlocal_ratio >= 0.12)

    gv0 = _get_cost("GV", s0)
    wc0 = _get_cost("WC", s0)

    # Choose initial method with feasibility and stable bias.
    gv_ok = (gv0 <= b)
    wc_ok = (wc0 <= b)
    if gv_ok and not wc_ok:
        chosen_method = "GV"
    elif wc_ok and not gv_ok:
        chosen_method = "WC"
    elif gv_ok and wc_ok:
        if prefer_wc:
            chosen_method = "WC" if wc0 <= int(gv0 * 1.15) else "GV"
        else:
            chosen_method = "GV" if gv0 <= int(wc0 * 1.15) else "WC"
    else:
        # Neither ok at s0: pick the cheaper one to guide the upward search.
        if prefer_wc:
            chosen_method = "WC" if wc0 <= gv0 else "GV"
        else:
            chosen_method = "GV" if gv0 <= wc0 else "WC"

    best_size, best_cost = _find_smallest_valid(chosen_method, s0)

    # If we had to fall back to no-op, pick a reasonable method label (doesn't matter downstream).
    if best_size >= nq:
        # Keep label stable with the initial preference and observed costs at s0.
        chosen_method = "WC" if (prefer_wc and wc0 <= gv0) else ("GV" if gv0 <= wc0 else "WC")
        return int(best_size), chosen_method

    return int(best_size), chosen_method
    # OE_END