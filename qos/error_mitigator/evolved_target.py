import os

from qos.error_mitigator.analyser import IsQAOACircuitPass, QAOAAnalysisPass
from qos.error_mitigator.optimiser import FrozenQubitsPass
from qos.types.types import Qernel


def evolved_run(self, q: Qernel):
    # OE_BEGIN
    # Revert to the previously stable structure:
    # - Use computeCuttingCosts-driven search to select size_to_reach under budget.
    # - Prefer GV (never WC) when feasible.
    # - Apply QR ONLY when we have evidence it's safe:
    #     (a) costs feasible at chosen size, and
    #     (b) circuit is not small/simple.
    #
    # Empirical: QR without this context explodes depth/CNOT for many benchmarks.
    if hasattr(self, "methods") and isinstance(self.methods, dict):
        for key in self.methods:
            self.methods[key] = True

    budget = int(getattr(self, "budget", 0) or 0)
    size_to_reach = int(getattr(self, "size_to_reach", 2) or 2)

    meta = {}
    try:
        meta = q.get_metadata() or {}
    except Exception:
        meta = {}

    num_qubits = int(meta.get("num_qubits", 0) or 0)
    depth = float(meta.get("depth", 0) or 0)
    num_cnots = int(meta.get("num_nonlocal_gates", 0) or 0)

    # Clamp size_to_reach.
    if num_qubits > 0:
        size_to_reach = max(2, min(size_to_reach, num_qubits))
    else:
        size_to_reach = max(2, size_to_reach)

    # Small/simple circuits: avoid QR (it explodes depth/CNOT in our tests).
    simple_circuit = (depth <= 15) or (num_cnots <= 12)

    def _safe_costs(q_in, s):
        try:
            c = self.computeCuttingCosts(q_in, s) or {}
        except Exception:
            c = {}
        gv = c.get("GV", float("inf"))
        wc = c.get("WC", float("inf"))
        try:
            gv = float(gv)
        except Exception:
            gv = float("inf")
        try:
            wc = float(wc)
        except Exception:
            wc = float("inf")
        return gv, wc

    # If no budget, do the minimum: do nothing (avoid QR).
    # (Earlier attempts to always QR were catastrophic.)
    if budget <= 0:
        return q

    # -------------------------
    # Budget-guided search for size_to_reach (bounded iterations)
    # -------------------------
    max_iters = int(os.getenv("QOS_COST_SEARCH_MAX_ITERS", "6"))
    if max_iters < 1:
        max_iters = 1

    lo = 2
    hi = num_qubits if num_qubits > 0 else max(size_to_reach, 2)

    # Start from requested size_to_reach, adjust to find a feasible size.
    best = size_to_reach
    gv0, wc0 = _safe_costs(q, best)

    def _gv_feasible(gv_val):
        return gv_val <= budget

    it = 0
    if _gv_feasible(gv0):
        # Try decreasing to find smallest feasible (reduces chance of QR blow-ups).
        step = 1
        cur = best
        last_good = cur
        first_bad = None

        while it < max_iters and cur - step >= lo:
            it += 1
            nxt = cur - step
            g2, _ = _safe_costs(q, nxt)
            if _gv_feasible(g2):
                last_good = nxt
                best = nxt
                cur = nxt
                step *= 2
            else:
                first_bad = nxt
                break

        # Binary search between first_bad and last_good.
        if first_bad is not None and last_good - 1 >= lo:
            left = first_bad
            right = last_good
            while it < max_iters and right - left > 1:
                it += 1
                mid = (left + right) // 2
                g3, _ = _safe_costs(q, mid)
                if _gv_feasible(g3):
                    right = mid
                    best = mid
                else:
                    left = mid
    else:
        # Try increasing to find a feasible size.
        step = 1
        cur = best
        found = None
        while it < max_iters:
            it += 1
            nxt = cur + step
            if num_qubits > 0 and nxt > num_qubits:
                break
            g2, _ = _safe_costs(q, nxt)
            if _gv_feasible(g2):
                found = nxt
                best = nxt
                break
            cur = nxt
            step *= 2

        # Binary search down to minimal feasible.
        if found is not None:
            left = cur
            right = found
            while it < max_iters and right - left > 1:
                it += 1
                mid = (left + right) // 2
                g3, _ = _safe_costs(q, mid)
                if _gv_feasible(g3):
                    right = mid
                    best = mid
                else:
                    left = mid

    size_to_reach = max(2, best)
    if num_qubits > 0:
        size_to_reach = min(size_to_reach, num_qubits)

    # Final feasibility at chosen size (for safety gating).
    gv_cost, _ = _safe_costs(q, size_to_reach)
    gv_ok = (gv_cost <= budget)

    # Apply GV only if feasible.
    if gv_ok:
        try:
            q = self.applyGV(q, size_to_reach)
        except Exception:
            pass

    # Apply QR only if:
    # - GV was feasible at this size (strong signal we are in the "normal" regime)
    # - AND circuit is not simple (QR tends to explode simple circuits)
    if gv_ok and (not simple_circuit):
        try:
            q = self.applyQR(q, size_to_reach)
        except Exception:
            pass

    # OE_END
    return q
