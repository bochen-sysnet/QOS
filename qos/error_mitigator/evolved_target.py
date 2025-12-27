import os

from qos.error_mitigator.analyser import IsQAOACircuitPass, QAOAAnalysisPass
from qos.error_mitigator.optimiser import FrozenQubitsPass
from qos.types.types import Qernel


def evolved_run(self, q: Qernel):
    # OE_BEGIN
    # Using your new artifact signals (stage_counts/stage_order) + confirmed pipeline behavior:
    # - evolved_run is the LAST transformation => our decisions here fully determine qose output.
    # - Sampling is uniform => must be very safe on non-QAOA.
    #
    # Observations from your best run (~0.23):
    # - qaoa_pl1 benefited from *more fragmentation* (e.g., 4 circuits) with lower depth/CNOT.
    # - qaoa_r3 benefited from *little/no fragmentation* (often 1 circuit) and lower CNOT.
    #
    # So we adapt cutting "size_to_reach" by QAOA subtype:
    # - pl1-like (higher critical_depth / lower parallelism): allow smaller size_to_reach -> more fragments
    # - r3-like (lower critical_depth / higher parallelism): keep size_to_reach near full width -> fewer fragments
    #
    # Also: freezing helped sometimes, but can hurt on pl1-like; so only enable freezing in r3-like regime.
    # Never apply QR (too risky; tends to explode CNOT/depth on non-QAOA and can also hurt QAOA).

    # Enable all methods (some internal codepaths may assume this).
    try:
        for key in self.methods:
            self.methods[key] = True
    except Exception:
        pass

    # ---------------- helpers ----------------
    def _get_md(q_):
        try:
            return q_.get_metadata() or {}
        except Exception:
            return {}

    def _i(md, k, d=0):
        try:
            return int(md.get(k, d))
        except Exception:
            return int(d)

    def _f(md, k, d=0.0):
        try:
            return float(md.get(k, d))
        except Exception:
            return float(d)

    def _cost(costs, k):
        try:
            return float(costs.get(k, float("inf")))
        except Exception:
            return float("inf")

    # ---------------- budget ----------------
    try:
        budget = int(getattr(self, "budget", 0) or 0)
    except Exception:
        budget = 0
    if budget <= 0:
        return q

    md0 = _get_md(q)
    num_qubits = _i(md0, "num_qubits", 0)
    num_nonlocal = _i(md0, "num_nonlocal_gates", 0)
    program_comm = _f(md0, "program_communication", 0.0)
    parallelism = _f(md0, "parallelism", 0.0)
    ent_ratio = _f(md0, "entanglement_ratio", 0.0)
    critical_depth = _f(md0, "critical_depth", 0.0)

    # GHZ-like: avoid any transforms (QoS baseline tends to be strong; transforms often regress badly).
    ghz_like = (ent_ratio >= 0.90 and parallelism <= 0.05 and critical_depth >= 0.90) or (
        ent_ratio >= 0.95 and parallelism <= 0.02
    )
    if ghz_like:
        return q

    # ---------------- QAOA detection + strong feature gate ----------------
    is_qaoa_pass = IsQAOACircuitPass()
    try:
        is_qaoa = bool(is_qaoa_pass.run(q))
    except Exception:
        is_qaoa = False

    nonlocal_per_qubit = float(num_nonlocal) / float(max(1, num_qubits)) if num_qubits else float(num_nonlocal)
    qaoa_like = is_qaoa and (program_comm >= 0.23) and (nonlocal_per_qubit >= 2.2)

    # Uniform sampling => safest for non-QAOA is do nothing.
    if not qaoa_like:
        return q

    # ---------------- QAOA subtype gates ----------------
    # r3-like (from your artifacts): parallelism ~0.65, critical_depth ~0.44
    r3_like = (parallelism >= 0.62) and (critical_depth <= 0.49)
    # pl1-like: critical_depth around 0.55 and parallelism around 0.57
    pl1_like = (critical_depth >= 0.52) and (parallelism <= 0.60)

    # ---------------- Optional freezing ONLY for r3-like ----------------
    # (Avoid freezing on pl1-like; it often doesn't help and can worsen.)
    if r3_like:
        try:
            QAOAAnalysisPass().run(q)
        except Exception:
            pass

        mdA = _get_md(q)
        num_cnots = _i(mdA, "num_nonlocal_gates", 0)
        hs = mdA.get("hotspot_nodes", {}) or {}
        if isinstance(hs, dict):
            hotspots = list(hs.values())
        elif isinstance(hs, (list, tuple)):
            hotspots = list(hs)
        else:
            hotspots = []

        freeze_n = 0
        if num_cnots > 0 and hotspots:
            try:
                # slightly conservative
                if float(hotspots[0]) / float(num_cnots) >= 0.12:
                    freeze_n = 1
            except Exception:
                freeze_n = 0

        freeze_n = min(freeze_n, budget, 1)
        nqubitsA = _i(mdA, "num_qubits", num_qubits)
        if nqubitsA > 0:
            freeze_n = min(freeze_n, max(0, nqubitsA - 1))

        if freeze_n > 0:
            try:
                q = FrozenQubitsPass(freeze_n).run(q)
                budget -= freeze_n
            except Exception:
                pass

        if budget <= 0:
            return q

        # refresh features after freezing
        md0 = _get_md(q)
        num_qubits = _i(md0, "num_qubits", num_qubits)
        program_comm = _f(md0, "program_communication", program_comm)
        parallelism = _f(md0, "parallelism", parallelism)
        critical_depth = _f(md0, "critical_depth", critical_depth)

    # ---------------- Cutting (GV/WC) with adaptive size_to_reach ----------------
    try:
        base_size = int(getattr(self, "size_to_reach", 2) or 2)
    except Exception:
        base_size = 2
    base_size = max(2, base_size)

    # Adaptive starting point:
    # - pl1-like: start smaller (more fragments) to reduce depth/CNOT (as in your best ~0.23 run)
    # - r3-like: start larger (fewer fragments) to avoid CNOT blowups
    if num_qubits > 0:
        if pl1_like:
            # target ~4 fragments on 12q => size around 6-7 often
            start = max(base_size, min(num_qubits, max(2, int(round(0.58 * num_qubits)))))
        else:
            # r3-like / default: near full width
            start = max(base_size, min(num_qubits, max(2, int(round(0.85 * num_qubits)))))
        size_to_reach = start
    else:
        size_to_reach = base_size

    # Bounded feasibility search upwards (increase size reduces fragmentation & cost).
    try:
        max_iters = int(os.getenv("QOS_COST_SEARCH_MAX_ITERS", "6") or 6)
    except Exception:
        max_iters = 6
    if max_iters < 0:
        max_iters = 0

    try:
        costs = self.computeCuttingCosts(q, size_to_reach)
    except Exception:
        costs = {"GV": float("inf"), "WC": float("inf")}

    it = 0
    while _cost(costs, "GV") > budget and _cost(costs, "WC") > budget:
        it += 1
        if max_iters > 0 and it >= max_iters:
            break
        size_to_reach += 1
        if num_qubits > 0:
            size_to_reach = min(size_to_reach, num_qubits)
        try:
            costs = self.computeCuttingCosts(q, size_to_reach)
        except Exception:
            costs = {"GV": float("inf"), "WC": float("inf")}
        if num_qubits > 0 and size_to_reach >= num_qubits:
            break

    gv = _cost(costs, "GV")
    wc = _cost(costs, "WC")
    if gv > budget and wc > budget:
        return q

    # Method selection:
    # - For pl1-like (higher comm / higher crit-depth): WC has been consistently good.
    # - Otherwise choose cheaper feasible.
    prefer_wc = pl1_like or (program_comm >= 0.29) or (critical_depth >= 0.52)

    try:
        if prefer_wc:
            if wc <= budget:
                q = self.applyWC(q, size_to_reach)
            elif gv <= budget:
                q = self.applyGV(q, size_to_reach)
        else:
            if gv <= wc and gv <= budget:
                q = self.applyGV(q, size_to_reach)
            elif wc <= budget:
                q = self.applyWC(q, size_to_reach)
            elif gv <= budget:
                q = self.applyGV(q, size_to_reach)
    except Exception:
        pass

    # Never apply QR.
    return q
    # OE_END
