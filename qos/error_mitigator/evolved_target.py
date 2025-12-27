import os

from qos.error_mitigator.analyser import IsQAOACircuitPass, QAOAAnalysisPass
from qos.error_mitigator.optimiser import FrozenQubitsPass
from qos.types.types import Qernel


def evolved_run(self, q: Qernel):
    # OE_BEGIN
    # You confirmed evolved_run is the LAST transformation.
    #
    # New objective: combined_score = 1 / (1 + rel_depth + rel_cnot + rel_overhead/10).
    # Since rel_* are vs QoS, we want to get as close as possible to QoS in depth/CNOT,
    # while keeping overhead reasonable.
    #
    # Empirical behavior from artifacts:
    # - Some transforms can catastrophically increase depth/CNOT (e.g., qaoa_pl1 when mis-applied).
    # - But doing nothing can also be worse than QoS (QoS often reduces depth/CNOT).
    #
    # Strategy here: "try a few safe candidates, measure via metadata, pick best."
    # Stateless + bounded tries:
    #   Candidates include: original, QR-only, (freeze + cut), cut-only, and (cut + QR) for QAOA.
    # We do acceptance by minimizing an internal objective:
    #   obj = depth + 1.1*cnot + 0.4*(num_circuits-1)
    # (Overhead is lightly penalized since rel_overhead/10 is smaller than rel_depth/rel_cnot terms.)

    # Ensure methods are enabled (some internal helpers expect this).
    try:
        for key in self.methods:
            self.methods[key] = True
    except Exception:
        pass

    # ---------------- helpers ----------------
    def _get_md(q_):
        try:
            md = q_.get_metadata() or {}
        except Exception:
            md = {}
        # Some pipelines store features nested; support both.
        if isinstance(md, dict) and isinstance(md.get("input_features", None), dict):
            # Keep both: top-level may include pass-produced fields (e.g., num_circuits),
            # while input_features has precomputed circuit stats.
            md2 = dict(md)
            md2["_input_features"] = md["input_features"]
            return md2
        return md

    def _pick(md, key, default=None):
        if not isinstance(md, dict):
            return default
        if key in md:
            return md.get(key, default)
        # also look in nested input_features if present
        feats = md.get("_input_features", None)
        if isinstance(feats, dict) and key in feats:
            return feats.get(key, default)
        return default

    def _i(md, key, default=0):
        try:
            return int(_pick(md, key, default))
        except Exception:
            return int(default)

    def _f(md, key, default=0.0):
        try:
            return float(_pick(md, key, default))
        except Exception:
            return float(default)

    def _num_circuits(md):
        # Try a few likely keys.
        for k in ("num_circuits", "qose_num_circuits", "num_fragment_circuits", "fragment_circuits"):
            v = _pick(md, k, None)
            if v is None:
                continue
            try:
                return max(1, int(v))
            except Exception:
                continue
        return 1

    def _depth(md):
        # Prefer post-pass fields if present, else fall back to input_features depth.
        for k in ("depth", "circuit_depth", "dag_depth"):
            v = _pick(md, k, None)
            if v is None:
                continue
            try:
                return int(v)
            except Exception:
                continue
        return 0

    def _cnot(md):
        for k in ("num_nonlocal_gates", "num_cnots", "cnot", "num_cnot"):
            v = _pick(md, k, None)
            if v is None:
                continue
            try:
                return int(v)
            except Exception:
                continue
        return 0

    def _obj(q_):
        md = _get_md(q_)
        d = _depth(md)
        c = _cnot(md)
        nc = _num_circuits(md)
        # If metadata missing, treat as bad.
        if d <= 0 and c <= 0:
            return float("inf")
        return float(d) + 1.1 * float(c) + 0.4 * float(max(0, nc - 1))

    def _cost(costs, k):
        try:
            return float(costs.get(k, float("inf")))
        except Exception:
            return float("inf")

    # ---------------- budget & base features ----------------
    try:
        budget = int(getattr(self, "budget", 0) or 0)
    except Exception:
        budget = 0

    md0 = _get_md(q)
    num_qubits = _i(md0, "num_qubits", 0)
    program_comm = _f(md0, "program_communication", 0.0)
    parallelism = _f(md0, "parallelism", 0.0)
    ent_ratio = _f(md0, "entanglement_ratio", 0.0)
    critical_depth = _f(md0, "critical_depth", 0.0)

    # GHZ-like: QoS often heavily optimizes these; aggressive transforms can blow up.
    # Still, allow QR-only candidate (sometimes routing helps), but avoid cutting.
    ghz_like = (ent_ratio >= 0.90 and parallelism <= 0.05 and critical_depth >= 0.90) or (
        ent_ratio >= 0.95 and parallelism <= 0.02
    )

    # ---------------- QAOA detection ----------------
    is_qaoa_pass = IsQAOACircuitPass()
    try:
        is_qaoa = bool(is_qaoa_pass.run(q))
    except Exception:
        is_qaoa = False

    # ---------------- size_to_reach search (for cutting) ----------------
    try:
        base_size = int(getattr(self, "size_to_reach", 2) or 2)
    except Exception:
        base_size = 2
    base_size = max(2, base_size)

    # Bounded iterations for cost feasibility
    try:
        max_iters = int(os.getenv("QOS_COST_SEARCH_MAX_ITERS", "6") or 6)
    except Exception:
        max_iters = 6
    if max_iters < 0:
        max_iters = 0

    def _find_feasible_size(q_):
        # choose conservative start near full width for QAOA (reduces fragmentation risk),
        # otherwise start at base_size.
        md = _get_md(q_)
        nq = _i(md, "num_qubits", num_qubits)
        start = base_size
        if nq > 0 and is_qaoa:
            start = max(start, min(nq, max(2, int(round(0.80 * nq)))))
        size = start

        try:
            costs = self.computeCuttingCosts(q_, size)
        except Exception:
            return size, float("inf"), float("inf")

        it = 0
        while _cost(costs, "GV") > budget and _cost(costs, "WC") > budget:
            it += 1
            if max_iters > 0 and it >= max_iters:
                break
            size += 1
            if nq > 0:
                size = min(size, nq)
            try:
                costs = self.computeCuttingCosts(q_, size)
            except Exception:
                break
            if nq > 0 and size >= nq:
                break

        return size, _cost(costs, "GV"), _cost(costs, "WC")

    # ---------------- candidate generation ----------------
    best_q = q
    best_obj = _obj(q)

    def _consider(qcand):
        nonlocal best_q, best_obj
        o = _obj(qcand)
        if o < best_obj:
            best_obj = o
            best_q = qcand

    # Candidate 0: do nothing
    _consider(q)

    # Candidate 1: QR-only (full width) — can reduce depth/CNOT for some non-QAOA (bv/vqe/hamsim),
    # but can also hurt; we score and only keep if better.
    if num_qubits >= 2:
        try:
            q_qr = self.applyQR(q, num_qubits)
            _consider(q_qr)
        except Exception:
            pass

    # If no budget, stop here
    if budget <= 0:
        return best_q

    # Avoid cutting on GHZ-like (too risky)
    if ghz_like:
        return best_q

    # QAOA candidates: allow (freeze<=1), cutting, and (cut+QR) optionally.
    if is_qaoa:
        # Conservative freezing (<=1) only if hotspot dominates; can help QoS-like improvements.
        q_fr = q
        try:
            QAOAAnalysisPass().run(q_fr)
        except Exception:
            pass
        mdA = _get_md(q_fr)
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
                if float(hotspots[0]) / float(num_cnots) >= 0.12:
                    freeze_n = 1
            except Exception:
                freeze_n = 0

        try:
            nq2 = _i(mdA, "num_qubits", num_qubits)
            freeze_n = min(freeze_n, budget, 1, max(0, nq2 - 1))
        except Exception:
            freeze_n = 0

        if freeze_n > 0:
            try:
                q_fr2 = FrozenQubitsPass(freeze_n).run(q_fr)
                _consider(q_fr2)
                q_fr = q_fr2
            except Exception:
                pass

        # Cutting candidates from (possibly frozen) q_fr
        size, gv_cost, wc_cost = _find_feasible_size(q_fr)
        feasible = (gv_cost <= budget) or (wc_cost <= budget)

        if feasible:
            # Decide preference: for high communication, WC tends to be better; else cheaper.
            mdF = _get_md(q_fr)
            pc = _f(mdF, "program_communication", program_comm)
            cd = _f(mdF, "critical_depth", critical_depth)

            prefer_wc = (pc >= 0.29) or (cd >= 0.52)

            # Try both feasible cuts (score-based selection)
            if gv_cost <= budget:
                try:
                    q_gv = self.applyGV(q_fr, size)
                    _consider(q_gv)
                    # Sometimes QR after cutting helps; try and score.
                    try:
                        q_gv_qr = self.applyQR(q_gv, size)
                        _consider(q_gv_qr)
                    except Exception:
                        pass
                except Exception:
                    pass

            if wc_cost <= budget:
                try:
                    q_wc = self.applyWC(q_fr, size)
                    _consider(q_wc)
                    try:
                        q_wc_qr = self.applyQR(q_wc, size)
                        _consider(q_wc_qr)
                    except Exception:
                        pass
                except Exception:
                    pass

            # A tiny bias: if both feasible, re-consider preferred method first (already done above),
            # but if preferred one wasn't feasible, the other still covered.

    # Non-QAOA: optionally try a very small cut only if program_comm is high (rare) AND budget allows,
    # otherwise cuts tend to regress; keep it minimal and score-gated.
    else:
        if program_comm >= 0.28 and num_qubits >= 4:
            size, gv_cost, wc_cost = _find_feasible_size(q)
            if gv_cost <= budget:
                try:
                    q_gv = self.applyGV(q, size)
                    _consider(q_gv)
                except Exception:
                    pass
            if wc_cost <= budget:
                try:
                    q_wc = self.applyWC(q, size)
                    _consider(q_wc)
                except Exception:
                    pass

    return best_q
    # OE_END
