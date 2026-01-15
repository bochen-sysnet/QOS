from multiprocessing import Process, Queue, Value
import multiprocessing as mp
import os
import time

from qos.error_mitigator.analyser import *
from qos.error_mitigator.optimiser import *

_COST_SEARCH_ATTRS = (
    "_gv_cost_calls",
    "_wc_cost_calls",
    "_qose_cost_search_input_size",
    "_qose_cost_search_budget",
    "_qose_cost_search_output_size",
    "_qose_cost_search_method",
    "_qose_gv_cost_trace",
    "_qose_wc_cost_trace",
    "_qose_gv_time_trace",
    "_qose_wc_time_trace",
)


def compute_gv_cost(q: Qernel, size_to_reach: int) -> int:
    gv_pass = GVOptimalDecompositionPass(size_to_reach)
    gv_cost_value = Value("i", 0)
    gv_pass.cost(q, gv_cost_value)
    return gv_cost_value.value


def compute_wc_cost(q: Qernel, size_to_reach: int) -> int:
    wc_pass = OptimalWireCuttingPass(size_to_reach)
    wc_cost_value = Value("i", 0)
    wc_pass.cost(q, wc_cost_value)
    return wc_cost_value.value


def _cost_search_worker(
    mitigator,
    q,
    size_to_reach,
    budget,
    queue,
    state,
    gv_cost_trace,
    wc_cost_trace,
    gv_time_trace,
    wc_time_trace,
):
    gv_cost_orig = GVOptimalDecompositionPass.cost
    wc_cost_orig = OptimalWireCuttingPass.cost
    try:
        mitigator._qose_cost_search_input_size = size_to_reach
        mitigator._qose_cost_search_budget = budget
        mitigator._qose_cost_search_output_size = None
        mitigator._qose_cost_search_method = None
        mitigator._qose_gv_cost_trace = gv_cost_trace
        mitigator._qose_wc_cost_trace = wc_cost_trace
        mitigator._qose_gv_time_trace = gv_time_trace
        mitigator._qose_wc_time_trace = wc_time_trace
        state["size"] = size_to_reach
        state["gv_cost"] = None
        state["wc_cost"] = None

        def _gv_cost(self, *args, **kwargs):
            t0 = time.perf_counter()
            result = gv_cost_orig(self, *args, **kwargs)
            dt = time.perf_counter() - t0
            cost_val = None
            if len(args) >= 2:
                cost_val = args[1]
            else:
                cost_val = kwargs.get("cost") or kwargs.get("cost_value")
            if hasattr(cost_val, "value"):
                cost_val = cost_val.value
            gv_cost_trace.append(cost_val)
            gv_time_trace.append(dt)
            state["size"] = getattr(self, "_size_to_reach", state.get("size"))
            state["gv_cost"] = cost_val
            return result

        def _wc_cost(self, *args, **kwargs):
            t0 = time.perf_counter()
            result = wc_cost_orig(self, *args, **kwargs)
            dt = time.perf_counter() - t0
            cost_val = None
            if len(args) >= 2:
                cost_val = args[1]
            else:
                cost_val = kwargs.get("cost") or kwargs.get("cost_value")
            if hasattr(cost_val, "value"):
                cost_val = cost_val.value
            wc_cost_trace.append(cost_val)
            wc_time_trace.append(dt)
            state["size"] = getattr(self, "_size_to_reach", state.get("size"))
            state["wc_cost"] = cost_val
            return result

        GVOptimalDecompositionPass.cost = _gv_cost
        OptimalWireCuttingPass.cost = _wc_cost

        size, method = mitigator._cost_search_impl(q, size_to_reach, budget)
        mitigator._qose_cost_search_output_size = size
        mitigator._qose_cost_search_method = method
        state["size"] = size
        state["method"] = method
        attrs = {name: getattr(mitigator, name, None) for name in _COST_SEARCH_ATTRS}
        queue.put({"ok": True, "size": size, "method": method, "attrs": attrs})
    except Exception as exc:
        queue.put({"ok": False, "error": str(exc)})
    finally:
        GVOptimalDecompositionPass.cost = gv_cost_orig
        OptimalWireCuttingPass.cost = wc_cost_orig

class ErrorMitigator():
    budget: int
    methods: dict[str, bool]
    size_to_reach: int
    ideal_size_to_reach: int
    use_cost_search: bool
    collect_timing: bool
    timings: dict[str, float]

    def __init__(
        self,
        size_to_reach: int = 7,
        ideal_size_to_reach: int = 2,
        budget: int = 4,
        methods: List[str] = [],
        use_cost_search: bool = True,
        collect_timing: bool = False,
    ) -> None:
        self.size_to_reach = size_to_reach
        self.ideal_size_to_reach = ideal_size_to_reach
        self.budget = budget
        self.use_cost_search = use_cost_search
        self.collect_timing = collect_timing
        self.timings = {}
        self.methods = {
            "QF": False,
            "GV": False,
            "WC": False,
            "QR": False
        }

        for method in methods:
            self.methods[method] = True

    def estimateOptimalCuttingMethod(self, q: Qernel):
        metadata = q.get_metadata()

        pc = metadata["program_communication"]
        liveness = metadata['liveness']

        if liveness > pc:
            return "WC"
        else:
            return "GV"
        
    def computeCuttingCosts(self, q: Qernel, size_to_reach: int):
        verbose = os.getenv("QOS_VERBOSE", "").lower() in {"1", "true", "yes", "y"}
        if not hasattr(self, "_gv_cost_calls"):
            self._gv_cost_calls = 0
        if not hasattr(self, "_wc_cost_calls"):
            self._wc_cost_calls = 0
        gv_sec = 0.0
        wc_sec = 0.0
        use_gv = self.methods.get("GV", True)
        use_wc = self.methods.get("WC", True)
        if verbose:
            print(f"[QOS] computeCuttingCosts start size_to_reach={size_to_reach}", flush=True)
        gv_pass = GVOptimalDecompositionPass(size_to_reach)
        wc_pass = OptimalWireCuttingPass(size_to_reach)
        
        gv_cost = Value("i", 1000)
        wc_cost = Value("i", 1000)

        if use_gv:
            self._gv_cost_calls += 1
            p = Process(target=gv_pass.cost, args=(q, gv_cost))
            t0 = time.perf_counter()
            p.start()
            p.join(600)
            if p.is_alive():
                p.terminate()
                p.join()
            gv_sec = time.perf_counter() - t0
            gv_cost = gv_cost.value
        else:
            gv_cost = 1000

        if use_wc:
            self._wc_cost_calls += 1
            p = Process(target=wc_pass.cost, args=(q, wc_cost))
            t0 = time.perf_counter()
            p.start()
            p.join(600)
            if p.is_alive():
                p.terminate()
                p.join()
            wc_sec = time.perf_counter() - t0
            wc_cost = wc_cost.value
        else:
            wc_cost = 1000
            
        if verbose:
            print(
                f"[QOS] computeCuttingCosts done GV={gv_cost} ({gv_sec:.2f}s) "
                f"WC={wc_cost} ({wc_sec:.2f}s)",
                flush=True,
            )
        self._last_gv_sec = gv_sec
        self._last_wc_sec = wc_sec
        return {"GV": gv_cost, "WC": wc_cost}

    def _cost_search_impl(self, q: Qernel, size_to_reach: int, budget: int):
        costs = self.computeCuttingCosts(q, size_to_reach)
        max_iters = int(os.getenv("QOS_COST_SEARCH_MAX_ITERS", "0"))
        if os.getenv("QOS_VERBOSE", "").lower() in {"1", "true", "yes", "y"}:
            print(
                f"[QOS] cost_search init size_to_reach={size_to_reach} "
                f"GV={costs['GV']} WC={costs['WC']} budget={budget} "
                f"max_iters={max_iters}",
                flush=True,
            )
        iter_ctr = 0
        while (costs["GV"] <= budget or costs["WC"] <= budget) and size_to_reach > 2:
            iter_ctr += 1
            if max_iters > 0 and iter_ctr > max_iters:
                break
            size_to_reach = size_to_reach - 1
            costs = self.computeCuttingCosts(q, size_to_reach)
            if os.getenv("QOS_VERBOSE", "").lower() in {"1", "true", "yes", "y"}:
                print(
                    f"[QOS] cost_search shrink iter={iter_ctr} size_to_reach={size_to_reach} "
                    f"GV={costs['GV']} WC={costs['WC']} max_iters={max_iters}",
                    flush=True,
                )

        iter_ctr = 0
        while costs["GV"] > budget and costs["WC"] > budget:
            iter_ctr += 1
            if max_iters > 0 and iter_ctr > max_iters:
                break
            size_to_reach = size_to_reach + 1
            costs = self.computeCuttingCosts(q, size_to_reach)
            if os.getenv("QOS_VERBOSE", "").lower() in {"1", "true", "yes", "y"}:
                print(
                    f"[QOS] cost_search grow iter={iter_ctr} size_to_reach={size_to_reach} "
                    f"GV={costs['GV']} WC={costs['WC']} max_iters={max_iters}",
                    flush=True,
                )

        method = None
        if costs["GV"] <= costs["WC"]:
            method = "GV"
        else:
            method = "WC" 
        return size_to_reach, method

    def cost_search(self, q: Qernel, size_to_reach: int, budget: int):
        t0 = time.perf_counter()
        timeout_sec = int(os.getenv("QOS_COST_SEARCH_TIMEOUT_SEC", "600"))
        if timeout_sec <= 0:
            self._qose_cost_search_input_size = size_to_reach
            self._qose_cost_search_budget = budget
            size_to_reach, method = self._cost_search_impl(q, size_to_reach, budget)
            cost_time = time.perf_counter() - t0
            self._qose_cost_search_output_size = size_to_reach
            self._qose_cost_search_method = method
            return size_to_reach, method, cost_time, False

        queue = Queue()
        manager = mp.Manager()
        state = manager.dict()
        gv_cost_trace = manager.list()
        wc_cost_trace = manager.list()
        gv_time_trace = manager.list()
        wc_time_trace = manager.list()
        proc = Process(
            target=_cost_search_worker,
            args=(
                self,
                q,
                size_to_reach,
                budget,
                queue,
                state,
                gv_cost_trace,
                wc_cost_trace,
                gv_time_trace,
                wc_time_trace,
            ),
        )
        proc.start()
        proc.join(timeout_sec)
        try:
            cost_time = time.perf_counter() - t0
            self._qose_cost_search_input_size = size_to_reach
            self._qose_cost_search_budget = budget
            self._qose_gv_cost_trace = list(gv_cost_trace)
            self._qose_wc_cost_trace = list(wc_cost_trace)
            self._qose_gv_time_trace = list(gv_time_trace)
            self._qose_wc_time_trace = list(wc_time_trace)

            if proc.is_alive():
                proc.terminate()
                proc.join()
                size = state.get("size", size_to_reach)
                gv_cost = state.get("gv_cost")
                wc_cost = state.get("wc_cost")
                if gv_cost is not None and wc_cost is not None:
                    method = "GV" if gv_cost <= wc_cost else "WC"
                else:
                    method = "GV"
                self._qose_cost_search_output_size = size
                self._qose_cost_search_method = method
                return size, method, cost_time, True

            if queue.empty():
                size = state.get("size", size_to_reach)
                gv_cost = state.get("gv_cost")
                wc_cost = state.get("wc_cost")
                if gv_cost is not None and wc_cost is not None:
                    method = "GV" if gv_cost <= wc_cost else "WC"
                else:
                    method = "GV"
                self._qose_cost_search_output_size = size
                self._qose_cost_search_method = method
                return size, method, cost_time, True

            result = queue.get()
            if not result.get("ok"):
                size = state.get("size", size_to_reach)
                gv_cost = state.get("gv_cost")
                wc_cost = state.get("wc_cost")
                if gv_cost is not None and wc_cost is not None:
                    method = "GV" if gv_cost <= wc_cost else "WC"
                else:
                    method = "GV"
                self._qose_cost_search_output_size = size
                self._qose_cost_search_method = method
                return size, method, cost_time, True

            for name, value in result.get("attrs", {}).items():
                if name in {
                    "_qose_gv_cost_trace",
                    "_qose_wc_cost_trace",
                    "_qose_gv_time_trace",
                    "_qose_wc_time_trace",
                }:
                    if value is None:
                        continue
                    try:
                        setattr(self, name, list(value))
                    except TypeError:
                        setattr(self, name, value)
                    continue
                setattr(self, name, value)
            return result["size"], result["method"], cost_time, False
        finally:
            try:
                manager.shutdown()
            except Exception:
                pass
    
    def applyGV(self, q: Qernel, size_to_reach: int):
        if self.methods["GV"]:
            gv_pass = GVOptimalDecompositionPass(size_to_reach)
            
            #cost = gv_pass.cost(q)

            #if cost <= self.budget:
            q = gv_pass.run(q, self.budget)
        
        return q

    def applyBestCut(self, q: Qernel, size_to_reach: int):
        method = self.estimateOptimalCuttingMethod(q)
        if method == "GV":
            return self.applyGV(q, size_to_reach)
        return self.applyWC(q, size_to_reach)
    
    def applyWC(self, q: Qernel, size_to_reach: int):
        if self.methods["WC"]:
            wc_pass = OptimalWireCuttingPass(size_to_reach)
            
            #cost = wc_pass.cost(q)

            #if cost <= self.budget:
            q = wc_pass.run(q, self.budget)
        
        return q
    
    def applyQR(self, q: Qernel, size_to_reach: int):
        if self.methods["QR"]:
            qr_pass = RandomQubitReusePass(size_to_reach)

            q = qr_pass.run(q)

        return q

    def run(self, q: Qernel):
        if self.collect_timing:
            self.timings = {}
            total_start = time.perf_counter()

        analysis_pass = BasicAnalysisPass()
        supermarq_features_pass = SupermarqFeaturesAnalysisPass()
        
        if self.collect_timing:
            t0 = time.perf_counter()
            analysis_pass.run(q)
            supermarq_features_pass.run(q)
            self.timings["analysis"] = time.perf_counter() - t0
        else:
            analysis_pass.run(q)
            supermarq_features_pass.run(q)

        flag = True
        for method in self.methods.values():
            if method:
                flag = False
        
        if flag:
            for k in self.methods.keys():
                self.methods[k] = True

        if self.methods["QF"]:
            is_qaoa_pass = IsQAOACircuitPass()
            budget = self.budget
            if is_qaoa_pass.run(q):
                qaoa_analysis_pass = QAOAAnalysisPass()
                if self.collect_timing:
                    t0 = time.perf_counter()
                    qaoa_analysis_pass.run(q)
                    self.timings["qaoa_analysis"] = time.perf_counter() - t0
                else:
                    qaoa_analysis_pass.run(q)
                metadata = q.get_metadata()
                num_cnots = metadata["num_nonlocal_gates"]
                hotspots = list(metadata["hotspot_nodes"].values())
                qubits_to_freeze = 0

                for i in range(2):
                    if hotspots[i] / num_cnots >= 0.07:                        
                        qubits_to_freeze = qubits_to_freeze + 1

                qubits_to_freeze = min(qubits_to_freeze, budget)

                if qubits_to_freeze > 0:
                    QF_pass = FrozenQubitsPass(qubits_to_freeze)
                    if self.collect_timing:
                        t0 = time.perf_counter()
                        q = QF_pass.run(q)
                        self.timings["qf"] = time.perf_counter() - t0
                    else:
                        q = QF_pass.run(q)
                    budget = budget - qubits_to_freeze

            if self.methods["GV"] and self.methods["WC"]:
                if not self.use_cost_search:
                    if self.collect_timing:
                        t0 = time.perf_counter()
                        q = self.applyBestCut(q, self.size_to_reach)
                        self.timings["cut_select"] = time.perf_counter() - t0
                    else:
                        q = self.applyBestCut(q, self.size_to_reach)
                else:
                    size_to_reach = self.size_to_reach
                    size_to_reach, method, cost_time, timed_out = self.cost_search(
                        q, size_to_reach, budget
                    )
                    if self.collect_timing:
                        self.timings["cost_search"] = cost_time

                    if method == "GV":
                        if self.collect_timing:
                            t0 = time.perf_counter()
                            q = self.applyGV(q, size_to_reach)
                            self.timings["gv"] = time.perf_counter() - t0
                        else:
                            q = self.applyGV(q, size_to_reach)
                    else:
                        if self.collect_timing:
                            t0 = time.perf_counter()
                            q = self.applyWC(q, size_to_reach)
                            self.timings["wc"] = time.perf_counter() - t0
                        else:
                            q = self.applyWC(q, size_to_reach)          
            elif self.methods["GV"]:
                if self.collect_timing:
                    t0 = time.perf_counter()
                    q = self.applyGV(q, self.size_to_reach)
                    self.timings["gv"] = time.perf_counter() - t0
                else:
                    q = self.applyGV(q, self.size_to_reach)

            elif self.methods["WC"]:
                if self.collect_timing:
                    t0 = time.perf_counter()
                    q = self.applyWC(q, self.size_to_reach)
                    self.timings["wc"] = time.perf_counter() - t0
                else:
                    q = self.applyWC(q, self.size_to_reach)
        
        elif self.methods["GV"] and self.methods["WC"]:
            if not self.use_cost_search:
                if self.collect_timing:
                    t0 = time.perf_counter()
                    q = self.applyBestCut(q, self.size_to_reach)
                    self.timings["cut_select"] = time.perf_counter() - t0
                else:
                    q = self.applyBestCut(q, self.size_to_reach)
            else:
                size_to_reach = self.size_to_reach
                size_to_reach, method, cost_time, timed_out = self.cost_search(
                    q, size_to_reach, self.budget
                )
                if self.collect_timing:
                    self.timings["cost_search"] = cost_time

                if method == "GV":
                    if self.collect_timing:
                        t0 = time.perf_counter()
                        q = self.applyGV(q, size_to_reach)
                        self.timings["gv"] = time.perf_counter() - t0
                    else:
                        q = self.applyGV(q, size_to_reach)
                else:
                    if self.collect_timing:
                        t0 = time.perf_counter()
                        q = self.applyWC(q, size_to_reach)
                        self.timings["wc"] = time.perf_counter() - t0
                    else:
                        q = self.applyWC(q, size_to_reach)
        elif self.methods["GV"]:
             if self.collect_timing:
                 t0 = time.perf_counter()
                 q = self.applyGV(q, self.size_to_reach)
                 self.timings["gv"] = time.perf_counter() - t0
             else:
                 q = self.applyGV(q, self.size_to_reach)
        elif self.methods["WC"]:
            if self.collect_timing:
                t0 = time.perf_counter()
                q = self.applyWC(q, self.size_to_reach)
                self.timings["wc"] = time.perf_counter() - t0
            else:
                q = self.applyWC(q, self.size_to_reach)
        
        if self.methods["QR"]:
            if self.collect_timing:
                t0 = time.perf_counter()
                q = self.applyQR(q, self.size_to_reach)
                self.timings["qr"] = time.perf_counter() - t0
            else:
                q = self.applyQR(q, self.size_to_reach)        

        if self.collect_timing:
            self.timings["total"] = time.perf_counter() - total_start
        
        return q
