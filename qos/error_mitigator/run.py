from multiprocessing import Process, Queue, Value
import multiprocessing as mp
import queue as queue_mod
import os
import time

from qos.error_mitigator.analyser import *
from qos.error_mitigator.optimiser import *

_TRACE_QUEUE = None


def _emit_trace(event: dict) -> None:
    if _TRACE_QUEUE is None:
        return
    try:
        _TRACE_QUEUE.put_nowait(event)
    except Exception:
        pass


def _drain_trace_queue(trace_queue) -> list:
    events = []
    while True:
        try:
            events.append(trace_queue.get_nowait())
        except queue_mod.Empty:
            break
        except Exception:
            break
    return events


def _choose_method(gv_cost, wc_cost) -> str:
    if gv_cost is not None and wc_cost is not None:
        return "GV" if gv_cost <= wc_cost else "WC"
    return "GV"


def _parse_trace_events(events, default_size):
    gv_cost_trace = []
    wc_cost_trace = []
    gv_time_trace = []
    wc_time_trace = []
    last_size = default_size
    last_gv_cost = None
    last_wc_cost = None
    for event in events:
        kind = event.get("kind")
        if kind == "PAIR":
            size = event.get("size", last_size)
            gv_cost = event.get("gv_cost")
            wc_cost = event.get("wc_cost")
            gv_sec = event.get("gv_sec")
            wc_sec = event.get("wc_sec")
            last_size = size
            last_gv_cost = gv_cost
            last_wc_cost = wc_cost
            gv_cost_trace.append(gv_cost)
            wc_cost_trace.append(wc_cost)
            gv_time_trace.append(gv_sec)
            wc_time_trace.append(wc_sec)
        elif kind == "GV":
            size = event.get("size", last_size)
            gv_cost = event.get("cost")
            gv_sec = event.get("sec")
            last_size = size
            last_gv_cost = gv_cost
            gv_cost_trace.append(gv_cost)
            gv_time_trace.append(gv_sec)
        elif kind == "WC":
            size = event.get("size", last_size)
            wc_cost = event.get("cost")
            wc_sec = event.get("sec")
            last_size = size
            last_wc_cost = wc_cost
            wc_cost_trace.append(wc_cost)
            wc_time_trace.append(wc_sec)
    return (
        gv_cost_trace,
        wc_cost_trace,
        gv_time_trace,
        wc_time_trace,
        last_size,
        last_gv_cost,
        last_wc_cost,
    )


def _is_verbose() -> bool:
    return os.getenv("QOS_VERBOSE", "").lower() in {"1", "true", "yes", "y"}


def _mark_timeout_trace(gv_time_trace, wc_time_trace) -> None:
    if gv_time_trace:
        gv_time_trace[-1] = -1.0
    else:
        gv_time_trace.append(-1.0)
    if wc_time_trace:
        wc_time_trace[-1] = -1.0
    else:
        wc_time_trace.append(-1.0)


def compute_gv_cost(q: Qernel, size_to_reach: int) -> int:
    gv_pass = GVOptimalDecompositionPass(size_to_reach)
    gv_cost_value = Value("i", 0)
    t0 = time.perf_counter()
    gv_pass.cost(q, gv_cost_value)
    gv_sec = time.perf_counter() - t0
    gv_cost = gv_cost_value.value
    _emit_trace(
        {"kind": "GV", "size": size_to_reach, "cost": gv_cost, "sec": gv_sec}
    )
    return gv_cost


def compute_wc_cost(q: Qernel, size_to_reach: int) -> int:
    wc_pass = OptimalWireCuttingPass(size_to_reach)
    wc_cost_value = Value("i", 0)
    t0 = time.perf_counter()
    wc_pass.cost(q, wc_cost_value)
    wc_sec = time.perf_counter() - t0
    wc_cost = wc_cost_value.value
    _emit_trace(
        {"kind": "WC", "size": size_to_reach, "cost": wc_cost, "sec": wc_sec}
    )
    return wc_cost


def _cost_search_worker(mitigator, q, size_to_reach, budget, result_queue, trace_queue):
    global _TRACE_QUEUE
    _TRACE_QUEUE = trace_queue
    try:
        size, method = mitigator._cost_search_impl(q, size_to_reach, budget)
        result_queue.put({"ok": True, "size": size, "method": method})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})
    finally:
        _TRACE_QUEUE = None

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
            t0 = time.perf_counter()
            gv_pass.cost(q, gv_cost)
            gv_sec = time.perf_counter() - t0
            gv_cost = gv_cost.value
        else:
            gv_cost = 1000

        if use_wc:
            self._wc_cost_calls += 1
            t0 = time.perf_counter()
            wc_pass.cost(q, wc_cost)
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
        _emit_trace(
            {
                "kind": "PAIR",
                "size": size_to_reach,
                "gv_cost": gv_cost,
                "wc_cost": wc_cost,
                "gv_sec": gv_sec,
                "wc_sec": wc_sec,
            }
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
        timeout_sec = int(os.getenv("QOS_COST_SEARCH_TIMEOUT_SEC", "60"))
        self._qose_cost_search_error = None
        if timeout_sec <= 0:
            trace_queue = queue_mod.Queue()
            global _TRACE_QUEUE
            _TRACE_QUEUE = trace_queue
            try:
                self._qose_cost_search_input_size = size_to_reach
                self._qose_cost_search_budget = budget
                size_to_reach, method = self._cost_search_impl(q, size_to_reach, budget)
            finally:
                _TRACE_QUEUE = None
            cost_time = time.perf_counter() - t0
            events = _drain_trace_queue(trace_queue)
            (
                gv_cost_trace,
                wc_cost_trace,
                gv_time_trace,
                wc_time_trace,
                last_size,
                last_gv_cost,
                last_wc_cost,
            ) = _parse_trace_events(events, size_to_reach)
            self._qose_gv_cost_trace = gv_cost_trace
            self._qose_wc_cost_trace = wc_cost_trace
            self._qose_gv_time_trace = gv_time_trace
            self._qose_wc_time_trace = wc_time_trace
            self._qose_cost_search_output_size = size_to_reach
            self._qose_cost_search_method = method
            return size_to_reach, method, cost_time, False

        mp_ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        result_queue = mp_ctx.Queue()
        trace_queue = mp_ctx.Queue()
        proc = mp_ctx.Process(
            target=_cost_search_worker,
            args=(self, q, size_to_reach, budget, result_queue, trace_queue),
        )
        proc.start()
        proc.join(timeout_sec)
        cost_time = time.perf_counter() - t0

        events = _drain_trace_queue(trace_queue)
        (
            gv_cost_trace,
            wc_cost_trace,
            gv_time_trace,
            wc_time_trace,
            last_size,
            last_gv_cost,
            last_wc_cost,
        ) = _parse_trace_events(events, size_to_reach)

        self._qose_cost_search_input_size = size_to_reach
        self._qose_cost_search_budget = budget
        self._qose_gv_cost_trace = gv_cost_trace
        self._qose_wc_cost_trace = wc_cost_trace
        self._qose_gv_time_trace = gv_time_trace
        self._qose_wc_time_trace = wc_time_trace

        if proc.is_alive():
            proc.terminate()
            proc.join()
            method = _choose_method(last_gv_cost, last_wc_cost)
            _mark_timeout_trace(gv_time_trace, wc_time_trace)
            self._qose_gv_time_trace = gv_time_trace
            self._qose_wc_time_trace = wc_time_trace
            self._qose_cost_search_output_size = last_size
            self._qose_cost_search_method = method
            return last_size, method, cost_time, True

        try:
            result = result_queue.get_nowait()
        except queue_mod.Empty:
            method = _choose_method(last_gv_cost, last_wc_cost)
            _mark_timeout_trace(gv_time_trace, wc_time_trace)
            self._qose_gv_time_trace = gv_time_trace
            self._qose_wc_time_trace = wc_time_trace
            self._qose_cost_search_output_size = last_size
            self._qose_cost_search_method = method
            return last_size, method, cost_time, True

        if not result.get("ok"):
            self._qose_cost_search_error = result.get("error")
            method = _choose_method(last_gv_cost, last_wc_cost)
            _mark_timeout_trace(gv_time_trace, wc_time_trace)
            self._qose_gv_time_trace = gv_time_trace
            self._qose_wc_time_trace = wc_time_trace
            self._qose_cost_search_output_size = last_size
            self._qose_cost_search_method = method
            return last_size, method, cost_time, True

        method = result["method"]
        size = result["size"]
        self._qose_cost_search_output_size = size
        self._qose_cost_search_method = method
        return size, method, cost_time, False
    
    def applyGV(self, q: Qernel, size_to_reach: int):
        if self.methods["GV"]:
            gv_pass = GVOptimalDecompositionPass(size_to_reach)
            
            #cost = gv_pass.cost(q)

            #if cost <= self.budget:
            if _is_verbose():
                print(f"[QOS] applyGV start size_to_reach={size_to_reach}", flush=True)
            t0 = time.perf_counter()
            q = gv_pass.run(q, self.budget)
            if _is_verbose():
                print(
                    f"[QOS] applyGV done size_to_reach={size_to_reach} "
                    f"sec={time.perf_counter() - t0:.2f}",
                    flush=True,
                )
        
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
            if _is_verbose():
                print(f"[QOS] applyWC start size_to_reach={size_to_reach}", flush=True)
            t0 = time.perf_counter()
            try:
                q = wc_pass.run(q, self.budget)
            except ValueError as exc:
                if _is_verbose():
                    print(
                        f"[QOS] applyWC failed size_to_reach={size_to_reach} "
                        f"error={exc}",
                        flush=True,
                    )
                return q
            if _is_verbose():
                print(
                    f"[QOS] applyWC done size_to_reach={size_to_reach} "
                    f"sec={time.perf_counter() - t0:.2f}",
                    flush=True,
                )
        
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
                    if _is_verbose():
                        print(
                            f"[QOS] cost_search start size_to_reach={size_to_reach} "
                            f"budget={budget}",
                            flush=True,
                        )
                    size_to_reach, method, cost_time, timed_out = self.cost_search(
                        q, size_to_reach, budget
                    )
                    if _is_verbose():
                        print(
                            f"[QOS] cost_search done size_to_reach={size_to_reach} "
                            f"method={method} sec={cost_time:.2f} timed_out={timed_out}",
                            flush=True,
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
                if _is_verbose():
                    print(
                        f"[QOS] cost_search start size_to_reach={size_to_reach} "
                        f"budget={self.budget}",
                        flush=True,
                    )
                size_to_reach, method, cost_time, timed_out = self.cost_search(
                    q, size_to_reach, self.budget
                )
                if _is_verbose():
                    print(
                        f"[QOS] cost_search done size_to_reach={size_to_reach} "
                        f"method={method} sec={cost_time:.2f} timed_out={timed_out}",
                        flush=True,
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
