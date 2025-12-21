from multiprocessing import Process, Value
import os
import time

from qos.error_mitigator.analyser import *
from qos.error_mitigator.optimiser import *

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
        gv_pass = GVOptimalDecompositionPass(size_to_reach)
        wc_pass = OptimalWireCuttingPass(size_to_reach)
        try:
            gv_cost = Value("i", 1000)
            wc_cost = Value("i", 1000)

            p = Process(target=gv_pass.cost, args=(q, gv_cost))
            p.start()
            p.join(600)
            if p.is_alive():
                p.terminate()
                p.join()
            gv_cost = gv_cost.value
            p = Process(target=wc_pass.cost, args=(q, wc_cost))
            p.start()
            p.join(600)
            if p.is_alive():
                p.terminate()
                p.join()
            wc_cost = wc_cost.value
        except Exception:
            class _CostBox:
                def __init__(self, value: int):
                    self.value = value

            gv_box = _CostBox(1000)
            wc_box = _CostBox(1000)
            try:
                gv_pass.cost(q, gv_box)
            except ValueError:
                gv_box.value = 1000
            try:
                wc_pass.cost(q, wc_box)
            except ValueError:
                wc_box.value = 1000
            gv_cost = gv_box.value
            wc_cost = wc_box.value

        return {"GV": gv_cost, "WC": wc_cost}
    
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
                    cost_time = 0.0
                    t0 = time.perf_counter()
                    costs = self.computeCuttingCosts(q, size_to_reach)
                    cost_time += time.perf_counter() - t0
                    max_iters = int(os.getenv("QOS_COST_SEARCH_MAX_ITERS", "6"))
                    iter_ctr = 0
                    while (costs["GV"] <= budget or costs["WC"] <=budget) and size_to_reach > 2:
                        iter_ctr += 1
                        if iter_ctr > max_iters:
                            break
                        size_to_reach = size_to_reach - 1
                        t0 = time.perf_counter()
                        costs = self.computeCuttingCosts(q, size_to_reach)
                        cost_time += time.perf_counter() - t0
                
                    iter_ctr = 0
                    while costs["GV"] > budget and costs["WC"] > budget:
                        iter_ctr += 1
                        if iter_ctr > max_iters:
                            break
                        size_to_reach = size_to_reach + 1
                        t0 = time.perf_counter()
                        costs = self.computeCuttingCosts(q, size_to_reach)
                        cost_time += time.perf_counter() - t0
                    if self.collect_timing:
                        self.timings["cost_search"] = cost_time

                    if costs["GV"] <= budget or costs["WC"] <= budget:
                        if costs["GV"] <= costs["WC"] or (costs["GV"] == 0 and costs["WC"] == 0):
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
                cost_time = 0.0
                t0 = time.perf_counter()
                costs = self.computeCuttingCosts(q, size_to_reach)
                cost_time += time.perf_counter() - t0
                max_iters = int(os.getenv("QOS_COST_SEARCH_MAX_ITERS", "6"))
                iter_ctr = 0

                while (costs["GV"] <= self.budget or costs["WC"] <= self.budget) and size_to_reach > 2:
                    iter_ctr += 1
                    if iter_ctr > max_iters:
                        break
                    size_to_reach = size_to_reach - 1
                    t0 = time.perf_counter()
                    costs = self.computeCuttingCosts(q, size_to_reach)                
                    cost_time += time.perf_counter() - t0
                
                iter_ctr = 0
                while costs["GV"] > self.budget and costs["WC"] > self.budget:
                    iter_ctr += 1
                    if iter_ctr > max_iters:
                        break
                    size_to_reach = size_to_reach + 1
                    t0 = time.perf_counter()
                    costs = self.computeCuttingCosts(q, size_to_reach)
                    cost_time += time.perf_counter() - t0
                if self.collect_timing:
                    self.timings["cost_search"] = cost_time

                if costs["GV"] <= self.budget or costs["WC"] <= self.budget:
                    if costs["GV"] <= costs["WC"] or (costs["GV"] == 0 and costs["WC"] == 0):
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
