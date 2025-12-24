import os
from itertools import permutations
from multiprocessing import Process, Value

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Reset, Measure
from qiskit.circuit.library.standard_gates import XGate
from qiskit.converters import circuit_to_dag

from qos.types.types import Qernel
from qvm.compiler.dag import DAG
from qvm.virtual_circuit import VirtualCircuit

try:
    from FrozenQubits.helper_FrozenQubits import (
        drop_hotspot_node,
        halt_qubits,
        get_nodes_sorted_by_degree,
    )
except Exception:
    drop_hotspot_node = None
    halt_qubits = None
    get_nodes_sorted_by_degree = None

try:
    from FrozenQubits.helper_qaoa import pqc_QAOA, bind_QAOA
except Exception:
    pqc_QAOA = None
    bind_QAOA = None


def evolved_run(self, q: Qernel):
    # OE_BEGIN
    def basic_analysis(qernel: Qernel) -> None:
        qc = qernel.get_circuit()
        metadata = {
            "depth": qc.depth(),
            "num_qubits": qc.num_qubits,
            "num_clbits": qc.num_clbits,
            "num_nonlocal_gates": qc.num_nonlocal_gates(),
            "num_connected_components": qc.num_connected_components(),
            "number_instructions": qc.size(),
        }
        for key, value in qc.count_ops().items():
            if key == "measure":
                metadata["num_measurements"] = value
            elif key == "cx":
                metadata["num_cnot_gates"] = value
        qernel.edit_metadata(metadata)

    def _program_communication(qc: QuantumCircuit) -> float:
        num_qubits = qc.num_qubits
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")
        graph = nx.Graph()
        for op in dag.two_qubit_ops():
            q1, q2 = op.qargs
            graph.add_edge(qc.find_bit(q1).index, qc.find_bit(q2).index)
        degree_sum = sum(graph.degree(n) for n in graph.nodes)
        return degree_sum / (num_qubits * (num_qubits - 1))

    def _liveness(qc: QuantumCircuit) -> float:
        num_qubits = qc.num_qubits
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")
        activity_matrix = np.zeros((num_qubits, dag.depth()))
        for i, layer in enumerate(dag.layers()):
            for op in layer["partition"]:
                for qubit in op:
                    activity_matrix[qc.find_bit(qubit).index, i] = 1
        return np.sum(activity_matrix) / (num_qubits * dag.depth())

    def _parallelism(qc: QuantumCircuit) -> float:
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")
        return max(1 - (qc.depth() / len(dag.gate_nodes())), 0)

    def _measurement(qc: QuantumCircuit) -> float:
        qc = qc.copy()
        qc.remove_final_measurements()
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")
        reset_moments = 0
        gate_depth = dag.depth()
        for layer in dag.layers():
            reset_present = False
            for op in layer["graph"].op_nodes():
                if op.name == "reset":
                    reset_present = True
            if reset_present:
                reset_moments += 1
        return reset_moments / gate_depth

    def _entanglement_ratio(qc: QuantumCircuit) -> float:
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")
        return len(dag.two_qubit_ops()) / len(dag.gate_nodes())

    def _critical_depth(qc: QuantumCircuit) -> float:
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")
        n_ed = 0
        two_q_gates = set(op.name for op in dag.two_qubit_ops())
        for name in two_q_gates:
            try:
                n_ed += dag.count_ops_longest_path()[name]
            except KeyError:
                continue
        n_e = len(dag.two_qubit_ops())
        if n_ed == 0:
            return 0
        return n_ed / n_e

    def supermarq_features(qernel: Qernel) -> None:
        qc = qernel.get_circuit().copy()
        metadata = {
            "program_communication": _program_communication(qc),
            "liveness": _liveness(qc),
            "parallelism": _parallelism(qc),
            "measurement": _measurement(qc),
            "entanglement_ratio": _entanglement_ratio(qc),
            "critical_depth": _critical_depth(qc),
        }
        qernel.edit_metadata(metadata)

    def _is_qaoa(qc: QuantumCircuit) -> bool:
        must_have_ops_cx = ["cx", "h", "rz", "rx"]
        must_have_ops_rzz = ["h", "rzz", "rx"]
        checklist = {}
        ops = qc.count_ops()
        for op in ops:
            if op == "measure" or op == "barrier":
                continue
            if op in must_have_ops_cx or op in must_have_ops_rzz:
                checklist[op] = True
            else:
                return False
        flag1 = True
        flag2 = True
        for op in must_have_ops_cx:
            try:
                if checklist.get(op) is None:
                    flag1 = False
            except Exception:
                flag1 = False
        for op in must_have_ops_rzz:
            try:
                if checklist.get(op) is None:
                    flag2 = False
            except Exception:
                flag2 = False
        return flag1 or flag2

    def _qaoa_analysis(qernel: Qernel) -> None:
        if get_nodes_sorted_by_degree is None:
            raise RuntimeError("FrozenQubits dependencies are missing; QAOA analysis unavailable.")
        qc = qernel.get_circuit()
        h = {i: 0.0 for i in range(qc.num_qubits)}
        J = {}
        prev_pair = None
        prev_op = None
        for i in range(qc.num_qubits):
            for instr in qc.data:
                if instr.operation.name == "rzz":
                    param = instr.operation.params[0]
                    if param > 0:
                        J[(instr.qubits[0].index, instr.qubits[1].index)] = 1
                    else:
                        J[(instr.qubits[0].index, instr.qubits[1].index)] = -1
                if instr.operation.name == "cx":
                    if qc.find_bit(instr.qubits[1]).index == i:
                        op1 = qc.find_bit(instr.qubits[0]).index
                        v = J.get((op1, i))
                        if v is not None:
                            continue
                        prev_pair = (op1, i)
                        prev_op = "cx"
                if instr.operation.name == "rz":
                    if prev_op != "cx":
                        continue
                    if qc.find_bit(instr.qubits[0]).index == i:
                        param = instr.operation.params[0]
                        if param > 0:
                            J[prev_pair] = 1
                        else:
                            J[prev_pair] = -1
                        prev_op = None
        G = nx.Graph()
        G.add_edges_from(list(J.keys()))
        G.add_nodes_from(list(h.keys()))
        qaoa_metadata = {
            "h": h,
            "J": J,
            "offset": 0.0,
            "num_layers": 1,
            "hotspot_nodes": get_nodes_sorted_by_degree(G.adj),
        }
        qernel.edit_metadata(qaoa_metadata)

    def _frozen_qubits(qernel: Qernel, qubits_to_freeze: int) -> Qernel:
        if drop_hotspot_node is None:
            raise RuntimeError("FrozenQubits dependencies are missing; qubit freezing unavailable.")
        circuit = qernel.get_circuit()
        metadata = qernel.get_metadata()
        h = metadata["h"]
        J = metadata["J"]
        offset = metadata["offset"]
        num_layers = metadata["num_layers"]
        G = nx.Graph()
        G.add_edges_from(list(J.keys()))
        G.add_nodes_from(list(h.keys()))
        list_of_halting_qubits = []
        for _ in range(qubits_to_freeze):
            G, list_of_halting_qubits = drop_hotspot_node(
                G, list_of_fixed_vars=list_of_halting_qubits, verbosity=0
            )
        sub_Ising_list = halt_qubits(
            J=J, h=h, offset=offset, halting_list=list_of_halting_qubits
        )
        for sub_problem in sub_Ising_list:
            new_QAOA = pqc_QAOA(J=sub_problem["J"], h=sub_problem["h"], num_layers=num_layers)
            new_circuit = new_QAOA["qc"]
            gamma = np.random.uniform(0, 2 * np.pi, 1)[0]
            beta = np.random.uniform(0, np.pi, 1)[0]
            new_circuit = bind_QAOA(new_circuit, new_QAOA["params"], beta, gamma)
            sub_qernel = Qernel()
            sub_qernel.set_circuit(new_circuit)
            qaoa_metadata = {
                "h": sub_problem["h"],
                "J": sub_problem["J"],
                "offset": sub_problem["offset"],
                "num_layers": 1,
                "num_clbits": new_circuit.num_clbits,
            }
            sub_qernel.set_metadata(qaoa_metadata)
            qernel.add_virtual_subqernel(sub_qernel)
        return qernel

    def _apply_gv(qernel: Qernel, size_to_reach: int, budget: int) -> Qernel:
        from qvm.compiler.virtualization import OptimalDecompositionPass
        optimal_decomposition_pass = OptimalDecompositionPass(size_to_reach)
        vsqs = qernel.get_virtual_subqernels()
        if len(vsqs) > 0:
            for vsq in vsqs:
                qc = vsq.get_circuit()
                new_circuit = optimal_decomposition_pass.run(qc, budget)
                vsq.set_circuit(new_circuit)
        else:
            qc = qernel.get_circuit()
            new_circuit = optimal_decomposition_pass.run(qc, budget)
            sub_qernel = Qernel()
            sub_qernel.set_circuit(new_circuit)
            sub_qernel.set_metadata(qernel.get_metadata())
            qernel.add_virtual_subqernel(sub_qernel)
        return qernel

    def _apply_wc(qernel: Qernel, size_to_reach: int, budget: int) -> Qernel:
        from qvm.compiler.virtualization.wire_decomp import OptimalWireCutter
        optimal_wire_cutting_pass = OptimalWireCutter(size_to_reach)
        vsqs = qernel.get_virtual_subqernels()
        if len(vsqs) > 0:
            for vsq in vsqs:
                qc = vsq.get_circuit()
                new_circuit = optimal_wire_cutting_pass.run(qc, budget)
                vsq.set_circuit(new_circuit)
        else:
            qc = qernel.get_circuit()
            new_circuit = optimal_wire_cutting_pass.run(qc, budget)
            sub_qernel = Qernel()
            sub_qernel.set_circuit(new_circuit)
            sub_qernel.set_metadata(qernel.get_metadata())
            qernel.add_virtual_subqernel(sub_qernel)
        return qernel

    def _apply_best_cut(qernel: Qernel, size_to_reach: int) -> Qernel:
        metadata = qernel.get_metadata()
        pc = metadata["program_communication"]
        liveness = metadata["liveness"]
        if liveness > pc:
            return _apply_wc(qernel, size_to_reach, self.budget)
        return _apply_gv(qernel, size_to_reach, self.budget)

    def _gv_cost(qernel: Qernel, final_cost, size_to_reach: int) -> int:
        from qvm.compiler.virtualization import OptimalDecompositionPass
        optimal_decomposition_pass = OptimalDecompositionPass(size_to_reach)
        vsqs = qernel.get_virtual_subqernels()
        cost = 0
        if len(vsqs) > 0:
            highest_cost = 0
            for vsq in vsqs:
                qc = vsq.get_circuit()
                try:
                    cost = optimal_decomposition_pass.get_budget(qc)
                except ValueError:
                    cost = 1000
                if cost > highest_cost:
                    highest_cost = cost
        else:
            qc = qernel.get_circuit()
            try:
                cost = optimal_decomposition_pass.get_budget(qc)
            except ValueError:
                cost = 1000
        final_cost.value = cost
        return cost

    def _wc_cost(qernel: Qernel, final_cost, size_to_reach: int) -> int:
        from qvm.compiler.virtualization.wire_decomp import OptimalWireCutter
        optimal_wire_cutting_pass = OptimalWireCutter(size_to_reach)
        vsqs = qernel.get_virtual_subqernels()
        cost = 0
        if len(vsqs) > 0:
            highest_cost = 0
            for vsq in vsqs:
                qc = vsq.get_circuit()
                try:
                    cost = optimal_wire_cutting_pass.get_budget(qc)
                except ValueError:
                    cost = 1000
                if cost > highest_cost:
                    highest_cost = cost
        else:
            qc = qernel.get_circuit()
            try:
                cost = optimal_wire_cutting_pass.get_budget(qc)
            except ValueError:
                cost = 1000
        final_cost.value = cost
        return cost

    def computeCuttingCosts(qernel: Qernel, size_to_reach: int):
        gv_cost = Value("i", 1000)
        wc_cost = Value("i", 1000)
        p = Process(target=_gv_cost, args=(qernel, gv_cost, size_to_reach))
        p.start()
        p.join(600)
        if p.is_alive():
            p.terminate()
            p.join()
        gv_cost = gv_cost.value
        p = Process(target=_wc_cost, args=(qernel, wc_cost, size_to_reach))
        p.start()
        p.join(600)
        if p.is_alive():
            p.terminate()
            p.join()
        wc_cost = wc_cost.value
        return {"GV": gv_cost, "WC": wc_cost}

    def _dynamic_measure_and_reset(dag: DAG) -> None:
        if not hasattr(XGate(), "c_if"):
            return
        nodes = list(dag.nodes())
        for node in nodes:
            instr = dag.get_node_instr(node)
            if not isinstance(instr.operation, Measure):
                continue
            clbit = instr.clbits[0]
            next_node = next(dag.successors(node), None)
            if next_node is None:
                continue
            next_instr = dag.get_node_instr(next_node)
            if not isinstance(next_instr.operation, Reset):
                continue
            new_op = XGate().c_if(clbit, 1)
            new_instr = CircuitInstruction(new_op, next_instr.qubits, next_instr.clbits)
            dag.nodes[next_node]["instr"] = new_instr

    def _reuse(dag: DAG, qubit, reused_qubit) -> None:
        first_node = next(dag.nodes_on_qubit(reused_qubit))
        last_node = list(dag.nodes_on_qubit(qubit))[-1]
        reset_instr = CircuitInstruction(operation=Reset(), qubits=(reused_qubit,))
        reset_node = dag.add_instr_node(reset_instr)
        dag.add_edge(last_node, reset_node)
        dag.add_edge(reset_node, first_node)
        for node in dag.nodes:
            instr = dag.get_node_instr(node)
            new_qubits = [
                reused_qubit if instr_qubit == qubit else instr_qubit
                for instr_qubit in instr.qubits
            ]
            new_instr = CircuitInstruction(instr.operation, new_qubits, instr.clbits)
            dag.nodes[node]["instr"] = new_instr

    def _is_dependent_qubit(dag: DAG, u_qubit, v_qubit) -> bool:
        u_node = next(dag.nodes_on_qubit(u_qubit))
        v_node = list(dag.nodes_on_qubit(v_qubit))[-1]
        return nx.has_path(dag, u_node, v_node)

    def _find_valid_reuse_pairs(dag: DAG):
        for qubit, reused_qubit in permutations(dag.qubits, 2):
            if not _is_dependent_qubit(dag, reused_qubit, qubit):
                yield qubit, reused_qubit

    def _random_qubit_reuse(dag: DAG, size_to_reach: int = 1) -> None:
        num_qubits = len(dag.qubits)
        while num_qubits > size_to_reach:
            print("entered")
            qubit_pair = next(_find_valid_reuse_pairs(dag), None)
            if qubit_pair is None:
                break
            _reuse(dag, *qubit_pair)
            dag.compact()
            num_qubits -= 1

    def _apply_random_qubit_reuse(qernel: Qernel, size_to_reach: int) -> Qernel:
        vsqs = qernel.get_virtual_subqernels()
        if len(vsqs) > 0:
            for vsq in vsqs:
                qc = vsq.get_circuit()
                virtual_circuit = VirtualCircuit(qc)
                for frag, frag_circ in virtual_circuit.fragment_circuits.items():
                    dag = DAG(frag_circ)
                    _random_qubit_reuse(dag, size_to_reach)
                    _dynamic_measure_and_reset(dag)
                    virtual_circuit.replace_fragment_circuit(frag, dag.to_circuit())
                vsq.set_circuit(virtual_circuit)
        else:
            qc = qernel.get_circuit()
            virtual_circuit = VirtualCircuit(qc)
            for frag, frag_circ in virtual_circuit.fragment_circuits.items():
                dag = DAG(frag_circ)
                _random_qubit_reuse(dag, size_to_reach)
                _dynamic_measure_and_reset(dag)
                virtual_circuit.replace_fragment_circuit(frag, dag.to_circuit())
            sub_qernel = Qernel()
            sub_qernel.set_circuit(virtual_circuit)
            sub_qernel.set_metadata(qernel.get_metadata())
            qernel.add_virtual_subqernel(sub_qernel)
        return qernel

    basic_analysis(q)
    supermarq_features(q)

    for key in self.methods:
        self.methods[key] = True

    budget = self.budget
    if _is_qaoa(q.get_circuit()):
        _qaoa_analysis(q)
        metadata = q.get_metadata()
        num_cnots = metadata["num_nonlocal_gates"]
        hotspots = list(metadata["hotspot_nodes"].values())
        qubits_to_freeze = 0
        for i in range(2):
            if hotspots[i] / num_cnots >= 0.07:
                qubits_to_freeze = qubits_to_freeze + 1
        qubits_to_freeze = min(qubits_to_freeze, budget)
        if qubits_to_freeze > 0:
            q = _frozen_qubits(q, qubits_to_freeze)
            budget = budget - qubits_to_freeze

    if self.use_cost_search:
        size_to_reach = self.size_to_reach
        costs = computeCuttingCosts(q, size_to_reach)
        max_iters = int(os.getenv("QOS_COST_SEARCH_MAX_ITERS", "0"))
        iter_ctr = 0
        while (costs["GV"] <= budget or costs["WC"] <= budget) and size_to_reach > 2:
            iter_ctr += 1
            if max_iters > 0 and iter_ctr > max_iters:
                break
            size_to_reach = size_to_reach - 1
            costs = computeCuttingCosts(q, size_to_reach)

        iter_ctr = 0
        while costs["GV"] > budget and costs["WC"] > budget:
            iter_ctr += 1
            if max_iters > 0 and iter_ctr > max_iters:
                break
            size_to_reach = size_to_reach + 1
            costs = computeCuttingCosts(q, size_to_reach)

        if costs["GV"] <= budget or costs["WC"] <= budget:
            if costs["GV"] <= costs["WC"] or (costs["GV"] == 0 and costs["WC"] == 0):
                q = _apply_gv(q, size_to_reach, self.budget)
            else:
                q = _apply_wc(q, size_to_reach, self.budget)
    else:
        q = _apply_best_cut(q, self.size_to_reach)

    q = _apply_random_qubit_reuse(q, self.size_to_reach)

    # OE_END
    return q
