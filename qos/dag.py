try:
    from qvm.compiler.dag import DAG, dag_to_qcg
except ModuleNotFoundError:
    class DAG:
        def __init__(self, circuit):
            self._circuit = circuit

        def to_circuit(self):
            return self._circuit

    def dag_to_qcg(*_args, **_kwargs):
        raise RuntimeError("dag_to_qcg is unavailable without qvm.")
