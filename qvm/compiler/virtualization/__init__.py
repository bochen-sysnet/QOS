try:
    from .gate_decomp import OptimalDecompositionPass, BisectionPass
except ModuleNotFoundError:
    OptimalDecompositionPass = None
    BisectionPass = None

from .reduce_deps import GreedyDependencyBreaker
