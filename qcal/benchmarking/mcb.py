"""Submodule for Mirror Circuit Benchmarking

Relevant papers:
- https://www.nature.com/articles/s41567-021-01409-7,
  https://arxiv.org/abs/2008.11294
"""
import logging
import numpy as np
# import sys

from typing import List, Dict

from qcal.circuit import CircuitSet
from qcal.compilation.compiler import Compiler, DEFAULT_COMPILER
from qcal.config import Config
from qcal.qpu.qpu import QPU

logger = logging.getLogger(__name__)


# https://colab.research.google.com/drive/1vLKdrdpiV7rc8-7mCDvCxw32311EfPve#scrollTo=lfKh72_hgYmi
# https://colab.research.google.com/drive/1gJdaX9vK8adBcK4xYEdO8gjyXFeTnK5D#scrollTo=eMoNNSL5pJCA

# libnames = ['pygsti', 'scipy', 'operator']
# for libname in libnames:
#     try:
#         lib = __import__(libname)
#     except ImportError:
#         print(sys.exc_info())
#     else:
#         globals()[libname] = lib

# import pygsti
# from pygsti.io import load_circuit_list, read_circuit_list
# from qiskit import QuantumCircuit


ESTIMATED_QUBIT_ERROR_RATE = 0.005
TARGET_POLARIZATION = 0.01 

def trim_depths(depths: List, width: int):
    """Heuristic function for automatically removing depths that are too long 
    
    This function can be used to trim MCB circuit depths so that they are not
    too long. If the circuit depths are too long, you will not get useful data
    and the runtime will be unnecessarily long. This function depends on
    ESTIMATED_QUBIT_ERROR_RATE and TARGET_POLARIZATION, which are global
    variables.

    Args:
        depths (List): list of circuit depths to trim
        width (int): circuit width

    Returns:
        _type_: _description_
    """
    max_depth = np.log(TARGET_POLARIZATION) / (
            width * np.log(1 - ESTIMATED_QUBIT_ERROR_RATE)
        )
    trimmed_depths = [d for d in depths if d < max_depth]
    n_depths = len(trimmed_depths)
    if n_depths < len(depths) and trimmed_depths[-1] < max_depth:
        trimmed_depths.append(depths[n_depths])
    
    return trimmed_depths


class MCB(QPU):

    def __init__(
            self, 
            config: Config, 
            compiler: Compiler = DEFAULT_COMPILER, 
            n_shots: int = 1024, 
            n_batches: int = 1, 
            n_circs_per_seq: int = 100,
            n_circs_per_depth_width: int = 30,
            qubits: List = None,
            depths: Dict = None,
            widths: List = None,
            qubit_subsets_per_width: Dict[List] = None,
            two_qubit_gate_density: float = 0.125,  # 1/8
        ) -> None:
        super().__init__(config, compiler, n_shots, n_batches, n_circs_per_seq)

        self._n_circs_per_depth_width = n_circs_per_depth_width
        self._qubits = qubits if qubits is not None else self._config.qubits
        self._depths = depths
        self._widths = widths
        self._qubit_subsets_per_width = qubit_subsets_per_width
        self._two_qubit_gate_density = two_qubit_gate_density

        self._n_qubits = len(self._qubits)
        self._exp_designs = {}  # Experiment designs (pygsti)
        self._proc_spec = None  # Processor spec (pygsti)

    @property
    def depths(self):
        return self._depths
    
    @property
    def n_qubits(self):
        return self.n_qubits
    
    @property
    def qubits(self):
        return self._qubits
    
    @property
    def qubit_subsets_per_width(self):
        return self._qubit_subsets_per_width
    
    @property
    def widths(self):
        return self._widths

    def generate_circuits(self):

        if self._widths is None:
            self._widths = [i for i in range(1, self._n_qubits + 1)]

        if self._depths is None:
            base_depths = [0,] +  [int(d) for d in 2**np.arange(1, 15)]
            self._depths = {w:trim_depths(base_depths,w) for w in self._widths}

        if self._qubit_subsets_per_width is None:
            self._qubit_subsets_per_width = {
                w: [tuple([q for q in self._qubits[:w]])] for w in self._widths
            }

        