""""Submodule for analyzing tomography experiments.

"""
import qcal.settings as settings

from qcal.characterization.characterize import Characterize
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.fitting.fit import FitDecayingCosine, FitExponential
from qcal.gate.single_qubit import Idle, X90, X, VirtualZ
from qcal.managers.classification_manager import ClassificationManager
from qcal.math.utils import reciprocal_uncertainty, round_to_order_error
from qcal.qpu.qpu import QPU
from qcal.units import MHz, us

import logging
import numpy as np
import pandas as pd

from numpy.typing import NDArray
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


def pauli_matrix(P: str) -> NDArray:
    """Calculates the matrix for a given Pauli operator (i.e., measurement basis).

    Args:
        P (str): Pauli operator/measurement basis (e.g., 'XY' for two qubits).

    Returns:
        NDArray: the matrix corresponding to the tensor product of the Pauli 
            operators.
    """
    I = np.eye(2, dtype=complex)
    sigma_x = np.array([[0, 1],
                        [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j],
                        [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0],
                        [0, -1]], dtype=complex)
    pauli_map = {'I': I,
                 'X': sigma_x,
                 'Y': sigma_y,
                 'Z': sigma_z}

    for i, pauli in enumerate(P):
        if i == 0:
            matrix = pauli_map[pauli]
        else:
            matrix = np.kron(matrix, pauli_map[pauli])
    return matrix


def rho_from_evals(evals: Dict) -> NDArray:
    """Calculate a density matrix from a dictionary of expectation values.

    Args:
        evals (Dict): expectation values. This should be a dictionary mapping
            a Pauli string to the expextation value.

    Returns:
        NDArray: density matrix.
    """
    n_qubits = len(list(evals.keys())[0])
    dim = 2 ** n_qubits

    if ''.join(['I'] * n_qubits) not in evals.keys():
        evals[''.join(['I'] * n_qubits)] = 1.0

    rho = np.zeros((dim,dim)).astype(complex)
    for p, ev in evals.items():
        rho += ev * pauli_matrix(p)

    return rho / dim