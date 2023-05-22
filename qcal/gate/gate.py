"""Submodule for defining the basic gate class.
"""
import numpy as np
# import sympy

from numpy.typing import NDArray
from sympy import Matrix
from typing import Tuple, Union


class Gate:

    __slots__ = ['_matrix', '_qubits']

    def __init__(self, 
            matrix: NDArray, 
            qubits: Union[int, Tuple] = None
        ) -> None:
        """Initialize a gate using its matrix definition.

        Args:
            matrix (NDArray): numpy array defining the unitary matrix.
            qubits (int | tuple): qubit label(s).
        """
        self._matrix = matrix
        self._qubits = qubits if type(qubits) is tuple else tuple(qubits)

    def __call__(self) -> Matrix:
        """Returns the sympy expression for the numpy array.

        Returns:
            Matrix: sympy matrix.
        """
        return Matrix(self._matrix.round(5))
    
    def __repr__(self) -> str:
        """Returns a string representation of the numpy array.

        Returns:
            str: string representation of the numpy array.
        """
        return np.array_repr(self._matrix)
    
    def __str__(self) -> str:
        """Returns a string representation of the numpy array.

        Returns:
            str: string representation of the numpy array.
        """
        return np.array_repr(self._matrix)
    
    @property
    def alias(self) -> str:
        """Returns the alias(es) of the gate.

        Returns:
            str: alias of the gate.
        """
        return None
    
    @property
    def dim(self) -> int:
        """Returns the Hilbert space dimension of the unitary operator.

        Returns:
            int: Hilbert space dimension.
        """
        return self._matrix.shape[0]
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return None

    @property
    def matrix(self) -> NDArray:
        """Returns the numpy array of the matrix.

        Returns:
            NDArray: numpy array of the matrix.
        """
        return self._matrix
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return None
    
    @property
    def qubits(self) -> tuple:
        """Returns the qubit(s) that the gate acts on.

        Returns:
            tupe: qubit label(s).
        """
        return self._qubits