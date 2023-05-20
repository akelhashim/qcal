"""Submodule for defining the basic gate class.
"""
import numpy as np
# import sympy

from numpy.typing import NDArray
from sympy import Matrix


class Gate:

    __slots__ = '_matrix'

    def __init__(self, matrix) -> None:
        """Initialize a gate using its matrix definition.
        """
        self._matrix = matrix

    def __call__(self) -> Matrix:
        """Returns the sympy expression for the numpy array.

        Returns:
            sympy.matrices.dense.MutableDenseMatrix: sympy matrix.
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