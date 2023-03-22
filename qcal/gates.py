"""Submodule for storing relevant gate definitions.
"""
import numpy as np
# import sympy

from numpy.typing import NDArray
from sympy import Matrix

Id = sigma0 = np.array([[1., 0.],
                        [0., 1.]])

X = X180 = sigma1 = sigma_x = np.array([[0., 1.],
                                        [1., 0.]])

Y = Y180 = sigma2 = sigma_y = np.array([[0., -1.j],
                                        [1.j, 0.]])

Z = Z180 = sigma3 = sigma_z = np.array([[1., 0.],
                                        [0., -1.]])


def rx(theta: float):
    """Unitary rotation about the x-axis.

    Args:
        theta (float): angle of rotation

    Returns:
        NDARRAY: matrix expression of a unitary rotation about the x-axis by an
            angle theta.
    """
    return np.array([[np.cos(theta/2.), -1.j*np.sin(theta/2.)],
                     [-1.j*np.sin(theta/2.), np.cos(theta/2.)]])


def ry(theta: float):
    """Unitary rotation about the y-axis.

    Args:
        theta (float): angle of rotation

    Returns:
        NDARRAY: matrix expression of a unitary rotation about the y-axis by an
            angle theta.
    """
    return np.array([[np.cos(theta/2.), -np.sin(theta/2.)],
                     [np.sin(theta/2.), np.cos(theta/2.)]])


def rz(theta: float):
    """Unitary rotation about the z-axis.

    Args:
        theta (float): angle of rotation

    Returns:
        NDARRAY: matrix expression of a unitary rotation about the z-axis by an
            angle theta.
    """
    return np.array([[np.exp(-1.j*theta/2), 0.],
                     [0., np.exp(1.j*theta/2)]])


class Rx:
    """Class for parametrized x rotations.

    Basic example usage:

        rx = Rx(theta)
        rx()       # Symbolic python view of the matrix
        rx.matrix  # Numpy array of the matrix
    """

    def __init__(self, theta: float) -> None:
        """Initialize using the rx function.

        Args:
            theta (float): angle of rotation
        """
        self._matrix = rx(theta)

    def __call__(self) -> Matrix:
        """Returns the sympy expression for the numpy array.

        Returns:
            sympy.matrices.dense.MutableDenseMatrix: sympy matrix
        """
        return Matrix(self._matrix)
    
    def __repr__(self) -> str:
        """Returns a string representation of the numpy array.

        Returns:
            str: string representation of the numpy array
        """
        return np.array_repr(self._matrix)
    
    def __str__(self) -> str:
        """Returns a string representation of the numpy array.

        Returns:
            str: string representation of the numpy array
        """
        return np.array_repr(self._matrix)

    @property
    def matrix(self) -> NDArray:
        """Returns the numpy array of

        Returns:
            NDArray: numpy array of the matrix
        """
        return self._matrix
    

class Ry:
    """Class for parametrized y rotations.

    Basic example usage:

        ry = Ry(theta)
        ry()       # Symbolic python view of the matrix
        ry.matrix  # Numpy array of the matrix
    """

    def __init__(self, theta: float) -> None:
        """Initialize using the ry function.

        Args:
            theta (float): angle of rotation
        """
        self._matrix = rz(theta)

    def __call__(self) -> Matrix:
        """Returns the sympy expression for the numpy array.

        Returns:
            sympy.matrices.dense.MutableDenseMatrix: sympy matrix
        """
        return Matrix(self._matrix)
    
    def __repr__(self) -> str:
        """Returns a string representation of the numpy array.

        Returns:
            str: string representation of the numpy array
        """
        return np.array_repr(self._matrix)
    
    def __str__(self) -> str:
        """Returns a string representation of the numpy array.

        Returns:
            str: string representation of the numpy array
        """
        return np.array_repr(self._matrix)

    @property
    def matrix(self) -> NDArray:
        """Returns the numpy array of

        Returns:
            NDArray: numpy array of the matrix
        """
        return self._matrix
    

class Rz:
    """Class for parametrized y rotations.

    Basic example usage:

        rz = Rz(theta)
        rz()       # Symbolic python view of the matrix
        rz.matrix  # Numpy array of the matrix
    """

    def __init__(self, theta: float) -> None:
        """Initialize using the rz function.

        Args:
            theta (float): angle of rotation
        """
        self._matrix = rz(theta)

    def __call__(self) -> Matrix:
        """Returns the sympy expression for the numpy array.

        Returns:
            sympy.matrices.dense.MutableDenseMatrix: sympy matrix
        """
        return Matrix(self._matrix)
    
    def __repr__(self) -> str:
        """Returns a string representation of the numpy array.

        Returns:
            str: string representation of the numpy array
        """
        return np.array_repr(self._matrix)
    
    def __str__(self) -> str:
        """Returns a string representation of the numpy array.

        Returns:
            str: string representation of the numpy array
        """
        return np.array_repr(self._matrix)

    @property
    def matrix(self) -> NDArray:
        """Returns the numpy array of

        Returns:
            NDArray: numpy array of the matrix
        """
        return self._matrix

        
