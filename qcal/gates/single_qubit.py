"""Submodule for storing single-qubit gate definitions.

See https://threeplusone.com/pubs/on_gates.pdf for relevant definitions.
"""
from qcal.gates.gate import Gate

import numpy as np

from numpy.typing import NDArray
from typing import List, Tuple, Union

# TODO: add Clifford gates


__all__ = ['C', 'H', 'Id', 'Rn', 'Rx', 'Ry', 'Rz', 'S', 'SDag', 'T', 'TDag',
           'U3', 'V', 'VDag', 'X', 'X90', 'Y', 'Y90', 'Z']


id = sigma0 = np.array([[1., 0.],
                        [0., 1.]])

h = hadamard = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                         [1/np.sqrt(2), -1/np.sqrt(2)]])

t = np.array([[1., 0.],
              [0., np.exp(1j*np.pi/4)]])

tdag = np.array([[1., 0.],
                 [0., np.exp(-1j*np.pi/4)]])

x = x180 = sigma1 = sigma_x = np.array([[0., 1.],
                                        [1., 0.]])

y = y180 = sigma2 = sigma_y = np.array([[0., -1.j],
                                        [1.j, 0.]])

z = z180 = sigma3 = sigma_z = np.array([[1., 0.],
                                        [0., -1.]])


def rn(theta: float, n: Union[List, Tuple, NDArray]) -> NDArray:
    """Unitary rotation about the n-axis by an angle theta.

    Args:
        theta (float): angle of rotation.
        n (list, tuple, NDArray): unit vector defining the rotation axis.

    Returns:
        NDArray: matrix expression of a unitary rotation about the n-axis by an
            angle theta.
    """
    nx, ny, nz = n
    length = np.sqrt(nx**2 + ny**2 + nz**2)
    assert length == 1., "n must be a unit vector!"

    return np.array([
        [np.cos(theta/2.) - 1.j*nz*np.sin(theta/2.),
         -ny*np.sin(theta/2.) - 1.j*nx*np.sin(theta/2.)],
        [ny*np.sin(theta/2.) - 1.j*nx*np.sin(theta/2.),
         np.cos(theta/2.) + 1.j*nz*np.sin(theta/2.)]
    ])


def rx(theta: float) -> NDArray:
    """Unitary rotation about the x-axis.

    Args:
        theta (float): angle of rotation.

    Returns:
        NDArray: matrix expression of a unitary rotation about the x-axis by an
            angle theta.
    """
    return np.array([[np.cos(theta/2.), -1.j*np.sin(theta/2.)],
                     [-1.j*np.sin(theta/2.), np.cos(theta/2.)]])


def ry(theta: float) -> NDArray:
    """Unitary rotation about the y-axis.

    Args:
        theta (float): angle of rotation.

    Returns:
        NDArray: matrix expression of a unitary rotation about the y-axis by an
            angle theta.
    """
    return np.array([[np.cos(theta/2.), -np.sin(theta/2.)],
                     [np.sin(theta/2.), np.cos(theta/2.)]])


def rz(theta: float) -> NDArray:
    """Unitary rotation about the z-axis.

    Args:
        theta (float): angle of rotation.

    Returns:
        NDArray: matrix expression of a unitary rotation about the z-axis by an
            angle theta.
    """
    return np.array([[np.exp(-1.j*theta/2), 0.],
                     [0., np.exp(1.j*theta/2)]])


def u3(theta: float, phi: float, gamma: float) -> NDArray:
    """Generic single-qubit unitary rotation parametrized by 3 Euler angles.

    https://qiskit.org/documentation/stubs/qiskit.circuit.library.U3Gate.html

    Args:
        theta (float): angle of rotation.
        phi (float): first phase angle.
        gamma (float): second phase angle.

    Returns:
        NDArray: matrix representation of a U3 gate.
    """
    return np.array([
        [np.cos(theta/2.), 
         -np.exp(1.j*gamma/2)*np.sin(theta/2.)],
        [np.exp(1.j*phi/2)*np.sin(theta/2.), 
         np.exp(1.j*(phi+gamma)/2)*np.cos(theta/2.)]
    ])


class C(Gate):
    """Class for the axis cycling (C) gate."""

    def __init__(self) -> None:
        """Initialize using the rn function."""
        super().__init__(rn(2*np.pi/3, (1, 1, 1)/np.sqrt(3)))

    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return 2*np.pi/3
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return str((1, 1, 1)/np.sqrt(3))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'C'


class H(Gate):
    """Class for the Hadamard (H) gate."""

    def __init__(self) -> None:
        """Initialize using the h gate."""
        super().__init__(h)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Hadamard'


class Id(Gate):
    """Class for the identity gate."""

    def __init__(self) -> None:
        """Initialize using the id gate."""
        super().__init__(id)

    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Identity'
    

class Rn(Gate):
    """Class for parametrized n rotations.

    Basic example usage:

        n = (1, 0, 0)  # x-axis
        rn = Rn(theta, n)
        rn()       # Symbolic python view of the matrix
        rn.matrix  # Numpy array of the matrix
    """

    def __init__(self, theta: float, n: Union[List, Tuple, NDArray]) -> None:
        """Initialize using the rn function.

        Args:
            theta (float): angle of rotation.
            n (list, tuple, NDArray): unit vector defining the rotation axis.
        """
        super().__init__(rn(theta, n))
        self._angle = theta
        if n == (1, 0, 0):
            self._axis = 'x'
        elif n == (0, 1, 0):
            self._axis = 'y'
        elif n == (0, 0, 1):
            self._axis = 'z'
        else:
            self._axis = str(n)

    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return self._axis
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Rx'
    

class Rx(Gate):
    """Class for parametrized x rotations.

    Basic example usage:

        rx = Rx(theta)
        rx()       # Symbolic python view of the matrix
        rx.matrix  # Numpy array of the matrix
    """

    def __init__(self, theta: float) -> None:
        """Initialize using the rx function.

        Args:
            theta (float): angle of rotation.
        """
        super().__init__(rx(theta))
        self._angle = theta

    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'x'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Rx'
    

class Ry(Gate):
    """Class for parametrized y rotations.

    Basic example usage:

        ry = Ry(theta)
        ry()       # Symbolic python view of the matrix
        ry.matrix  # Numpy array of the matrix
    """

    def __init__(self, theta: float) -> None:
        """Initialize using the ry function.

        Args:
            theta (float): angle of rotation.
        """
        super().__init__(ry(theta))
        self._angle = theta

    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'y'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Ry'
    

class Rz(Gate):
    """Class for parametrized z rotations.

    Basic example usage:

        rz = Rz(theta)
        rz()       # Symbolic python view of the matrix
        rz.matrix  # Numpy array of the matrix
    """

    def __init__(self, theta: float) -> None:
        """Initialize using the rz function.

        Args:
            theta (float): angle of rotation.
        """
        super().__init__(rz(theta))
        self._angle = theta

    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'z'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Rz'
    

class S(Gate):
    """Class for the phase S = sqrt(Z) gate."""

    def __init__(self) -> None:
        """Initialize using the Rz gate."""
        super().__init__(rz(np.pi/2))
        self._angle = np.pi/2

    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'z'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'S'
    

class SDag(Gate):
    """Class for the phase S^dagger gate."""

    def __init__(self) -> None:
        """Initialize using the Rz gate."""
        super().__init__(rz(-np.pi/2))
        self._angle = -np.pi/2

    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'z'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'SDag'
    

class T(Gate):
    """Class for the fourth-root of Z (T) gate."""

    def __init__(self) -> None:
        """Initialize using the t gate."""
        super().__init__(t)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'T'
    

class TDag(Gate):
    """Class for the inverse fourth-root of Z gate."""

    def __init__(self) -> None:
        """Initialize using the tdag gate."""
        super().__init__(tdag)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'TDag'
    

class U3(Gate):
    """Class for the U3 gate."""

    def __init__(self, theta: float, phi: float, gamma: float) -> None:
        """Initialize using the u3 function."""
        super().__init__(u3(theta, phi, gamma))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'U3'
    

class V(Gate):
    """Class for the V = X90 gate."""

    def __init__(self) -> None:
        """Initialize using the rx gate."""
        super().__init__(rx(np.pi/2))
        self._angle = np.pi/2

    @property
    def alias(self) -> str:
        """Returns the alias(es) of the gate.

        Returns:
            str: alias of the gate.
        """
        return 'X90'
    
    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'x'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'V'
    

class VDag(Gate):
    """Class for the VDag = X-90 gate."""

    def __init__(self) -> None:
        """Initialize using the rx gate."""
        super().__init__(rx(-np.pi/2))
        self._angle = -np.pi/2

    @property
    def alias(self) -> str:
        """Returns the alias(es) of the gate.

        Returns:
            str: alias of the gate.
        """
        return 'X-90'
    
    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'x'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'VDag'
    

class X(Gate):
    """Class for the Pauli X gate."""

    def __init__(self) -> None:
        """Initialize using the x gate."""
        super().__init__(x)
        self._angle = np.pi

    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'x'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Pauli-X'
    

class X90(Gate):
    """Class for the X90 = sqrt(X) gate."""

    def __init__(self) -> None:
        """Initialize using the rx gate."""
        super().__init__(rx(np.pi/2))
        self._angle = np.pi/2

    @property
    def alias(self) -> str:
        """Returns the alias(es) of the gate.

        Returns:
            str: alias of the gate.
        """
        return 'V'
    
    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'x'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'X90'
    

class Y(Gate):
    """Class for the Pauli Y gate."""

    def __init__(self) -> None:
        """Initialize using the Y gate."""
        super().__init__(y)
        self._angle = np.pi

    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'y'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Pauli-Y'
    

class Y90(Gate):
    """Class for the Y90 = sqrt(Y) gate."""

    def __init__(self) -> None:
        """Initialize using the ry gate."""
        super().__init__(ry(np.pi/2))
        self._angle = np.pi/2

    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'y'

    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Y90'
    

class Z(Gate):
    """Class for the Pauli Z gate."""

    def __init__(self) -> None:
        """Initialize using the z gate."""
        super().__init__(z)
        self._angle = np.pi

    @property
    def angle(self) -> float:
        """Returns the angle of rotation.

        Returns:
            float: angle of rotation.
        """
        return self._angle
    
    @property
    def axis(self) -> str:
        """Returns the axis of rotation.

        Returns:
            str: axis of rotation.
        """
        return 'z'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Pauli-Z'