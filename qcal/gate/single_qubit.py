"""Submodule for storing single-qubit gate definitions.

See https://threeplusone.com/pubs/on_gates.pdf for relevant definitions.
"""
from qcal.gate.gate import Gate
from qcal.units import ns

import numpy as np

from numpy.typing import NDArray
from typing import List, Tuple, Union

# TODO: add Clifford gates


__all__ = ['C', 'H', 'Id', 'Rn', 'Rx', 'Ry', 'Rz', 'S', 'Sdag', 'T', 'Tdag',
           'U3', 'V', 'Vdag', 'X', 'X90', 'Y', 'Y90', 'Z']


id = sigma0 = np.array([[1., 0.],
                        [0., 1.]])

h = hadamard = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                         [1/np.sqrt(2), -1/np.sqrt(2)]])

meas = np.array([[1., 0.],
                 [0., 0.]])

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

paulis = [id, x, y, z]


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
        phi (float):   first phase angle.
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

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the rn function.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(rn(2*np.pi/3, (1, 1, 1)/np.sqrt(3)), qubit)
        self._properties['name'] = 'C'
        self._properties['gate'] = {
            'angle': 2*np.pi/3,
            'axis': str((1, 1, 1)/np.sqrt(3))
        }


class H(Gate):
    """Class for the Hadamard (H) gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the h gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(h, qubit)
        self._properties['alias'] = 'QFT'
        self._properties['name'] = 'H'


class Id(Gate):
    """Class for the identity gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the id gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(id, qubit)
        self._properties['name'] = 'I'
        self._properties['gate'] = {
            'angle': 0,
        }


class Idle(Gate):
    """Class for the idle gate."""

    def __init__(self, qubit: int = 0, duration: float = 30*ns) -> None:
        """Initialize using the id gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
            duration (float): idle duration. Defaults to 30 ns.
        """
        super().__init__(id, qubit)
        self._properties['alias'] = 'I'
        self._properties['name'] = 'Idle'
        self._properties['gate'] = {
            'angle': 0,
            'duration': duration
        }


class Meas(Gate):
    """Class for a single-qubit measurement operation."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the meas matrix.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(meas, qubit)
        self._properties['name'] = 'Meas'
    

class Rn(Gate):
    """Class for parametrized n rotations.

    Basic example usage:

        n = (1, 0, 0)  # x-axis
        rn = Rn(theta, n)
        rn()       # Symbolic python view of the matrix
        rn.matrix  # Numpy array of the matrix
    """

    def __init__(self,
            theta: float, 
            n: Union[List, Tuple, NDArray],
            qubit: int = 0
        ) -> None:
        """Initialize using the rn function.

        Args:
            theta (float): angle of rotation.
            n (List, Tuple, NDArray): unit vector defining the rotation axis.
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(rn(theta, n), qubit)
        self._properties['name'] = 'Rn'
        self._properties['gate'] = {
            'angle': theta,
            'axis':  n,
        }
    

class Rx(Gate):
    """Class for parametrized x rotations.

    Basic example usage:

        rx = Rx(theta)
        rx()       # Symbolic python view of the matrix
        rx.matrix  # Numpy array of the matrix
    """

    def __init__(self, theta: float, qubit: int = 0) -> None:
        """Initialize using the rx function.

        Args:
            theta (float): angle of rotation.
            qubit (int):   qubit label. Defaults to 0.
        """
        super().__init__(rx(theta), qubit)
        self._properties['name'] = 'Rx'
        self._properties['gate'] = {
            'angle': theta,
            'axis':  'x'
        }
    

class Ry(Gate):
    """Class for parametrized y rotations.

    Basic example usage:

        ry = Ry(theta)
        ry()       # Symbolic python view of the matrix
        ry.matrix  # Numpy array of the matrix
    """

    def __init__(self, theta: float, qubit: int = 0) -> None:
        """Initialize using the ry function.

        Args:
            theta (float): angle of rotation.
            qubit (int):   qubit label. Defaults to 0.
        """
        super().__init__(ry(theta), qubit)
        self._properties['name'] = 'Ry'
        self._properties['gate'] = {
            'angle': theta,
            'axis':  'y',
        }
    

class Rz(Gate):
    """Class for parametrized z rotations.

    Basic example usage:

        rz = Rz(theta)
        rz()       # Symbolic python view of the matrix
        rz.matrix  # Numpy array of the matrix
    """

    def __init__(self, theta: float, qubit: int = 0) -> None:
        """Initialize using the rz function.

        Args:
            theta (float): angle of rotation.
            qubit (int):   qubit label. Defaults to 0.
        """
        super().__init__(rz(theta), qubit)
        self._properties['name'] = 'Rz'
        self._properties['gate'] = {
            'angle': theta,
            'axis':  'z',
        }
    

class S(Gate):
    """Class for the phase S = sqrt(Z) gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the Rz gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(rz(np.pi/2), qubit)
        self._properties['alias'] = 'sqrt(Z)\nZ90'
        self._properties['name'] = 'S'
        self._properties['gate'] = {
            'angle': np.pi/2,
            'axis':  'z',
        }
    

class Sdag(Gate):
    """Class for the phase S^dagger gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the Rz gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(rz(-np.pi/2), qubit)
        self._properties['alias'] = 'SqrtZdag\nZ-90'
        self._properties['name'] = 'Sdag'
        self._properties['gate'] = {
            'angle': -np.pi/2,
            'axis':  'z',
        }
    

class T(Gate):
    """Class for the fourth-root of Z (T) gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the t gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(t, qubit)
        self._properties['alias'] = 'Z^(1/4)'
        self._properties['name'] = 'T'
        self._properties['gate'] = {
            'angle': np.pi/4,
            'axis':  'z',
        }
    

class Tdag(Gate):
    """Class for the inverse fourth-root of Z gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the tdag gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(tdag, qubit)
        self._properties['alias'] = 'Z^(1/4)dag'
        self._properties['name'] = 'Tdag'
        self._properties['gate'] = {
            'angle': -np.pi/4,
            'axis':  'z',
        }
    

class U3(Gate):
    """Class for the U3 gate."""

    def __init__(self, 
            theta: float,
            phi:   float,
            gamma: float,
            qubit: int = 0
        ) -> None:
        """Initialize using the u3 function.
        
        Args:
            theta (float): angle of rotation.
            phi (float):   first phase angle.
            gamma (float): second phase angle.
            qubit (int):   qubit label.
        """
        super().__init__(u3(theta, phi, gamma), qubit)
        self._properties['name'] = 'U3'
        self._properties['gate'] = {
            'theta': theta,
            'phi':   phi,
            'gamma': gamma
        }
    

class V(Gate):
    """Class for the V = X90 gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the rx gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(rx(np.pi/2), qubit)
        self._properties['alias'] = 'X90'
        self._properties['name'] = 'V'
        self._properties['gate'] = {
            'angle': np.pi/2,
            'axis':  'x',
        }
    

class Vdag(Gate):
    """Class for the Vdag = X-90 gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the rx gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(rx(-np.pi/2), qubit)
        self._properties['alias'] = 'X-90'
        self._properties['name'] = 'Vdag'
        self._properties['gate'] = {
            'angle': -np.pi/2,
            'axis':  'x',
        }
    

class X(Gate):
    """Class for the Pauli X gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the x gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(x, qubit)
        self._properties['name'] = 'X'
        self._properties['gate'] = {
            'angle': np.pi,
            'axis':  'x',
        }
    

class X90(Gate):
    """Class for the X90 = sqrt(X) gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the rx gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(rx(np.pi/2), qubit)
        self._properties['alias'] = 'V'
        self._properties['name'] = 'X90'
        self._properties['gate'] = {
            'angle': np.pi/2,
            'axis':  'x',
        }


class Y(Gate):
    """Class for the Pauli Y gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the Y gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(y, qubit)
        self._properties['name'] = 'Y'
        self._properties['gate'] = {
            'angle': np.pi,
            'axis':  'y',
        }
    

class Y90(Gate):
    """Class for the Y90 = sqrt(Y) gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the ry gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(ry(np.pi/2), qubit)
        self._properties['name'] = 'Y90'
        self._properties['gate'] = {
            'angle': np.pi/2,
            'axis':  'y',
        }
    

class Z(Gate):
    """Class for the Pauli Z gate."""

    def __init__(self, qubit: int = 0) -> None:
        """Initialize using the z gate.
        
        Args:
            qubit (int): qubit label. Defaults to 0.
        """
        super().__init__(z, qubit)
        self._properties['name'] = 'Z'
        self._properties['gate'] = {
            'angle': np.pi,
            'axis':  'z',
        }