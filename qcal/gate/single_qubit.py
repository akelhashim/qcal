"""Submodule for storing single-qubit gate definitions.

See https://threeplusone.com/pubs/on_gates.pdf for relevant definitions.
"""
from qcal.gate.gate import Gate
from qcal.units import ns

import numpy as np

from collections import defaultdict
from numpy.typing import NDArray
from random import gauss, randint
from typing import List, Dict, Tuple, Union

# TODO: add Clifford gates


__all__ = (
    'C',
    'H',
    'Id',
    'Idle',
    'Meas',
    'MCM',
    'RandSU2',
    'Reset',
    'Rn',
    'Rx',
    'Ry',
    'Rz',
    'S',
    'SX',
    'SY',
    'Sdag',
    'SXdag',
    'SYdag',
    'T',
    'Tdag',
    'U3',
    'V',
    'Vdag',
    'VirtualZ',
    'X',
    'X90',
    'Y',
    'Y90',
    'Z',
    'Z90'
)


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

paulis = (id, x, y, z)


def basis_rotation(meas):
    """Returns the gate which rotates the qubit to the measurement basis.

    Args:
        meas (Meas): measurement object.

    Returns:
        Gate: gate object.
    """
    basis_map = {
        'X': Ry(meas.qubits[0], -np.pi/2),
        'Y': Rx(meas.qubits[0], np.pi/2),
        'Z': Id(meas.qubits[0])
    }
    return basis_map[meas.properties['params']['basis'].upper()]


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
    assert round(length, 5) == 1., "n must be a unit vector!"

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

    def __init__(self, qubit: int) -> None:
        """Initialize using the rn function.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(rn(2*np.pi/3, (1, 1, 1)/np.sqrt(3)), qubit)
        self._properties['name'] = 'C'
        self._properties['params'] = {
            'angle': 2*np.pi/3,
            'axis': str((1, 1, 1)/np.sqrt(3))
        }


class H(Gate):
    """Class for the Hadamard (H) gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the h gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(h, qubit)
        self._properties['alias'] = 'QFT'
        self._properties['name'] = 'H'


class Id(Gate):
    """Class for the identity gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the id gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(id, qubit)
        self._properties['name'] = 'I'
        self._properties['params'] = {
            'angle': 0,
            'duration': 0.
        }


class Idle(Gate):
    """Class for the idle gate."""

    def __init__(self, qubit: int, duration: float = 0*ns) -> None:
        """Initialize using the id gate.
        
        Args:
            qubit (int): qubit label.
            duration (float): idle duration. Defaults to 0 ns.
        """
        super().__init__(id, qubit)
        self._properties['alias'] = 'I'
        self._properties['name'] = 'Idle'
        self._properties['params'] = {
            'angle': 0,
            'duration': duration
        }


class Meas(Gate):
    """Class for a single-qubit measurement operation."""

    def __init__(self, qubit: int, basis: str = 'Z') -> None:
        """Initialize using the meas matrix.
        
        Args:
            qubit (int): qubit label.
            basis (str): measurement basis. Defaults to Z.
        """
        super().__init__(meas, qubit)
        self._properties['name'] = 'Meas'
        self._properties['params']['basis'] = basis

    @property
    def is_measurement(self) -> bool:
        """Whether or not gate is a measurement operation.

        Returns:
            bool: measurement or not.
        """
        return True
    

class MCM(Gate):
    """Class for a single-qubit mid-circuit measurement operation."""

    def __init__(
            self, 
            qubits:      int | Tuple[int], 
            basis:       str = 'Z',
            apply:       Dict = {}, 
            dd_qubits:   List | Tuple = [],
            dd_method:   str = 'XY',
            n_dd_pulses: int = 8,
        ) -> None:
        """Initialize using the meas matrix.
        
        Args:
            qubits (int | Tuple[int]): qubit label(s).
            basis (str): measurement basis. Defaults to Z.
            apply (Dict): conditional gates to apply to another qubit depending
                on the outcomes of the mid-circuit measurement. Defaults to {}.
                Example: apply = {'0': Cycle({Id(1)}), '1': Cycle({X(1)})}.
            dd_qubits (List | Tuple, optional): which qubits to dynamical 
                decouple during a mid-circuit measurement. Defaults to [].
            dd_method (str): dynamical decoupling protocol. Defaults to 'XY'.
            n_dd_pulses (int): number of pulses for dd protocol. Defaults to 8.
        """
        super().__init__(meas, qubits)
        self._properties['name'] = 'MCM'
        self._properties['params']['basis'] = basis
        self._properties['params']['apply'] = apply
        self._properties['params']['dd_qubits'] = dd_qubits
        self._properties['params']['dd_method'] = dd_method
        self._properties['params']['n_dd_pulses'] = n_dd_pulses

    @property
    def is_measurement(self) -> bool:
        """Whether or not gate is a measurement operation.

        Returns:
            bool: measurement or not.
        """
        return True
    
    @property
    def is_single_qubit(self) -> bool:
        """Whether or not the gate acts on a single qubit.

        Returns:
            bool: single-qubit gate or not.
        """
        return True


class Reset(Gate):
    """Class for qubit reset in the middle of a circuit."""

    def __init__(
            self, 
            qubit:         int,
            measure_first: bool = True,
            dd_qubits:     List | Tuple = [],
            dd_method:     str = 'XY',
            n_dd_pulses:   int = 8,
        ) -> None:
        """Initialize using the meas matrix.

        Args:
            qubit (int): qubit label.
            measure_first (bool, optional): whether to measure before the
                reset. Defauls to True. This is unnecessary if a previous
                measurement was already made for a mid-circuit operation.
            dd_qubits (List | Tuple, optional): which qubits to dynamical 
                decouple during a mid-circuit measurement. Defaults to [].
            dd_method (str): dynamical decoupling protocol. Defaults to 'XY'.
            n_dd_pulses (int): number of pulses for dd protocol. Defaults to 8.
        """
        super().__init__(meas, qubit)
        self._properties['name'] = 'Reset'
        self._properties['params']['measure_first'] = measure_first
        self._properties['params']['meas'] = MCM(
            qubit, 
            dd_qubits=dd_qubits, dd_method=dd_method, n_dd_pulses=n_dd_pulses
        )

class RandSU2(Gate):
    """Class for a random SU(2) gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the rn function.

        Args:
            qubit (int): qubit label.
        """
        vec = [gauss(0, 1) for i in range(3)]
        mag = sum(x**2 for x in vec) ** .5
        n = [x/mag for x in vec]
        theta = randint(-180, 180)
        super().__init__(rn(theta, n), qubit)
        self._properties['name'] = 'RandSU2'
        self._properties['params'] = {
            'angle': theta,
            'axis':  n,
        }

class Rn(Gate):
    """Class for parametrized n rotations.

    Basic example usage:

        n = (1, 0, 0)  # x-axis
        rn = Rn(theta, n)
        rn()       # Symbolic python view of the matrix
        rn.matrix  # Numpy array of the matrix
    """

    def __init__(self,
            qubit: int,
            theta: float, 
            n: Union[List, Tuple, NDArray]
        ) -> None:
        """Initialize using the rn function.

        Args:
            qubit (int):              qubit label.
            theta (float):            angle of rotation.
            n (List, Tuple, NDArray): unit vector defining the rotation axis.
        """
        super().__init__(rn(theta, n), qubit)
        self._properties['name'] = 'Rn'
        self._properties['params'] = {
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

    def __init__(self, qubit: int, theta: float) -> None:
        """Initialize using the rx function.

        Args:
            qubit (int):   qubit label.
            theta (float): angle of rotation.
        """
        super().__init__(rx(theta), qubit)
        self._properties['name'] = 'Rx'
        self._properties['params'] = {
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

    def __init__(self, qubit: int, theta: float) -> None:
        """Initialize using the ry function.

        Args:
            qubit (int):   qubit label.
            theta (float): angle of rotation.
        """
        super().__init__(ry(theta), qubit)
        self._properties['name'] = 'Ry'
        self._properties['params'] = {
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

    def __init__(self,
            qubit: int, theta: float, subspace: str = 'GE'
        ) -> None:
        """Initialize using the rz function.

        Args:
            qubit (int):    qubit label.
            theta (float):  angle of rotation.
            subspace (str): subspace within which the gate acts. Defaults to
                GE.
        """
        super().__init__(rz(theta), qubit)
        self._properties['name'] = 'Rz'
        self._properties['params'] = {
            'phase': theta,
            'axis':  'z',
        }
        self._properties['subspace'] = subspace
    

class S(Gate):
    """Class for the phase S = sqrt(Z) gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the Rz gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(rz(np.pi/2), qubit)
        self._properties['alias'] = 'sqrt(Z)\nZ90'
        self._properties['name'] = 'S'
        self._properties['params'] = {
            'phase': np.pi/2,
            'axis':  'z',
        }


class SX(Gate):
    """Class for the SX = X90 gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the rx gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(rx(np.pi/2), qubit)
        self._properties['alias'] = 'V, X90'
        self._properties['name'] = 'SX'
        self._properties['params'] = {
            'angle': np.pi/2,
            'axis':  'x',
        }


class SY(Gate):
    """Class for the SY = Y90 gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the ry gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(ry(np.pi/2), qubit)
        self._properties['alias'] = 'Y90'
        self._properties['name'] = 'SY'
        self._properties['params'] = {
            'angle': np.pi/2,
            'axis':  'y',
        }
    

class Sdag(Gate):
    """Class for the phase S^dagger gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the Rz gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(rz(-np.pi/2), qubit)
        self._properties['alias'] = 'SqrtZdag\nZ-90'
        self._properties['name'] = 'Sdag'
        self._properties['params'] = {
            'phase': -np.pi/2,
            'axis':  'z',
        }


class SXdag(Gate):
    """Class for the SXdag = X-90 gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the rx gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(rx(-np.pi/2), qubit)
        self._properties['alias'] = 'Vdag, X-90'
        self._properties['name'] = 'SXdag'
        self._properties['params'] = {
            'angle': -np.pi/2,
            'axis':  'x',
        }


class SYdag(Gate):
    """Class for the SYdag = Y-90 gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the ry gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(ry(-np.pi/2), qubit)
        self._properties['alias'] = 'Y-90'
        self._properties['name'] = 'SYdag'
        self._properties['params'] = {
            'angle': -np.pi/2,
            'axis':  'y',
        }
    

class T(Gate):
    """Class for the fourth-root of Z (T) gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the t gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(t, qubit)
        self._properties['alias'] = 'Z^(1/4)'
        self._properties['name'] = 'T'
        self._properties['params'] = {
            'phase': np.pi/4,
            'axis':  'z',
        }
    

class Tdag(Gate):
    """Class for the inverse fourth-root of Z gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the tdag gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(tdag, qubit)
        self._properties['alias'] = 'Z^(1/4)dag'
        self._properties['name'] = 'Tdag'
        self._properties['params'] = {
            'phase': -np.pi/4,
            'axis':  'z',
        }
    

class U3(Gate):
    """Class for the U3 gate."""

    def __init__(self,
            qubit: int, theta: float, phi: float, gamma: float,
        ) -> None:
        """Initialize using the u3 function.
        
        Args:
            qubit (int):   qubit label.
            theta (float): angle of rotation.
            phi (float):   first phase angle.
            gamma (float): second phase angle.
        """
        super().__init__(u3(theta, phi, gamma), qubit)
        self._properties['name'] = 'U3'
        self._properties['params'] = {
            'theta': theta,
            'phi':   phi,
            'gamma': gamma
        }
    

class V(Gate):
    """Class for the V = X90 gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the rx gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(rx(np.pi/2), qubit)
        self._properties['alias'] = 'SX, X90'
        self._properties['name'] = 'V'
        self._properties['params'] = {
            'angle': np.pi/2,
            'axis':  'x',
        }
    

class Vdag(Gate):
    """Class for the Vdag = X-90 gate."""

    def __init__(self, qubit: int) -> None:
        """Initialize using the rx gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(rx(-np.pi/2), qubit)
        self._properties['alias'] = 'SXdag, X-90'
        self._properties['name'] = 'Vdag'
        self._properties['params'] = {
            'angle': -np.pi/2,
            'axis':  'x',
        }


class VirtualZ(Gate):
    """Class for virtual Z gate."""

    def __init__(self,
            qubit: int, theta: float, subspace: str = 'GE'
        ) -> None:
        """Initialize using the rz function.

        Args:
            qubit (int):    qubit label.
            theta (float):  angle of rotation.
            subspace (str): subspace within which the gate acts. Defaults to
                GE.
        """
        super().__init__(rz(theta), qubit)
        self._properties['name'] = 'VirtualZ'
        self._properties['params'] = {
            'phase': theta,
            'axis':  'z',
        }
        self._properties['subspace'] = subspace
    

class X(Gate):
    """Class for the Pauli X gate."""

    def __init__(self, qubit: int, subspace: str = 'GE') -> None:
        """Initialize using the x gate.
        
        Args:
            qubit (int):    qubit label.
            subspace (str): subspace within which the gate acts. Defaults to
                GE.
        """
        super().__init__(x, qubit)
        self._properties['name'] = 'X'
        self._properties['params'] = {
            'angle': np.pi,
            'axis':  'x',
        }
        self._properties['subspace'] = subspace
    

class X90(Gate):
    """Class for the X90 = sqrt(X) gate."""

    def __init__(self, qubit: int, subspace: str = 'GE') -> None:
        """Initialize using the rx gate.
        
        Args:
            qubit (int):    qubit label.
            subspace (str): subspace within which the gate acts. Defaults to
                GE.
        """
        super().__init__(rx(np.pi/2), qubit)
        self._properties['alias'] = 'V'
        self._properties['name'] = 'X90'
        self._properties['params'] = {
            'angle':    np.pi/2,
            'axis':     'x'
        }
        self._properties['subspace'] = subspace


class Y(Gate):
    """Class for the Pauli Y gate."""

    def __init__(self, qubit: int, subspace: str = 'GE') -> None:
        """Initialize using the Y gate.
        
        Args:
            qubit (int):    qubit label.
            subspace (str): subspace within which the gate acts. Defaults to
                GE.
        """
        super().__init__(y, qubit)
        self._properties['name'] = 'Y'
        self._properties['params'] = {
            'angle': np.pi,
            'axis':  'y',
        }
        self._properties['subspace'] = subspace
    

class Y90(Gate):
    """Class for the Y90 = sqrt(Y) gate."""

    def __init__(self, qubit: int, subspace: str = 'GE') -> None:
        """Initialize using the ry gate.
        
        Args:
            qubit (int): qubit label.
            subspace (str): subspace within which the gate acts. Defaults to
                GE.
        """
        super().__init__(ry(np.pi/2), qubit)
        self._properties['name'] = 'Y90'
        self._properties['params'] = {
            'angle': np.pi/2,
            'axis':  'y',
        }
        self._properties['subspace'] = subspace
    

class Z(Gate):
    """Class for the Pauli Z gate."""

    def __init__(self, qubit: int, subspace: str = 'GE') -> None:
        """Initialize using the z gate.
        
        Args:
            qubit (int): qubit label.
            subspace (str): subspace within which the gate acts. Defaults to
                GE.
        """
        super().__init__(z, qubit)
        self._properties['name'] = 'Z'
        self._properties['params'] = {
            'phase': np.pi,
            'axis':  'z',
        }
        self._properties['subspace'] = subspace


class Z90(Gate):
    """Class for the Z90 = sqrt(Z) gate."""

    def __init__(self, qubit: int, subspace: str = 'GE') -> None:
        """Initialize using the Rz gate.
        
        Args:
            qubit (int): qubit label.
        """
        super().__init__(rz(np.pi/2), qubit)
        self._properties['alias'] = 'sqrt(Z)\nS'
        self._properties['name'] = 'Z90'
        self._properties['params'] = {
            'phase': np.pi/2,
            'axis':  'z',
        }
        self._properties['subspace'] = subspace


single_qubit_gates = defaultdict(lambda: 'Gate not currently supported!', {
    'C':        C,
    'H':        H,
    'Id':       Id,
    'Idle':     Idle,
    'Meas':     Meas,
    'MCM':      MCM,
    'RandSU2':  RandSU2,
    'Reset':    Reset,
    'Rn':       Rn,
    'Rx':       Rx, 
    'Ry':       Ry,
    'Rz':       Rz,
    'S':        S,
    'SX':       SX,
    'SY':       SY,
    'Sdag':     Sdag,
    'SXdag':    SXdag,
    'SYdag':    SYdag,
    'T':        T,
    'Tdag':     Tdag,
    'U3':       U3,
    'V':        V,
    'Vdag':     Vdag,
    'VirtualZ': VirtualZ,
    'X':        X,
    'X90':      X90,
    'Y':        Y, 
    'Y90':      Y90,
    'Z':        Z,
    'Z90':      Z90
})