"""Submodule for storing two-qubit gate definitions.

See https://threeplusone.com/pubs/on_gates.pdf for relevant definitions.
"""
from qcal.gate.gate import Gate
from qcal.gate.single_qubit import id, x, y, z

import numpy as np
import scipy

from numpy.typing import NDArray
from typing import List, Tuple, Union


__all__ = [
    'CNot', 'CX', 'CY', 'CZ', 'CH', 'M', 'MS',
    'DCNot', 'fSWAP', 'iSWAP', 
    'SWAP',
    'XX', 'YY', 'ZZ',
    'CPhase', 'CS', 'CSdag', 'CT', 'CTdag', 'CRot', 'Barenco', 'CV',
    'XY', 'Givens', 'DB',
    'SWAPAlpha', 'SqrtSWAP', 'SqrtSWAPdag',
    'pSWAP', 'QFT2',
    'BGate', 'ECP', 'WGate', 'AGate',
    'FSim', 'Syc'
]

paulis = [np.kron(id, id),
          np.kron(id, x),
          np.kron(id, y),
          np.kron(id, z),
          np.kron(x, id),
          np.kron(x, x),
          np.kron(x, y),
          np.kron(x, z),
          np.kron(y, id),
          np.kron(y, x),
          np.kron(y, y),
          np.kron(y, z),
          np.kron(z, id),
          np.kron(z, x),
          np.kron(z, y),
          np.kron(z, z)
]


b_gate = berkeley = np.sqrt(2 - np.sqrt(2)) / 2 * np.array([
    [1. + np.sqrt(2), 0., 0., 1.j],
    [0., 1., 1.j*(1 + np.sqrt(2)), 0.],
    [0., 1.j*(1 + np.sqrt(2)), 1., 0.],
    [1.j, 0., 0., 1. + np.sqrt(2)]
])

cnot = cx = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 1.],
                      [0., 0., 1., 0.]])

cy = np.array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 0., -1.j],
               [0., 0., 1.j, 0.]])

cz = np.array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., -1.]])

ch = np.array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1/np.sqrt(2), 1/np.sqrt(2)],
               [0., 0., 1/np.sqrt(2), -1/np.sqrt(2)]])

cv = sqrtcnot = np.array([[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., (1 + 1.j)/2, (1 - 1.j)/2],
                          [0., 0., (1 - 1.j)/2, (1 + 1.j)/2]])

db = np.array([[1, 0., 0., 0],
               [0., np.cos(3*np.pi/8), -1j*np.sin(3*np.pi/8), 0.],
               [0., -np.sin(3*np.pi/8), np.cos(3*np.pi/8), 0.],
               [0, 0., 0., 1]])

dcnot = np.array([[1., 0., 0., 0.],
                  [0., 0., 0., 1.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.]])

c = np.cos(np.pi / 8)
s = np.sin(np.pi / 8)
ecp = 0.5 * np.array([[2*c, 0., 0., -1j*2*s],
                      [0., (1+1.j)*(c-s), (1-1.j)*(c+s), 0.],
                      [0., (1-1.j)*(c+s), (1+1.j)*(c-s), 0.],
                      [-1j*2*s, 0., 1., 2*c]])

fswap = np.array([[1., 0., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 0., -1.]])

iswap = np.array([[1., 0., 0., 0.],
                  [0., 0., 1.j, 0.],
                  [0., 1.j, 0., 0.],
                  [0., 0., 0., 1.]])

m = 1/np.sqrt(2) * np.array([[1., 1.j, 0., 0.],
                             [0., 0., 1.j, 1.],
                             [0., 0., 1.j, -1.],
                             [1., -1.j, 0., 0.]])

ms = 1/np.sqrt(2) * np.array([[1., 0., 0., 1.j],
                              [0., 1., 1.j, 0.],
                              [0., 1.j, 0., 0.],
                              [1.j, 0., 0., 1.]])

qft2 = np.array([[1., 1., 1., 1.],
                 [1., 1.j, -1., -1.j],
                 [1., -1., 1., -1.],
                 [1., -1.j, -1., 1.j]])

swap = np.array([[1., 0., 0., 0.],
                 [0., 0., 1., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 0., 1.]])

sqrt_swap = np.array([[1., 0., 0., 0.],
                      [0., 0.5*(1+1j), 0.5*(1-1j), 0.],
                      [0., 0.5*(1-1j), 0.5*(1+1j), 0.],
                      [0., 0., 0., 1.]])

sqrt_swap_dag = np.array([[1., 0., 0., 0.],
                          [0., 0.5*(1-1j), 0.5*(1+1j), 0.],
                          [0., 0.5*(1+1j), 0.5*(1-1j), 0.],
                          [0., 0., 0., 1.]])

syc = np.array([[1., 0., 0., 0.],
                [0., 0., -1.j, 0.],
                [0., -1.j, 0., 0.],
                [0., 0., 0., np.exp(-1j*np.pi/6)]])

w_gate =  np.array([[1., 0., 0., 0.],
                    [0., 1/np.sqrt(2), 1/np.sqrt(2), 0.],
                    [0., 1/np.sqrt(2), -1/np.sqrt(2), 0.],
                    [0., 0., 0., 1.]])


def a_gate(theta: Union[int, float], phi: Union[int, float]) -> NDArray:
    """A-gate defintion.

    Args:
        theta (int, float): first gate parameter.
        phi (int, float):   second gate parameter.

    Returns:
        NDArray: A-gate with angles theta and phi.
    """
    return np.array([[1., 0., 0., 0.],
                     [0., np.cos(theta), np.exp(1j*phi)*np.sin(theta), 0.],
                     [0., np.exp(-1j*phi)*np.sin(theta), -np.cos(theta), 0.],
                     [0., 0., 0., 1.]])


def barenco(phi:   Union[int, float],
            alpha: Union[int, float],
            theta: Union[int, float]
    ) -> NDArray:
    """Barenco gate defintion.

    Args:
        phi (int, float): off-diagonal phase angle.
        alpha (int, float): diagonal phase angle.
        theta (int, float): rotation angle.

    Returns:
        NDArray: Barenco gate with angles phi, alpha, theta.
    """
    u00 = np.exp(1j*alpha)*np.cos(theta)
    u01 = -1j*np.exp(1j*(alpha-phi))*np.sin(theta)
    u10 = -1j*np.exp(1j*(alpha+phi))*np.sin(theta)
    u11 = np.exp(1j*alpha)*np.cos(theta)
    return np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., u00, u01],
                     [0., 0., u10, u11]])


def cphase(theta: Union[int, float] = 1) -> NDArray:
    """Controlled-Phase gate defintion.

    Args:
        theta (int, float): phase factor. Defaults to 1.

    Returns:
        NDArray: ZZ gate with a rotation angle of pi*theta.
    """
    return np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., np.exp(1j*np.pi*theta)]])


def crot(theta: Union[int, float],
         n:     Union[List, NDArray]
    ) -> NDArray:
    """Controlled-Rotation gate defintion.

    Args:
        theta (int, float): rotation angle.
        n (List, NDArray):  unit vector defining the axis of rotation.

    Returns:
        NDArray: Crot gate with a rotation angle of theta about the axis
            defined by n = (nx, ny, nz).
    """
    nx, ny, nz = n
    assert nx**2 + ny**2 + nz**2 == 1, 'n must be a unit vector!'
    return scipy.linalg.expm(-1j*theta/2 * np.kron(id - z, nx*x + ny*y + nz*z))


def fsim(theta: Union[int, float], phi: Union[int, float]) -> NDArray:
    """FSim gate defintion.

    Args:
        theta (int, float): rotation angle parameter.
        phi (int, float):   ZZ phase parameter.

    Returns:
        NDArray: FSim with rotation angle theta and phase phi.
    """
    return np.array([[1., 0., 0., 0.],
                     [0., np.cos(theta), -1.j*np.sin(theta), 0.],
                     [0., -1.j*np.sin(theta), np.cos(theta), 0.],
                     [0., 0., 0., np.exp(-1.j*phi)]])


def givens(theta: Union[int, float]) -> NDArray:
    """Givens gate defintion.

    Args:
        theta (int, float): rotation angle.

    Returns:
        NDArray: Givens gate with a rotation angle of theta.
    """
    return np.array([[1., 0., 0., 0],
                     [0., np.cos(theta), -np.sin(theta), 0.],
                     [0., np.sin(theta), np.cos(theta), 0.],
                     [0, 0., 0., 1.]])


def pswap(theta: Union[int, float]) -> NDArray:
    """pSWAP gate defintion.

    Args:
        theta (int, float): rotation angle.

    Returns:
        NDArray: pSWAP gate with a rotation angle of theta.
    """
    return np.array([[1., 0., 0., 0],
                     [0., 0., np.exp(1j*theta), 0.],
                     [0., np.exp(1j*theta), 0., 0.],
                     [0, 0., 0., 1.]])


def swap_alpha(alpha: Union[int, float]) -> NDArray:
    """SWAP-alpha gate defintion.

    Args:
        alpha (int, float): power of the SWAP gate.

    Returns:
        NDArray: SWAP-alpha gate with a power given by alpha.
    """
    angle = np.pi * alpha / 2
    return  np.exp(angle) * np.array([
        [np.exp(-1j*angle), 0., 0., 0],
        [0., np.cos(angle), 1j*np.sin(angle), 0.],
        [0., 1j*np.sin(angle), np.cos(angle), 0.],
        [0, 0., 0., np.exp(-angle)]
    ])


def xx(t: Union[int, float] = 1) -> NDArray:
    """XX gate defintion.

    Args:
        t (int, float): phase factor. Defaults to 1.

    Returns:
        NDArray: XX gate with a phase factor t.
    """
    return np.array([[np.cos(np.pi*t/2), 0., 0., -1j*np.sin(np.pi*t/2)],
                     [0., np.cos(np.pi*t/2), -1j*np.sin(np.pi*t/2), 0.],
                     [0., -1j*np.sin(np.pi*t/2), np.cos(np.pi*t/2), 0.],
                     [-1j*np.sin(np.pi*t/2), 0., 0., np.cos(np.pi*t/2)]])


def xy(t: Union[int, float] = 1) -> NDArray:
    """XY gate defintion.

    Args:
        t (int, float): phase factor. Defaults to 1.

    Returns:
        NDArray: XY gate with a phase factor t.
    """
    return np.array([[1, 0., 0., 0],
                     [0., np.cos(np.pi*t), -1j*np.sin(np.pi*t), 0.],
                     [0., -1j*np.sin(np.pi*t), np.cos(np.pi*t), 0.],
                     [0, 0., 0., 1]])


def yy(t: Union[int, float] = 1) -> NDArray:
    """YY gate defintion.

    Args:
        t (int, float): phase factor. Defaults to 1.

    Returns:
        NDArray: YY gate with a phase factor t.
    """
    return np.array([[np.cos(np.pi*t/2), 0., 0., 1j*np.sin(np.pi*t/2)],
                     [0., np.cos(np.pi*t/2), -1j*np.sin(np.pi*t/2), 0.],
                     [0., -1j*np.sin(np.pi*t/2), np.cos(np.pi*t/2), 0.],
                     [1j*np.sin(np.pi*t/2), 0., 0., np.cos(np.pi*t/2)]])


def zz(t: Union[int, float] = 1) -> NDArray:
    """ZZ (Ising) gate defintion.

    Args:
        t (int, float): phase factor. Defaults to 1.

    Returns:
        NDArray: ZZ gate with a phase factor t.
    """
    return np.array([[np.exp(-1j*np.pi*t/2), 0., 0., 0.],
                     [0., np.exp(1j*np.pi*t/2), 0., 0.],
                     [0., 0., np.exp(1j*np.pi*t/2), 0.],
                     [0., 0., 0., np.exp(-1j*np.pi*t/2)]])


def nearest_kronecker_product(C) -> Tuple[NDArray]:
    """Finds the closest Kronecker product to C in the Frobenius norm.

    Args:
        C (NDArray): matrix to decompose into the Kronecker product of two
            other matrices.

    Returns:
        tuple: tuple of matrices whose Kronecker product form C.
    """
    C = C.reshape(2, 2, 2, 2)
    C = C.transpose(0, 2, 1, 3)
    C = C.reshape(4, 4)
    u, sv, vh = np.linalg.svd(C)
    A = np.sqrt(sv[0]) * u[:, 0].reshape(2, 2)
    B = np.sqrt(sv[0]) * vh[0, :].reshape(2, 2)
    return A, B
    

class AGate(Gate):
    """Class for the A-gate."""

    def __init__(self,
            theta:  Union[int, float], 
            phi:    Union[int, float],
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the a_gate gate.
        
        Args:
            theta (int, float):   first gate parameter.
            phi (int, float):     second gate parameter.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(a_gate(theta, phi), qubits)
        
        if theta == 0 and phi == 0:
            loc_equiv = 'CX'
        elif theta == np.pi/4 and phi == 0:
            loc_equiv = 'WGate'
        elif theta == np.pi/2 and phi == 0:
            loc_equiv = 'SWAP'
        else:
            loc_equiv = None

        self._properties['locally_equivalent'] = loc_equiv
        self._properties['name'] = 'AGate'
        self._properties['params'] = {
            'theta': theta,
            'phi':   phi,
        }


class Barenco(Gate):
    """Class for the Barenco gate."""

    def __init__(self,
            phi:    Union[int, float],
            alpha:  Union[int, float],
            theta:  Union[int, float],
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the barenco gate.
        
        Args:
            phi (int, float):     off-diagonal phase angle.
            alpha (int, float):   diagonal phase angle.
            theta (int, float):   rotation angle.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(barenco(phi, alpha, theta), qubits)
        self._properties['locally_equivalent'] = 'XX'
        self._properties['name'] = 'Barenco'
        self._properties['params'] = {
            'theta': theta,
            'alpha': alpha,
            'phi':   phi
        }
    

class BGate(Gate):
    """Class for the Berkeley (B) gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the berkeley gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(b_gate, qubits)
        self._properties['alias'] = 'Berkeley'
        self._properties['name'] = 'BGate'
    

class CH(Gate):
    """Class for the Controlled-Hadamard gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the ch gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(ch, qubits)
        self._properties['alias'] = 'Controlled-H'
        self._properties['locally_equivalent'] = 'CX'
        self._properties['name'] = 'CH'


class CNot(Gate):
    """Class for the Controlled-Not gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the cnot gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(cnot, qubits)
        self._properties['alias'] = 'CX'
        self._properties['locally_equivalent'] = 'CY, CZ'
        self._properties['name'] = 'CNot'


class CPhase(Gate):
    """Class for the Controlled-Phase gate."""

    def __init__(self, 
            theta:  Union[int, float] = 1,
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the cphase parametrized gate.
        
        Args:
            theta (int, float):   phase factor. Defaults to 1.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(cphase(theta), qubits)

        if theta == 1:
            alias = 'CZ'
        elif theta == 0.5:
            alias = 'CS'
        elif theta == -0.5:
            alias = 'CSdag'
        elif theta == 0.25:
            alias = 'CT'
        elif theta == -0.25:
            alias = 'CTdag'
        else:
            alias = None

        if theta == 1:
            loc_equiv = 'CX, CY'
        else:
            loc_equiv = None

        self._properties['alias'] = alias
        self._properties['locally_equivalent'] = loc_equiv
        self._properties['name'] = 'CPhase'
        self._properties['params'] = {
            'angle': theta,
        }
    

class CRot(Gate):
    """Class for the Controlled-Rotation gate."""

    def __init__(self,
            theta:  Union[int, float],
            n:      Union[List, NDArray],
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the crot gate.
        
        Args:
            theta (int, float):   rotation angle.
            n (List, NDarray):    unit vector defining the axis of rotation.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(crot(theta, n), qubits)
        self._properties['name'] = 'CRot'
        self._properties['params'] = {
            'angle': theta,
            'axis':  n
        }
    

class CS(Gate):
    """Class for the Controlled-S gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the cphase parametrized gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(cphase(0.5), qubits)
        self._properties['alias'] = 'sqrt(CZ)'
        self._properties['name'] = 'CS'
        self._properties['params'] = {
            'angle': np.pi/2,
        }
    

class CSdag(Gate):
    """Class for the Controlled-S^dagger gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the cphase parametrized gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(cphase(-0.5), qubits)
        self._properties['alias'] = 'sqrt(CZ)dag'
        self._properties['name'] = 'CSDdag'
        self._properties['params'] = {
            'angle': -np.pi/2,
        }
    

class CT(Gate):
    """Class for the Controlled-T gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the cphase parametrized gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(cphase(0.25), qubits)
        self._properties['alias'] = 'CZ^(1/4)'
        self._properties['name'] = 'CT'
        self._properties['params'] = {
            'angle': np.pi/4,
        }
    

class CTdag(Gate):
    """Class for the Controlled-T^dagger gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the cphase parametrized gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(cphase(-0.25), qubits)
        self._properties['alias'] = 'CZ^(-1/4)'
        self._properties['name'] = 'CTdag'
        self._properties['params'] = {
            'angle': -np.pi/4,
        }
    

class CV(Gate):
    """Class for the Controlled-V (Square-root CNot) gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the cv gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(cv, qubits)
        self._properties['alias'] = 'SqrtCNot'
        self._properties['locally_equivalent'] = 'CVdag'
        self._properties['name'] = 'CV'
        self._properties['params'] = {
            'angle': np.pi/2
        }
    

class CX(Gate):
    """Class for the Controlled-X gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the cx gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(cx, qubits)
        self._properties['alias'] = 'CNot'
        self._properties['locally_equivalent'] = 'CY, CZ'
        self._properties['name'] = 'CX'
        self._properties['params'] = {
            'angle': np.pi
        }
    

class CY(Gate):
    """Class for the Controlled-Y gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the cy gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(cy, qubits)
        self._properties['locally_equivalent'] = 'CX, CZ'
        self._properties['name'] = 'CY'
        self._properties['params'] = {
            'angle': np.pi
        }
    

class CZ(Gate):
    """Class for the Controlled-Z gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the cz gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(cz, qubits)
        self._properties['locally_equivalent'] = 'CX, CY'
        self._properties['name'] = 'CZ'
        self._properties['params'] = {
            'angle': np.pi
        }
    

class DB(Gate):
    """Class for the Dagwood-Bumstead gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the db gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(db, qubits)
        self._properties['locally_equivalent'] = 'XY'
        self._properties['name'] = 'DB'
    

class DCNot(Gate):
    """Class for the Double Controlled-NOT gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the dcnot gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(dcnot, qubits)
        self._properties['locally_equivalent'] = 'fSWAP, iSWAP'
        self._properties['name'] = 'DCNot'
    

class ECP(Gate):
    """Class for the ECP gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the ecp gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(ecp, qubits)
        self._properties['locally_equivalent'] = 'WGate'
        self._properties['name'] = 'ECP'
    

class FSim(Gate):
    """Class for the FSim gate."""

    def __init__(self,
            theta:  Union[int, float], 
            phi:    Union[int, float],
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the fsim gate.
        
        Args:
            theta (int, float): gate rotation angle.
            phi (int, float):   ZZ phase parameter.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(fsim(theta, phi), qubits)
        self._properties['name'] = 'FSim'
        self._properties['params'] = {
            'theta': theta,
            'phi':   phi,
        }
    

class fSWAP(Gate):
    """Class for the fSWAP gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the fswap gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(fswap, qubits)
        self._properties['locally_equivalent'] = 'iSWAP, DCNot'
        self._properties['name'] = 'fSWAP'
    

class Givens(Gate):
    """Class for the Givens gate."""

    def __init__(self,
            theta: Union[int, float],
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the Givens parametrized gate.
        
        Args:
            theta (int, float):   rotation angle.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(givens(theta), qubits)
        self._properties['locally_equivalent'] = 'XY'
        self._properties['name'] = 'Givens'
        self._properties['params'] = {
            'angle': theta
        }


class iSWAP(Gate):
    """Class for the iSWAP gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the iswap gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(iswap, qubits)
        self._properties['locally_equivalent'] = 'fSWAP, DCNot'
        self._properties['name'] = 'iSWAP'
    

class M(Gate):
    """Class for the Magic gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the m gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(m, qubits)
        self._properties['alias'] = 'Magic'
        self._properties['locally_equivalent'] = 'CX'
        self._properties['name'] = 'M'
    

class MS(Gate):
    """Class for the Mølmer-Sørensen gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the ms gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(ms, qubits)
        self._properties['locally_equivalent'] = 'CX'
        self._properties['name'] = 'MS'
    

class pSWAP(Gate):
    """Class for the Parametric SWAP (pSWAP) gate."""

    def __init__(self,
            theta: Union[int, float],
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the pswap gate.
        
        Args:
            theta (int, float): rotation angle.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(pswap(theta), qubits)
        self._properties['locally_equivalent'] = 'QFT2'
        self._properties['name'] = 'pSWAP'
        self._properties['params'] = {
            'angle': theta
        }


class QFT2(Gate):
    """Class for the QFT gate on 2 qubits."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the pswap gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(qft2, qubits)
        self._properties['locally_equivalent'] = 'pSWAP'
        self._properties['name'] = 'QFT2'


class SWAP(Gate):
    """Class for the SWAP gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the swap gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(swap, qubits)
        self._properties['name'] = 'SWAP'
    

class SWAPAlpha(Gate):
    """Class for the SWAP-alpha gate."""

    def __init__(self,
            alpha: Union[int, float],
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the swap-alpha gate.
        
        Args:
            alpha (int, float): power of the SWAP gate.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(swap_alpha(alpha), qubits)
        self._properties['locally_equivalent'] = 'SqrtSWAP, SqrtSWAPdag'
        self._properties['name'] = 'SWAPAlpha'
        self._properties['params'] = {
            'alpha': alpha
        }
    

class SqrtSWAP(Gate):
    """Class for the Sqrt SWAP gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the sqrt_swap gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(sqrt_swap, qubits)
        self._properties['locally_equivalent'] = 'SWAPAlpha, SqrtSWAPdag'
        self._properties['name'] = 'SqrtSWAP'
    

class SqrtSWAPdag(Gate):
    """Class for the Sqrt(SWAP)^dagger gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the sqrt_swap_dag gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(sqrt_swap_dag, qubits)
        self._properties['locally_equivalent'] = 'SWAPAlpha, SqrtSWAP'
        self._properties['name'] = 'SqrtSWAPdag'
    

class Syc(Gate):
    """Class for the Sycamore (Syc) gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the syc gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(syc, qubits)
        self._properties['alias'] = 'Sycamore'
        self._properties['name'] = 'Syc'
    

class WGate(Gate):
    """Class for the W gate."""

    def __init__(self, qubits: Tuple = (0, 1)) -> None:
        """Initialize using the w gate.
        
        Args:
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(w_gate, qubits)
        self._properties['locally_equivalent'] = 'ECP'
        self._properties['name'] = 'WGate'
    

class XX(Gate):
    """Class for the XX gate."""

    def __init__(self,
            t: Union[int, float] = 1,
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the xx parametrized gate.
        
        Args:
            t (int, float):       phase factor. Defaults to 1.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(xx(t), qubits)
        self._properties['name'] = 'XX'
        self._properties['params'] = {
            'phase factor': t
        }
    

class XY(Gate):
    """Class for the XY gate."""

    def __init__(self,
            t: Union[int, float] = 1,
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the xy parametrized gate.
        
        Args:
            t (int, float):       phase factor. Defaults to 1.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(xy(t), qubits)
        self._properties['alias'] = 'piSWAP'
        self._properties['name'] = 'XY'
        self._properties['params'] = {
            'phase factor': t
        }
    

class YY(Gate):
    """Class for the YY gate."""

    def __init__(self,
            t: Union[int, float] = 1,
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the yy parametrized gate.
        
        Args:
            t (int, float):       phase factor. Defaults to 1.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(yy(t), qubits)
        self._properties['name'] = 'YY'
        self._properties['params'] = {
            'phase factor': t
        }
    

class ZZ(Gate):
    """Class for the ZZ gate."""

    def __init__(self,
            t: Union[int, float] = 1,
            qubits: Tuple = (0, 1)
        ) -> None:
        """Initialize using the zz parametrized gate.
        
        Args:
            t (int, float):       phase factor. Defaults to 1.
            qubits (int | tuple): qubit labels. Defaults to (0, 1).
        """
        super().__init__(zz(t), qubits)
        self._properties['name'] = 'ZZ'
        self._properties['params'] = {
            'phase factor': t
        }