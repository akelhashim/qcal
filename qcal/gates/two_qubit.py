"""Submodule for storing two-qubit gate definitions.

See https://threeplusone.com/pubs/on_gates.pdf for relevant definitions.
"""
from qcal.gates.gate import Gate
from qcal.gates.single_qubit import id, x, y, z

import numpy as np
import scipy

from numpy.typing import NDArray
from typing import Tuple, Union


__all__ = [
    'CNot', 'CX', 'CY', 'CZ', 'CH', 'M', 'MS',
    'DCNot', 'fSWAP', 'iSWAP', 
    'SWAP',
    'XX', 'YY', 'ZZ',
    'CPhase', 'CS', 'CSdag', 'CT', 'CTdag', 'CRot', 'Barenco', 'CV',
    'XY', 'Givens', 'DB',
    'SWAPAlpha', 'SqrtSWAP', 'SqrtSWAPDag',
    'pSWAP', 'QFT2',
    'BGate', 'ECP', 'WGate', 'AGate',
    'FSim', 'Syc'
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
         nx: Union[int, float],
         ny: Union[int, float],
         nz: Union[int, float]
    ) -> NDArray:
    """Controlled-Rotation gate defintion.

    Args:
        theta (int, float): rotation angle.
        nx (int, float):    x component of unit vector.
        ny (int, float):    y component of unit vector.
        nz (int, float):    z component of unit vector.

    Returns:
        NDArray: Crot gate with a rotation angle of theta about the axis
            defined by (nx, ny, nz).
    """
    assert nx**2 + ny**2 + nz**2 == 1, '(nx, ny, nz) must be a unit vector!'
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
            theta: Union[int, float], 
            phi:   Union[int, float],
        ) -> None:
        """Initialize using the a_gate gate.
        
        Args:
            theta (int, float): first gate parameter.
            phi (int, float):   secont gate parameter.
        """
        super().__init__(a_gate(theta, phi))
        self._theta = theta
        self._phi = phi
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        if self._theta == 0 and self._phi == 0:
            return 'CX'
        elif self._theta == np.pi/4 and self._phi == 0:
            return 'WGate'
        elif self._theta == np.pi/2 and self._phi == 0:
            return 'SWAP'
        else:
            return None
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'AGate'


class Barenco(Gate):
    """Class for the Barenco gate."""

    def __init__(self,
            phi:   Union[int, float],
            alpha: Union[int, float],
            theta: Union[int, float]
        ) -> None:
        """Initialize using the barenco gate.
        
        Args:
            phi (int, float):   off-diagonal phase angle.
            alpha (int, float): diagonal phase angle.
            theta (int, float): rotation angle.
        """
        super().__init__(barenco(phi, alpha, theta))
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'XX'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Barenco'
    

class BGate(Gate):
    """Class for the Berkeley (B) gate."""

    def __init__(self) -> None:
        """Initialize using the berkeley gate."""
        super().__init__(b_gate)
    
    @property
    def alias(self) -> str:
        """Returns the alias(es) of the gate.

        Returns:
            str: alias of the gate.
        """
        return 'Berkeley'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'BGate'
    

class CH(Gate):
    """Class for the Controlled-Hadamard gate."""

    def __init__(self) -> None:
        """Initialize using the ch gate."""
        super().__init__(ch)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CH'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'CX'


class CNot(Gate):
    """Class for the Controlled-Not gate."""

    def __init__(self) -> None:
        """Initialize using the cnot gate."""
        super().__init__(cnot)
    
    @property
    def alias(self) -> str:
        """Returns the alias(es) of the gate.

        Returns:
            str: alias of the gate.
        """
        return 'CX'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'CY\nCZ'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CNot'
    

class CPhase(Gate):
    """Class for the Controlled-Phase gate."""

    def __init__(self, theta: Union[int, float] = 1) -> None:
        """Initialize using the cphase parametrized gate.
        
        Args:
            theta (int, float): phase factor. Defaults to 1.
        """
        self._theta = theta
        super().__init__(cphase(theta))

    @property
    def alias(self) -> str:
        """Returns the alias(es) of the gate.

        Returns:
            str: alias of the gate.
        """
        if self._theta == 1:
            return 'CZ'
        elif self._theta == 0.5:
            return 'CS'
        elif self._theta == -0.5:
            return 'CSdag'
        elif self._theta == 0.25:
            return 'CT'
        elif self._theta == -0.25:
            return 'CTdag'
        else:
            return None
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        if self._theta == 1:
            return 'CX\nCY'
        else:
            return None
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CPhase'
    

class CRot(Gate):
    """Class for the Controlled-Rotation gate."""

    def __init__(self,
            theta: Union[int, float],
            nx: Union[int, float],
            ny: Union[int, float],
            nz: Union[int, float]
        ) -> None:
        """Initialize using the crot gate.
        
        Args:
            theta (int, float): rotation angle.
            nx (int, float):    x component of unit vector.
            ny (int, float):    y component of unit vector.
            nz (int, float):    z component of unit vector.
        """
        super().__init__(crot(theta, nx, ny, nz))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CRot'
    

class CS(Gate):
    """Class for the Controlled-S gate."""

    def __init__(self) -> None:
        """Initialize using the cphase parametrized gate."""
        super().__init__(cphase(0.5))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CS'
    

class CSdag(Gate):
    """Class for the Controlled-S^dagger gate."""

    def __init__(self) -> None:
        """Initialize using the cphase parametrized gate."""
        super().__init__(cphase(-0.5))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CSdag'
    

class CT(Gate):
    """Class for the Controlled-T gate."""

    def __init__(self) -> None:
        """Initialize using the cphase parametrized gate."""
        super().__init__(cphase(0.25))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CT'
    

class CTdag(Gate):
    """Class for the Controlled-T^dagger gate."""

    def __init__(self) -> None:
        """Initialize using the cphase parametrized gate."""
        super().__init__(cphase(-0.25))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CTdag'
    

class CV(Gate):
    """Class for the Controlled-V (Square-root CNot) gate."""

    def __init__(self) -> None:
        """Initialize using the cv gate."""
        super().__init__(cv)
    
    @property
    def alias(self) -> str:
        """Returns the alias(es) of the gate.

        Returns:
            str: alias of the gate.
        """
        return 'SqrtCNot'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'CY\nCZ'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CV'
    

class CX(Gate):
    """Class for the Controlled-X gate."""

    def __init__(self) -> None:
        """Initialize using the cx gate."""
        super().__init__(cx)
    
    @property
    def alias(self) -> str:
        """Returns the alias(es) of the gate.

        Returns:
            str: alias of the gate.
        """
        return 'CNot'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'CY\nCZ'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CX'
    

class CY(Gate):
    """Class for the Controlled-Y gate."""

    def __init__(self) -> None:
        """Initialize using the cy gate."""
        super().__init__(cy)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CY'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'CX\nCZ'
    

class CZ(Gate):
    """Class for the Controlled-Z gate."""

    def __init__(self) -> None:
        """Initialize using the cz gate."""
        super().__init__(cz)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'CZ'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'CY\nCZ'
    

class DB(Gate):
    """Class for the Dagwood-Bumstead gate."""

    def __init__(self) -> None:
        """Initialize using the db gate."""
        super().__init__(db)

    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'XY'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'DB'
    

class DCNot(Gate):
    """Class for the Double Controlled-NOT gate."""

    def __init__(self) -> None:
        """Initialize using the dcnot gate."""
        super().__init__(dcnot)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'DCNot'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'fSWAP\niSWAP'
    

class ECP(Gate):
    """Class for the ECP gate."""

    def __init__(self) -> None:
        """Initialize using the ecp gate."""
        super().__init__(ecp)

    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'WGate'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'ECP'
    

class FSim(Gate):
    """Class for the FSim gate."""

    def __init__(self,
            theta: Union[int, float], 
            phi:   Union[int, float],
        ) -> None:
        """Initialize using the fsim gate.
        
        Args:
            theta (int, float): gate rotation angle.
            phi (int, float):   ZZ phase parameter.
        """
        super().__init__(fsim(theta, phi))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'FSim'
    

class fSWAP(Gate):
    """Class for the fSWAP gate."""

    def __init__(self) -> None:
        """Initialize using the fswap gate."""
        super().__init__(fswap)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'fSWAP'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'iSWAP\nDCNot'
    

class Givens(Gate):
    """Class for the Givens gate."""

    def __init__(self, theta: Union[int, float]) -> None:
        """Initialize using the Givens parametrized gate.
        
        Args:
            theta (int, float): rotation angle.
        """
        super().__init__(givens(theta))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'Givens'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'XY'


class iSWAP(Gate):
    """Class for the iSWAP gate."""

    def __init__(self) -> None:
        """Initialize using the iswap gate."""
        super().__init__(iswap)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'iSWAP'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'fSWAP\nDCNot'
    

class M(Gate):
    """Class for the Magic gate."""

    def __init__(self) -> None:
        """Initialize using the m gate."""
        super().__init__(m)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'M'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'CX'
    

class MS(Gate):
    """Class for the Mølmer-Sørensen gate."""

    def __init__(self) -> None:
        """Initialize using the ms gate."""
        super().__init__(ms)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate
        """
        return 'MS'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'CX'
    

class pSWAP(Gate):
    """Class for the Parametric SWAP (pSWAP) gate."""

    def __init__(self, theta: Union[int, float]) -> None:
        """Initialize using the pswap gate."""
        super().__init__(pswap(theta))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'pSWAP'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'QFT2'


class QFT2(Gate):
    """Class for the QFT gate on 2 qubits."""

    def __init__(self) -> None:
        """Initialize using the pswap gate."""
        super().__init__(qft2)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'QFT2'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'pSWAP'


class SWAP(Gate):
    """Class for the SWAP gate."""

    def __init__(self) -> None:
        """Initialize using the swap gate."""
        super().__init__(swap)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'SWAP'
    

class SWAPAlpha(Gate):
    """Class for the SWAP-alpha gate."""

    def __init__(self, alpha: Union[int, float]) -> None:
        """Initialize using the swap-alpha gate."""
        super().__init__(swap_alpha(alpha))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'SWAP-alpha'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates
        """
        return 'SqrtSWAP\nSqrtSWAPDag'
    

class SqrtSWAP(Gate):
    """Class for the Sqrt SWAP gate."""

    def __init__(self) -> None:
        """Initialize using the sqrt_swap gate."""
        super().__init__(sqrt_swap)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'SqrtSWAP'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates
        """
        return 'SWAPAlpha\nSqrtSWAPDag'
    

class SqrtSWAPDag(Gate):
    """Class for the Sqrt SWAP dagger gate."""

    def __init__(self) -> None:
        """Initialize using the sqrt_swap_dag gate."""
        super().__init__(sqrt_swap_dag)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'SqrtSWAPDag'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'SWAPAlpha\nSqrtSWAP'
    

class Syc(Gate):
    """Class for the Sycamore (Syc) gate."""

    def __init__(self) -> None:
        """Initialize using the syc gate."""
        super().__init__(syc)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'SWAP'
    

class WGate(Gate):
    """Class for the W gate."""

    def __init__(self) -> None:
        """Initialize using the w gate."""
        super().__init__(w_gate)
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'WGate'
    
    @property
    def locally_equivalent(self) -> str:
        """Returns the names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return 'ECP'
    

class XX(Gate):
    """Class for the XX gate."""

    def __init__(self, t: Union[int, float] = 1) -> None:
        """Initialize using the xx parametrized gate.
        
        Args:
            t (int, float): phase factor. Defaults to 1.
        """
        super().__init__(xx(t))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'XX'
    

class XY(Gate):
    """Class for the XY gate."""

    def __init__(self, t: Union[int, float] = 1) -> None:
        """Initialize using the xy parametrized gate.
        
        Args:
            t (int, float): phase factor. Defaults to 1.
        """
        super().__init__(xy(t))
    
    @property
    def alias(self) -> str:
        """Returns the alias(es) of the gate.

        Returns:
            str: alias of the gate.
        """
        return 'piSWAP'
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'XY'
    

class YY(Gate):
    """Class for the YY gate."""

    def __init__(self, t: Union[int, float] = 1) -> None:
        """Initialize using the yy parametrized gate.
        
        Args:
            t (int, float): phase factor. Defaults to 1.
        """
        super().__init__(yy(t))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'YY'
    

class ZZ(Gate):
    """Class for the ZZ gate."""

    def __init__(self, t: Union[int, float] = 1) -> None:
        """Initialize using the zz parametrized gate.
        
        Args:
            t (int, float): phase factor. Defaults to 1.
        """
        super().__init__(zz(t))
    
    @property
    def name(self) -> str:
        """Returns the name of the gate.

        Returns:
            str: name of the gate.
        """
        return 'ZZ'