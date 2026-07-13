"""Submodule for storing two-qubit gate definitions.

See https://threeplusone.com/pubs/on_gates.pdf for relevant definitions.
"""
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import List, Tuple, Union

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray

from qcal.gate.gate import Gate
from qcal.gate.single_qubit import id, x, y, z

__all__ = (
    'AGate',
    'Barenco',
    'BGate',
    'bSWAP',
    'CH',
    'CNOT',
    'CPhase',
    'CRot',
    'CS',
    'CSdag',
    'CT',
    'CTdag',
    'CV',
    'CX',
    'CY',
    'CZ',
    'DB',
    'DCNOT',
    'ECP',
    'FSim',
    'fSWAP',
    'Givens',
    'iSWAP',
    'M',
    'MS',
    'pSWAP',
    'QFT2',
    'SqrtSWAP',
    'SqrtSWAPdag',
    'SWAPAlpha',
    'SWAP',
    'Syc',
    'WGate',
    'XX',
    'XY',
    'YY',
    'ZZ'
)

paulis = (
    np.kron(id, id),
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
)


b_gate = berkeley = np.sqrt(2 - np.sqrt(2)) / 2 * np.array([
    [1. + np.sqrt(2), 0., 0., 1.j],
    [0., 1., 1.j*(1 + np.sqrt(2)), 0.],
    [0., 1.j*(1 + np.sqrt(2)), 1., 0.],
    [1.j, 0., 0., 1. + np.sqrt(2)]
])

bswap = np.array([
    [0., 0., 0., 1.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [1., 0., 0., 0.]
])

cnot = cx = np.array([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
    [0., 0., 1., 0.]
])

cy = np.array([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., -1.j],
    [0., 0., 1.j, 0.]
])

cz = np.array([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., -1.]
])

ch = np.array([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1/np.sqrt(2), 1/np.sqrt(2)],
    [0., 0., 1/np.sqrt(2), -1/np.sqrt(2)]
])

cv = sqrtcnot = np.array([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., (1 + 1.j)/2, (1 - 1.j)/2],
    [0., 0., (1 - 1.j)/2, (1 + 1.j)/2]
])

db = np.array([
    [1, 0., 0., 0],
    [0., np.cos(3*np.pi/8), -1j*np.sin(3*np.pi/8), 0.],
    [0., -np.sin(3*np.pi/8), np.cos(3*np.pi/8), 0.],
    [0, 0., 0., 1]
])

dcnot = np.array([
    [1., 0., 0., 0.],
    [0., 0., 0., 1.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.]
])

c = np.cos(np.pi / 8)
s = np.sin(np.pi / 8)
ecp = 0.5 * np.array([
    [2*c, 0., 0., -1j*2*s],
    [0., (1+1.j)*(c-s), (1-1.j)*(c+s), 0.],
    [0., (1-1.j)*(c+s), (1+1.j)*(c-s), 0.],
    [-1j*2*s, 0., 1., 2*c]
])

fswap = np.array([
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., -1.]
])

iswap = np.array([
    [1., 0., 0., 0.],
    [0., 0., 1.j, 0.],
    [0., 1.j, 0., 0.],
    [0., 0., 0., 1.]
])

m = 1/np.sqrt(2) * np.array([
    [1., 1.j, 0., 0.],
    [0., 0., 1.j, 1.],
    [0., 0., 1.j, -1.],
    [1., -1.j, 0., 0.]
])

ms = 1/np.sqrt(2) * np.array([
    [1., 0., 0., 1.j],
    [0., 1., 1.j, 0.],
    [0., 1.j, 0., 0.],
    [1.j, 0., 0., 1.]
])

qft2 = np.array([
    [1., 1., 1., 1.],
    [1., 1.j, -1., -1.j],
    [1., -1., 1., -1.],
    [1., -1.j, -1., 1.j]
])

swap = np.array([
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.]
])

sqrt_swap = np.array([
    [1., 0., 0., 0.],
    [0., 0.5*(1+1j), 0.5*(1-1j), 0.],
    [0., 0.5*(1-1j), 0.5*(1+1j), 0.],
    [0., 0., 0., 1.]
])

sqrt_swap_dag = np.array([
    [1., 0., 0., 0.],
    [0., 0.5*(1-1j), 0.5*(1+1j), 0.],
    [0., 0.5*(1+1j), 0.5*(1-1j), 0.],
    [0., 0., 0., 1.]
])

syc = np.array([
    [1., 0., 0., 0.],
    [0., 0., -1.j, 0.],
    [0., -1.j, 0., 0.],
    [0., 0., 0., np.exp(-1j*np.pi/6)]
])

w_gate =  np.array([
    [1., 0., 0., 0.],
    [0., 1/np.sqrt(2), 1/np.sqrt(2), 0.],
    [0., 1/np.sqrt(2), -1/np.sqrt(2), 0.],
    [0., 0., 0., 1.]
])


def a_gate(theta: float, phi: float) -> NDArray:
    """A-gate defintion.

    Args:
        theta (float): first gate parameter.
        phi (float): second gate parameter.

    Returns:
        NDArray: A-gate with angles theta and phi.
    """
    return np.array([
        [1., 0., 0., 0.],
        [0., np.cos(theta), np.exp(1j*phi)*np.sin(theta), 0.],
        [0., np.exp(-1j*phi)*np.sin(theta), -np.cos(theta), 0.],
        [0., 0., 0., 1.]
    ])


def barenco(phi: float, alpha: float, theta: float) -> NDArray:
    """Barenco gate defintion.

    Args:
        phi (float): off-diagonal phase angle.
        alpha (float): diagonal phase angle.
        theta (float): rotation angle.

    Returns:
        NDArray: Barenco gate with angles phi, alpha, theta.
    """
    u00 = np.exp(1j*alpha)*np.cos(theta)
    u01 = -1j*np.exp(1j*(alpha-phi))*np.sin(theta)
    u10 = -1j*np.exp(1j*(alpha+phi))*np.sin(theta)
    u11 = np.exp(1j*alpha)*np.cos(theta)
    return np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., u00, u01],
        [0., 0., u10, u11]
    ])


def cphase(theta: float = 1.) -> NDArray:
    """Controlled-Phase gate defintion.

    Args:
        theta (float): phase factor. Defaults to 1.

    Returns:
        NDArray: ZZ gate with a rotation angle of pi*theta.
    """
    return np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., np.exp(1j*np.pi*theta)]
    ])


def crot(
    theta: float,
    n:     ArrayLike,
) -> NDArray:
    """Controlled-Rotation gate defintion.

    Args:
        theta (float): rotation angle.
        n (ArrayLike): unit vector defining the axis of rotation.

    Returns:
        NDArray: Crot gate with a rotation angle of theta about the axis
            defined by n = (nx, ny, nz).
    """
    nx, ny, nz = n
    if nx**2 + ny**2 + nz**2 != 1.:
        raise ValueError('n must be a unit vector!')
    return scipy.linalg.expm(-1j*theta/2 * np.kron(id - z, nx*x + ny*y + nz*z))


def fsim(theta: float, phi: float) -> NDArray:
    """FSim gate defintion.

    Args:
        theta (float): rotation angle parameter.
        phi (float):   ZZ phase parameter.

    Returns:
        NDArray: FSim with rotation angle theta and phase phi.
    """
    return np.array([
        [1., 0., 0., 0.],
        [0., np.cos(theta), -1.j*np.sin(theta), 0.],
        [0., -1.j*np.sin(theta), np.cos(theta), 0.],
        [0., 0., 0., np.exp(-1.j*phi)]
    ])


def givens(theta: float) -> NDArray:
    """Givens gate defintion.

    Args:
        theta (float): rotation angle.

    Returns:
        NDArray: Givens gate with a rotation angle of theta.
    """
    return np.array([
        [1., 0., 0., 0],
        [0., np.cos(theta), -np.sin(theta), 0.],
        [0., np.sin(theta), np.cos(theta), 0.],
        [0, 0., 0., 1.]]
    )


def pswap(theta: float) -> NDArray:
    """pSWAP gate defintion.

    Args:
        theta (float): rotation angle.

    Returns:
        NDArray: pSWAP gate with a rotation angle of theta.
    """
    return np.array([
        [1., 0., 0., 0],
        [0., 0., np.exp(1j*theta), 0.],
        [0., np.exp(1j*theta), 0., 0.],
        [0, 0., 0., 1.]
    ])


def swap_alpha(alpha: float) -> NDArray:
    """SWAP-alpha gate defintion.

    Args:
        alpha (float): power of the SWAP gate.

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


def xx(t: float = 1.) -> NDArray:
    """XX gate defintion.

    Args:
        t (float): phase factor. Defaults to 1.

    Returns:
        NDArray: XX gate with a phase factor t.
    """
    return np.array([
        [np.cos(np.pi*t/2), 0., 0., -1j*np.sin(np.pi*t/2)],
        [0., np.cos(np.pi*t/2), -1j*np.sin(np.pi*t/2), 0.],
        [0., -1j*np.sin(np.pi*t/2), np.cos(np.pi*t/2), 0.],
        [-1j*np.sin(np.pi*t/2), 0., 0., np.cos(np.pi*t/2)]
    ])


def xy(t: float = 1.) -> NDArray:
    """XY gate defintion.

    Args:
        t (float): phase factor. Defaults to 1.

    Returns:
        NDArray: XY gate with a phase factor t.
    """
    return np.array([
        [1, 0., 0., 0],
        [0., np.cos(np.pi*t), -1j*np.sin(np.pi*t), 0.],
        [0., -1j*np.sin(np.pi*t), np.cos(np.pi*t), 0.],
        [0, 0., 0., 1]]
    )


def yy(t: float = 1.) -> NDArray:
    """YY gate defintion.

    Args:
        t (float): phase factor. Defaults to 1.

    Returns:
        NDArray: YY gate with a phase factor t.
    """
    return np.array([
        [np.cos(np.pi*t/2), 0., 0., 1j*np.sin(np.pi*t/2)],
        [0., np.cos(np.pi*t/2), -1j*np.sin(np.pi*t/2), 0.],
        [0., -1j*np.sin(np.pi*t/2), np.cos(np.pi*t/2), 0.],
        [1j*np.sin(np.pi*t/2), 0., 0., np.cos(np.pi*t/2)]
    ])


def zz(t: float = 1.) -> NDArray:
    """ZZ (Ising) gate defintion.

    Args:
        t (float): phase factor. Defaults to 1.

    Returns:
        NDArray: ZZ gate with a phase factor t.
    """
    return np.array([
        [np.exp(-1j*np.pi*t/2), 0., 0., 0.],
        [0., np.exp(1j*np.pi*t/2), 0., 0.],
        [0., 0., np.exp(1j*np.pi*t/2), 0.],
        [0., 0., 0., np.exp(-1j*np.pi*t/2)]
    ])


def _normalize_qubits(qubits: tuple) -> Tuple[int, int]:
    """Normalize variadic qubits args to a (q0, q1) tuple.

    Handles both GATE((q0, q1)) and GATE(q0, q1) call forms.
    """
    return qubits[0] if len(qubits) == 1 else qubits


def nearest_kronecker_product(C: NDArray) -> Tuple[NDArray]:
    """Finds the closest Kronecker product to C in the Frobenius norm.

    Args:
        C (NDArray): matrix to decompose into the Kronecker product of two
            other matrices.

    Returns:
        tuple[NDArray]: tuple of matrices whose Kronecker product form C.
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

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        theta:   float,
        phi:     float,
        **kwargs,
    ) -> None:
        """Initialize using the a_gate gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            theta (float): first gate parameter.
            phi (float): second gate parameter.
        """
        super().__init__(a_gate(theta, phi), _normalize_qubits(qubits))

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

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        phi:     float,
        alpha:   float,
        theta:   float,
        **kwargs,
    ) -> None:
        """Initialize using the barenco gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            phi (float): off-diagonal phase angle.
            alpha (float): diagonal phase angle.
            theta (float): rotation angle.
        """
        super().__init__(barenco(phi, alpha, theta), _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'XX'
        self._properties['name'] = 'Barenco'
        self._properties['params'] = {
            'theta': theta,
            'alpha': alpha,
            'phi':   phi
        }


class BGate(Gate):
    """Class for the Berkeley (B) gate."""

    def __init__(self, *qubits: int | Tuple[int, int], **kwargs) -> None:
        """Initialize using the berkeley gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(b_gate, _normalize_qubits(qubits))
        self._properties['alias'] = 'Berkeley'
        self._properties['name'] = 'BGate'


class bSWAP(Gate):
    """Class for the bSWAP gate."""

    def __init__(self, *qubits: int | Tuple[int, int], **kwargs) -> None:
        """Initialize using the bSWAP gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(bswap, _normalize_qubits(qubits))
        self._properties['name'] = 'bSWAP'


class CH(Gate):
    """Class for the Controlled-Hadamard gate."""

    def __init__(self, *qubits: int | Tuple[int, int], **kwargs) -> None:
        """Initialize using the ch gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(ch, _normalize_qubits(qubits))
        self._properties['alias'] = 'Controlled-H'
        self._properties['locally_equivalent'] = 'CX'
        self._properties['name'] = 'CH'


class CNOT(Gate):
    """Class for the Controlled-Not gate."""

    def __init__(self, *qubits: int | Tuple[int, int], **kwargs) -> None:
        """Initialize using the cnot gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(cnot, _normalize_qubits(qubits))
        self._properties['alias'] = 'CX'
        self._properties['locally_equivalent'] = 'CY, CZ'
        self._properties['name'] = 'CNOT'


class CPhase(Gate):
    """Class for the Controlled-Phase gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        theta:   float = 1.,
        **kwargs,
    ) -> None:
        """Initialize using the cphase parametrized gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            theta (float): phase factor. Defaults to 1.
        """
        super().__init__(cphase(theta), _normalize_qubits(qubits))

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

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        theta:   float,
        n:       Union[List, NDArray],
        **kwargs,
    ) -> None:
        """Initialize using the crot gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            theta (float): rotation angle.
            n (List, NDarray):   unit vector defining the axis of rotation.
        """
        super().__init__(crot(theta, n), _normalize_qubits(qubits))
        self._properties['name'] = 'CRot'
        self._properties['params'] = {
            'angle': theta,
            'axis':  n
        }


class CS(Gate):
    """Class for the Controlled-S gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the cphase parametrized gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(cphase(0.5), _normalize_qubits(qubits))
        self._properties['alias'] = 'sqrt(CZ)'
        self._properties['name'] = 'CS'
        self._properties['params'] = {
            'angle': np.pi/2,
        }


class CSdag(Gate):
    """Class for the Controlled-S^dagger gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the cphase parametrized gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(cphase(-0.5), _normalize_qubits(qubits))
        self._properties['alias'] = 'sqrt(CZ)dag'
        self._properties['name'] = 'CSDdag'
        self._properties['params'] = {
            'angle': -np.pi/2,
        }


class CT(Gate):
    """Class for the Controlled-T gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the cphase parametrized gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(cphase(0.25), _normalize_qubits(qubits))
        self._properties['alias'] = 'CZ^(1/4)'
        self._properties['name'] = 'CT'
        self._properties['params'] = {
            'angle': np.pi/4,
        }


class CTdag(Gate):
    """Class for the Controlled-T^dagger gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the cphase parametrized gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(cphase(-0.25), _normalize_qubits(qubits))
        self._properties['alias'] = 'CZ^(-1/4)'
        self._properties['name'] = 'CTdag'
        self._properties['params'] = {
            'angle': -np.pi/4,
        }


class CV(Gate):
    """Class for the Controlled-V (Square-root CNOT) gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the cv gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(cv, _normalize_qubits(qubits))
        self._properties['alias'] = 'SqrtCNOT'
        self._properties['locally_equivalent'] = 'CVdag'
        self._properties['name'] = 'CV'
        self._properties['params'] = {
            'angle': np.pi/2
        }


class CX(Gate):
    """Class for the Controlled-X gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the cx gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(cx, _normalize_qubits(qubits))
        self._properties['alias'] = 'CNOT'
        self._properties['locally_equivalent'] = 'CY, CZ'
        self._properties['name'] = 'CX'
        self._properties['params'] = {
            'angle': np.pi
        }


class CY(Gate):
    """Class for the Controlled-Y gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the cy gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(cy, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'CX, CZ'
        self._properties['name'] = 'CY'
        self._properties['params'] = {
            'angle': np.pi
        }


class CZ(Gate):
    """Class for the Controlled-Z gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the cz gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(cz, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'CX, CY'
        self._properties['name'] = 'CZ'
        self._properties['params'] = {
            'angle': np.pi
        }


class DB(Gate):
    """Class for the Dagwood-Bumstead gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the db gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(db, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'XY'
        self._properties['name'] = 'DB'


class DCNOT(Gate):
    """Class for the Double Controlled-NOT gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the dcnot gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(dcnot, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'fSWAP, iSWAP'
        self._properties['name'] = 'DCNOT'


class ECP(Gate):
    """Class for the ECP gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the ecp gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(ecp, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'WGate'
        self._properties['name'] = 'ECP'


class FSim(Gate):
    """Class for the FSim gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        theta:   float,
        phi:     float,
        **kwargs,
    ) -> None:
        """Initialize using the fsim gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            theta (float): gate rotation angle.
            phi (float): ZZ phase parameter.
        """
        super().__init__(fsim(theta, phi), _normalize_qubits(qubits))
        self._properties['name'] = 'FSim'
        self._properties['params'] = {
            'theta': theta,
            'phi':   phi,
        }


class fSWAP(Gate):
    """Class for the fSWAP gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the fswap gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(fswap, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'iSWAP, DCNOT'
        self._properties['name'] = 'fSWAP'


class Givens(Gate):
    """Class for the Givens gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        theta:   float,
        **kwargs,
    ) -> None:
        """Initialize using the Givens parametrized gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            theta (float): rotation angle.
        """
        super().__init__(givens(theta), _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'XY'
        self._properties['name'] = 'Givens'
        self._properties['params'] = {
            'angle': theta
        }


class iSWAP(Gate):
    """Class for the iSWAP gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the iswap gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(iswap, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'fSWAP, DCNOT'
        self._properties['name'] = 'iSWAP'


class M(Gate):
    """Class for the Magic gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the m gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(m, _normalize_qubits(qubits))
        self._properties['alias'] = 'Magic'
        self._properties['locally_equivalent'] = 'CX'
        self._properties['name'] = 'M'


class MS(Gate):
    """Class for the Mølmer-Sørensen gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the ms gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(ms, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'CX'
        self._properties['name'] = 'MS'


class pSWAP(Gate):
    """Class for the Parametric SWAP (pSWAP) gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        theta:   float,
        **kwargs,
    ) -> None:
        """Initialize using the pswap gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            theta (float): rotation angle.
        """
        super().__init__(pswap(theta), _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'QFT2'
        self._properties['name'] = 'pSWAP'
        self._properties['params'] = {
            'angle': theta
        }


class QFT2(Gate):
    """Class for the QFT gate on 2 qubits."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the qft2 gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(qft2, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'pSWAP'
        self._properties['name'] = 'QFT2'


class SWAP(Gate):
    """Class for the SWAP gate."""

    def __init__(self, *qubits: int | Tuple[int, int], **kwargs) -> None:
        """Initialize using the swap gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(swap, _normalize_qubits(qubits))
        self._properties['name'] = 'SWAP'


class SWAPAlpha(Gate):
    """Class for the SWAP-alpha gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        alpha:   float,
        **kwargs
    ) -> None:
        """Initialize using the swap-alpha gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            alpha (float): power of the SWAP gate.
        """
        super().__init__(swap_alpha(alpha), _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'SqrtSWAP, SqrtSWAPdag'
        self._properties['name'] = 'SWAPAlpha'
        self._properties['params'] = {
            'alpha': alpha
        }


class SqrtSWAP(Gate):
    """Class for the Sqrt SWAP gate."""

    def __init__(self, *qubits: int | Tuple[int, int], **kwargs) -> None:
        """Initialize using the sqrt_swap gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(sqrt_swap, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'SWAPAlpha, SqrtSWAPdag'
        self._properties['name'] = 'SqrtSWAP'


class SqrtSWAPdag(Gate):
    """Class for the Sqrt(SWAP)^dagger gate."""

    def __init__(self, *qubits: int | Tuple[int, int], **kwargs) -> None:
        """Initialize using the sqrt_swap_dag gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(sqrt_swap_dag, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'SWAPAlpha, SqrtSWAP'
        self._properties['name'] = 'SqrtSWAPdag'


class Syc(Gate):
    """Class for the Sycamore (Syc) gate."""

    def __init__(self, *qubits: int | Tuple[int, int], **kwargs) -> None:
        """Initialize using the syc gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(syc, _normalize_qubits(qubits))
        self._properties['alias'] = 'Sycamore'
        self._properties['name'] = 'Syc'


class WGate(Gate):
    """Class for the W gate."""

    def __init__(self, *qubits: int | Tuple[int, int], **kwargs) -> None:
        """Initialize using the w gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
        """
        super().__init__(w_gate, _normalize_qubits(qubits))
        self._properties['locally_equivalent'] = 'ECP'
        self._properties['name'] = 'WGate'


class XX(Gate):
    """Class for the XX gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        t:       float = 1.,
        **kwargs,
    ) -> None:
        """Initialize using the xx parametrized gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            t (float): phase factor. Defaults to 1.
        """
        super().__init__(xx(t), _normalize_qubits(qubits))
        self._properties['name'] = 'XX'
        self._properties['params'] = {
            'phase factor': t
        }


class XY(Gate):
    """Class for the XY gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        t:       float = 1.,
        **kwargs,
    ) -> None:
        """Initialize using the xy parametrized gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            t (float): phase factor. Defaults to 1.
        """
        super().__init__(xy(t), _normalize_qubits(qubits))
        self._properties['alias'] = 'piSWAP'
        self._properties['name'] = 'XY'
        self._properties['params'] = {
            'phase factor': t
        }


class YY(Gate):
    """Class for the YY gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        t:       float = 1.,
        **kwargs,
    ) -> None:
        """Initialize using the yy parametrized gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            t (float): phase factor. Defaults to 1.
        """
        super().__init__(yy(t), _normalize_qubits(qubits))
        self._properties['name'] = 'YY'
        self._properties['params'] = {
            'phase factor': t
        }


class ZZ(Gate):
    """Class for the ZZ gate."""

    def __init__(
        self,
        *qubits: int | Tuple[int, int],
        t:       float = 1.,
        **kwargs,
    ) -> None:
        """Initialize using the zz parametrized gate.

        Args:
            *qubits (int | Tuple[int, int]): qubit labels as a tuple (q0, q1)
                or two separate ints.
            t (float): phase factor. Defaults to 1.
        """
        super().__init__(zz(t), _normalize_qubits(qubits))
        self._properties['name'] = 'ZZ'
        self._properties['params'] = {
            'phase factor': t
        }


TWO_QUBIT_GATES: Mapping[str, Callable] = defaultdict(
    lambda: 'Gate not currently supported!', {
        'AGate':       AGate,
        'Barenco':     Barenco,
        'BGate':       BGate,
        'bSWAP':       bSWAP,
        'CH':          CH,
        'CNOT':        CNOT,
        'CPhase':      CPhase,
        'CRot':        CRot,
        'CS':          CS,
        'CSdag':       CSdag,
        'CT':          CT,
        'CTdag':       CTdag,
        'CV':          CV,
        'CX':          CX,
        'CY':          CY,
        'CZ':          CZ,
        'DB':          DB,
        'DCNOT':       DCNOT,
        'ECP':         ECP,
        'FSim':        FSim,
        'fSWAP':       fSWAP,
        'Givens':      Givens,
        'iSWAP':       iSWAP,
        'M':           M,
        'MS':          MS,
        'pSWAP':       pSWAP,
        'QFT2':        QFT2,
        'SqrtSWAP':    SqrtSWAP,
        'SqrtSWAPdag': SqrtSWAPdag,
        'SWAPAlpha':   SWAPAlpha,
        'SWAP':        SWAP,
        'Syc':         Syc,
        'WGate':       WGate,
        'XX':          XX,
        'XY':          XY,
        'YY':          YY,
        'ZZ':          ZZ
    }
)
