"""Submodule for storing single-qutrit gate definitions.

See the supplement of arXiv:2206.07216 (Goss et al.) for relevant
definitions.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from qcal.gates.gate import Gate


__all__ = (
    'H3',
    'Id3',
    'RGM1',
    'RGM2',
    'RGM3',
    'RGM4',
    'RGM5',
    'RGM6',
    'RGM7',
    'RGM8',
    'Rx01',
    'Rx12',
    'Ry01',
    'Ry12',
    'Rz01',
    'Rz12',
    'VirtualZ01',
    'VirtualZ12',
    'WeylX',
    'WeylX2',
    'WeylX2Z',
    'WeylX2Z2',
    'WeylXZ',
    'WeylXZ2',
    'WeylZ',
    'WeylZ2',
    'X01',
    'X02',
    'X12',
    'X9001',
    'X9012',
    'Y01',
    'Y02',
    'Y12',
    'Y9001',
    'Y9012',
    'Z01',
    'Z12',
)

# ---------------------------------------------------------------------------
# Matrix constants
# ---------------------------------------------------------------------------

id3 = np.eye(3, dtype=complex)

# Pauli-X type: π rotation in each two-level subspace
x01 = np.array(
    [[0., 1., 0.],
     [1., 0., 0.],
     [0., 0., 1.]]
)

x12 = np.array(
    [[1., 0., 0.],
     [0., 0., 1.],
     [0., 1., 0.]]
)

x02 = np.array(
    [[0., 0., 1.],
     [0., 1., 0.],
     [1., 0., 0.]]
)

# Pauli-Y type: π rotation in each two-level subspace
y01 = np.array(
    [[ 0., -1.j,  0.],
     [ 1.j,  0.,  0.],
     [ 0.,   0.,  1.]]
)

y12 = np.array(
    [[1.,  0.,   0.],
     [0.,  0.,  -1.j],
     [0.,  1.j,  0.]]
)

y02 = np.array(
    [[ 0.,  0., -1.j],
     [ 0.,  1.,  0.],
     [ 1.j, 0.,  0.]]
)

# Pauli-Z type: phase flip in each two-level subspace
z01 = np.array(
    [[1.,  0., 0.],
     [0., -1., 0.],
     [0.,  0., 1.]]
)

z12 = np.array(
    [[1., 0.,  0.],
     [0., 1.,  0.],
     [0., 0., -1.]]
)

# Qutrit Hadamard / 3-level QFT
_omega = np.exp(2j * np.pi / 3)
h3 = (1 / np.sqrt(3)) * np.array(
    [[1.,  1.,        1.      ],
     [1.,  _omega,    _omega**2],
     [1.,  _omega**2, _omega  ]]
)

# ---------------------------------------------------------------------------
# Gell-Mann matrices  (Hermitian, traceless; SU(3) generators)
# λ1-λ3 act in the GE (|0⟩-|1⟩) subspace
# λ4-λ5 act in the GF (|0⟩-|2⟩) subspace
# λ6-λ7 act in the EF (|1⟩-|2⟩) subspace
# λ8     is diagonal across all three levels
# ---------------------------------------------------------------------------
lam1 = np.array(
    [[0., 1., 0.],
     [1., 0., 0.],
     [0., 0., 0.]]
)

lam2 = np.array(
    [[ 0., -1.j, 0.],
     [ 1.j,  0., 0.],
     [ 0.,   0., 0.]]
)

lam3 = np.array(
    [[1.,  0., 0.],
     [0., -1., 0.],
     [0.,  0., 0.]]
)

lam4 = np.array(
    [[0., 0., 1.],
     [0., 0., 0.],
     [1., 0., 0.]]
)

lam5 = np.array(
    [[ 0., 0., -1.j],
     [ 0., 0.,  0. ],
     [ 1.j, 0., 0. ]]
)

lam6 = np.array(
    [[0., 0., 0.],
     [0., 0., 1.],
     [0., 1., 0.]]
)

lam7 = np.array(
    [[0.,  0.,   0. ],
     [0.,  0.,  -1.j],
     [0.,  1.j,  0. ]]
)

lam8 = (1. / np.sqrt(3)) * np.array(
    [[1., 0.,  0.],
     [0., 1.,  0.],
     [0., 0., -2.]]
)

# ---------------------------------------------------------------------------
# Weyl (clock-shift) operators  (unitary; qutrit Pauli group basis)
# W_{ab} = X^a Z^b  where X is the shift and Z is the clock operator.
# Basis ordering: |00⟩=|0⟩, |01⟩=|1⟩, |02⟩=|2⟩ (single-qutrit).
# ---------------------------------------------------------------------------
# X: shift  |j⟩ → |(j+1) mod 3⟩
weyl_x = np.array(
    [[0., 0., 1.],
     [1., 0., 0.],
     [0., 1., 0.]]
)

# Z: clock  |j⟩ → ω^j |j⟩
weyl_z = np.diag([1., _omega, _omega**2])

# Higher powers and mixed products
weyl_x2   = weyl_x @ weyl_x
weyl_z2   = weyl_z @ weyl_z
weyl_xz   = weyl_x @ weyl_z
weyl_xz2  = weyl_x @ weyl_z2
weyl_x2z  = weyl_x2 @ weyl_z
weyl_x2z2 = weyl_x2 @ weyl_z2


# ---------------------------------------------------------------------------
# Parametric rotation functions
# ---------------------------------------------------------------------------

_GELL_MANN = [lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8]


def _expm_herm(H: NDArray, theta: float) -> NDArray:
    """Compute exp(-i·theta/2·H) for a Hermitian matrix H.

    Uses eigendecomposition: exp(-i t H) = U diag(exp(-i t λ)) U†.
    """
    vals, vecs = np.linalg.eigh(H)
    return (vecs * np.exp(-0.5j * theta * vals)) @ vecs.conj().T


def rgm(k: int, theta: float) -> NDArray:
    """Rotation about Gell-Mann axis λ_k by angle theta.

    Computes exp(-i·theta/2·λ_k) for k ∈ {1, …, 8}.

    Args:
        k (int):     Gell-Mann index (1-indexed).
        theta (float): rotation angle in radians.

    Returns:
        NDArray: 3x3 unitary matrix.
    """
    return _expm_herm(_GELL_MANN[k - 1], theta)


def rx01(theta: float) -> NDArray:
    """Rx rotation in the GE (|0⟩-|1⟩) subspace.

    Args:
        theta (float): rotation angle.

    Returns:
        NDArray: 3x3 unitary matrix.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[c,      -1.j*s, 0.],
         [-1.j*s,  c,     0.],
         [0.,      0.,    1.]]
    )


def rx12(theta: float) -> NDArray:
    """Rx rotation in the EF (|1⟩-|2⟩) subspace.

    Args:
        theta (float): rotation angle.

    Returns:
        NDArray: 3x3 unitary matrix.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[1.,  0.,      0.    ],
         [0.,  c,      -1.j*s],
         [0., -1.j*s,   c    ]]
    )


def ry01(theta: float) -> NDArray:
    """Ry rotation in the GE (|0⟩-|1⟩) subspace.

    Args:
        theta (float): rotation angle.

    Returns:
        NDArray: 3x3 unitary matrix.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[ c, -s, 0.],
         [ s,  c, 0.],
         [0., 0., 1.]]
    )


def ry12(theta: float) -> NDArray:
    """Ry rotation in the EF (|1⟩-|2⟩) subspace.

    Args:
        theta (float): rotation angle.

    Returns:
        NDArray: 3x3 unitary matrix.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[1., 0.,  0.],
         [0.,  c,  -s],
         [0.,  s,   c]]
    )


def rz01(theta: float) -> NDArray:
    """Rz rotation in the GE (|0⟩-|1⟩) subspace.

    Args:
        theta (float): rotation angle.

    Returns:
        NDArray: 3x3 unitary matrix.
    """
    p, m = np.exp(-1.j * theta / 2), np.exp(1.j * theta / 2)
    return np.array(
        [[p,  0., 0.],
         [0., m,  0.],
         [0., 0., 1.]]
    )


def rz12(theta: float) -> NDArray:
    """Rz rotation in the EF (|1⟩-|2⟩) subspace.

    Args:
        theta (float): rotation angle.

    Returns:
        NDArray: 3x3 unitary matrix.
    """
    p, m = np.exp(-1.j * theta / 2), np.exp(1.j * theta / 2)
    return np.array(
        [[1., 0., 0.],
         [0., p,  0.],
         [0., 0., m ]]
    )


# ---------------------------------------------------------------------------
# Gate classes
# ---------------------------------------------------------------------------

class H3(Gate):
    """Class for the qutrit Hadamard (3-level QFT) gate."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the h3 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(h3, qudit)
        self._properties['alias'] = 'QFT3'
        self._properties['name'] = 'H3'
        self._properties['subspace'] = 'GEF'


class Id3(Gate):
    """Class for the single-qutrit identity gate."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the 3x3 identity matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(id3, qudit)
        self._properties['name'] = 'I3'
        self._properties['params'] = {'angle': 0, 'duration': 0.}
        self._properties['subspace'] = 'GEF'


class Rx01(Gate):
    """Class for parametric Rx rotation in the GE subspace."""

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using the rx01 function.

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle.
        """
        super().__init__(rx01(theta), qudit)
        self._properties['name'] = 'Rx01'
        self._properties['params'] = {'angle': theta, 'axis': 'x'}
        self._properties['subspace'] = 'GE'


class Rx12(Gate):
    """Class for parametric Rx rotation in the EF subspace."""

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using the rx12 function.

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle.
        """
        super().__init__(rx12(theta), qudit)
        self._properties['name'] = 'Rx12'
        self._properties['params'] = {'angle': theta, 'axis': 'x'}
        self._properties['subspace'] = 'EF'


class Ry01(Gate):
    """Class for parametric Ry rotation in the GE subspace."""

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using the ry01 function.

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle.
        """
        super().__init__(ry01(theta), qudit)
        self._properties['name'] = 'Ry01'
        self._properties['params'] = {'angle': theta, 'axis': 'y'}
        self._properties['subspace'] = 'GE'


class Ry12(Gate):
    """Class for parametric Ry rotation in the EF subspace."""

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using the ry12 function.

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle.
        """
        super().__init__(ry12(theta), qudit)
        self._properties['name'] = 'Ry12'
        self._properties['params'] = {'angle': theta, 'axis': 'y'}
        self._properties['subspace'] = 'EF'


class Rz01(Gate):
    """Class for parametric Rz rotation in the GE subspace."""

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using the rz01 function.

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle.
        """
        super().__init__(rz01(theta), qudit)
        self._properties['name'] = 'Rz01'
        self._properties['params'] = {'phase': theta, 'axis': 'z'}
        self._properties['subspace'] = 'GE'


class Rz12(Gate):
    """Class for parametric Rz rotation in the EF subspace."""

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using the rz12 function.

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle.
        """
        super().__init__(rz12(theta), qudit)
        self._properties['name'] = 'Rz12'
        self._properties['params'] = {'phase': theta, 'axis': 'z'}
        self._properties['subspace'] = 'EF'


class VirtualZ01(Gate):
    """Class for the virtual Z gate in the GE subspace."""

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using the rz01 function.

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle.
        """
        super().__init__(rz01(theta), qudit)
        self._properties['name'] = 'VirtualZ01'
        self._properties['params'] = {'phase': theta, 'axis': 'z'}
        self._properties['subspace'] = 'GE'


class VirtualZ12(Gate):
    """Class for the virtual Z gate in the EF subspace."""

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using the rz12 function.

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle.
        """
        super().__init__(rz12(theta), qudit)
        self._properties['name'] = 'VirtualZ12'
        self._properties['params'] = {'phase': theta, 'axis': 'z'}
        self._properties['subspace'] = 'EF'


class X01(Gate):
    """Class for the Pauli-X gate in the GE (|0⟩-|1⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the x01 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(x01, qudit)
        self._properties['name'] = 'X01'
        self._properties['params'] = {'angle': np.pi, 'axis': 'x'}
        self._properties['subspace'] = 'GE'


class X02(Gate):
    """Class for the Pauli-X gate in the GF (|0⟩-|2⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the x02 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(x02, qudit)
        self._properties['name'] = 'X02'
        self._properties['params'] = {'angle': np.pi, 'axis': 'x'}
        self._properties['subspace'] = 'GF'


class X12(Gate):
    """Class for the Pauli-X gate in the EF (|1⟩-|2⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the x12 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(x12, qudit)
        self._properties['name'] = 'X12'
        self._properties['params'] = {'angle': np.pi, 'axis': 'x'}
        self._properties['subspace'] = 'EF'


class X90_01(Gate):
    """Class for the sqrt(X) gate in the GE (|0⟩-|1⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the rx01 function at π/2.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(rx01(np.pi / 2), qudit)
        self._properties['alias'] = 'SX01'
        self._properties['name'] = 'X90_01'
        self._properties['params'] = {'angle': np.pi / 2, 'axis': 'x'}
        self._properties['subspace'] = 'GE'


class X90_12(Gate):
    """Class for the sqrt(X) gate in the EF (|1⟩-|2⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the rx12 function at π/2.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(rx12(np.pi / 2), qudit)
        self._properties['alias'] = 'SX12'
        self._properties['name'] = 'X90_12'
        self._properties['params'] = {'angle': np.pi / 2, 'axis': 'x'}
        self._properties['subspace'] = 'EF'


class Y01(Gate):
    """Class for the Pauli-Y gate in the GE (|0⟩-|1⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the y01 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(y01, qudit)
        self._properties['name'] = 'Y01'
        self._properties['params'] = {'angle': np.pi, 'axis': 'y'}
        self._properties['subspace'] = 'GE'


class Y02(Gate):
    """Class for the Pauli-Y gate in the GF (|0⟩-|2⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the y02 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(y02, qudit)
        self._properties['name'] = 'Y02'
        self._properties['params'] = {'angle': np.pi, 'axis': 'y'}
        self._properties['subspace'] = 'GF'


class Y12(Gate):
    """Class for the Pauli-Y gate in the EF (|1⟩-|2⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the y12 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(y12, qudit)
        self._properties['name'] = 'Y12'
        self._properties['params'] = {'angle': np.pi, 'axis': 'y'}
        self._properties['subspace'] = 'EF'


class Y90_01(Gate):
    """Class for the sqrt(Y) gate in the GE (|0⟩-|1⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the ry01 function at π/2.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(ry01(np.pi / 2), qudit)
        self._properties['alias'] = 'SY01'
        self._properties['name'] = 'Y90_01'
        self._properties['params'] = {'angle': np.pi / 2, 'axis': 'y'}
        self._properties['subspace'] = 'GE'


class Y90_12(Gate):
    """Class for the sqrt(Y) gate in the EF (|1⟩-|2⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the ry12 function at π/2.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(ry12(np.pi / 2), qudit)
        self._properties['alias'] = 'SY12'
        self._properties['name'] = 'Y90_12'
        self._properties['params'] = {'angle': np.pi / 2, 'axis': 'y'}
        self._properties['subspace'] = 'EF'


class Z01(Gate):
    """Class for the Pauli-Z gate in the GE (|0⟩-|1⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the z01 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(z01, qudit)
        self._properties['name'] = 'Z01'
        self._properties['params'] = {'phase': np.pi, 'axis': 'z'}
        self._properties['subspace'] = 'GE'


class Z12(Gate):
    """Class for the Pauli-Z gate in the EF (|1⟩-|2⟩) subspace."""

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the z12 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(z12, qudit)
        self._properties['name'] = 'Z12'
        self._properties['params'] = {'phase': np.pi, 'axis': 'z'}
        self._properties['subspace'] = 'EF'


class WeylX(Gate):
    """Qutrit shift (increment) operator X: |j⟩ → |(j+1) mod 3⟩.

    Corresponds to the Weyl operator W_{10}.
    """

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the weyl_x matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(weyl_x, qudit)
        self._properties['alias'] = 'W10'
        self._properties['name'] = 'WeylX'
        self._properties['subspace'] = 'GEF'


class WeylX2(Gate):
    """Qutrit double-shift operator X²: |j⟩ → |(j+2) mod 3⟩.

    Corresponds to the Weyl operator W_{20}.
    """

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the weyl_x2 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(weyl_x2, qudit)
        self._properties['alias'] = 'W20'
        self._properties['name'] = 'WeylX2'
        self._properties['subspace'] = 'GEF'


class WeylZ(Gate):
    """Qutrit clock operator Z: |j⟩ → ω^j |j⟩, ω = exp(2πi/3).

    Corresponds to the Weyl operator W_{01}.
    """

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the weyl_z matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(weyl_z, qudit)
        self._properties['alias'] = 'W01'
        self._properties['name'] = 'WeylZ'
        self._properties['subspace'] = 'GEF'


class WeylZ2(Gate):
    """Qutrit double-clock operator Z²: |j⟩ → ω^(2j) |j⟩.

    Corresponds to the Weyl operator W_{02}.
    """

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the weyl_z2 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(weyl_z2, qudit)
        self._properties['alias'] = 'W02'
        self._properties['name'] = 'WeylZ2'
        self._properties['subspace'] = 'GEF'


class WeylXZ(Gate):
    """Weyl operator W_{11} = X·Z.

    Applies clock then shift: |j⟩ → ω^j |(j+1) mod 3⟩.
    """

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the weyl_xz matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(weyl_xz, qudit)
        self._properties['alias'] = 'W11'
        self._properties['name'] = 'WeylXZ'
        self._properties['subspace'] = 'GEF'


class WeylXZ2(Gate):
    """Weyl operator W_{12} = X·Z².

    Applies double-clock then shift: |j⟩ → ω^(2j) |(j+1) mod 3⟩.
    """

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the weyl_xz2 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(weyl_xz2, qudit)
        self._properties['alias'] = 'W12'
        self._properties['name'] = 'WeylXZ2'
        self._properties['subspace'] = 'GEF'


class WeylX2Z(Gate):
    """Weyl operator W_{21} = X²·Z.

    Applies clock then double-shift: |j⟩ → ω^j |(j+2) mod 3⟩.
    """

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the weyl_x2z matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(weyl_x2z, qudit)
        self._properties['alias'] = 'W21'
        self._properties['name'] = 'WeylX2Z'
        self._properties['subspace'] = 'GEF'


class WeylX2Z2(Gate):
    """Weyl operator W_{22} = X²·Z².

    Applies double-clock then double-shift:
    |j⟩ → ω^(2j) |(j+2) mod 3⟩.
    """

    def __init__(self, qudit: int, **kwargs) -> None:
        """Initialize using the weyl_x2z2 matrix.

        Args:
            qudit (int): qudit label.
        """
        super().__init__(weyl_x2z2, qudit)
        self._properties['alias'] = 'W22'
        self._properties['name'] = 'WeylX2Z2'
        self._properties['subspace'] = 'GEF'


class RGM1(Gate):
    """Rotation about Gell-Mann axis λ₁: exp(-i·theta/2·λ₁).

    λ₁ acts in the GE (|0⟩-|1⟩) subspace; it is the off-diagonal
    real part of the SU(2) generators embedded in SU(3).
    """

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using rgm(1, theta).

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle in radians.
        """
        super().__init__(rgm(1, theta), qudit)
        self._properties['name'] = 'RGM1'
        self._properties['params'] = {'angle': theta, 'axis': 'lam1'}
        self._properties['subspace'] = 'GE'


class RGM2(Gate):
    """Rotation about Gell-Mann axis λ₂: exp(-i·theta/2·λ₂).

    λ₂ acts in the GE (|0⟩-|1⟩) subspace; it is the off-diagonal
    imaginary part (Y-type generator).
    """

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using rgm(2, theta).

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle in radians.
        """
        super().__init__(rgm(2, theta), qudit)
        self._properties['name'] = 'RGM2'
        self._properties['params'] = {'angle': theta, 'axis': 'lam2'}
        self._properties['subspace'] = 'GE'


class RGM3(Gate):
    """Rotation about Gell-Mann axis λ₃: exp(-i·theta/2·λ₃).

    λ₃ is diagonal in the GE (|0⟩-|1⟩) subspace (Z-type generator).
    """

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using rgm(3, theta).

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle in radians.
        """
        super().__init__(rgm(3, theta), qudit)
        self._properties['name'] = 'RGM3'
        self._properties['params'] = {'angle': theta, 'axis': 'lam3'}
        self._properties['subspace'] = 'GE'


class RGM4(Gate):
    """Rotation about Gell-Mann axis λ₄: exp(-i·theta/2·λ₄).

    λ₄ acts in the GF (|0⟩-|2⟩) subspace (X-type generator).
    """

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using rgm(4, theta).

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle in radians.
        """
        super().__init__(rgm(4, theta), qudit)
        self._properties['name'] = 'RGM4'
        self._properties['params'] = {'angle': theta, 'axis': 'lam4'}
        self._properties['subspace'] = 'GF'


class RGM5(Gate):
    """Rotation about Gell-Mann axis λ₅: exp(-i·theta/2·λ₅).

    λ₅ acts in the GF (|0⟩-|2⟩) subspace (Y-type generator).
    """

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using rgm(5, theta).

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle in radians.
        """
        super().__init__(rgm(5, theta), qudit)
        self._properties['name'] = 'RGM5'
        self._properties['params'] = {'angle': theta, 'axis': 'lam5'}
        self._properties['subspace'] = 'GF'


class RGM6(Gate):
    """Rotation about Gell-Mann axis λ₆: exp(-i·theta/2·λ₆).

    λ₆ acts in the EF (|1⟩-|2⟩) subspace (X-type generator).
    """

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using rgm(6, theta).

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle in radians.
        """
        super().__init__(rgm(6, theta), qudit)
        self._properties['name'] = 'RGM6'
        self._properties['params'] = {'angle': theta, 'axis': 'lam6'}
        self._properties['subspace'] = 'EF'


class RGM7(Gate):
    """Rotation about Gell-Mann axis λ₇: exp(-i·theta/2·λ₇).

    λ₇ acts in the EF (|1⟩-|2⟩) subspace (Y-type generator).
    """

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using rgm(7, theta).

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle in radians.
        """
        super().__init__(rgm(7, theta), qudit)
        self._properties['name'] = 'RGM7'
        self._properties['params'] = {'angle': theta, 'axis': 'lam7'}
        self._properties['subspace'] = 'EF'


class RGM8(Gate):
    """Rotation about Gell-Mann axis λ₈: exp(-i·theta/2·λ₈).

    λ₈ is the diagonal hypercharge generator acting across all three
    levels: diag(1, 1, -2) / sqrt(3).
    """

    def __init__(self, qudit: int, theta: float, **kwargs) -> None:
        """Initialize using rgm(8, theta).

        Args:
            qudit (int):   qudit label.
            theta (float): rotation angle in radians.
        """
        super().__init__(rgm(8, theta), qudit)
        self._properties['name'] = 'RGM8'
        self._properties['params'] = {'angle': theta, 'axis': 'lam8'}
        self._properties['subspace'] = 'GEF'


SINGLE_QUTRIT_GATES: Mapping[str, Callable] = defaultdict(
    lambda: 'Gate not currently supported!', {
        'H3':         H3,
        'Id3':        Id3,
        'RGM1':       RGM1,
        'RGM2':       RGM2,
        'RGM3':       RGM3,
        'RGM4':       RGM4,
        'RGM5':       RGM5,
        'RGM6':       RGM6,
        'RGM7':       RGM7,
        'RGM8':       RGM8,
        'Rx01':       Rx01,
        'Rx12':       Rx12,
        'Ry01':       Ry01,
        'Ry12':       Ry12,
        'Rz01':       Rz01,
        'Rz12':       Rz12,
        'VirtualZ01': VirtualZ01,
        'VirtualZ12': VirtualZ12,
        'WeylX':      WeylX,
        'WeylX2':     WeylX2,
        'WeylX2Z':    WeylX2Z,
        'WeylX2Z2':   WeylX2Z2,
        'WeylXZ':     WeylXZ,
        'WeylXZ2':    WeylXZ2,
        'WeylZ':      WeylZ,
        'WeylZ2':     WeylZ2,
        'X01':        X01,
        'X02':        X02,
        'X12':        X12,
        'X90_01':     X90_01,
        'X90_12':     X90_12,
        'Y01':        Y01,
        'Y02':        Y02,
        'Y12':        Y12,
        'Y90_01':     Y90_01,
        'Y90_12':     Y90_12,
        'Z01':        Z01,
        'Z12':        Z12,
    }
)
