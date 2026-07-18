"""Submodule for storing two-qutrit gate definitions.

See the supplement of arXiv:2206.07216 (Goss et al.) for relevant
definitions.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from qcal.gate.gate import Gate


__all__ = (
    'CP3',
    'CSUM',
    'CSUMdag',
    'CZ3',
    'CZ3dag',
    'SWAP3',
)

# ---------------------------------------------------------------------------
# Matrix constants
# ---------------------------------------------------------------------------

# ω = e^(2πi/3), the primitive cube root of unity
_omega = np.exp(2j * np.pi / 3)

# CZ3: applies phase ω^(j·k) to |j⟩|k⟩ in basis ordering
# |00⟩, |01⟩, |02⟩, |10⟩, |11⟩, |12⟩, |20⟩, |21⟩, |22⟩
cz3 = np.diag([
    1.,      1.,        1.,
    1.,      _omega,    _omega**2,
    1.,      _omega**2, _omega,
])

cz3dag = cz3.conj()

# CSUM: |j⟩|k⟩ → |j⟩|k+j mod 3⟩  (qutrit CNOT analog)
csum = np.array([
    [1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0.],
])

# CSUMdag: |j⟩|k⟩ → |j⟩|k-j mod 3⟩  (inverse of CSUM)
csumd = np.array([
    [1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0.],
])

# SWAP3: |j⟩|k⟩ → |k⟩|j⟩
swap3 = np.array([
    [1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1.],
])


# ---------------------------------------------------------------------------
# Parametric gate functions
# ---------------------------------------------------------------------------

def cp3(theta: float) -> NDArray:
    """Generalized controlled-phase gate for qutrits.

    Applies phase e^(i·j·k·theta) to computational basis state
    |j⟩|k⟩. When theta = 2π/3 this reduces to CZ3.

    Args:
        theta (float): phase angle per unit of j·k.

    Returns:
        NDArray: 9x9 diagonal unitary matrix.
    """
    diag = np.array([
        np.exp(1.j * j * k * theta)
        for j in range(3) for k in range(3)
    ])
    return np.diag(diag)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _normalize_qudits(qudits: tuple) -> tuple:
    """Normalize variadic qudit args to a (q0, q1) tuple.

    Handles both GATE((q0, q1)) and GATE(q0, q1) call forms.
    """
    return qudits[0] if len(qudits) == 1 else qudits


# ---------------------------------------------------------------------------
# Gate classes
# ---------------------------------------------------------------------------

class CP3(Gate):
    """Class for the generalized two-qutrit controlled-phase gate.

    Applies phase e^(i·j·k·theta) to |j⟩|k⟩. At theta = 2π/3 this
    gate is equivalent to CZ3.
    """

    def __init__(
        self,
        *qudits: int | Tuple[int, int],
        theta:   float,
        **kwargs,
    ) -> None:
        """Initialize using the cp3 function.

        Args:
            *qudits (int | Tuple[int, int]): qudit labels as a tuple
                (q0, q1) or two separate ints.
            theta (float): phase angle per unit of j·k.
        """
        super().__init__(cp3(theta), _normalize_qudits(qudits))
        self._properties['name'] = 'CP3'
        self._properties['params'] = {'angle': theta}
        self._properties['subspace'] = 'GEF'


class CSUM(Gate):
    """Class for the two-qutrit CSUM (controlled-SUM) gate.

    The qutrit analog of the CNOT gate. Maps |j⟩|k⟩ → |j⟩|k+j mod 3⟩,
    entangling the two qutrits via modular addition.
    """

    def __init__(
        self,
        *qudits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the csum matrix.

        Args:
            *qudits (int | Tuple[int, int]): qudit labels as a tuple
                (q0, q1) or two separate ints.
        """
        super().__init__(csum, _normalize_qudits(qudits))
        self._properties['name'] = 'CSUM'
        self._properties['subspace'] = 'GEF'


class CSUMdag(Gate):
    """Class for the two-qutrit CSUM† (inverse controlled-SUM) gate.

    Maps |j⟩|k⟩ → |j⟩|k-j mod 3⟩.
    """

    def __init__(
        self,
        *qudits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the csumd matrix.

        Args:
            *qudits (int | Tuple[int, int]): qudit labels as a tuple
                (q0, q1) or two separate ints.
        """
        super().__init__(csumd, _normalize_qudits(qudits))
        self._properties['name'] = 'CSUMdag'
        self._properties['subspace'] = 'GEF'


class CZ3(Gate):
    """Class for the two-qutrit CZ3 gate.

    Applies phase ω^(j·k) to the two-qutrit computational basis state
    |j⟩|k⟩, where ω = exp(2πi/3). This generalizes the qubit CZ gate
    to the qutrit setting.

    Reference: arXiv:2206.07216 (Goss et al.)
    """

    def __init__(
        self,
        *qudits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the cz3 matrix.

        Args:
            *qudits (int | Tuple[int, int]): qudit labels as a tuple
                (q0, q1) or two separate ints.
        """
        super().__init__(cz3, _normalize_qudits(qudits))
        self._properties['name'] = 'CZ3'
        self._properties['subspace'] = 'GEF'


class CZ3dag(Gate):
    """Class for the two-qutrit CZ3† (dagger) gate.

    The Hermitian conjugate of CZ3; applies phase ω^(-j·k) = ω^(2j·k)
    to |j⟩|k⟩, where ω = exp(2πi/3).

    Reference: arXiv:2206.07216 (Goss et al.)
    """

    def __init__(
        self,
        *qudits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the cz3dag matrix.

        Args:
            *qudits (int | Tuple[int, int]): qudit labels as a tuple
                (q0, q1) or two separate ints.
        """
        super().__init__(cz3dag, _normalize_qudits(qudits))
        self._properties['name'] = 'CZ3dag'
        self._properties['subspace'] = 'GEF'


class SWAP3(Gate):
    """Class for the two-qutrit SWAP gate.

    Maps |j⟩|k⟩ → |k⟩|j⟩, exchanging the states of the two qutrits.
    """

    def __init__(
        self,
        *qudits: int | Tuple[int, int],
        **kwargs,
    ) -> None:
        """Initialize using the swap3 matrix.

        Args:
            *qudits (int | Tuple[int, int]): qudit labels as a tuple
                (q0, q1) or two separate ints.
        """
        super().__init__(swap3, _normalize_qudits(qudits))
        self._properties['name'] = 'SWAP3'
        self._properties['subspace'] = 'GEF'


TWO_QUTRIT_GATES: Mapping[str, Callable] = defaultdict(
    lambda: 'Gate not currently supported!', {
        'CP3':     CP3,
        'CSUM':    CSUM,
        'CSUMdag': CSUMdag,
        'CZ3':     CZ3,
        'CZ3dag':  CZ3dag,
        'SWAP3':   SWAP3,
    }
)
