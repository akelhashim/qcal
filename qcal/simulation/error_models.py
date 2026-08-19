"""Error model wrappers for the DensityMatrixSimulator.

Each :class:`ErrorModel` maps broad gate categories
(``'single_qubit'``, ``'two_qubit'``, ``'single_qutrit'``,
``'two_qutrit'``) to ``quax`` noise channels, rather than
requiring a channel per gate class.

Gate membership is resolved through four frozensets built from the
canonical gate-name dictionaries in ``qcal.gates``:

- :data:`SINGLE_QUBIT_NAMES`  — keys of ``SINGLE_QUBIT_GATES``
- :data:`TWO_QUBIT_NAMES`     — keys of ``TWO_QUBIT_GATES``
- :data:`SINGLE_QUTRIT_NAMES` — keys of ``SINGLE_QUTRIT_GATES``
- :data:`TWO_QUTRIT_NAMES`    — keys of ``TWO_QUTRIT_GATES``

Use :func:`gate_category` to find the category for any gate class
name. The :class:`~qcal.simulation.simulators.DensityMatrixSimulator`
calls :meth:`ErrorModel.channel_for` on each gate during simulation.

Most error models accept a *dims* tuple implicitly through separate
per-category parameters: qubit categories always use ``dims=(2,)``
and qutrit categories always use ``dims=(3,)`` when calling the
underlying ``quax.channels.*`` functions.

Example::

    from qcal.simulation import DensityMatrixSimulator
    from qcal.simulation.error_models import DepolarizingNoise

    noise = DepolarizingNoise(single_qubit=0.001, two_qubit=0.01)
    sim = DensityMatrixSimulator(noise_model=noise)
    sim.run(circuit)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product as _iproduct
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import quax

from qcal.gates.gate import Gate
from qcal.gates.single_qubit import SINGLE_QUBIT_GATES
from qcal.gates.single_qutrit import SINGLE_QUTRIT_GATES
from qcal.gates.two_qubit import TWO_QUBIT_GATES
from qcal.gates.two_qutrit import TWO_QUTRIT_GATES

logger = logging.getLogger(__name__)

__all__ = (
    'SINGLE_QUBIT_NAMES',
    'TWO_QUBIT_NAMES',
    'SINGLE_QUTRIT_NAMES',
    'TWO_QUTRIT_NAMES',
    'gate_category',
    'RelaxationParams',
    'ErrorModel',
    'DepolarizingNoise',
    'AmplitudeDamping',
    'DephasingNoise',
    'BitFlipNoise',
    'PhaseFlipNoise',
    'LeakageNoise',
    'SeepageNoise',
    'RelaxationNoise',
    'UnitaryError',
    'CustomErrorModel',
)

# ---------------------------------------------------------------------------
# Thermal-relaxation parameter bundle
# ---------------------------------------------------------------------------

@dataclass
class RelaxationParams:
    """Parameters for one category's relaxation channel.

    Passed to :class:`RelaxationNoise` per gate category.
    Qubit categories model GE (|0⟩–|1⟩) relaxation; qutrit categories
    model EF (|1⟩–|2⟩) relaxation.

    .. note::
        ``tphi`` is the *pure-dephasing* time Tφ, **not** T₂.
        Convert with ``Tφ = 1 / (1/T₂ − 1/(2·T₁))``.

    Attributes:
        t1 (float): T1 relaxation time (energy relaxation). Positive.
        tphi (float): Pure-dephasing time Tφ. Positive.
        p1 (float): Equilibrium excited-state population
            (0 = zero temperature). Defaults to ``0.0``.
        t (float): Gate duration. Defaults to ``1.0``.
    """
    t1:   float
    tphi: float
    p1:   float = 0.0
    t:    float = 1.0


# ---------------------------------------------------------------------------
# Gate-name category sets (built once at import time)
# ---------------------------------------------------------------------------

SINGLE_QUBIT_NAMES: frozenset = frozenset(SINGLE_QUBIT_GATES)
TWO_QUBIT_NAMES: frozenset = frozenset(TWO_QUBIT_GATES)
SINGLE_QUTRIT_NAMES: frozenset = frozenset(SINGLE_QUTRIT_GATES)
TWO_QUTRIT_NAMES: frozenset = frozenset(TWO_QUTRIT_GATES)

# Ordered priority: a gate in multiple sets (unlikely) gets the first match.
_CATEGORY_LOOKUP: Tuple = (
    ('single_qubit',  SINGLE_QUBIT_NAMES),
    ('two_qubit',     TWO_QUBIT_NAMES),
    ('single_qutrit', SINGLE_QUTRIT_NAMES),
    ('two_qutrit',    TWO_QUTRIT_NAMES),
)


def gate_category(gate_name: str) -> Optional[str]:
    """Return the broad category of a gate given its class name.

    Args:
        gate_name (str): ``type(gate).__name__``, e.g. ``'X'``,
            ``'CZ'``, ``'X01'``.

    Returns:
        str | None: one of ``'single_qubit'``, ``'two_qubit'``,
            ``'single_qutrit'``, ``'two_qutrit'``, or ``None`` if
            the gate class name is not in any known category.
    """
    for category, names in _CATEGORY_LOOKUP:
        if gate_name in names:
            return category
    return None


# ---------------------------------------------------------------------------
# Qutrit channel helpers (for channels quax doesn't expose with dims)
# ---------------------------------------------------------------------------

def _qutrit_bit_flip_channel(
    gamma_ge: float, gamma_ef: float
) -> quax.KrausMap:
    """Build a single-qutrit bit-flip KrausMap.

    Composes independent GE and EF bit-flip channels so each rate is
    unconstrained. Kraus operators (GE channel applied first):

        K₀₀ = √(1−γ_ge)·√(1−γ_ef) I₃
        K₀₁ = √γ_ge·√(1−γ_ef)      X₀₁
        K₁₀ = √(1−γ_ge)·√γ_ef      X₁₂
        K₁₁ = √(γ_ge·γ_ef)          X₁₂·X₀₁

    Args:
        gamma_ge (float): GE bit-flip rate (0 ≤ γ ≤ 1).
        gamma_ef (float): EF bit-flip rate (0 ≤ γ ≤ 1).

    Returns:
        quax.KrausMap: single-qutrit bit-flip channel.
    """
    x01 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)
    x12 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    eye = np.eye(3, dtype=complex)
    k00 = np.sqrt((1.0 - gamma_ge) * (1.0 - gamma_ef)) * eye
    k01 = np.sqrt(gamma_ge * (1.0 - gamma_ef)) * x01
    k10 = np.sqrt((1.0 - gamma_ge) * gamma_ef) * x12
    k11 = np.sqrt(gamma_ge * gamma_ef) * (x12 @ x01)
    matrix = jnp.array(np.stack([k00, k01, k10, k11]))
    return quax.KrausMap.from_matrix(matrix, dims=((3,), (3,)))


def _qutrit_phase_flip_channel(
    gamma_ge: float, gamma_ef: float
) -> quax.KrausMap:
    """Build a single-qutrit phase-flip KrausMap.

    Composes independent GE and EF phase-flip channels so each rate is
    unconstrained. Kraus operators (GE channel applied first):

        K₀₀ = √(1−γ_ge)·√(1−γ_ef) I₃
        K₀₁ = √γ_ge·√(1−γ_ef)      Z₀₁
        K₁₀ = √(1−γ_ge)·√γ_ef      Z₁₂
        K₁₁ = √(γ_ge·γ_ef)          Z₁₂·Z₀₁

    Args:
        gamma_ge (float): GE phase-flip rate (0 ≤ γ ≤ 1).
        gamma_ef (float): EF phase-flip rate (0 ≤ γ ≤ 1).

    Returns:
        quax.KrausMap: single-qutrit phase-flip channel.
    """
    z01 = np.diag([1.0, -1.0, 1.0]).astype(complex)
    z12 = np.diag([1.0, 1.0, -1.0]).astype(complex)
    eye = np.eye(3, dtype=complex)
    k00 = np.sqrt((1.0 - gamma_ge) * (1.0 - gamma_ef)) * eye
    k01 = np.sqrt(gamma_ge * (1.0 - gamma_ef)) * z01
    k10 = np.sqrt((1.0 - gamma_ge) * gamma_ef) * z12
    k11 = np.sqrt(gamma_ge * gamma_ef) * (z12 @ z01)
    matrix = jnp.array(np.stack([k00, k01, k10, k11]))
    return quax.KrausMap.from_matrix(matrix, dims=((3,), (3,)))


def _qutrit_dephasing_channel(
    gamma_ge: float, gamma_ef: float
) -> quax.KrausMap:
    """Build a single-qutrit dephasing KrausMap.

    Kills off-diagonal coherences independently in the GE and EF
    subspaces. Kraus operators are:

        K₀ = diag(1, √(1−γ_ge), √(1−γ_ef))
        K₁ = √γ_ge  |1⟩⟨1|
        K₂ = √γ_ef  |2⟩⟨2|

    The GE rate (γ_ge) is supplied by the ``single_qubit`` /
    ``two_qubit`` parameter of the parent :class:`DephasingNoise`;
    the EF rate (γ_ef) by ``single_qutrit`` / ``two_qutrit``.

    Args:
        gamma_ge (float): GE dephasing parameter (0 ≤ γ ≤ 1).
        gamma_ef (float): EF dephasing parameter (0 ≤ γ ≤ 1).

    Returns:
        quax.KrausMap: single-qutrit dephasing channel.
    """
    k0 = np.diag(
        [1.0, np.sqrt(1.0 - gamma_ge), np.sqrt(1.0 - gamma_ef)]
    ).astype(complex)
    k1 = np.zeros((3, 3), dtype=complex)
    k1[1, 1] = np.sqrt(gamma_ge)
    k2 = np.zeros((3, 3), dtype=complex)
    k2[2, 2] = np.sqrt(gamma_ef)
    # from_matrix expects shape (n_kraus, d_out, d_in)
    matrix = jnp.array(np.stack([k0, k1, k2]))
    return quax.KrausMap.from_matrix(matrix, dims=((3,), (3,)))


def _qutrit_thermal_relaxation_channel(
    ge: Optional['RelaxationParams'],
    ef: Optional['RelaxationParams'],
) -> quax.SuperOp:
    """Build a qutrit thermal-relaxation SuperOp.

    Constructs a Lindbladian with up to six jump operators — GE and EF
    decay, excitation, and pure-dephasing — then evolves it for the
    gate duration ``t``:

        L_ge↓  = √((1−p1_ge)/t1_ge) |0⟩⟨1|
        L_ge↑  = √(p1_ge/t1_ge)     |1⟩⟨0|
        L_ge,φ = √(1/(2·tphi_ge))   diag(1,−1, 0)
        L_ef↓  = √((1−p1_ef)/t1_ef) |1⟩⟨2|
        L_ef↑  = √(p1_ef/t1_ef)     |2⟩⟨1|
        L_ef,φ = √(1/(2·tphi_ef))   diag(0, 1,−1)

    Either *ge* or *ef* may be ``None`` (that subspace has no noise).
    Gate time ``t`` is taken from *ef* if provided, else from *ge*.

    Args:
        ge (RelaxationParams | None): GE subspace parameters.
        ef (RelaxationParams | None): EF subspace parameters.

    Returns:
        quax.SuperOp: qutrit thermal-relaxation channel.
    """
    t = ef.t if ef is not None else ge.t

    jump_ops = []

    if ge is not None:
        sigma_minus_ge = np.zeros((3, 3), dtype=complex)
        sigma_minus_ge[0, 1] = 1.0                          # |0⟩⟨1|
        sigma_plus_ge = np.zeros((3, 3), dtype=complex)
        sigma_plus_ge[1, 0] = 1.0                           # |1⟩⟨0|
        z_ge = np.diag([1.0, -1.0, 0.0]).astype(complex)

        inv_sqrt_t1 = 1.0 / np.sqrt(ge.t1)
        jump_ops.append(np.sqrt(1.0 - ge.p1) * inv_sqrt_t1 * sigma_minus_ge)
        if ge.p1 > 0.0:
            jump_ops.append(np.sqrt(ge.p1) * inv_sqrt_t1 * sigma_plus_ge)
        jump_ops.append(np.sqrt(0.5 / ge.tphi) * z_ge)

    if ef is not None:
        sigma_minus_ef = np.zeros((3, 3), dtype=complex)
        sigma_minus_ef[1, 2] = 1.0                          # |1⟩⟨2|
        sigma_plus_ef = np.zeros((3, 3), dtype=complex)
        sigma_plus_ef[2, 1] = 1.0                           # |2⟩⟨1|
        z_ef = np.diag([0.0, 1.0, -1.0]).astype(complex)

        inv_sqrt_t1 = 1.0 / np.sqrt(ef.t1)
        jump_ops.append(np.sqrt(1.0 - ef.p1) * inv_sqrt_t1 * sigma_minus_ef)
        if ef.p1 > 0.0:
            jump_ops.append(np.sqrt(ef.p1) * inv_sqrt_t1 * sigma_plus_ef)
        jump_ops.append(np.sqrt(0.5 / ef.tphi) * z_ef)

    L_stack = jnp.array(np.stack(jump_ops))
    lindbladian = quax.Lindbladian(
        hamiltonian=None,
        jump_operators=quax.Operator.from_matrix(L_stack, ((3,), (3,))),
    )
    return quax.evolve(lindbladian, t)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ErrorModel(ABC):
    """Abstract base class for category-based noise models.

    Subclasses build a ``quax`` channel for each gate category and
    expose it via :meth:`channel_for`. The
    :class:`~qcal.simulation.simulators.DensityMatrixSimulator`
    resolves gate categories using :func:`gate_category` and then
    calls this interface.

    Per-qudit readout confusion matrices can be registered with
    :meth:`add_readout_noise` and retrieved as ``quax``
    instruments via :meth:`instrument_for`.
    """

    def __init__(self) -> None:
        # qudit label (e.g. 'Q0') → column-stochastic confusion matrix
        self._confusion_matrices: Dict[str, np.ndarray] = {}

    def add_readout_noise(
        self,
        qudit: str | int,
        cmat:  'pd.DataFrame | np.ndarray',
    ) -> None:
        """Register a per-qudit readout confusion matrix.

        Accepts either a ``pandas`` DataFrame (e.g. the per-qubit slice
        ``ReadoutFidelity.cmat['Q0']``) or a plain ``numpy`` array.
        Both must use the **qcal row-stochastic convention**:
        ``C[prep, meas]`` = P(measure *meas* | prepared *prep*).
        The matrix is transposed internally to the column-stochastic
        form expected by ``quax``.

        Args:
            qudit (str | int): qudit label. Both ``'Q0'`` and the
                raw integer ``0`` are accepted; integers are
                normalised to the ``'Q{n}'`` string key.
            cmat (pd.DataFrame | np.ndarray): confusion matrix in
                the qcal row-stochastic convention.
        """
        key = f'Q{qudit}' if isinstance(qudit, int) else qudit
        if isinstance(cmat, pd.DataFrame):
            arr = cmat.to_numpy().astype(float)
        else:
            arr = np.asarray(cmat, dtype=float)
        # quax convention: C[i,j] = P(report i | system in j)
        self._confusion_matrices[key] = arr.T

    def confusion_matrix_for(
        self, qudit: str
    ) -> Optional[np.ndarray]:
        """Return the stored column-stochastic confusion matrix, or ``None``.

        Args:
            qudit (str): qudit label, e.g. ``'Q0'``.

        Returns:
            np.ndarray | None: column-stochastic confusion matrix, or
                ``None`` if none was registered for this qudit.
        """
        return self._confusion_matrices.get(qudit)

    def instrument_for(
        self, qudit: str, d: int = 2
    ) -> Optional['quax.QuantumInstrument']:
        """Build a noisy ``QuantumInstrument`` from a stored confusion matrix.

        Uses ``quax.instrument_from_confusion_and_transition`` with an
        identity transition matrix (post-measurement state = actual
        pre-measurement state; classical bit-flip model only).

        Args:
            qudit (str): qudit label, e.g. ``'Q0'``.
            d (int): Hilbert-space dimension. Defaults to ``2``.

        Returns:
            quax.QuantumInstrument | None: noisy measurement instrument,
                or ``None`` if no confusion matrix was registered for
                this qudit.
        """
        arr = self._confusion_matrices.get(qudit)
        if arr is None:
            return None
        transition = np.eye(d, dtype=float)
        return quax.instrument_from_confusion_and_transition(
            jnp.array(arr),
            jnp.array(transition),
            dims=(d,),
        )

    @abstractmethod
    def channel_for(
        self, gate_name: str
    ) -> Optional[quax.KrausMap | quax.SuperOp]:
        """Return the noise channel for a gate, or ``None``.

        Args:
            gate_name (str): ``type(gate).__name__``.

        Returns:
            quax.KrausMap | quax.SuperOp | None: channel to apply
                after this gate, or ``None`` for no noise.
        """

    def channel_for_gate(
        self, gate
    ) -> Optional[quax.KrausMap | quax.SuperOp]:
        """Return a per-instance noise channel for a gate, or ``None``.

        The default implementation always returns ``None``. Override in
        subclasses (e.g. :class:`CustomErrorModel`) to provide channels
        keyed to specific gate instances rather than gate categories.

        Args:
            gate: gate instance, e.g. ``X(0)``.

        Returns:
            quax.KrausMap | quax.SuperOp | None: channel to apply
                after this gate, or ``None`` for no noise.
        """
        return None


# ---------------------------------------------------------------------------
# Typical noise models
# ---------------------------------------------------------------------------

class DepolarizingNoise(ErrorModel):
    """Depolarizing noise applied uniformly by gate category.

    Delegates to ``quax.channels.depolarizing(rate, dims=(d,))``
    where *d* is 2 for qubit categories and 3 for qutrit categories.

    Example::

        noise = DepolarizingNoise(single_qubit=0.001, two_qubit=0.01)
        sim = DensityMatrixSimulator(noise_model=noise)

        # qutrit circuit
        noise = DepolarizingNoise(
            single_qutrit=0.002, two_qutrit=0.02
        )

    Args:
        single_qubit (float): depolarizing rate for single-qubit
            gates. Defaults to ``0.0`` (no noise).
        two_qubit (float): depolarizing rate for two-qubit gates.
            Defaults to ``0.0``.
        single_qutrit (float): depolarizing rate for single-qutrit
            gates. Defaults to ``0.0``.
        two_qutrit (float): depolarizing rate for two-qutrit gates,
            applied independently to each qutrit. Defaults to ``0.0``.
    """

    def __init__(
        self,
        single_qubit:  float = 0.0,
        two_qubit:     float = 0.0,
        single_qutrit: float = 0.0,
        two_qutrit:    float = 0.0,
    ) -> None:
        super().__init__()
        self._channels: Dict[str, Optional[quax.SuperOp]] = {
            'single_qubit': (
                quax.channels.depolarizing(single_qubit, dims=(2,))
                if single_qubit > 0.0 else None
            ),
            'two_qubit': (
                quax.channels.depolarizing(two_qubit, dims=(2,))
                if two_qubit > 0.0 else None
            ),
            'single_qutrit': (
                quax.channels.depolarizing(single_qutrit, dims=(3,))
                if single_qutrit > 0.0 else None
            ),
            'two_qutrit': (
                quax.channels.depolarizing(two_qutrit, dims=(3,))
                if two_qutrit > 0.0 else None
            ),
        }

    def channel_for(
        self, gate_name: str
    ) -> Optional[quax.SuperOp]:
        """Return the depolarizing channel for a gate, or ``None``.

        Args:
            gate_name (str): ``type(gate).__name__``.

        Returns:
            quax.SuperOp | None: the channel, or ``None`` if the
                gate's category has zero error rate or is unknown.
        """
        category = gate_category(gate_name)
        if category is None:
            return None
        return self._channels.get(category)


class AmplitudeDamping(ErrorModel):
    """Amplitude-damping (T1) noise applied uniformly by gate category.

    Delegates to ``quax.channels.amplitude_damping(rate, dims=(d,))``
    where *d* is 2 for qubit categories and 3 for qutrit categories.

    Example::

        noise = AmplitudeDamping(single_qubit=0.005, two_qubit=0.02)
        sim = DensityMatrixSimulator(noise_model=noise)

        # qutrit circuit
        noise = AmplitudeDamping(single_qutrit=0.005, two_qutrit=0.02)

    Args:
        single_qubit (float): damping rate γ for single-qubit gates
            (0 ≤ γ ≤ 1). Defaults to ``0.0``.
        two_qubit (float): damping rate γ for two-qubit gates,
            applied independently to each qubit. Defaults to ``0.0``.
        single_qutrit (float): damping rate γ for single-qutrit
            gates. Defaults to ``0.0``.
        two_qutrit (float): damping rate γ for two-qutrit gates,
            applied independently to each qutrit. Defaults to ``0.0``.
    """

    def __init__(
        self,
        single_qubit:  float = 0.0,
        two_qubit:     float = 0.0,
        single_qutrit: float = 0.0,
        two_qutrit:    float = 0.0,
    ) -> None:
        super().__init__()
        self._channels: Dict[str, Optional[quax.SuperOp]] = {
            'single_qubit': (
                quax.channels.amplitude_damping(single_qubit, dims=(2,))
                if single_qubit > 0.0 else None
            ),
            'two_qubit': (
                quax.channels.amplitude_damping(two_qubit, dims=(2,))
                if two_qubit > 0.0 else None
            ),
            'single_qutrit': (
                quax.channels.amplitude_damping(single_qutrit, dims=(3,))
                if single_qutrit > 0.0 else None
            ),
            'two_qutrit': (
                quax.channels.amplitude_damping(two_qutrit, dims=(3,))
                if two_qutrit > 0.0 else None
            ),
        }

    def channel_for(
        self, gate_name: str
    ) -> Optional[quax.SuperOp]:
        """Return the amplitude-damping channel for a gate, or ``None``.

        Args:
            gate_name (str): ``type(gate).__name__``.

        Returns:
            quax.SuperOp | None: the channel, or ``None`` if the
                gate's category has zero damping rate or is unknown.
        """
        category = gate_category(gate_name)
        if category is None:
            return None
        return self._channels.get(category)


class DephasingNoise(ErrorModel):
    """Dephasing (T2) noise applied uniformly by gate category.

    Qubit categories use ``quax.channels.dephasing(rate)`` and model
    GE (|0⟩–|1⟩) dephasing. Qutrit categories use a
    manually-constructed Kraus map (since ``quax.channels.dephasing``
    does not accept a *dims* argument) where the qubit rates supply
    γ_ge and the qutrit rates supply γ_ef:

    +---------------------+----------+----------+
    | Parameter           | γ_ge     | γ_ef     |
    +=====================+==========+==========+
    | ``single_qubit``    | ✓        |          |
    | ``single_qutrit``   |          | ✓        |
    | ``two_qubit``       | ✓        |          |
    | ``two_qutrit``      |          | ✓        |
    +---------------------+----------+----------+

    For a circuit that mixes qubit and qutrit gates, supply all four
    parameters and the appropriate rate is used per gate.

    Example::

        # qubit-only circuit
        noise = DephasingNoise(single_qubit=0.003, two_qubit=0.015)

        # qutrit-only circuit
        noise = DephasingNoise(single_qutrit=0.006, two_qutrit=0.012)

        # mixed: qutrit channel gets gamma_ge=0.003, gamma_ef=0.006
        noise = DephasingNoise(
            single_qubit=0.003, single_qutrit=0.006,
        )
        sim = DensityMatrixSimulator(noise_model=noise)

    Args:
        single_qubit (float): GE dephasing rate for single-qubit
            gates; also sets γ_ge for the single-qutrit channel
            (0 ≤ γ ≤ 1). Defaults to ``0.0``.
        two_qubit (float): GE dephasing rate for two-qubit gates;
            also sets γ_ge for the two-qutrit channel. Applied
            independently to each qubit. Defaults to ``0.0``.
        single_qutrit (float): EF dephasing rate (γ_ef) for the
            single-qutrit channel (0 ≤ γ ≤ 1). Defaults to ``0.0``.
        two_qutrit (float): EF dephasing rate (γ_ef) for the
            two-qutrit channel. Applied independently to each qutrit.
            Defaults to ``0.0``.
    """

    def __init__(
        self,
        single_qubit:  float = 0.0,
        two_qubit:     float = 0.0,
        single_qutrit: float = 0.0,
        two_qutrit:    float = 0.0,
    ) -> None:
        super().__init__()
        _sq_active = single_qubit > 0.0 or single_qutrit > 0.0
        _tq_active = two_qubit > 0.0 or two_qutrit > 0.0
        self._channels: Dict[str, Optional[quax.SuperOp | quax.KrausMap]] = {
            'single_qubit': (
                quax.channels.dephasing(single_qubit)
                if single_qubit > 0.0 else None
            ),
            'two_qubit': (
                quax.channels.dephasing(two_qubit)
                if two_qubit > 0.0 else None
            ),
            'single_qutrit': (
                _qutrit_dephasing_channel(single_qubit, single_qutrit)
                if _sq_active else None
            ),
            'two_qutrit': (
                _qutrit_dephasing_channel(two_qubit, two_qutrit)
                if _tq_active else None
            ),
        }

    def channel_for(
        self, gate_name: str
    ) -> Optional[quax.SuperOp | quax.KrausMap]:
        """Return the dephasing channel for a gate, or ``None``.

        Args:
            gate_name (str): ``type(gate).__name__``.

        Returns:
            quax.SuperOp | None: for qubit categories.
            quax.KrausMap | None: for qutrit categories (built from
                γ_ge = qubit rate, γ_ef = qutrit rate).
            None: if all relevant rates are zero or the gate is
                unknown.
        """
        category = gate_category(gate_name)
        if category is None:
            return None
        return self._channels.get(category)


class BitFlipNoise(ErrorModel):
    """Bit-flip noise applied uniformly by gate category.

    Qubit categories use ``quax.channels.bit_flip(rate)`` (GE flip).
    Qutrit categories use a manually-constructed Kraus map where the
    qubit rates supply γ_ge (X₀₁) and the qutrit rates supply γ_ef
    (X₁₂). Requires γ_ge + γ_ef ≤ 1 for qutrit channels.

    Example::

        noise = BitFlipNoise(single_qubit=0.01, two_qubit=0.02)
        sim = DensityMatrixSimulator(noise_model=noise)

        # qutrit circuit
        noise = BitFlipNoise(single_qutrit=0.01, two_qutrit=0.02)

    Args:
        single_qubit (float): GE bit-flip rate for single-qubit
            gates; also sets γ_ge for the single-qutrit channel.
            Defaults to ``0.0``.
        two_qubit (float): GE bit-flip rate for two-qubit gates;
            also sets γ_ge for the two-qutrit channel. Applied
            independently to each qubit. Defaults to ``0.0``.
        single_qutrit (float): EF bit-flip rate (γ_ef) for the
            single-qutrit channel (0 ≤ γ ≤ 1). Defaults to ``0.0``.
        two_qutrit (float): EF bit-flip rate (γ_ef) for the
            two-qutrit channel. Applied independently to each qutrit.
            Defaults to ``0.0``.
    """

    def __init__(
        self,
        single_qubit:  float = 0.0,
        two_qubit:     float = 0.0,
        single_qutrit: float = 0.0,
        two_qutrit:    float = 0.0,
    ) -> None:
        super().__init__()
        _sq_active = single_qubit > 0.0 or single_qutrit > 0.0
        _tq_active = two_qubit > 0.0 or two_qutrit > 0.0
        self._channels: Dict[str, Optional[quax.SuperOp | quax.KrausMap]] = {
            'single_qubit': (
                quax.channels.bit_flip(single_qubit)
                if single_qubit > 0.0 else None
            ),
            'two_qubit': (
                quax.channels.bit_flip(two_qubit)
                if two_qubit > 0.0 else None
            ),
            'single_qutrit': (
                _qutrit_bit_flip_channel(single_qubit, single_qutrit)
                if _sq_active else None
            ),
            'two_qutrit': (
                _qutrit_bit_flip_channel(two_qubit, two_qutrit)
                if _tq_active else None
            ),
        }

    def channel_for(
        self, gate_name: str
    ) -> Optional[quax.SuperOp | quax.KrausMap]:
        """Return the bit-flip channel for a gate, or ``None``.

        Args:
            gate_name (str): ``type(gate).__name__``.

        Returns:
            quax.SuperOp | None: for qubit categories.
            quax.KrausMap | None: for qutrit categories.
            None: if all relevant rates are zero or the gate is
                unknown.
        """
        category = gate_category(gate_name)
        if category is None:
            return None
        return self._channels.get(category)


class PhaseFlipNoise(ErrorModel):
    """Phase-flip noise applied uniformly by gate category.

    Qubit categories use ``quax.channels.phase_flip(rate)`` (GE
    phase flip). Qutrit categories use a manually-constructed Kraus
    map where the qubit rates supply γ_ge (Z₀₁) and the qutrit rates
    supply γ_ef (Z₁₂). Requires γ_ge + γ_ef ≤ 1 for qutrit channels.

    Example::

        noise = PhaseFlipNoise(single_qubit=0.01, two_qubit=0.02)
        sim = DensityMatrixSimulator(noise_model=noise)

        # qutrit circuit
        noise = PhaseFlipNoise(single_qutrit=0.01, two_qutrit=0.02)

    Args:
        single_qubit (float): GE phase-flip rate for single-qubit
            gates; also sets γ_ge for the single-qutrit channel.
            Defaults to ``0.0``.
        two_qubit (float): GE phase-flip rate for two-qubit gates;
            also sets γ_ge for the two-qutrit channel. Applied
            independently to each qubit. Defaults to ``0.0``.
        single_qutrit (float): EF phase-flip rate (γ_ef) for the
            single-qutrit channel (0 ≤ γ ≤ 1). Defaults to ``0.0``.
        two_qutrit (float): EF phase-flip rate (γ_ef) for the
            two-qutrit channel. Applied independently to each qutrit.
            Defaults to ``0.0``.
    """

    def __init__(
        self,
        single_qubit:  float = 0.0,
        two_qubit:     float = 0.0,
        single_qutrit: float = 0.0,
        two_qutrit:    float = 0.0,
    ) -> None:
        super().__init__()
        _sq_active = single_qubit > 0.0 or single_qutrit > 0.0
        _tq_active = two_qubit > 0.0 or two_qutrit > 0.0
        self._channels: Dict[str, Optional[quax.SuperOp | quax.KrausMap]] = {
            'single_qubit': (
                quax.channels.phase_flip(single_qubit)
                if single_qubit > 0.0 else None
            ),
            'two_qubit': (
                quax.channels.phase_flip(two_qubit)
                if two_qubit > 0.0 else None
            ),
            'single_qutrit': (
                _qutrit_phase_flip_channel(single_qubit, single_qutrit)
                if _sq_active else None
            ),
            'two_qutrit': (
                _qutrit_phase_flip_channel(two_qubit, two_qutrit)
                if _tq_active else None
            ),
        }

    def channel_for(
        self, gate_name: str
    ) -> Optional[quax.SuperOp | quax.KrausMap]:
        """Return the phase-flip channel for a gate, or ``None``.

        Args:
            gate_name (str): ``type(gate).__name__``.

        Returns:
            quax.SuperOp | None: for qubit categories.
            quax.KrausMap | None: for qutrit categories.
            None: if all relevant rates are zero or the gate is
                unknown.
        """
        category = gate_category(gate_name)
        if category is None:
            return None
        return self._channels.get(category)


class RelaxationNoise(ErrorModel):
    """Relaxation noise (T1 + pure dephasing Tφ) applied by gate category.

    Combines T1 (energy relaxation) and pure dephasing (Tφ) into a
    single CPTP channel per category.

    Qubit categories use ``quax.channels.thermal_relaxation`` directly.
    Qutrit categories are built from a manually-constructed Lindbladian
    with separate GE and EF jump operators, following the convention
    that qubit params supply γ_ge and qutrit params supply γ_ef:

    +---------------------+------+------+
    | Parameter           | γ_ge | γ_ef |
    +=====================+======+======+
    | ``single_qubit``    | ✓    |      |
    | ``single_qutrit``   |      | ✓    |
    | ``two_qubit``       | ✓    |      |
    | ``two_qutrit``      |      | ✓    |
    +---------------------+------+------+

    The qutrit channel uses the gate duration ``t`` from the qutrit
    (EF) params when provided, otherwise falls back to the qubit (GE)
    params.

    Example::

        from qcal.simulation.error_models import (
            RelaxationNoise, RelaxationParams,
        )
        noise = RelaxationNoise(
            single_qubit=RelaxationParams(t1=50e-6, tphi=30e-6, t=20e-9),
            two_qubit=RelaxationParams(t1=50e-6, tphi=30e-6, t=60e-9),
        )
        sim = DensityMatrixSimulator(noise_model=noise)

        # qutrit: GE from single_qubit, EF from single_qutrit
        noise = ThermalRelaxationNoise(
            single_qubit=RelaxationParams(t1=50e-6, tphi=30e-6, t=20e-9),
            single_qutrit=RelaxationParams(t1=30e-6, tphi=20e-6, t=20e-9),
        )

    Args:
        single_qubit (RelaxationParams | None): params for single-qubit
            gates; also GE component of the single-qutrit channel.
            Defaults to ``None``.
        two_qubit (RelaxationParams | None): params for two-qubit
            gates; also GE component of the two-qutrit channel.
            Defaults to ``None``.
        single_qutrit (RelaxationParams | None): EF params for the
            single-qutrit channel. Defaults to ``None``.
        two_qutrit (RelaxationParams | None): EF params for the
            two-qutrit channel. Defaults to ``None``.
    """

    def __init__(
        self,
        single_qubit:  Optional[RelaxationParams] = None,
        two_qubit:     Optional[RelaxationParams] = None,
        single_qutrit: Optional[RelaxationParams] = None,
        two_qutrit:    Optional[RelaxationParams] = None,
    ) -> None:
        super().__init__()
        self._channels: Dict[str, Optional[quax.SuperOp]] = {
            'single_qubit': (
                quax.channels.thermal_relaxation(
                    single_qubit.t1, single_qubit.tphi,
                    single_qubit.p1, single_qubit.t,
                ) if single_qubit is not None else None
            ),
            'two_qubit': (
                quax.channels.thermal_relaxation(
                    two_qubit.t1, two_qubit.tphi,
                    two_qubit.p1, two_qubit.t,
                ) if two_qubit is not None else None
            ),
            'single_qutrit': (
                _qutrit_thermal_relaxation_channel(
                    single_qubit, single_qutrit
                ) if single_qubit is not None or single_qutrit is not None
                else None
            ),
            'two_qutrit': (
                _qutrit_thermal_relaxation_channel(
                    two_qubit, two_qutrit
                ) if two_qubit is not None or two_qutrit is not None
                else None
            ),
        }

    def channel_for(
        self, gate_name: str
    ) -> Optional[quax.SuperOp]:
        """Return the relaxation channel for a gate, or ``None``.

        Args:
            gate_name (str): ``type(gate).__name__``.

        Returns:
            quax.SuperOp | None: the channel, or ``None`` if no params
                were provided for this gate's category.
        """
        category = gate_category(gate_name)
        if category is None:
            return None
        return self._channels.get(category)


class LeakageNoise(ErrorModel):
    """Leakage noise (|1⟩→|2⟩) for qutrit gate categories.

    Delegates to ``quax.channels.leakage(rate)``. Qutrit-only;
    qubit categories always return ``None``.

    Example::

        noise = LeakageNoise(single_qutrit=0.005, two_qutrit=0.01)
        sim = DensityMatrixSimulator(noise_model=noise)

    Args:
        single_qutrit (float): leakage rate for single-qutrit gates.
            Defaults to ``0.0``.
        two_qutrit (float): leakage rate for two-qutrit gates,
            applied independently to each qutrit. Defaults to ``0.0``.
    """

    def __init__(
        self,
        single_qutrit: float = 0.0,
        two_qutrit:    float = 0.0,
    ) -> None:
        super().__init__()
        self._channels: Dict[str, Optional[quax.SuperOp]] = {
            'single_qubit':  None,
            'two_qubit':     None,
            'single_qutrit': (
                quax.channels.leakage(single_qutrit)
                if single_qutrit > 0.0 else None
            ),
            'two_qutrit': (
                quax.channels.leakage(two_qutrit)
                if two_qutrit > 0.0 else None
            ),
        }

    def channel_for(
        self, gate_name: str
    ) -> Optional[quax.SuperOp]:
        """Return the leakage channel for a gate, or ``None``.

        Args:
            gate_name (str): ``type(gate).__name__``.

        Returns:
            quax.SuperOp | None: the channel, or ``None`` if the
                gate's category has zero rate or is unknown.
        """
        category = gate_category(gate_name)
        if category is None:
            return None
        return self._channels.get(category)


class SeepageNoise(ErrorModel):
    """Seepage noise (|2⟩→|1⟩) for qutrit gate categories.

    Delegates to ``quax.channels.seepage(rate)``. Qutrit-only;
    qubit categories always return ``None``.

    Example::

        noise = SeepageNoise(single_qutrit=0.005, two_qutrit=0.01)
        sim = DensityMatrixSimulator(noise_model=noise)

    Args:
        single_qutrit (float): seepage rate for single-qutrit gates.
            Defaults to ``0.0``.
        two_qutrit (float): seepage rate for two-qutrit gates,
            applied independently to each qutrit. Defaults to ``0.0``.
    """

    def __init__(
        self,
        single_qutrit: float = 0.0,
        two_qutrit:    float = 0.0,
    ) -> None:
        super().__init__()
        self._channels: Dict[str, Optional[quax.SuperOp]] = {
            'single_qubit':  None,
            'two_qubit':     None,
            'single_qutrit': (
                quax.channels.seepage(single_qutrit)
                if single_qutrit > 0.0 else None
            ),
            'two_qutrit': (
                quax.channels.seepage(two_qutrit)
                if two_qutrit > 0.0 else None
            ),
        }

    def channel_for(
        self, gate_name: str
    ) -> Optional[quax.SuperOp]:
        """Return the seepage channel for a gate, or ``None``.

        Args:
            gate_name (str): ``type(gate).__name__``.

        Returns:
            quax.SuperOp | None: the channel, or ``None`` if the
                gate's category has zero rate or is unknown.
        """
        category = gate_category(gate_name)
        if category is None:
            return None
        return self._channels.get(category)


# ---------------------------------------------------------------------------
# Unitary (coherent) channel helper
# ---------------------------------------------------------------------------

_UNITARY_DIMS: dict = {
    2: ((2,),    (2,)),
    3: ((3,),    (3,)),
    4: ((2, 2),  (2, 2)),
    9: ((3, 3),  (3, 3)),
}


def _unitary_channel(
    U: Optional[np.ndarray],
) -> Optional[quax.KrausMap]:
    """Build a single-Kraus-operator KrausMap from a unitary U.

    Returns ``None`` if *U* is ``None`` or numerically the identity
    (``np.allclose(U, np.eye(d))``).

    Supported shapes and inferred ``dims``:

    - ``(2, 2)`` → ``dims=((2,), (2,))`` (single qubit)
    - ``(3, 3)`` → ``dims=((3,), (3,))`` (single qutrit)
    - ``(4, 4)`` → ``dims=((2, 2), (2, 2))`` (joint two-qubit)
    - ``(9, 9)`` → ``dims=((3, 3), (3, 3))`` (joint two-qutrit)

    Args:
        U (np.ndarray | None): square unitary matrix, or ``None``.

    Returns:
        quax.KrausMap | None: rank-1 channel, or ``None``.

    Raises:
        ValueError: if the matrix shape is not one of the four
            recognised forms.
    """
    if U is None:
        return None
    U = np.asarray(U, dtype=complex)
    d = U.shape[0]
    if np.allclose(U, np.eye(d)):
        return None
    dims = _UNITARY_DIMS.get(d)
    if dims is None:
        raise ValueError(
            f'UnitaryError: unsupported matrix size {d}×{d}. '
            'Expected 2×2, 3×3, 4×4, or 9×9.'
        )
    kraus = jnp.array(U[np.newaxis])   # shape (1, d, d)
    return quax.KrausMap.from_matrix(kraus, dims=dims)


class UnitaryError(ErrorModel):
    """Coherent (unitary) error channel applied by gate category or instance.

    After each gate, the error channel ρ → U ρ U† is applied, where U
    is a fixed unitary matrix.  This is a rank-1 (single-Kraus-operator)
    channel and therefore preserves state purity.

    Two levels of granularity are supported:

    * **Category-level** (*single_qubit*, *two_qubit*, etc.) — the same
      unitary is applied after every gate in that category.
    * **Per-gate-instance** (*gate_unitaries*) — a specific unitary for
      an individual gate instance (e.g. ``CZ((0, 1))``).  These take
      priority over the category-level channel: if a gate matches an
      entry in *gate_unitaries*, only the per-gate unitary is applied;
      the category channel is not applied for that gate.

    When no channel is configured for a gate (identity or ``None``), no
    channel is added and the simulation is equivalent to a state-vector
    simulation.

    The *two_qubit* (and *two_qutrit*) parameter accepts either a
    per-qudit local unitary (2×2 or 3×3, broadcast independently to
    each qudit of the gate) or a joint unitary over the whole gate
    subsystem (4×4 or 9×9).

    Example::

        import numpy as np
        from qcal.simulation.error_models import UnitaryError
        from qcal.gates.single_qubit import X
        from qcal.gates.two_qubit import CZ

        # Category-level: small Rz over-rotation on every single-qubit gate
        theta = 0.05  # radians
        Rz = np.array([
            [np.exp(-0.5j * theta), 0],
            [0,                     np.exp(0.5j * theta)],
        ])
        noise = UnitaryError(single_qubit=Rz)

        # Per-gate: different unitary on a specific CZ instance
        noise = UnitaryError(
            single_qubit=Rz,
            gate_unitaries={CZ((0, 1)): U_cz_error},
        )

        # U = I → equivalent to the state-vector simulator
        noise = UnitaryError()   # all None, no channels added

    Args:
        single_qubit (np.ndarray | None): 2×2 unitary error for
            single-qubit gates. ``None`` or the identity means no
            channel. Defaults to ``None``.
        two_qubit (np.ndarray | None): unitary error for two-qubit
            gates.  May be 2×2 (applied independently to each qubit)
            or 4×4 (applied jointly to the gate subsystem). ``None``
            or the identity means no channel. Defaults to ``None``.
        single_qutrit (np.ndarray | None): 3×3 unitary error for
            single-qutrit gates. Defaults to ``None``.
        two_qutrit (np.ndarray | None): unitary error for two-qutrit
            gates.  May be 3×3 (per-qutrit) or 9×9 (joint). Defaults
            to ``None``.
        gate_unitaries (dict | None): mapping from gate instances (e.g.
            ``CZ((0, 1))``) to unitary error matrices. Gate instances
            are compared by value using
            :meth:`~qcal.gates.gate.Gate.__eq__`. A ``None`` matrix or
            the identity matrix is treated as no channel. Defaults to
            ``None``.
    """

    def __init__(
        self,
        single_qubit:   Optional[np.ndarray] = None,
        two_qubit:      Optional[np.ndarray] = None,
        single_qutrit:  Optional[np.ndarray] = None,
        two_qutrit:     Optional[np.ndarray] = None,
        gate_unitaries: Optional[dict[Gate, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self._channels: Dict[str, Optional[quax.KrausMap]] = {
            'single_qubit':  _unitary_channel(single_qubit),
            'two_qubit':     _unitary_channel(two_qubit),
            'single_qutrit': _unitary_channel(single_qutrit),
            'two_qutrit':    _unitary_channel(two_qutrit),
        }
        self._gate_channels: dict = {
            gate: _unitary_channel(U)
            for gate, U in (gate_unitaries or {}).items()
        }

    def channel_for(
        self, gate_name: str
    ) -> Optional[quax.KrausMap]:
        """Return the category-level unitary channel for a gate, or ``None``.

        Args:
            gate_name (str): ``type(gate).__name__``.

        Returns:
            quax.KrausMap | None: single-Kraus-operator channel, or
                ``None`` if the category has no configured unitary.
        """
        category = gate_category(gate_name)
        if category is None:
            return None
        return self._channels.get(category)

    def channel_for_gate(
        self, gate
    ) -> Optional[quax.KrausMap]:
        """Return the per-instance unitary channel for ``gate``, or ``None``.

        Looks up the gate by value in the *gate_unitaries* dict supplied
        at construction.  Returns ``None`` if no entry was registered for
        this gate instance.

        Args:
            gate: gate instance, e.g. ``CZ((0, 1))``.

        Returns:
            quax.KrausMap | None: single-Kraus-operator channel, or
                ``None``.
        """
        return self._gate_channels.get(gate)


# ---------------------------------------------------------------------------
# Channel composition helpers
# ---------------------------------------------------------------------------

_PAULIS_1Q = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]
_PAULI_1Q_MAP: dict = dict(zip('IXYZ', _PAULIS_1Q, strict=True))


def _to_superop_matrix(channel) -> np.ndarray:
    """Return the d²×d² superoperator matrix for a quax channel.

    Uses the column-stacking (Liouville) convention: for a Kraus operator K
    the contribution is K ⊗ conj(K).

    Args:
        channel: quax.SuperOp or quax.KrausMap.

    Returns:
        np.ndarray: real or complex d²×d² superoperator matrix.
    """
    if isinstance(channel, quax.SuperOp):
        return np.asarray(channel.matrix)
    # KrausMap.matrix shape: (n_kraus, d_out, d_in)
    mat = np.asarray(channel.matrix)
    d_out, d_in = mat.shape[-2], mat.shape[-1]
    S = np.zeros((d_out * d_in, d_out * d_in), dtype=complex)
    for K in mat:  # quax convention: S = Σ_i conj(K_i) ⊗ K_i
        S += np.kron(np.conj(K), K)
    return S


def _compose_channels(channels: list) -> quax.SuperOp:
    """Compose a list of quax channels into a single SuperOp.

    Channels are applied left to right (``channels[0]`` first, i.e. nearest
    to the gate). All channels must act on the same Hilbert-space dimension.

    Args:
        channels (list): list of ``quax.KrausMap`` or ``quax.SuperOp``,
            length ≥ 2.

    Returns:
        quax.SuperOp: composed channel.
    """
    matrices = [_to_superop_matrix(ch) for ch in channels]
    composed = matrices[0]
    for m in matrices[1:]:
        composed = m @ composed
    dims = channels[0].dims  # preserve qudit structure (e.g. ((2,2),(2,2)))
    return quax.SuperOp.from_matrix(jnp.array(composed), dims=dims)


def _pauli_basis_unitary(d: int) -> np.ndarray:
    """Build the d²×d² unitary whose columns are vec(P_k / sqrt(d)).

    P_k are the n-qubit Pauli operators (tensor products of {I, X, Y, Z})
    in lexicographic order.  Requires d = 2^n.

    Args:
        d (int): Hilbert-space dimension; must be a power of 2.

    Returns:
        np.ndarray: complex d²×d² unitary matrix.

    Raises:
        ValueError: if d is not a positive power of 2.
    """
    n = int(round(np.log2(d))) if d > 1 else 0
    if 2 ** n != d:
        raise ValueError(
            f'PTM conversion requires a power-of-2 dimension; got d={d}. '
            'Qutrit gates are not supported by from_PTM.'
        )
    d2 = d * d
    U = np.zeros((d2, d2), dtype=complex)
    for k, combo in enumerate(_iproduct(range(4), repeat=n)):
        op = _PAULIS_1Q[combo[0]]
        for idx in combo[1:]:
            op = np.kron(op, _PAULIS_1Q[idx])
        U[:, k] = op.flatten(order='C') / np.sqrt(d)
    return U


def _ptm_to_superop(
    ptm: np.ndarray,
    qudit_dims: tuple = (2,),
) -> quax.SuperOp:
    """Convert a Pauli Transfer Matrix to a quax SuperOp.

    The PTM convention used here is
    ``R[i, j] = Tr(P_i @ E(P_j)) / d``
    where ``P_k`` are the (unnormalised) n-qubit Pauli operators in
    lexicographic order and ``E`` is the channel being represented.
    This matches the convention used by Qiskit and most process-
    tomography tools.

    Internally the PTM is converted to the row-stacking (Liouville)
    superoperator via ``L = U @ R @ U†``, where the columns of ``U``
    are ``vec(P_k / sqrt(d))``.

    Args:
        ptm (np.ndarray): real or complex d²×d² PTM array.
        qudit_dims (tuple): per-qudit dimensions, e.g. ``(2,)`` for a
            single qubit or ``(2, 2)`` for a two-qubit gate.  Product
            must equal ``sqrt(ptm.shape[0])``.

    Returns:
        quax.SuperOp: equivalent Liouville superoperator.

    Raises:
        ValueError: if ptm is not square or d is not a power of 2.
    """
    ptm = np.asarray(ptm, dtype=complex)
    if ptm.ndim != 2 or ptm.shape[0] != ptm.shape[1]:
        raise ValueError(
            f'PTM must be a square 2-D array; got shape {ptm.shape}.'
        )
    d2 = ptm.shape[0]
    d = int(round(d2 ** 0.5))
    U = _pauli_basis_unitary(d)
    L = U @ ptm @ U.conj().T
    dims = (qudit_dims, qudit_dims)
    return quax.SuperOp.from_matrix(jnp.array(L), dims=dims)


def _pauli_str_to_matrix(pauli_str: str) -> np.ndarray:
    """Return the tensor-product Pauli matrix for a string like ``'ZI'``.

    Each character must be one of ``'I'``, ``'X'``, ``'Y'``, ``'Z'``
    (case-insensitive).

    Args:
        pauli_str (str): Pauli string of length n for n qubits.

    Returns:
        np.ndarray: complex (2^n × 2^n) Pauli matrix.
    """
    mat = _PAULI_1Q_MAP[pauli_str[0].upper()]
    for c in pauli_str[1:]:
        mat = np.kron(mat, _PAULI_1Q_MAP[c.upper()])
    return mat


def _pauli_noise_to_kraus(pauli_probs: dict) -> quax.KrausMap:
    """Build a :class:`quax.KrausMap` from a Pauli → probability mapping.

    The channel is ``E(ρ) = Σ_k p_k P_k ρ P_k†``.  If the
    probabilities sum to less than one, an implicit identity Pauli
    ``'I…I'`` is added with the remaining weight, ensuring the channel
    is trace-preserving.

    All Pauli strings must have the same length (one character per qubit).

    Args:
        pauli_probs (dict): mapping from Pauli string (e.g. ``'ZI'``) to
            its probability weight.  Weights must be non-negative and
            sum to ≤ 1.

    Returns:
        quax.KrausMap: Pauli noise channel with ``dims=((2,)*n, (2,)*n)``.

    Raises:
        ValueError: if *pauli_probs* is empty, Pauli strings have
            inconsistent lengths, any probability is negative, or the
            weights sum to more than one.
    """
    if not pauli_probs:
        raise ValueError('pauli_probs must be non-empty.')
    first_key = next(iter(pauli_probs))
    n = len(first_key)
    d = 2 ** n

    total_prob = 0.0
    kraus_ops = []
    for pauli_str, prob in pauli_probs.items():
        if len(pauli_str) != n:
            raise ValueError(
                f'All Pauli strings must have the same length; '
                f'expected {n}, got length {len(pauli_str)!r} '
                f'for {pauli_str!r}.'
            )
        if prob < 0:
            raise ValueError(
                f'Probability for {pauli_str!r} is negative: {prob}.'
            )
        if prob == 0:
            continue
        kraus_ops.append(
            np.sqrt(prob) * _pauli_str_to_matrix(pauli_str)
        )
        total_prob += prob

    if total_prob > 1.0 + 1e-10:
        raise ValueError(
            f'Pauli probabilities sum to {total_prob:.6g} > 1.'
        )
    remaining = 1.0 - total_prob
    if remaining > 1e-10:
        kraus_ops.insert(0, np.sqrt(remaining) * np.eye(d, dtype=complex))

    ops = jnp.array(np.stack(kraus_ops).astype(complex))
    dims = ((2,) * n, (2,) * n)
    return quax.KrausMap.from_matrix(ops, dims=dims)


# ---------------------------------------------------------------------------
# Custom (composite) noise model
# ---------------------------------------------------------------------------

class CustomErrorModel(ErrorModel):
    """Compose multiple :class:`ErrorModel` instances into one.

    Channels from each sub-model are composed sequentially in the order the
    sub-models are passed (left to right). Confusion matrices are resolved
    from the instance's own dict first, then from sub-models in order —
    allowing per-instance overrides without modifying sub-models.

    ``CustomErrorModel`` is itself a :class:`ErrorModel` and may therefore
    appear as a sub-model inside another ``CustomErrorModel``.

    Example::

        noise = CustomErrorModel(
            DepolarizingNoise(single_qubit=0.01),
            RelaxationNoise(
                single_qubit=RelaxationParams(
                    t1=50e-6, tphi=30e-6, t=50e-9
                )
            ),
        )
        noise.add_readout_noise('Q0', cmat)
        sim = DensityMatrixSimulator(noise_model=noise)

    Args:
        *error_models (ErrorModel): sub-models to compose. Accepts zero or
            more sub-models; an empty ``CustomErrorModel`` returns ``None``
            for all channels.
    """

    def __init__(self, *error_models: ErrorModel) -> None:
        super().__init__()
        self._error_models: List[ErrorModel] = list(error_models)
        # gate instance → SuperOp (populated by from_PTM)
        self._gate_channels: dict = {}

    @property
    def error_models(self) -> List[ErrorModel]:
        """Sub-models held by this composite."""
        return self._error_models

    def channel_for(
        self, gate_name: str
    ) -> Optional[quax.KrausMap | quax.SuperOp]:
        """Return the composed channel for a gate, or ``None``.

        Collects the channel each sub-model provides for ``gate_name`` and
        composes them in order. Returns ``None`` if no sub-model has a
        channel for that gate.

        Args:
            gate_name (str): ``type(gate).__name__``, e.g. ``'X90'``.

        Returns:
            quax.KrausMap | quax.SuperOp | None: composed channel, or
                ``None``.
        """
        channels = [
            ch for m in self._error_models
            if (ch := m.channel_for(gate_name)) is not None
        ]
        if not channels:
            return None
        if len(channels) == 1:
            return channels[0]
        return _compose_channels(channels)

    def confusion_matrix_for(self, qudit: str) -> Optional[np.ndarray]:
        """Return the confusion matrix for ``qudit``, or ``None``.

        Own matrices (added via :meth:`add_readout_noise`) take priority.
        Falls back to sub-models in order.

        Args:
            qudit (str): qudit label, e.g. ``'Q0'``.

        Returns:
            np.ndarray | None: column-stochastic confusion matrix, or
                ``None``.
        """
        mat = self._confusion_matrices.get(qudit)
        if mat is not None:
            return mat
        for m in self._error_models:
            mat = m.confusion_matrix_for(qudit)
            if mat is not None:
                return mat
        return None

    def instrument_for(
        self, qudit: str, d: int = 2
    ) -> Optional['quax.QuantumInstrument']:
        """Build a :class:`quax.QuantumInstrument` from the resolved cmat.

        Uses :meth:`confusion_matrix_for` (own dict + sub-model fall-through)
        so that sub-model confusion matrices are picked up automatically.

        Args:
            qudit (str): qudit label, e.g. ``'Q0'``.
            d (int): Hilbert-space dimension. Defaults to ``2``.

        Returns:
            quax.QuantumInstrument | None: noisy instrument, or ``None`` if
                no confusion matrix is registered for this qudit.
        """
        arr = self.confusion_matrix_for(qudit)
        if arr is None:
            return None
        transition = np.eye(d, dtype=float)
        return quax.instrument_from_confusion_and_transition(
            jnp.array(arr), jnp.array(transition), dims=(d,),
        )

    def channel_for_gate(
        self, gate
    ) -> Optional[quax.SuperOp]:
        """Return the per-instance channel for ``gate``, or ``None``.

        Checks the instance's own ``_gate_channels`` dict (populated by
        :meth:`from_PTM`) first, then falls through to sub-models in order.
        Two gate instances are considered equal when they have the same name,
        qubits, subspace, and params (see :meth:`qcal.gates.gate.Gate.__eq__`).

        Args:
            gate: gate instance, e.g. ``X(0)``.

        Returns:
            quax.SuperOp | None: error channel, or ``None``.
        """
        ch = self._gate_channels.get(gate)
        if ch is not None:
            return ch
        for m in self._error_models:
            ch = m.channel_for_gate(gate)
            if ch is not None:
                return ch
        return None

    @classmethod
    def from_PTM(
        cls,
        gate_ptm_dict: dict,
    ) -> 'CustomErrorModel':
        """Create a :class:`CustomErrorModel` from per-gate PTMs.

        Each entry maps a gate instance to its error channel expressed as a
        Pauli Transfer Matrix.  The PTM convention is
        ``R[i, j] = Tr(P_i @ E(P_j)) / d``, where ``P_k`` are the
        (unnormalised) n-qubit Pauli operators in lexicographic order and
        ``E`` is the error channel (not the full gate unitary + error).

        Gate instances are compared by value — ``X(0)`` created in this call
        and ``X(0)`` created independently in the circuit compare as equal
        provided they share the same name, qubits, subspace, and params.

        Only qubit gates (Hilbert-space dimension that is a power of 2) are
        supported; qutrit gates raise :class:`ValueError`.

        Example::

            import qcal.gates.single_qubit as gates
            import numpy as np

            # Identity-like error on X(0) (slightly depolarised)
            ptm = np.diag([1, 0.99, 0.99, 0.99])
            noise = CustomErrorModel.from_PTM({gates.X(0): ptm})
            sim = DensityMatrixSimulator(noise_model=noise)

        Args:
            gate_ptm_dict (dict): mapping from gate instance to PTM array.
                Each PTM must be a square (d²×d²) real or complex array.

        Returns:
            CustomErrorModel: new model carrying per-gate PTM channels.
        """
        model = cls()
        for gate, ptm in gate_ptm_dict.items():
            n = len(gate.qudits)
            qudit_dims = (2,) * n
            model._gate_channels[gate] = _ptm_to_superop(
                np.asarray(ptm, dtype=complex), qudit_dims
            )
        return model

    @classmethod
    def from_PNR(
        cls,
        gate_pauli_dict: dict,
    ) -> 'CustomErrorModel':
        """Create a :class:`CustomErrorModel` from Pauli Noise Reconstruction.

        Each entry maps a gate instance to a ``{Pauli_string: probability}``
        dict representing the stochastic Pauli error channel on that gate.
        The channel is ``E(ρ) = Σ_k p_k P_k ρ P_k†``.

        Pauli strings are composed of ``'I'``, ``'X'``, ``'Y'``, ``'Z'``
        (case-insensitive), one character per qubit.  Probabilities must be
        non-negative and sum to ≤ 1; any remainder is assigned to the all-
        identity Pauli automatically.  Single- and multi-qubit Pauli errors
        are both supported (e.g. ``'Z'`` for 1-qubit, ``'ZZ'`` for 2-qubit
        joint errors).

        Gate instances are compared by value — ``CZ(0, 1)`` created here
        and ``CZ(0, 1)`` in the circuit compare as equal when they share
        the same name, qubits, subspace, and params.

        Example::

            import qcal.gates.single_qubit as sq
            import qcal.gates.two_qubit as tq

            noise = CustomErrorModel.from_PNR({
                sq.X(0):      {'Z': 0.002, 'X': 0.001},
                tq.CZ((0,1)): {'ZI': 0.005, 'IZ': 0.003, 'ZZ': 0.001},
            })
            sim = DensityMatrixSimulator(noise_model=noise)

        Args:
            gate_pauli_dict (dict): mapping from gate instance to a
                ``{Pauli_string: probability}`` dict.  All Pauli strings
                for a given gate must have the same length, which must
                equal the gate's qudit count.

        Returns:
            CustomErrorModel: new model carrying per-gate Pauli channels.

        Raises:
            ValueError: if Pauli string lengths do not match the gate's
                qudit count, if probabilities are negative, or if they
                sum to more than 1.
        """
        model = cls()
        for gate, pauli_probs in gate_pauli_dict.items():
            if not pauli_probs:
                continue
            n_paulis = len(next(iter(pauli_probs)))
            n_gate = len(gate.qudits)
            if n_paulis != n_gate:
                raise ValueError(
                    f'Gate {gate.name}{gate.qubits} acts on {n_gate} '
                    f'qudit(s), but Pauli strings have length {n_paulis}.'
                )
            model._gate_channels[gate] = _pauli_noise_to_kraus(pauli_probs)
        return model
