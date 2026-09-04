"""ZXZXZ decomposition dispatch table for non-parametrized single-qubit gates.

Each entry maps a gate name (str) or gate class (type) to a callable f(qubit)
that returns the 5-gate sequence [Rz(a), X90, Rz(b), X90, Rz(c)] in
application order (first gate applied first to the qubit state). The effective
unitary is therefore:

    U = Rz(c) · X90 · Rz(b) · X90 · Rz(a)   (up to global phase)

Derivation: normalizing U to SU(2) via U' = U/sqrt(det(U)), the elements of
U' satisfy:

    U'[0,0] = -i · exp(-i(a+c)/2) · sin(b/2)
    U'[0,1] = -i · exp( i(a-c)/2) · cos(b/2)

from which a, b, c are extracted directly.
"""
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from qcal.circuit import Circuit, Cycle
from qcal.gates.single_qubit import (
    SX,
    SY,
    X90,
    Y90,
    Z90,
    C,
    Cliff0,
    Cliff1,
    Cliff2,
    Cliff3,
    Cliff4,
    Cliff5,
    Cliff6,
    Cliff7,
    Cliff8,
    Cliff9,
    Cliff10,
    Cliff11,
    Cliff12,
    Cliff13,
    Cliff14,
    Cliff15,
    Cliff16,
    Cliff17,
    Cliff18,
    Cliff19,
    Cliff20,
    Cliff21,
    Cliff22,
    Cliff23,
    H,
    Id,
    Rz,
    S,
    Sdag,
    SXdag,
    SYdag,
    T,
    Tdag,
    V,
    Vdag,
    X,
    Y,
    Z,
    h,
    rn,
    rx,
    ry,
    rz,
    t,
    tdag,
    x,
    y,
    z,
)

__all__ = (
    'ZXZXZ_DECOMPOSITIONS', 'unitary_to_zxzxz', 'pauli_to_cycle'
)

PauliString = tuple[str, ...]


def unitary_to_zxzxz(U: NDArray) -> tuple[float, float, float]:
    """Return (a, b, c) for [Rz(a), X90, Rz(b), X90, Rz(c)] in order.

    The effective unitary is Rz(c)·X90·Rz(b)·X90·Rz(a) = U up to global phase.

    Args:
        U (NDArray): 2x2 unitary matrix.

    Returns:
        tuple[float, float, float]: angles (a, b, c) in radians
    """
    # Cast to complex so sqrt(-1) works for real matrices like X, H, Z
    d = complex(np.linalg.det(U))
    U = U / np.sqrt(d)
    u00, u01 = U[0, 0], U[0, 1]
    b = 2.0 * np.arctan2(abs(u00), abs(u01))
    if abs(u00) < 1e-10:
        # b ≈ 0: only a−c is determined; set c = 0
        a, c = 2.0 * np.angle(u01) + np.pi, 0.0
    elif abs(u01) < 1e-10:
        # b ≈ π: only a+c is determined; set a = 0
        a, c = 0.0, -2.0 * np.angle(u00) - np.pi
    else:
        a = np.angle(u01) - np.angle(u00)
        c = -np.angle(u00) - np.angle(u01) - np.pi
    return float(np.real(a)), float(np.real(b)), float(np.real(c))


def _decomp(matrix: NDArray):
    """Return f(qubit) -> [Rz(a), X90, Rz(b), X90, Rz(c)] for the given unitary.

    Args:
        matrix (NDArray): 2x2 unitary matrix.

    Returns:
        callable: f(qubit) -> list of 5 gates implementing the unitary.
    """
    a, b, c = unitary_to_zxzxz(matrix)

    def _factory(qubit: int) -> list:
        return [
            Rz(qubit, a), X90(qubit), Rz(qubit, b), X90(qubit), Rz(qubit, c)
        ]

    return _factory


@lru_cache(maxsize=None)
def decompose_to_zxzxz(label: str, qubit: int) -> tuple:
    """Return the cached ZXZXZ gate decomposition for one (label, qubit).

    Cache domain is naturally bounded: at most len(ZXZXZ_DECOMPOSITIONS)
    distinct labels times the number of qubits on the device, so every
    reachable (label, qubit) pair can be cached without unbounded
    growth. This turns per-qubit gate construction (Rz/X90 matrix
    building, Gate hashing, etc.) into a one-time cost per pair,
    regardless of how many callers or circuits request it.

    Args:
        label (str): single-qubit gate label, e.g. 'X' or 'Cliff5'.
        qubit (int): qubit label.

    Returns:
        tuple: the 5-gate [Rz, X90, Rz, X90, Rz] decomposition, in
            application order.
    """
    return tuple(ZXZXZ_DECOMPOSITIONS[label](qubit))


ZXZXZ_DECOMPOSITIONS: dict[str | type, callable] = {
    'C':       _decomp(rn(2 * np.pi / 3, np.array([1., 1., 1.]) / np.sqrt(3))),
    'Cliff0':  _decomp(np.eye(2)),
    'Cliff1':  _decomp(x),
    'Cliff2':  _decomp(y),
    'Cliff3':  _decomp(z),
    'Cliff4':  _decomp(rx(np.pi / 2)),
    'Cliff5':  _decomp(rx(-np.pi / 2)),
    'Cliff6':  _decomp(ry(np.pi / 2)),
    'Cliff7':  _decomp(ry(-np.pi / 2)),
    'Cliff8':  _decomp(rz(np.pi / 2)),
    'Cliff9':  _decomp(rz(-np.pi / 2)),
    'Cliff10': _decomp(rn(np.pi, np.array([1., 1., 0.]) / np.sqrt(2))),
    'Cliff11': _decomp(rn(np.pi, np.array([1., -1., 0.]) / np.sqrt(2))),
    'Cliff12': _decomp(rn(np.pi, np.array([0., 1., 1.]) / np.sqrt(2))),
    'Cliff13': _decomp(rn(np.pi, np.array([0., 1., -1.]) / np.sqrt(2))),
    'Cliff14': _decomp(h),
    'Cliff15': _decomp(rn(np.pi, np.array([1., 0., -1.]) / np.sqrt(2))),
    'Cliff16': _decomp(
        rn(2 * np.pi / 3, np.array([1., 1., 1.]) / np.sqrt(3))
    ),
    'Cliff17': _decomp(
        rn(-2 * np.pi / 3, np.array([1., 1., 1.]) / np.sqrt(3))
    ),
    'Cliff18': _decomp(
        rn(2 * np.pi / 3, np.array([1., 1., -1.]) / np.sqrt(3))
    ),
    'Cliff19': _decomp(
        rn(-2 * np.pi / 3, np.array([1., 1., -1.]) / np.sqrt(3))
    ),
    'Cliff20': _decomp(
        rn(2 * np.pi / 3, np.array([1., -1., 1.]) / np.sqrt(3))
    ),
    'Cliff21': _decomp(
        rn(-2 * np.pi / 3, np.array([1., -1., 1.]) / np.sqrt(3))
    ),
    'Cliff22': _decomp(
        rn(2 * np.pi / 3, np.array([-1., 1., 1.]) / np.sqrt(3))
    ),
    'Cliff23': _decomp(
        rn(-2 * np.pi / 3, np.array([-1., 1., 1.]) / np.sqrt(3))
    ),
    'H':       _decomp(h),
    'I':       _decomp(np.eye(2)),
    'Id':      _decomp(np.eye(2)),
    'S':       _decomp(rz(np.pi / 2)),
    'Sdag':    _decomp(rz(-np.pi / 2)),
    'SX':      _decomp(rx(np.pi / 2)),
    'SXdag':   _decomp(rx(-np.pi / 2)),
    'SY':      _decomp(ry(np.pi / 2)),
    'SYdag':   _decomp(ry(-np.pi / 2)),
    'T':       _decomp(t),
    'Tdag':    _decomp(tdag),
    'V':       _decomp(rx(np.pi / 2)),
    'Vdag':    _decomp(rx(-np.pi / 2)),
    'X':       _decomp(x),
    'X90':     _decomp(rx(np.pi / 2)),
    'Y':       _decomp(y),
    'Y90':     _decomp(ry(np.pi / 2)),
    'Z':       _decomp(z),
    'Z90':     _decomp(rz(np.pi / 2)),
}

# Extend with gate class keys pointing to the same factories as the string keys
ZXZXZ_DECOMPOSITIONS.update({
    C:       ZXZXZ_DECOMPOSITIONS['C'],
    Cliff0:  ZXZXZ_DECOMPOSITIONS['Cliff0'],
    Cliff1:  ZXZXZ_DECOMPOSITIONS['Cliff1'],
    Cliff2:  ZXZXZ_DECOMPOSITIONS['Cliff2'],
    Cliff3:  ZXZXZ_DECOMPOSITIONS['Cliff3'],
    Cliff4:  ZXZXZ_DECOMPOSITIONS['Cliff4'],
    Cliff5:  ZXZXZ_DECOMPOSITIONS['Cliff5'],
    Cliff6:  ZXZXZ_DECOMPOSITIONS['Cliff6'],
    Cliff7:  ZXZXZ_DECOMPOSITIONS['Cliff7'],
    Cliff8:  ZXZXZ_DECOMPOSITIONS['Cliff8'],
    Cliff9:  ZXZXZ_DECOMPOSITIONS['Cliff9'],
    Cliff10: ZXZXZ_DECOMPOSITIONS['Cliff10'],
    Cliff11: ZXZXZ_DECOMPOSITIONS['Cliff11'],
    Cliff12: ZXZXZ_DECOMPOSITIONS['Cliff12'],
    Cliff13: ZXZXZ_DECOMPOSITIONS['Cliff13'],
    Cliff14: ZXZXZ_DECOMPOSITIONS['Cliff14'],
    Cliff15: ZXZXZ_DECOMPOSITIONS['Cliff15'],
    Cliff16: ZXZXZ_DECOMPOSITIONS['Cliff16'],
    Cliff17: ZXZXZ_DECOMPOSITIONS['Cliff17'],
    Cliff18: ZXZXZ_DECOMPOSITIONS['Cliff18'],
    Cliff19: ZXZXZ_DECOMPOSITIONS['Cliff19'],
    Cliff20: ZXZXZ_DECOMPOSITIONS['Cliff20'],
    Cliff21: ZXZXZ_DECOMPOSITIONS['Cliff21'],
    Cliff22: ZXZXZ_DECOMPOSITIONS['Cliff22'],
    Cliff23: ZXZXZ_DECOMPOSITIONS['Cliff23'],
    H:       ZXZXZ_DECOMPOSITIONS['H'],
    Id:      ZXZXZ_DECOMPOSITIONS['Id'],
    S:       ZXZXZ_DECOMPOSITIONS['S'],
    Sdag:    ZXZXZ_DECOMPOSITIONS['Sdag'],
    SX:      ZXZXZ_DECOMPOSITIONS['SX'],
    SXdag:   ZXZXZ_DECOMPOSITIONS['SXdag'],
    SY:      ZXZXZ_DECOMPOSITIONS['SY'],
    SYdag:   ZXZXZ_DECOMPOSITIONS['SYdag'],
    T:       ZXZXZ_DECOMPOSITIONS['T'],
    Tdag:    ZXZXZ_DECOMPOSITIONS['Tdag'],
    V:       ZXZXZ_DECOMPOSITIONS['V'],
    Vdag:    ZXZXZ_DECOMPOSITIONS['Vdag'],
    X:       ZXZXZ_DECOMPOSITIONS['X'],
    X90:     ZXZXZ_DECOMPOSITIONS['X90'],
    Y:       ZXZXZ_DECOMPOSITIONS['Y'],
    Y90:     ZXZXZ_DECOMPOSITIONS['Y90'],
    Z:       ZXZXZ_DECOMPOSITIONS['Z'],
    Z90:     ZXZXZ_DECOMPOSITIONS['Z90'],
})


# Local alias so `pauli_to_cycle`'s `decompose_to_zxzxz` parameter can
# shadow the module-level function name without losing the reference.
_zxzxz_gates = decompose_to_zxzxz


@lru_cache(maxsize=1024)
def pauli_to_cycle(
        pauli: PauliString, qubits: tuple, decompose_to_zxzxz: bool = False
) -> Circuit:
    """Convert a PauliString to a Cycle (or subcircuit) of Pauli gates.

    Cached (bounded): the number of distinct n-qubit Pauli strings is 4^n,
    so for small n (few-qubit benchmarked cycles) the same twirl/prep
    pattern recurs often across randomizations/depths and caching avoids
    rebuilding it; for large n the cache simply stays capped at `maxsize`
    (a fixed, bounded footprint) rather than growing without bound. The
    returned Circuit is shared across cache hits rather than copied: callers
    only ever `.extend()` it (never mutate its cycles in place).

    Args:
        pauli (PauliString): tuple of single-qubit Pauli labels,
            e.g. ('X', 'Z', 'I').
        qubits (tuple): qubit labels corresponding to each position in
            pauli.
        decompose_to_zxzxz (bool): whether to decompose all single-qubit gates
            to ZXZXZ decomposition. Defaults to False. Setting to True can be
            useful when implementing CB using hardware-efficient randomization.

    Returns:
        Circuit: a Circuit containing a Cycle of I/X/Y/Z gate for each
            non-identity entry of pauli, or a Circuit containing the ZXZXZ
            decomposition of each Pauli gate.
    """
    circuit = Circuit()
    if decompose_to_zxzxz:
        # Each qubit's decomposition is the same fixed-length [Rz, X90, Rz,
        # X90, Rz] sequence, so transposing across qubits and appending one
        # shared Cycle per step gives the same result as `.join()`-ing each
        # qubit's sub-circuit in one at a time, without `.join()`'s
        # per-call rebuild of every existing Cycle.
        gate_lists = [
            _zxzxz_gates(p, q)
            for p, q in zip(pauli, qubits, strict=True)
        ]
        for gates_at_step in zip(*gate_lists, strict=True):
            circuit.append(Cycle(gates_at_step))

    else:
        cycle = Cycle()
        for p, q in zip(pauli, qubits, strict=True):
            match p:
                case 'I':
                    cycle.append(Id(q))
                case 'X':
                    cycle.append(X(q))
                case 'Y':
                    cycle.append(Y(q))
                case 'Z':
                    cycle.append(Z(q))

        circuit.append(cycle)

    return circuit
