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
import numpy as np
from numpy.typing import NDArray

from qcal.gate.single_qubit import (
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


def _zxzxz_angles(U: NDArray) -> tuple[float, float, float]:
    """Return (a, b, c) for [Rz(a), X90, Rz(b), X90, Rz(c)] in order.

    The effective unitary is Rz(c)·X90·Rz(b)·X90·Rz(a) = U up to global phase.
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
    """
    a, b, c = _zxzxz_angles(matrix)

    def _factory(qubit: int) -> list:
        return [
            Rz(qubit, a), X90(qubit), Rz(qubit, b), X90(qubit), Rz(qubit, c)
        ]

    return _factory


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

__all__ = ('ZXZXZ_DECOMPOSITIONS',)
