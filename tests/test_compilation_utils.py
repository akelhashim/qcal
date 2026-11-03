"""Unit tests for qcal.compilation.utils.composes_to_identity.

Every expectation is cross-checked against a brute-force test on the
full unitary (U^depth proportional to the identity), so the
Pauli-conjugation shortcut is validated against ground truth rather
than against hand-written booleans alone.
"""
import numpy as np
import pytest

from qcal.circuit import Barrier, Circuit, Cycle
from qcal.compilation.merge import merge_cycles
from qcal.compilation.utils import composes_to_identity
from qcal.gates.single_qubit import H, Id, Meas, S, X, Y, Z
from qcal.gates.two_qubit import CNOT, CZ, SWAP, iSWAP


def _is_identity_up_to_phase(U):
    phase = U[0, 0]
    return np.isclose(abs(phase), 1) and np.allclose(
        U, phase * np.eye(U.shape[0])
    )


def _brute_force(cycle_or_circuit, depth):
    U = cycle_or_circuit.unitary
    return bool(
        _is_identity_up_to_phase(np.linalg.matrix_power(U, depth))
    )


CASES = [
    ('X', Cycle(X(0)), 1, False),
    ('X', Cycle(X(0)), 2, True),
    ('Z sign only', Cycle(Z(0)), 1, False),
    ('Z', Cycle(Z(0)), 2, True),
    ('H', Cycle(H(0)), 1, False),
    ('H', Cycle(H(0)), 2, True),
    ('S', Cycle(S(0)), 2, False),
    ('S', Cycle(S(0)), 4, True),
    ('Id', Cycle(Id(0)), 1, True),
    ('XY', Cycle([X(0), Y(1)]), 1, False),
    ('XY', Cycle([X(0), Y(1)]), 2, True),
    ('XH', Cycle([X(0), H(1)]), 3, False),
    ('CNOT', Cycle(CNOT(0, 1)), 1, False),
    ('CNOT', Cycle(CNOT(0, 1)), 2, True),
    ('CZ', Cycle(CZ(0, 1)), 2, True),
    ('SWAP', Cycle(SWAP(0, 1)), 2, True),
    ('iSWAP', Cycle(iSWAP(0, 1)), 2, False),
    ('iSWAP', Cycle(iSWAP(0, 1)), 4, True),
    ('HS circuit', Circuit([Cycle(H(0)), Cycle(S(0))]), 1, False),
    ('HS circuit', Circuit([Cycle(H(0)), Cycle(S(0))]), 2, False),
    ('HS circuit', Circuit([Cycle(H(0)), Cycle(S(0))]), 3, True),
    (
        'H-CNOT circuit',
        Circuit([Cycle([H(0), H(1)]), Cycle(CNOT(0, 1))]),
        2, False,
    ),
    (
        'H-CNOT circuit',
        Circuit([Cycle([H(0), H(1)]), Cycle(CNOT(0, 1))]),
        3, False,
    ),
    (
        'H-CNOT circuit',
        Circuit([Cycle([H(0), H(1)]), Cycle(CNOT(0, 1))]),
        6, True,
    ),
]


class TestComposesToIdentity:

    @pytest.mark.parametrize(
        'label, obj, depth, expected', CASES,
        ids=[f'{c[0]}-d{c[2]}' for c in CASES],
    )
    def test_cases(self, label, obj, depth, expected):
        # Sanity-check the expectation itself against the full unitary.
        assert _brute_force(obj, depth) is expected
        assert composes_to_identity(obj, depth) is expected

    def test_depth_zero_is_identity(self):
        assert composes_to_identity(Cycle(H(0)), 0) is True

    def test_sign_matters_not_just_pauli_string(self):
        # Z maps X -> -X: the string is preserved but the sign is not,
        # so Z alone is not the identity.
        assert composes_to_identity(Cycle(Z(0)), 1) is False

    def test_circuit_with_barrier(self):
        circuit = Circuit([Cycle(X(0)), Barrier((0,)), Cycle(X(0))])
        assert composes_to_identity(circuit, 1) is True

    def test_measurement_treated_as_identity(self):
        assert composes_to_identity(Cycle(Meas(0)), 1) is True

    def test_non_clifford_raises(self):
        from qcal.gates.single_qubit import T
        with pytest.raises(ValueError):
            composes_to_identity(Cycle(T(0)), 1)

    def test_merged_cycle_identity(self):
        # Unique qubit labels: merged generic gates collide in the
        # module's lru_caches with any other merge on the same qubits
        # (see the xfail test below).
        merged = merge_cycles(
            Cycle([H(80), S(81)]), Cycle([H(80), S(81)])
        )
        # H^2 = I on qubit 0, S^2 = Z on qubit 1.
        assert composes_to_identity(merged, 1) is False
        assert composes_to_identity(merged, 2) is True

    @pytest.mark.xfail(
        reason=(
            'composes_to_identity (and conjugate_pauli*, which it calls) '
            'are lru_cached on Cycle/Gate, whose hash/eq ignore the '
            'matrix of generic merged gates; two merged cycles on the '
            'same qubits collide and the first cached answer is reused.'
        ),
        strict=True,
    )
    def test_merged_cycles_with_different_matrices_not_conflated(self):
        # Both are Cycle(Gate(<matrix>, (93,))) and compare equal.
        is_x = merge_cycles(Cycle(X(93)), Cycle(Id(93)))
        is_identity = merge_cycles(Cycle(X(93)), Cycle(X(93)))
        assert is_x == is_identity
        assert composes_to_identity(is_x, 1) is False
        assert composes_to_identity(is_identity, 1) is True
