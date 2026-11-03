"""Unit tests for qcal.compilation.pauli_conjugation.

Checks the hardcoded single-/two-qubit Clifford conjugation tables
against direct matrix conjugation U P U^dagger (so a typo in any table
entry is caught), a handful of hand-derived identities (H X H = Z,
CNOT: XI -> XX, ...), the tensor-factor convention (the first Pauli
label acts on the first qubit of the gate / cycle), Pauli-string
identification, and Cycle/Circuit-level conjugation.

Convention (verified numerically): `CNOT(0, 1).unitary` has its control
on the first tensor factor, and conjugating by a Circuit applies its
cycles in order, i.e. U = U_last ... U_first.
"""
from functools import reduce

import numpy as np
import pytest

import qcal.gates.single_qubit as sq
import qcal.gates.two_qubit as tq
from qcal.circuit import Barrier, Circuit, Cycle
from qcal.compilation.merge import merge_gates
from qcal.compilation.pauli_conjugation import (
    ONE_QUBIT_PAULIS,
    SINGLE_QUBIT_PAULI_CONJUGATION_TABLE,
    TWO_QUBIT_PAULI_CONJUGATION_TABLE,
    TWO_QUBIT_PAULIS,
    conjugate_pauli,
    conjugate_pauli_by_cycle,
    conjugate_pauli_by_gate,
    conjugate_pauli_by_unitary,
    get_pauli_conjugation,
    identify_pauli_str,
    pauli_str_to_matrix,
)
from qcal.gates.gate import Gate
from qcal.gates.single_qubit import H, Meas, S, X, Y, Z
from qcal.gates.two_qubit import CNOT, CZ

PAULI_MATS = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli(labels):
    return reduce(np.kron, [PAULI_MATS[p] for p in labels])


def _conj(U, labels):
    return U @ _pauli(labels) @ U.conj().T


def _embed(gates, qubits):
    """Full unitary of a cycle of single-qubit gates over `qubits`."""
    by_q = {g.qubits[0]: g.unitary for g in gates}
    return reduce(np.kron, [by_q.get(q, np.eye(2)) for q in qubits])


class TestConstants:

    def test_one_qubit_paulis(self):
        assert ONE_QUBIT_PAULIS == ('I', 'X', 'Y', 'Z')

    def test_two_qubit_paulis(self):
        assert len(TWO_QUBIT_PAULIS) == 16
        assert len(set(TWO_QUBIT_PAULIS)) == 16
        assert TWO_QUBIT_PAULIS[0] == ('I', 'I')
        assert ('X', 'Z') in TWO_QUBIT_PAULIS

    def test_every_table_entry_is_complete(self):
        for table in SINGLE_QUBIT_PAULI_CONJUGATION_TABLE.values():
            assert set(table) == set(ONE_QUBIT_PAULIS)
        for table in TWO_QUBIT_PAULI_CONJUGATION_TABLE.values():
            assert set(table) == set(TWO_QUBIT_PAULIS)


class TestTablesMatchMatrices:

    @pytest.mark.parametrize(
        'name', sorted(SINGLE_QUBIT_PAULI_CONJUGATION_TABLE)
    )
    def test_single_qubit_table(self, name):
        U = getattr(sq, name)(0).unitary
        for p, (q, sign) in SINGLE_QUBIT_PAULI_CONJUGATION_TABLE[
            name
        ].items():
            np.testing.assert_allclose(
                _conj(U, p), sign * _pauli(q), atol=1e-12,
                err_msg=f'{name}: {p} -> {sign}{q}',
            )

    @pytest.mark.parametrize(
        'name', sorted(TWO_QUBIT_PAULI_CONJUGATION_TABLE)
    )
    def test_two_qubit_table(self, name):
        U = getattr(tq, name)(0, 1).unitary
        for p, (q, sign) in TWO_QUBIT_PAULI_CONJUGATION_TABLE[
            name
        ].items():
            np.testing.assert_allclose(
                _conj(U, p), sign * _pauli(q), atol=1e-12,
                err_msg=f'{name}: {p} -> {sign}{q}',
            )

    def test_signs_are_plus_minus_one(self):
        for tables in (
            SINGLE_QUBIT_PAULI_CONJUGATION_TABLE,
            TWO_QUBIT_PAULI_CONJUGATION_TABLE,
        ):
            for table in tables.values():
                assert {s for _, s in table.values()} <= {1, -1}

    def test_identity_always_maps_to_identity(self):
        for table in SINGLE_QUBIT_PAULI_CONJUGATION_TABLE.values():
            assert table['I'] == ('I', 1)
        for table in TWO_QUBIT_PAULI_CONJUGATION_TABLE.values():
            assert table[('I', 'I')] == (('I', 'I'), 1)


class TestHandDerived:

    def test_h_x_h_is_z(self):
        assert conjugate_pauli_by_gate(('X',), H(0)) == (('Z',), 1)

    def test_h_y_h_is_minus_y(self):
        assert conjugate_pauli_by_gate(('Y',), H(0)) == (('Y',), -1)

    def test_s_x_sdag_is_y(self):
        assert conjugate_pauli_by_gate(('X',), S(0)) == (('Y',), 1)

    def test_x_flips_z_sign(self):
        assert conjugate_pauli_by_gate(('Z',), X(0)) == (('Z',), -1)

    def test_cnot_xi_is_xx(self):
        # X on the control spreads to the target.
        assert conjugate_pauli_by_gate(('X', 'I'), CNOT(0, 1)) == (
            ('X', 'X'), 1
        )

    def test_cnot_iz_is_zz(self):
        # Z on the target spreads back to the control.
        assert conjugate_pauli_by_gate(('I', 'Z'), CNOT(0, 1)) == (
            ('Z', 'Z'), 1
        )

    def test_cnot_reversed_control(self):
        # CNOT(1, 0): the first factor (qubit 1) is the control.
        assert conjugate_pauli_by_gate(('X', 'I'), CNOT(1, 0)) == (
            ('X', 'X'), 1
        )

    def test_cz_xi_is_xz(self):
        assert conjugate_pauli_by_gate(('X', 'I'), CZ(0, 1)) == (
            ('X', 'Z'), 1
        )


class TestGetPauliConjugation:

    def test_two_qubit_by_name_and_string(self):
        assert get_pauli_conjugation('CNOT', 'XI') == (('X', 'X'), 1)

    def test_two_qubit_by_gate_and_tuple(self):
        assert get_pauli_conjugation(CNOT(0, 1), ('X', 'I')) == (
            ('X', 'X'), 1
        )

    def test_single_qubit_by_string(self):
        assert get_pauli_conjugation('H', 'X') == ('Z', 1)

    def test_single_qubit_by_tuple(self):
        assert get_pauli_conjugation('H', ('X',)) == ('Z', 1)

    def test_untabulated_returns_none(self):
        assert get_pauli_conjugation('NotAGate', 'X') is None
        assert get_pauli_conjugation(sq.T(0), 'X') is None

    def test_wrong_length_for_two_qubit_raises(self):
        with pytest.raises(ValueError):
            get_pauli_conjugation('CNOT', 'X')

    def test_invalid_label_for_two_qubit_raises(self):
        with pytest.raises(ValueError):
            get_pauli_conjugation('CNOT', 'XA')

    def test_wrong_length_for_single_qubit_raises(self):
        with pytest.raises(ValueError):
            get_pauli_conjugation('H', 'XX')


class TestPauliStrToMatrix:

    @pytest.mark.parametrize('label', ['I', 'X', 'Y', 'Z'])
    def test_single(self, label):
        np.testing.assert_allclose(
            pauli_str_to_matrix((label,)), PAULI_MATS[label]
        )

    def test_kron_order_first_label_is_first_factor(self):
        np.testing.assert_allclose(
            pauli_str_to_matrix(('X', 'Z')),
            np.kron(PAULI_MATS['X'], PAULI_MATS['Z']),
        )

    def test_three_qubits(self):
        labels = ('Y', 'I', 'Z')
        M = pauli_str_to_matrix(labels)
        assert M.shape == (8, 8)
        np.testing.assert_allclose(M, _pauli(labels))


class TestIdentifyPauliStr:

    @pytest.mark.parametrize('labels', [
        ('I',), ('X',), ('Y',), ('Z',),
        ('X', 'Z'), ('Z', 'X'), ('Y', 'I'), ('I', 'Y'),
        ('X', 'Y', 'Z'), ('Z', 'I', 'Y'),
    ])
    def test_round_trip(self, labels):
        assert identify_pauli_str(_pauli(labels), len(labels)) == labels

    @pytest.mark.parametrize('phase', [1, -1, 1j, -1j, np.exp(0.3j)])
    def test_ignores_phase(self, phase):
        labels = ('Y', 'Z')
        assert identify_pauli_str(phase * _pauli(labels), 2) == labels


class TestConjugatePauliByUnitary:

    def test_returns_scalar(self):
        new, scalar = conjugate_pauli_by_unitary(sq.h, ('Y',))
        assert new == ('Y',)
        assert np.isclose(scalar, -1)

    def test_scalar_reconstructs_conjugate(self):
        U = tq.iSWAP(0, 1).unitary
        for p in TWO_QUBIT_PAULIS:
            new, scalar = conjugate_pauli_by_unitary(U, p)
            np.testing.assert_allclose(
                _conj(U, p), scalar * _pauli(new), atol=1e-12
            )


class TestConjugatePauliByGate:

    def test_single_label_wrapped_as_tuple(self):
        result, _ = conjugate_pauli_by_gate(('X',), H(0))
        assert result == ('Z',)

    def test_measurement_is_identity(self):
        for p in ONE_QUBIT_PAULIS:
            assert conjugate_pauli_by_gate((p,), Meas(0)) == ((p,), 1)

    def test_untabulated_clifford_falls_back_to_unitary(self):
        # A 3-qubit gate is never tabulated: CNOT(0, 1) x I(2).
        U = np.kron(CNOT(0, 1).unitary, np.eye(2))
        gate = Gate(U, (10, 11, 12))
        for p in [('X', 'I', 'I'), ('I', 'Z', 'Y'), ('Y', 'Y', 'X')]:
            new, sign = conjugate_pauli_by_gate(p, gate)
            np.testing.assert_allclose(
                _conj(U, p), sign * _pauli(new), atol=1e-12
            )

    def test_non_clifford_raises(self):
        with pytest.raises(ValueError, match='Clifford'):
            conjugate_pauli_by_gate(('X',), sq.T(0))

    @pytest.mark.xfail(
        reason=(
            'conjugate_pauli_by_unitary only checks one matrix entry '
            'and identify_pauli_str never validates its input, so some '
            'non-Clifford gates (e.g. CS on IY) are silently accepted '
            'with a wrong (Pauli, sign) instead of raising ValueError.'
        ),
        strict=True,
    )
    def test_non_clifford_two_qubit_raises(self):
        with pytest.raises(ValueError):
            conjugate_pauli_by_gate(('I', 'Y'), tq.CS(0, 1))

    @pytest.mark.xfail(
        reason=(
            'conjugate_pauli_by_gate is lru_cached on Gate, whose hash/eq '
            'ignore the matrix; two generic Gates on the same qubits '
            '(e.g. outputs of merge_gates) collide in the cache.'
        ),
        strict=True,
    )
    def test_generic_gates_with_different_matrices_not_conflated(self):
        conjugate_pauli_by_gate.cache_clear()
        hadamard = Gate(sq.h, 91)
        identity = Gate(np.eye(2, dtype=complex), 91)
        assert conjugate_pauli_by_gate(('X',), hadamard) == (('Z',), 1)
        assert conjugate_pauli_by_gate(('X',), identity) == (('X',), 1)


class TestConjugatePauliByCycle:

    def test_local_conjugation_on_each_qubit(self):
        cycle = Cycle([H(0), S(1)])
        assert conjugate_pauli_by_cycle(('X', 'X'), (0, 1), cycle) == (
            ('Z', 'Y'), 1
        )

    def test_signs_multiply(self):
        cycle = Cycle([H(0), X(1)])
        # H Y H = -Y, X Z X = -Z -> overall +1.
        assert conjugate_pauli_by_cycle(('Y', 'Z'), (0, 1), cycle) == (
            ('Y', 'Z'), 1
        )

    def test_qubits_not_in_cycle_untouched(self):
        cycle = Cycle(H(1))
        assert conjugate_pauli_by_cycle(
            ('Y', 'X', 'Z'), (0, 1, 2), cycle
        ) == (('Y', 'Z', 'Z'), 1)

    def test_qubit_order_from_qubits_argument(self):
        cycle = Cycle([H(5), S(2)])
        # qubits (5, 2): position 0 -> qubit 5 (H), position 1 -> 2 (S).
        assert conjugate_pauli_by_cycle(('X', 'X'), (5, 2), cycle) == (
            ('Z', 'Y'), 1
        )

    def test_two_qubit_gate_with_reversed_qubits(self):
        cycle = Cycle(CNOT(1, 0))
        # qubits (0, 1): X on qubit 1 (the control) spreads to qubit 0.
        assert conjugate_pauli_by_cycle(('I', 'X'), (0, 1), cycle) == (
            ('X', 'X'), 1
        )


class TestConjugatePauli:

    def test_cycle_uses_sorted_qubits(self):
        cycle = Cycle([H(5), S(2)])
        # cycle.qubits == (2, 5): position 0 -> S, position 1 -> H.
        assert conjugate_pauli(('X', 'X'), cycle) == (('Y', 'Z'), 1)

    def test_circuit_matches_full_unitary(self):
        circuit = Circuit([
            Cycle([H(0), S(1)]),
            Cycle(CNOT(0, 1)),
            Cycle([S(0), H(1)]),
            Cycle(CZ(0, 1)),
        ])
        U = circuit.unitary
        for p in TWO_QUBIT_PAULIS:
            new, sign = conjugate_pauli(p, circuit)
            np.testing.assert_allclose(
                _conj(U, p), sign * _pauli(new), atol=1e-12,
                err_msg=f'{p}',
            )

    def test_circuit_applies_cycles_in_order(self):
        # H then S: S H X H S^dag = S Z S^dag = Z.
        # (S then H would give H Y H = -Y.)
        circuit = Circuit([Cycle(H(0)), Cycle(S(0))])
        assert conjugate_pauli(('X',), circuit) == (('Z',), 1)
        reversed_circuit = Circuit([Cycle(S(0)), Cycle(H(0))])
        assert conjugate_pauli(('X',), reversed_circuit) == (('Y',), -1)

    def test_circuit_skips_barriers(self):
        circuit = Circuit([Cycle(H(0)), Barrier((0,)), Cycle(S(0))])
        assert conjugate_pauli(('X',), circuit) == (('Z',), 1)

    def test_three_qubit_circuit(self):
        qubits = (0, 1, 2)
        c1 = Cycle([H(0), Y(1), S(2)])
        c2 = Cycle([CNOT(1, 2), Z(0)])
        circuit = Circuit([c1, c2])
        U1 = _embed(c1.gates, qubits)
        U2 = np.kron(Z(0).unitary, CNOT(1, 2).unitary)
        U = U2 @ U1
        for p in [('X', 'Y', 'Z'), ('Z', 'X', 'I'), ('I', 'I', 'X')]:
            new, sign = conjugate_pauli(p, circuit)
            np.testing.assert_allclose(
                _conj(U, p), sign * _pauli(new), atol=1e-12
            )

    def test_merged_generic_gate(self):
        # Unique qubit label to avoid the generic-Gate cache collision
        # covered by the xfail test above.
        merged = merge_gates(H(82), S(82))
        new, sign = conjugate_pauli(('X',), Cycle(merged))
        np.testing.assert_allclose(
            _conj(merged.unitary, ('X',)), sign * _pauli(new), atol=1e-12
        )
