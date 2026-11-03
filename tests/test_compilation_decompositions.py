"""Unit tests for qcal.compilation.decompositions.

Covers the ZXZXZ (Rz, X90, Rz, X90, Rz) single-qubit decomposition:
`unitary_to_zxzxz` / `zxzxz_to_unitary` round trips (up to global
phase), the `ZXZXZ_DECOMPOSITIONS` dispatch table, the cached
`decompose_to_zxzxz`, `decompose_cycle`, and `pauli_to_cycle`.

Convention (verified numerically): the angles (a, b, c) are applied in
order, so the effective unitary is Rz(c) @ X90 @ Rz(b) @ X90 @ Rz(a).
"""
from functools import reduce

import numpy as np
import pytest
from scipy.stats import unitary_group

import qcal.gates.single_qubit as sq
from qcal.circuit import Circuit, Cycle
from qcal.compilation.decompositions import (
    ZXZXZ_DECOMPOSITIONS,
    decompose_cycle,
    decompose_to_zxzxz,
    pauli_to_cycle,
    unitary_to_zxzxz,
    zxzxz_to_unitary,
)
from qcal.compilation.merge import merge_cycles, merge_gates
from qcal.gates.single_qubit import X90, H, Rz, S, X, Y, Z

NAMED_GATES = (
    'Id', 'X', 'Y', 'Z', 'H', 'S', 'Sdag', 'T', 'Tdag', 'X90', 'Y90',
    'Z90', 'SX', 'SXdag', 'SY', 'SYdag', 'V', 'Vdag', 'C',
) + tuple(f'Cliff{i}' for i in range(24))


def _equal_up_to_phase(a, b, atol=1e-8):
    idx = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    phase = b[idx] / a[idx]
    return np.isclose(abs(phase), 1, atol=atol) and np.allclose(
        a * phase, b, atol=atol
    )


def _gate_sequence_unitary(gates):
    return reduce(lambda u, g: g.unitary @ u, gates, np.eye(2))


def _qubit_unitary(circuit, qubit):
    """Product of a circuit's gates on one qubit, in application order."""
    gates = [g for cycle in circuit for g in cycle if g.qubits == (qubit,)]
    return _gate_sequence_unitary(gates)


class TestZxzxzToUnitary:

    def test_matches_explicit_product(self):
        a, b, c = 0.3, -1.2, 2.1
        expected = (
            sq.rz(c) @ sq.rx(np.pi / 2) @ sq.rz(b) @ sq.rx(np.pi / 2)
            @ sq.rz(a)
        )
        np.testing.assert_allclose(
            zxzxz_to_unitary((a, b, c)), expected, atol=1e-12
        )

    def test_matches_x90_gate_class(self):
        # X90's matrix agrees with rx(pi/2) up to phase; the explicit
        # product with the gate classes must match exactly.
        a, b, c = 0.5, 0.6, 0.7
        expected = (
            Rz(0, c).unitary @ X90(0).unitary @ Rz(0, b).unitary
            @ X90(0).unitary @ Rz(0, a).unitary
        )
        np.testing.assert_allclose(
            zxzxz_to_unitary((a, b, c)), expected, atol=1e-12
        )

    def test_is_unitary(self):
        U = zxzxz_to_unitary((0.1, 0.2, 0.3))
        np.testing.assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-12)

    def test_zero_middle_angle_is_x(self):
        # X90 @ X90 = X (up to phase) when a = b = c = 0.
        assert _equal_up_to_phase(zxzxz_to_unitary((0., 0., 0.)), sq.x)


class TestUnitaryToZxzxz:

    @pytest.mark.parametrize('seed', range(25))
    def test_round_trip_random_unitaries(self, seed):
        U = unitary_group.rvs(2, random_state=seed)
        angles = unitary_to_zxzxz(U)
        assert _equal_up_to_phase(zxzxz_to_unitary(angles), U)

    @pytest.mark.parametrize('name', NAMED_GATES)
    def test_round_trip_named_gates(self, name):
        U = getattr(sq, name)(0).unitary
        assert _equal_up_to_phase(
            zxzxz_to_unitary(unitary_to_zxzxz(U)), U
        )

    @pytest.mark.parametrize('theta', [0.0, 0.1, -2.0, np.pi, 3.0])
    def test_round_trip_diagonal_rz(self, theta):
        # Diagonal unitaries hit the b ~ pi degenerate branch.
        U = sq.rz(theta)
        assert _equal_up_to_phase(
            zxzxz_to_unitary(unitary_to_zxzxz(U)), U
        )

    @pytest.mark.parametrize('theta', [0.1, -2.0, np.pi])
    def test_round_trip_rx(self, theta):
        U = sq.rx(theta)
        assert _equal_up_to_phase(
            zxzxz_to_unitary(unitary_to_zxzxz(U)), U
        )

    def test_antidiagonal_hits_b_zero_branch(self):
        # X: |u00| = 0, so b = 0 and c is fixed to 0.
        a, b, c = unitary_to_zxzxz(sq.x)
        assert np.isclose(b, 0.0)
        assert c == 0.0

    def test_diagonal_hits_b_pi_branch(self):
        a, b, c = unitary_to_zxzxz(sq.z)
        assert np.isclose(b, np.pi)
        assert a == 0.0

    def test_global_phase_is_ignored(self):
        U = unitary_group.rvs(2, random_state=7)
        a1 = np.array(unitary_to_zxzxz(U))
        a2 = np.array(unitary_to_zxzxz(np.exp(1j * 0.9) * U))
        assert _equal_up_to_phase(
            zxzxz_to_unitary(tuple(a1)), zxzxz_to_unitary(tuple(a2))
        )

    def test_returns_real_floats(self):
        angles = unitary_to_zxzxz(sq.h)
        assert len(angles) == 3
        assert all(type(x) is float for x in angles)

    def test_middle_angle_in_zero_to_pi(self):
        for seed in range(10):
            _, b, _ = unitary_to_zxzxz(
                unitary_group.rvs(2, random_state=seed)
            )
            assert -1e-12 <= b <= np.pi + 1e-12


class TestZxzxzDecompositionsTable:

    @pytest.mark.parametrize(
        'key', [k for k in ZXZXZ_DECOMPOSITIONS if isinstance(k, str)]
    )
    def test_string_entries_match_gate_unitary(self, key):
        gates = ZXZXZ_DECOMPOSITIONS[key](0)
        cls = sq.Id if key == 'I' else getattr(sq, key)
        assert _equal_up_to_phase(
            _gate_sequence_unitary(gates), cls(0).unitary
        )

    @pytest.mark.parametrize(
        'key', [k for k in ZXZXZ_DECOMPOSITIONS if isinstance(k, type)],
        ids=lambda k: k.__name__,
    )
    def test_class_entries_match_gate_unitary(self, key):
        gates = ZXZXZ_DECOMPOSITIONS[key](0)
        assert _equal_up_to_phase(
            _gate_sequence_unitary(gates), key(0).unitary
        )

    def test_class_and_string_keys_share_factory(self):
        assert ZXZXZ_DECOMPOSITIONS[sq.H] is ZXZXZ_DECOMPOSITIONS['H']
        assert ZXZXZ_DECOMPOSITIONS[sq.Cliff7] is (
            ZXZXZ_DECOMPOSITIONS['Cliff7']
        )

    def test_factory_structure(self):
        gates = ZXZXZ_DECOMPOSITIONS['H'](4)
        assert [g.name for g in gates] == ['Rz', 'X90', 'Rz', 'X90', 'Rz']
        assert all(g.qubits == (4,) for g in gates)


class TestDecomposeToZxzxz:

    def test_returns_tuple_of_five_gates(self):
        gates = decompose_to_zxzxz('Y', 2)
        assert isinstance(gates, tuple)
        assert [g.name for g in gates] == ['Rz', 'X90', 'Rz', 'X90', 'Rz']
        assert all(g.qubits == (2,) for g in gates)

    def test_implements_labelled_gate(self):
        for label, gate in [('X', X), ('Y', Y), ('Z', Z), ('H', H)]:
            gates = decompose_to_zxzxz(label, 1)
            assert _equal_up_to_phase(
                _gate_sequence_unitary(gates), gate(1).unitary
            )

    def test_is_cached(self):
        assert decompose_to_zxzxz('S', 3) is decompose_to_zxzxz('S', 3)

    def test_unknown_label_raises(self):
        with pytest.raises(KeyError):
            decompose_to_zxzxz('NotAGate', 0)


class TestDecomposeCycle:

    def test_structure_is_five_rz_x90_cycles(self):
        circuit = decompose_cycle(Cycle([H(0), S(1)]))
        assert isinstance(circuit, Circuit)
        assert len(circuit) == 5
        names = [{g.name for g in cycle} for cycle in circuit]
        assert names == [{'Rz'}, {'X90'}, {'Rz'}, {'X90'}, {'Rz'}]
        assert all(cycle.qubits == (0, 1) for cycle in circuit)

    def test_each_qubit_implements_original_gate(self):
        gates = [H(0), S(1), Y(2), Rz(5, 0.37)]
        circuit = decompose_cycle(Cycle(gates))
        for gate in gates:
            q = gate.qubits[0]
            assert _equal_up_to_phase(
                _qubit_unitary(circuit, q), gate.unitary
            )

    def test_merged_generic_gates(self):
        merged = merge_cycles(
            Cycle([H(0), X90(1)]), Cycle([S(0), Rz(1, 0.4)])
        )
        circuit = decompose_cycle(merged)
        np.testing.assert_equal(circuit.qubits, (0, 1))
        assert _equal_up_to_phase(
            _qubit_unitary(circuit, 0), S(0).unitary @ H(0).unitary
        )
        assert _equal_up_to_phase(
            _qubit_unitary(circuit, 1),
            Rz(1, 0.4).unitary @ X90(1).unitary,
        )

    def test_not_fooled_by_equal_hash_merged_cycles(self):
        # Two merged cycles compare equal (generic Gate equality ignores
        # the matrix) but implement different unitaries; decompose_cycle
        # must decompose each from its actual matrix.
        c1 = Cycle(merge_gates(X(0), X(0)))
        c2 = Cycle(merge_gates(X(0), H(0)))
        assert c1 == c2
        u1 = _qubit_unitary(decompose_cycle(c1), 0)
        u2 = _qubit_unitary(decompose_cycle(c2), 0)
        assert _equal_up_to_phase(u1, np.eye(2))
        assert _equal_up_to_phase(u2, sq.h @ sq.x)

    def test_full_cycle_unitary_matches(self):
        cycle = Cycle([H(0), S(1)])
        circuit = decompose_cycle(cycle)
        assert _equal_up_to_phase(circuit.unitary, cycle.unitary)


class TestPauliToCycle:

    def test_plain_cycle_gates(self):
        circuit = pauli_to_cycle(('X', 'I', 'Z'), (0, 1, 2))
        assert len(circuit) == 1
        cycle = circuit[0]
        by_qubit = {g.qubits[0]: g for g in cycle}
        assert isinstance(by_qubit[0], sq.X)
        assert isinstance(by_qubit[1], sq.Id)
        assert isinstance(by_qubit[2], sq.Z)

    def test_plain_cycle_unitary_is_pauli_tensor(self):
        circuit = pauli_to_cycle(('Y', 'X'), (3, 7))
        np.testing.assert_allclose(
            circuit[0].unitary, np.kron(sq.y, sq.x)
        )

    def test_qubit_labels_follow_positions(self):
        circuit = pauli_to_cycle(('Z', 'X'), (5, 2))
        by_qubit = {g.qubits[0]: g.name for g in circuit[0]}
        assert by_qubit == {5: 'Z', 2: 'X'}

    def test_zxzxz_structure(self):
        circuit = pauli_to_cycle(('X', 'I'), (0, 1), True)
        assert len(circuit) == 5
        names = [[g.name for g in cycle] for cycle in circuit]
        assert names == [
            ['Rz', 'Rz'], ['X90', 'X90'], ['Rz', 'Rz'], ['X90', 'X90'],
            ['Rz', 'Rz'],
        ]

    @pytest.mark.parametrize('pauli', ['I', 'X', 'Y', 'Z'])
    def test_zxzxz_implements_pauli(self, pauli):
        circuit = pauli_to_cycle(('I', pauli), (0, 1), True)
        expected = sq.SINGLE_QUBIT_PAULIS[pauli](1).unitary
        assert _equal_up_to_phase(_qubit_unitary(circuit, 1), expected)
        assert _equal_up_to_phase(_qubit_unitary(circuit, 0), np.eye(2))

    def test_is_cached(self):
        a = pauli_to_cycle(('X', 'Y'), (0, 1))
        b = pauli_to_cycle(('X', 'Y'), (0, 1))
        assert a is b

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            pauli_to_cycle(('X', 'Y'), (0,))
