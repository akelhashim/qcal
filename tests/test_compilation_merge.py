"""Unit tests for qcal.compilation.merge (merge_gates, merge_cycles).

Ordering convention (verified numerically): merging `first` then
`second` yields `second.unitary @ first.unitary`, i.e. the same unitary
as a two-cycle Circuit that applies `first` and then `second`. Merged
gates are generic `Gate(matrix, qubits)` objects whose equality/hash
ignore the matrix, so these tests always compare `.matrix`/`.unitary`
explicitly.
"""
import numpy as np
import pytest

from qcal.circuit import Circuit, Cycle
from qcal.compilation.merge import merge_cycles, merge_gates
from qcal.gates.gate import Gate
from qcal.gates.single_qubit import X90, H, Rz, S, X, Y
from qcal.gates.two_qubit import CNOT, CZ


class TestMergeGates:

    def test_matrix_is_second_times_first(self):
        first, second = X90(0), Rz(0, 0.3)
        merged = merge_gates(first, second)
        np.testing.assert_allclose(
            merged.matrix, second.unitary @ first.unitary
        )

    def test_order_matters_for_noncommuting_gates(self):
        a, b = X90(0), Rz(0, 0.3)
        ab = merge_gates(a, b).matrix
        ba = merge_gates(b, a).matrix
        assert not np.allclose(ab, ba)

    def test_matches_circuit_applying_first_then_second(self):
        first, second = H(0), S(0)
        merged = merge_gates(first, second)
        circuit = Circuit([Cycle(first), Cycle(second)])
        np.testing.assert_allclose(merged.unitary, circuit.unitary)

    def test_h_then_s_by_hand(self):
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        # qcal's S is Rz(pi/2), i.e. diag(1, i) up to global phase.
        s = np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)])
        merged = merge_gates(H(0), S(0))
        np.testing.assert_allclose(merged.matrix, s @ h)

    def test_returns_generic_gate_on_same_qubits(self):
        merged = merge_gates(X(3), Y(3))
        assert type(merged) is Gate
        assert merged.name == 'Gate'
        assert merged.qubits == (3,)

    def test_two_qubit_merge(self):
        local = Gate(np.kron(H(0).unitary, np.eye(2)), (0, 1))
        merged = merge_gates(local, CNOT(0, 1))
        np.testing.assert_allclose(
            merged.matrix, CNOT(0, 1).unitary @ local.unitary
        )
        assert merged.qubits == (0, 1)

    def test_merge_with_inverse_is_identity(self):
        merged = merge_gates(Rz(0, 0.7), Rz(0, -0.7))
        np.testing.assert_allclose(merged.matrix, np.eye(2), atol=1e-12)

    def test_merged_result_is_unitary(self):
        merged = merge_gates(H(0), Rz(0, 1.1))
        np.testing.assert_allclose(
            merged.matrix @ merged.matrix.conj().T, np.eye(2), atol=1e-12
        )

    def test_different_qubits_raises(self):
        with pytest.raises(ValueError, match='different qubits'):
            merge_gates(X(0), X(1))

    def test_reversed_qubit_order_raises(self):
        with pytest.raises(ValueError):
            merge_gates(CNOT(0, 1), CNOT(1, 0))

    def test_merges_with_different_matrices_are_distinct(self):
        # Merged gates are all generic Gates named 'Gate' with empty
        # params, so equality/hashing must tell them apart by matrix.
        m1 = merge_gates(X(0), X(0))
        m2 = merge_gates(X(0), Y(0))
        assert m1 != m2
        assert len({m1, m2}) == 2

    def test_merges_with_equal_matrices_are_equal(self):
        # X @ X is a real identity, Y @ Y a complex one: equal values
        # with different dtypes must still be equal and hash equally.
        m1 = merge_gates(X(0), X(0))
        m2 = merge_gates(Y(0), Y(0))
        assert m1 == m2
        assert hash(m1) == hash(m2)


class TestMergeCycles:

    def test_single_qubit_cycles_merge_per_qubit(self):
        first = Cycle([H(0), X90(1)])
        second = Cycle([S(0), Rz(1, 0.4)])
        merged = merge_cycles(first, second)
        assert merged.qubits == (0, 1)
        assert merged.n_gates == 2
        by_qubit = {g.qubits: g for g in merged}
        np.testing.assert_allclose(
            by_qubit[(0,)].matrix, S(0).unitary @ H(0).unitary
        )
        np.testing.assert_allclose(
            by_qubit[(1,)].matrix, Rz(1, 0.4).unitary @ X90(1).unitary
        )

    def test_merged_cycle_unitary_matches_circuit(self):
        first = Cycle([H(0), X90(1), Y(2)])
        second = Cycle([S(0), Rz(1, 0.4), X90(2)])
        merged = merge_cycles(first, second)
        circuit = Circuit([first, second])
        np.testing.assert_allclose(merged.unitary, circuit.unitary)

    def test_pairs_gates_by_qubits_not_insertion_order(self):
        first = Cycle([X(1), H(0)])
        second = Cycle([S(0), Y(1)])
        merged = merge_cycles(first, second)
        by_qubit = {g.qubits: g for g in merged}
        np.testing.assert_allclose(
            by_qubit[(0,)].matrix, S(0).unitary @ H(0).unitary
        )
        np.testing.assert_allclose(
            by_qubit[(1,)].matrix, Y(1).unitary @ X(1).unitary
        )

    def test_two_qubit_gates_merge(self):
        first = Cycle(CNOT(0, 1))
        second = Cycle(CZ(0, 1))
        merged = merge_cycles(first, second)
        assert merged.n_gates == 1
        np.testing.assert_allclose(
            merged.gates[0].matrix, CZ(0, 1).unitary @ CNOT(0, 1).unitary
        )

    def test_different_qubits_raises(self):
        with pytest.raises(ValueError, match='different qubits'):
            merge_cycles(Cycle(X(0)), Cycle(X(1)))

    def test_different_grouping_raises(self):
        with pytest.raises(ValueError, match='grouped'):
            merge_cycles(Cycle(CNOT(0, 1)), Cycle([X(0), X(1)]))

    def test_different_grouping_raises_other_direction(self):
        with pytest.raises(ValueError, match='grouped'):
            merge_cycles(Cycle([X(0), X(1)]), Cycle(CNOT(0, 1)))
