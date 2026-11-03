"""Unit tests for qcal.gates.gate.Gate, the base class every concrete
gate (single/two-qubit, single/two-qutrit) subclasses.
"""
import numpy as np
import sympy

from qcal.gates.gate import Gate

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0., 1.], [1., 0.]], dtype=complex)


class TestConstruction:

    def test_int_qubit_is_normalized_to_tuple(self):
        gate = Gate(I2, 0)
        assert gate.qubits == (0,)

    def test_tuple_qubits_kept_as_is(self):
        gate = Gate(np.eye(4, dtype=complex), (0, 1))
        assert gate.qubits == (0, 1)

    def test_matrix_and_unitary_both_start_as_given_matrix(self):
        gate = Gate(X2, 0)
        np.testing.assert_array_equal(gate.matrix, X2)
        np.testing.assert_array_equal(gate.unitary, X2)

    def test_dim_is_matrix_dimension(self):
        assert Gate(I2, 0).dim == 2
        assert Gate(np.eye(4, dtype=complex), (0, 1)).dim == 4
        assert Gate(np.eye(3, dtype=complex), 0).dim == 3

    def test_default_properties(self):
        gate = Gate(I2, 0)
        assert gate.alias is None
        assert gate.locally_equivalent is None
        assert gate.name == 'Gate'
        assert gate.subspace == 'GE'
        assert gate.properties['params'] == {}


class TestQubitQuditClassification:

    def test_single_qubit(self):
        gate = Gate(I2, 0)
        assert gate.is_single_qubit
        assert gate.is_single_qudit
        assert not gate.is_multi_qubit
        assert not gate.is_multi_qudit

    def test_multi_qubit(self):
        gate = Gate(np.eye(4, dtype=complex), (0, 1))
        assert not gate.is_single_qubit
        assert not gate.is_single_qudit
        assert gate.is_multi_qubit
        assert gate.is_multi_qudit

    def test_single_qutrit_is_not_a_qubit_gate(self):
        gate = Gate(np.eye(3, dtype=complex), 0)
        assert not gate.is_single_qubit
        assert gate.is_single_qudit
        assert not gate.is_multi_qubit
        assert not gate.is_multi_qudit

    def test_two_qutrit_is_not_a_qubit_gate(self):
        gate = Gate(np.eye(9, dtype=complex), (0, 1))
        assert not gate.is_single_qubit
        assert not gate.is_multi_qubit
        assert gate.is_multi_qudit

    def test_qudits_aliases_qubits(self):
        gate = Gate(np.eye(4, dtype=complex), (0, 1))
        assert gate.qudits == gate.qubits


class TestIsMeasurement:

    def test_default_is_false(self):
        assert Gate(I2, 0).is_measurement is False


class TestUnitary:

    def test_ge_subspace_returns_matrix_unchanged(self):
        gate = Gate(X2, 0)
        np.testing.assert_array_equal(gate.unitary, X2)

    def test_ef_subspace_embeds_2x2_into_3x3(self):
        gate = Gate(X2, 0)
        gate._properties['subspace'] = 'EF'
        expected = np.eye(3, dtype=complex)
        expected[1:3, 1:3] = X2
        np.testing.assert_array_equal(gate.unitary, expected)

    def test_ef_subspace_leaves_already_3x3_matrix_unchanged(self):
        # The EF embedding only triggers for a 2x2 matrix -- a gate
        # that is already defined on the full qutrit space (e.g. any
        # single_qutrit.py gate) passes through unmodified regardless
        # of its subspace label.
        matrix = np.eye(3, dtype=complex) * 1j
        gate = Gate(matrix, 0)
        gate._properties['subspace'] = 'EF'
        np.testing.assert_array_equal(gate.unitary, matrix)

    def test_setter_overrides_measured_unitary(self):
        gate = Gate(X2, 0)
        measured = X2 * 0.99
        gate.unitary = measured
        np.testing.assert_array_equal(gate.unitary, measured)

    def test_setter_does_not_affect_matrix_property(self):
        gate = Gate(X2, 0)
        gate.unitary = X2 * 0.99
        np.testing.assert_array_equal(gate.matrix, X2)

    def test_setter_then_ef_embedding_uses_new_unitary(self):
        gate = Gate(X2, 0)
        gate._properties['subspace'] = 'EF'
        measured = X2 * 0.99
        gate.unitary = measured
        expected = np.eye(3, dtype=complex)
        expected[1:3, 1:3] = measured
        np.testing.assert_array_equal(gate.unitary, expected)


class TestEquality:

    def test_equal_when_name_qubits_subspace_params_match(self):
        a = Gate(X2, 0)
        b = Gate(X2, 0)
        assert a == b

    def test_not_equal_with_different_qubits(self):
        a = Gate(X2, 0)
        b = Gate(X2, 1)
        assert a != b

    def test_not_equal_with_different_params(self):
        a = Gate(X2, 0)
        b = Gate(X2, 0)
        b._properties['params'] = {'angle': 1.0}
        assert a != b

    def test_not_equal_to_non_gate(self):
        assert Gate(X2, 0) != 'not a gate'

    def test_hash_consistent_with_equality(self):
        a = Gate(X2, 0)
        b = Gate(X2, 0)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


class TestDunderMethods:

    def test_call_returns_sympy_matrix(self):
        gate = Gate(X2, 0)
        assert isinstance(gate(), sympy.Matrix)

    def test_repr_contains_name_and_qubits(self):
        gate = Gate(X2, 0)
        assert 'Gate' in repr(gate)
        assert '(0,)' in repr(gate)

    def test_str_matches_repr(self):
        gate = Gate(X2, 0)
        assert str(gate) == repr(gate)
