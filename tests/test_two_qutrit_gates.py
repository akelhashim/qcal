"""Unit tests for qcal.gates.two_qutrit.

Covers the _normalize_qudits helper, the cp3 parametric function,
every gate class, and the TWO_QUTRIT_GATES registry.
"""
import numpy as np
import pytest

from qcal.gates.two_qutrit import (
    CP3,
    CSUM,
    CZ3,
    SWAP3,
    TWO_QUTRIT_GATES,
    CSUMdag,
    CZ3dag,
    _normalize_qudits,
    cp3,
    csum,
    csumd,
    cz3,
    cz3dag,
    swap3,
)


def _is_unitary(matrix, atol=1e-8):
    return np.allclose(
        matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=atol
    )


class TestNormalizeQudits:

    def test_tuple_form(self):
        assert _normalize_qudits(((0, 1),)) == (0, 1)

    def test_separate_args_form(self):
        assert _normalize_qudits((0, 1)) == (0, 1)


class TestCp3Function:

    def test_is_unitary(self):
        assert _is_unitary(cp3(0.42))

    def test_zero_angle_is_identity(self):
        np.testing.assert_allclose(cp3(0.0), np.eye(9), atol=1e-12)

    def test_matches_cz3_at_two_pi_thirds(self):
        np.testing.assert_allclose(cp3(2 * np.pi / 3), cz3, atol=1e-10)


class TestMatrixConstants:

    @pytest.mark.parametrize('matrix', [cz3, cz3dag, csum, csumd, swap3])
    def test_is_unitary(self, matrix):
        assert _is_unitary(matrix)

    def test_cz3dag_is_conjugate_of_cz3(self):
        np.testing.assert_allclose(cz3dag, cz3.conj())

    def test_cz3_times_cz3dag_is_identity(self):
        np.testing.assert_allclose(cz3 @ cz3dag, np.eye(9), atol=1e-10)

    def test_csum_times_csumdag_is_identity(self):
        np.testing.assert_allclose(csum @ csumd, np.eye(9), atol=1e-10)

    def test_swap3_is_its_own_inverse(self):
        np.testing.assert_allclose(swap3 @ swap3, np.eye(9), atol=1e-10)


class TestGateClasses:

    def test_cp3_matches_function_and_subspace(self):
        theta = 0.42
        gate = CP3(0, 1, theta=theta)
        np.testing.assert_allclose(gate.unitary, cp3(theta))
        assert gate.subspace == 'GEF'
        assert gate.properties['params'] == {'angle': theta}

    @pytest.mark.parametrize('gate_cls, expected', [
        (CSUM, csum), (CSUMdag, csumd), (CZ3, cz3), (CZ3dag, cz3dag),
        (SWAP3, swap3),
    ])
    def test_fixed_matrix_gates(self, gate_cls, expected):
        gate = gate_cls(0, 1)
        np.testing.assert_allclose(gate.unitary, expected)
        assert gate.subspace == 'GEF'
        assert gate.qubits == (0, 1)

    @pytest.mark.parametrize('gate_cls, kwargs', [
        (CP3, {'theta': 0.42}), (CSUM, {}), (CSUMdag, {}), (CZ3, {}),
        (CZ3dag, {}), (SWAP3, {}),
    ])
    def test_is_multi_qudit_and_unitary(self, gate_cls, kwargs):
        gate = gate_cls(0, 1, **kwargs)
        assert gate.is_multi_qudit
        assert not gate.is_multi_qubit
        assert _is_unitary(gate.unitary)

    @pytest.mark.parametrize('gate_cls, kwargs', [
        (CSUM, {}), (CZ3, {}), (SWAP3, {}),
    ])
    def test_tuple_and_separate_args_agree(self, gate_cls, kwargs):
        np.testing.assert_allclose(
            gate_cls(0, 1, **kwargs).unitary,
            gate_cls((0, 1), **kwargs).unitary,
        )


class TestRegistry:

    @pytest.mark.parametrize('name, gate_cls', [
        ('CP3', CP3), ('CSUM', CSUM), ('CSUMdag', CSUMdag),
        ('CZ3', CZ3), ('CZ3dag', CZ3dag), ('SWAP3', SWAP3),
    ])
    def test_registry_maps_name_to_class(self, name, gate_cls):
        assert TWO_QUTRIT_GATES[name] is gate_cls

    def test_unsupported_name_returns_default_message_not_keyerror(self):
        assert (
            TWO_QUTRIT_GATES['bogus'] == 'Gate not currently supported!'
        )
