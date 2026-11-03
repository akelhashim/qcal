"""Unit tests for qcal.gates.single_qubit.

Covers the raw rotation-matrix helper functions, every concrete gate
class (unitary matrix, subspace, params, aliases), the
SINGLE_QUBIT_GATES registry, and the basis_rotation() helper.
"""
import numpy as np
import pytest

from qcal.gates.single_qubit import (
    MCM,
    SINGLE_QUBIT_GATES,
    SX,
    SY,
    U3,
    X90,
    Y90,
    Z90,
    C,
    H,
    Id,
    Idle,
    Meas,
    RandSU2,
    Reset,
    Rn,
    Rx,
    Ry,
    Rz,
    S,
    Sdag,
    SXdag,
    SYdag,
    T,
    Tdag,
    V,
    Vdag,
    VirtualZ,
    X,
    Y,
    Z,
    basis_rotation,
    h,
    id,
    rn,
    rx,
    ry,
    rz,
    t,
    tdag,
    u3,
    x,
    y,
    z,
)
from qcal.units import ns


def _is_unitary(matrix):
    return np.allclose(
        matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=1e-8
    )


def _equal_up_to_phase(a, b, atol=1e-8):
    idx = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    phase = b[idx] / a[idx]
    return np.allclose(a * phase, b, atol=atol)


class TestRotationFunctions:

    def test_rx_zero_is_identity(self):
        np.testing.assert_allclose(rx(0.0), np.eye(2), atol=1e-12)

    def test_ry_zero_is_identity(self):
        np.testing.assert_allclose(ry(0.0), np.eye(2), atol=1e-12)

    def test_rz_zero_is_identity(self):
        np.testing.assert_allclose(rz(0.0), np.eye(2), atol=1e-12)

    def test_rx_pi_equals_x_up_to_phase(self):
        assert _equal_up_to_phase(rx(np.pi), x)

    def test_ry_pi_equals_y_up_to_phase(self):
        assert _equal_up_to_phase(ry(np.pi), y)

    def test_rz_pi_equals_z_up_to_phase(self):
        assert _equal_up_to_phase(rz(np.pi), z)

    @pytest.mark.parametrize('theta', [0.0, 0.3, 1.0, np.pi, 2.5])
    def test_rotations_are_unitary(self, theta):
        assert _is_unitary(rx(theta))
        assert _is_unitary(ry(theta))
        assert _is_unitary(rz(theta))

    def test_rn_requires_unit_vector(self):
        with pytest.raises(AssertionError):
            rn(0.5, (1, 1, 1))

    def test_rn_z_axis_matches_rz(self):
        np.testing.assert_allclose(rn(0.7, (0, 0, 1)), rz(0.7), atol=1e-10)

    def test_rn_x_axis_matches_rx(self):
        np.testing.assert_allclose(rn(0.7, (1, 0, 0)), rx(0.7), atol=1e-10)

    def test_u3_is_unitary(self):
        assert _is_unitary(u3(0.3, 0.5, 0.7))

    def test_u3_zero_theta_is_diagonal_phase(self):
        U = u3(0.0, 0.5, 0.7)
        np.testing.assert_allclose(U[0, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(U[1, 0], 0.0, atol=1e-12)


class TestNonParametricGates:
    """Fixed-matrix gates, checked against hand/independently derived
    unitaries.
    """

    @pytest.mark.parametrize('gate_cls, expected', [
        (H, h),
        (Id, id),
        (X, x),
        (Y, y),
        (Z, z),
        (T, t),
        (Tdag, tdag),
    ])
    def test_matches_expected_matrix(self, gate_cls, expected):
        np.testing.assert_allclose(gate_cls(0).unitary, expected)

    @pytest.mark.parametrize('gate_cls, expected', [
        (SX, rx(np.pi / 2)),
        (SY, ry(np.pi / 2)),
        (SXdag, rx(-np.pi / 2)),
        (SYdag, ry(-np.pi / 2)),
        (S, rz(np.pi / 2)),
        (Sdag, rz(-np.pi / 2)),
        (V, rx(np.pi / 2)),
        (Vdag, rx(-np.pi / 2)),
        (X90, rx(np.pi / 2)),
        (Y90, ry(np.pi / 2)),
        (Z90, rz(np.pi / 2)),
    ])
    def test_matches_expected_rotation(self, gate_cls, expected):
        np.testing.assert_allclose(gate_cls(0).unitary, expected)

    @pytest.mark.parametrize('gate_cls', [
        H, Id, X, Y, Z, T, Tdag, SX, SY, SXdag, SYdag, S, Sdag, V, Vdag,
        X90, Y90, Z90,
    ])
    def test_is_single_qubit_and_unitary(self, gate_cls):
        gate = gate_cls(0)
        assert gate.is_single_qubit
        assert gate.qubits == (0,)
        assert _is_unitary(gate.unitary)


class TestAliasEquivalences:

    def test_x90_equals_sx_equals_v(self):
        np.testing.assert_allclose(X90(0).unitary, SX(0).unitary)
        np.testing.assert_allclose(X90(0).unitary, V(0).unitary)

    def test_sxdag_equals_vdag(self):
        np.testing.assert_allclose(SXdag(0).unitary, Vdag(0).unitary)

    def test_y90_equals_sy(self):
        np.testing.assert_allclose(Y90(0).unitary, SY(0).unitary)

    def test_z90_equals_s(self):
        np.testing.assert_allclose(Z90(0).unitary, S(0).unitary)


class TestIdAndIdle:

    def test_id_params(self):
        gate = Id(0)
        assert gate.properties['params'] == {'angle': 0, 'duration': 0.}

    def test_idle_default_duration_is_zero(self):
        gate = Idle(0)
        assert gate.properties['params']['duration'] == 0 * ns

    def test_idle_custom_duration(self):
        gate = Idle(0, duration=50 * ns)
        assert gate.properties['params']['duration'] == 50 * ns

    def test_idle_matrix_is_identity(self):
        np.testing.assert_allclose(Idle(0, duration=50 * ns).unitary, id)


class TestParametrizedRotations:

    @pytest.mark.parametrize('gate_cls, fn', [(Rx, rx), (Ry, ry)])
    def test_matches_rotation_function(self, gate_cls, fn):
        theta = 0.42
        np.testing.assert_allclose(gate_cls(0, theta).unitary, fn(theta))

    def test_rz_matches_rotation_function(self):
        theta = 0.42
        np.testing.assert_allclose(Rz(0, theta).unitary, rz(theta))

    def test_rz_default_subspace_is_ge(self):
        assert Rz(0, 0.3).subspace == 'GE'

    def test_rz_custom_subspace(self):
        assert Rz(0, 0.3, subspace='EF').subspace == 'EF'

    def test_virtualz_matches_rz_matrix_but_different_name(self):
        theta = 0.6
        np.testing.assert_allclose(
            VirtualZ(0, theta).unitary, Rz(0, theta).unitary
        )
        assert VirtualZ(0, theta).name == 'VirtualZ'
        assert Rz(0, theta).name == 'Rz'

    def test_rn_gate_matches_rn_function(self):
        theta, n = 0.5, (0, 1, 0)
        np.testing.assert_allclose(Rn(0, theta, n).unitary, rn(theta, n))

    def test_u3_gate_matches_u3_function(self):
        theta, phi, gamma = 0.3, 0.5, 0.7
        np.testing.assert_allclose(
            U3(0, theta, phi, gamma).unitary, u3(theta, phi, gamma)
        )

    def test_c_gate_cubed_is_identity_up_to_phase(self):
        U = C(0).unitary
        assert _equal_up_to_phase(U @ U @ U, np.eye(2))

    def test_randsu2_is_unitary_single_qubit(self):
        gate = RandSU2(0)
        assert _is_unitary(gate.unitary)
        assert gate.is_single_qubit


class TestSubspace:

    def test_x_default_subspace_is_ge(self):
        assert X(0).subspace == 'GE'

    def test_x_ef_subspace_embeds_into_3x3(self):
        gate = X(0, subspace='EF')
        expected = np.eye(3, dtype=complex)
        expected[1:3, 1:3] = x
        np.testing.assert_allclose(gate.unitary, expected)
        # matrix (the raw 2x2 definition) is untouched by the embedding.
        np.testing.assert_allclose(gate.matrix, x)


class TestMeasurementAndNonUnitaryGates:

    def test_meas_is_measurement_and_non_unitary(self):
        gate = Meas(0)
        assert gate.is_measurement
        assert gate.unitary is None

    def test_meas_default_basis_is_z(self):
        assert Meas(0).properties['params']['basis'] == 'Z'

    def test_meas_custom_basis(self):
        assert Meas(0, basis='X').properties['params']['basis'] == 'X'

    def test_mcm_is_measurement_non_unitary_and_single_qubit(self):
        gate = MCM(0)
        assert gate.is_measurement
        assert gate.unitary is None
        assert gate.is_single_qubit

    def test_reset_is_non_unitary(self):
        assert Reset(0).unitary is None

    def test_reset_default_method_is_active(self):
        assert Reset(0).properties['params']['method'] == ['active']

    def test_reset_invalid_method_raises(self):
        with pytest.raises(ValueError):
            Reset(0, method=['bogus'])

    def test_reset_contains_a_mcm(self):
        gate = Reset(0)
        assert isinstance(gate.properties['params']['meas'], MCM)


class TestBasisRotation:

    def test_x_basis_returns_ry_minus_half_pi(self):
        gate = basis_rotation(Meas(3, basis='X'))
        assert isinstance(gate, Ry)
        assert gate.qubits == (3,)
        assert gate.properties['params']['angle'] == pytest.approx(
            -np.pi / 2
        )

    def test_y_basis_returns_rx_half_pi(self):
        gate = basis_rotation(Meas(3, basis='Y'))
        assert isinstance(gate, Rx)
        assert gate.properties['params']['angle'] == pytest.approx(
            np.pi / 2
        )

    def test_z_basis_returns_id(self):
        gate = basis_rotation(Meas(3, basis='Z'))
        assert isinstance(gate, Id)

    def test_lowercase_basis_is_accepted(self):
        gate = basis_rotation(Meas(3, basis='x'))
        assert isinstance(gate, Ry)

    def test_unsupported_basis_raises(self):
        with pytest.raises(KeyError):
            basis_rotation(Meas(3, basis='bogus'))


class TestRegistry:

    @pytest.mark.parametrize('name, gate_cls', [
        ('H', H), ('X', X), ('Y', Y), ('Z', Z), ('X90', X90), ('Y90', Y90),
        ('Id', Id), ('Meas', Meas), ('Rz', Rz),
    ])
    def test_registry_maps_name_to_class(self, name, gate_cls):
        assert SINGLE_QUBIT_GATES[name] is gate_cls

    def test_unsupported_name_returns_default_message_not_keyerror(self):
        assert (
            SINGLE_QUBIT_GATES['bogus']
            == 'Gate not currently supported!'
        )
