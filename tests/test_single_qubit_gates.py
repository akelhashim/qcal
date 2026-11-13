"""Unit tests for qcal.gates.single_qubit.

Covers the raw rotation-matrix helper functions, every concrete gate
class (unitary matrix, subspace, params, aliases), the
SINGLE_QUBIT_GATES registry, and the basis_rotation() helper.
"""
import itertools
import re

import numpy as np
import pytest

from qcal.gates.single_qubit import (
    MCM,
    SINGLE_QUBIT_CLIFFORDS,
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
    prep_rotation,
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

    def test_x_basis_returns_sydag(self):
        gate = basis_rotation(Meas(3, basis='X'))
        assert isinstance(gate, SYdag)
        assert gate.qubits == (3,)
        assert gate.properties['params']['angle'] == pytest.approx(
            -np.pi / 2
        )

    def test_y_basis_returns_sx(self):
        gate = basis_rotation(Meas(3, basis='Y'))
        assert isinstance(gate, SX)
        assert gate.properties['params']['angle'] == pytest.approx(
            np.pi / 2
        )

    def test_z_basis_returns_id(self):
        gate = basis_rotation(Meas(3, basis='Z'))
        assert isinstance(gate, Id)

    def test_lowercase_basis_is_accepted(self):
        gate = basis_rotation(Meas(3, basis='x'))
        assert isinstance(gate, SYdag)

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


def _clifford_matrices():
    return [SINGLE_QUBIT_CLIFFORDS[i](0).matrix for i in range(24)]


def _clifford_index(matrix, cliffords):
    """Indices of every Clifford equal to `matrix` up to global phase."""
    return [
        i for i, c in enumerate(cliffords)
        if _equal_up_to_phase(c, matrix)
    ]


class TestCliffordGates:

    @pytest.mark.parametrize('index', range(24))
    def test_registered_by_index_and_name(self, index):
        gate_cls = SINGLE_QUBIT_CLIFFORDS[index]
        gate = gate_cls(4)

        assert gate.name == f'Cliff{index}'
        assert gate.qubits == (4,)
        assert gate.subspace == 'GE'
        assert SINGLE_QUBIT_GATES[f'Cliff{index}'] is gate_cls

    @pytest.mark.parametrize('index', range(24))
    def test_is_single_qubit_unitary(self, index):
        gate = SINGLE_QUBIT_CLIFFORDS[index](0)

        assert gate.matrix.shape == (2, 2)
        assert gate.is_single_qubit
        assert _is_unitary(gate.matrix)

    def test_pairwise_distinct_up_to_phase(self):
        cliffords = _clifford_matrices()
        for i, j in itertools.combinations(range(24), 2):
            assert not _equal_up_to_phase(cliffords[i], cliffords[j]), (
                f'Cliff{i} == Cliff{j} up to phase'
            )

    def test_closed_under_multiplication(self):
        cliffords = _clifford_matrices()
        for a, b in itertools.product(cliffords, repeat=2):
            assert len(_clifford_index(a @ b, cliffords)) == 1

    def test_closed_under_inversion(self):
        cliffords = _clifford_matrices()
        for c in cliffords:
            assert len(_clifford_index(c.conj().T, cliffords)) == 1

    @pytest.mark.parametrize('index', range(24))
    def test_maps_paulis_to_signed_paulis(self, index):
        c = SINGLE_QUBIT_CLIFFORDS[index](0).matrix
        images = []
        for pauli in (x, y, z):
            image = c @ pauli @ c.conj().T
            matches = [
                k for k, p in enumerate((x, y, z))
                if np.allclose(image, p) or np.allclose(image, -p)
            ]
            assert len(matches) == 1
            images.append(matches[0])
        # Distinct Paulis must map to distinct Paulis.
        assert sorted(images) == [0, 1, 2]

    @pytest.mark.parametrize('index, expected', [
        (0, id), (1, x), (2, y), (3, z), (14, h),
    ])
    def test_named_cliffords_match_exact_matrix(self, index, expected):
        np.testing.assert_allclose(
            SINGLE_QUBIT_CLIFFORDS[index](0).matrix, expected, atol=1e-12
        )

    @pytest.mark.parametrize('index, expected', [
        (4, rx(np.pi / 2)), (5, rx(-np.pi / 2)),
        (6, ry(np.pi / 2)), (7, ry(-np.pi / 2)),
        (8, rz(np.pi / 2)), (9, rz(-np.pi / 2)),
    ])
    def test_half_pi_rotations(self, index, expected):
        np.testing.assert_allclose(
            SINGLE_QUBIT_CLIFFORDS[index](0).matrix, expected, atol=1e-12
        )

    @pytest.mark.parametrize('index', range(10, 24))
    def test_matrix_matches_params_rotation(self, index):
        gate = SINGLE_QUBIT_CLIFFORDS[index](0)
        params = gate.properties['params']
        assert _equal_up_to_phase(
            gate.matrix, rn(params['angle'], params['axis'])
        )

    def test_cliff16_cycles_x_to_y_to_z(self):
        c = SINGLE_QUBIT_CLIFFORDS[16](0).matrix

        np.testing.assert_allclose(c @ x @ c.conj().T, y, atol=1e-12)
        np.testing.assert_allclose(c @ y @ c.conj().T, z, atol=1e-12)
        np.testing.assert_allclose(c @ z @ c.conj().T, x, atol=1e-12)

    @pytest.mark.parametrize('index, inverse', [
        (4, 5), (6, 7), (8, 9), (16, 17), (18, 19), (20, 21), (22, 23),
    ])
    def test_inverse_pairs(self, index, inverse):
        a = SINGLE_QUBIT_CLIFFORDS[index](0).matrix
        b = SINGLE_QUBIT_CLIFFORDS[inverse](0).matrix
        assert _equal_up_to_phase(a @ b, np.eye(2))

    @pytest.mark.parametrize('index', [0, 1, 2, 3, 10, 11, 12, 13, 14, 15])
    def test_involutions_square_to_identity(self, index):
        c = SINGLE_QUBIT_CLIFFORDS[index](0).matrix
        assert _equal_up_to_phase(c @ c, np.eye(2))

    @pytest.mark.parametrize('name', [
        'C', 'H', 'Id', 'S', 'Sdag', 'SX', 'SXdag', 'SY', 'SYdag', 'V',
        'Vdag', 'X', 'X90', 'Y', 'Y90', 'Z', 'Z90',
    ])
    def test_cliff_alias_matches_clifford_matrix(self, name):
        gate = SINGLE_QUBIT_GATES[name](0)
        indices = [int(k) for k in re.findall(r'Cliff(\d+)', gate.alias)]

        assert len(indices) == 1
        assert _equal_up_to_phase(
            gate.matrix, SINGLE_QUBIT_CLIFFORDS[indices[0]](0).matrix
        )


class TestPrepRotation:

    @pytest.mark.parametrize('pauli, operator', [
        ('X', x), ('Y', y), ('Z', z),
    ])
    def test_prepares_plus_one_eigenstate(self, pauli, operator):
        gate = prep_rotation(2, pauli)
        state = gate.matrix @ np.array([1, 0])

        assert gate.qubits == (2,)
        np.testing.assert_allclose(operator @ state, state, atol=1e-12)

    @pytest.mark.parametrize('pauli, operator', [
        ('-X', x), ('-Y', y), ('-Z', z),
    ])
    def test_prepares_minus_one_eigenstate(self, pauli, operator):
        gate = prep_rotation(2, pauli)
        state = gate.matrix @ np.array([1, 0])

        assert gate.qubits == (2,)
        np.testing.assert_allclose(operator @ state, -state, atol=1e-12)

    @pytest.mark.parametrize('pauli', ['+X', '+Y', '+Z', '-x', '-y', '-z'])
    def test_signed_paulis_accepted(self, pauli):
        sign = -1 if pauli[0] == '-' else 1
        operator = {'X': x, 'Y': y, 'Z': z}[pauli[1].upper()]
        state = prep_rotation(0, pauli).matrix @ np.array([1, 0])
        np.testing.assert_allclose(
            operator @ state, sign * state, atol=1e-12
        )

    @pytest.mark.parametrize('pauli, expected', [
        ('X', np.array([1, 1]) / np.sqrt(2)),
        ('Y', np.array([1, 1j]) / np.sqrt(2)),
        ('Z', np.array([1, 0])),
    ])
    def test_prepared_state_vectors(self, pauli, expected):
        state = prep_rotation(0, pauli).matrix @ np.array([1, 0])
        assert np.isclose(abs(np.vdot(expected, state)), 1)

    @pytest.mark.parametrize('pauli, gate_cls, angle', [
        ('X', SY, np.pi / 2), ('Y', SXdag, -np.pi / 2),
        ('-X', SYdag, -np.pi / 2), ('-Y', SX, np.pi / 2),
    ])
    def test_rotation_gate_and_angle(self, pauli, gate_cls, angle):
        gate = prep_rotation(0, pauli)

        assert isinstance(gate, gate_cls)
        assert gate.properties['params']['angle'] == pytest.approx(angle)

    @pytest.mark.parametrize('pauli', ['I', 'Z'])
    def test_identity_for_i_and_z(self, pauli):
        assert isinstance(prep_rotation(0, pauli), Id)

    def test_lowercase_is_accepted(self):
        assert isinstance(prep_rotation(0, 'x'), SY)

    def test_invalid_pauli_raises(self):
        with pytest.raises(ValueError, match='Invalid Pauli'):
            prep_rotation(0, 'W')

    @pytest.mark.parametrize('basis', ['X', 'Y', 'Z'])
    def test_inverts_basis_rotation(self, basis):
        # Preparing the +1 eigenstate and rotating back into the
        # measurement basis lands on |0>.
        prep = prep_rotation(0, basis).matrix
        meas = basis_rotation(Meas(0, basis=basis)).matrix
        state = meas @ prep @ np.array([1, 0])
        assert np.isclose(abs(state[0]), 1)
