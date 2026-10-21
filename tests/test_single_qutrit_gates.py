"""Unit tests for qcal.gates.single_qutrit.

Covers the Gell-Mann/Weyl matrix constants, the parametric rotation
functions (rx01/rx12/ry01/ry12/rz01/rz12, rgm), every gate class, and
the SINGLE_QUTRIT_GATES registry.

Gate.unitary's EF-subspace embedding (see qcal.gates.gate) only
triggers for a 2x2 matrix, so for every gate here -- already defined
on the full 3-level space -- .unitary always equals .matrix exactly,
regardless of the gate's own subspace label ('GE', 'EF', 'GF', or
'GEF').
"""
import numpy as np
import pytest

from qcal.gates.single_qubit import rx, ry, rz
from qcal.gates.single_qutrit import (
    H3,
    RGM1,
    RGM2,
    RGM3,
    RGM4,
    RGM5,
    RGM6,
    RGM7,
    RGM8,
    SINGLE_QUTRIT_GATES,
    X01,
    X02,
    X12,
    X90_01,
    X90_12,
    Y01,
    Y02,
    Y12,
    Y90_01,
    Y90_12,
    Z01,
    Z12,
    Id3,
    Rx01,
    Rx12,
    Ry01,
    Ry12,
    Rz01,
    Rz12,
    VirtualZ01,
    VirtualZ12,
    WeylX,
    WeylX2,
    WeylXZ,
    WeylZ,
    WeylZ2,
    _omega,
    h3,
    id3,
    lam1,
    lam2,
    lam3,
    lam4,
    lam5,
    lam6,
    lam7,
    lam8,
    rgm,
    rx01,
    rx12,
    ry01,
    ry12,
    rz01,
    rz12,
    weyl_x,
    weyl_x2,
    weyl_xz,
    weyl_z,
    weyl_z2,
    x01,
    x02,
    x12,
    y01,
    y02,
    y12,
    z01,
    z12,
)

GELL_MANN = [lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8]


def _is_unitary(matrix, atol=1e-8):
    return np.allclose(
        matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=atol
    )


class TestMatrixConstants:

    @pytest.mark.parametrize('matrix', [
        id3, x01, x12, x02, y01, y02, y12, z01, z12, h3, weyl_x, weyl_z,
        weyl_x2, weyl_z2, weyl_xz,
    ])
    def test_is_unitary(self, matrix):
        assert _is_unitary(matrix)

    @pytest.mark.parametrize('lam', GELL_MANN)
    def test_gell_mann_is_hermitian_and_traceless(self, lam):
        np.testing.assert_allclose(lam, lam.conj().T, atol=1e-12)
        assert np.trace(lam) == pytest.approx(0.0, abs=1e-12)

    def test_weyl_x_cubed_is_identity(self):
        np.testing.assert_allclose(
            weyl_x @ weyl_x @ weyl_x, np.eye(3), atol=1e-10
        )

    def test_weyl_z_cubed_is_identity(self):
        np.testing.assert_allclose(
            weyl_z @ weyl_z @ weyl_z, np.eye(3), atol=1e-10
        )

    def test_weyl_clock_shift_commutation(self):
        # Canonical qutrit clock-shift relation: X Z = omega^-1 Z X.
        np.testing.assert_allclose(
            weyl_x @ weyl_z, np.conj(_omega) * weyl_z @ weyl_x, atol=1e-10
        )

    def test_weyl_xz_is_x_then_z_product(self):
        np.testing.assert_allclose(weyl_xz, weyl_x @ weyl_z)


class TestRotationFunctions:

    @pytest.mark.parametrize('fn', [rx01, rx12, ry01, ry12, rz01, rz12])
    def test_zero_angle_is_identity(self, fn):
        np.testing.assert_allclose(fn(0.0), np.eye(3), atol=1e-12)

    @pytest.mark.parametrize('fn', [rx01, rx12, ry01, ry12, rz01, rz12])
    def test_is_unitary(self, fn):
        assert _is_unitary(fn(0.42))

    @pytest.mark.parametrize('fn01, fn', [
        (rx01, rx), (ry01, ry), (rz01, rz),
    ])
    def test_01_subspace_embeds_qubit_rotation_in_top_left_block(
        self, fn01, fn
    ):
        theta = 0.37
        np.testing.assert_allclose(fn01(theta)[0:2, 0:2], fn(theta))

    def test_01_subspace_leaves_level_2_untouched(self):
        theta = 0.37
        U = rx01(theta)
        np.testing.assert_allclose(U[2, :], [0, 0, 1])
        np.testing.assert_allclose(U[:, 2], [0, 0, 1])

    @pytest.mark.parametrize('fn12, fn', [
        (rx12, rx), (ry12, ry), (rz12, rz),
    ])
    def test_12_subspace_embeds_qubit_rotation_in_bottom_right_block(
        self, fn12, fn
    ):
        theta = 0.37
        np.testing.assert_allclose(fn12(theta)[1:3, 1:3], fn(theta))

    def test_12_subspace_leaves_level_0_untouched(self):
        theta = 0.37
        U = rx12(theta)
        np.testing.assert_allclose(U[0, :], [1, 0, 0])
        np.testing.assert_allclose(U[:, 0], [1, 0, 0])

    @pytest.mark.parametrize('k', range(1, 9))
    def test_rgm_zero_angle_is_identity(self, k):
        np.testing.assert_allclose(rgm(k, 0.0), np.eye(3), atol=1e-12)

    @pytest.mark.parametrize('k', range(1, 9))
    def test_rgm_is_unitary(self, k):
        assert _is_unitary(rgm(k, 0.6))

    @pytest.mark.parametrize('k, fn', [
        (1, rx01), (2, ry01), (3, rz01), (6, rx12), (7, ry12),
    ])
    def test_rgm_matches_named_rotation(self, k, fn):
        theta = 0.6
        np.testing.assert_allclose(rgm(k, theta), fn(theta), atol=1e-10)


class TestGateClasses:

    def test_h3_matches_matrix_and_subspace(self):
        gate = H3(0)
        np.testing.assert_allclose(gate.unitary, h3)
        assert gate.subspace == 'GEF'
        assert gate.alias == 'QFT3'

    def test_id3_is_identity_with_params(self):
        gate = Id3(0)
        np.testing.assert_allclose(gate.unitary, id3)
        assert gate.properties['params'] == {'angle': 0, 'duration': 0.}

    @pytest.mark.parametrize('gate_cls, expected, subspace', [
        (X01, x01, 'GE'), (X02, x02, 'GF'), (X12, x12, 'EF'),
        (Y01, y01, 'GE'), (Y02, y02, 'GF'), (Y12, y12, 'EF'),
        (Z01, z01, 'GE'), (Z12, z12, 'EF'),
    ])
    def test_pauli_type_gates(self, gate_cls, expected, subspace):
        gate = gate_cls(0)
        np.testing.assert_allclose(gate.unitary, expected)
        assert gate.subspace == subspace
        assert gate.qubits == (0,)

    @pytest.mark.parametrize('gate_cls, fn, subspace', [
        (X90_01, rx01, 'GE'), (X90_12, rx12, 'EF'),
        (Y90_01, ry01, 'GE'), (Y90_12, ry12, 'EF'),
    ])
    def test_90_degree_gates(self, gate_cls, fn, subspace):
        gate = gate_cls(0)
        np.testing.assert_allclose(gate.unitary, fn(np.pi / 2))
        assert gate.subspace == subspace

    @pytest.mark.parametrize('gate_cls, fn, subspace', [
        (Rx01, rx01, 'GE'), (Rx12, rx12, 'EF'),
        (Ry01, ry01, 'GE'), (Ry12, ry12, 'EF'),
        (Rz01, rz01, 'GE'), (Rz12, rz12, 'EF'),
    ])
    def test_parametric_rotation_gates(self, gate_cls, fn, subspace):
        theta = 0.42
        gate = gate_cls(0, theta)
        np.testing.assert_allclose(gate.unitary, fn(theta))
        assert gate.subspace == subspace

    def test_virtualz01_matches_rz01_but_different_name(self):
        theta = 0.5
        np.testing.assert_allclose(
            VirtualZ01(0, theta).unitary, Rz01(0, theta).unitary
        )
        assert VirtualZ01(0, theta).name == 'VirtualZ01'

    def test_virtualz12_matches_rz12_but_different_name(self):
        theta = 0.5
        np.testing.assert_allclose(
            VirtualZ12(0, theta).unitary, Rz12(0, theta).unitary
        )
        assert VirtualZ12(0, theta).name == 'VirtualZ12'

    @pytest.mark.parametrize('gate_cls, expected, alias', [
        (WeylX, weyl_x, 'W10'), (WeylX2, weyl_x2, 'W20'),
        (WeylZ, weyl_z, 'W01'), (WeylZ2, weyl_z2, 'W02'),
        (WeylXZ, weyl_xz, 'W11'),
    ])
    def test_weyl_gates(self, gate_cls, expected, alias):
        gate = gate_cls(0)
        np.testing.assert_allclose(gate.unitary, expected)
        assert gate.alias == alias
        assert gate.subspace == 'GEF'

    @pytest.mark.parametrize('gate_cls, k, subspace', [
        (RGM1, 1, 'GE'), (RGM2, 2, 'GE'), (RGM3, 3, 'GE'),
        (RGM4, 4, 'GF'), (RGM5, 5, 'GF'),
        (RGM6, 6, 'EF'), (RGM7, 7, 'EF'),
        (RGM8, 8, 'GEF'),
    ])
    def test_rgm_gates(self, gate_cls, k, subspace):
        theta = 0.6
        gate = gate_cls(0, theta)
        np.testing.assert_allclose(gate.unitary, rgm(k, theta))
        assert gate.subspace == subspace

    def test_is_single_qudit_not_single_qubit(self):
        gate = H3(0)
        assert gate.is_single_qudit
        assert not gate.is_single_qubit
        assert gate.dim == 3

    def test_ef_subspace_does_not_further_embed_3x3_matrix(self):
        # Already a 3x3 matrix, so Gate.unitary's EF-embedding logic
        # (which only fires for a 2x2 matrix) is a no-op here.
        gate = Rx12(0, 0.42)
        np.testing.assert_allclose(gate.unitary, gate.matrix)


class TestRegistry:

    @pytest.mark.parametrize('name, gate_cls', [
        ('H3', H3), ('Id3', Id3), ('X01', X01), ('Z12', Z12),
        ('RGM1', RGM1), ('WeylX', WeylX),
    ])
    def test_registry_maps_name_to_class(self, name, gate_cls):
        assert SINGLE_QUTRIT_GATES[name] is gate_cls

    def test_unsupported_name_returns_default_message_not_keyerror(self):
        assert (
            SINGLE_QUTRIT_GATES['bogus'] == 'Gate not currently supported!'
        )
