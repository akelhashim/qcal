"""Unit tests for qcal.gates.two_qubit.

Covers the _normalize_qubits/nearest_kronecker_product helpers, every
concrete gate class (checked against its underlying matrix constant
or function, plus a generic unitarity check), and the TWO_QUBIT_GATES
registry.
"""
import numpy as np
import pytest

from qcal.gates.single_qubit import x, y
from qcal.gates.two_qubit import (
    CH,
    CNOT,
    CS,
    CT,
    CV,
    CX,
    CY,
    CZ,
    DB,
    DCNOT,
    ECP,
    MS,
    QFT2,
    SWAP,
    TWO_QUBIT_GATES,
    XX,
    XY,
    YY,
    ZZ,
    AGate,
    Barenco,
    BGate,
    CPhase,
    CRot,
    CSdag,
    CTdag,
    FSim,
    Givens,
    M,
    SqrtSWAP,
    SqrtSWAPdag,
    SWAPAlpha,
    Syc,
    WGate,
    _normalize_qubits,
    a_gate,
    b_gate,
    barenco,
    bSWAP,
    bswap,
    ch,
    cnot,
    crot,
    cv,
    cy,
    cz,
    db,
    dcnot,
    ecp,
    fSWAP,
    fswap,
    iSWAP,
    iswap,
    m,
    ms,
    nearest_kronecker_product,
    pSWAP,
    qft2,
    sqrt_swap,
    sqrt_swap_dag,
    swap,
    swap_alpha,
    syc,
    w_gate,
)


def _is_unitary(matrix, atol=1e-8):
    return np.allclose(
        matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=atol
    )


class TestNormalizeQubits:

    def test_tuple_form(self):
        assert _normalize_qubits(((0, 1),)) == (0, 1)

    def test_separate_args_form(self):
        assert _normalize_qubits((0, 1)) == (0, 1)


class TestNearestKroneckerProduct:

    def test_recovers_kronecker_product(self):
        C = np.kron(x, y)
        A, B = nearest_kronecker_product(C)
        np.testing.assert_allclose(np.kron(A, B), C, atol=1e-10)


class TestFixedMatrixGates:
    """Parameter-free gates, checked against the module's own matrix
    constants.
    """

    @pytest.mark.parametrize('gate_cls, expected', [
        (BGate, b_gate),
        (bSWAP, bswap),
        (CH, ch),
        (CNOT, cnot),
        (CV, cv),
        (CX, cnot),
        (CY, cy),
        (CZ, cz),
        (DB, db),
        (DCNOT, dcnot),
        (ECP, ecp),
        (fSWAP, fswap),
        (iSWAP, iswap),
        (M, m),
        (MS, ms),
        (QFT2, qft2),
        (SWAP, swap),
        (SqrtSWAP, sqrt_swap),
        (SqrtSWAPdag, sqrt_swap_dag),
        (Syc, syc),
        (WGate, w_gate),
    ])
    def test_matches_expected_matrix(self, gate_cls, expected):
        np.testing.assert_allclose(gate_cls(0, 1).unitary, expected)

    @pytest.mark.parametrize('gate_cls', [
        BGate, bSWAP, CH, CNOT, CV, CX, CY, CZ, DB, DCNOT, ECP, fSWAP,
        iSWAP, M, MS, QFT2, SWAP, SqrtSWAP, SqrtSWAPdag, Syc, WGate,
    ])
    def test_is_multi_qubit_and_unitary(self, gate_cls):
        gate = gate_cls(0, 1)
        assert gate.is_multi_qubit
        assert gate.qubits == (0, 1)
        assert _is_unitary(gate.unitary)

    @pytest.mark.parametrize('gate_cls', [CNOT, CZ, SWAP])
    def test_tuple_and_separate_args_agree(self, gate_cls):
        np.testing.assert_allclose(
            gate_cls(0, 1).unitary, gate_cls((0, 1)).unitary
        )


class TestCPhaseFamily:
    """CS/CSdag/CT/CTdag are CPhase at fixed angles."""

    def test_cphase_theta_1_matches_cz(self):
        np.testing.assert_allclose(CPhase(0, 1, theta=1.).unitary, cz)
        assert CPhase(0, 1, theta=1.).properties['alias'] == 'CZ'

    @pytest.mark.parametrize('gate_cls, theta', [
        (CS, 0.5), (CSdag, -0.5), (CT, 0.25), (CTdag, -0.25),
    ])
    def test_matches_cphase_at_fixed_theta(self, gate_cls, theta):
        np.testing.assert_allclose(
            gate_cls(0, 1).unitary, CPhase(0, 1, theta=theta).unitary
        )

    def test_csdag_name_matches_class_name(self):
        assert CSdag(0, 1).name == 'CSdag'


class TestParametrizedGates:

    def test_crot_matches_function(self):
        theta, n = 0.4, (0, 0, 1)
        np.testing.assert_allclose(
            CRot(0, 1, theta=theta, n=n).unitary, crot(theta, n)
        )

    def test_fsim_matches_function(self):
        from qcal.gates.two_qubit import fsim
        theta, phi = 0.3, 0.5
        np.testing.assert_allclose(
            FSim(0, 1, theta=theta, phi=phi).unitary, fsim(theta, phi)
        )

    def test_givens_matches_function(self):
        from qcal.gates.two_qubit import givens
        theta = 0.4
        np.testing.assert_allclose(
            Givens(0, 1, theta=theta).unitary, givens(theta)
        )

    def test_pswap_matches_function(self):
        from qcal.gates.two_qubit import pswap
        theta = 0.4
        np.testing.assert_allclose(
            pSWAP(0, 1, theta=theta).unitary, pswap(theta)
        )

    @pytest.mark.parametrize('gate_cls, fn_name', [
        (XX, 'xx'), (XY, 'xy'), (YY, 'yy'), (ZZ, 'zz'),
    ])
    def test_xx_family_matches_function_and_identity_at_t0(
        self, gate_cls, fn_name
    ):
        import qcal.gates.two_qubit as tq
        fn = getattr(tq, fn_name)
        np.testing.assert_allclose(gate_cls(0, 1, t=0.3).unitary, fn(0.3))
        np.testing.assert_allclose(
            gate_cls(0, 1, t=0.).unitary, np.eye(4), atol=1e-10
        )

    def test_agate_special_cases(self):
        np.testing.assert_allclose(
            AGate(0, 1, theta=0., phi=0.).unitary, a_gate(0., 0.)
        )
        assert AGate(
            0, 1, theta=0., phi=0.
        ).properties['locally_equivalent'] == 'CX'

    def test_barenco_matches_function(self):
        phi, alpha, theta = 0.2, 0.3, 0.4
        np.testing.assert_allclose(
            Barenco(0, 1, phi=phi, alpha=alpha, theta=theta).unitary,
            barenco(phi, alpha, theta),
        )

    def test_swap_alpha_matches_function(self):
        alpha = 0.4
        np.testing.assert_allclose(
            SWAPAlpha(0, 1, alpha=alpha).unitary, swap_alpha(alpha)
        )

    @pytest.mark.parametrize('gate_cls, kwargs', [
        (CRot, {'theta': 0.4, 'n': (0, 0, 1)}),
        (FSim, {'theta': 0.3, 'phi': 0.5}),
        (Givens, {'theta': 0.4}),
        (pSWAP, {'theta': 0.4}),
        (XX, {'t': 0.3}),
        (XY, {'t': 0.3}),
        (YY, {'t': 0.3}),
        (ZZ, {'t': 0.3}),
        (AGate, {'theta': 0.2, 'phi': 0.1}),
        (Barenco, {'phi': 0.2, 'alpha': 0.3, 'theta': 0.4}),
        (SWAPAlpha, {'alpha': 0.4}),
    ])
    def test_is_unitary(self, gate_cls, kwargs):
        gate = gate_cls(0, 1, **kwargs)
        assert _is_unitary(gate.unitary)


class TestRegistry:

    @pytest.mark.parametrize('name, gate_cls', [
        ('CNOT', CNOT), ('CZ', CZ), ('SWAP', SWAP), ('iSWAP', iSWAP),
    ])
    def test_registry_maps_name_to_class(self, name, gate_cls):
        assert TWO_QUBIT_GATES[name] is gate_cls

    def test_unsupported_name_returns_default_message_not_keyerror(self):
        assert (
            TWO_QUBIT_GATES['bogus'] == 'Gate not currently supported!'
        )
