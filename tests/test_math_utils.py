"""Unit tests for qcal.math.utils.

Regression coverage for a bug found while testing JAZZ
(test_zz.py): when a decaying-cosine fit's stderr is None on both
branches, callers like qcal.characterization.phase_estimation.zz
fall back to err=0.0 (e.g. via uncertainty_of_sum([0., 0.])).
round_to_order_error used to compute np.log10(0) in that case, which
is -inf, and int(np.around(-inf)) raised an OverflowError instead of
returning a sensible result.
"""
import numpy as np
import pytest

from qcal.math.utils import (
    reciprocal_uncertainty,
    round_sig_figures,
    round_to_order_error,
    uncertainty_of_exp,
    uncertainty_of_product,
    uncertainty_of_sum,
    wrap_phase,
)


class TestRoundToOrderError:

    def test_normal_rounding(self):
        val, err = round_to_order_error(1234.5678, 12.3)
        assert val == pytest.approx(1230.0)
        assert err == pytest.approx(12.0)

    def test_zero_error_returns_value_unrounded(self):
        val, err = round_to_order_error(-199277.48425938096, 0.0)
        assert val == pytest.approx(-199277.48425938096)
        assert err == 0.0

    def test_zero_value_and_zero_error(self):
        val, err = round_to_order_error(0.0, 0.0)
        assert val == 0.0
        assert err == 0.0

    def test_error_greater_than_value(self):
        val, err = round_to_order_error(0.0, 5.0)
        assert val == pytest.approx(0.0)
        assert err == pytest.approx(5.0)


class TestReciprocalUncertainty:

    def test_matches_error_propagation_formula(self):
        rec_x, rec_sigma_x = reciprocal_uncertainty(2.0, 0.1)
        assert rec_x == pytest.approx(0.5)
        assert rec_sigma_x == pytest.approx(0.025)

    def test_uncertainty_is_always_nonnegative(self):
        _, rec_sigma_x = reciprocal_uncertainty(-2.0, 0.1)
        assert rec_sigma_x >= 0


class TestRoundSigFigures:

    def test_rounds_to_requested_significant_figures(self):
        assert round_sig_figures(1234.5678, 1) == 1000.0
        assert round_sig_figures(1234.5678, 2) == 1200.0
        assert round_sig_figures(1234.5678, 3) == 1230.0

    def test_small_number(self):
        assert round_sig_figures(0.0004567, 2) == pytest.approx(0.00046)

    def test_negative_number(self):
        assert round_sig_figures(-1234.5678, 3) == pytest.approx(-1230.0)

    def test_zero_does_not_crash(self):
        # Regression test: this used to compute log10(0) == -inf, and
        # int(floor(-inf)) raised OverflowError (same bug class fixed
        # separately in round_to_order_error).
        assert round_sig_figures(0.0) == 0.0


class TestUncertaintyOfExp:

    def test_matches_error_propagation_formula(self):
        # f(val) = val**p, sigma_f = |p| * val**(p - 1) * sigma_val
        assert uncertainty_of_exp(2.0, 0.1, 2) == pytest.approx(0.4)

    def test_negative_power(self):
        assert uncertainty_of_exp(2.0, 0.1, -1) == pytest.approx(0.025)


class TestUncertaintyOfProduct:

    def test_matches_formula_without_covariance(self):
        err = uncertainty_of_product(
            np.array([2.0, 3.0]), stds=np.array([0.1, 0.2])
        )
        expected = abs(2.0 * 3.0) * np.sqrt(
            (0.1 / 2.0) ** 2 + (0.2 / 3.0) ** 2
        )
        assert err == pytest.approx(expected)

    def test_diagonal_covariance_matches_no_covariance_case(self):
        stds = np.array([0.1, 0.2])
        values = np.array([2.0, 3.0])
        cov = np.diag(stds ** 2)

        err_cov = uncertainty_of_product(values, cov=cov)
        err_stds = uncertainty_of_product(values, stds=stds)

        assert err_cov == pytest.approx(err_stds)

    def test_missing_stds_and_cov_raises(self):
        with pytest.raises(ValueError):
            uncertainty_of_product(np.array([2.0, 3.0]))


class TestUncertaintyOfSum:

    def test_quadrature_sum(self):
        assert uncertainty_of_sum([3.0, 4.0]) == pytest.approx(5.0)

    def test_empty_list_gives_zero(self):
        assert uncertainty_of_sum([]) == 0.0

    def test_single_error_is_unchanged(self):
        assert uncertainty_of_sum([7.0]) == pytest.approx(7.0)


class TestWrapPhase:

    def test_phase_within_range_is_unchanged(self):
        assert wrap_phase(0.0) == pytest.approx(0.0)
        assert wrap_phase(np.pi / 2) == pytest.approx(np.pi / 2)

    def test_phase_above_pi_wraps_around(self):
        assert wrap_phase(3 * np.pi) == pytest.approx(-np.pi)

    def test_phase_below_negative_pi_wraps_around(self):
        assert wrap_phase(-3 * np.pi) == pytest.approx(-np.pi)

    def test_boundary_pi_wraps_to_negative_pi(self):
        assert wrap_phase(np.pi) == pytest.approx(-np.pi)
