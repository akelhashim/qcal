"""Unit tests for qcal.math.utils.

Regression coverage for a bug found while testing JAZZ
(test_zz.py): when a decaying-cosine fit's stderr is None on both
branches, callers like qcal.characterization.phase_estimation.zz
fall back to err=0.0 (e.g. via np.linalg.norm([0., 0.])).
round_to_order_error used to compute np.log10(0) in that case, which
is -inf, and int(np.around(-inf)) raised an OverflowError instead of
returning a sensible result.
"""
import numpy as np
import pytest

from qcal.math.utils import (
    round_sig_figures,
    round_to_order_error,
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
