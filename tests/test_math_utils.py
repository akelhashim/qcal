"""Unit tests for qcal.math.utils.round_to_order_error.

Regression coverage for a bug found while testing JAZZ
(test_zz.py): when a decaying-cosine fit's stderr is None on both
branches, callers like qcal.characterization.phase_estimation.zz
fall back to err=0.0 (e.g. via uncertainty_of_sum([0., 0.])).
round_to_order_error used to compute np.log10(0) in that case, which
is -inf, and int(np.around(-inf)) raised an OverflowError instead of
returning a sensible result.
"""
import pytest

from qcal.math.utils import round_to_order_error


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
