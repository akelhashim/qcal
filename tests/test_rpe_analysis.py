"""Unit tests for qcal.characterization.phase_estimation.analysis.rectify_angle.

rectify_angle wraps an angle into [-pi, pi]. The full RPE run tests
(test_rpe.py) only exercise this indirectly -- through whatever angle
estimates the RNG happens to produce -- so the wrap-around boundary
behavior at +/- pi is tested directly here instead.
"""
import numpy as np
import pytest

from qcal.characterization.phase_estimation.analysis import rectify_angle


class TestRectifyAngle:

    def test_angle_inside_range_is_unchanged(self):
        assert rectify_angle(0.1) == pytest.approx(0.1)
        assert rectify_angle(-0.1) == pytest.approx(-0.1)

    def test_zero_is_unchanged(self):
        assert rectify_angle(0.0) == pytest.approx(0.0)

    def test_positive_pi_wraps_to_negative_pi(self):
        assert rectify_angle(np.pi) == pytest.approx(-np.pi)

    def test_negative_pi_is_unchanged(self):
        assert rectify_angle(-np.pi) == pytest.approx(-np.pi)

    def test_angle_just_past_positive_pi_wraps_to_negative_side(self):
        assert rectify_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)

    def test_angle_just_past_negative_pi_wraps_to_positive_side(self):
        assert rectify_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)

    def test_three_halves_pi_wraps_to_negative_half_pi(self):
        assert rectify_angle(3 * np.pi / 2) == pytest.approx(-np.pi / 2)

    def test_negative_three_halves_pi_wraps_to_positive_half_pi(self):
        assert rectify_angle(-3 * np.pi / 2) == pytest.approx(np.pi / 2)

    def test_multiple_full_turns_reduce_to_equivalent_angle(self):
        assert rectify_angle(0.3 + 4 * np.pi) == pytest.approx(0.3)
        assert rectify_angle(0.3 - 4 * np.pi) == pytest.approx(0.3)
