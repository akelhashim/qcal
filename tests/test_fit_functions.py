"""Unit tests for qcal.fitting.fit_functions.

Each function is a plain, vectorized math expression, so these tests
just check the formula against hand-computed values (both scalar and
array inputs) rather than exercising any fitting machinery.
"""
import numpy as np
import pytest

from qcal.fitting.fit_functions import (
    absolute_value,
    base_exponential,
    cosine,
    decaying_cosine,
    decaying_cosine_exponential,
    exponential,
    linear,
    parabola,
)


class TestAbsoluteValue:

    def test_scalar(self):
        assert absolute_value(5.0, a=2.0, b=1.0, c=3.0) == pytest.approx(
            2.0 * abs(5.0 - 1.0) + 3.0
        )

    def test_minimum_at_x_offset(self):
        assert absolute_value(2.0, a=3.0, b=2.0, c=1.0) == pytest.approx(1.0)

    def test_array(self):
        x = np.array([0.0, 1.0, 2.0])
        expected = np.array([1.0, 0.0, 1.0])
        np.testing.assert_allclose(
            absolute_value(x, a=1.0, b=1.0, c=0.0), expected
        )


class TestBaseExponential:

    def test_matches_exponential_for_base_e(self):
        # base_exponential uses base**(b*x); exponential uses exp(-b*x),
        # so the decay constants are negatives of each other.
        x = np.linspace(0, 5, 11)
        np.testing.assert_allclose(
            base_exponential(x, np.e, a=2.0, b=-0.5, c=1.0),
            exponential(x, a=2.0, b=0.5, c=1.0),
        )

    def test_scalar(self):
        assert base_exponential(
            2.0, base=2.0, a=1.0, b=1.0, c=0.0
        ) == pytest.approx(2.0 ** 2.0)


class TestCosine:

    def test_zero_phase_at_origin(self):
        assert cosine(
            0.0, amp=2.0, freq=1.0, phase=0.0, offset=0.0
        ) == pytest.approx(2.0)

    def test_quarter_period_is_offset(self):
        freq = 0.5
        quarter_period = 1 / freq / 4
        assert cosine(
            quarter_period, amp=3.0, freq=freq, phase=0.0, offset=1.0
        ) == pytest.approx(1.0, abs=1e-9)

    def test_periodicity(self):
        x = np.linspace(0, 3, 13)
        freq = 2.0
        period = 1 / freq
        np.testing.assert_allclose(
            cosine(x, amp=1.0, freq=freq, phase=0.3, offset=0.0),
            cosine(x + period, amp=1.0, freq=freq, phase=0.3, offset=0.0),
            atol=1e-9,
        )


class TestDecayingCosine:

    def test_at_origin_equals_amplitude_plus_offset(self):
        assert decaying_cosine(
            0.0, a=2.0, b=0.5, c=1.0, d=0.0, e=0.5
        ) == pytest.approx(2.5)

    def test_decays_toward_offset(self):
        value = decaying_cosine(100.0, a=2.0, b=0.5, c=1.0, d=0.0, e=0.5)
        assert value == pytest.approx(0.5, abs=1e-6)


class TestDecayingCosineExponential:

    def test_at_origin(self):
        expected = 2.0 * np.cos(0.0) + 1.0 + 0.5
        assert decaying_cosine_exponential(
            0.0, a=2.0, b=0.5, c=1.0, d=0.0, e=1.0, f=0.3, g=0.5
        ) == pytest.approx(expected)

    def test_decays_toward_constant_offset(self):
        value = decaying_cosine_exponential(
            100.0, a=2.0, b=0.5, c=1.0, d=0.0, e=1.0, f=0.3, g=0.5
        )
        assert value == pytest.approx(0.5, abs=1e-6)


class TestExponential:

    def test_at_origin_equals_amplitude_plus_offset(self):
        assert exponential(0.0, a=2.0, b=0.5, c=1.0) == pytest.approx(3.0)

    def test_decays_toward_offset(self):
        assert exponential(
            100.0, a=2.0, b=0.5, c=1.0
        ) == pytest.approx(1.0, abs=1e-6)


class TestLinear:

    def test_scalar(self):
        assert linear(3.0, m=2.0, b=1.0) == pytest.approx(7.0)

    def test_array(self):
        x = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            linear(x, m=2.0, b=1.0), np.array([1.0, 3.0, 5.0])
        )


class TestParabola:

    def test_vertex(self):
        assert parabola(0.0, a=1.0, b=0.0, c=5.0) == pytest.approx(5.0)

    def test_array(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            parabola(x, a=1.0, b=-2.0, c=1.0),
            np.array([4.0, 1.0, 0.0, 1.0]),
        )
