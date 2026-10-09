"""Unit tests for qcal.characterization.spectroscopy.utils.

find_inflection_points() locates curvature (sign-of-second-derivative)
changes in a Gaussian-smoothed signal. The Gaussian filter's edge
handling introduces spurious curvature near the first/last ~20 samples
of any input (even a pure parabola, whose true curvature never changes
sign), so these tests restrict assertions to the interior of the
array, away from that boundary artifact.
"""
import numpy as np
import pytest

from qcal.characterization.spectroscopy.utils import find_inflection_points

_EDGE_MARGIN = 20


def _interior(idx, x):
    """idx entries that are not within _EDGE_MARGIN samples of an edge."""
    return idx[(idx > _EDGE_MARGIN) & (idx < len(x) - _EDGE_MARGIN)]


class TestFindInflectionPoints:

    def test_parabola_has_no_interior_inflection(self):
        # A parabola's curvature never changes sign.
        x = np.linspace(-10, 10, 501)
        y = x ** 2
        idx = find_inflection_points(x, y, sigma=5)
        assert len(_interior(idx, x)) == 0

    def test_cubic_has_single_interior_inflection_at_zero(self):
        # x**3 has curvature 6x, which changes sign only at x=0.
        x = np.linspace(-10, 10, 501)
        y = x ** 3
        idx = find_inflection_points(x, y, sigma=1)
        interior = _interior(idx, x)
        assert len(interior) == 1
        assert x[interior[0]] == pytest.approx(0.0, abs=0.2)

    def test_sine_has_periodic_inflections_at_multiples_of_pi(self):
        # sin(x) has curvature -sin(x), which changes sign at n*pi.
        x = np.linspace(0, 4 * np.pi, 801)
        y = np.sin(x)
        idx = find_inflection_points(x, y, sigma=1)
        interior = _interior(idx, x)
        expected = np.array([1, 2, 3]) * np.pi
        assert len(interior) == len(expected)
        assert x[interior] == pytest.approx(expected, abs=0.1)
