"""Unit tests for qcal.sequence.utils."""
import numpy as np
import pytest

from qcal.sequence.utils import (
    clip_amplitude,
    compute_matrix_A,
    cosine_basis_function,
    fourier_transform_basis,
    solve_coefficients,
)


class TestClipAmplitude:

    def test_scalar_within_range_is_unchanged(self):
        assert clip_amplitude(0.5) == pytest.approx(0.5)

    def test_scalar_above_max_is_clipped(self):
        assert clip_amplitude(2.0) == pytest.approx(1.0)

    def test_scalar_below_min_is_clipped(self):
        assert clip_amplitude(-2.0) == pytest.approx(-1.0)

    def test_custom_bounds(self):
        assert (
            clip_amplitude(5.0, min_amp=0.0, max_amp=3.0) == pytest.approx(3.0)
        )

    def test_array_is_clipped_elementwise(self):
        result = clip_amplitude(np.array([-2.0, 0.0, 2.0]))
        np.testing.assert_allclose(result, [-1.0, 0.0, 1.0])


class TestCosineBasisFunction:

    def test_zero_outside_window(self):
        t = np.array([-1.0, 11.0])
        result = cosine_basis_function(1, t, t_p=10.0)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_zero_at_window_edges(self):
        t = np.array([0.0, 10.0])
        result = cosine_basis_function(2, t, t_p=10.0)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-12)

    def test_symmetric_about_window_center(self):
        t_p = 10.0
        t = np.linspace(0, t_p, 101)
        result = cosine_basis_function(3, t, t_p)
        np.testing.assert_allclose(result, result[::-1], atol=1e-10)


class TestFourierTransformBasis:

    def test_zero_frequency_returns_half_pulse_duration(self):
        result = fourier_transform_basis(1, np.array([0.0]), t_p=10.0)
        assert result[0] == pytest.approx(5.0)

    def test_handles_scalar_and_array_frequencies_consistently(self):
        f = np.array([0.0, 1.0, -1.0, 5.0])
        result = fourier_transform_basis(2, f, t_p=10.0)
        assert result.shape == f.shape
        assert np.all(np.isfinite(result))

    def test_near_zero_frequency_uses_zero_branch(self):
        result = fourier_transform_basis(1, np.array([1e-13]), t_p=10.0)
        assert result[0] == pytest.approx(5.0)


class TestComputeMatrixA:

    def test_shape(self):
        A = compute_matrix_A(3, [[0.0, 1.0]], [1.0], t_p=10.0)
        assert A.shape == (3, 3)

    def test_symmetric(self):
        A = compute_matrix_A(
            4, [[0.0, 1.0], [2.0, 3.0]], [1.0, 0.5], t_p=10.0
        )
        np.testing.assert_allclose(A, A.T)

    def test_real_valued(self):
        A = compute_matrix_A(3, [[0.0, 1.0]], [1.0], t_p=10.0)
        assert np.isrealobj(A)


class TestSolveCoefficients:

    def test_returns_n_coefficients(self):
        coeffs = solve_coefficients(
            N=3,
            t_p=10.0,
            theta=np.pi / 2,
            freq_intervals=[[0.0, 1.0], [2.0, 3.0]],
            weights=[1.0, 0.5],
        )
        assert coeffs.shape == (3,)

    def test_satisfies_rotation_angle_constraint(self):
        theta = np.pi / 2
        t_p = 10.0
        coeffs = solve_coefficients(
            N=4,
            t_p=t_p,
            theta=theta,
            freq_intervals=[[0.0, 1.0], [2.0, 3.0]],
            weights=[1.0, 0.5],
        )
        assert np.sum(coeffs) == pytest.approx(theta / t_p)

    def test_falls_back_to_pseudoinverse_on_singular_matrix(self, caplog):
        # N=0 produces a degenerate 1x1 augmented matrix (just the
        # Lagrange-multiplier block, which is exactly zero), forcing
        # the except-branch pseudoinverse fallback.
        with caplog.at_level('WARNING'):
            coeffs = solve_coefficients(
                N=0,
                t_p=10.0,
                theta=np.pi / 2,
                freq_intervals=[[0.0, 1.0]],
                weights=[1.0],
            )
        assert coeffs.shape == (0,)
        assert 'pseudoinverse' in caplog.text
