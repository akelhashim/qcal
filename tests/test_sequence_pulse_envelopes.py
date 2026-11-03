"""Unit tests for qcal.sequence.pulse_envelopes."""
import numpy as np
import pytest

from qcal.sequence.pulse_envelopes import (
    CW,
    DRAG,
    FAST,
    FAST_DRAG,
    IPM,
    RAP,
    STARAP,
    cosine_square,
    custom,
    gaussian,
    gaussian_square,
    linear,
    pulse_envelopes,
    sine,
    square,
    super_gaussian,
    virtualz,
)

SR = 1e9
LENGTH = 100e-9
N_POINTS = int(round(LENGTH * SR))


class TestCosineSquare:

    def test_default_length_and_dtype(self):
        result = cosine_square(LENGTH, SR)
        assert len(result) == N_POINTS
        assert result.dtype == np.complex64

    def test_zero_ramp_fraction_is_a_plain_square_pulse(self):
        result = cosine_square(LENGTH, SR, ramp_fraction=0.0)
        assert len(result) == N_POINTS
        np.testing.assert_allclose(result, np.ones(N_POINTS))

    @pytest.mark.parametrize('ramp_fraction', [0.0, 0.1, 0.25, 0.5])
    def test_symmetric_ramp_keeps_total_length(self, ramp_fraction):
        result = cosine_square(LENGTH, SR, ramp_fraction=ramp_fraction)
        assert len(result) == N_POINTS

    @pytest.mark.parametrize(
        'ramp_fraction', [-0.1, -0.25, -0.5, -0.6, 0.6]
    )
    def test_one_sided_ramp_keeps_total_length(self, ramp_fraction):
        result = cosine_square(LENGTH, SR, ramp_fraction=ramp_fraction)
        assert len(result) == N_POINTS

    def test_positive_ramp_starts_low_and_flattens_to_amp(self):
        result = cosine_square(LENGTH, SR, amp=1.0, ramp_fraction=0.25).real
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[N_POINTS // 2] == pytest.approx(1.0)

    def test_negative_ramp_fraction_ramps_down_at_the_end_only(self):
        result = cosine_square(LENGTH, SR, amp=1.0, ramp_fraction=-0.25).real
        # One-sided: flat (amp) at the start, ramping down near the end.
        assert result[0] == pytest.approx(1.0)
        assert result[-1] < 0.01

    def test_large_positive_ramp_fraction_warns_and_ramps_up_only(
        self, caplog
    ):
        with caplog.at_level('WARNING'):
            result = cosine_square(LENGTH, SR, amp=1.0, ramp_fraction=0.6).real
        assert 'greater than 0.5' in caplog.text
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[-1] == pytest.approx(1.0)

    def test_amp_and_phase_are_applied(self):
        result = cosine_square(LENGTH, SR, amp=2.0, phase=np.pi, ramp_fraction=0.0)
        np.testing.assert_allclose(result, -2.0 * np.ones(N_POINTS), atol=1e-6)


class TestCW:

    def test_returns_literal_cw_string(self):
        assert CW(LENGTH, SR) == 'cw'


class TestCustom:

    def test_loads_npy_file_as_complex64(self, tmp_path):
        data = np.array([0.1, 0.5, 1.0], dtype=np.complex64)
        path = tmp_path / 'pulse.npy'
        np.save(path, data)

        result = custom(LENGTH, SR, str(path))

        np.testing.assert_allclose(result, data)
        assert result.dtype == np.complex64


class TestDRAG:

    def test_zero_alpha_matches_plain_gaussian(self):
        drag = DRAG(LENGTH, SR, alpha=0.0)
        gauss = gaussian(LENGTH, SR)
        np.testing.assert_allclose(drag, gauss)

    def test_nonzero_alpha_adds_an_imaginary_component(self):
        drag = DRAG(LENGTH, SR, alpha=0.5)
        assert np.any(drag.imag != 0.0)

    def test_length_and_dtype(self):
        drag = DRAG(LENGTH, SR, alpha=0.5)
        assert len(drag) == N_POINTS
        assert drag.dtype == np.complex64


class TestGaussian:

    def test_length_and_dtype(self):
        result = gaussian(LENGTH, SR)
        assert len(result) == N_POINTS
        assert result.dtype == np.complex64

    def test_peak_is_at_pulse_center(self):
        result = gaussian(LENGTH, SR).real
        assert np.argmax(result) == pytest.approx(N_POINTS / 2, abs=1)

    def test_amplitude_scales_peak(self):
        result = gaussian(LENGTH, SR, amp=2.0).real
        assert np.max(result) == pytest.approx(2.0, rel=1e-2)


class TestGaussianSquare:

    def test_length_and_dtype(self):
        result = gaussian_square(LENGTH, SR)
        assert len(result) == N_POINTS
        assert result.dtype == np.complex64

    def test_ramp_fraction_above_half_raises_value_error(self):
        with pytest.raises(ValueError, match='ramp_fraction'):
            gaussian_square(LENGTH, SR, ramp_fraction=0.6)

    def test_zero_ramp_fraction_is_a_plain_square_pulse(self):
        result = gaussian_square(LENGTH, SR, ramp_fraction=0.0).real
        np.testing.assert_allclose(result, np.ones(N_POINTS))

    def test_flat_top_reaches_amp(self):
        result = gaussian_square(LENGTH, SR, amp=1.0, ramp_fraction=0.25).real
        assert result[N_POINTS // 2] == pytest.approx(1.0)


class TestIPM:

    def test_length_matches_underlying_envelope(self):
        result = IPM(LENGTH, SR, env='square', freq=50e6)
        assert len(result) == N_POINTS

    def test_normalized_to_unit_peak(self):
        result = IPM(LENGTH, SR, env='square', freq=50e6)
        assert np.max(np.abs(result)) == pytest.approx(1.0)

    def test_zero_beat_amplitude_is_unmodulated(self):
        result = IPM(LENGTH, SR, env='square', freq=50e6, amp2=0.0)
        np.testing.assert_allclose(np.abs(result), np.ones(N_POINTS))


class TestLinear:

    def test_starts_at_zero_ends_at_amp(self):
        result = linear(LENGTH, SR, amp=2.0).real
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(2.0)

    def test_monotonically_increasing(self):
        result = linear(LENGTH, SR).real
        assert np.all(np.diff(result) >= 0)


class TestRAP:

    def test_length_matches_underlying_envelope(self):
        result = RAP(LENGTH, SR, env='square', f0=-10e6, f1=10e6)
        assert len(result) == N_POINTS

    def test_starts_at_zero_phase(self):
        result = RAP(LENGTH, SR, env='square', f0=-10e6, f1=10e6)
        assert result[0] == pytest.approx(1.0 + 0.0j)

    def test_envelope_magnitude_is_preserved(self):
        result = RAP(LENGTH, SR, env='square', amp=0.5, f0=-10e6, f1=10e6)
        np.testing.assert_allclose(np.abs(result), 0.5, atol=1e-6)


class TestSTARAP:

    def test_length_and_dtype(self):
        result = STARAP(LENGTH, SR, env='gaussian', f0=-10e6, f1=10e6)
        assert len(result) == N_POINTS
        assert result.dtype == np.complex64

    def test_normalized_to_unit_peak(self):
        result = STARAP(LENGTH, SR, env='gaussian', f0=-10e6, f1=10e6)
        assert np.max(np.abs(result)) == pytest.approx(1.0, rel=1e-3)


class TestSuperGaussian:

    def test_length_and_dtype(self):
        result = super_gaussian(LENGTH, SR)
        assert len(result) == N_POINTS
        assert result.dtype == np.complex64

    def test_flatter_top_than_plain_gaussian_near_center(self):
        # A super Gaussian should stay closer to its peak amplitude
        # near the center than a plain Gaussian of the same
        # length/n_sigma (its steeper walls only kick in further out).
        g = gaussian(LENGTH, SR, n_sigma=3).real
        sg = super_gaussian(LENGTH, SR, n_sigma=3).real
        near_center = N_POINTS // 2 - N_POINTS // 10
        assert sg[near_center] > g[near_center]

    def test_steeper_walls_than_plain_gaussian_far_from_center(self):
        g = gaussian(LENGTH, SR, n_sigma=3).real
        sg = super_gaussian(LENGTH, SR, n_sigma=3).real
        far_from_center = N_POINTS // 4
        assert sg[far_from_center] < g[far_from_center]


class TestSine:

    def test_matches_analytic_complex_exponential(self):
        freq = 50e6
        phase = 0.3
        result = sine(LENGTH, SR, amp=1.0, freq=freq, phase=phase)
        t = np.linspace(0, LENGTH, N_POINTS)
        expected = np.exp(1j * (2.0 * np.pi * freq * t + phase))
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_zero_frequency_is_constant_phase_factor(self):
        result = sine(LENGTH, SR, amp=1.0, freq=0.0, phase=np.pi / 2)
        np.testing.assert_allclose(result, 1j * np.ones(N_POINTS), atol=1e-6)


class TestSquare:

    def test_constant_amplitude_and_phase(self):
        result = square(LENGTH, SR, amp=2.0, phase=np.pi / 2)
        np.testing.assert_allclose(result, 2j * np.ones(N_POINTS), atol=1e-6)

    def test_length_and_dtype(self):
        result = square(LENGTH, SR)
        assert len(result) == N_POINTS
        assert result.dtype == np.complex64


class TestVirtualZ:

    def test_returns_phase_factor(self):
        result = virtualz(phase=np.pi / 2)
        assert result == pytest.approx(1j)

    def test_length_and_sample_rate_are_unused(self):
        assert virtualz(length=123.0, sample_rate=456.0, phase=0.0) == \
            pytest.approx(1.0)

    def test_amp_is_unused(self):
        assert virtualz(amp=5.0, phase=0.0) == pytest.approx(1.0)


class TestPulseEnvelopesRegistry:

    @pytest.mark.parametrize('name, func', [
        ('cosine_square', cosine_square),
        ('CW', CW),
        ('custom', custom),
        ('DRAG', DRAG),
        ('FAST', FAST),
        ('FAST_DRAG', FAST_DRAG),
        ('gaussian', gaussian),
        ('gaussian_square', gaussian_square),
        ('IPM', IPM),
        ('linear', linear),
        ('RAP', RAP),
        ('STARAP', STARAP),
        ('super_gaussian', super_gaussian),
        ('sine', sine),
        ('square', square),
        ('virtualz', virtualz),
    ])
    def test_registry_maps_name_to_function(self, name, func):
        assert pulse_envelopes[name] is func

    def test_unsupported_envelope_returns_sentinel_string(self):
        assert pulse_envelopes['not_a_real_envelope'] == (
            'Pulse envelope not supported!'
        )


class TestFAST:

    def test_length_and_dtype(self):
        result = FAST(
            LENGTH, SR, freq_intervals=[[0.0, 10e6]], weights=[1.0]
        )
        assert len(result) == N_POINTS
        assert result.dtype == np.complex64

    def test_normalized_to_unit_peak_amplitude(self):
        result = FAST(
            LENGTH, SR, amp=1.0, freq_intervals=[[0.0, 10e6]], weights=[1.0]
        )
        assert np.max(np.abs(result)) == pytest.approx(1.0)

    def test_amp_scales_peak(self):
        result = FAST(
            LENGTH, SR, amp=0.5, freq_intervals=[[0.0, 10e6]], weights=[1.0]
        )
        assert np.max(np.abs(result)) == pytest.approx(0.5)


class TestFastDrag:

    def test_default_frequency_intervals_and_weights_are_used(self):
        result = FAST_DRAG(LENGTH, SR, anh=-200e6)
        assert len(result) == N_POINTS
        assert result.dtype == np.complex64

    def test_normalized_to_unit_peak_amplitude(self):
        result = FAST_DRAG(LENGTH, SR, anh=-200e6)
        assert np.max(np.abs(result)) == pytest.approx(1.0)

    def test_zero_alpha_still_normalizes_the_i_component(self):
        result = FAST_DRAG(LENGTH, SR, anh=-200e6, alpha=0.0)
        np.testing.assert_allclose(result.imag, 0.0, atol=1e-6)
        assert np.max(np.abs(result.real)) == pytest.approx(1.0)
