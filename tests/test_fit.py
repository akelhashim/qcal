"""Unit tests for qcal.fitting.fit.

lmfit's generic `Model` has no way to guess sensible starting values
for an arbitrary user function (Model.make_params() defaults every
parameter to -inf here), so every `Fit.fit()` call below passes
explicit initial guesses -- either as individual kwargs matching the
fit function's parameter names, or as an `lmfit.Parameters` object via
`params=` -- exactly as real callers (e.g.
qcal.characterization.coherence) do. For noiseless synthetic data,
starting from the true values converges exactly.
"""
import lmfit
import numpy as np
import pytest

from qcal.fitting.fit import (
    Fit,
    FitAbsoluteValue,
    FitCosine,
    FitDecayingCosine,
    FitDecayingCosineExponential,
    FitExponential,
    FitLinear,
    FitParabola,
)
from qcal.fitting.fit_functions import exponential, linear

X = np.linspace(0, 10, 50)


class TestFitBase:

    def test_fit_fuction_property_returns_constructor_function(self):
        fit = Fit(linear)
        assert fit.fit_fuction is linear

    def test_defaults_before_any_fit(self):
        fit = Fit(linear)
        assert fit.fit_success is False
        assert fit.result is None

    def test_predict_before_fit_raises(self):
        fit = Fit(linear)
        with pytest.raises(AttributeError):
            fit.predict(X)

    def test_fit_with_params_object(self):
        y = 2.5 * X + 1.2
        fit = Fit(linear)
        params = lmfit.Parameters()
        params.add('m', value=1.0)
        params.add('b', value=0.0)
        fit.fit(X, y, params=params)

        assert fit.fit_success
        assert fit.fit_params['m'].value == pytest.approx(2.5)
        assert fit.fit_params['b'].value == pytest.approx(1.2)

    def test_fit_with_kwarg_initial_guesses(self):
        y = 2.5 * X + 1.2
        fit = Fit(linear)
        fit.fit(X, y, m=1.0, b=0.0)

        assert fit.fit_success
        assert fit.fit_params['m'].value == pytest.approx(2.5)

    def test_error_is_near_zero_for_noiseless_data(self):
        y = 2.5 * X + 1.2
        fit = Fit(linear)
        fit.fit(X, y, m=1.0, b=0.0)

        assert fit.error == pytest.approx(0.0, abs=1e-15)

    def test_predict_matches_data_after_successful_fit(self):
        y = 2.5 * X + 1.2
        fit = Fit(linear)
        fit.fit(X, y, m=1.0, b=0.0)

        np.testing.assert_allclose(fit.predict(X), y, atol=1e-8)

    def test_mismatched_lengths_fails_gracefully(self, caplog):
        fit = Fit(linear)
        fit.fit(X, X[:-1], m=1.0, b=0.0)

        assert fit.fit_success is False
        assert fit.result is None
        assert 'Failed to fit data' in caplog.text

    def test_model_wraps_fit_function(self):
        fit = Fit(linear)
        assert fit.model.func is linear


class TestFitLinear:

    def test_recovers_known_params(self):
        y = -3.0 * X + 7.0
        fit = FitLinear()
        fit.fit(X, y, m=1.0, b=0.0)

        assert fit.fit_success
        assert fit.fit_params['m'].value == pytest.approx(-3.0)
        assert fit.fit_params['b'].value == pytest.approx(7.0)


class TestFitParabola:

    def test_recovers_known_params(self):
        y = 3.0 * X**2 - 2.0 * X + 5.0
        fit = FitParabola()
        fit.fit(X, y, a=1.0, b=1.0, c=1.0)

        assert fit.fit_success
        for name, true_value in zip('abc', (3.0, -2.0, 5.0), strict=True):
            assert fit.fit_params[name].value == pytest.approx(true_value)


class TestFitAbsoluteValue:

    def test_recovers_known_params(self):
        y = 1.5 * np.abs(X - 5.0) + 0.5
        fit = FitAbsoluteValue()
        fit.fit(X, y, a=1.0, b=4.0, c=0.0)

        assert fit.fit_success
        for name, true_value in zip('abc', (1.5, 5.0, 0.5), strict=True):
            assert fit.fit_params[name].value == pytest.approx(true_value)


class TestFitExponential:

    def test_default_fit_function_is_exponential(self):
        assert FitExponential().fit_fuction is exponential

    def test_recovers_known_params(self):
        y = 4.0 * np.exp(-0.5 * X) + 1.0
        fit = FitExponential()
        fit.fit(X, y, a=1.0, b=1.0, c=0.0)

        assert fit.fit_success
        for name, true_value in zip('abc', (4.0, 0.5, 1.0), strict=True):
            assert fit.fit_params[name].value == pytest.approx(true_value)

    def test_custom_base_recovers_known_params(self):
        y = 5.0 * 2.0 ** (-0.2 * X) + 0.5
        fit = FitExponential(base=2.0)
        fit.fit(X, y, a=1.0, b=0.1, c=0.0)

        assert fit.fit_success
        for name, true_value in zip('abc', (5.0, -0.2, 0.5), strict=True):
            assert fit.fit_params[name].value == pytest.approx(true_value)


class TestFitCosine:

    def test_recovers_known_params(self):
        y = 2.0 * np.cos(2 * np.pi * 0.3 * X + 0.5) + 0.1
        fit = FitCosine()
        fit.fit(X, y, amp=1.0, freq=0.3, phase=0.0, offset=0.0)

        assert fit.fit_success
        params = fit.fit_params
        assert params['amp'].value == pytest.approx(2.0)
        assert params['freq'].value == pytest.approx(0.3)
        assert params['phase'].value == pytest.approx(0.5)
        assert params['offset'].value == pytest.approx(0.1)


class TestFitDecayingCosine:

    def test_recovers_known_params(self):
        y = (
            2.0 * np.exp(-0.3 * X) * np.cos(2 * np.pi * 0.4 * X + 0.2) + 1.0
        )
        fit = FitDecayingCosine()
        fit.fit(X, y, a=1.0, b=0.1, c=0.4, d=0.0, e=0.0)

        assert fit.fit_success
        for name, true_value in zip(
            'abcde', (2.0, 0.3, 0.4, 0.2, 1.0), strict=True
        ):
            assert fit.fit_params[name].value == pytest.approx(true_value)


class TestFitDecayingCosineExponential:

    def test_recovers_known_params(self):
        y = (
            2.0 * np.exp(-0.3 * X) * np.cos(2 * np.pi * 0.4 * X)
            + 1.5 * np.exp(-0.1 * X) + 0.5
        )
        fit = FitDecayingCosineExponential()
        fit.fit(
            X, y, a=1.0, b=0.1, c=0.4, d=0.0, e=1.0, f=0.05, g=0.0
        )

        assert fit.fit_success
        for name, true_value in zip(
            'abcdefg', (2.0, 0.3, 0.4, 0.0, 1.5, 0.1, 0.5), strict=True
        ):
            assert fit.fit_params[name].value == pytest.approx(
                true_value, abs=1e-4
            )
