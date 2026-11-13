"""Fit classes for fitting experimental data to model functions.

This module wraps :class:`lmfit.Model` in a small set of convenience
classes. :class:`Fit` is the generic base class, which accepts any model
function, and subclasses such as :class:`FitExponential`,
:class:`FitCosine`, :class:`FitDecayingCosine`, :class:`FitLinear`, and
:class:`FitParabola` are pre-configured with the corresponding functions
from :mod:`qcal.fitting.fit_functions`. Initial guesses for the model
parameters are passed as keyword arguments to :meth:`Fit.fit`.

Example:
    Fit a synthetic :math:`T_1` decay curve:

    .. code-block:: python

        import numpy as np
        from qcal.fitting.fit import FitExponential

        x = np.linspace(0, 100e-6, 51)
        y = 0.9 * np.exp(-x / 30e-6) + 0.05

        fit = FitExponential()
        fit.fit(x, y, a=1.0, b=1 / 20e-6, c=0.0)
        fit.fit_success                # True
        T1 = 1 / fit.fit_params['b'].value   # ~3.0e-05
        y_fit = fit.predict(x)
"""
import logging
import warnings
from typing import Callable, Dict

import lmfit
from lmfit import Model
from numpy.typing import ArrayLike

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

logger = logging.getLogger(__name__)


__all__ = (
    'FitAbsoluteValue',
    'FitCosine',
    'FitDecayingCosine',
    'FitExponential',
    'FitLinear',
    'FitParabola',
)


class FitError(Exception):
    """Custom FitError class."""
    pass


class Fit:
    """Generic fit class wrapping an :class:`lmfit.Model`.

    The data is fit by calling :meth:`fit`, after which the optimized
    parameters are available via :attr:`fit_params` and the fit can be
    evaluated with :meth:`predict`. If the fit fails, a warning is logged
    and :attr:`fit_success` remains ``False``.
    """

    __slots__ = ('_fit_function', '_fit_success', '_model', '_result')

    def __init__(self, fit_function: Callable) -> None:
        """Initialize a fitter for a given fit function.

        Args:
            fit_function (Callable): function to fit the data to.
        """
        self._fit_function = fit_function
        self._fit_success = False
        self._model = Model(fit_function)
        self._result = None

    @property
    def fit_function(self) -> Callable:
        """Function used to fit the data.

        Returns:
            Callable: fit function.
        """
        return self._fit_function

    @property
    def fit_fuction(self) -> Callable:
        """Deprecated misspelled alias of :attr:`fit_function`.

        Returns:
            Callable: fit function.
        """
        warnings.warn(
            "'fit_fuction' is deprecated; use 'fit_function' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._fit_function

    @property
    def fit_success(self) -> bool:
        """Whether or not the data was successfully fit to the function.

        Returns:
            bool: fit success or not.
        """
        return self._fit_success

    @property
    def fit_params(self) -> Dict:
        """Optimized fit params.

        Returns:
            Dict: a dictionary of Parameter objects.
        """
        return self._result.params

    @property
    def error(self) -> float:
        """Reduced chi-squared error.

        Returns:
            float: reduced chi-squared value.
        """
        return self._result.redchi

    @property
    def model(self) -> Model:
        """Model for fitting.

        Returns:
            lmfit.Model: model.
        """
        return self._model

    @property
    def result(self) -> lmfit.model.ModelResult:
        """Model fit result.

        Returns:
            lmfit.model.ModelResult: result object.
        """
        return self._result

    def fit(self, x: ArrayLike, y: ArrayLike, **kwargs) -> None:
        """Fit data to the fit function provided.

        Args:
            x (ArrayLike): x data.
            y (ArrayLike): y data.
            **kwargs: additional keyword arguments passed to
                :meth:`lmfit.Model.fit` (e.g. initial guesses for the
                model parameters, such as ``a=1.0``).
        """
        try:
            self._result = self._model.fit(y, x=x, **kwargs)
            assert self._result.success
            self._fit_success = self._result.success
            # logger.info(f' {self._result.fit_report()}')
        except Exception:
            logger.warning(f' Failed to fit data to {self._fit_function}!')

    def predict(self, x: ArrayLike) -> ArrayLike:
        """Predict data given the fit function and the optimized fit params.

        Args:
            x (ArrayLike): x data.

        Returns:
            ArrayLike: y data.
        """
        return self._result.eval(x=x)


class FitAbsoluteValue(Fit):
    """Absolute value fit class."""

    def __init__(self, fit_function: Callable = absolute_value) -> None:
        """Initialize an absolute value fitter.

        Args:
            fit_function (Callable, optional): function to fit the data to.
                Defaults to
                :func:`~qcal.fitting.fit_functions.absolute_value`.
        """
        super().__init__(fit_function)


class FitCosine(Fit):
    """Cosine fit class."""

    def __init__(self, fit_function: Callable = cosine) -> None:
        """Initialize a cosine fitter.

        Args:
            fit_function (Callable, optional): function to fit the data to.
                Defaults to :func:`~qcal.fitting.fit_functions.cosine`.
        """
        super().__init__(fit_function)


class FitDecayingCosine(Fit):
    """Decaying cosine fit class."""

    def __init__(self, fit_function: Callable = decaying_cosine) -> None:
        """Initialize a decaying cosine fitter.

        Args:
            fit_function (Callable, optional): function to fit the data to.
                Defaults to
                :func:`~qcal.fitting.fit_functions.decaying_cosine`.
        """
        super().__init__(fit_function)


class FitDecayingCosineExponential(Fit):
    """Decaying cosine with exponential baseline fit class."""

    def __init__(
        self, fit_function: Callable = decaying_cosine_exponential
    ) -> None:
        """Initialize a decaying cosine with exponential baseline fitter.

        Args:
            fit_function (Callable, optional): function to fit the data to.
                Defaults to :func:`.decaying_cosine_exponential`.
        """
        super().__init__(fit_function)


class FitExponential(Fit):
    """Exponential fit class."""

    __slots__ = ('_base',)

    def __init__(
        self,
        fit_function: Callable = exponential,
        base: float | None = None,
    ) -> None:
        """Initialize an exponential fitter.

        Args:
            fit_function (Callable, optional): function to fit the data to.
                Ignored if ``base`` is not ``None``. Defaults to
                :func:`~qcal.fitting.fit_functions.exponential`.
            base (float | None, optional): base of the exponential. If
                provided, the data is fit to
                :func:`~qcal.fitting.fit_functions.base_exponential` with
                this fixed base. Defaults to None.
        """
        if base is not None:
            def _fit_function(
                x: ArrayLike, a: float, b: float, c: float
            ) -> ArrayLike:
                return base_exponential(x, base, a, b, c)
        else:
            _fit_function = fit_function

        super().__init__(_fit_function)


class FitLinear(Fit):
    """Linear fit class."""

    def __init__(self, fit_function: Callable = linear) -> None:
        """Initialize a linear fitter.

        Args:
            fit_function (Callable, optional): function to fit the data to.
                Defaults to :func:`~qcal.fitting.fit_functions.linear`.
        """
        super().__init__(fit_function)


class FitParabola(Fit):
    """Parabola fit class."""

    def __init__(self, fit_function: Callable = parabola) -> None:
        """Initialize a parabola fitter.

        Args:
            fit_function (Callable, optional): function to fit the data to.
                Defaults to :func:`~qcal.fitting.fit_functions.parabola`.
        """
        super().__init__(fit_function)
