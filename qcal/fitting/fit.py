"""Submodule for different fit classes.

"""
import logging
from typing import Callable, Dict

import lmfit
from lmfit import Model
from numpy.typing import ArrayLike, NDArray

from qcal.fitting.fit_functions import (
    absolute_value,
    base_exponential,
    cosine,
    decaying_cosine,
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
    """Main fit class."""

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
    def fit_fuction(self) -> Callable:
        """Function used to fit the data.

        Returns:
            Callable: fit function.
        """
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
    def error(self) -> NDArray:
        """Reduced chi-squared error.

        Returns:
            NDArray: error array.
        """
        return self._result.redchi

    @property
    def model(self) -> Model:
        """Model for fitting.

        Returns:
            lmfit.model: model
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
        """Fit data to the fit_function provided.

        Args:
            x (ArrayLike): x data.
            y (ArrayLike): y data.
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
        """Initialize an absolute value fitter using the absolute value
        function.
        """
        super().__init__(fit_function)


class FitCosine(Fit):
    """Cosine fit class."""

    def __init__(self, fit_function: Callable = cosine) -> None:
        """Initialize a cosine fitter using the cosine function."""
        super().__init__(fit_function)


class FitDecayingCosine(Fit):
    """Decaying cosine fit class."""

    def __init__(self, fit_function: Callable = decaying_cosine) -> None:
        """Initialize a decaying cosine fitter using the decaying cosine
        function.
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
        """Initialize an exponential fitter using the exponential function."""

        if base is not None:
            def _fit_function(
                x: ArrayLike, a: float, b: float, c: float
            ) -> NDArray:
                return base_exponential(x, base, a, b, c)
        else:
            _fit_function = fit_function

        super().__init__(_fit_function)


class FitLinear(Fit):
    """Linear fit class."""

    def __init__(self, fit_function: Callable = linear) -> None:
        """Initialize an exponential fitter using the exponential function."""
        super().__init__(fit_function)


class FitParabola(Fit):
    """Parabola fit class."""

    def __init__(self, fit_function: Callable = parabola) -> None:
        """Initialize a parabola fitter using the parabola function."""
        super().__init__(fit_function)
