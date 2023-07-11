"""Submodule for different fit classes.

"""
from qcal.fitting.fit_functions import *

import logging

from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit
from typing import Callable

logger = logging.getLogger(__name__)


__all__ = (
    'CosineFit'
)


class FitError(Exception):
    """Custom FitError class."""
    pass


class Fit:
    """Main fit class."""

    __slots__ = ('_fit_function', '_fit_success', '_popt', '_pcov')

    def __init__(self, fit_function: Callable) -> None:
        """Initialize a fitter for a given fit function.

        Args:
            fit_function (Callable): function to fit the data to.
        """
        self._fit_function = fit_function
        self._fit_success = False
        self._popt = None
        self._pcov = None

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
    def fit_params(self) -> NDArray:
        """Optimized fit params.

        Returns:
            NDArray: param array.
        """
        return self._popt
    
    @property
    def covariance(self) -> NDArray:
        """The estimated approximate covariance of the optimized fit params.

        Returns:
            NDArray: covariance matrix.
        """
        return self._pcov
    
    @property
    def error(self) -> NDArray:
        """One standard deviation on the error of the optimized fit params.

        Returns:
            NDArray: error array.
        """
        return np.sqrt(np.diag(self._pcov))
    
    def fit(self, x: ArrayLike, y: ArrayLike, **kwargs) -> None:
        """Fit data to the fit_function provided.

        Args:
            x (ArrayLike): x data.
            y (ArrayLike): y data.
        """
        try:
            self._popt, self._pcov = curve_fit(
                self._fit_function, x, y, **kwargs
            )
            self._fit_success = True
        except FitError:
            logger.warning(f' Failed to fit data to {self._fit_function}!')

    def predict(self, x: ArrayLike) -> ArrayLike:
        """Predict data given the fit function and the optimized fit params.

        Args:
            x (ArrayLike): x data.

        Returns:
            ArrayLike: y data.
        """
        return self._fit_function(x, *self._popt)
    

class CosineFit(Fit):
    """Cosine fit class."""

    def __init__(self, fit_function: Callable = cosine) -> None:
        """Initialize a cosine fitter using the cosine function."""
        super().__init__(fit_function)


class ParabolaFit(Fit):
    """Parabola fit class."""

    def __init__(self, fit_function: Callable = parabola) -> None:
        """Initialize a parabola fitter using the parabola function."""
        super().__init__(fit_function)
