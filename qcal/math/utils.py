"""Submodule for helper math functions.

"""
import logging
import numpy as np

from numpy.typing import NDArray
from typing import List, Tuple

logger = logging.getLogger(__name__)


def reciprocal_uncertainty(x: float, sigma_x: float) -> Tuple[float]:
    """Compute the reciprocal of a value and its uncertainty.

    Args:
        x (float): value.
        sigma_x (float): uncertainty.

    Returns:
        Tuple[float]: reciprocal value, reciprocal uncertainty
    """
    rec_x = 1 / x
    rec_sigma_x = np.abs((1 / x**2) * sigma_x)
    return rec_x, rec_sigma_x


def round_sig_figures(x: float, n_sig_figs: int = 1) -> float:
    """Rounds a number to a given number of significant figures.

    Args:
        x (float): number to round.
        n_sig_decimals (int, optional): number of significant figures. 
            Defaults to 1.

    Returns:
        float: rounded number.
    """
    return round(x, n_sig_figs - int(np.floor(np.log10(abs(x)))) - 1)


def round_to_order_error(
        val: float, err: float, error_precision: int = 1
    ) -> Tuple[float]:
    """Rounds a value to its meaningful precision given an uncertainty.

    Args:
        val (float): value; number to round.
        err (float): error used for rounding the value.
        error_precision (int, optional): number of significant digits to round 
            the error to. Defaults to 1.

    Returns:
        Tuple[float]: rounded value, rounded error
    """
    if abs(err) >= abs(val):
        logger.warning(" Uncertainty greater than value!")
        val_precision = 0
    else:
        order_val = np.log10(abs(val))  # exponent of order
        order_err = np.log10(err)
        val_precision = int(np.around(order_val - order_err))

    val_rounded = float(
        np.format_float_scientific(val, precision=val_precision)
    )
    err_rounded = float(
        np.format_float_scientific(err, precision=error_precision)
    )
    return val_rounded, err_rounded


def uncertainty_of_exp(val: float, err: float, p: float | int) -> float:
    """Computes the error propagation for power functions.

    Args:
        val (float): value.
        err (float): error in value.
        p (float | int): power of the exponential.

    Returns:
        float: error of the exponential.
    """
    return abs(p) * err * val**(p - 1)


def uncertainty_of_product(values: NDArray, errors: NDArray) -> float:
    """Computes the uncertainty of a product of values.

    c = a * b
    err(c) = c * sqrt( (err(a)/a)^2 +  (err(b)/b)^2 )

    Args:
        values (NDArray): values.
        errors (NDArray): uncertainty of each value.

    Returns:
        float: uncertainty of product of values.
    """
    return np.prod(values) * np.sqrt(np.sum((errors / values) ** 2))


def uncertainty_of_sum(errors: List | NDArray) -> float:
    """Computes the quadrature sum of errors from a list/array of errors.

    Args:
        errors (List | NDArray): list/array of errors.

    Returns:
        float: error for sum of terms.
    """
    return np.sqrt(np.sum(np.array([err ** 2 for err in errors])))


def wrap_phase(phase: float) -> float:
    """Wrap a phase to be bounded by [-pi, pi].

    Args:
        phase (float): phase to wrap.

    Returns:
        float: phase wrapped between [-pi, pi].
    """
    return (phase + np.pi) % (2 * np.pi) - np.pi