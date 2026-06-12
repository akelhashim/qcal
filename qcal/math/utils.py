"""Submodule for helper math functions.

"""
import logging
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

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


def uncertainty_of_product(
        values: NDArray,
        stds:   NDArray | None = None,
        cov:    NDArray | None = None,
    ) -> float:
    """Computes the uncertainty of a product of values.

    Without covariance:
        c = a * b
        err(c) = |c| * sqrt( (err(a)/a)^2 + (err(b)/b)^2 )

    With covariance matrix Cov:
        (err(f)/f)^2 = sum_{i,j} Cov(x_i, x_j) / (x_i * x_j) = v^T C v,
        with v_i = 1 / x_i.
        => err(c) = |c| * sqrt( v^T @ Cov @ v )

    Note: values must be non-zero (relative-error formulation).

    Args:
        values (NDArray): values.
        stds (NDArray | None): standard deviations of each value. Required
            when cov is ``None``.
        cov (NDArray | None): covariance matrix of the values. Defaults to
            ``None``. When provided, ``stds`` is ignored and the full
            off-diagonal correlations are used.

    Returns:
        float: uncertainty of the product.
    """
    if cov is None and stds is None:
        raise ValueError("Must provide either `stds` or `cov`.")

    c = np.prod(values)
    v = 1.0 / values

    if cov is not None:
        rel_var = v @ cov @ v
    else:
        rel_var = np.sum((stds / values) ** 2)

    return float(np.abs(c) * np.sqrt(max(rel_var, 0.0)))


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
