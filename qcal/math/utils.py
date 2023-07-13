"""Submodule for helper math functions.

"""
import logging
import numpy as np

from typing import Tuple

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
        logger.wanring("Uncertainty greater than value!")
        val_precision = 0
    else:
        order_val = np.log10(abs(val))  # exponent of order
        order_err = np.log10(err)
        val_precision = int(np.around(order_val - order_err))

    val_rounded = float(np.format_float_scientific(val, precision=val_precision))
    err_rounded = float(np.format_float_scientific(err, precision=error_precision))
    return val_rounded, err_rounded