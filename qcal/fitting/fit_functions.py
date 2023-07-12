"""Submodule for storing different fit functions.

"""
import logging
import numpy as np

from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


def cosine(
        x: ArrayLike, amp: float, freq: float, phase: float, offset: float
    ) -> NDArray:
    """Cosine function.

    Args:
        x (ArrayLike):  data.
        amp (float):    amplitude.
        freq (float):   frequency.
        phase (float):  phase.
        offset (float): phase offset.

    Returns:
        NDArray: cosine curve.
    """
    return amp * np.cos(2 * np.pi * freq * x + phase) + offset


def exponential(x: ArrayLike, a: float, b: float, c: float) -> NDArray:
    """Exponential function.

    Args:
        x (ArrayLike): data.
        a (float): amplitude.
        b (float): x-offset.
        c (float): y-offset

    Returns:
        NDArray: exponential curve.
    """
    return a * np.exp(-b * x) + c


def parabola(x: ArrayLike, a: float, b: float, c: float) -> ArrayLike:
    """Parabola function.

    Args:
        x (ArrayLike): data.
        a (float):     quadratic coefficient.
        b (float):     linear coefficient; sets the x-offset.
        c (float):     y offset.

    Returns:
        ArrayLike: parabola curve.
    """
    return a * x**2 + b * x + c


