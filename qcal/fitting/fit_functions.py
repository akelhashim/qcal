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
        amp (float):    amplitude of cosine curve.
        freq (float):   frequency of cosine curve.
        phase (float):  phase of cosine curve.
        offset (float): phase offset of cosine curve.

    Returns:
        NDArray: cosine fit to data.
    """
    return amp * np.cos(2 * np.pi * freq * x + phase) + offset


def parabola(x: ArrayLike, a: float, b: float, c: float) -> ArrayLike:
    """Parabola function.

    Args:
        x (ArrayLike): data.
        a (float):     quadratic coefficient.
        b (float):     linear coefficient; sets the x offset.
        c (float):     y offset.

    Returns:
        ArrayLike: parabola fit to data.
    """
    return a * x**2 + b * x + c


