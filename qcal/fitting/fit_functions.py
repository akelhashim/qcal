"""Submodule for storing different fit functions.

"""
import logging
import numpy as np

from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


__all__ = (
    'absolute_value',
    'cosine',
    'decaying_cosine',
    'exponential',
    'constrained_exponential',
    'linear',
    'parabola'
)


def absolute_value(x: ArrayLike, a: float, b: float, c: float) -> NDArray:
    """Absolute value function.

    Args:
        x (ArrayLike): data.
        a (float): amplitude.
        b (float): x-offset.
        c (float): y-offset

    Returns:
        NDArray: absolute value curve.
    """
    return a * np.abs(x - b)  + c


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

def decaying_cosine(
        x: ArrayLike, a: float, b: float, c: float, d: float, e: float
    ) -> NDArray:
    """Decaying cosine function.

    Args:
        x (ArrayLike): data.
        a (float): amplitude.
        b (float): exponential constant.
        c (float): cosine frequency.
        d (float): cosine phase.
        e (float): y-offset.

    Returns:
        NDArray: decaying cosine curve.
    """
    return a * np.exp(-b * x) * np.cos(2 * np.pi * c * x + d) + e


def exponential(x: ArrayLike, a: float, b: float, c: float) -> NDArray:
    """Exponential function.

    Args:
        x (ArrayLike): data.
        a (float): amplitude.
        b (float): exponential constant.
        c (float): y-offset

    Returns:
        NDArray: exponential curve.
    """
    return a * np.exp(-b * x) + c

def constrained_exponential(x: ArrayLike, a: float, b: float) -> NDArray:
    """Exponential function.

    Args:
        x (ArrayLike): data.
        a (float): y-amplitude/offset
        b (float): exponential constant.

    Returns:
        NDArray: exponential curve.
    """
    return a*(1 - np.exp(-b * x))



def linear(x: ArrayLike, m: float, b: float) -> ArrayLike:
    """Linear function

    Args:
        x (ArrayLike): data.
        m (float): slope of the line.
        b (float): y-intercept.

    Returns:
        ArrayLike: linear curve.
    """
    return m * x + b


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


