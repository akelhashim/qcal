"""Model functions used for fitting experimental data.

Each function takes the independent variable ``x`` as its first argument
followed by the model parameters, so it can be passed directly to
:class:`lmfit.Model` (see :mod:`qcal.fitting.fit`).
"""
import logging

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


__all__ = (
    'absolute_value',
    'base_exponential',
    'cosine',
    'decaying_cosine',
    'decaying_cosine_exponential',
    'exponential',
    'linear',
    'parabola'
)


def absolute_value(x: ArrayLike, a: float, b: float, c: float) -> ArrayLike:
    r"""Absolute value function.

    .. math::

        f(x) = a |x - b| + c

    Args:
        x (ArrayLike): data.
        a (float): amplitude.
        b (float): x-offset.
        c (float): y-offset.

    Returns:
        ArrayLike: absolute value curve.
    """
    return a * np.abs(x - b)  + c


def base_exponential(
    x:    ArrayLike,
    base: float,
    a:    float,
    b:    float,
    c:    float,
) -> ArrayLike:
    r"""Exponential function with an arbitrary base.

    .. math::

        f(x) = a \, \mathrm{base}^{b x} + c

    Args:
        x (ArrayLike): data.
        base (float): base of the exponential.
        a (float): amplitude.
        b (float): exponential constant.
        c (float): y-offset.

    Returns:
        ArrayLike: exponential curve.
    """
    return a * np.power(base, b * x) + c


def cosine(
    x: ArrayLike, amp: float, freq: float, phase: float, offset: float
) -> ArrayLike:
    r"""Cosine function.

    .. math::

        f(x) = \mathrm{amp} \cos(2 \pi \, \mathrm{freq} \, x
        + \mathrm{phase}) + \mathrm{offset}

    Args:
        x (ArrayLike): data.
        amp (float): amplitude.
        freq (float): frequency.
        phase (float): phase.
        offset (float): y-offset.

    Returns:
        ArrayLike: cosine curve.
    """
    return amp * np.cos(2 * np.pi * freq * x + phase) + offset


def decaying_cosine(
    x: ArrayLike, a: float, b: float, c: float, d: float, e: float
) -> ArrayLike:
    r"""Decaying cosine function.

    .. math::

        f(x) = a e^{-b x} \cos(2 \pi c x + d) + e

    Args:
        x (ArrayLike): data.
        a (float): amplitude.
        b (float): exponential constant.
        c (float): cosine frequency.
        d (float): cosine phase.
        e (float): y-offset.

    Returns:
        ArrayLike: decaying cosine curve.
    """
    return a * np.exp(-b * x) * np.cos(2 * np.pi * c * x + d) + e


def decaying_cosine_exponential(
    x: ArrayLike, a: float, b: float, c: float, d: float, e: float,
    f: float, g: float
) -> ArrayLike:
    r"""Decaying cosine function with an exponential baseline.

    .. math::

        f(x) = a e^{-b x} \cos(2 \pi c x + d) + e \, e^{-f x} + g

    Args:
        x (ArrayLike): data.
        a (float): amplitude.
        b (float): exponential constant.
        c (float): cosine frequency.
        d (float): cosine phase.
        e (float): baseline y-offset.
        f (float): baseline exponential constant.
        g (float): constant y-offset.

    Returns:
        ArrayLike: decaying cosine curve.
    """
    return (
        a * np.exp(-b * x) * np.cos(2 * np.pi * c * x + d)
        + e * np.exp(-f * x) + g
    )


def exponential(x: ArrayLike, a: float, b: float, c: float) -> ArrayLike:
    r"""Exponential decay function.

    .. math::

        f(x) = a e^{-b x} + c

    Args:
        x (ArrayLike): data.
        a (float): amplitude.
        b (float): exponential constant.
        c (float): y-offset.

    Returns:
        ArrayLike: exponential curve.
    """
    return a * np.exp(-b * x) + c


def linear(x: ArrayLike, m: float, b: float) -> ArrayLike:
    r"""Linear function.

    .. math::

        f(x) = m x + b

    Args:
        x (ArrayLike): data.
        m (float): slope of the line.
        b (float): y-intercept.

    Returns:
        ArrayLike: linear curve.
    """
    return m * x + b


def parabola(x: ArrayLike, a: float, b: float, c: float) -> ArrayLike:
    r"""Parabola function.

    .. math::

        f(x) = a x^2 + b x + c

    Args:
        x (ArrayLike): data.
        a (float): quadratic coefficient.
        b (float): linear coefficient; sets the x-offset.
        c (float): y-offset.

    Returns:
        ArrayLike: parabola curve.
    """
    return a * x**2 + b * x + c
