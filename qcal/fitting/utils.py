"""Helper functions for fitting.

"""
import logging

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def est_freq_fft(
        x: ArrayLike, y: ArrayLike, padding_factor=10
    ) -> float:
    """Estimate frequency using zero-padded FFT.

    Args:
        x (ArrayLike): x data.
        y (ArrayLike): y data.
        padding_factor (int, optional): zero-padding multiplier applied to the
            signal before FFT; larger values increase frequency resolution.
            Defaults to 10.

    Returns:
        float: estimated frequency.
    """
    # Ensure x is evenly spaced for FFT
    if not np.allclose(np.diff(x), np.diff(x)[0]):
        # Interpolate to evenly spaced grid if needed
        x_even = np.linspace(min(x), max(x), len(x))
        y_interp = np.interp(x_even, x, y)
        x, y = x_even, y_interp

    # Remove mean to reduce DC component
    y_detrend = y - np.mean(y)

    # Calculate spacing and sampling frequency
    dx = x[1] - x[0]
    # fs = 1/dx

    # Zero pad the signal
    n_orig = len(y_detrend)
    n_padded = n_orig * padding_factor
    y_padded = np.zeros(n_padded, dtype=y_detrend.dtype)
    y_padded[:n_orig] = y_detrend

    # Compute FFT
    if np.iscomplexobj(y):
        yf = np.fft.fft(y_padded.astype(complex))
        xf = np.fft.fftfreq(n_padded, dx)
        # Exclude DC bin (index 0) when searching for peak
        magnitudes = np.abs(yf)
        magnitudes[0] = 0
        idx = np.argmax(magnitudes)
    else:
        yf = np.fft.rfft(y_padded)
        xf = np.fft.rfftfreq(n_padded, dx)
        idx = np.argmax(np.abs(yf[1:])) + 1

    freq = xf[idx]
    return freq
