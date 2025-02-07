"""Submodule for helper spectroscopy functions.

"""
import logging
import numpy as np

from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


def find_inflection_points(x: NDArray, y: NDArray, sigma=10) -> NDArray:
    """Find inflection points for resonator fitting.

    Args:
        x (NDArray): independent variable.
        y (NDArray): dependent variable.
        sigma (int, optional): standard deviation for Gaussian kernel. Defaults 
            to 10.

    Returns:
        NDArray: indices of inflection points.
    """
    y_smoothed = gaussian_filter1d(y, sigma=sigma)
    dy_dx = np.gradient(y_smoothed, x)
    d2y_dx2 = np.gradient(dy_dx, x)
    inflection_idxs = np.where(np.diff(np.sign(d2y_dx2)))[0]
    
    return inflection_idxs