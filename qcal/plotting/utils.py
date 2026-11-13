"""Submodule for helper plotting functions."""
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def calculate_nrows_ncols(n_elements: int, max_ncols: int = 4) -> Tuple:
    """Calculate the number of rows and columns for a grid of subplots.

    The grid holds ``n_elements`` total elements with at most
    ``max_ncols`` columns.

    Args:
        n_elements (int): total number of elements to plot.
        max_ncols (int, optional): maximum number of columns. Defaults
            to 4.

    Returns:
        Tuple: number of rows, number of columns.
    """
    if n_elements <= 0:
        return 0, 0
    ncols = min(n_elements, max_ncols)
    nrows = (n_elements + ncols - 1) // ncols
    return nrows, ncols
