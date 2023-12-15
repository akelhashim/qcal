"""Submodule for helper plotting functions

"""
import logging

from typing import Tuple

logger = logging.getLogger(__name__)


def calculate_nrows_ncols(n_elements: int, max_ncols: int = 4) -> Tuple:
    """Calculate the number of rows and columns given some number of total
    elements and a maximum number of columns.

    Args:
        n_elements (int): total number of elements to plot.
        max_ncols (int): maximum number of columns.

    Returns:
        Tuple: number of rows, columns
    """
    ncols = min(n_elements, max_ncols)
    nrows = (n_elements + ncols - 1) // ncols
    return nrows, ncols