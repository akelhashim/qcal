"""Helper functions for ML.

"""
from numpy.typing import ArrayLike
from typing import Dict


def find_mapping(array1: ArrayLike, array2: ArrayLike) -> Dict:
    """Find the mapping between two arrays.

    Args:
        array1 (ArrayLike): first array.
        array2 (ArrayLike): second array.

    Raises:
        ValueError: if the arrays are not the same length.

    Returns:
        Dict: mapping between the first and second arrays.
    """
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")

    mapping = {}  # Initialize an empty dictionary to store the mapping

    for num1, num2 in zip(array1, array2):
        mapping[num1] = num2

    return mapping