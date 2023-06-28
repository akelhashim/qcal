"""Helper functions.
"""
import logging
import pickle

from typing import Any

logger = logging.getLogger(__name__)


def save(data: Any, filename: str) -> None:
    """Save data to a pickle file.

    Args:
        data (Any): data of any datatype.
        filename (str): filename for saved data.
    """
    with open(f'{filename}.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename: str) -> Any:
    """Load data from a pickle file.

    Args:
        filename (str): filename of the saved data.

    Returns:
        Any: loaded data.
    """
    with open(f'{filename}.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data

