"""Helper functions.
"""
import logging
import pandas as pd
import pickle

from typing import Any

logger = logging.getLogger(__name__)


def save_to_csv(data: pd.DataFrame, filename: str) -> None:
    """Save a dataframe to a csv file.

    Args:
        data (pd.DataFrame): data in a pandas DataFrame.
        filename (str): filename for the saved data.
    """
    assert isinstance(data, pd.DataFrame), 'Data must be in a DataFrame!'
    data.to_csv(f'{filename}.csv')


def save_to_pickle(data: Any, filename: str) -> None:
    """Save data to a pickle file.

    Args:
        data (Any): data of any datatype.
        filename (str): filename for saved data.
    """
    if isinstance(data, pd.DataFrame):
        data.to_pickle(f'{filename}.pkl')
    with open(f'{filename}.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename: str) -> Any:
    """Load data from a pickle file.

    Args:
        filename (str): filename of the saved data.

    Returns:
        Any: loaded data.
    """
    with open(f'{filename}', 'rb') as handle:
        data = pickle.load(handle)
    return data

