"""Helper functions.
"""
import functools
import inspect
import logging
import pandas as pd
import pickle

from collections.abc import Iterable
from typing import Any, Callable, List

logger = logging.getLogger(__name__)


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


def flatten(xs: List):
    """Flatten a list with arbitrary nesting.

    Args:
        xs (List): List of nested lists.

    Yields:
        Any: the flattened item in the list.
    """
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def save_init(init: Callable) -> Callable:
    """Decorator that saves the arguments passed to a class's __init__ method.

    Stores the initialization arguments in `_init_args` and `_init_kwargs` 
    on the instance, allowing for later re-instantiation or debugging.

    Args:
        init (Callable): original __init__ method of the class.

    Returns:
        Callable: wrapped __init__ method with argument-saving functionality.
    """
    @functools.wraps(init)
    def wrapped(self, *args, **kwargs) -> Any:
        """Wrapped __init__ that saves init arguments to instance variables.

        Args:
            *args: positional arguments passed to __init__.
            **kwargs: keyword arguments passed to __init__.

        Returns:
            Any: whatever the original __init__ returns (typically None).
        """
        sig = inspect.signature(init)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self", None)

        self._init_args = args
        self._init_kwargs = kwargs

        return init(self, *args, **kwargs)

    return wrapped


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