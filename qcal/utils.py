"""Helper functions."""
import functools
import inspect
import logging
import pathlib
import pickle
import warnings
from collections.abc import Iterable
from typing import Any, Callable, List

import pandas as pd

logger = logging.getLogger(__name__)


def flatten(xs: List):
    """Flatten a list with arbitrary nesting.

    Args:
        xs (List): list of nested lists.

    Yields:
        Any: the flattened item in the list.
    """
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def get_package_directory() -> pathlib.Path:
    """Return the root directory of the qcal package.

    Returns:
        pathlib.Path: repo root when installed in editable mode, or the
            site-packages directory when installed normally.
    """
    return pathlib.Path(__file__).parent.parent


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


def pop_deprecated_kwarg(
    kwargs: dict, old: str, new: str, value: Any, default: Any = None
) -> Any:
    """Resolve a renamed keyword argument, warning if the old name is used.

    Removes ``old`` from ``kwargs`` (so it is not forwarded elsewhere) and
    returns the value to use for ``new``.

    Example:
        .. code-block:: python

            def f(propagate_correction=False, **kwargs):
                propagate_correction = pop_deprecated_kwarg(
                    kwargs, 'propogate_correction', 'propagate_correction',
                    propagate_correction, default=False,
                )

    Args:
        kwargs (dict): keyword arguments that may contain ``old``. Modified
            in place.
        old (str): deprecated argument name.
        new (str): current argument name.
        value (Any): value passed for ``new``.
        default (Any, optional): default value of ``new``, used to detect
            whether ``new`` was passed explicitly. Defaults to None.

    Returns:
        Any: the value passed for ``old`` if it was given, otherwise
        ``value``.

    Raises:
        TypeError: if both ``old`` and ``new`` were passed with different
            values.
    """
    if old not in kwargs:
        return value
    old_value = kwargs.pop(old)
    warnings.warn(
        f'{old!r} is deprecated; use {new!r} instead.',
        DeprecationWarning,
        stacklevel=3,
    )
    if value != default and value != old_value:
        raise TypeError(
            f'Got values for both {new!r} and its deprecated alias '
            f'{old!r}.'
        )
    return old_value


def save_init(init: Callable) -> Callable:
    """Decorator that saves the arguments passed to a class's ``__init__``.

    Stores the initialization arguments in ``_init_args`` and
    ``_init_kwargs`` on the instance, allowing for later re-instantiation
    or debugging.

    Args:
        init (Callable): original ``__init__`` method of the class.

    Returns:
        Callable: wrapped ``__init__`` method with argument-saving
            functionality.
    """
    @functools.wraps(init)
    def wrapped(self, *args, **kwargs) -> Any:
        """Wrapped ``__init__`` that saves init arguments to the instance.

        Args:
            self (Any): the instance being initialized.
            *args: positional arguments passed to ``__init__``.
            **kwargs: keyword arguments passed to ``__init__``.

        Returns:
            Any: whatever the original ``__init__`` returns (typically
                None).
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
    if filename.endswith('.pkl'):
        pkl_filename = filename
    else:
        pkl_filename = f'{filename}.pkl'
    if isinstance(data, pd.DataFrame):
        data.to_pickle(pkl_filename)
    else:
        with open(pkl_filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
