"""Helper functions for transpilation.

"""
import logging
from collections.abc import Iterable

from qcal.gate.gate import Gate

logger = logging.getLogger(__name__)


def transpilation_error(key: str):
    """Raise an error for non-native gates.

    Args:
        key (str): name of input gate.

    Raises:
        Exception: transpilation error for non-native gate.
    """
    raise Exception(
        f"Cannot transpile '{key}' (non-native gate)!"
    )


class GateMapper(dict):
    """
    A dictionary mapping gate names (strings) to handler functions.
    If a gate is missing, returns a function that raises a transpilation error
    including the gate name.
    """

    def __init__(self, mapping_or_iterable=None, **kwargs):
        """
        Init GateMapper from another dict, iterable of pairs, or keyword args.

        Examples:
            GateMapper({'a': func_a, 'b': func_b})
            GateMapper([('a', func_a), ('b', func_b)])
            GateMapper(a=func_a, b=func_b)
        """
        # If it's another dict-like, convert to dict
        if mapping_or_iterable is None:
            super().__init__(**kwargs)
        elif isinstance(mapping_or_iterable, dict):
            super().__init__(mapping_or_iterable, **kwargs)
        else:
            # Handle an iterable of key/value pairs
            super().__init__(mapping_or_iterable, **kwargs)

    def __missing__(self, key):
        """
        Return a callable that raises transpilation_error with the missing key.
        """
        return lambda *args, **kwargs: transpilation_error(key)

    def call(self, key, *args, **kwargs):
        result = self[key](*args, **kwargs)
        if isinstance(result, Gate):
            return {result}
        elif isinstance(result, Iterable):
            return set(result)
        else:
            raise TypeError(
                f"Gate function for {key} returned an unsupported type."
            )
