"""Helper functions for transpilation."""
import logging
from collections.abc import Iterable, Iterator

from qcal.circuit import Circuit, Cycle
from qcal.gates.gate import Gate

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
    """A dictionary mapping gate names (strings) to handler functions.

    If a gate is missing, returns a function that raises a transpilation
    error including the gate name.

    Example:
        .. code-block:: python

            from qcal.gates.single_qubit import X
            from qcal.transpilation.utils import GateMapper

            gate_mapper = GateMapper({'X': X})
            gates = gate_mapper.call('X', 0)  # {X(0)}
            gate_mapper['Y']  # callable that raises a transpilation error
    """

    def __init__(self, mapping_or_iterable=None, **kwargs):
        """Initialize from a dict, iterable of pairs, or keyword args.

        Args:
            mapping_or_iterable (dict | Iterable | None, optional): mapping
                of gate names to handler functions, or an iterable of
                ``(name, handler)`` pairs. Defaults to ``None``.
            **kwargs: additional ``name=handler`` entries.

        Examples:
            .. code-block:: python

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
        """Return a callable that raises a transpilation error for ``key``."""
        return lambda *args, **kwargs: transpilation_error(key)

    def call(self, key, *args, **kwargs):
        """Call the handler for a gate and normalize its output.

        Args:
            key (str): name of the gate.
            *args: positional arguments passed to the handler.
            **kwargs: keyword arguments passed to the handler.

        Returns:
            set | Circuit: a set of gates if the handler returns a single
                ``Gate`` or an iterator of gates, or a
                :class:`~qcal.circuit.Circuit` with one gate per cycle if it
                returns any other (non-string) iterable of gates.

        Raises:
            TypeError: if the handler returns an unsupported type.
        """
        result = self[key](*args, **kwargs)
        if isinstance(result, Gate):
            return {result}
        elif isinstance(result, Iterator):
            return set(result)
        elif isinstance(result, Iterable) and not isinstance(result, str):
            return Circuit([
                Cycle({gate}) for gate in result
            ])
        else:
            raise TypeError(
                f"Gate function for {key} returned an unsupported type."
            )
