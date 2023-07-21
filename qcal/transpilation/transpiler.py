"""Main submodule for handing transpilation.

Transpilation describes the process of mapping one circuit type to another
circuit type at the same level of abstraction.
"""
from collections import defaultdict
from typing import Any, List

class Transpiler:
    """Parent Transpiler class."""

    def __init__(self, gate_mapper: defaultdict) -> None:
        """Initialize a transpiler.

        Args:
            gate_mapper (defaultdict | None, optional): a dictionary which maps
                gates of one kind to gates of another kind.
        """
        self._gate_mapper = gate_mapper

    @property
    def gate_mapper(self) -> defaultdict:
        """Gate mapper.

        This dictionary controls how gates are mapped from one type to another.

        Returns:
            defaultdict: gate mapper.
        """
        return self._gate_mapper
    
    def transpile(self, circuits: Any) -> List[Any]:
        """Transpile all circuits.

        Args:
            circuits (Any): circuits to transpile.

        Returns:
            List[Any]: transpiled circuits.
        """
        raise NotImplementedError(
            'This method should be handled by the child class!'
        )