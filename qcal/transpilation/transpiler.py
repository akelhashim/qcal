"""Main submodule for handing transpilation.

Transpilation describes the process of mapping one circuit type to another
circuit type at the same level of abstraction.
"""
import logging
from typing import Any

from qcal.circuit import CircuitSet
from qcal.transpilation.utils import GateMapper

logger = logging.getLogger(__name__)


class Transpiler:
    """Parent Transpiler class."""

    def __init__(self, gate_mapper: GateMapper) -> None:
        """Initialize a transpiler.

        Args:
            gate_mapper (GateMapper | None, optional): a dictionary which maps
                gates of one kind to gates of another kind.
        """
        self._gate_mapper = gate_mapper

    @property
    def gate_mapper(self) -> GateMapper:
        """Gate mapper.

        This dictionary controls how gates are mapped from one type to another.

        Returns:
            GateMapper: gate mapper.
        """
        return self._gate_mapper

    def transpile(self, circuits: Any) -> CircuitSet:
        """Transpile all circuits.

        Args:
            circuits (Any): circuits to transpile.

        Returns:
            CircuitSet[Any]: transpiled circuits.
        """
        raise NotImplementedError(
            'This method should be handled by the child class!'
        )
