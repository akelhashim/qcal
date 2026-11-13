"""Main submodule for handling transpilation.

Transpilation describes the process of mapping one circuit type to another
circuit type at the same level of abstraction. This module defines the
abstract :class:`Transpiler` base class; concrete transpilers (e.g.
:class:`~qcal.interface.bqskit.transpiler.BQSKitTranspiler`,
:class:`~qcal.interface.pygsti.transpiler.PyGSTiTranspiler`) live in
:mod:`qcal.interface` and the hardware backends.

Example:
    .. code-block:: python

        from qcal.circuit import Circuit, CircuitSet, Cycle
        from qcal.gates.single_qubit import X
        from qcal.interface.bqskit.transpiler import BQSKitTranspiler
        from qcal.transpilation.transpiler import Transpiler
        from qcal.transpilation.utils import GateMapper

        # Use a concrete transpiler (qcal -> BQSKit)
        to_bqskit = BQSKitTranspiler(to_bqskit=True)
        bq_circuits = to_bqskit.transpile(Circuit([Cycle({X(0)})]))

        # Or define a custom transpiler
        class NameTranspiler(Transpiler):
            # Map each qcal circuit to a list of gate names.

            def __init__(self) -> None:
                super().__init__(gate_mapper=GateMapper())

            def transpile(self, circuits):
                return CircuitSet(circuits=[
                    [gate.name for cycle in circuit for gate in cycle]
                    for circuit in circuits
                ])

        circuits = CircuitSet(circuits=[Circuit([Cycle({X(0)})])])
        names = NameTranspiler().transpile(circuits)  # names[0] == ['X']
"""
import logging
from abc import ABC, abstractmethod
from typing import Any

from qcal.circuit import CircuitSet
from qcal.transpilation.utils import GateMapper

logger = logging.getLogger(__name__)


class Transpiler(ABC):
    """Abstract base class for all transpilers.

    Subclasses must implement :meth:`transpile`, which maps a collection
    of circuits of one type to a :class:`~qcal.circuit.CircuitSet` of
    circuits of another type, typically using :attr:`gate_mapper`.
    """

    def __init__(self, gate_mapper: GateMapper) -> None:
        """Initialize a transpiler.

        Args:
            gate_mapper (GateMapper): a dictionary which maps gates of one
                kind to gates of another kind.
        """
        self._gate_mapper = gate_mapper

    @property
    def gate_mapper(self) -> GateMapper:
        """Gate mapper.

        This dictionary controls how gates are mapped from one type to
        another.
        """
        return self._gate_mapper

    @abstractmethod
    def transpile(self, circuits: Any) -> CircuitSet:
        """Transpile all circuits.

        Args:
            circuits (Any): circuits to transpile.

        Returns:
            CircuitSet[Any]: transpiled circuits.
        """
