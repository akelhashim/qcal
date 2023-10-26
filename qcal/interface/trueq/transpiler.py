"""Submodule for handling transpilation from True-Q to qcal circuits.

"""
from qcal.circuit import Barrier, Circuit, CircuitSet
from qcal.gate.single_qubit import single_qubit_gates
from qcal.gate.two_qubit import two_qubit_gates
from qcal.transpilation.transpiler import Transpiler

import logging
import multiprocessing as mp
import numpy as np

from collections import defaultdict, deque

logger = logging.getLogger(__name__)


def to_qcal(
        circuit, 
        gate_mapper: defaultdict,
        barrier_between_all: bool = False
    ) -> Circuit:
    """Transpile a True-Q circuit to a qcal circuit.

    Args:
        circuit (tq.Circuit):      True-Q circuit.
        gate_mapper (defaultdict): map between True-Q to qcal gates.
        barrier_between_all (bool, optional): whether to place a barrier
            between all cycles. Defaults to False. This is useful for
            benchmarking circuits to ensure that the circuit structure is
            preserved.

    Returns:
        Circuit: qcal circuit.
    """
    import trueq as tq

    tcircuit = deque()
    for i, cycle in enumerate(circuit):
        tcycle = deque()

        if cycle.marker > 0 and not barrier_between_all:
            tcircuit.append(Barrier(cycle.labels))
        
        for q, gate in cycle:
            if isinstance(gate, tq.Meas):
                tcycle.append(gate_mapper['Meas'](q))
                # pass
            elif gate.name == 'Rz':
                tcycle.append(
                    gate_mapper[gate.name](
                        np.deg2rad(gate.parameters['phi']), q
                    )
                )
            else:
                tcycle.append(gate_mapper[gate.name](q))

        tcircuit.append(tcycle)

        if cycle.marker > 0 and not barrier_between_all:
            tcircuit.append(Barrier(circuit.labels))
        elif barrier_between_all and i < (len(circuit) - 1):
            tcircuit.append(Barrier(circuit.labels))
    
    tcircuit = Circuit(tcircuit)
    return tcircuit


class Transpiler(Transpiler):
    """True-Q to qcal Transpiler."""
    import trueq as tq

    __slots__ = ('_gate_mapper', '_barrier_between_all')

    def __init__(
            self, 
            gate_mapper:         dict | None = None, 
            barrier_between_all: bool = False
        ) -> None:
        """Initialize with a gate_mapper.

        Args:
            gate_mapper (dict | None, optional): dictionary which maps
                TrueQ gates to Qubic gates. Defaults to None.
            barrier_between_all (bool, optional): whether to place a barrier
                between all cycles. Defaults to False. This is useful for
                benchmarking circuits to ensure that the circuit structure is
                preserved.
        """
        if gate_mapper is None:
            gate_mapper = {**single_qubit_gates, **two_qubit_gates}
        super().__init__(gate_mapper=gate_mapper)
        self._barrier_between_all = barrier_between_all

    def transpile(
            self, circuits: tq.Circuit | tq.CircuitCollection
        ) -> CircuitSet:
        """Transpile all circuits.

        Args:
            circuits (tq.Circuit | tq.CircuitCollection): circuits to 
                transpile.

        Returns:
            CircuitSet: transpiled circuits.
        """
        import trueq as tq
        if not isinstance(circuits, tq.CircuitCollection):
            circuits = tq.CircuitCollection(circuits)

        tcircuits = []
        for circuit in circuits:
            tcircuits.append(
                to_qcal(
                    circuit, 
                    self._gate_mapper,
                    self._barrier_between_all
                )
            )
        
        tcircuits = CircuitSet(circuits=tcircuits)
        tcircuits['key'] = [str(circ.key) for circ in circuits]
        return tcircuits