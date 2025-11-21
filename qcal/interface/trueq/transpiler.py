"""Submodule for handling transpilation from True-Q to qcal circuits.

"""
import logging
from collections import defaultdict, deque
from typing import Dict

import numpy as np

from qcal.circuit import Barrier, Circuit, CircuitSet, Cycle
from qcal.gate.single_qubit import single_qubit_gates
from qcal.gate.two_qubit import two_qubit_gates
from qcal.transpilation.transpiler import Transpiler
from qcal.transpilation.utils import GateMapper

logger = logging.getLogger(__name__)


def transpile_cycle(cycle, gate_mapper: defaultdict) -> deque:
    """Transpile a True-Q cycle to qcal Cycle.

    Args:
        cycle (trueq.Cycle): True-Q cycle to transpile.
        gate_mapper (defaultdict): map between True-Q to qcal gates.

    Returns:
        deque: transpiled cycle.
    """
    try:
        import trueq as tq
    except ImportError:
        logger.warning(' Unable to import trueq!')

    tcycle = Cycle()
    for q, gate in cycle:

        if isinstance(gate, tq.Meas):
            tcycle.append(gate_mapper['Meas'](q))

        else:
            args = (q,)
            kwargs = {}
            if 'Rz' in gate.name:
                kwargs['theta'] = np.deg2rad(gate.parameters['phi'])

            if 'GE' in gate.name:
                kwargs['subspace'] = 'GE'
                gate_name = gate.name.replace('GE', '')
            elif 'EF' in gate.name:
                kwargs['subspace'] = 'EF'
                gate_name = gate.name.replace('EF', '')
            else:
                gate_name = gate.name

            tcycle.append(
                gate_mapper.call(gate_name, *args, **kwargs)
            )

    return tcycle


def to_qcal(
        circuit,
        gate_mapper:         defaultdict,
        cycle_replacement:   Cycle | Circuit | Dict | None = None,
        barrier_between_all: bool = False
    ) -> Circuit:
    """Transpile a True-Q circuit to a qcal circuit.

    Args:
        circuit (trueq.Circuit): True-Q circuit.
        gate_mapper (defaultdict): map between True-Q to qcal gates.
        cycle_replacement (Cycle | Circuit | Dict | None, optional): dictionary
            which specifies how a marked cycle should be transpiled.
            Defaults to None. For example, ```cycle_replacement = {'marker':
            1, 'cycle': qc.Cycle({qc.MCM(0)})}```.
        barrier_between_all (bool, optional): whether to place a barrier
            between all cycles. Defaults to False. This is useful for
            benchmarking circuits to ensure that the circuit structure is
            preserved.

    Returns:
        Circuit: qcal circuit.
    """
    tcircuit = deque()
    for i, cycle in enumerate(circuit):

        if cycle.marker > 0 and not barrier_between_all:
            tcircuit.append(Barrier(cycle.labels))

        if (cycle.marker > 0 and
            cycle_replacement is not None and
            i < len(circuit) -1):
            # If a marker is specified, only replace the specified cycle
            if isinstance(cycle_replacement, dict):
                if cycle.marker in cycle_replacement.keys():
                    tcycle = cycle_replacement[cycle.marker]
                    if isinstance(tcycle, Cycle):
                        tcircuit.append(tcycle)
                    elif isinstance(tcycle, Circuit):
                        tcircuit.extend(tcycle)
                else:
                    tcircuit.append(
                        transpile_cycle(cycle, gate_mapper)
                    )

            # If no marker is specified, replace every marked cycle
            else:
                tcycle = cycle_replacement
                if isinstance(tcycle, Cycle):
                    tcircuit.append(tcycle)
                elif isinstance(tcycle, Circuit):
                    tcircuit.extend(tcycle)

        else:
            if len(cycle) > 0:
                tcircuit.append(
                    transpile_cycle(cycle, gate_mapper)
                )

        if cycle.marker > 0 and not barrier_between_all and i < len(circuit)-1:
            tcircuit.append(Barrier(circuit.labels))
        elif barrier_between_all and i < (len(circuit) - 1):
            tcircuit.append(Barrier(circuit.labels))

    tcircuit = Circuit(tcircuit)
    return tcircuit


class TrueqTranspiler(Transpiler):
    """True-Q to qcal Transpiler."""
    try:
        import trueq as tq
    except ImportError:
        logger.warning(' Unable to import trueq!')

    # __slots__ = ('_gate_mapper', '_barrier_between_all', '_cycle_replacement')

    def __init__(
            self,
            gate_mapper:         Dict | GateMapper | None = None,
            cycle_replacement:   Cycle | Circuit | Dict | None = None,
            barrier_between_all: bool = False
        ) -> None:
        """Initialize with a gate_mapper.

        Args:
            gate_mapper (Dict | GateMapper | None, optional): dictionary which
                maps TrueQ gates to Qubic gates. Defaults to None.
            cycle_replacement (Cycle | Circuit | Dict | None, optional):
                dictionary which specifies how a marked cycle should be
                transpiled. Defaults to None. For example, ```cycle_replacement
                = {1: qc.Cycle({qc.MCM(0)})}```.
            barrier_between_all (bool, optional): whether to place a barrier
                between all cycles. Defaults to False. This is useful for
                benchmarking circuits to ensure that the circuit structure is
                preserved.
        """
        if gate_mapper is None:
            gate_mapper = GateMapper(
                {**single_qubit_gates, **two_qubit_gates}
            )
        super().__init__(gate_mapper=gate_mapper)
        self._cycle_replacement = cycle_replacement
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
                    self._cycle_replacement,
                    self._barrier_between_all
                )
            )

        tcircuits = CircuitSet(circuits=tcircuits)
        tcircuits['key'] = [str(circ.key) for circ in circuits]
        return tcircuits
