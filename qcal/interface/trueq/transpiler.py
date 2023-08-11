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
from typing import List

logger = logging.getLogger(__name__)


def to_qcal(
        circuit, gate_mapper: defaultdict, cs: CircuitSet, element: int,
        barrier_between_all: bool = False
    ) -> None:
    """Compile a qcal circuit to a qubic circuit.

    Args:
        circuit (tq.Circuit):      True-Q circuit.
        gate_mapper (defaultdict): map between True-Q to qcal gates.
        cs (CircuitSet):           set of transpiled circuits.
        element (int):             element of the CircuitSet.
        barrier_between_all (bool, optional): whether to place a barrier
            between all cycles. Defaults to False. This is useful for
            benchmarking circuits to ensure that the circuit structure is
            preserved.
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
            elif gate.name == 'Rz':
                tcycle.append(
                    gate_mapper[gate.name](
                        gate.parameters['phi'], q
                    )
                )
            else:
                tcycle.append(gate_mapper[gate.name](q))

        tcircuit.append(tcycle)

        if cycle.marker > 0 and not barrier_between_all:
            tcircuit.append(Barrier(cycle.labels))
        elif barrier_between_all and i < (len(circuit) - 1):
            tcircuit.append(Barrier(cycle.labels))

    cs.circuit[element] = Circuit(tcircuit)


class Transpiler(Transpiler):
    """True-Q to qcal Transpiler."""
    import trueq as tq

    __slots__ = ('_gate_mapper', '_barrier_between_all', '_parallelize')

    def __init__(
            self, 
            gate_mapper:         defaultdict | None = None, 
            barrier_between_all: bool = False,
            parallelize:         bool = False
        ) -> None:
        """Initialize with a gate_mapper.

        Args:
            gate_mapper (defaultdict | None, optional): dictionary which maps
                TrueQ gates to Qubic gates. Defaults to None.
            barrier_between_all (bool, optional): whether to place a barrier
                between all cycles. Defaults to False. This is useful for
                benchmarking circuits to ensure that the circuit structure is
                preserved.
            parallelize (bool, optional): whether to use multiprocessing to
                parallelize the circuit transpilation. Defaults to False. This
                can speed up the transpilation process if there are many
                circuits and/or long circuit depths.
        """
        if gate_mapper is None:
            gate_mapper = defaultdict(
                lambda: 'Gate not currently supported!',
                {**single_qubit_gates, **two_qubit_gates}
            )
        super().__init__(gate_mapper=gate_mapper)
        self._barrier_between_all = barrier_between_all
        self._parallelize = parallelize

    def transpile(
            self, circuits: tq.Circuit | tq.CircuitCollection
        ) -> List[Circuit]:
        """Transpile all circuits.

        Args:
            circuits (tq.Circuit | tq.CircuitCollection): circuits to 
                transpile.

        Returns:
            List[Circuit]: transpiled circuits.
        """
        import trueq as tq
        if not isinstance(circuits, tq.CircuitCollection):
            circuits = tq.CircuitCollection(circuits)

        tcircuits = CircuitSet([np.nan for _ in range(len(circuits))])
        if self._parallelize and mp.cpu_count() > 4:
            logger.info(
                f" Pooling {mp.cpu_count() - 2} processes for transpilation..."
            )
            pool = mp.Pool(mp.cpu_count())
            try:
                for i, circuit in enumerate(circuits):
                    pool.apply_async(
                        to_qcal(
                            circuit, self._gate_mapper, tcircuits, i, 
                            self._barrier_between_all
                        )
                    )
            finally:
                pool.close()
                pool.join()

        else:
            for i, circuit in enumerate(circuits):
                to_qcal(
                    circuit, self._gate_mapper, tcircuits, i, 
                    self._barrier_between_all
                )

        tcircuits['key'] = [str(circ.key) for circ in circuits]
        return tcircuits