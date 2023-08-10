"""Submodule for handling transpilation from True-Q to qcal circuits.

"""
from qcal.circuit import Barrier, Circuit
from qcal.gate.single_qubit import single_qubit_gates
from qcal.gate.two_qubit import two_qubit_gates
from qcal.transpilation.transpiler import Transpiler

from collections import defaultdict
from typing import List


class Transpiler(Transpiler):
    """True-Q to qcal Transpiler."""
    import trueq as tq

    __slots__ = ('_gate_mapper', '_barrier_between_all')

    def __init__(
            self, 
            gate_mapper:         defaultdict | None = None, 
            barrier_between_all: bool = False
        ) -> None:
        """Initialize with a gate_mapper.

        Args:
            gate_mapper (defaultdict | None, optional): dictionary which maps
                TrueQ gates to Qubic gates. Defaults to None.
            barrier_between_all (bool, optional): whether to place a barrier
                between all cycles. Defaults to False. This is useful for
                benchmarking circuits to ensure that the circuit structure is
                preserved.
        """
        if gate_mapper is None:
            gate_mapper = defaultdict(
                lambda: 'Gate not currently supported!',
                {**single_qubit_gates, **two_qubit_gates}
            )
        super().__init__(gate_mapper=gate_mapper)
        self._barrier_between_all = barrier_between_all

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
        
        transpiled_circuits = []
        for circuit in circuits:
            tcircuit = Circuit()
            
            for i, cycle in enumerate(circuit):
                tcycle = []

                if cycle.marker > 0 and not self._barrier_between_all:
                    tcircuit.append(Barrier(cycle.labels))
                
                for q, gate in cycle:
                    if isinstance(gate, tq.Meas):
                        tcycle.append(self._gate_mapper['Meas'](q))
                    elif gate.name == 'Rz':
                        tcycle.append(
                            self._gate_mapper[gate.name](
                                gate.parameters['phi'], q
                            )
                        )
                    else:
                        tcycle.append(self._gate_mapper[gate.name](q))

                tcircuit.append(tcycle)

                if cycle.marker > 0 and not self._barrier_between_all:
                    tcircuit.append(Barrier(cycle.labels))
                elif self._barrier_between_all and i < (len(circuit) - 1):
                    tcircuit.append(Barrier(cycle.labels))

            transpiled_circuits.append(tcircuit)

        return transpiled_circuits