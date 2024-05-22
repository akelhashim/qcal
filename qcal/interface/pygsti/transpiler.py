"""Submodule for handling transpilation from PyGSTi to qcal circuits.

"""
from qcal.circuit import Barrier, Layer, Circuit, CircuitSet
from qcal.gate.single_qubit import single_qubit_gates
from qcal.gate.two_qubit import two_qubit_gates
from qcal.transpilation.transpiler import Transpiler

import logging
import numpy as np

from collections import defaultdict, deque
from typing import Dict, List

logger = logging.getLogger(__name__)


__all__ = ('Transpiler',)


def to_qcal(
        circuit, 
        gate_mapper: Dict | defaultdict,
    ) -> Circuit:
    """Transpile a PyGSTi circuit to a qcal circuit.

    Args:
        circuit (pygsti.circuits.circuit.Circuit): PyGSTi circuit.
        gate_mapper (Dict | defaultdict): map between PyGSTi to qcal gates.

    Returns:
        Circuit: qcal circuit.
    """
    from pygsti.baseobjs.label import LabelTupTup
    qubits = [int(str(q).replace('Q','')) for q in circuit.line_labels]
    
    tcircuit = Circuit()

    if len(circuit) == 0:
       tcircuit.measure(qubits)

    else:
        for i, layer in enumerate(circuit):
            
            # Parallel gate layer
            if isinstance(layer, LabelTupTup):  
                if len(layer) == 0:  # Idling cycle
                    tcircuit.append(
                        Layer({gate_mapper['Gidle'](q) for q in qubits})
                    )
                else:
                    tlayer = Layer()
                    for gate in layer:
                        gqubits = tuple(
                            int(str(q).replace('Q','')) for q in gate.qubits
                        )
                        if gate.name == 'Gzr':
                            tlayer.append(
                                gate_mapper['Gzr'](
                                    gqubits, float(gate.args[0])
                                )
                            )
                        # elif gate.name == 'Gypi2':
                        #     layer.extend()
                        else:
                            tlayer.append(
                                gate_mapper[gate.name](gqubits)
                            )
                    tcircuit.append(tlayer)
            
            # Isolated gate layer
            else:
                gqubits = tuple(
                    int(str(q).replace('Q','')) for q in layer.qubits
                )
                if layer.name == 'Gzr':
                    tcircuit.append(
                        Layer(
                            {gate_mapper['Gzr'](gqubits, float(layer.args[0]))}
                        )
                    )
                else:
                    tcircuit.append(Layer({gate_mapper[layer.name](gqubits)}))

    tcircuit.measure(qubits=qubits)
    return tcircuit


class Transpiler(Transpiler):
    """PyGSTi to qcal Transpiler."""

    # __slots__ = ('_gate_mapper')

    def __init__(
            self, 
            gate_mapper: Dict | None = None,
        ) -> None:
        """Initialize with a gate_mapper.

        Args:
            gate_mapper (Dict | None, optional): dictionary which maps
                PyGSTi gates to qcal gates. Defaults to None.
        """
        if gate_mapper is None:
            gate_mapper = {
                'Gcnot':   two_qubit_gates['CX'], 
                'Gcphase': two_qubit_gates['CZ'],
                'Gidle':   single_qubit_gates['Id'],
                'Gxpi2':   single_qubit_gates['X90'], 
                'Gypi2':   single_qubit_gates['Y90'],
                'Gzpi2':   single_qubit_gates['Z90'],
                'Gzr':     single_qubit_gates['Rz']
            }
        super().__init__(gate_mapper=gate_mapper)

    def transpile(self, circuits: List | str | CircuitSet) -> CircuitSet:
        """Transpile all circuits.

        Args:
            circuits (List | str | CircuitSet): circuits to transpile.

        Returns:
            CircuitSet: transpiled circuits.
        """
        from pygsti.io import read_circuit_list
        if isinstance(circuits, str):
            circuits = read_circuit_list(circuits)
            circuit_list = [circ.str for circ in circuits]
        elif isinstance(circuits, list):
            circuit_list = [circ.str for circ in circuits]
        elif isinstance(circuits, CircuitSet):
            circuit_list = circuits['pygsti_circuit']

        tcircuits = []
        for circuit in circuits:
            tcircuits.append(
                to_qcal(circuit, self._gate_mapper)
            )
        
        tcircuits = CircuitSet(circuits=tcircuits)
        tcircuits['pygsti_circuit'] = circuit_list
        return tcircuits