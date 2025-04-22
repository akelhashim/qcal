"""Submodule for transpiler from cirq and qiskit circuits..

"""
from qcal.circuit import Barrier, Cycle, Layer, Circuit, CircuitSet
from qcal.gate.single_qubit import Meas, Rz, X90, X
from qcal.gate.two_qubit import CNOT, CX, CZ
from qcal.transpilation.transpiler import Transpiler

import logging

from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)


def cirq_to_qcal(circuit, gate_mapper: defaultdict) -> Circuit:
    """Compile a Cirq circuit to a qcal circuit.

    Args:
        circuit (cirq.Circuit): Cirq circuit.
        gate_mapper (defaultdict): map between Cirq to qcal gates.

    Returns:
        Circuit: qcal circuit.
    """
    tcircuit = Circuit()
    for moment in circuit:
        tcycle = Cycle()
        for op in moment:
            if 'Measurement' in str(op):
                # tcircuit.measure(tuple([q.x for q in op.qubits]))
                tcycle = Cycle({Meas(q.x) for q in op.qubits})
            else:
                if 'Rz' in str(op.gate):
                    tcycle.append(Rz(op.qubits[0].x, op.gate._rads))
                else:
                    tcycle.append(
                        gate_mapper[str(op.gate)](
                            tuple([q.x for q in op.qubits])
                        )
                    )
        tcircuit.append(tcycle)

    return tcircuit


def qiskit_to_qcal(circuit, gate_mapper: defaultdict) -> Circuit:
    """Compile a Qiskit circuit to a qcal circuit.

    Args:
        circuit (qiskit.QuantumCircuit): Qiskit circuit.
        gate_mapper (defaultdict): map between Qiskit to qcal gates.

    Returns:
        Circuit: qcal circuit.
    """
    tcircuit = Circuit()
    tlayer = Layer()
    for instr in circuit:
        qubits = tuple([circuit.find_bit(q).index for q in instr.qubits])
        gate = instr.operation.name
        
        if any(q in tlayer.qubits for q in qubits):
            tcircuit.append(tlayer)
            tlayer = Layer()

        if gate == 'barrier':
            tlayer = Barrier(qubits)
        elif gate == 'rz':
            tlayer.append(gate_mapper[gate](
                qubits, theta=instr.operation.params[0])
            )
        else:
            tlayer.append(gate_mapper[gate](qubits))
    tcircuit.append(tlayer)

    return tcircuit


def transpilation_error(*args):
    """Generic transpilation error.

    Raises:
        Exception: transpilation error for non-native gate.
    """
    raise Exception(
        f'Cannot transpile {str(args)} (non-native gate)!'
    ) 


class CirqTranspiler(Transpiler):
    """Cirq Transpiler."""

    __slots__ = ('_gate_mapper',)

    def __init__(self, gate_mapper: defaultdict | Dict | None = None) -> None:
        """Initialize with gate_mapper.

        Args:
            gate_mapper (defaultdict | Dict | None, optional): dictionary which 
                maps str names of Cirq gates to qcal gates. Defaults to None.
        """
        if gate_mapper is None:
            self._gate_mapper = defaultdict(lambda: transpilation_error,
                {'CNOT':     CX,
                 'CZ':       CZ,
                 'CZ**-1.0': CZ,
                 'Rz':       Rz,
                 'Rx(0.5π)': X90,
                 'X':        X
                }
            )
        else:
            self._gate_mapper = gate_mapper
        super().__init__(gate_mapper=self._gate_mapper)


    def transpile(self, circuits: Circuit | CircuitSet | List) -> CircuitSet:
        """Transpile all circuits.

        Args:
            circuits (Circuit | CircuitSet | List): circuits to 
                transpile.

        Returns:
            CircuitSet: transpiled circuits.
        """
        if not isinstance(circuits, CircuitSet):
            circuits = CircuitSet(circuits=circuits)

        tcircuits = []
        for circuit in circuits:
            tcircuits.append(cirq_to_qcal(circuit, self._gate_mapper))
    
        tcircuits = CircuitSet(circuits=tcircuits)
        return tcircuits
    

class QiskitTranspiler(Transpiler):
    """Qiskit Transpiler."""

    __slots__ = ('_gate_mapper',)

    def __init__(self, gate_mapper:  defaultdict | Dict | None = None) -> None:
        """Initialize with gate_mapper.

        Args:
            gate_mapper (defaultdict | Dict | None, optional): dictionary which 
                maps str names of Qiskit gates to qcal gates. Defaults to None.
        """
        if gate_mapper is None:
            self._gate_mapper = defaultdict(lambda: transpilation_error,
                {'barrier': Barrier,
                 'measure': Meas,
                 'cnot': CNOT,
                 'cx':   CX,
                 'cz':   CZ,
                 'rz':   Rz,
                 'rx':   X90,
                 'x':    X
                }
            )
        else:
            self._gate_mapper = gate_mapper
        super().__init__(gate_mapper=self._gate_mapper)


    def transpile(self, circuits: Circuit | CircuitSet | List) -> CircuitSet:
        """Transpile all circuits.

        Args:
            circuits (Circuit | CircuitSet | List): circuits to 
                transpile.

        Returns:
            CircuitSet: transpiled circuits.
        """
        if not isinstance(circuits, CircuitSet):
            circuits = CircuitSet(circuits=circuits)

        tcircuits = []
        for circuit in circuits:
            tcircuits.append(qiskit_to_qcal(circuit, self._gate_mapper))
    
        tcircuits = CircuitSet(circuits=tcircuits)
        return tcircuits