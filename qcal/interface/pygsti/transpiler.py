"""Submodule for handling transpilation from pyGSTi to qcal circuits.

"""
import logging
from collections import defaultdict
from collections.abc import Iterator
from typing import Dict, List, Tuple

from qcal.circuit import Circuit, CircuitSet, Layer
from qcal.gate.single_qubit import X90, Y90, Idle, single_qubit_gates
from qcal.gate.two_qubit import two_qubit_gates
from qcal.transpilation.transpiler import Transpiler
from qcal.transpilation.utils import GateMapper
from qcal.units import ns

logger = logging.getLogger(__name__)


__all__ = ('Transpiler',)


def add_idle(qubit: int) -> Idle:
    """Add idle gate.

    Args:
        qubit (int): qubit label.

    Returns:
        Idle: idle gate of duration 100 ns.
    """
    return Idle(qubit, duration=100*ns)


def add_global_identity(qubits: Tuple[int]) -> None:
    """Add a global identity gate.

    Args:
        qubits (Tuple[int]): qubit labels.

    Returns:
        None: No cycle is returned.
    """
    return set()


def add_parallel_X90s(qubits: Tuple[int]) -> Iterator[X90]:
    """Add parallel X90 gates.

    Args:
        qubits (Tuple[int]): qubit labels.

    Yields:
        Iterator[X90]: X90 gate on each qubit.
    """
    for q in qubits:
        yield X90(q)


def add_parallel_X90_Y90(qubits: Tuple[int]) -> Iterator[X90 | Y90]:
    """Add parallel X90 and Y90 gates.

    Args:
        qubits (Tuple[int]): qubit labels.

    Yields:
        Iterator[X90 | Y90]: X90 and Y90 gates.
    """
    yield X90(qubits[0])
    yield Y90(qubits[1])


def add_parallel_Y90s(qubits: Tuple[int]) -> Iterator[Y90]:
    """Add parallel Y90 gates.

    Args:
        qubits (Tuple[int]): qubit labels.

    Yields:
        Iterator[Y90]: Y90 gate on each qubit.
    """
    for q in qubits:
        yield Y90(q)


def add_parallel_Y90_X90(qubits: Tuple[int]) -> Iterator[Y90 | X90]:
    """Add parallel Y90 and X90 gates.

    Args:
        qubits (Tuple[int]): qubit labels.

    Yields:
        Iterator[Y90 | X90]: Y90 and X90 gates.
    """
    yield Y90(qubits[0])
    yield X90(qubits[1])


def to_qcal(
        circuit,
        gate_mapper: Dict | defaultdict,
    ) -> Circuit:
    """Transpile a pyGSTi circuit to a qcal circuit.

    Args:
        circuit (pygsti.circuits.circuit.Circuit): pyGSTi circuit.
        gate_mapper (Dict | defaultdict): map between pyGSTi to qcal gates.

    Returns:
        Circuit: qcal circuit.
    """
    from pygsti.baseobjs.label import LabelTupTup
    qubits = [int(str(q).replace('Q','')) for q in circuit.line_labels]

    tcircuit = Circuit()

    if len(circuit) == 0:
       tcircuit.measure(qubits)

    else:
        for _, layer in enumerate(circuit):
            # Parallel gate layer
            if isinstance(layer, LabelTupTup):
                if len(layer) == 0:  # Idling cycle
                    tcircuit.append(
                        Layer({gate_mapper['Empty'](q) for q in qubits})
                    )
                else:
                    tlayer = Layer()
                    for gate in layer:
                        gqubits = tuple(
                            int(str(q).replace('Q','')) for q in gate.qubits
                        )
                        args = (
                            (gqubits,) if len(gate.args) == 0 else
                            (gqubits, float(gate.args[0]))
                        )
                        tlayer.append(
                            gate_mapper.call(gate.name, *args)
                        )

                    tcircuit.append(tlayer)

            # Isolated gate layer
            else:
                gqubits = tuple(
                    int(str(q).replace('Q','')) for q in layer.qubits
                )
                args = (
                    (gqubits,) if len(layer.args) == 0 else
                    (gqubits, float(layer.args[0]))
                )
                tcircuit.append(
                    Layer(gate_mapper.call(layer.name, *args))
                )

        tcircuit.measure(qubits=qubits)

    return tcircuit


class PyGSTiTranspiler(Transpiler):
    """pyGSTi to qcal Transpiler."""

    # __slots__ = ('_gate_mapper',)

    def __init__(
            self,
            gate_mapper: Dict | GateMapper | None = None,
        ) -> None:
        """Initialize with a GateMapper.

        Args:
            gate_mapper (Dict | GateMapper | None, optional): dictionary which
                maps pyGSTi gates to qcal gates. Defaults to None.
        """
        if gate_mapper is None:
            gate_mapper = GateMapper(
                {
                    'Empty':   single_qubit_gates['Id'],
                    'Gcnot':   two_qubit_gates['CX'],
                    'Gcphase': two_qubit_gates['CZ'],
                    'Gi':      single_qubit_gates['Id'],
                    'Gii':     add_global_identity,
                    'Gidle':   add_idle,
                    'Grz':     single_qubit_gates['Rz'],
                    'Gxpi2':   single_qubit_gates['X90'],
                    'Gypi2':   single_qubit_gates['Y90'],
                    'Gzpi2':   single_qubit_gates['Z90'],
                    'Gzpi':    single_qubit_gates['Z'],
                    'Gzr':     single_qubit_gates['Rz'],
                    'Gzmpi2':  single_qubit_gates['Sdag'],
                    'Gxx':     add_parallel_X90s,
                    'Gxy':     add_parallel_X90_Y90,
                    'Gyx':     add_parallel_Y90_X90,
                    'Gyy':     add_parallel_Y90s
                }
            )

        elif isinstance(gate_mapper, dict):
            gate_mapper = GateMapper(gate_mapper)

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
        for _, circuit in enumerate(circuits):
            tcircuits.append(
                to_qcal(circuit, self._gate_mapper)
            )

        tcircuits = CircuitSet(circuits=tcircuits)
        tcircuits['pygsti_circuit'] = circuit_list
        return tcircuits
