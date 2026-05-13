"""Submodule for handling transpilation from pyGSTi to qcal circuits.

"""
import logging
from collections import defaultdict
from collections.abc import Iterator
from typing import Dict, List, Tuple

import numpy as np
from pygsti.baseobjs.label import LabelTupTup
from pygsti.io import read_circuit_list

from qcal.circuit import Barrier, Circuit, CircuitSet, Layer
from qcal.gate.single_qubit import SINGLE_QUBIT_GATES, X90, Y90, Idle, Rz
from qcal.gate.two_qubit import TWO_QUBIT_GATES
from qcal.transpilation.transpiler import Transpiler
from qcal.transpilation.utils import GateMapper
from qcal.units import ns

logger = logging.getLogger(__name__)

__all__ = ('Transpiler',)


def add_Clifford_C0(qubit: int) -> List:
    """Add a Clifford C0 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C0 gate.
    """
    return [Rz(qubit, theta=0), Rz(qubit, theta=0), Rz(qubit, theta=0)]


def add_Clifford_C1(qubit: int) -> List:
    """Add a Clifford C1 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C1 gate.
    """
    return [Rz(qubit, theta=0), X90(qubit), Rz(qubit, theta=np.pi/2)]


def add_Clifford_C2(qubit: int) -> List:
    """Add a Clifford C2 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C2 gate.
    """
    return [Rz(qubit, theta=np.pi/2), X90(qubit), Rz(qubit, theta=np.pi)]


def add_Clifford_C3(qubit: int) -> List:
    """Add a Clifford C3 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C3 gate.
    """
    return [Rz(qubit, theta=0), X90(qubit), X90(qubit)]


def add_Clifford_C4(qubit: int) -> List:
    """Add a Clifford C4 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C4 gate.
    """
    return [Rz(qubit, theta=np.pi), X90(qubit), Rz(qubit, theta=-np.pi/2)]


def add_Clifford_C5(qubit: int) -> List:
    """Add a Clifford C5 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C5 gate.
    """
    return [Rz(qubit, theta=np.pi/2), X90(qubit), Rz(qubit, theta=0)]


def add_Clifford_C6(qubit: int) -> List:
    """Add a Clifford C6 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C6 gate.
    """
    return [Rz(qubit, theta=np.pi), X90(qubit), X90(qubit)]


def add_Clifford_C7(qubit: int) -> List:
    """Add a Clifford C7 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C7 gate.
    """
    return [Rz(qubit, theta=0), X90(qubit), Rz(qubit, theta=-np.pi/2)]


def add_Clifford_C8(qubit: int) -> List:
    """Add a Clifford C8 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C8 gate.
    """
    return [Rz(qubit, theta=-np.pi/2), X90(qubit), Rz(qubit, theta=0)]


def add_Clifford_C9(qubit: int) -> List:
    """Add a Clifford C9 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C9 gate.
    """
    return [Rz(qubit, theta=np.pi), Rz(qubit, theta=0), Rz(qubit, theta=0)]


def add_Clifford_C10(qubit: int) -> List:
    """Add a Clifford C10 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C10 gate.
    """
    return [Rz(qubit, theta=np.pi), X90(qubit), Rz(qubit, theta=np.pi/2)]


def add_Clifford_C11(qubit: int) -> List:
    """Add a Clifford C11 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C11 gate.
    """
    return [Rz(qubit, theta=-np.pi/2), X90(qubit), Rz(qubit, theta=np.pi)]


def add_Clifford_C12(qubit: int) -> List:
    """Add a Clifford C12 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C12 gate.
    """
    return [Rz(qubit, theta=np.pi/2), X90(qubit), Rz(qubit, theta=np.pi/2)]


def add_Clifford_C13(qubit: int) -> List:
    """Add a Clifford C13 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C13 gate.
    """
    return [Rz(qubit, theta=np.pi), X90(qubit), Rz(qubit, theta=-np.pi)]


def add_Clifford_C14(qubit: int) -> List:
    """Add a Clifford C14 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C14 gate.
    """
    return [Rz(qubit, theta=np.pi/2), Rz(qubit, theta=0), Rz(qubit, theta=0)]


def add_Clifford_C15(qubit: int) -> List:
    """Add a Clifford C15 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C15 gate.
    """
    return [Rz(qubit, theta=np.pi/2), X90(qubit), Rz(qubit, theta=-np.pi/2)]


def add_Clifford_C16(qubit: int) -> List:
    """Add a Clifford C16 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C16 gate.
    """
    return [Rz(qubit, theta=0), X90(qubit), Rz(qubit, theta=0)]


def add_Clifford_C17(qubit: int) -> List:
    """Add a Clifford C17 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C17 gate.
    """
    return [Rz(qubit, theta=-np.pi/2), X90(qubit), X90(qubit)]


def add_Clifford_C18(qubit: int) -> List:
    """Add a Clifford C18 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C18 gate.
    """
    return [Rz(qubit, theta=-np.pi/2), X90(qubit), Rz(qubit, theta=-np.pi/2)]


def add_Clifford_C19(qubit: int) -> List:
    """Add a Clifford C19 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C19 gate.
    """
    return [Rz(qubit, theta=np.pi), X90(qubit), Rz(qubit, theta=0)]


def add_Clifford_C20(qubit: int) -> List:
    """Add a Clifford C20 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C20 gate.
    """
    return [Rz(qubit, theta=np.pi/2), X90(qubit), X90(qubit)]


def add_Clifford_C21(qubit: int) -> List:
    """Add a Clifford C21 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C21 gate.
    """
    return [Rz(qubit, theta=-np.pi/2), X90(qubit), Rz(qubit, theta=np.pi/2)]


def add_Clifford_C22(qubit: int) -> List:
    """Add a Clifford C22 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C22 gate.
    """
    return [Rz(qubit, theta=0), X90(qubit), Rz(qubit, theta=np.pi)]


def add_Clifford_C23(qubit: int) -> List:
    """Add a Clifford C23 gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Clifford C23 gate.
    """
    return [Rz(qubit, theta=-np.pi/2), Rz(qubit, theta=0), Rz(qubit, theta=0)]


def add_Hadamard(qubit: int) -> List:
    """Add a Hadamard gate.

    Args:
        qubit (int): qubit label.

    Returns:
        List: list of gates corresponding to the Hadamard gate.
    """
    return [Rz(qubit, theta=np.pi/2), X90(qubit), Rz(qubit, theta=np.pi/2)]


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
    circuit: Circuit,
    gate_mapper: Dict | defaultdict,
) -> Circuit:
    """Transpile a pyGSTi circuit to a qcal circuit.

    Args:
        circuit (pyGSTi.circuits.circuit.Circuit): pyGSTi circuit.
        gate_mapper (Dict | defaultdict): map between pyGSTi to qcal gates.

    Returns:
        Circuit: qcal circuit.
    """
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
                    # For handling cases where a single pyGSTi gate maps to
                    # multiple qcal layers
                    tsubcircuit = Circuit()
                    for gate in layer:
                        gqubits = tuple(
                            int(str(q).replace('Q','')) for q in gate.qubits
                        )
                        args = (
                            (gqubits,) if len(gate.args) == 0 else
                            (gqubits, float(gate.args[0]))
                        )
                        out = gate_mapper.call(gate.name, *args)
                        if isinstance(out, set):
                            tlayer.append(out)
                        elif isinstance(out, Circuit):
                            tsubcircuit.join(out)

                    if tlayer.n_gates > 0 and tsubcircuit.n_cycles == 0:
                        tcircuit.append(tlayer)
                    elif tlayer.n_gates == 0 and tsubcircuit.n_cycles > 0:
                        tcircuit.extend(tsubcircuit)
                    elif tlayer.n_gates > 0 and tsubcircuit.n_cycles > 0:
                        tsubcircuit.join(Circuit([tlayer]))
                        tcircuit.extend(tsubcircuit)

            # Isolated gate layer
            else:
                gqubits = tuple(
                    int(str(q).replace('Q','')) for q in layer.qubits
                )
                args = (
                    (gqubits,) if len(layer.args) == 0 else
                    (gqubits, float(layer.args[0]))
                )
                out = gate_mapper.call(layer.name, *args)
                if isinstance(out, set):
                    tcircuit.append(Layer(out))
                elif isinstance(out, Circuit):
                    tcircuit.extend(out)

            # Add a barrier between each layer to prevent compression via compilation
            tcircuit.append(Barrier(qubits))

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
                    'Empty':   SINGLE_QUBIT_GATES['Id'],
                    'Gcnot':   TWO_QUBIT_GATES['CX'],
                    'Gcphase': TWO_QUBIT_GATES['CZ'],
                    'Gc0':     add_Clifford_C0,
                    'Gc1':     add_Clifford_C1,
                    'Gc2':     add_Clifford_C2,
                    'Gc3':     add_Clifford_C3,
                    'Gc4':     add_Clifford_C4,
                    'Gc5':     add_Clifford_C5,
                    'Gc6':     add_Clifford_C6,
                    'Gc7':     add_Clifford_C7,
                    'Gc8':     add_Clifford_C8,
                    'Gc9':     add_Clifford_C9,
                    'Gc10':    add_Clifford_C10,
                    'Gc11':    add_Clifford_C11,
                    'Gc12':    add_Clifford_C12,
                    'Gc13':    add_Clifford_C13,
                    'Gc14':    add_Clifford_C14,
                    'Gc15':    add_Clifford_C15,
                    'Gc16':    add_Clifford_C16,
                    'Gc17':    add_Clifford_C17,
                    'Gc18':    add_Clifford_C18,
                    'Gc19':    add_Clifford_C19,
                    'Gc20':    add_Clifford_C20,
                    'Gc21':    add_Clifford_C21,
                    'Gc22':    add_Clifford_C22,
                    'Gc23':    add_Clifford_C23,
                    'Gh':      add_Hadamard,
                    'Gi':      SINGLE_QUBIT_GATES['Id'],
                    'Gii':     add_global_identity,
                    'Gidle':   add_idle,
                    'Grz':     SINGLE_QUBIT_GATES['Rz'],
                    'Gxpi2':   SINGLE_QUBIT_GATES['X90'],
                    'Gypi2':   SINGLE_QUBIT_GATES['Y90'],
                    'Gzpi2':   SINGLE_QUBIT_GATES['Z90'],
                    'Gzpi':    SINGLE_QUBIT_GATES['Z'],
                    'Gzr':     SINGLE_QUBIT_GATES['Rz'],
                    'Gzmpi2':  SINGLE_QUBIT_GATES['Sdag'],
                    'Gxx':     add_parallel_X90s,
                    'Gxy':     add_parallel_X90_Y90,
                    'Gyx':     add_parallel_Y90_X90,
                    'Gyy':     add_parallel_Y90s,
                    'Iz':      SINGLE_QUBIT_GATES['MCM']
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
