"""Submodule for handling transpilation from qcal to pyquil circuits.

"""
import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Dict, List

import numpy as np

from qcal.circuit import Barrier, Circuit, CircuitSet
from qcal.transpilation.transpiler import Transpiler
from qcal.transpilation.utils import GateMapper
from qcal.units import ns

logger = logging.getLogger(__name__)


__all__ = ('Transpiler',)


def add_CNOT(q0: int, q1: int, **kwargs) -> Iterator:
    """Add a CNOT gate.

    Args:
        q0 (int): control qubit label.
        q1 (int): target qubit label.

    Yields:
        Iterator: CNOT gate.
    """
    try:
        from pyquil.gates import CNOT
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield CNOT(q0, q1)


def add_CZ(q0: int, q1: int, **kwargs) -> Iterator:
    """Add a CZ gate.

    Args:
        q0 (int): control qubit label.
        q1 (int): target qubit label.

    Yields:
        Iterator: CZ gate.
    """
    try:
        from pyquil.gates import CZ
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield CZ(q0, q1)


def add_Idle(qubit: int, duration: float, **kwargs) -> Iterator:
    """Add an idle gate.

    Args:
        qubit (int): qubit label.
        duration (float): idle duration (in seconds).

    Yields:
        Iterator: DELAY gate.
    """
    try:
        from pyquil.gates import DELAY
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield DELAY(qubit, round(duration / (4 * ns)) * 4 * ns)


def add_ISWAP(q0: int, q1: int, **kwargs) -> Iterator:
    """Add an ISWAP gate.

    Args:
        q0 (int): control qubit label.
        q1 (int): target qubit label.

    Yields:
        Iterator: ISWAP gate.
    """
    try:
        from pyquil.gates import ISWAP
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield ISWAP(q0, q1)


# def add_Measure(qubit: int, **kwargs) -> Iterator:
#     """Add a Measurement.

#     Args:
#         qubit (int): qubit label.

#     Yields:
#         Iterator: Measurement
#     """
#     try:
#         from pyquil.gates import Measure
#     except ImportError:
#         logger.warning(' Unable to import pyquil!')
#         return

#     yield Measure(qubit)


def add_SXdag(qubit: int, **kwargs) -> Iterator:
    """Add an SXdag (X-90) gate.

    Args:
        qubit (int): qubit label.

    Yields:
        Iterator: RZ and RX gates.
    """
    try:
        from pyquil.gates import RX, RZ
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield RZ(np.pi, qubit)
    yield RX(np.pi/2, qubit)
    yield RZ(-np.pi, qubit)


def add_SYdag(qubit: int, **kwargs) -> Iterator:
    """Add an SYdag (Y-90) gate.

    Args:
        qubit (int): qubit label.

    Yields:
        Iterator: RZ and RX gates.
    """
    try:
        from pyquil.gates import RZ
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield RZ(np.pi, qubit)
    yield from add_Y90(qubit, **kwargs)
    yield RZ(-np.pi, qubit)


def add_X(qubit: int, **kwargs) -> Iterator:
    """Add an X gate.

    Args:
        qubit (int): qubit label.


    Yields:
        Iterator: RX gate.
    """
    try:
        from pyquil.gates import RX
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield RX(np.pi, qubit)


def add_X90(qubit: int, **kwargs) -> Iterator:
    """Add an X90 gate.

    Args:
        qubit (int): qubit label.


    Yields:
        Iterator: RX gate.
    """
    try:
        from pyquil.gates import RX
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield RX(np.pi/2, qubit)


def add_Y90(qubit: int, **kwargs) -> Iterator:
    """Add a Y90 gate.

    Args:
        qubit (int): qubit label.

    Yields:
        Iterator: RZ and RX gates.
    """
    try:
        from pyquil.gates import RX, RZ
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield RZ(np.pi/2, qubit)
    yield RX(np.pi/2, qubit)
    yield RZ(-np.pi/2, qubit)


def add_Y(qubit: int, **kwargs) -> Iterator:
    """Add a Y gate.

    Args:
        qubit (int): qubit label.

    Yields:
        Iterator: RZ and RX gates.
    """
    try:
        from pyquil.gates import RX, RZ
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield RZ(np.pi/2, qubit)
    yield RX(np.pi, qubit)
    yield RZ(-np.pi/2, qubit)


def add_Rz(qubit: int, phase: float, **kwargs) -> Iterator:
    """Add an Rz gate.

    Args:
        qubit (int): qubit label.
        phase (float): phase.

    Yields:
        Iterator: RZ gate.
    """
    try:
        from pyquil.gates import RZ
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield RZ(phase, qubit)


def to_pyquil(
    circuit:              Circuit,
    gate_mapper:          GateMapper,
    fence_between_cycles: bool = False,
):
    """Transpile a qcal circuit to a pyquil Program.

    Args:
        circuit (Circuit): qcal circuit.
        gate_mapper (GateMapper): map between qcal to quil gates.
        fence_between_cycles (bool, optional): whether to add a fence
            between every cycle. Defaults to False.

    Returns:
        Program: pyquil Program.
    """
    try:
        from pyquil.gates import FENCE
        from pyquil.quil import Program
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    tprogram = Program()
    ro = tprogram.declare('ro', 'BIT', circuit.n_qubits)
    for cycle in circuit:
        if fence_between_cycles:
            tprogram += FENCE(*circuit.qubits)

        if isinstance(cycle, Barrier):
            tprogram += FENCE(*cycle.qubits)
        else:
            for gate in cycle:
                if gate.name in ['Meas', 'MCM']:
                    tprogram += gate_mapper[gate.name](
                        gate.qubits[0],
                        ro[circuit.qubits.index(gate.qubits[0])]
                    )
                else:
                    tprogram += gate_mapper[gate.name](
                        *gate.qubits, **gate.properties['params']
                    )

    return tprogram


class PyquilTranspiler(Transpiler):
    """qcal to Pyquil Transpiler."""

    # __slots__ = ('_gate_mapper',)

    def __init__(
        self,
        gate_mapper:          Dict | GateMapper | None = None,
        fence_between_cycles: bool = False,
    ) -> None:
        """Initialize with a GateMapper.

        Args:
            gate_mapper (Dict | GateMapper | None, optional): dictionary which
                maps qcal gates to pyquil gates. Defaults to ``None``.
            fence_between_cycles (bool, optional): whether to add a fence
                between every cycle. Defaults to ``True``.
        """
        try:
            import pyquil  # noqa: F401
            from pyquil.gates import MEASURE
        except ImportError:
            logger.warning(' Unable to import pyquil!')
            return

        if gate_mapper is None:
            gate_mapper: Mapping[str, Callable] = GateMapper(
                {
                    'CNOT':     add_CNOT,
                    'CZ':       add_CZ,
                    'I':        add_Idle,
                    'Idle':     add_Idle,
                    'iSWAP':    add_ISWAP,
                    'MCM':      MEASURE,
                    'Meas':     MEASURE,
                    'VirtualZ': add_Rz,
                    'SXdag':    add_SXdag,
                    'SYdag':    add_SYdag,
                    'Rz':       add_Rz,
                    'X':        add_X,
                    'X90':      add_X90,
                    'Y':        add_Y,
                    'Y90':      add_Y90,
                    'Z':        add_Rz,
                    'Z90':      add_Rz,
                }
            )
        elif isinstance(gate_mapper, dict):
            gate_mapper = GateMapper(gate_mapper)

        self._fence_between_cycles = fence_between_cycles

        super().__init__(gate_mapper=gate_mapper)

    def transpile(self, circuits: List | CircuitSet) -> CircuitSet:
        """Transpile all circuits.

        Args:
            circuits (List | CircuitSet): circuits to transpile.

        Returns:
            CircuitSet: transpiled circuits.
        """
        if not isinstance(circuits, List) and 'n_circuits' not in dir(circuits):
            circuits = [circuits]
        if isinstance(circuits, List):
            circuits = CircuitSet(circuits=circuits)

        tprograms = []
        for circuit in circuits:
            tprograms.append(
                to_pyquil(
                    circuit, self._gate_mapper, self._fence_between_cycles
                )
            )

        tprograms = CircuitSet(circuits=tprograms)
        return tprograms
