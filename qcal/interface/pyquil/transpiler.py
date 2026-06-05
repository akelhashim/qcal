"""Submodule for handling transpilation from qcal to pyquil circuits.

NOTE: we do not use TYPE_CHECKING for PyQuil types because this might fail if
PyQuil is not installed when building docs.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Dict, Iterable, List

import numpy as np

from qcal.circuit import Barrier, Circuit, CircuitSet, Cycle
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


def add_CX(q0: int, q1: int, **kwargs) -> Iterator:
    """Add a CX gate.

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
        Iterator: DELAY gate with delay rounded to the nearest 4ns.
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


def add_Measure(
    qubit: int,
    classical_ref: pyquil.quilatom.MemoryReference | None = None,  # noqa: F821 # type: ignore
    **kwargs
) -> Iterator:
    """Add a Measurement.

    Args:
        qubit (int): qubit label.
        classical_ref (pyquil.quilatom.MemoryReference | None, optional):
            classical memory reference to store the measurement result.
            Defaults to ``None``.

    Yields:
        Iterator: Measurement
    """
    try:
        from pyquil.gates import MEASURE
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    yield MEASURE(qubit, classical_ref)


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

    if kwargs['subspace'] == 'GE':
        yield RZ(np.pi, qubit)
        yield RX(np.pi/2, qubit)
        yield RZ(-np.pi, qubit)
    elif kwargs['subspace'] == 'EF':
        yield RZ_F12(np.pi, qubit)
        yield RX_F12(np.pi/2, qubit)
        yield RZ_F12(-np.pi, qubit)
    else:
        raise ValueError(f'Invalid subspace: {kwargs["subspace"]}')


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

    if kwargs['subspace'] == 'GE':
        yield RZ(np.pi, qubit)
        yield from add_Y90(qubit, **kwargs)
        yield RZ(-np.pi, qubit)
    elif kwargs['subspace'] == 'EF':
        yield RZ_F12(np.pi, qubit)
        yield from add_Y90(qubit, **kwargs)
        yield RZ_F12(-np.pi, qubit)
    else:
        raise ValueError(f'Invalid subspace: {kwargs["subspace"]}')


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

    if kwargs['subspace'] == 'GE':
        yield RX(np.pi, qubit)
    elif kwargs['subspace'] == 'EF':
        yield RX_F12(np.pi, qubit)
    else:
        raise ValueError(f'Invalid subspace: {kwargs["subspace"]}')


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

    if kwargs['subspace'] == 'GE':
        yield RX(np.pi/2, qubit)
    elif kwargs['subspace'] == 'EF':
        yield RX_F12(np.pi/2, qubit)
    else:
        raise ValueError(f'Invalid subspace: {kwargs["subspace"]}')


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

    if kwargs['subspace'] == 'GE':
        yield RZ(np.pi/2, qubit)
        yield RX(np.pi/2, qubit)
        yield RZ(-np.pi/2, qubit)
    elif kwargs['subspace'] == 'EF':
        yield RZ_F12(np.pi/2, qubit)
        yield RX_F12(np.pi/2, qubit)
        yield RZ_F12(-np.pi/2, qubit)
    else:
        raise ValueError(f'Invalid subspace: {kwargs["subspace"]}')


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

    if kwargs['subspace'] == 'GE':
        yield RZ(np.pi/2, qubit)
        yield RX(np.pi, qubit)
        yield RZ(-np.pi/2, qubit)
    elif kwargs['subspace'] == 'EF':
        yield RZ_F12(np.pi/2, qubit)
        yield RX_F12(np.pi, qubit)
        yield RZ_F12(-np.pi/2, qubit)
    else:
        raise ValueError(f'Invalid subspace: {kwargs["subspace"]}')


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

    if kwargs['subspace'] == 'GE':
        yield RZ(phase, qubit)
    elif kwargs['subspace'] == 'EF':
        yield RZ_F12(phase, qubit)
    else:
        raise ValueError(f'Invalid subspace: {kwargs["subspace"]}')


def RX_F12(theta: float, q: int) -> Any:
    """Create an RX gate in the 1-2 subspace.

    Args:
        theta (float): rotation angle in radians.
        q (int): qubit label.

    Returns:
        pyquil.quilbase.Gate: RX_F12 gate.
    """
    try:
        from pyquil.quilbase import Gate
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return
    return Gate("RX_F12", [theta], [q])


def RZ_F12(theta, q) -> Any:
    """Create an RZ gate in the 1-2 subspace.

    Args:
        theta (float): rotation angle in radians.
        q (int): qubit label.

    Returns:
        pyquil.quilbase.Gate: RZ_F12 gate.
    """
    try:
        from pyquil.quilbase import Gate
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return
    return Gate("RZ_F12", [theta], [q])


def to_pyquil(
    circuit:               Circuit,
    gate_mapper:           GateMapper,
    circuit_for_loop:      bool = False,
    cycles_to_defcircuits: bool = False,
    fence_between_cycles:  bool = True,
) -> Program:  # type: ignore # noqa: F821
    """Transpile a qcal circuit to a PyQuil Program.

    Args:
        circuit (Circuit): qcal circuit.
        gate_mapper (GateMapper): map between qcal to quil gates.
        circuit_for_loop (bool, optional): loops over circuit partitions for
                circuits with repeated structures. Defaults to ``False``.
        cycles_to_defcircuits (bool, optional): whether to write each
            distinct cycle as a DEFCIRCUIT definition and invoke it by name.
            Defaults to ``False``.
        fence_between_cycles (bool, optional): whether to add a fence
            between every cycle. Defaults to ``True``.

    Returns:
        Program: PyQuil Program.
    """
    try:
        from pyquil import Program
        from pyquil.quilatom import LabelPlaceholder
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    tprogram = Program()
    qubit_to_cref = {
        q: tprogram.declare(f'ro{q}', 'BIT', 1)
        for q in circuit.qubits
    }
    if circuit_for_loop:
        for sub_circuit, n_reps in circuit.partitions:
            if n_reps == 1:
                tprogram += transpile_circuit(
                        circuit=Circuit(sub_circuit),
                        gate_mapper=gate_mapper,
                        qubits=circuit.qubits,
                        qubit_to_cref=qubit_to_cref,
                        cycles_to_defcircuits=cycles_to_defcircuits,
                        fence_between_cycles=fence_between_cycles
                )

            elif n_reps > 1:
                counter = tprogram.declare(
                    f'counter{to_pyquil._counter}', 'INTEGER'
                )
                tsub_program = transpile_circuit(
                    circuit=Circuit(sub_circuit),
                    gate_mapper=gate_mapper,
                    qubits=circuit.qubits,
                    qubit_to_cref=qubit_to_cref,
                    cycles_to_defcircuits=cycles_to_defcircuits,
                    fence_between_cycles=fence_between_cycles
                )

                loop = tsub_program.with_loop(
                    num_iterations=n_reps,
                    iteration_count_reference=counter,
                    start_label=LabelPlaceholder(f'START{to_pyquil._counter}'),
                    end_label=LabelPlaceholder(f'END{to_pyquil._counter}'),
                )
                tprogram += loop

                to_pyquil._counter += 1

        tprogram.resolve_label_placeholders()

    else:
        tprogram += transpile_circuit(
            circuit=circuit,
            gate_mapper=gate_mapper,
            qubit_to_cref=qubit_to_cref,
            cycles_to_defcircuits=cycles_to_defcircuits,
            fence_between_cycles=fence_between_cycles
        )

    return tprogram


def transpile_circuit(
    circuit:               Circuit,
    gate_mapper:           GateMapper,
    qubits:                Iterable[int] | None = None,
    qubit_to_cref:         Dict | None = None,
    cycles_to_defcircuits: bool = False,
    fence_between_cycles:  bool = True,
):
    """Transpile a qcal circuit to a PyQuil Program.

    Args:
        circuit (Circuit): qcal circuit.
        gate_mapper (GateMapper): map between qcal to quil gates.
        qubits (Iterable[int] | None, optional): qubits to include in the
            program. Defaults to ``None``, in which case all qubits in the
            passed circuit are included. Being able to pass the circuit qubits
            is useful for transpiling circuit partitions into for-loops, in
            which case the qubits in the partition may be a subset of the qubits
            in the entire circuit.
        qubit_to_cref (Dict | None, optional): mapping from qubit index to
            classical memory reference. When ``None``, a fresh ``ro`` register
            is declared and the mapping is built from ``circuit.qubits``.
            Defaults to ``None``.
        cycles_to_defcircuits (bool, optional): whether to write each
            distinct cycle as a DEFCIRCUIT definition and invoke it by name.
            Defaults to ``False``.
        fence_between_cycles (bool, optional): whether to add a fence
            between every cycle. Defaults to ``True``.

    Returns:
        Program: PyQuil Program.
    """
    try:
        from pyquil.gates import FENCE
        from pyquil.quil import Program
        from pyquil.quilatom import FormalArgument
        from pyquil.quilbase import DefCircuit
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    qubits = circuit.qubits if qubits is None else qubits
    tprogram = Program()
    tprogram_body = Program()

    if qubit_to_cref is None:
        qubit_to_cref = {
            q: tprogram.declare(f'ro{q}', 'BIT', 1)
            for q in circuit.qubits
        }

    if cycles_to_defcircuits:
        cycle_defs = {}

    for i, cycle in enumerate(circuit):
        if fence_between_cycles:
            tprogram_body += FENCE(*qubits)

        if isinstance(cycle, Barrier):
            tprogram_body += FENCE(*cycle.qubits)

        else:
            if cycles_to_defcircuits:
                tcycle = transpile_cycle(
                    cycle, gate_mapper, qubit_to_cref, cycles_to_defcircuits
                )
                cycle_key = next(
                    (k for k, c in cycle_defs.items() if c == cycle),
                    None
                )
                if not cycle_key:
                    cycle_key = f'Cycle_{i}'
                    cycle_defs[cycle_key] = cycle

                    # Create the DefCircuit and add it to the program
                    tprogram += DefCircuit(
                        name=cycle_key,
                        parameters=[],
                        qubits=[
                            FormalArgument(f'q{i}') for i in cycle.qubits
                        ],
                        instructions=tcycle.instructions,
                    )

                tprogram_body += Program(
                    f"{cycle_key} {' '.join(str(q) for q in qubits)}"
                )

            else:
                tprogram_body += transpile_cycle(
                    cycle, gate_mapper, qubit_to_cref, cycles_to_defcircuits
                )

    tprogram += tprogram_body
    return tprogram


def transpile_cycle(
    cycle:                 Cycle,
    gate_mapper:           GateMapper,
    qubit_to_cref:         Dict,
    cycles_to_defcircuits: bool = False,
) -> Program:  # type: ignore # noqa: F821
    """Transpile a single qcal Cycle to a PyQuil Program.

    Args:
        cycle: qcal Cycle.
        gate_mapper (GateMapper): map between qcal to PyQuil gates.
        qubit_to_cref (Dict): mapping from qubit label to classical memory
            reference, used to route measurement results.
        cycles_to_defcircuits (bool, optional): whether to write each
            distinct cycle as a DEFCIRCUIT definition and invoke it by name.
            Defaults to ``False``.

    Returns:
        Program: PyQuil Program for this cycle.
    """
    try:
        from pyquil.quil import Program
        from pyquil.quilatom import FormalArgument
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    tprogram = Program()
    for gate in cycle:
        if gate.name in ['Meas', 'MCM']:
            tprogram += gate_mapper[gate.name](
                gate.qubits[0],
                qubit_to_cref[gate.qubits[0]]
            )
        else:
            if cycles_to_defcircuits:
                qubits = [FormalArgument(f'q{i}') for i in gate.qubits]
            else:
                qubits = gate.qubits

            tprogram += gate_mapper[gate.name](
                *qubits, **{
                    **{'subspace': gate.properties.get('subspace', 'GE')},
                    **gate.properties.get('params', {})
                }
            )
    return tprogram


to_pyquil._counter = 0
DEFAULT_GATEMAPPER: Mapping[str, Callable] = GateMapper(
    {
        'CNOT':     add_CNOT,
        'CX':       add_CNOT,
        'CZ':       add_CZ,
        'I':        add_Idle,
        'Idle':     add_Idle,
        'iSWAP':    add_ISWAP,
        'MCM':      add_Measure,
        'Meas':     add_Measure,
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
class PyQuilTranspiler(Transpiler):
    """qcal to PyQuil Transpiler."""

    # __slots__ = ('_gate_mapper',)

    def __init__(
        self,
        gate_mapper:           Dict | GateMapper | None = None,
        circuit_for_loop:      bool = False,
        cycles_to_defcircuits: bool = False,
        fence_between_cycles:  bool = True,
    ) -> None:
        """Initialize with a GateMapper.

        Args:
            gate_mapper (Dict | GateMapper | None, optional): dictionary which
                maps qcal gates to pyquil gates. Defaults to ``None``.
            circuit_for_loop (bool, optional): loops over circuit partitions for
                circuits with repeated structures. Defaults to ``False``.
            cycles_to_defcircuits (bool, optional): whether to write each
                distinct cycle as a DEFCIRCUIT definition and invoke it by name.
                Defaults to ``False``.
            fence_between_cycles (bool, optional): whether to add a fence
                between every cycle. Defaults to ``True``.
        """
        try:
            import pyquil  # noqa: F401
        except ImportError:
            logger.warning(' Unable to import pyquil!')
            return

        if gate_mapper is None:
            gate_mapper = DEFAULT_GATEMAPPER
        elif isinstance(gate_mapper, dict):
            gate_mapper = GateMapper(gate_mapper)

        self._circuit_for_loop = circuit_for_loop
        self._cycle_to_defcircuits = cycles_to_defcircuits
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
                    circuit=circuit,
                    gate_mapper=self._gate_mapper,
                    circuit_for_loop=self._circuit_for_loop,
                    cycles_to_defcircuits=self._cycle_to_defcircuits,
                    fence_between_cycles=self._fence_between_cycles
                )
            )

        if self._circuit_for_loop:
            to_pyquil._counter = 0

        tprograms = CircuitSet(circuits=tprograms)
        return tprograms
