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
from qcal.interface.pyquil.randomized_compiling import (
    PHASE_GATES,
    PULSE_GATES,
    RCLayerTracker,
    ReadoutLayerTracker,
    _U2Randomization,
    build_rc_configuration,
    effective_layer_period,
    final_layer_cycle_indices,
    has_trailing_single_qubit_layer,
    is_rc_layer,
)
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


def add_MCM(
    qubits: Iterable[int],
    classical_refs: Iterable[
        pyquil.quilatom.MemoryReference | None  # type: ignore # noqa: F821
    ],
    **kwargs
) -> Iterator:
    """Add a mid-circuit measurement, possibly over multiple qubits.

    Args:
        qubits (Iterable[int]): qubit label(s).
        classical_refs (Iterable[pyquil.quilatom.MemoryReference | None]):
            classical memory reference(s) to store the measurement
            result(s), one per qubit in `qubits`.

    Yields:
        Iterator: Measurement(s).
    """
    try:
        from pyquil.gates import MEASURE
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    for qubit, classical_ref in zip(qubits, classical_refs, strict=True):
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
    cycle_replacement:     Dict[Cycle, str] | None = None,
    circuit_for_loop:      bool = False,
    cycles_to_defcircuits: bool = False,
    fence_between_cycles:  bool = True,
    randomized_compiling:  bool = False,
    randomize_readout:     bool = False,
    rc_kwargs:             Dict | None = None,
) -> Program:  # type: ignore # noqa: F821
    """Transpile a qcal circuit to a PyQuil Program.

    Args:
        circuit (Circuit): qcal circuit.
        gate_mapper (GateMapper): map between qcal to quil gates.
        cycle_replacement (Dict[Cycle, str] | None, optional): mapping from
                qcal Cycles to PyQuil DEFCIRCUIT name, used to replace entire
                cycle with a single DEFCIRCUIT call. Defaults to ``None``.
        circuit_for_loop (bool, optional): loops over circuit partitions for
                circuits with repeated structures. Defaults to ``False``.
        cycles_to_defcircuits (bool, optional): whether to write each
            distinct cycle as a DEFCIRCUIT definition and invoke it by name.
            Defaults to ``False``.
        fence_between_cycles (bool, optional): whether to add a fence
            between every cycle. Defaults to ``True``.
        randomized_compiling (bool, optional): whether to randomly compile
            the circuit. Defaults to ``False``.
        randomize_readout (bool, optional): whether to randomize the readout.
            Defaults to ``False``.
        rc_kwargs (Dict | None, optional): keyword arguments forwarded to
            `RandomizedCompilingConfiguration` when `randomized_compiling`
            is ``True`` (e.g. `invert_random_paulis`,
            `shots_per_randomization`, `base_cycle_repetitions`,
            `layer_period`). Defaults to ``None``.

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
    declarations = Program()
    qubit_to_cref = {
        q: declarations.declare(f'ro{q}', 'BIT', 1)
        for q in circuit.qubits
    }
    if circuit_for_loop and not randomized_compiling:
        if randomize_readout:
            logger.warning(
                'Randomized readout is not compatible with repeated '
                'sub-circuits!'
            )
        for sub_circuit, n_reps in circuit.partitions:
            if n_reps == 1:
                _declarations, _tprogram = transpile_circuit(
                        circuit=Circuit(sub_circuit),
                        gate_mapper=gate_mapper,
                        qubits=circuit.qubits,
                        cycle_replacement=cycle_replacement,
                        qubit_to_cref=qubit_to_cref,
                        cycles_to_defcircuits=cycles_to_defcircuits,
                        fence_between_cycles=fence_between_cycles,
                        randomize_readout=randomize_readout
                )
                declarations += _declarations
                tprogram += _tprogram

            elif n_reps > 1:
                counter = declarations.declare(
                    f'counter{to_pyquil._counter}', 'INTEGER'
                )
                _declarations, _tsub_program = transpile_circuit(
                    circuit=Circuit(sub_circuit),
                    gate_mapper=gate_mapper,
                    qubits=circuit.qubits,
                    cycle_replacement=cycle_replacement,
                    qubit_to_cref=qubit_to_cref,
                    cycles_to_defcircuits=cycles_to_defcircuits,
                    fence_between_cycles=fence_between_cycles,
                    # randomize_readout=randomize_readout
                )
                declarations += _declarations

                loop = _tsub_program.with_loop(
                    num_iterations=n_reps,
                    iteration_count_reference=counter,
                    start_label=LabelPlaceholder(f'START{to_pyquil._counter}'),
                    end_label=LabelPlaceholder(f'END{to_pyquil._counter}'),
                )
                tprogram += loop

                to_pyquil._counter += 1

        tprogram.resolve_label_placeholders()

    elif randomized_compiling and not circuit_for_loop:
        _declarations, _tprogram = transpile_circuit(
            circuit=circuit,
            gate_mapper=gate_mapper,
            cycle_replacement=cycle_replacement,
            qubit_to_cref=qubit_to_cref,
            cycles_to_defcircuits=cycles_to_defcircuits,
            fence_between_cycles=fence_between_cycles,
            randomized_compiling=randomized_compiling,
            randomize_readout=randomize_readout,
            rc_kwargs=rc_kwargs,
        )
        rc_configuration = _tprogram.rc_configuration
        rc_source_phases = _tprogram.source_phases
        if randomize_readout:
            readout_configuation = _tprogram.readout_configuation
        declarations += _declarations
        tprogram += _tprogram

    elif randomized_compiling and circuit_for_loop:
        raise ValueError(
            'Randomized compiling builds for-loops internally, ' \
            'so cannot be used with circuit_for_loop.'
        )

    else:
        _declarations, _tprogram = transpile_circuit(
            circuit=circuit,
            gate_mapper=gate_mapper,
            cycle_replacement=cycle_replacement,
            qubit_to_cref=qubit_to_cref,
            cycles_to_defcircuits=cycles_to_defcircuits,
            fence_between_cycles=fence_between_cycles,
            randomize_readout=randomize_readout
        )
        if randomize_readout:
            readout_configuation = _tprogram.readout_configuation
            readout_source_phases = _tprogram.source_phases
        declarations += _declarations
        tprogram += _tprogram

    final_program = declarations + tprogram
    if randomized_compiling:
        final_program.rc_configuration = rc_configuration
        final_program.source_phases = rc_source_phases
    if randomize_readout:
        final_program.readout_configuation = readout_configuation
        if not randomized_compiling:
            final_program.source_phases = readout_source_phases

    return final_program


def transpile_circuit(
    circuit:               Circuit,
    gate_mapper:           GateMapper,
    qubits:                Iterable[int] | None = None,
    cycle_replacement:     Dict[Cycle, str] | None = None,
    qubit_to_cref:         (
        Dict[int, pyquil.quilatom.MemoryReference] | None  # type: ignore # noqa: F821
    ) = None,
    cycles_to_defcircuits: bool = False,
    fence_between_cycles:  bool = True,
    randomized_compiling:  bool = False,
    randomize_readout:     bool = False,
    rc_kwargs:             Dict | None = None,
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
        cycle_replacement (Dict[Cycle, str] | None, optional): mapping from
                qcal Cycles to PyQuil DEFCIRCUIT name, used to replace entire
                cycle with a single DEFCIRCUIT call. Defaults to ``None``.
        qubit_to_cref (Dict[int, pyquil.quilatom.MemoryReference] | None,
            optional): mapping from qubit index to classical memory reference.
            When ``None``, a fresh ``ro`` register is declared and the mapping
            is built from ``circuit.qubits``. Defaults to ``None``.
        cycles_to_defcircuits (bool, optional): whether to write each
            distinct cycle as a DEFCIRCUIT definition and invoke it by name.
            Defaults to ``False``.
        fence_between_cycles (bool, optional): whether to add a fence
            between every cycle. Defaults to ``True``.
        randomized_compiling (bool, optional): whether to randomly compile
            the circuit. Defaults to ``False``.
        randomize_readout (bool, optional): whether to randomize the readout.
            Defaults to ``False``.
        rc_kwargs (Dict | None, optional): keyword arguments forwarded to
            `RandomizedCompilingConfiguration` when `randomized_compiling`
            is ``True`` (e.g. `invert_random_paulis`,
            `shots_per_randomization`, `base_cycle_repetitions`,
            `layer_period`). Defaults to ``None``.

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

    if randomized_compiling and (cycle_replacement or cycles_to_defcircuits):
        raise ValueError(
            'Randomized compiling requires literal per-layer gates (the '
            'twirled phase differs at every layer_index), so cannot be '
            'combined with cycle_replacement/cycles_to_defcircuits.'
        )

    if randomize_readout and (cycle_replacement or cycles_to_defcircuits):
        raise ValueError(
            'Randomized readout requires literal gates in the final '
            'pre-measurement cycle, so cannot be combined with '
            'cycle_replacement/cycles_to_defcircuits.'
        )

    qubits = circuit.qubits if qubits is None else qubits
    tprogram = Program()
    declarations = Program()

    if qubit_to_cref is None:
        qubit_to_cref = {
            q: declarations.declare(f'ro{q}', 'BIT', 1)
            for q in circuit.qubits
        }

    layer_period = effective_layer_period(
        circuit, (rc_kwargs or {}).get('layer_period')
    )

    rc_tracker = None
    if randomized_compiling:
        rc_configuration = build_rc_configuration(
            circuit, qubits, **(rc_kwargs or {})
        )
        rc_tracker = RCLayerTracker(
            rc_configuration,
            reserve_final_layer=has_trailing_single_qubit_layer(
                circuit, layer_period
            ),
        )

    readout_tracker = None
    readout_final_layer = frozenset()
    if randomize_readout:
        readout_configuation = _U2Randomization(
            qubits=qubits,
            destination_names=lambda qubit: f"measurement_unitary_q{qubit}",
            seed_names=lambda qubit: f"measurement_seed_q{qubit}",
            source_name="measurement_unitaries",
            source_unitary_phases=np.array(
                [[ 0.  ,  0.5 ,  0.5 ], # I
                 [-0.25,  0.  , -0.25], # X
                 [ 0.5 ,  0.  ,  0.  ], # Y
                 [ 0.25,  0.5 , -0.25]] # Z
            )
        )
        # RC's own RCLayerTracker already tracks this same final layer
        # (see RCLayerTracker.__init__), so only build a dedicated
        # tracker for it when RC is off.
        if not randomized_compiling:
            readout_tracker = ReadoutLayerTracker(readout_configuation)
            readout_final_layer = final_layer_cycle_indices(circuit)

    non_barrier_idx = 0
    cycle_defs = {}
    for i, cycle in enumerate(circuit):
        if fence_between_cycles:
            tprogram += FENCE()

        if isinstance(cycle, Barrier):
            tprogram += FENCE(*cycle.qubits)

        else:
            # Determine defcircuit name: caller-supplied takes priority,
            # otherwise auto-generate if cycles_to_defcircuits is set.
            if cycle_replacement and cycle in cycle_replacement:
                cycle_key = cycle_replacement[cycle]
            elif cycles_to_defcircuits:
                cycle_key = next(
                    (k for k, c in cycle_defs.items() if c == cycle),
                    f'Cycle_{i}'
                )
            else:
                cycle_key = None

            if cycle_key is not None:
                if cycle_key not in cycle_defs:
                    cycle_defs[cycle_key] = cycle
                    tcycle = transpile_cycle(
                        cycle=cycle,
                        gate_mapper=gate_mapper,
                        qubit_to_cref=qubit_to_cref,
                        cycles_to_defcircuits=True
                    )
                    declarations += DefCircuit(
                        name=cycle_key,
                        parameters=[],
                        qubits=[
                            FormalArgument(f'q{i}') for i in cycle.qubits
                        ],
                        instructions=tcycle.instructions,
                    )
                tprogram += Program(
                    f"{cycle_key} {' '.join(str(q) for q in cycle.qubits)}"
                )

            else:
                tprogram += transpile_cycle(
                    cycle=cycle,
                    gate_mapper=gate_mapper,
                    qubit_to_cref=qubit_to_cref,
                    cycles_to_defcircuits=False,
                    rc_tracker=rc_tracker,
                    readout_tracker=(
                        readout_tracker if i in readout_final_layer
                        else None
                    ),
                )
                if rc_tracker is not None and is_rc_layer(
                    cycle, non_barrier_idx, layer_period
                ):
                    rc_tracker.close_layer()
                non_barrier_idx += 1

    if randomized_compiling:
        rc_program = rc_configuration.build_quil_program()
        if randomize_readout:
            rc_program += readout_configuation.build_quil_program()
            for qubit in qubits:
                call = rc_configuration.apply_pauli_pair(
                    qubit,
                    # rc_configuration.cycle_count,  # TODO: switch
                    rc_configuration._cycle_count,
                    source_unitaries=(
                        readout_configuation.destination_names(qubit)
                    ),
                    target_unitaries=(
                        readout_configuation.destination_names(qubit)
                    ),
                    unitary_offset=0,
                )
                if call is not None:
                    rc_program += call

        tprogram = rc_program + tprogram
        tprogram.rc_configuration = rc_configuration
        tprogram.source_phases = rc_tracker.source_phases
        if randomize_readout:
            tprogram.readout_configuation = readout_configuation

    elif randomize_readout:
        tprogram = readout_configuation.build_quil_program() + tprogram
        tprogram.readout_configuation = readout_configuation
        tprogram.source_phases = readout_tracker.source_phases

    return (declarations, tprogram)


def transpile_cycle(
    cycle:                 Cycle,
    gate_mapper:           GateMapper,
    qubit_to_cref:         Dict[int, pyquil.quilatom.MemoryReference],  # type: ignore # noqa: F821
    cycles_to_defcircuits: bool = False,
    rc_tracker:            RCLayerTracker | None = None,
    readout_tracker:       ReadoutLayerTracker | None = None,
) -> Program:  # type: ignore # noqa: F821
    """Transpile a single qcal Cycle to a PyQuil Program.

    Args:
        cycle: qcal Cycle.
        gate_mapper (GateMapper): map between qcal to PyQuil gates.
        qubit_to_cref (Dict[int, pyquil.quilatom.MemoryReference]): mapping from
            qubit label to classical memory reference, used to route measurement
            results.
        cycles_to_defcircuits (bool, optional): whether to write each
            distinct cycle as a DEFCIRCUIT definition and invoke it by name.
            Defaults to ``False``.
        rc_tracker (RCLayerTracker | None, optional): when randomized
            compiling, the tracker whose `emit_phase_gate` replaces the
            literal Rz gates this Cycle would otherwise emit; X90/SX
            gates still go through `gate_mapper` but get an extra
            per-qubit FENCE. Defaults to ``None``.
        readout_tracker (ReadoutLayerTracker | None, optional): when
            randomizing readout without randomized compiling, the
            tracker whose `emit_phase_gate` replaces the literal Rz
            gates of this Cycle (X90/SX handled as with `rc_tracker`).
            Only ever passed for the final pre-measurement Cycle (see
            `final_layer_cycle_indices`). Defaults to ``None``.

    Returns:
        Program: PyQuil Program for this cycle.
    """
    try:
        from pyquil.gates import FENCE
        from pyquil.quil import Program
        from pyquil.quilatom import FormalArgument
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    tracking = rc_tracker is not None or readout_tracker is not None
    tprogram = Program()
    for gate in cycle:
        if gate.name == 'Meas':
            tprogram += gate_mapper[gate.name](
                gate.qubits[0],
                qubit_to_cref[gate.qubits[0]]
            )
        elif gate.name == 'MCM':
            tprogram += gate_mapper[gate.name](
                gate.qubits,
                [qubit_to_cref[q] for q in gate.qubits]
            )
        elif rc_tracker is not None and gate.name in PHASE_GATES:
            tprogram += rc_tracker.emit_phase_gate(gate)
        elif readout_tracker is not None and gate.name in PHASE_GATES:
            tprogram += readout_tracker.emit_phase_gate(gate)
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
            # A tracker is only ever handed the final layer of a ZXZXZ
            # template (RC's layer boundaries, or the pre-measurement
            # layer -- see final_layer_cycle_indices), so fence its
            # X90/SX pulses per-qubit to preserve that template's
            # ordering when fence_between_cycles is off.
            if tracking and gate.name in PULSE_GATES:
                tprogram += FENCE(*qubits)

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
        'MCM':      add_MCM,
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
        cycle_replacement:     Dict[Cycle, str] | None = None,
        circuit_for_loop:      bool = False,
        cycles_to_defcircuits: bool = False,
        fence_between_cycles:  bool = True,
        randomized_compiling:  bool = False,
        randomize_readout:     bool = False,
        rc_kwargs:             Dict | None = None,
    ) -> None:
        """Initialize with a GateMapper.

        Args:
            gate_mapper (Dict | GateMapper | None, optional): dictionary which
                maps qcal gates to pyquil gates. Defaults to ``None``.
            cycle_replacement (Dict[Cycle, str] | None, optional): mapping from
                qcal Cycles to PyQuil DEFCIRCUIT name, used to replace entire
                cycle with a single DEFCIRCUIT call. Defaults to ``None``.
            circuit_for_loop (bool, optional): loops over circuit partitions for
                circuits with repeated structures. Defaults to ``False``.
            cycles_to_defcircuits (bool, optional): whether to write each
                distinct cycle as a DEFCIRCUIT definition and invoke it by name.
                Defaults to ``False``.
            fence_between_cycles (bool, optional): whether to add a fence
                between every cycle. Defaults to ``True``.
            randomized_compiling (bool, optional): whether to randomly
                compile the circuit. Defaults to ``False``.
            randomize_readout (bool, optional): whether to randomize the
                readout. Defaults to ``False``.
            rc_kwargs (Dict | None, optional): keyword arguments forwarded to
                `RandomizedCompilingConfiguration` when
                `randomized_compiling` is ``True`` (e.g.
                `invert_random_paulis`, `shots_per_randomization`,
                `base_cycle_repetitions`, `layer_period`). Defaults to
                ``None``, in which case `invert_random_paulis=True` and
                `shots_per_randomization=None` are used.
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

        self._cycle_replacement = cycle_replacement
        self._circuit_for_loop = circuit_for_loop
        self._cycles_to_defcircuits = cycles_to_defcircuits
        self._fence_between_cycles = fence_between_cycles
        self._randomized_compiling = randomized_compiling
        self._randomize_readout = randomize_readout
        self._rc_kwargs = rc_kwargs if rc_kwargs is not None else {
            'invert_random_paulis': True,
            # Defaults to 1 if None, but can be set to > 1 for more shots per
            # randomization.
            'shots_per_randomization': None,
            'base_cycle_repetitions': None,
            'layer_period': 5
        }

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
                    cycle_replacement=self._cycle_replacement,
                    circuit_for_loop=self._circuit_for_loop,
                    cycles_to_defcircuits=self._cycles_to_defcircuits,
                    fence_between_cycles=self._fence_between_cycles,
                    randomized_compiling=self._randomized_compiling,
                    randomize_readout=self._randomize_readout,
                    rc_kwargs=self._rc_kwargs,
                )
            )

        if self._circuit_for_loop:
            to_pyquil._counter = 0

        tprograms = CircuitSet(circuits=tprograms)
        return tprograms
