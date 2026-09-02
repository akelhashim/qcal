"""Submodule for hardware-level randomized compiling via PyQuil.

Circuits reaching this module are assumed to already be compiled to the
Rz/X90/two-qubit-gate native gate set (see `qcal.interface.bqskit`), i.e.
every single-qubit layer is already expressed as the ZXZXZ template
(Rz-X90-Rz-X90-Rz). This module does not re-derive that decomposition; it
provides the pieces specific to randomized compiling -- `base_cycles`
extraction and (layer_index, angle_index) tracking -- that
`qcal.interface.pyquil.transpiler.transpile_circuit`/`transpile_cycle`
call into while doing their own single walk of the circuit, merging
random Pauli twirls into the existing Rz phases at pulse-execution time
via `pyquil._qpu.randomized_compiling` instead of compiling a distinct
circuit per randomization.

NOTE: we do not use TYPE_CHECKING for PyQuil types because this might fail
if PyQuil is not installed when building docs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Sequence,
    Tuple,
    final,
)

import numpy as np
from numpy.typing import NDArray

from qcal.circuit import Circuit, Cycle

logger = logging.getLogger(__name__)


__all__ = (
    'PHASE_GATES',
    'PULSE_GATES',
    'is_rc_layer',
    'cycle_to_base_cycle',
    'build_rc_configuration',
    'RCLayerTracker',
    'final_layer_cycle_indices',
    'ReadoutLayerTracker',
)


PHASE_GATES = ('Rz', 'VirtualZ')
PULSE_GATES = ('X90', 'SX')
_ANGLES_PER_UNITARY = 3  # ZXZXZ decomposition


def is_rc_layer(cycle: Cycle) -> bool:
    """Whether a Cycle is a randomized-compiling layer boundary.

    A Cycle advances the RC layer index if it contains a two-qubit gate
    (an edge to twirl) or a mid-circuit measurement (`Meas`/`MCM`). MCM
    layers must still be represented as their own `base_cycles` entry --
    a tuple of bare ints for every qubit, including the measured ones --
    so `RCLayerTracker.layer_index` stays aligned with `base_cycles`.
    This is used both to filter `base_cycles` in `build_rc_configuration`
    and to decide when `RCLayerTracker.close_layer` fires in
    `qcal.interface.pyquil.transpiler.transpile_circuit`; both call
    sites must agree.

    Args:
        cycle (Cycle): qcal Cycle.

    Returns:
        bool: True if `cycle` should count as an RC layer.
    """
    return any(
        len(gate.qubits) == 2 or gate.is_measurement
        for gate in cycle.gates
    )


def cycle_to_base_cycle(
    cycle: Cycle, qubits: Sequence[int]
) -> Tuple[Tuple[int, int] | int, ...]:
    """Convert a qcal Cycle into a PyQuil RC 'base cycle' entry.

    Two-qubit gates become edges ``(q0, q1)``. Every other qubit in
    `qubits` becomes a bare int, since ``pyquil._qpu.randomized_compiling``
    only distinguishes "edge" from "identity" for Pauli-propagation
    bookkeeping. This function is only called on RC-layer-boundary cycles
    (see `is_rc_layer`), so any non-edge qubit here is assumed to be
    truly idle or running an identity-equivalent operation (e.g., a
    dynamical-decoupling sequence during a 2-qubit gate or MCM). Real
    single-qubit content lives in the separate ZXZXZ-template cycles
    between boundaries, which are not RC layers and never reach this function;
    those angles are tracked instead via `RCLayerTracker`'s source-unitary
    phases. A Meas/MCM gate is deliberately treated the same as an idle
    qubit here (see the `gate.is_measurement` check below, which just
    skips the edge/degree check for it): its qubit(s) fall through to
    the bare-int branch below, so pyquil's RC engine twirls it as an
    identity op like any other idle qubit, whether the gate spans one
    qubit or several.

    Args:
        cycle (Cycle): qcal Cycle containing at least one two-qubit gate,
            or a mid-circuit measurement layer (see `is_rc_layer`).
        qubits (Sequence[int]): full qubit register to twirl.

    Returns:
        Tuple[Tuple[int, int] | int, ...]: e.g. ``((0, 1), 2)``, or, for
            an MCM layer, ``(0, 1, 2)``.

    Raises:
        ValueError: if a non-measurement gate in `cycle` acts on more
            than two qubits.
    """
    entries: List[Tuple[int, int] | int] = []
    edge_qubits = set()
    for gate in cycle.gates:
        if gate.is_measurement:
            continue
        if len(gate.qubits) == 2:
            edge = tuple(sorted(gate.qubits))
            entries.append(edge)
            edge_qubits.update(edge)
        elif len(gate.qubits) > 2:
            raise ValueError(
                f"Gate '{gate.name}' on {gate.qubits} touches more than "
                "two qubits; randomized compiling only supports 1- and "
                "2-qubit gates."
            )

    for q in qubits:
        if q not in edge_qubits:
            entries.append(q)

    return tuple(entries)


def build_rc_configuration(
    circuit: Circuit,
    qubits:  Sequence[int],
    base_cycle_repetitions: int | None = None,
    **rc_kwargs,
) -> pyquil._qpu.randomized_compiling.RandomizedCompilingConfiguration: # type: ignore  # noqa: F821
    """Build a RandomizedCompilingConfiguration from a compiled Circuit.

    `circuit` is the fully unrolled sequence to run. Its RC-layer
    Cycles (see `is_rc_layer`) are walked in order into a flat
    per-cycle layer sequence, which must tile exactly into
    `base_cycle_repetitions` repetitions of an equal-length unit --
    that unit becomes `base_cycles`. E.g. a repetition code's two
    entangling cycles plus an MCM cycle, repeated N times, collapses
    to a 3-cycle `base_cycles` with `base_cycle_repetitions=N`; a
    circuit with no repeated structure yields `base_cycle_repetitions
    == 1` and `base_cycles` equal to the full layer sequence.

    Args:
        circuit (Circuit): compiled qcal circuit to twirl.
        qubits (Sequence[int]): full qubit register to twirl.
        base_cycle_repetitions (int | None, optional): number of times
            `base_cycles` repeats in `circuit`. Defaults to `None`,
            which auto-detects the smallest repeating unit (and its
            repeat count) from the circuit's actual RC-layer sequence
            (see `_minimal_period`). Pass an explicit value to require
            a specific repeat count instead -- a mismatch raises
            `ValueError`.
        **rc_kwargs: forwarded to `RandomizedCompilingConfiguration`
            (e.g. `invert_random_paulis`, `shots_per_randomization`).

    Returns:
        RandomizedCompilingConfiguration: PyQuil RC configuration.

    Raises:
        ValueError: if an explicit `base_cycle_repetitions` does not
            evenly divide the circuit's RC-layer count, or the layer
            sequence turns out not to actually be that many
            repetitions of the resulting unit.
    """
    try:
        from pyquil._qpu import randomized_compiling as rc
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    layers = tuple(
        cycle_to_base_cycle(cycle, qubits)
        for cycle in circuit.cycles[:-1]  # Exclude terminal measurements
        if not cycle.is_barrier and is_rc_layer(cycle)
    )

    if base_cycle_repetitions is None:
        period = _minimal_period(layers)
        base_cycle_repetitions = len(layers) // period if period else 1
    else:
        if base_cycle_repetitions < 1:
            raise ValueError(
                "base_cycle_repetitions must be >= 1, got "
                f"{base_cycle_repetitions}."
            )
        if len(layers) % base_cycle_repetitions:
            raise ValueError(
                f"The circuit has {len(layers)} RC layers, which does "
                f"not divide evenly into base_cycle_repetitions="
                f"{base_cycle_repetitions}."
            )
        period = len(layers) // base_cycle_repetitions

    base_cycles = layers[:period]
    if base_cycles * base_cycle_repetitions != layers:
        raise ValueError(
            f"The circuit's {len(layers)} RC layers are not "
            f"{base_cycle_repetitions} repetitions of the same "
            f"{period}-cycle unit; pass the base_cycle_repetitions "
            "that matches the circuit's actual repeated structure."
        )

    return rc.RandomizedCompilingConfiguration(
        base_cycles=base_cycles,
        base_cycle_repetitions=base_cycle_repetitions,
        variables=rc.RandomizedCompilingVariables(
            unitaries_prefix="source_phases"
        ),
        **rc_kwargs,
    )


class RCLayerTracker:
    """Tracks (layer_index, angle_index) state for randomized compiling.

    `qcal.interface.pyquil.transpiler.transpile_circuit` owns the actual
    circuit/cycle walk; it calls into this tracker only for the piece
    specific to randomized compiling -- swapping a Rz/VirtualZ gate's
    literal phase for a memory-reference-parametrized one (recording the
    original phase into `source_phases`) -- and advances to the
    next layer once a Cycle's two-qubit gates have been emitted.
    X90/SX pulses are still emitted through the ordinary `gate_mapper`
    (see `transpile_cycle`), which additionally fences them per-qubit
    while a tracker is active.

    `layer_index` is a single counter shared by every qubit. `angle_index`
    (0, 1, 2) is a per-qubit counter that resets to 0 whenever
    `close_layer` is called.
    """

    def __init__(self, configuration) -> None:
        """Initialize the tracker from an RC configuration.

        Args:
            configuration (RandomizedCompilingConfiguration): from
                `build_rc_configuration`.
        """
        self.configuration = configuration
        self.layer_index = 0
        self.angle_index: Dict[int, int] = dict.fromkeys(
            configuration.qubits_sorted, 0
        )

        n_layers = (
            configuration.base_cycle_repetitions
            * len(configuration.base_cycles) + 1
        )
        self.source_phases: Dict[str, List[float]] = {
            configuration.variables.source_unitaries(q): (
                [0.0] * (n_layers * 3)
            ) for q in configuration.qubits_sorted
        }

    def emit_phase_gate(self, gate) -> 'Program':  # type: ignore  # noqa: F821
        """Replace a Rz/VirtualZ gate with its twirled memory reference.

        Records the gate's original phase into `source_phases` at
        (layer_index, angle_index) and advances angle_index for `gate`'s
        qubit.

        Args:
            gate: qcal Rz or VirtualZ gate.

        Returns:
            Program: single-instruction PyQuil Program.
        """
        from pyquil.gates import RZ

        q = gate.qubits[0]
        phase = gate.properties['params']['phase']
        key = self.configuration.variables.source_unitaries(q)
        self.source_phases[key][
            self.layer_index * 3 + self.angle_index[q]
        ] = phase / (2 * np.pi)

        ref = self.configuration.variables.twirled_unitaries_ref(
            q, self.layer_index, self.angle_index[q]
        )
        self.angle_index[q] += 1
        return RZ(2 * np.pi * ref, q)

    def close_layer(self) -> None:
        """Advance to the next layer, resetting per-qubit angle_index."""
        self.layer_index += 1
        self.angle_index = dict.fromkeys(
            self.configuration.qubits_sorted, 0
        )


def final_layer_cycle_indices(circuit: Circuit) -> FrozenSet[int]:
    """Indices of every Cycle in the final pre-measurement layer.

    The final layer is the full ZXZXZ template -- however many Cycles
    it spans (up to 5: Rz-X90-Rz-X90-Rz) -- immediately preceding a
    circuit's terminal measurement(s): every non-barrier Cycle after
    the last RC-layer boundary (see `is_rc_layer`), excluding the
    boundary cycles themselves (barriers and the terminal Meas/MCM
    cycle(s)). This is the layer `ReadoutLayerTracker` should be
    scoped to when randomizing readout without randomized compiling
    (RC's own `RCLayerTracker` already tracks every layer, including
    this one, via its reserved final layer -- see `RCLayerTracker.
    __init__`'s `n_layers = ... + 1` -- so this is only needed when RC
    is off).

    Args:
        circuit (Circuit): qcal circuit.

    Returns:
        FrozenSet[int]: indices into `circuit`, possibly empty (e.g. a
            circuit with no single-qubit content before its terminal
            measurement(s), or with no measurement at all).
    """
    i = len(circuit) - 1
    while i >= 0:
        cycle = circuit[i]
        if cycle.is_barrier:
            i -= 1
        elif all(gate.is_measurement for gate in cycle.gates):
            i -= 1
        else:
            break

    indices = []
    while i >= 0:
        cycle = circuit[i]
        if cycle.is_barrier:
            i -= 1
        elif is_rc_layer(cycle):
            break
        else:
            indices.append(i)
            i -= 1
    return frozenset(indices)


class ReadoutLayerTracker:
    """Tracks and sets the final pre-measurement phases for readout
    randomization when randomized compiling is not also active.

    Mirrors `RCLayerTracker`'s "record the literal phase, then emit a
    memory-reference-parametrized gate in its place" pattern, but is
    scoped to the single ZXZXZ template immediately preceding each
    qubit's measurement (see `final_layer_cycle_indices`) -- there is
    exactly one layer, so, unlike `RCLayerTracker`, there is no
    `layer_index`/`close_layer` to advance. X90/SX pulses are still
    emitted through the ordinary `gate_mapper` (see `transpile_cycle`),
    which additionally fences them per-qubit while this tracker is
    active.

    When randomized compiling *is* active, `RCLayerTracker` already
    tracks this same final layer on its own, so this tracker is unused
    in that case (see `qcal.interface.pyquil.transpiler.
    transpile_circuit`).
    """

    def __init__(self, configuration: _U2Randomization) -> None:
        """Initialize the tracker from a readout-randomization config.

        Args:
            configuration (_U2Randomization): built in
                `transpile_circuit` when `randomize_readout` is True.
        """
        self.configuration = configuration
        self.angle_index: Dict[int, int] = dict.fromkeys(
            configuration.qubits, 0
        )
        self.source_phases: Dict[int, List[float]] = {
            q: [0.0] * 3 for q in configuration.qubits
        }

    def emit_phase_gate(self, gate) -> 'Program':  # type: ignore  # noqa: F821
        """Replace a Rz/VirtualZ gate with its readout-randomized ref.

        Records the gate's original phase into `source_phases` and
        advances `angle_index` for `gate`'s qubit.

        Args:
            gate: qcal Rz or VirtualZ gate.

        Returns:
            Program: single-instruction PyQuil Program.
        """
        from pyquil.gates import RZ
        from pyquil.quilatom import MemoryReference

        q = gate.qubits[0]
        phase = gate.properties['params']['phase']
        self.source_phases[q][self.angle_index[q]] = phase / (2 * np.pi)

        ref = MemoryReference(
            self.configuration.destination_names(q), self.angle_index[q]
        )
        self.angle_index[q] += 1
        return RZ(2 * np.pi * ref, q)


def _minimal_period(layers: Sequence[object]) -> int:
    """Length of the smallest unit that exactly tiles `layers`.

    Used to auto-detect `base_cycle_repetitions` when the caller
    doesn't specify one: the returned period, and
    `len(layers) // period` repetitions of it, always reconstruct
    `layers` exactly.

    Args:
        layers (Sequence[object]): a circuit's full per-cycle RC-layer
            sequence (see `build_rc_configuration`).

    Returns:
        int: smallest `p` dividing `len(layers)` such that
            `layers == layers[:p] * (len(layers) // p)`. Equals
            `len(layers)` if no smaller unit tiles it exactly (or if
            `layers` is empty).
    """
    n = len(layers)
    if n == 0:
        return 0
    # Any p < n dividing n satisfies p <= n // 2, so checking beyond that
    # can only find p == n itself, which trivially tiles -- skip straight
    # to the fallback below instead of paying an O(n) pass to confirm it.
    for p in range(1, n // 2 + 1):
        if n % p:
            continue
        if all(layers[i] == layers[i % p] for i in range(n)):
            return p
    return n


@final
@dataclass(frozen=True, kw_only=True)
class _U2Randomization:
    qubits: Sequence[int]
    destination_names: Callable[[int], str]
    seed_names: Callable[[int], str]
    source_name: str
    source_unitary_phases: list[float]

    @property
    def source_unitary_count(self) -> int:
        return len(self.source_unitary_phases) // _ANGLES_PER_UNITARY

    def build_quil_program(self) -> 'Program': # type: ignore  # noqa: F821
        try:
            from pyquil import Program
            from pyquil.quilbase import Call, Declare
            from quil import instructions as inst
        except ImportError:
            logger.warning(' Unable to import pyquil!')
            return

        program = Program()
        program += Declare(
            self.source_name, "REAL", len(self.source_unitary_phases)
        )
        for qubit in self.qubits:
            program += Declare(
                self.destination_names(qubit), "REAL", _ANGLES_PER_UNITARY
            )
            program += Declare(self.seed_names(qubit), "INTEGER", 1)
        for qubit in self.qubits:
            program += Call(
                "choose_random_real_sub_regions",
                [
                    inst.CallArgument.from_identifier(
                        self.destination_names(qubit)
                    ),
                    inst.CallArgument.from_identifier(self.source_name),
                    inst.CallArgument.from_immediate(
                        complex(_ANGLES_PER_UNITARY)),
                    inst.CallArgument.from_identifier(
                        self.seed_names(qubit)
                    ),
                ],
            )
        return program

    def generate_seeds(self, rng: np.random.Generator) -> Mapping[int, int]:
        # return {
        #     qubit: rng.integers(
        #         -(2**47), 2**47 - 1, dtype=np.int64
        #     ).item() for qubit in self.qubits
        # }
        return {
            qubit: rng.integers(
                0, 2**47 - 1, dtype=np.int64
            ).item() for qubit in self.qubits
        }

    def build_memory_map(
            self, seeds: Mapping[int, int]
    ) -> Mapping[str, list[float] | list[int]]:
        memory_map = {self.source_name: self.source_unitary_phases}
        for qubit in self.qubits:
            memory_map[self.destination_names(qubit)] = [0.0, 0.5, 0.5]
            memory_map[self.seed_names(qubit)] = [seeds[qubit]]
        return memory_map

    def generate_indices(
            self, shot_count: int, seeds: Mapping[int, int]
    ) -> NDArray[np.int8]:
        from qcs_sdk.qpu.experimental.random import (
            PrngSeedValue,
            choose_random_real_sub_region_indices,
        )

        indices = np.zeros((shot_count, len(self.qubits)), dtype=np.int8)
        for qubit_index, qubit in enumerate(self.qubits):
            sequence = choose_random_real_sub_region_indices(
                PrngSeedValue(seeds[qubit]),
                start_index=0,
                series_length=shot_count,
                sub_region_count=self.source_unitary_count,
            )
            indices[:, qubit_index] = np.asarray(sequence, dtype=np.int8)
        return indices
