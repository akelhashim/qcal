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
from typing import Dict, List, Sequence, Tuple

import numpy as np

from qcal.circuit import Circuit, Cycle

logger = logging.getLogger(__name__)


__all__ = (
    'PHASE_GATES',
    'PULSE_GATES',
    'cycle_to_base_cycle',
    'build_rc_configuration',
    'RCLayerTracker',
)


PHASE_GATES = ('Rz', 'VirtualZ')
PULSE_GATES = ('X90', 'SX')


def cycle_to_base_cycle(
    cycle: Cycle, qubits: Sequence[int]
) -> Tuple[Tuple[int, int] | int, ...]:
    """Convert a qcal Cycle into a PyQuil RC "base cycle" entry.

    Two-qubit gates become edges ``(q0, q1)``. Every other qubit in
    `qubits` becomes a bare int -- this includes qubits truly idle at
    this position *and* qubits carrying single-qubit content, since
    ``pyquil._qpu.randomized_compiling`` only distinguishes "edge" from
    "identity" for Pauli-propagation bookkeeping; the actual single-qubit
    content is captured separately as the source-unitary angles (the
    circuit's own Rz phases, tracked by `RCLayerTracker`).

    Args:
        cycle (Cycle): qcal Cycle containing at least one two-qubit gate.
        qubits (Sequence[int]): full qubit register to twirl.

    Returns:
        Tuple[Tuple[int, int] | int, ...]: e.g. ``((0, 1), 2)``.

    Raises:
        ValueError: if a gate in `cycle` acts on more than two qubits.
    """
    entries: List[Tuple[int, int] | int] = []
    edge_qubits = set()
    for gate in cycle.gates:
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
    **rc_kwargs,
):  # -> pyquil._qpu.randomized_compiling.RandomizedCompilingConfiguration
    """Build a RandomizedCompilingConfiguration from a compiled Circuit.

    `circuit` is the fully unrolled sequence to run (not a repeated
    template), so every two-qubit-gate Cycle becomes its own entry in
    `base_cycles` and `base_cycle_repetitions` is fixed at 1.

    Args:
        circuit (Circuit): compiled qcal circuit to twirl.
        qubits (Sequence[int]): full qubit register to twirl.
        **rc_kwargs: forwarded to `RandomizedCompilingConfiguration`
            (e.g. `invert_random_paulis`, `shots_per_randomization`).

    Returns:
        RandomizedCompilingConfiguration: PyQuil RC configuration.
    """
    try:
        from pyquil._qpu import randomized_compiling as rc
    except ImportError:
        logger.warning(' Unable to import pyquil!')
        return

    base_cycles = tuple(
        cycle_to_base_cycle(cycle, qubits)
        for cycle in circuit.cycles
        if not cycle.is_barrier and
        any(len(gate.qubits) == 2 for gate in cycle.gates)
    )
    return rc.RandomizedCompilingConfiguration(
        base_cycles=base_cycles,
        base_cycle_repetitions=1,
        **rc_kwargs,
    )


class RCLayerTracker:
    """Tracks (layer_index, angle_index) state for randomized compiling.

    `qcal.interface.pyquil.transpiler.transpile_circuit` owns the actual
    circuit/cycle walk; it calls into this tracker only for the pieces
    specific to randomized compiling -- swapping a Rz/VirtualZ gate's
    literal phase for a memory-reference-parametrized one (recording the
    original phase into `source_unitaries`), fencing X90/SX pulses, and
    advancing to the next layer once a Cycle's two-qubit gates have been
    emitted.

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
        self.source_unitaries: Dict[str, List[float]] = {
            configuration.variables.source_unitaries(q): (
                [0.0] * (n_layers * 3)
            ) for q in configuration.qubits_sorted
        }

    def emit_phase_gate(self, gate) -> 'Program':  # noqa: F821
        """Replace a Rz/VirtualZ gate with its twirled memory reference.

        Records the gate's original phase into `source_unitaries` at
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
        self.source_unitaries[key][
            self.layer_index * 3 + self.angle_index[q]
        ] = phase / (2 * np.pi)

        ref = self.configuration.variables.twirled_unitaries_ref(
            q, self.layer_index, self.angle_index[q]
        )
        self.angle_index[q] += 1
        return RZ(2 * np.pi * ref, q)

    def emit_pulse_gate(self, gate) -> 'Program':  # noqa: F821
        """Re-emit a fixed X90/SX pulse, fenced per the ZXZXZ template.

        Args:
            gate: qcal X90 or SX gate.

        Returns:
            Program: two-instruction PyQuil Program (RX, FENCE).
        """
        from pyquil.gates import FENCE, RX
        from pyquil.quil import Program

        q = gate.qubits[0]
        return Program(RX(np.pi / 2, q), FENCE(q))

    def close_layer(self) -> None:
        """Advance to the next layer, resetting per-qubit angle_index."""
        self.layer_index += 1
        self.angle_index = dict.fromkeys(
            self.configuration.qubits_sorted, 0
        )
