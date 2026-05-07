"""Submodule for handling dynamical decoupling sequences.

"""
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

from qcal.circuit import Circuit, Cycle
from qcal.gate.single_qubit import X90, Y90, Idle, Rz

# from typing import Any, List

logger = logging.getLogger(__name__)


__all__ = ('XY', 'DD_SEQUENCES')


def XY(
    qubits:     Sequence[int],
    total_time: float,
    gate_time:  float,
    n_pulses:   int | None = None,
    phase:      float | None = None,
    subspace:   str = 'GE',
) -> Circuit:
    """Subcircuit for performing an XY-type dynamical decoupling (DD) sequence.

    See:
    https://aws.amazon.com/blogs/quantum-computing/suppressing-errors-with-
    dynamical-decoupling-using-pulse-control-on-amazon-braket/

    Args:
        qubits (Sequence[int]): qubit labels.
        total_time (float): total time over which to perform the DD.
        gate_time (float): time of the native X90 gate.
        n_pulses (int | None, optional): number of pulses. Defaults to None. For
            example, for n_pulses = 4, this performs an XY4 DD sequence. If
            None, the number of pulses is set to the maximum number of pulses
            that can fit within the specified time, with a minimum of 4.
        phase (float | None, optional): optional (virtual) phase to add to the
            sequence for each XY4 duration.
        subspace (str, optional): qubits subspace for the DD sequence. Defaults
            'GE'. 'EF' is also supported.

    Returns:
        Circuit: qcal Circuit.
    """
    if n_pulses is None:
        n_pulses = max(4, int(total_time / (gate_time * 8)) * 4)
    else:
        if n_pulses % 4 != 0:
            raise ValueError("'n_pulses' must be a multiple of 4!")

    tau = total_time / (2 * n_pulses)  # Base interval
    idle_time = tau - gate_time
    idle_time = idle_time if idle_time > 0 else 0.

    if phase:
        DD_sequence = Circuit([
            # Idle + phase
            Cycle({Idle(q, duration=idle_time) for q in qubits}),
            Cycle({
                Rz(q, phase/(2*n_pulses), subspace=subspace) for q in qubits
            }),
            # X
            Cycle({X90(q, subspace=subspace) for q in qubits}),
            Cycle({X90(q, subspace=subspace) for q in qubits}),
            # Idle + phase
            Cycle({
                Rz(q, phase/(2*n_pulses), subspace=subspace) for q in qubits
            }),
            Cycle({Idle(q, duration=2 * idle_time) for q in qubits}),
            Cycle({
                Rz(q, phase/(2*n_pulses), subspace=subspace) for q in qubits
            }),
            # Y
            Cycle({Y90(q, subspace=subspace) for q in qubits}),
            Cycle({Y90(q, subspace=subspace) for q in qubits}),
            # Idle + phase
            Cycle({
                Rz(q, phase/(2*n_pulses), subspace=subspace) for q in qubits
            }),
            Cycle({Idle(q, duration=2 * idle_time) for q in qubits}),
            Cycle({
                Rz(q, phase/(2*n_pulses), subspace=subspace) for q in qubits
            }),
            # X
            Cycle({X90(q, subspace=subspace) for q in qubits}),
            Cycle({X90(q, subspace=subspace) for q in qubits}),
            # Idle + phase
            Cycle({
                Rz(q, phase/(2*n_pulses), subspace=subspace) for q in qubits
            }),
            Cycle({Idle(q, duration=2 * idle_time) for q in qubits}),
            Cycle({
                Rz(q, phase/(2*n_pulses), subspace=subspace) for q in qubits
            }),
            # Y
            Cycle({Y90(q, subspace=subspace) for q in qubits}),
            Cycle({Y90(q, subspace=subspace) for q in qubits}),
            # Idle + phase
            Cycle({
                Rz(q, phase/(2*n_pulses), subspace=subspace) for q in qubits
            }),
            Cycle({Idle(q, duration=idle_time) for q in qubits}),
            # Barrier((q for q in qubits))
        ])

    else:
        DD_sequence = Circuit([
            # Idle
            Cycle({Idle(q, duration=idle_time) for q in qubits}),
            # X
            Cycle({X90(q) for q in qubits}),
            Cycle({X90(q) for q in qubits}),
            # Idle
            Cycle({Idle(q, duration=2 * idle_time) for q in qubits}),
            # Y
            Cycle({Y90(q) for q in qubits}),
            Cycle({Y90(q) for q in qubits}),
            # Idle
            Cycle({Idle(q, duration=2 * idle_time) for q in qubits}),
            # X
            Cycle({X90(q) for q in qubits}),
            Cycle({X90(q) for q in qubits}),
            # Idle
            Cycle({Idle(q, duration=2 * idle_time) for q in qubits}),
            # Y
            Cycle({Y90(q) for q in qubits}),
            Cycle({Y90(q) for q in qubits}),
            # Idle
            Cycle({Idle(q, duration=idle_time) for q in qubits}),
            # Barrier((q for q in qubits))
        ])

    DD_circuit = Circuit()
    for _ in range(int(n_pulses / 4)):
        DD_circuit.extend(DD_sequence.copy())

    return DD_circuit


DD_SEQUENCES: Mapping[str, Callable] = defaultdict(
    lambda: 'DD sequence not currently supported!', {
        'XY': XY,
    }
)
