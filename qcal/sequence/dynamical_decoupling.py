"""Submodule for handling dynamical decoupling sequences.

"""
from qcal.calibration.utils import find_pulse_index
from qcal.circuit import Barrier, Circuit, Cycle
from qcal.config import Config
from qcal.gate.single_qubit import Idle, X90, Y90, Rz

import logging

from collections import defaultdict
from collections.abc import Iterable
# from typing import Any, List

logger = logging.getLogger(__name__)


__all__ = ('XY', 'dd_sequences')


def XY(
        config:   Config, 
        qubits:   Iterable[int], 
        length:   float, 
        n_pulses: int | None = None,
        phase:    float | None = None,
        subspace: str = 'GE',
    ) -> Circuit:
    """Subcircuit for performing an XY-type dynamical decoupling (DD) sequence.

    See:
    https://aws.amazon.com/blogs/quantum-computing/suppressing-errors-with-
    dynamical-decoupling-using-pulse-control-on-amazon-braket/

    Args:
        config (Config): qcal Config object.
        qubits (Iterable[int]): qubit labels.
        length (float): length of time over which to perform the DD.
        n_pulses (int | None, optional): number of pulses. Defaults to 4. For 
            example, for n_pulses = 4, this performs an XY4 DD sequence.
        phase (float | None, optional): optional (virtual) phase to add to the
            sequence for each XY4 duration.
        subspace (str, optional): qubits subsace for the DD sequence. Defaults 
            'GE'. 'EF' is also supported.

    Returns:
        Circuit: qcal Circuit.
    """
    idx = find_pulse_index(
        config, f'single_qubit/{qubits[0]}/{subspace}/X90/pulse'
    )
    gate_time = config[
        f'single_qubit/{qubits[0]}/{subspace}/X90/pulse/{idx}/length'
    ]

    if n_pulses is None:
        n_pulses = max(4, int(length / (gate_time * 8)) * 4)

    assert n_pulses % 4 == 0, "'n_pulses' must be a multiple of 4!"
    
    tau = length / (2 * n_pulses)  # Base interval
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


dd_sequences = defaultdict(lambda: 'DD sequence not currently supported!', {
    'XY': XY,
})
