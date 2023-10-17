"""Submodule for handling dynamical decoupling sequences.

"""
from qcal.calibration.utils import find_pulse_index
from qcal.circuit import Barrier, Circuit, Cycle
from qcal.config import Config
from qcal.gate.single_qubit import Idle, X, Rz

import logging
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Any, List

logger = logging.getLogger(__name__)


__all__ = ('XY')


def XY(
        config: Config, qubit: int, length: float, n_pulses: int = 4
    ) -> Circuit:
    """Subcircuit for performing an XY-type dynamical decoupling (DD) sequence.

    See:
    https://aws.amazon.com/blogs/quantum-computing/suppressing-errors-with-
    dynamical-decoupling-using-pulse-control-on-amazon-braket/

    Args:
        config (Config): qcal Config object.
        qubit (int): qubit label.
        length (float): length of time over which to perform the DD.
        n_pulses (int, optional): number of pulses. Defaults to 4. For example,
            for n_pulses = 4, this performs an XY4 DD sequence.

    Returns:
        Circuit: qcal Circuit.
    """
    assert n_pulses % 4 == 0, "'n_pulses' must be a multiple of 4!"
    tau = length / (2 * n_pulses)  # Base interval
    idx = find_pulse_index(config, f'single_qubit/{qubit}/GE/X/pulse')
    gate_time = config[f'single_qubit/{qubit}/GE/X/pulse'][idx]['length']
    idle_time = tau - gate_time / 2

    DD_sequence = Circuit([
        # Barrier((qubit,)),
        # Idle
        Cycle({Idle(qubit, duration=idle_time)}),
        # X
        Cycle({X(qubit)}),
        # Idle
        Cycle({Idle(qubit, duration=2 * idle_time)}),
        # Y
        Cycle({Rz(np.pi/2, qubit)}),
        Cycle({X(qubit)}),
        Cycle({Rz(-np.pi/2, qubit)}),
        # Idle
        Cycle({Idle(qubit, duration=2 * idle_time)}),
        # X
        Cycle({X(qubit)}),
        # Idle
        Cycle({Idle(qubit, duration=2 * idle_time)}),
        # Y
        Cycle({Rz(np.pi/2, qubit)}),
        Cycle({X(qubit)}),
        Cycle({Rz(-np.pi/2, qubit)}),
        # Idle
        Cycle({Idle(qubit, duration=idle_time)}),
        # Barrier((qubit,)),
    ])
    
    DD_circuit = Circuit()
    for _ in range(int(n_pulses / 4)):
        DD_circuit.extend(DD_sequence)

    return DD_circuit


dd_sequences = defaultdict(lambda: 'DD sequence not currently supported!', {
    'XY': XY,
})
