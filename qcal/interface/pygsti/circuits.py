"""Submodule for handling pyGSTi circuits.

"""
from qcal.circuit import CircuitSet

from typing import List

import logging

logger = logging.getLogger(__name__)


__all__ = ('load_circuits',)


def load_circuits(circuits: List | str) -> CircuitSet:
    """Load pyGSTi circuits from a file.

    Args:
        circuits (List | str): list of pyGSTi circuits. This can also be a text
            file containing all of the circuits.

    Returns:
        CircuitSet: set of pyGSTi circuits.
    """
    from pygsti.io import load_circuit_list

    if isinstance(circuits, str):
        circuits = load_circuit_list(circuits)
        circuit_list = [circ.str for circ in circuits]
    elif isinstance(circuits, list):
        circuit_list = [circ.str for circ in circuits]

    cs = CircuitSet(circuits=circuits)
    cs['pygsti_circuit'] = circuit_list

    return cs

