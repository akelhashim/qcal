"""Helper functions for True-Q compilation.

NOTE: we do not use TYPE_CHECKING for trueq types because this might fail if
trueq is not installed when building docs.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def set_cycles(
        circuit: trueq.Circuit, list_of_cycles: List[trueq.Cycle]  # noqa: F821 # type: ignore
) -> trueq.Circuit:  # noqa: F821 # type: ignore
    """Function that changes the cycles of a circuit.

    Args:
        circuit (trueq.Circuit): True-Q circuit.
        list_of_cycles (list[trueq.Cycle]): List of new True-Q cycles.

    Returns:
        trueq.Circuit: same True-Q circuit but with different cycles.
    """
    try:
        import trueq as tq
    except ImportError:
        logger.warning(' Unable to import trueq!')

    circuit_dict = circuit.to_dict()
    circuit_dict['cycles'] = tq.Circuit(list_of_cycles).to_dict()['cycles']

    return tq.Circuit.from_dict(circuit_dict)


def set_marker(cycle: trueq.Cycle, marker: int) -> trueq.Cycle:  # noqa: F821 # type: ignore
    """Function that sets the marker of a cycle.

    Args:
        cycle (trueq.Cycle): True-Q cycle.
        marker (int): cycle marker label.

    Returns:
        trueq.Cycle: True-Q cycle with a modified marker.
    """
    try:
        import trueq as tq
    except ImportError:
        logger.warning(' Unable to import trueq!')

    dict_cycle = cycle.to_dict()
    dict_cycle['marker']= marker

    return tq.Cycle.from_dict(dict_cycle)


def serialize_cycles(circuit: trueq.Circuit) -> trueq.Circuit:  # noqa: F821 # type: ignore
    """Function to serialize all single-qubit and two-qubit gate cycles.

    This function separates all single-qubit gates from two-qubit gates that
    occur simultaneously within a given cycle, and then merges the single-qubit
    gates into the previous cycle.

    Args:
        circuit (trueq.Circuit): True-Q circuit.

    Returns:
        trueq.Circuit: same True-Q circuit but with serialized cycles.
    """
    try:
        import trueq as tq
    except ImportError:
        logger.warning(' Unable to import trueq!')

    # To keep track of original cycle markers
    markers = []
    for cycle in circuit:
        markers.append(cycle.marker)

    # Unmark to allow for Merging
    circuit = tq.compilation.UnmarkCycles().apply(circuit)

    list_of_cycles=[]
    for i, cycle in enumerate(circuit):
        if cycle.gates_multi:  # If the cycle contains multi qubit gates,
            # Remove previous cycle from list, and merge it with 1q cycle from
            # the 2q cycle
            list_of_cycles.pop(-1)
            list_of_cycles.append(
                tq.compilation.Merge().apply(
                    [circuit[i-1], tq.Cycle(cycle.gates_single)]
                )[0]
            )
            # Follow with entangling cycle
            list_of_cycles.append(tq.Cycle(cycle.gates_multi))
        else:
            list_of_cycles.append(cycle)
    circuit = set_cycles(circuit, list_of_cycles)

    # Re-mark the cycles using the original pattern
    list_of_cycles = []
    for i, cycle in enumerate(circuit):
        # Mark all cycles using the original markers
        list_of_cycles.append(set_marker(cycle, markers[i]))
    circuit = set_cycles(circuit, list_of_cycles)

    return circuit


def X90_phases(
        qubits: Iterable[int], compiler: trueq.Compiler, in_radians: bool = True  # noqa: F821 # type: ignore
    ) -> Dict[int, List[float]]:
    """Extracts the phases of a decomposed X90.

    Args:
        qubits (Iterable[int]): qubit labels
        compiler (trueq.Compiler): True-Q compiler
        in_radians (bool, optional): phases in radians. Defaults to True.

    Returns:
        Dict[int, List[float]]: list of phases for each qubit.
    """
    try:
        import trueq as tq
    except ImportError:
        logger.warning(' Unable to import trueq!')

    phases = {q: [] for q in qubits}
    compiled_X90 = compiler.compile(
        tq.Circuit([tq.Cycle({(q,): tq.Gate.sx for q in qubits})])
    )
    for i in range(0, compiled_X90.n_cycles, 2):
        for q in qubits:
            phases[q].append(
                np.deg2rad(compiled_X90[i][q].parameters['phi']) if in_radians
                else compiled_X90[i][q].parameters['phi']
            )

    return phases
