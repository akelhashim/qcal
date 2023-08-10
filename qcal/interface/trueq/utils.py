"""Helper functions for True-Q compilation.

"""
import logging

logger = logging.getLogger(__name__)


def set_cycles(circuit, list_of_cycles):
    """Function that changes the cycles of a circuit.

    Args:
        circuit (tq.Circuit): True-Q circuit.
        list_of_cycles (list[tq.Cycle]): List of new True-Q cycles.

    Returns:
        tq.Circuit: same True-Q circuit but with different cycles.
    """
    import trueq as tq

    circuit_dict = circuit.to_dict()
    circuit_dict['cycles'] = tq.Circuit(list_of_cycles).to_dict()['cycles']

    return tq.Circuit.from_dict(circuit_dict)


def set_marker(cycle, marker):
    """Function that sets the marker of a cycle.

    Args:
        cycle (tq.Cycle): True-Q cycle.
        marker (int): cycle marker label.

    Returns:
        tq.Cycle: True-Q cycle with a modified marker.
    """
    import trueq as tq

    dict_cycle = cycle.to_dict()
    dict_cycle['marker']= marker

    return tq.Cycle.from_dict(dict_cycle)


def serialize_cycles(circuit):
    """Function to serialize all single-qubit and two-qubit gate cycles.

    This function separates all single-qubit gates from two-qubit gates that
    occur simultaneously within a given cycle, and then merges the single-qubit 
    gates into the previous cycle.

    Args:
        circuit (tq.Circuit): True-Q circuit.

    Returns:
        tq.Circuit: same True-Q circuit but with serialized cycles.
    """
    import trueq as tq

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

    # Remark the cycles using the original pattern
    list_of_cycles = []
    for i, cycle in enumerate(circuit): 
        # Mark all cycles using the original markers
        list_of_cycles.append(set_marker(cycle, markers[i]))
    circuit = set_cycles(circuit, list_of_cycles)
            
    return circuit