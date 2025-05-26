""""Submodule for generating RPE circuits.

"""
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def interleaved_circuit_depths(circuit_depths: List[int]) -> List[int]:
    """Circuit depths for interleaved X90 sequences.

    Args:
        circuit_depths (List[int]): circuit depths.

    Returns:
        List[int]: circuit depths with 1/4 the maximum depth.
    """
    max_depth = int(max(circuit_depths) / 4)
    idx = circuit_depths.index(max_depth)
    return circuit_depths[:idx+1]


def make_idle_circuits(
        circuit_depths: List[int],
        qubits:         Tuple[int],
        gate_layer:     List = None,
    ) -> List:
    """Generate the idle RPE circuits.

    Args:
        circuit_depths (List[int]): circuit depths.
        qubits (Tuple[int]): qubit labels.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
        List: pyGSTi circuits.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    circuits = (
        [make_idle_cos_circ(d, qubits, gate_layer) for d in circuit_depths] +
        [make_idle_sin_circ(d, qubits, gate_layer) for d in circuit_depths]
    )

    return pygsti.remove_duplicates(circuits)


def make_idle_cos_circ(
        d: int, qubits: Tuple[int] | List[int], gate_layer: List = None,
    ):
    """Make the cosine circuit for idle RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    Gi_prep = pygsti.circuits.Circuit(
        [[('Gypi2', q) for q in qubits]], line_labels=qubits
    )
    Gi_germ = pygsti.circuits.Circuit(
        gate_layer if gate_layer else [[('Gidle', q) for q in qubits]], 
        line_labels=qubits
    ) * d
    Gi_meas = pygsti.circuits.Circuit(
        [[('Gypi2', q) for q in qubits]], line_labels=qubits
    ) * 3

    return Gi_prep + Gi_germ + Gi_meas


def make_idle_sin_circ(
        d: int, qubits: Tuple[int] | List[int], gate_layer: List = None,
    ):
    """Make the cosine circuit for idle RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    Gi_prep = pygsti.circuits.Circuit(
        [[('Gxpi2', q) for q in qubits]], line_labels=qubits
    )
    Gi_germ = pygsti.circuits.Circuit(
        gate_layer if gate_layer else [[('Gidle', q) for q in qubits]], 
        line_labels=qubits
    ) * d
    Gi_meas = pygsti.circuits.Circuit(
        [[('Gypi2', q) for q in qubits]], line_labels=qubits
    ) * 3

    return Gi_prep + Gi_germ + Gi_meas


def make_x90_circuits(
        circuit_depths: List[int],
        qubits:         Tuple[int],
        gate_layer:     List = None,
    ) -> List:
    """Generate the axis X90 RPE circuits.

    Args:
        circuit_depths (List[int]): circuit depths.
        qubits (Tuple[int]): qubit labels.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
        List: list of pyGSTi circuits.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    circuits = (
        [make_x90_cos_circ(d, qubits, gate_layer) for d in circuit_depths] +
        [make_x90_sin_circ(d, qubits, gate_layer) for d in circuit_depths] +
        [make_X90_icos_circ(d, qubits, gate_layer) for d in circuit_depths] +
        [make_X90_isin_circ(d, qubits, gate_layer) for d in circuit_depths]
    )

    return pygsti.remove_duplicates(circuits)


def make_x90_cos_circ(
        d: int, qubits: Tuple[int] | List[int], gate_layer: List = None,
    ):
    """Make the cosine circuit for X90 RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')
    
    return pygsti.circuits.Circuit(
            gate_layer if gate_layer else [[('Gxpi2', q) for q in qubits]], 
            line_labels=qubits
        ) * d


def make_x90_sin_circ(
        d: int, qubits: Tuple[int] | List[int], gate_layer: List = None,
    ):
    """Make the sine circuit for X90 RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')
    
    return pygsti.circuits.Circuit(
            gate_layer if gate_layer else [[('Gxpi2', q) for q in qubits]], 
            line_labels=qubits
        ) * (d + 1)


def make_X90_icos_circ(
        d: int, qubits: Tuple[int] | List[int], gate_layer: List = None,
    ):
    """Make the interleaved cosine circuit for X90 axis error RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    Gz_layer = pygsti.circuits.Circuit(
        [[('Gzpi2', q) for q in qubits]], line_labels=qubits
    )
    Gx_layer = pygsti.circuits.Circuit(
        gate_layer if gate_layer else [[('Gxpi2', q) for q in qubits]], 
        line_labels=qubits
    )

    return (
        Gz_layer + Gx_layer + Gx_layer + Gz_layer + Gz_layer + Gx_layer + 
        Gx_layer + Gz_layer
    ) * d


def make_X90_isin_circ(
        d: int, qubits: Tuple[int] | List[int], gate_layer: List = None,
    ):
    """Make the interleaved sine circuit for X90 axis error RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    Gz_layer = pygsti.circuits.Circuit(
        [[('Gzpi2', q) for q in qubits]], line_labels=qubits
    )
    Gx_layer = pygsti.circuits.Circuit(
        gate_layer if gate_layer else [[('Gxpi2', q) for q in qubits]], 
        line_labels=qubits
    )

    return (
        Gz_layer + Gx_layer + Gx_layer + Gz_layer + Gz_layer + Gx_layer + 
        Gx_layer + Gz_layer
    ) * d + Gx_layer


def make_cz_circuits(
        circuit_depths: List[int], 
        qubit_pairs:    List[Tuple[int]], 
        gate_layer:     List = None,
    ) -> List:
    """Generate CZ RPE circuits.

    Args:
        circuit_depths (List[int]): circuit depths.
        qubit_pairs (List[Tuple[int]]): pairs of qubits for two-qubit gates.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
        List: list of pyGSTi circuits.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    state_pairs = [(0, 1), (2, 3), (3, 1)]
    sin_dict = {
        state_pair: {
            i: make_cz_sin_circ(
                i, state_pair, qubit_pairs, gate_layer
            ) for i in circuit_depths
        } for state_pair in state_pairs
    }
    cos_dict = {
        state_pair: {
            i: make_cz_cos_circ(
                i, state_pair, qubit_pairs, gate_layer
            ) for i in circuit_depths
        } for state_pair in state_pairs
    }

    circuits = []
    for trig_dict in [sin_dict, cos_dict]:
        for state_pair in state_pairs:
            circuits += list(trig_dict[state_pair].values())
    
    return pygsti.remove_duplicates(circuits)


def make_cz_cos_circ(
        d:           int, 
        state_pair:  Tuple[int], 
        qubit_pairs: List[Tuple[int]], 
        gate_layer:  List = None,
    ):
    """Make the cosine circuit for CZ RPE.

    Args:
        d (int): circuit depth.
        state_pair (Tuple[int]): state pair.
        qubit_pairs (List[Tuple[int]]): pairs of qubits for two-qubit gates.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    line_labels = []
    for qubit_pair in qubit_pairs:
        line_labels.extend(qubit_pair)

    # <01| for 1/2 (1+cos(1/2 (theta_iz + theta_zz))
    # <00| for 1/2 (1-cos(1/2 (theta_iz + theta_zz))
    if state_pair in [(0, 1), (1, 0)]:
       circ = (
           pygsti.circuits.Circuit(
               [[('Gypi2', qp[1]) for qp in qubit_pairs]], 
               line_labels=line_labels
           ) +
           pygsti.circuits.Circuit(
               gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
           ) * d +
           pygsti.circuits.Circuit(
               [[('Gypi2', qp[1]) for qp in qubit_pairs]]
           )
       )
    
    # <11| for 1/2 (1+cos(1/2 (theta_iz - theta_zz))
    elif state_pair in [(2, 3), (3, 2)]:
        circ = (
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]], 
                line_labels=line_labels
            ) +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[1]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
            ) * d +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[1]) for qp in qubit_pairs]]
            )
        )
    
    # <11| for 1/2 (1+cos(1/2 (theta_zi - theta_zz))
    elif state_pair in [(1, 3), (3, 1)]:
        circ = (
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]], 
                line_labels=line_labels
            ) +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[0]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
            ) * d +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[0]) for qp in qubit_pairs]]
            )
        )
    
    else:
        assert False, (
             "state_pair must be in [(0,1), (1,0), (2,3), (3,2), (1,3), (3,1)]"
        )

    return circ

    
def make_cz_sin_circ(
        d:           int, 
        state_pair:  Tuple[int], 
        qubit_pairs: List[Tuple[int]], 
        gate_layer:  List = None,
    ):
    """Make the sine circuit for CZ RPE.

    Args:
        d (int): circuit depth.
        state_pair (Tuple[int]): state pair.
        qubit_pairs (List[Tuple[int]]): pairs of qubits for two-qubit gates.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')
    
    line_labels = []
    for qubit_pair in qubit_pairs:
        line_labels.extend(qubit_pair)

    # <00| for 1/2 (1+sin(1/2 (theta_iz + theta_zz))
    if state_pair in [(0, 1), (1, 0)]:
        circ = (
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[1]) for qp in qubit_pairs]], 
                line_labels=line_labels
            ) +
            pygsti.circuits.Circuit(
                gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
            ) * d +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            )
        ) 
    
    # <10| for 1/2 (1+sin(1/2 (theta_iz - theta_zz))
    elif state_pair in [(2, 3),(3, 2)]:
        circ = (
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]], 
                line_labels=line_labels
            ) +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[1]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
            ) * d +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            )
        )

    # <01| for 1/2 (1+sin(1/2 (theta_zi - theta_zz))    
    elif state_pair in [(1, 3), (3, 1)]:
        circ = (
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]], 
                line_labels=line_labels
            ) +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[0]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
            ) * d +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]]
            )
        )
    else:
        assert False, (
            "state_pair must be in [(0,1), (1,0), (2,3), (3,2), (1,3), (3,1)]"
        )
    return circ
