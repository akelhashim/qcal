""""Submodule for generating RPE circuits.

"""
import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pygsti import remove_duplicates
from pygsti.circuits import Circuit as PyGSTiCircuit

from qcal.gate import gate
from qcal.utils import flatten

GateLayer = Optional[List[List[tuple]]]

logger = logging.getLogger(__name__)


# Centralizer of ZZ in the 2-qubit Pauli group: all P⊗Q that commute with Z⊗Z.
# Commutation holds when both qubits commute with Z, or both anticommute with Z:
#
#   Operator | Qubit 0 vs Z | Qubit 1 vs Z
#   ---------+--------------+--------------
#   II       | commutes     | commutes
#   IZ       | commutes     | commutes
#   ZI       | commutes     | commutes
#   ZZ       | commutes     | commutes
#   XX       | anticommutes | anticommutes
#   XY       | anticommutes | anticommutes
#   YX       | anticommutes | anticommutes
#   YY       | anticommutes | anticommutes
ZZ_CENTRALIZER = frozenset({
    'XX', 'YY', 'ZZ', 'XY', 'YX', 'IZ', 'ZI', 'II'
})
_ZZ_CENTRALIZER_SORTED = sorted(ZZ_CENTRALIZER)
_N_CENTRALIZERS: int = len(ZZ_CENTRALIZER)  # 8
_PAULI_GATE = {'X': 'Gxpi2', 'Y': 'Gypi2', 'Z': 'Gzpi2', 'I': 'Gi'}
# Single-qubit Pauli multiplication, modulo global phase (phases cancel in RPE).
_PAULI_PRODUCT = {
    ('I', 'I'): 'I', ('I', 'X'): 'X', ('I', 'Y'): 'Y', ('I', 'Z'): 'Z',
    ('X', 'I'): 'X', ('X', 'X'): 'I', ('X', 'Y'): 'Z', ('X', 'Z'): 'Y',
    ('Y', 'I'): 'Y', ('Y', 'X'): 'Z', ('Y', 'Y'): 'I', ('Y', 'Z'): 'X',
    ('Z', 'I'): 'Z', ('Z', 'X'): 'Y', ('Z', 'Y'): 'X', ('Z', 'Z'): 'I',
}


def _pauli_product_layer(
    paulis_a: List[str],
    paulis_b: List[str],
    qubit_pairs: Sequence[Tuple[int, int]],
) -> Optional[List[tuple]]:
    """Gate layer for the element-wise Pauli product P_a · P_b.

    Args:
        paulis_a (List[str]): Pauli strings for the first operand, one per
            qubit pair (e.g. ``['XX', 'YZ']``).
        paulis_b (List[str]): Pauli strings for the second operand, one per
            qubit pair.
        qubit_pairs (Sequence[Tuple[int, int]]): qubit pair labels; must
            have the same length as ``paulis_a`` and ``paulis_b``.

    Returns:
        Optional[List[tuple]]: a flat gate layer of ``(gate_name, qubit)``
            tuples representing P_a · P_b, or ``None`` if the product is
            the identity on every qubit.
    """
    result = [
        (_PAULI_GATE[_PAULI_PRODUCT[(a[qi], b[qi])]], qp[qi])
        for a, b, qp in zip(paulis_a, paulis_b, qubit_pairs, strict=True)
        for qi in (0, 1)
    ]
    return None if all(g == 'Gi' for g, _ in result) else result


def interleaved_circuit_depths(circuit_depths: Sequence[int]) -> List[int]:
    """Circuit depths for interleaved X90 sequences.

    Args:
        circuit_depths (Sequence[int]): circuit depths.

    Returns:
        List[int]: circuit depths with 1/4 the maximum depth.
    """
    max_depth = int(max(circuit_depths) / 4)
    idx = circuit_depths.index(max_depth)
    return circuit_depths[:idx+1]


def make_idle_circuits(
    circuit_depths: Sequence[int],
    qubits:         Sequence[int],
    gate_layer:     GateLayer = None,
) -> List[PyGSTiCircuit]:
    """Generate the idle RPE circuits.

    Args:
        circuit_depths (Sequence[int]): circuit depths.
        qubits (Sequence[int]): qubit labels.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
        List[PyGSTiCircuit]: pyGSTi circuits.
    """
    circuits = (
        [make_idle_cos_circ(d, qubits, gate_layer) for d in circuit_depths] +
        [make_idle_sin_circ(d, qubits, gate_layer) for d in circuit_depths]
    )

    return remove_duplicates(circuits)


def make_idle_cos_circ(
    depth: int, qubits: Sequence[int], gate_layer: GateLayer = None,
) -> PyGSTiCircuit:
    """Make the cosine circuit for idle RPE.

    Args:
        depth (int): circuit depth.
        qubits (Sequence[int]): qubit labels.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
       PyGSTiCircuit: pyGSTi circuit.
    """
    Gi_prep = PyGSTiCircuit(
        [[('Gypi2', q) for q in qubits]], line_labels=qubits
    )
    Gi_germ = PyGSTiCircuit(
        gate_layer if gate_layer else [[('Gidle', q) for q in qubits]],
        line_labels=qubits
    ) * depth
    Gi_meas = PyGSTiCircuit(
        [[('Gypi2', q) for q in qubits]], line_labels=qubits
    ) * 3

    return Gi_prep + Gi_germ + Gi_meas


def make_idle_sin_circ(
    depth: int, qubits: Sequence[int], gate_layer: GateLayer = None,
) -> PyGSTiCircuit:
    """Make the sine circuit for idle RPE.

    Args:
        depth (int): circuit depth.
        qubits (Sequence[int]): qubit labels.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
       PyGSTiCircuit: pyGSTi circuit.
    """
    Gi_prep = PyGSTiCircuit(
        [[('Gxpi2', q) for q in qubits]], line_labels=qubits
    )
    Gi_germ = PyGSTiCircuit(
        gate_layer if gate_layer else [[('Gidle', q) for q in qubits]],
        line_labels=qubits
    ) * depth
    Gi_meas = PyGSTiCircuit(
        [[('Gypi2', q) for q in qubits]], line_labels=qubits
    ) * 3

    return Gi_prep + Gi_germ + Gi_meas


def make_x90_circuits(
    circuit_depths: Sequence[int],
    qubits:         Sequence[int],
    gate_layer:     GateLayer = None,
) -> List[PyGSTiCircuit]:
    """Generate the axis X90 RPE circuits.

    Args:
        circuit_depths (Sequence[int]): circuit depths.
        qubits (Sequence[int]): qubit labels.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
        List[PyGSTiCircuit]: list of pyGSTi circuits.
    """
    circuits = (
        [make_x90_cos_circ(d, qubits, gate_layer) for d in circuit_depths] +
        [make_x90_sin_circ(d, qubits, gate_layer) for d in circuit_depths] +
        [make_X90_icos_circ(d, qubits, gate_layer) for d in circuit_depths] +
        [make_X90_isin_circ(d, qubits, gate_layer) for d in circuit_depths]
    )

    return remove_duplicates(circuits)


def make_x90_cos_circ(
    depth: int, qubits: Sequence[int], gate_layer: GateLayer = None,
) -> PyGSTiCircuit:
    """Make the cosine circuit for X90 RPE.

    Args:
        depth (int): circuit depth.
        qubits (Sequence[int]): qubit labels.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
       PyGSTiCircuit: pyGSTi circuit.
    """
    return PyGSTiCircuit(
            gate_layer if gate_layer else [[('Gxpi2', q) for q in qubits]],
            line_labels=qubits
        ) * depth


def make_x90_sin_circ(
    depth: int, qubits: Sequence[int], gate_layer: GateLayer = None,
) -> PyGSTiCircuit:
    """Make the sine circuit for X90 RPE.

    Args:
        depth (int): circuit depth.
        qubits (Sequence[int]): qubit labels.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
       PyGSTiCircuit: pyGSTi circuit.
    """
    return PyGSTiCircuit(
            gate_layer if gate_layer else [[('Gxpi2', q) for q in qubits]],
            line_labels=qubits
        ) * (depth + 1)


def make_X90_icos_circ(
    depth: int, qubits: Sequence[int], gate_layer: GateLayer = None,
) -> PyGSTiCircuit:
    """Make the interleaved cosine circuit for X90 axis error RPE.

    Args:
        depth (int): circuit depth.
        qubits (Sequence[int]): qubit labels.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
       PyGSTiCircuit: pyGSTi circuit.
    """
    Gz_layer = PyGSTiCircuit(
        [[('Gzpi2', q) for q in qubits]], line_labels=qubits
    )
    Gx_layer = PyGSTiCircuit(
        gate_layer if gate_layer else [[('Gxpi2', q) for q in qubits]],
        line_labels=qubits
    )

    return (
        Gz_layer + Gx_layer + Gx_layer + Gz_layer + Gz_layer + Gx_layer +
        Gx_layer + Gz_layer
    ) * depth


def make_X90_isin_circ(
    depth: int, qubits: Sequence[int], gate_layer: GateLayer = None,
) -> PyGSTiCircuit:
    """Make the interleaved sine circuit for X90 axis error RPE.

    Args:
        depth (int): circuit depth.
        qubits (Sequence[int]): qubit labels.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
       PyGSTiCircuit: pyGSTi circuit.
    """
    Gz_layer = PyGSTiCircuit(
        [[('Gzpi2', q) for q in qubits]], line_labels=qubits
    )
    Gx_layer = PyGSTiCircuit(
        gate_layer if gate_layer else [[('Gxpi2', q) for q in qubits]],
        line_labels=qubits
    )

    return (
        Gz_layer + Gx_layer + Gx_layer + Gz_layer + Gz_layer + Gx_layer +
        Gx_layer + Gz_layer
    ) * depth + Gx_layer


def make_cz_circuits(
    circuit_depths: Sequence[int],
    qubit_pairs:    Sequence[Tuple[int, int]],
    gate_layer:     GateLayer = None,
) -> List[PyGSTiCircuit]:
    """Generate CZ RPE circuits.

    Args:
        circuit_depths (Sequence[int]): circuit depths.
        qubit_pairs (Sequence[Tuple[int, int]]): pairs of qubits for two-qubit
            gates.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
        List[PyGSTiCircuit]: list of pyGSTi circuits.
    """
    state_pairs = [(0, 1), (2, 3), (3, 1)]
    sin_dict = {
        state_pair: {
            d: make_cz_sin_circ(
                d, state_pair, qubit_pairs, gate_layer
            ) for d in circuit_depths
        } for state_pair in state_pairs
    }
    cos_dict = {
        state_pair: {
            d: make_cz_cos_circ(
                d, state_pair, qubit_pairs, gate_layer
            ) for d in circuit_depths
        } for state_pair in state_pairs
    }

    circuits = []
    for trig_dict in [sin_dict, cos_dict]:
        for state_pair in state_pairs:
            circuits += list(trig_dict[state_pair].values())

    return circuits


def make_cz_cos_circ(
    depth:       int,
    state_pair:  Tuple[int, int],
    qubit_pairs: Sequence[Tuple[int, int]],
    gate_layer:  GateLayer = None,
) -> PyGSTiCircuit:
    """Make the cosine circuit for CZ RPE.

    Args:
        depth (int): circuit depth.
        state_pair (Tuple[int, int]): state pair.
        qubit_pairs (Sequence[Tuple[int, int]]): pairs of qubits for two-qubit
            gates.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
       PyGSTiCircuit: pyGSTi circuit.
    """
    line_labels = []
    for qubit_pair in qubit_pairs:
        line_labels.extend(qubit_pair)

    # <01| for 1/2 (1+cos(1/2 (theta_iz + theta_zz))
    # <00| for 1/2 (1-cos(1/2 (theta_iz + theta_zz))
    if state_pair in [(0, 1), (1, 0)]:
       circ = (
           PyGSTiCircuit(
               [[('Gypi2', qp[1]) for qp in qubit_pairs]],
               line_labels=line_labels
           ) +
           PyGSTiCircuit(
               gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
           ) * depth +
           PyGSTiCircuit(
               [[('Gypi2', qp[1]) for qp in qubit_pairs]]
           )
       )

    # <11| for 1/2 (1+cos(1/2 (theta_iz - theta_zz))
    elif state_pair in [(2, 3), (3, 2)]:
        circ = (
            PyGSTiCircuit(
                [[
                    gate for qp in qubit_pairs for gate in [
                        ('Gxpi2', qp[0]), ('Gypi2', qp[1])
                    ]
                ]],
                line_labels=line_labels
            ) +
            PyGSTiCircuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]]
            ) +
            PyGSTiCircuit(
                gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
            ) * depth +
            PyGSTiCircuit(
                [[('Gypi2', qp[1]) for qp in qubit_pairs]]
            )
        )

    # <11| for 1/2 (1+cos(1/2 (theta_zi - theta_zz))
    elif state_pair in [(1, 3), (3, 1)]:
        circ = (
            PyGSTiCircuit(
                [[
                    gate for qp in qubit_pairs for gate in [
                        ('Gypi2', qp[0]), ('Gxpi2', qp[1])
                    ]
                ]],
                line_labels=line_labels
            ) +
            PyGSTiCircuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            ) +
            PyGSTiCircuit(
                gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
            ) * depth +
            PyGSTiCircuit(
                [[('Gypi2', qp[0]) for qp in qubit_pairs]]
            )
        )

    else:
        raise AssertionError(
            "state_pair must be in [(0,1), (1,0), (2,3), (3,2), (1,3), (3,1)]"
        )

    return circ


def make_cz_sin_circ(
    depth:       int,
    state_pair:  Tuple[int, int],
    qubit_pairs: Sequence[Tuple[int, int]],
    gate_layer:  GateLayer = None,
) -> PyGSTiCircuit:
    """Make the sine circuit for CZ RPE.

    Args:
        depth (int): circuit depth.
        state_pair (Tuple[int, int]): state pair.
        qubit_pairs (Sequence[Tuple[int, int]]): pairs of qubits for two-qubit
            gates.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
       PyGSTiCircuit: pyGSTi circuit.
    """
    line_labels = []
    for qubit_pair in qubit_pairs:
        line_labels.extend(qubit_pair)

    # <00| for 1/2 (1+sin(1/2 (theta_iz + theta_zz))
    if state_pair in [(0, 1), (1, 0)]:
        circ = (
            PyGSTiCircuit(
                [[('Gypi2', qp[1]) for qp in qubit_pairs]],
                line_labels=line_labels
            ) +
            PyGSTiCircuit(
                gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
            ) * depth +
            PyGSTiCircuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            )
        )

    # <10| for 1/2 (1+sin(1/2 (theta_iz - theta_zz))
    elif state_pair in [(2, 3),(3, 2)]:
        circ = (
            PyGSTiCircuit(
                [[
                    gate for qp in qubit_pairs for gate in [
                        ('Gxpi2', qp[0]), ('Gypi2', qp[1])
                    ]
                ]],
                line_labels=line_labels
            ) +
            PyGSTiCircuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]]
            ) +
            PyGSTiCircuit(
                gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
            ) * depth +
            PyGSTiCircuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            )
        )

    # <01| for 1/2 (1+sin(1/2 (theta_zi - theta_zz))
    elif state_pair in [(1, 3), (3, 1)]:
        circ = (
            PyGSTiCircuit(
                [[
                    gate for qp in qubit_pairs for gate in [
                        ('Gypi2', qp[0]), ('Gxpi2', qp[1]),
                    ]
                ]],
                line_labels=line_labels
            ) +
            PyGSTiCircuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            ) +
            PyGSTiCircuit(
                gate_layer if gate_layer else [
                   [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
               ]
            ) * depth +
            PyGSTiCircuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]]
            )
        )
    else:
        raise AssertionError(
            "state_pair must be in [(0,1), (1,0), (2,3), (3,2), (1,3), (3,1)]"
        )
    return circ


def make_zz_circuits(
    circuit_depths: Sequence[int],
    qubit_pairs:    Sequence[Tuple[int, int]],
    gate_layer:     GateLayer = None,
) -> List[PyGSTiCircuit]:
    """Make the circuits for ZZ RPE.

    Each circuit depth produces exactly 8 randomized variants — one for each
    element of the ZZ centralizer. Each qubit pair gets an independent random
    permutation of the 8 centralizer elements, so no element is repeated across
    the 8 randomizations.

    Args:
        circuit_depths (Sequence[int]): circuit depths.
        qubit_pairs (Sequence[Tuple[int, int]]): pairs of qubits for two-qubit
            gates.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
        List[PyGSTiCircuit]: pyGSTi circuits for ZZ RPE.
    """
    circuits = []
    for d in circuit_depths:
        # Fresh independent shuffle per qubit pair at each depth, so the Pauli
        # sequences are uncorrelated across circuit depths.
        permutations = [
            list(np.random.permutation(_ZZ_CENTRALIZER_SORTED))
            for _ in qubit_pairs
        ]
        for r in range(_N_CENTRALIZERS):
            circuits.extend(
                make_cz_circuits(
                    circuit_depths=[1],
                    qubit_pairs=qubit_pairs,
                    gate_layer=make_twirled_zz_subcircuit(
                        depth=d,
                        qubit_pairs=qubit_pairs,
                        permutations=permutations,
                        offset=r,
                        gate_layer=gate_layer,
                    )
                )
            )
    return circuits


def make_twirled_zz_subcircuit(
    depth:        int,
    qubit_pairs:  Sequence[tuple[int, int]],
    permutations: List[List[str]],
    offset:       int,
    gate_layer:   GateLayer = None,
) -> List:
    """Make a twirled ZZ subcircuit for RPE.

    At each gate repetition j, applies the Pauli frame P·gate·P where P is
    drawn from the pre-shuffled permutation at index (offset + j) % 8. This
    ensures two properties simultaneously:
      - Within a circuit: each depth step uses a different centralizer element.
      - Across 8 randomizations (offsets 0-7): every centralizer element
        appears exactly once at each depth position.

    Args:
        depth (int): number of gate repetitions.
        qubit_pairs (Sequence[tuple[int, int]]): qubit pairs.
        permutations (List[List[str]]): one shuffled list of the 8 ZZ
            centralizer elements per qubit pair.
        offset (int): starting index into each permutation (0-7), equal to the
            randomization index from the outer loop.
        gate_layer (GateLayer, optional): custom gate layer for the gate of
            interest. Defaults to None.

    Returns:
        List: raw circuit layers for use as a gate_layer argument.
    """
    sub_circuit = []
    paulis_prev = None
    for j in range(depth):
        paulis = [
            permutations[i][(offset + j) % _N_CENTRALIZERS]
            for i in range(len(qubit_pairs))
        ]
        layer = [
            (_PAULI_GATE[pauli[qi]], qp[qi])
            for pauli, qp in zip(paulis, qubit_pairs, strict=True)
            for qi in (0, 1)
        ]
        if j == 0:
            sub_circuit.extend([layer, layer])
        else:
            # Merge the trailing P_{j-1} with the leading P_j into one Pauli.
            combined = _pauli_product_layer(paulis_prev, paulis, qubit_pairs)
            if combined is not None:
                sub_circuit.extend([combined, combined])

        sub_circuit.extend(
            gate_layer if gate_layer
            else [[('Gidle', q) for q in flatten(qubit_pairs)]]
        )

        if j == depth - 1:
            sub_circuit.extend([layer, layer])

        paulis_prev = paulis

    return sub_circuit
