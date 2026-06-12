"""Submodule for decomposing pyGSTi circuits.

Optimized version:
- precomputes Clifford update table for 1-qubit Clifford + Pauli dressing
- avoids repeated qubit index linear searches
- avoids constructing temporary Pauli Label layers in update_clifford_labels
- uses sets for fast membership checks in padding
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pygsti
from numpy.typing import NDArray
from pygsti.baseobjs import Label
from pygsti.tools import internalgates

logger = logging.getLogger(__name__)


CLIFFORD_UNITARIES = {
    k: v for k, v in internalgates.standard_gatename_unitaries().items()
    if 'Gc' in k and v.shape == (2, 2)
}

CLIFFORD_1Q_NAMES = frozenset(CLIFFORD_UNITARIES.keys())

PAULI_TO_CLIFFORD = {
    (0, 0): 'Gc0',   # I
    (2, 0): 'Gc9',   # Z
    (0, 2): 'Gc3',   # X
    (2, 2): 'Gc6',   # Y
}

PAULI_NAMES = tuple(PAULI_TO_CLIFFORD.values())


def decompose_X90(
        phases: Dict[int, List]
) -> List[List[L | tuple]]:
    """Decompose an X90 gate into ZXZ or ZXZXZ.

    Args:
        phases (Dict[int, List]): local phases for the X90 on each qubit.

    Returns:
        List[List[Label | tuple]]: circuit layers, alternating between
            Gzr Label layers and Gxpi2 tuple layers.
    """
    qubits = list(phases.keys())
    n_layers = max(len(phases[q]) for q in qubits)

    layers = []
    for i in range(2 * n_layers - 1):
        if i % 2 == 0:
            layers.append([
                Label('Gzr', q, args=(phases[q][i // 2],))
                for q in qubits if i // 2 < len(phases[q])
            ])
        else:
            layers.append([
                ('Gxpi2', q) for q in qubits if (i // 2) < (len(phases[q]) - 1)
            ])

    # Remove any empty layers (in case some qubits have fewer phase corrections)
    layers = [layer for layer in layers if layer]

    return layers


def decompose_CZ(
        phases: Dict[Tuple[int, int], List]
) -> List[List[L | tuple]]:
    """Decompose a CZ gate into CZ + IZ + ZI.

    Args:
        phases (Dict[Tuple[int, int], List]): local phases for the CZ on each
            qubit pair.

    Returns:
        List[List[Label | tuple]]: circuit layers — a Gcphase tuple layer
            followed by a Gzr Label layer.
    """
    qubit_pairs = list(phases.keys())
    layers = [
        [('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]
    ]

    layer = []
    for qp, phase in phases.items():
        layer.extend([
            Label('Gzr', qp[0], args=(phase[0],)),
            Label('Gzr', qp[1], args=(phase[1],))
        ])
    layers.append(layer)

    return layers


def get_clifford_from_unitary(U: NDArray) -> str:
    """Get a Clifford gate from a unitary matrix.

    Args:
        U (NDArray): unitary matrix.

    Raises:
        ValueError: if it fails to look up a Clifford gate from the unitary.

    Returns:
        str: Name of the Clifford gate.
    """
    for k, v in CLIFFORD_UNITARIES.items():
        for phase in (1, -1, 1j, -1j):
            if np.allclose(U, phase * v):
                return k

    raise ValueError(
        f'Failed to look up Clifford for unitary:\n{U}\n'
        'This is not a recognized 1-qubit Clifford (up to global phase).'
    )


def _build_clifford_update_table() -> dict[Tuple[str, str, str], str]:
    """Precompute 1-qubit Clifford label updates for all Pauli dressings."""
    table = {}
    for p_name in PAULI_NAMES:
        P = CLIFFORD_UNITARIES[p_name]
        for g_name in CLIFFORD_1Q_NAMES:
            U = CLIFFORD_UNITARIES[g_name]
            for q_name in PAULI_NAMES:
                Q = CLIFFORD_UNITARIES[q_name]
                table[
                    (p_name, g_name, q_name)
                ] = get_clifford_from_unitary(Q @ U @ P)
    return table


CLIFFORD_UPDATE_TABLE = _build_clifford_update_table()


def _pauli_entry_to_clifford_name(p_vec: NDArray, idx: int, n: int) -> str:
    """Convert a single Pauli-vector entry pair to a 1-qubit Clifford name."""
    try:
        return PAULI_TO_CLIFFORD[(int(p_vec[idx]), int(p_vec[idx + n]))]
    except KeyError as exc:
        raise ValueError(
            f'Invalid Pauli-vector entry at qubit index {idx}: '
            f'({p_vec[idx]}, {p_vec[idx + n]}).'
        ) from exc


def pad_layer_clifford(layer: Label, qubits: list[int]) -> Label:
    """Pad a layer of Clifford gates with identity gates for unused qubits.

    Args:
        layer (Label): layer of Clifford gates.
        qubits (list[int]): qubit labels.

    Returns:
        Label: padded layer of Clifford gates.
    """
    padded_layer = list(layer)
    used_qubits = {q for g in layer for q in g.qubits}

    for q in qubits:
        if q not in used_qubits:
            padded_layer.append(Label('Gc0', q))

    return Label(padded_layer)


def pauli_randomize_clifford_circuit(
    circuit: pygsti.circuits.Circuit
) -> tuple[pygsti.circuits.Circuit, str, NDArray]:
    """Pauli randomize a Clifford circuit.

    Args:
        circuit (pygsti.circuits.Circuit): Clifford circuit.

    Returns:
        Circuit: Pauli randomized Clifford circuit.
        str: bitstring of measurement bitflips.
        NDArray: random Pauli string.
    """
    d = circuit.depth
    n = circuit.width
    p_vec = np.zeros(2 * n, int)
    q_vec = np.zeros(2 * n, int)
    rc_circ = pygsti.circuits.Circuit(
        line_labels=circuit.line_labels, editable=True
    )
    qubits = list(circuit.line_labels)
    qubit_to_idx = {qb: i for i, qb in enumerate(qubits)}

    measurement_bitflips = []
    for i in range(d):
        layer = circuit.layer_label(i).components

        if len(layer) == 0 or layer[0].name in CLIFFORD_1Q_NAMES:
            q_vec = 2 * np.random.randint(0, 2, 2 * n)
            padded_layer = pad_layer_clifford(layer, qubits)
            rc_layer = update_clifford_labels(
                padded_layer, p_vec, q_vec, qubits, qubit_to_idx
            )
            rc_circ.insert_layer_inplace(rc_layer, i)
            p_vec = q_vec

        else:
            rc_circ.insert_layer_inplace(layer, i)
            for g in layer:
                if g.name == 'Gcnot':
                    control, target = g.qubits
                    c_idx = qubit_to_idx[control]
                    t_idx = qubit_to_idx[target]
                    p_vec[c_idx] = (p_vec[c_idx] + p_vec[t_idx]) % 4
                    p_vec[n + t_idx] = (p_vec[n + c_idx] + p_vec[n + t_idx]) % 4

                elif g.name == 'Gcphase':
                    control, target = g.qubits
                    c_idx = qubit_to_idx[control]
                    t_idx = qubit_to_idx[target]
                    # target: forward propagate phase based on state of control
                    p_vec[t_idx] = (p_vec[n + c_idx] + p_vec[t_idx]) % 4
                    # control: back propagate a phase based on state of target
                    p_vec[c_idx] = (p_vec[c_idx] + p_vec[n + t_idx]) % 4

                elif g.name == 'Gc0':
                    continue

                elif g.name == 'Iz':
                    idx = qubit_to_idx[g.qubits[0]]
                    # Need to record if X was applied pre-measurement
                    x = int(p_vec[idx + n] == 2)
                    measurement_bitflips.append(str(x))

                else:
                    raise ValueError(
                        "Circuit can only contain Gcnot, Gcphase, Gc[0-23], "
                        "and/or Iz gates in layers!"
                    )

    # Final measurement bitstring is determined by whether an X gate was applied
    # pre-measurement
    bs = ''.join(measurement_bitflips + [str(b // 2) for b in q_vec[n:]])
    rc_circ.done_editing()

    return rc_circ, bs, q_vec


def pauli_vector_to_clifford_layer(p_vec: NDArray, qubits: list[int]) -> Label:
    """Convert a Pauli vector to a layer of Clifford gates.

    Args:
        p_vec (NDArray): Pauli vector
        qubits (list[int]): qubit labels

    Returns:
        Label: Clifford layer.
    """
    n = len(qubits)
    layer = []
    for i, q in enumerate(qubits):
        label = PAULI_TO_CLIFFORD[(int(p_vec[i]), int(p_vec[i + n]))]
        layer.append(Label(label, q))

    return Label(layer)


def update_clifford_labels(
    layer: Label,
    p_vec: NDArray,
    q_vec: NDArray,
    qubits: list[int],
    qubit_to_idx: dict[int, int] | None = None,
) -> Label:
    """Update Clifford labels in a layer based on Pauli vectors.

    Args:
        layer (Label): layer of gates.
        p_vec (NDArray): Pauli vector for left side.
        q_vec (NDArray): Pauli vector for right side.
        qubits (list[int]): qubit labels.
        qubit_to_idx (dict[int, int] | None): optional precomputed qubit->index
            map.

    Returns:
        Label: updated layer with Clifford labels.
    """
    if qubit_to_idx is None:
        qubit_to_idx = {qb: i for i, qb in enumerate(qubits)}

    n = len(qubits)
    new_layer = []
    used_qubits = set()

    for g in layer:
        idx = qubit_to_idx[g.qubits[0]]
        p_name = _pauli_entry_to_clifford_name(p_vec, idx, n)
        q_name = _pauli_entry_to_clifford_name(q_vec, idx, n)
        cPUQ = CLIFFORD_UPDATE_TABLE[(p_name, g.name, q_name)]
        new_layer.append(Label(cPUQ, g.qubits[0]))
        used_qubits.add(g.qubits[0])

    expected_set = set(qubits)
    if used_qubits != expected_set:
        missing = sorted(expected_set - used_qubits)
        extra = sorted(used_qubits - expected_set)
        raise ValueError(
            'Layer does not act on the expected qubits. '
            f'Missing: {missing}. Extra: {extra}.'
        )

    return Label(new_layer)
