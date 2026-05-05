""""Submodule for decomposing pyGSTi circuits.

"""
import logging
from typing import Dict, List, Tuple

from pygsti.baseobjs import Label as L

logger = logging.getLogger(__name__)


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
                L('Gzr', q, args=(phases[q][i // 2],))
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
            L('Gzr', qp[0], args=(phase[0],)),
            L('Gzr', qp[1], args=(phase[1],))
        ])
    layers.append(layer)

    return layers
