"""Precomputed Pauli-conjugation tables for Clifford gates.

For a Clifford gate U and Pauli string P (a tensor product of single-qubit
Pauli labels 'I', 'X', 'Y', 'Z', one per qubit U acts on), conjugation
U P U† is always proportional to another Pauli Q of the same weight, with a
real ±1 sign: U P U† = sign * Q. TWO_QUBIT_PAULI_CONJUGATION_TABLE and
SINGLE_QUBIT_PAULI_CONJUGATION_TABLE hardcode this P -> (Q, sign) map for
every gate class in qcal.gate.two_qubit / qcal.gate.single_qubit,
respectively, that is Clifford at its default parameters, so callers can
look the result up directly instead of doing the matrix conjugation and
Pauli-string identification every time.

These mappings are fixed facts about each gate's matrix definition, not
data that depends on qubits, config, or runtime state, so they are written
here as literal values rather than recomputed on every import.

Convention: for a two-qubit gate with `gate.qubits == (q0, q1)`, a Pauli
string `(p0, p1)` means p0 acts on q0 and p1 acts on q1 (matching the
tensor-factor order of `gate.unitary`); a single-qubit Pauli is just its
bare label, e.g. 'X'. Both tables are keyed by `gate.name` (e.g. 'CZ',
'CNOT', 'iSWAP', 'Cliff4', 'H'), not by qubit labels, since the mapping
only depends on the gate's unitary, not on which physical qubits it acts
on. `SINGLE_QUBIT_PAULI_CONJUGATION_TABLE` includes entries for gate
classes that are aliases of the same underlying Clifford (e.g. 'SX', 'V',
and 'X90' are all sqrt(X) = Cliff4), since each is a distinct `gate.name`.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from qcal.circuit import Circuit, Cycle
from qcal.gate.gate import Gate
from qcal.gate.single_qubit import SINGLE_QUBIT_PAULIS

__all__ = (
    'ONE_QUBIT_PAULIS',
    'TWO_QUBIT_PAULI_CONJUGATION_TABLE',
    'SINGLE_QUBIT_PAULI_CONJUGATION_TABLE',
    'TWO_QUBIT_PAULIS',
    'get_pauli_conjugation',
    'pauli_str_to_matrix',
    'identify_pauli_str',
    'conjugate_pauli_by_unitary',
    'conjugate_pauli_by_gate',
    'conjugate_pauli_by_cycle',
    'conjugate_pauli',
)

NQubitPauliString = tuple[str, ...]

PauliString = tuple[str, str]

_PAULI_LABELS = ('I', 'X', 'Y', 'Z')

ONE_QUBIT_PAULIS: tuple[str, ...] = _PAULI_LABELS

TWO_QUBIT_PAULIS: tuple[PauliString, ...] = tuple(
    (p0, p1) for p0 in _PAULI_LABELS for p1 in _PAULI_LABELS
)

TWO_QUBIT_PAULI_CONJUGATION_TABLE: dict[
    str, dict[PauliString, tuple[PauliString, int]]
] = {
    'CNOT': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('I', 'X'), 1),
        ('I', 'Y'): (('Z', 'Y'), 1),
        ('I', 'Z'): (('Z', 'Z'), 1),
        ('X', 'I'): (('X', 'X'), 1),
        ('X', 'X'): (('X', 'I'), 1),
        ('X', 'Y'): (('Y', 'Z'), 1),
        ('X', 'Z'): (('Y', 'Y'), -1),
        ('Y', 'I'): (('Y', 'X'), 1),
        ('Y', 'X'): (('Y', 'I'), 1),
        ('Y', 'Y'): (('X', 'Z'), -1),
        ('Y', 'Z'): (('X', 'Y'), 1),
        ('Z', 'I'): (('Z', 'I'), 1),
        ('Z', 'X'): (('Z', 'X'), 1),
        ('Z', 'Y'): (('I', 'Y'), 1),
        ('Z', 'Z'): (('I', 'Z'), 1),
    },
    'CPhase': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('Z', 'X'), 1),
        ('I', 'Y'): (('Z', 'Y'), 1),
        ('I', 'Z'): (('I', 'Z'), 1),
        ('X', 'I'): (('X', 'Z'), 1),
        ('X', 'X'): (('Y', 'Y'), 1),
        ('X', 'Y'): (('Y', 'X'), -1),
        ('X', 'Z'): (('X', 'I'), 1),
        ('Y', 'I'): (('Y', 'Z'), 1),
        ('Y', 'X'): (('X', 'Y'), -1),
        ('Y', 'Y'): (('X', 'X'), 1),
        ('Y', 'Z'): (('Y', 'I'), 1),
        ('Z', 'I'): (('Z', 'I'), 1),
        ('Z', 'X'): (('I', 'X'), 1),
        ('Z', 'Y'): (('I', 'Y'), 1),
        ('Z', 'Z'): (('Z', 'Z'), 1),
    },
    'CX': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('I', 'X'), 1),
        ('I', 'Y'): (('Z', 'Y'), 1),
        ('I', 'Z'): (('Z', 'Z'), 1),
        ('X', 'I'): (('X', 'X'), 1),
        ('X', 'X'): (('X', 'I'), 1),
        ('X', 'Y'): (('Y', 'Z'), 1),
        ('X', 'Z'): (('Y', 'Y'), -1),
        ('Y', 'I'): (('Y', 'X'), 1),
        ('Y', 'X'): (('Y', 'I'), 1),
        ('Y', 'Y'): (('X', 'Z'), -1),
        ('Y', 'Z'): (('X', 'Y'), 1),
        ('Z', 'I'): (('Z', 'I'), 1),
        ('Z', 'X'): (('Z', 'X'), 1),
        ('Z', 'Y'): (('I', 'Y'), 1),
        ('Z', 'Z'): (('I', 'Z'), 1),
    },
    'CY': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('Z', 'X'), 1),
        ('I', 'Y'): (('I', 'Y'), 1),
        ('I', 'Z'): (('Z', 'Z'), 1),
        ('X', 'I'): (('X', 'Y'), 1),
        ('X', 'X'): (('Y', 'Z'), -1),
        ('X', 'Y'): (('X', 'I'), 1),
        ('X', 'Z'): (('Y', 'X'), 1),
        ('Y', 'I'): (('Y', 'Y'), 1),
        ('Y', 'X'): (('X', 'Z'), 1),
        ('Y', 'Y'): (('Y', 'I'), 1),
        ('Y', 'Z'): (('X', 'X'), -1),
        ('Z', 'I'): (('Z', 'I'), 1),
        ('Z', 'X'): (('I', 'X'), 1),
        ('Z', 'Y'): (('Z', 'Y'), 1),
        ('Z', 'Z'): (('I', 'Z'), 1),
    },
    'CZ': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('Z', 'X'), 1),
        ('I', 'Y'): (('Z', 'Y'), 1),
        ('I', 'Z'): (('I', 'Z'), 1),
        ('X', 'I'): (('X', 'Z'), 1),
        ('X', 'X'): (('Y', 'Y'), 1),
        ('X', 'Y'): (('Y', 'X'), -1),
        ('X', 'Z'): (('X', 'I'), 1),
        ('Y', 'I'): (('Y', 'Z'), 1),
        ('Y', 'X'): (('X', 'Y'), -1),
        ('Y', 'Y'): (('X', 'X'), 1),
        ('Y', 'Z'): (('Y', 'I'), 1),
        ('Z', 'I'): (('Z', 'I'), 1),
        ('Z', 'X'): (('I', 'X'), 1),
        ('Z', 'Y'): (('I', 'Y'), 1),
        ('Z', 'Z'): (('Z', 'Z'), 1),
    },
    'DCNOT': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('X', 'I'), 1),
        ('I', 'Y'): (('Y', 'Z'), 1),
        ('I', 'Z'): (('Z', 'Z'), 1),
        ('X', 'I'): (('X', 'X'), 1),
        ('X', 'X'): (('I', 'X'), 1),
        ('X', 'Y'): (('Z', 'Y'), 1),
        ('X', 'Z'): (('Y', 'Y'), -1),
        ('Y', 'I'): (('X', 'Y'), 1),
        ('Y', 'X'): (('I', 'Y'), 1),
        ('Y', 'Y'): (('Z', 'X'), -1),
        ('Y', 'Z'): (('Y', 'X'), 1),
        ('Z', 'I'): (('I', 'Z'), 1),
        ('Z', 'X'): (('X', 'Z'), 1),
        ('Z', 'Y'): (('Y', 'I'), 1),
        ('Z', 'Z'): (('Z', 'I'), 1),
    },
    'M': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('X', 'Y'), -1),
        ('I', 'Y'): (('I', 'Z'), -1),
        ('I', 'Z'): (('X', 'X'), 1),
        ('X', 'I'): (('Y', 'Z'), 1),
        ('X', 'X'): (('Z', 'X'), 1),
        ('X', 'Y'): (('Y', 'I'), -1),
        ('X', 'Z'): (('Z', 'Y'), 1),
        ('Y', 'I'): (('X', 'I'), -1),
        ('Y', 'X'): (('I', 'Y'), 1),
        ('Y', 'Y'): (('X', 'Z'), 1),
        ('Y', 'Z'): (('I', 'X'), -1),
        ('Z', 'I'): (('Z', 'Z'), 1),
        ('Z', 'X'): (('Y', 'X'), -1),
        ('Z', 'Y'): (('Z', 'I'), -1),
        ('Z', 'Z'): (('Y', 'Y'), -1),
    },
    'SWAP': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('X', 'I'), 1),
        ('I', 'Y'): (('Y', 'I'), 1),
        ('I', 'Z'): (('Z', 'I'), 1),
        ('X', 'I'): (('I', 'X'), 1),
        ('X', 'X'): (('X', 'X'), 1),
        ('X', 'Y'): (('Y', 'X'), 1),
        ('X', 'Z'): (('Z', 'X'), 1),
        ('Y', 'I'): (('I', 'Y'), 1),
        ('Y', 'X'): (('X', 'Y'), 1),
        ('Y', 'Y'): (('Y', 'Y'), 1),
        ('Y', 'Z'): (('Z', 'Y'), 1),
        ('Z', 'I'): (('I', 'Z'), 1),
        ('Z', 'X'): (('X', 'Z'), 1),
        ('Z', 'Y'): (('Y', 'Z'), 1),
        ('Z', 'Z'): (('Z', 'Z'), 1),
    },
    'XX': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('I', 'X'), 1),
        ('I', 'Y'): (('I', 'Y'), -1),
        ('I', 'Z'): (('I', 'Z'), -1),
        ('X', 'I'): (('X', 'I'), 1),
        ('X', 'X'): (('X', 'X'), 1),
        ('X', 'Y'): (('X', 'Y'), -1),
        ('X', 'Z'): (('X', 'Z'), -1),
        ('Y', 'I'): (('Y', 'I'), -1),
        ('Y', 'X'): (('Y', 'X'), -1),
        ('Y', 'Y'): (('Y', 'Y'), 1),
        ('Y', 'Z'): (('Y', 'Z'), 1),
        ('Z', 'I'): (('Z', 'I'), -1),
        ('Z', 'X'): (('Z', 'X'), -1),
        ('Z', 'Y'): (('Z', 'Y'), 1),
        ('Z', 'Z'): (('Z', 'Z'), 1),
    },
    'XY': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('I', 'X'), -1),
        ('I', 'Y'): (('I', 'Y'), -1),
        ('I', 'Z'): (('I', 'Z'), 1),
        ('X', 'I'): (('X', 'I'), -1),
        ('X', 'X'): (('X', 'X'), 1),
        ('X', 'Y'): (('X', 'Y'), 1),
        ('X', 'Z'): (('X', 'Z'), -1),
        ('Y', 'I'): (('Y', 'I'), -1),
        ('Y', 'X'): (('Y', 'X'), 1),
        ('Y', 'Y'): (('Y', 'Y'), 1),
        ('Y', 'Z'): (('Y', 'Z'), -1),
        ('Z', 'I'): (('Z', 'I'), 1),
        ('Z', 'X'): (('Z', 'X'), -1),
        ('Z', 'Y'): (('Z', 'Y'), -1),
        ('Z', 'Z'): (('Z', 'Z'), 1),
    },
    'YY': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('I', 'X'), -1),
        ('I', 'Y'): (('I', 'Y'), 1),
        ('I', 'Z'): (('I', 'Z'), -1),
        ('X', 'I'): (('X', 'I'), -1),
        ('X', 'X'): (('X', 'X'), 1),
        ('X', 'Y'): (('X', 'Y'), -1),
        ('X', 'Z'): (('X', 'Z'), 1),
        ('Y', 'I'): (('Y', 'I'), 1),
        ('Y', 'X'): (('Y', 'X'), -1),
        ('Y', 'Y'): (('Y', 'Y'), 1),
        ('Y', 'Z'): (('Y', 'Z'), -1),
        ('Z', 'I'): (('Z', 'I'), -1),
        ('Z', 'X'): (('Z', 'X'), 1),
        ('Z', 'Y'): (('Z', 'Y'), -1),
        ('Z', 'Z'): (('Z', 'Z'), 1),
    },
    'ZZ': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('I', 'X'), -1),
        ('I', 'Y'): (('I', 'Y'), -1),
        ('I', 'Z'): (('I', 'Z'), 1),
        ('X', 'I'): (('X', 'I'), -1),
        ('X', 'X'): (('X', 'X'), 1),
        ('X', 'Y'): (('X', 'Y'), 1),
        ('X', 'Z'): (('X', 'Z'), -1),
        ('Y', 'I'): (('Y', 'I'), -1),
        ('Y', 'X'): (('Y', 'X'), 1),
        ('Y', 'Y'): (('Y', 'Y'), 1),
        ('Y', 'Z'): (('Y', 'Z'), -1),
        ('Z', 'I'): (('Z', 'I'), 1),
        ('Z', 'X'): (('Z', 'X'), -1),
        ('Z', 'Y'): (('Z', 'Y'), -1),
        ('Z', 'Z'): (('Z', 'Z'), 1),
    },
    'bSWAP': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('X', 'I'), 1),
        ('I', 'Y'): (('Y', 'I'), -1),
        ('I', 'Z'): (('Z', 'I'), -1),
        ('X', 'I'): (('I', 'X'), 1),
        ('X', 'X'): (('X', 'X'), 1),
        ('X', 'Y'): (('Y', 'X'), -1),
        ('X', 'Z'): (('Z', 'X'), -1),
        ('Y', 'I'): (('I', 'Y'), -1),
        ('Y', 'X'): (('X', 'Y'), -1),
        ('Y', 'Y'): (('Y', 'Y'), 1),
        ('Y', 'Z'): (('Z', 'Y'), 1),
        ('Z', 'I'): (('I', 'Z'), -1),
        ('Z', 'X'): (('X', 'Z'), -1),
        ('Z', 'Y'): (('Y', 'Z'), 1),
        ('Z', 'Z'): (('Z', 'Z'), 1),
    },
    'fSWAP': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('X', 'Z'), 1),
        ('I', 'Y'): (('Y', 'Z'), 1),
        ('I', 'Z'): (('Z', 'I'), 1),
        ('X', 'I'): (('Z', 'X'), 1),
        ('X', 'X'): (('Y', 'Y'), 1),
        ('X', 'Y'): (('X', 'Y'), -1),
        ('X', 'Z'): (('I', 'X'), 1),
        ('Y', 'I'): (('Z', 'Y'), 1),
        ('Y', 'X'): (('Y', 'X'), -1),
        ('Y', 'Y'): (('X', 'X'), 1),
        ('Y', 'Z'): (('I', 'Y'), 1),
        ('Z', 'I'): (('I', 'Z'), 1),
        ('Z', 'X'): (('X', 'I'), 1),
        ('Z', 'Y'): (('Y', 'I'), 1),
        ('Z', 'Z'): (('Z', 'Z'), 1),
    },
    'iSWAP': {
        ('I', 'I'): (('I', 'I'), 1),
        ('I', 'X'): (('Y', 'Z'), 1),
        ('I', 'Y'): (('X', 'Z'), -1),
        ('I', 'Z'): (('Z', 'I'), 1),
        ('X', 'I'): (('Z', 'Y'), 1),
        ('X', 'X'): (('X', 'X'), 1),
        ('X', 'Y'): (('Y', 'X'), 1),
        ('X', 'Z'): (('I', 'Y'), 1),
        ('Y', 'I'): (('Z', 'X'), -1),
        ('Y', 'X'): (('X', 'Y'), 1),
        ('Y', 'Y'): (('Y', 'Y'), 1),
        ('Y', 'Z'): (('I', 'X'), -1),
        ('Z', 'I'): (('I', 'Z'), 1),
        ('Z', 'X'): (('Y', 'I'), 1),
        ('Z', 'Y'): (('X', 'I'), -1),
        ('Z', 'Z'): (('Z', 'Z'), 1),
    },
}


# Keyed by gate.name. Cliff0-Cliff23 appear in their canonical numeric
# order; every other gate name is an alias of one of them (e.g. 'SX', 'V',
# and 'X90' are all sqrt(X) = Cliff4) and is listed alphabetically after.
SINGLE_QUBIT_PAULI_CONJUGATION_TABLE: dict[
    str, dict[str, tuple[str, int]]
] = {
    'Cliff0': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Y', 1),
        'Z': ('Z', 1),
    },
    'Cliff1': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Y', -1),
        'Z': ('Z', -1),
    },
    'Cliff2': {
        'I': ('I', 1),
        'X': ('X', -1),
        'Y': ('Y', 1),
        'Z': ('Z', -1),
    },
    'Cliff3': {
        'I': ('I', 1),
        'X': ('X', -1),
        'Y': ('Y', -1),
        'Z': ('Z', 1),
    },
    'Cliff4': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Z', 1),
        'Z': ('Y', -1),
    },
    'Cliff5': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Z', -1),
        'Z': ('Y', 1),
    },
    'Cliff6': {
        'I': ('I', 1),
        'X': ('Z', -1),
        'Y': ('Y', 1),
        'Z': ('X', 1),
    },
    'Cliff7': {
        'I': ('I', 1),
        'X': ('Z', 1),
        'Y': ('Y', 1),
        'Z': ('X', -1),
    },
    'Cliff8': {
        'I': ('I', 1),
        'X': ('Y', 1),
        'Y': ('X', -1),
        'Z': ('Z', 1),
    },
    'Cliff9': {
        'I': ('I', 1),
        'X': ('Y', -1),
        'Y': ('X', 1),
        'Z': ('Z', 1),
    },
    'Cliff10': {
        'I': ('I', 1),
        'X': ('Y', 1),
        'Y': ('X', 1),
        'Z': ('Z', -1),
    },
    'Cliff11': {
        'I': ('I', 1),
        'X': ('Y', -1),
        'Y': ('X', -1),
        'Z': ('Z', -1),
    },
    'Cliff12': {
        'I': ('I', 1),
        'X': ('X', -1),
        'Y': ('Z', 1),
        'Z': ('Y', 1),
    },
    'Cliff13': {
        'I': ('I', 1),
        'X': ('X', -1),
        'Y': ('Z', -1),
        'Z': ('Y', -1),
    },
    'Cliff14': {
        'I': ('I', 1),
        'X': ('Z', 1),
        'Y': ('Y', -1),
        'Z': ('X', 1),
    },
    'Cliff15': {
        'I': ('I', 1),
        'X': ('Z', -1),
        'Y': ('Y', -1),
        'Z': ('X', -1),
    },
    'Cliff16': {
        'I': ('I', 1),
        'X': ('Y', 1),
        'Y': ('Z', 1),
        'Z': ('X', 1),
    },
    'Cliff17': {
        'I': ('I', 1),
        'X': ('Z', 1),
        'Y': ('X', 1),
        'Z': ('Y', 1),
    },
    'Cliff18': {
        'I': ('I', 1),
        'X': ('Z', -1),
        'Y': ('X', 1),
        'Z': ('Y', -1),
    },
    'Cliff19': {
        'I': ('I', 1),
        'X': ('Y', 1),
        'Y': ('Z', -1),
        'Z': ('X', -1),
    },
    'Cliff20': {
        'I': ('I', 1),
        'X': ('Z', 1),
        'Y': ('X', -1),
        'Z': ('Y', -1),
    },
    'Cliff21': {
        'I': ('I', 1),
        'X': ('Y', -1),
        'Y': ('Z', -1),
        'Z': ('X', 1),
    },
    'Cliff22': {
        'I': ('I', 1),
        'X': ('Z', -1),
        'Y': ('X', -1),
        'Z': ('Y', 1),
    },
    'Cliff23': {
        'I': ('I', 1),
        'X': ('Y', -1),
        'Y': ('Z', 1),
        'Z': ('X', -1),
    },
    'C': {
        'I': ('I', 1),
        'X': ('Y', 1),
        'Y': ('Z', 1),
        'Z': ('X', 1),
    },
    'H': {
        'I': ('I', 1),
        'X': ('Z', 1),
        'Y': ('Y', -1),
        'Z': ('X', 1),
    },
    'Id': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Y', 1),
        'Z': ('Z', 1),
    },
    'Idle': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Y', 1),
        'Z': ('Z', 1),
    },
    'S': {
        'I': ('I', 1),
        'X': ('Y', 1),
        'Y': ('X', -1),
        'Z': ('Z', 1),
    },
    'SX': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Z', 1),
        'Z': ('Y', -1),
    },
    'SXdag': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Z', -1),
        'Z': ('Y', 1),
    },
    'SY': {
        'I': ('I', 1),
        'X': ('Z', -1),
        'Y': ('Y', 1),
        'Z': ('X', 1),
    },
    'SYdag': {
        'I': ('I', 1),
        'X': ('Z', 1),
        'Y': ('Y', 1),
        'Z': ('X', -1),
    },
    'Sdag': {
        'I': ('I', 1),
        'X': ('Y', -1),
        'Y': ('X', 1),
        'Z': ('Z', 1),
    },
    'V': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Z', 1),
        'Z': ('Y', -1),
    },
    'Vdag': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Z', -1),
        'Z': ('Y', 1),
    },
    'X': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Y', -1),
        'Z': ('Z', -1),
    },
    'X90': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Z', 1),
        'Z': ('Y', -1),
    },
    'Y': {
        'I': ('I', 1),
        'X': ('X', -1),
        'Y': ('Y', 1),
        'Z': ('Z', -1),
    },
    'Y90': {
        'I': ('I', 1),
        'X': ('Z', -1),
        'Y': ('Y', 1),
        'Z': ('X', 1),
    },
    'Z': {
        'I': ('I', 1),
        'X': ('X', -1),
        'Y': ('Y', -1),
        'Z': ('Z', 1),
    },
    'Z90': {
        'I': ('I', 1),
        'X': ('Y', 1),
        'Y': ('X', -1),
        'Z': ('Z', 1),
    },
}


@lru_cache(maxsize=1024)
def get_pauli_conjugation(
    gate: str | Gate, pauli: str | PauliString
) -> tuple[PauliString, int] | tuple[str, int] | None:
    """Look up how `gate` conjugates `pauli`, or None if not tabulated.

    Checks the two-qubit table first, then the single-qubit table, and
    parses `pauli` accordingly (a 2-qubit Pauli for the former, a bare
    single-qubit label for the latter).

    Cached (bounded): a dict lookup is already O(1), but a cache hit skips
    this function's own Python-level overhead.

    Args:
        gate (str | Gate): gate name (e.g. 'CZ', 'Cliff4') or Gate
            instance.
        pauli (str | PauliString): Pauli matching the number of qubits
            `gate` acts on, e.g. 'XZ' or ('X', 'Z') for a two-qubit gate,
            or 'X' for a single-qubit gate.

    Returns:
        tuple[PauliString, int] | tuple[str, int] | None: (conjugated
            Pauli, sign), or None if `gate` isn't a tabulated Clifford.
            Callers should fall back to direct unitary conjugation in
            that case.
    """
    name = gate.name if isinstance(gate, Gate) else gate
    entry = TWO_QUBIT_PAULI_CONJUGATION_TABLE.get(name)
    if entry is not None:
        return entry[_parse_two_qubit_pauli(pauli)]
    entry = SINGLE_QUBIT_PAULI_CONJUGATION_TABLE.get(name)
    if entry is not None:
        return entry[_parse_single_qubit_pauli(pauli)]
    return None


def _parse_two_qubit_pauli(pauli: str | tuple) -> PauliString:
    """Parse a 2-qubit Pauli given as a 2-char string or a 2-tuple.

    Args:
        pauli (str | tuple): e.g. 'XZ' or ('X', 'Z').

    Returns:
        PauliString: canonical 2-tuple, e.g. ('X', 'Z').
    """
    pauli = tuple(pauli)
    if len(pauli) != 2 or any(p not in _PAULI_LABELS for p in pauli):
        raise ValueError(
            f"'{pauli}' is not a valid 2-qubit Pauli string; expected two "
            "labels from {'I', 'X', 'Y', 'Z'}."
        )
    return pauli


def _parse_single_qubit_pauli(pauli: str | tuple) -> str:
    """Parse a 1-qubit Pauli given as a 1-char string or a 1-tuple.

    Args:
        pauli (str | tuple): e.g. 'X' or ('X',).

    Returns:
        str: canonical Pauli label, e.g. 'X'.
    """
    if len(pauli) != 1 or pauli[0] not in _PAULI_LABELS:
        raise ValueError(
            f"'{pauli}' is not a valid single-qubit Pauli; expected one "
            "label from {'I', 'X', 'Y', 'Z'}."
        )
    return pauli[0]


@lru_cache(maxsize=1024)
def pauli_str_to_matrix(pauli: NQubitPauliString) -> np.ndarray:
    """Build the n-qubit Pauli matrix for a PauliString.

    Cached (bounded): keyed on the Pauli string alone (not on which gate
    is being conjugated), so this is shared across e.g. multiple gates of
    the same type that only differ by which qubits they act on.

    Args:
        pauli (NQubitPauliString): tuple of single-qubit Pauli labels,
            e.g. ('X', 'Z', 'I').

    Returns:
        np.ndarray: the 2^n x 2^n matrix given by the Kronecker product
            of the corresponding single-qubit Pauli matrices.
    """
    mat = SINGLE_QUBIT_PAULIS[pauli[0]](0).matrix.astype(complex)
    for p in pauli[1:]:
        mat = np.kron(mat, SINGLE_QUBIT_PAULIS[p](0).matrix)
    return mat


def identify_pauli_str(M: np.ndarray, n: int) -> NQubitPauliString:
    """Return Pauli labels for an n-qubit Pauli matrix, ignoring phase.

    Thin wrapper around `_identify_pauli_str_cached`: ndarrays aren't
    hashable, so this converts M to a hashable (bytes, dtype, shape) key
    before delegating to the cached implementation.

    Args:
        M (np.ndarray): 2^n x 2^n matrix proportional to an n-qubit
            Pauli operator.
        n (int): number of qubits.

    Returns:
        NQubitPauliString: tuple of single-qubit Pauli labels identifying
            M, e.g. ('X', 'Z', 'I').
    """
    return _identify_pauli_str_cached(M.tobytes(), M.dtype.str, M.shape, n)


@lru_cache(maxsize=1024)
def _identify_pauli_str_cached(
    data: bytes, dtype: str, shape: tuple, n: int
) -> NQubitPauliString:
    """Cached (bounded) implementation of `identify_pauli_str`.

    Uses recursive block decomposition: for σ_k ⊗ rest, the 2x2 block
    structure of M in the first qubit's index uniquely identifies σ_k.
    Keyed on the matrix's raw content rather than gate identity, so this
    is shared across e.g. multiple gates of the same type that only
    differ by which qubits they act on.

    Args:
        data (bytes): raw bytes of M (`M.tobytes()`).
        dtype (str): dtype string of M (`M.dtype.str`).
        shape (tuple): shape of M.
        n (int): number of qubits.

    Returns:
        NQubitPauliString: tuple of single-qubit Pauli labels identifying
            M, e.g. ('X', 'Z', 'I').
    """
    if n == 0:
        return ()
    M = np.frombuffer(data, dtype=dtype).reshape(shape)
    half = M.shape[0] // 2
    M00, M01 = M[:half, :half], M[:half, half:]
    M10, M11 = M[half:, :half], M[half:, half:]

    if np.allclose(M01, 0) and np.allclose(M10, 0):
        if np.allclose(M00, M11):
            label, rest = 'I', M00
        else:
            label, rest = 'Z', M00
    elif np.allclose(M01, M10):
        label, rest = 'X', M01
    else:
        label, rest = 'Y', -1j * M10

    return (label,) + identify_pauli_str(rest, n - 1)


def conjugate_pauli_by_unitary(
    U: np.ndarray, pauli: NQubitPauliString
) -> tuple[NQubitPauliString, complex]:
    """Conjugate a Pauli matrix by U and identify the resulting scalar.

    Builds M = U P U† for P = matrix(pauli), identifies which Pauli
    operator M is proportional to, and recovers the scalar of
    proportionality from a nonzero entry shared by both matrices.

    Args:
        U (np.ndarray): unitary matrix to conjugate by.
        pauli (NQubitPauliString): Pauli labels for the qubits acted on
            by U, in U's tensor-factor order.

    Returns:
        tuple[NQubitPauliString, complex]: the Pauli string of U P U†,
            and the scalar such that U P U† = scalar * (matrix of the
            returned string).
    """
    P = pauli_str_to_matrix(pauli)
    M = U @ P @ U.conj().T
    new_pauli = identify_pauli_str(M, len(pauli))
    Q = pauli_str_to_matrix(new_pauli)
    row, col = next(
        (r, c)
        for r in range(Q.shape[0]) for c in range(Q.shape[1])
        if abs(Q[r, c]) > 1e-9
    )
    scalar = M[row, col] / Q[row, col]
    return new_pauli, scalar


@lru_cache(maxsize=None)
def conjugate_pauli_by_gate(
    pauli: NQubitPauliString, gate: Gate
) -> tuple[NQubitPauliString, int]:
    """Conjugate a small Pauli string by a gate's unitary.

    Cached: the domain is tiny (at most 4^len(gate.qubits) Pauli strings
    per distinct gate), so this saturates almost immediately regardless of
    how many circuits/randomizations reuse the same gate.

    Tries `get_pauli_conjugation` first, which looks `gate` up by name in
    the precomputed single-/two-qubit Clifford tables; only falls back to
    conjugating the actual unitary and identifying the resulting Pauli
    string (the expensive path) when `gate` isn't tabulated (e.g. a
    gate with more than two qubits).

    Args:
        pauli (NQubitPauliString): Pauli labels for the qubits acted on
            by gate, in gate.unitary's tensor-factor order.
        gate (Gate): gate whose unitary to conjugate by.

    Returns:
        tuple[NQubitPauliString, int]: the Pauli string of U P U†, and
            the sign such that U P U† = sign * (matrix of the returned
            string).

    Raises:
        ValueError: if U P U† is not proportional to a Pauli operator with
            real sign ±1, i.e. U is not a Clifford unitary.
    """
    computed = get_pauli_conjugation(gate, pauli)
    if computed is not None:
        conjugate_pauli, sign = computed
        if isinstance(conjugate_pauli, str):
            conjugate_pauli = (conjugate_pauli,)
        return conjugate_pauli, sign

    new_pauli, scalar = conjugate_pauli_by_unitary(gate.unitary, pauli)
    if not np.isclose(scalar.imag, 0, atol=1e-6) or not np.isclose(
        abs(scalar.real), 1, atol=1e-6
    ):
        raise ValueError(
            "cycle_or_circuit does not appear to be a Clifford operation: "
            f"conjugating Pauli '{''.join(pauli)}' did not yield a Pauli "
            "operator with a real ±1 sign. CB requires cycle_or_circuit to "
            "normalize the Pauli group."
        )
    return new_pauli, int(round(scalar.real))


@lru_cache(maxsize=None)
def conjugate_pauli_by_cycle(
    pauli: NQubitPauliString, qubits: tuple, cycle: Cycle
) -> tuple[NQubitPauliString, int]:
    """Conjugate an n-qubit Pauli string by one Cycle of disjoint gates.

    Since a Cycle's gates act on pairwise-disjoint qubits, each gate
    conjugates only the Pauli substring on its own qubits; results are
    combined without ever forming the full n-qubit unitary.

    Cached: only depth+1 distinct twirl layers and a handful of decay
    Paulis are ever conjugated by the same (fixed) cycle across an
    entire CB run, so repeats are common.

    Args:
        pauli (NQubitPauliString): n-qubit Pauli string, ordered by
            `qubits`.
        qubits (tuple): qubit labels corresponding to each entry of pauli.
        cycle (Cycle): cycle to conjugate by.

    Returns:
        tuple[NQubitPauliString, int]: conjugated Pauli string and
            accumulated sign.
    """
    new_pauli = list(pauli)
    sign = 1
    for gate in cycle.gates:
        idx = [qubits.index(q) for q in gate.qubits]
        local_pauli = tuple(pauli[i] for i in idx)
        conjugate_pauli, local_sign = conjugate_pauli_by_gate(
            local_pauli, gate
        )
        for i, p in zip(idx, conjugate_pauli, strict=True):
            new_pauli[i] = p
        sign *= local_sign
    return tuple(new_pauli), sign


@lru_cache(maxsize=None)
def conjugate_pauli(
    pauli: NQubitPauliString, cycle_or_circuit: Cycle | Circuit
) -> tuple[NQubitPauliString, int]:
    """Return the Pauli string (and sign) of U P U† for U = cycle_or_circuit.

    Composes per-cycle, per-gate local conjugation (each gate's dimension is
    fixed by its own qubit count, independent of the total number of
    qubits), so this never constructs the full 2^n x 2^n unitary.

    Cached: in CB's `_propagate_sign`, this result depends only on
    decay_pauli (not on depth, randomization, or twirl_strings), so
    across an entire CB run only ~n_decays distinct calls ever do real
    work.

    Args:
        pauli (NQubitPauliString): n-qubit Pauli string, ordered by
            `cycle_or_circuit.qubits`.
        cycle_or_circuit (Cycle | Circuit): cycle or circuit to conjugate
            by.

    Returns:
        tuple[NQubitPauliString, int]: conjugated Pauli string and
            accumulated sign.
    """
    qubits = cycle_or_circuit.qubits
    if isinstance(cycle_or_circuit, Cycle):
        return conjugate_pauli_by_cycle(pauli, qubits, cycle_or_circuit)

    sign = 1
    for cycle in cycle_or_circuit.cycles:
        if cycle.is_barrier:
            continue
        pauli, cycle_sign = conjugate_pauli_by_cycle(pauli, qubits, cycle)
        sign *= cycle_sign
    return pauli, sign
