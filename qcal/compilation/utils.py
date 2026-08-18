"""Helper functions.

"""
from __future__ import annotations

from functools import lru_cache

from qcal.circuit import Circuit, Cycle
from qcal.compilation.pauli_conjugation import conjugate_pauli

__all__ = ('composes_to_identity',)


@lru_cache(maxsize=None)
def composes_to_identity(
    cycle_or_circuit: Cycle | Circuit, depth: int
) -> bool:
    """Check whether cycle_or_circuit^depth ≈ I (up to global phase).

    Strategy: a unitary U is proportional to the identity iff it conjugates
    every generator of the n-qubit Pauli group (X_q and Z_q for each qubit
    q) back to itself exactly, string and sign intact. Conjugation by c*I
    (|c| = 1) always fixes every operator exactly, while the n-qubit Pauli
    group's conjugation representation is irreducible, so by Schur's lemma
    the only unitaries that fix every generator are scalar multiples of the
    identity. Composing cycle_or_circuit's action `depth` times this way
    (via `conjugate_pauli`) only ever requires each gate's own small
    unitary, never the full 2^n x 2^n matrix.

    Args:
        cycle_or_circuit (Cycle | Circuit): the cycle or circuit to check.
        depth (int): the number of times to compose cycle_or_circuit.

    Returns:
        bool: True if cycle_or_circuit^depth is identity (up to global
            phase), False otherwise.
    """
    qubits = cycle_or_circuit.qubits
    n = len(qubits)
    for i in range(n):
        for label in ('X', 'Z'):
            generator = tuple('I' if j != i else label for j in range(n))
            pauli, sign = generator, 1
            for _ in range(depth):
                pauli, step_sign = conjugate_pauli(pauli, cycle_or_circuit)
                sign *= step_sign
            if pauli != generator or sign != 1:
                return False
    return True
