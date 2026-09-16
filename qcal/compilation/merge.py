"""Utilities for merging two gates or cycles acting on the same qubits.

For example, a state-preparation cycle and the first Pauli twirl of a
randomized-benchmarking circuit both act on the same qubits; before being
decomposed into a hardware-native gate sequence (e.g. ZXZXZ), it is more
efficient to apply them as a single combined gate per qubit rather than
as two back-to-back gates.
"""
from __future__ import annotations

from functools import lru_cache

from qcal.circuit import Cycle
from qcal.gates.gate import Gate

__all__ = ('merge_gates', 'merge_cycles')


@lru_cache(maxsize=None)
def merge_gates(first: Gate, second: Gate) -> Gate:
    """Merge two gates acting on the same qubits into a single gate.

    Computes the unitary of applying `first` then `second`:
    `first.unitary @ second.unitary`, matching the ordering convention
    used by `Cycle.unitary`/`Circuit.unitary` (the earlier-applied factor
    is the left factor).

    Note: the returned Gate is a generic `Gate(matrix, qubits)`, so its
    `name` is always 'Gate', and its `==`/hash (which only consider name,
    qubits, subspace, and params) will not distinguish it from a
    different merge on the same qubits. Use its `.unitary`/`.matrix`
    directly rather than relying on its equality or hashing it.

    Args:
        first (Gate): gate applied first.
        second (Gate): gate applied second, acting on the same qubits as
            `first`.

    Returns:
        Gate: a generic gate whose unitary is `first.unitary @
            second.unitary`.

    Raises:
        ValueError: if `first` and `second` do not act on the same
            qubits, in the same order.
    """
    if first.qubits != second.qubits:
        raise ValueError(
            f"Cannot merge gates on different qubits: {first.qubits} "
            f"!= {second.qubits}."
        )
    return Gate(first.unitary @ second.unitary, first.qubits)


@lru_cache(maxsize=None)
def merge_cycles(first: Cycle, second: Cycle) -> Cycle:
    """Merge two cycles acting on the same qubits into a single cycle.

    Pairs each gate in `first` with the gate in `second` that acts on the
    same qubits, and merges each pair via `merge_gates` (see its
    docstring for the ordering convention and the merged gates' equality
    caveat).

    Args:
        first (Cycle): cycle applied first.
        second (Cycle): cycle applied second, acting on the same qubits
            as `first` and grouped into gates the same way (e.g. both
            purely single-qubit, one gate per qubit).

    Returns:
        Cycle: a cycle of merged gates, one per qubit grouping.

    Raises:
        ValueError: if `first` and `second` do not act on the same
            qubits, or are grouped into gates differently (e.g. one has
            a two-qubit gate spanning qubits that the other splits
            across two single-qubit gates).
    """
    if first.qubits != second.qubits:
        raise ValueError(
            f"Cannot merge cycles on different qubits: {first.qubits} "
            f"!= {second.qubits}."
        )

    second_gates_by_qubits = {gate.qubits: gate for gate in second.gates}
    merged_gates = []
    for gate in first.gates:
        other = second_gates_by_qubits.get(gate.qubits)
        if other is None:
            raise ValueError(
                f"No gate on qubits {gate.qubits} in `second`; `first` "
                "and `second` must be grouped into gates the same way."
            )
        merged_gates.append(merge_gates(gate, other))

    return Cycle(merged_gates)
