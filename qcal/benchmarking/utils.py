"""Helper functions for benchmarking protocols.

"""
import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, product

logger = logging.getLogger(__name__)


PauliString = tuple[str, ...]

@dataclass
class PauliMeasurementGroup:
    """Container for a set of co-measurable Pauli strings.

    Attributes:
        measurement_basis: The combined Pauli string that defines the
            measurement basis for the entire group.
        paulis: The original Pauli strings that can be measured in this basis.
    """
    measurement_basis: PauliString
    paulis: list[PauliString]


def generate_n_qubit_paulis(
    qubits:          Sequence[int],
    measured_qubits: Iterable[int] | None = None,
) -> list[tuple[str, ...]]:
    """Generates all n-qubit Pauli strings.

    The allowed local Paulis depend on whether or not a qubit is measured.
    Unmeasured qubits allow full {I, X, Y, Z}, while measured qubits are
    restricted to {I, Z}.

    Args:
        qubits (Sequence[int]): sequence of qubits labels.
        measured_qubits (Iterable[int] | None): iterable of measured qubit
            labels. Defaults to None.

    Returns:
        list[tuple[str, ...]]: list of groupings of n-qubit Paulis
    """
    qubits = list(qubits)
    measured = set(measured_qubits) if measured_qubits else set()
    paulis_by_qubit = [
        ['I', 'Z'] if q in measured else ['I', 'X', 'Y', 'Z'] for q in qubits
    ]
    paulis = list(product(*paulis_by_qubit))
    return paulis


def generate_n_qubit_paulis_up_to_weight_k(
    qubits:          Sequence[int],
    measured_qubits: Iterable[int] | None = None,
    weight_k:        int = 1,
    connectivity:    list[tuple[int, int]] | None = None,
) -> list[tuple[str, ...]]:
    """Generates Pauli strings up to a maximum weight.

    Args:
        qubits (Sequence[int]): sequence of qubits labels.
        measured_qubits (Iterable[int] | None): iterable of measured qubit
            labels. Defaults to None.
        weight_k (int): maximum number of non-identity entries allowed in each
            Pauli string. Defaults to 1.
        connectivity (list[tuple[int, int]] | None): list of edges (pairs of
            qubit labels) defining the qubit connectivity graph. When provided,
            weight-k Pauli strings (k > 1) are only generated for qubit subsets
            that form a connected subgraph of size k. Defaults to None.

    Returns:
        list[tuple[str, ...]]: a list of groupings of n-qubit Paulis with weight
            <= weight_k.
    """
    qubits = list(qubits)
    measured = set(measured_qubits) if measured_qubits else set()
    paulis_by_qubit = [
        ['Z'] if q in measured else ['X', 'Y', 'Z'] for q in qubits
    ]

    adjacency: dict[int, set[int]] = {}
    if connectivity is not None:
        for q1, q2 in connectivity:
            adjacency.setdefault(q1, set()).add(q2)
            adjacency.setdefault(q2, set()).add(q1)

    paulis = []
    for w in range(1, weight_k + 1):
        for positions in combinations(range(len(qubits)), w):
            if connectivity is not None and w > 1:
                selected = [qubits[p] for p in positions]
                if not _is_connected_subset(selected, adjacency):
                    continue
            # positions might be (0, 3), pick non-identity ops for those spots
            for ops in product(*[paulis_by_qubit[p] for p in positions]):
                pauli = ['I'] * len(qubits)
                for pos, op in zip(positions, ops, strict=True):
                    pauli[pos] = op
                paulis.append(tuple(pauli))

    return paulis


def generate_random_n_qubit_paulis(
    qubits:          Sequence[int],
    measured_qubits: Iterable[int] | None = None,
    n_random_paulis: int = 1,
) -> list[PauliString]:
    """Generates random n-qubit Pauli strings.

    Args:
        qubits (Sequence[int]): sequence of qubits labels.
        measured_qubits (Iterable[int] | None): iterable of measured qubit
            labels. Defaults to None.
        n_random_paulis (int): number of random Pauli strings to generate.
            Defaults to 1.

    Returns:
        list[tuple[str, ...]]: a list of randomly generated n-qubit Pauli
            strings.
    """
    qubits = list(qubits)
    measured = set(measured_qubits) if measured_qubits else set()
    options = [
        ['I', 'Z'] if q in measured else ['I', 'X', 'Y', 'Z'] for q in qubits
    ]

    return [
        tuple(random.choice(opts) for opts in options)
        for _ in range(n_random_paulis)
    ]


def generate_n_qubit_pauli_measurement_groups(
    paulis: list[tuple[str, ...]],
) -> list[PauliMeasurementGroup]:
    """Groups Pauli strings by commuting measurement basis.

    This function first groups Pauli strings into sets that can be measured
    simultaneously under a single product-basis measurement.

    Args:
        paulis (list[tuple[str, ...]]): list of n-qubit Pauli strings to group.

    Returns:
        list[PauliMeasurementGroup]: a list of PauliMeasurementGroup objects.
    """
    return _group_paulis_for_simultaneous_measurement(paulis)


def generate_n_qubit_pauli_measurement_map(
    paulis: list[tuple[str, ...]],
) -> dict[PauliString, list[PauliString]]:
    """Generates a mapping from measurement bases to original Pauli strings.

    Args:
        paulis (list[tuple[str, ...]]): list of n-qubit Pauli strings to group.


    Returns:
        dict[PauliString, List[PauliString]]: a dictionary mapping
            measurement_basis -> list of Pauli strings measured in that basis.
    """
    groups = generate_n_qubit_pauli_measurement_groups(paulis)
    return {g.measurement_basis: g.paulis for g in groups}


def _combine_qwc_paulis(paulis: list[PauliString]) -> PauliString:
    """Combines qubit-wise commuting Pauli strings into one basis.

    The resulting Pauli string represents a measurement basis that can
    simultaneously measure all input Pauli strings.

    Args:
        paulis (list[PauliString]): list of qubit-wise commuting Pauli strings.

    Returns:
        PauliString: a single Pauli string representing the measurement basis.

    Raises:
        ValueError: If the input Paulis are not qubit-wise commuting.
    """
    combined = []
    for qubit_ops in zip(*paulis, strict=True):
        non_identity = {op for op in qubit_ops if op != 'I'}
        if len(non_identity) > 1:
            raise ValueError("Paulis are not qubit-wise commuting.")
        combined.append(non_identity.pop() if non_identity else 'I')
    return tuple(combined)


def _group_paulis_for_simultaneous_measurement(
    paulis: list[PauliString],
) -> list[PauliMeasurementGroup]:
    """Groups Pauli strings into qubit-wise commuting measurement sets.

    This uses a greedy algorithm: each Pauli string is assigned to the
    first compatible group, or a new group is created if none match.

    Args:
        paulis (list[PauliString]): list of Pauli strings to group.

    Returns:
        list[PauliMeasurementGroup]: A list of PauliMeasurementGroup objects.
            Each group contains:
                - measurement_basis: The basis needed to measure the group.
                - paulis: The original Pauli strings assigned to the group.
    """
    groups: list[PauliMeasurementGroup] = []

    # Sort by weight (descending) for better packing.
    paulis_sorted = sorted(
        paulis,
        key=lambda p: sum(x != 'I' for x in p),
        reverse=True,
    )

    for pauli in paulis_sorted:
        placed = False

        for group in groups:
            if _qwc_compatible(pauli, group.measurement_basis):
                group.paulis.append(pauli)
                group.measurement_basis = _combine_qwc_paulis(group.paulis)
                placed = True
                break

        if not placed:
            groups.append(
                PauliMeasurementGroup(
                    measurement_basis=pauli,
                    paulis=[pauli],
                )
            )

    return groups


def _is_connected_subset(
    qubits_subset: list[int],
    adjacency:     dict[int, set[int]],
) -> bool:
    """Checks if a subset of qubits forms a connected subgraph.

    Args:
        qubits_subset (list[int]): qubit labels to check.
        adjacency (dict[int, set[int]]): adjacency map of the full connectivity
            graph.

    Returns:
        bool: True if the subset forms a connected subgraph.
    """
    if len(qubits_subset) <= 1:
        return True
    subset = set(qubits_subset)
    visited = {qubits_subset[0]}
    queue = [qubits_subset[0]]
    while queue:
        node = queue.pop()
        for neighbor in adjacency.get(node, set()):
            if neighbor in subset and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited == subset


def _qwc_compatible(p1: PauliString, p2: PauliString) -> bool:
    """Checks if two Pauli strings are qubit-wise commuting.

    Two Pauli strings are qubit-wise commuting if, on every qubit,
    either:
      - at least one is identity, or
      - both are the same Pauli.

    Args:
        p1 (PauliString): First Pauli string.
        p2 (PauliString): Second Pauli string.

    Returns:
        bool: True if the Pauli strings are qubit-wise commuting, False
            otherwise.
    """
    for a, b in zip(p1, p2, strict=True):
        if a != 'I' and b != 'I' and a != b:
            return False
    return True
