"""Unit tests for qcal.benchmarking.utils.

Every function here is a pure, deterministic function of its inputs
(no Config, Emulator, or pyGSTi involved), so these tests are all
fast, direct checks of the combinatorics.
"""
import pytest

from qcal.benchmarking.utils import (
    _combine_qwc_paulis,
    _is_connected_subset,
    _qwc_compatible,
    generate_n_qubit_pauli_measurement_groups,
    generate_n_qubit_pauli_measurement_map,
    generate_n_qubit_paulis,
    generate_n_qubit_paulis_up_to_weight_k,
    generate_random_n_qubit_paulis,
)


class TestGenerateNQubitPaulis:

    def test_no_measured_qubits_uses_full_pauli_set(self):
        paulis = generate_n_qubit_paulis([0, 1])
        # 4 (I, X, Y, Z) per qubit, 2 qubits.
        assert len(paulis) == 16
        assert set(paulis) == {
            (a, b) for a in 'IXYZ' for b in 'IXYZ'
        }

    def test_measured_qubits_restricted_to_i_and_z(self):
        paulis = generate_n_qubit_paulis([0, 1], measured_qubits={1})
        # 4 (unmeasured qubit 0) x 2 (measured qubit 1).
        assert len(paulis) == 8
        assert ('X', 'Z') in paulis
        assert ('X', 'I') in paulis
        assert ('X', 'Y') not in paulis  # qubit 1 can't be Y when measured


class TestGenerateNQubitPaulisUpToWeightK:

    def test_weight_one_excludes_identity(self):
        paulis = generate_n_qubit_paulis_up_to_weight_k([0, 1], weight_k=1)
        # Weight 0 (the all-identity string) is never generated.
        assert ('I', 'I') not in paulis
        assert set(paulis) == {
            ('X', 'I'), ('Y', 'I'), ('Z', 'I'),
            ('I', 'X'), ('I', 'Y'), ('I', 'Z'),
        }

    def test_measured_qubit_restricted_to_z_at_weight_one(self):
        paulis = generate_n_qubit_paulis_up_to_weight_k(
            [0, 1], measured_qubits={1}, weight_k=1
        )
        assert set(paulis) == {
            ('X', 'I'), ('Y', 'I'), ('Z', 'I'), ('I', 'Z'),
        }

    def test_weight_two_adds_all_pairs(self):
        paulis = generate_n_qubit_paulis_up_to_weight_k([0, 1], weight_k=2)
        # 6 weight-1 strings plus 9 (3x3) weight-2 strings.
        assert len(paulis) == 15
        assert ('X', 'Z') in paulis
        assert all(sum(op != 'I' for op in p) <= 2 for p in paulis)

    def test_connectivity_restricts_weight_two_to_connected_pairs(self):
        # Qubit 2 is disconnected from 0 and 1, so no weight-2 string
        # may touch both qubit 2 and another qubit.
        paulis = generate_n_qubit_paulis_up_to_weight_k(
            [0, 1, 2], weight_k=2, connectivity=[(0, 1)]
        )
        for p in paulis:
            if sum(op != 'I' for op in p) == 2:
                assert p[2] == 'I'
        # 9 weight-1 strings (3 qubits x 3 letters) plus 9 weight-2
        # strings (only from the connected (0, 1) pair).
        assert len(paulis) == 18


class TestGenerateRandomNQubitPaulis:

    def test_returns_requested_count_and_length(self):
        paulis = generate_random_n_qubit_paulis([0, 1, 2], n_random_paulis=10)
        assert len(paulis) == 10
        assert all(len(p) == 3 for p in paulis)

    def test_measured_qubits_never_sampled_as_x_or_y(self):
        paulis = generate_random_n_qubit_paulis(
            [0, 1], measured_qubits={1}, n_random_paulis=200
        )
        assert all(p[1] in ('I', 'Z') for p in paulis)
        assert all(p[0] in ('I', 'X', 'Y', 'Z') for p in paulis)


class TestPauliMeasurementGrouping:

    def test_commuting_paulis_are_combined_into_one_group(self):
        groups = generate_n_qubit_pauli_measurement_groups(
            [('X', 'I'), ('I', 'X')]
        )
        assert len(groups) == 1
        assert groups[0].measurement_basis == ('X', 'X')
        assert groups[0].paulis == [('X', 'I'), ('I', 'X')]

    def test_non_commuting_paulis_get_separate_groups(self):
        groups = generate_n_qubit_pauli_measurement_groups([('X',), ('Z',)])
        assert len(groups) == 2
        bases = {g.measurement_basis for g in groups}
        assert bases == {('X',), ('Z',)}

    def test_measurement_map_keys_are_the_combined_bases(self):
        mapping = generate_n_qubit_pauli_measurement_map(
            [('X', 'I'), ('I', 'X')]
        )
        assert mapping == {('X', 'X'): [('X', 'I'), ('I', 'X')]}


class TestCombineQwcPaulis:

    def test_combines_into_single_basis(self):
        assert _combine_qwc_paulis([('X', 'I'), ('I', 'X')]) == ('X', 'X')

    def test_all_identity_positions_stay_identity(self):
        assert _combine_qwc_paulis([('I', 'I'), ('I', 'I')]) == ('I', 'I')

    def test_non_commuting_paulis_raise(self):
        with pytest.raises(ValueError):
            _combine_qwc_paulis([('X',), ('Z',)])


class TestIsConnectedSubset:

    def test_single_qubit_is_trivially_connected(self):
        assert _is_connected_subset([0], {}) is True

    def test_empty_subset_is_trivially_connected(self):
        assert _is_connected_subset([], {}) is True

    def test_indirectly_linked_qubits_are_not_connected(self):
        # 0-1-2 path: 0 and 2 are only connected via 1, which is
        # excluded from this subset, so the induced subgraph on
        # {0, 2} alone is disconnected.
        adjacency = {0: {1}, 1: {0, 2}, 2: {1}}
        assert _is_connected_subset([0, 2], adjacency) is False

    def test_full_path_is_connected(self):
        adjacency = {0: {1}, 1: {0, 2}, 2: {1}}
        assert _is_connected_subset([0, 1, 2], adjacency) is True

    def test_disconnected_component_is_not_connected(self):
        # Only 0-1 is an edge; 2 is isolated.
        adjacency = {0: {1}, 1: {0}}
        assert _is_connected_subset([0, 1, 2], adjacency) is False


class TestQwcCompatible:

    def test_disjoint_support_is_compatible(self):
        assert _qwc_compatible(('X', 'I'), ('I', 'X')) is True

    def test_identical_paulis_are_compatible(self):
        assert _qwc_compatible(('X', 'X'), ('X', 'X')) is True

    def test_differing_non_identity_ops_are_incompatible(self):
        assert _qwc_compatible(('X', 'Y'), ('X', 'Z')) is False

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            _qwc_compatible(('X',), ('X', 'Y'))
