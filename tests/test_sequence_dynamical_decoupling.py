"""Unit tests for qcal.sequence.dynamical_decoupling."""
import pytest

from qcal.circuit import Barrier, Cycle
from qcal.sequence.dynamical_decoupling import DD_SEQUENCES, XX_N, XY_N
from qcal.units import ns

TOTAL_TIME = 500 * ns
GATE_TIME = 20 * ns


def _gate_names(circuit):
    names = set()
    for cycle in circuit:
        if isinstance(cycle, Cycle):
            names.update(gate.name for gate in cycle)
    return names


class TestXYN:

    def test_n_pulses_must_be_a_multiple_of_four(self):
        with pytest.raises(ValueError, match='multiple of 4'):
            XY_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=5)

    def test_default_n_pulses_runs_without_error(self):
        circuit = XY_N([0, 1], TOTAL_TIME, GATE_TIME)
        assert len(circuit) > 0

    def test_syncopated_for_adjacent_even_odd_labels(self):
        circuit = XY_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=4)
        assert len(circuit) == 25

    def test_not_syncopated_when_all_labels_share_parity(self):
        # Both qubits are odd-labeled, so XY_N has no "even" partner
        # and falls back to the non-syncopated (single-partition)
        # sequence, same as a single qubit would.
        circuit = XY_N([1, 3], TOTAL_TIME, GATE_TIME, n_pulses=4)
        single = XY_N([1], TOTAL_TIME, GATE_TIME, n_pulses=4)
        assert len(circuit) == len(single)

    def test_single_qubit_is_not_syncopated(self):
        circuit = XY_N([0], TOTAL_TIME, GATE_TIME, n_pulses=4)
        assert len(circuit) == 14

    def test_scales_linearly_with_n_pulses(self):
        c4 = XY_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=4)
        c8 = XY_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=8)
        assert len(c8) == 2 * len(c4)

    def test_includes_all_requested_qubits(self):
        circuit = XY_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=4)
        assert circuit._qubits == {0, 1}

    def test_ends_with_a_barrier_over_all_qubits(self):
        circuit = XY_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=4)
        last_cycle = circuit[-1]
        assert isinstance(last_cycle, Barrier)
        assert set(last_cycle.qubits) == {0, 1}

    def test_only_uses_x90_y90_and_idle_gates(self):
        circuit = XY_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=4)
        assert _gate_names(circuit) <= {'X90', 'Y90', 'Idle'}

    def test_subspace_is_propagated_to_gates(self):
        circuit = XY_N(
            [0, 1], TOTAL_TIME, GATE_TIME, n_pulses=4, subspace='EF'
        )
        pulse_gates = [
            gate for cycle in circuit if isinstance(cycle, Cycle)
            for gate in cycle if gate.name in ('X90', 'Y90')
        ]
        assert pulse_gates
        assert all(gate.subspace == 'EF' for gate in pulse_gates)

    def test_warns_when_total_time_is_too_small_syncopated(self, caplog):
        with caplog.at_level('WARNING'):
            XY_N([0, 1], 10 * ns, GATE_TIME, n_pulses=4)
        assert 'Syncopated XY4' in caplog.text

    def test_warns_when_total_time_is_too_small_non_syncopated(self, caplog):
        with caplog.at_level('WARNING'):
            XY_N([0], 10 * ns, GATE_TIME, n_pulses=4)
        assert 'XY4' in caplog.text


class TestXXN:

    def test_n_pulses_must_be_a_multiple_of_two(self):
        with pytest.raises(ValueError, match='multiple of 2'):
            XX_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=3)

    def test_default_n_pulses_runs_without_error(self):
        circuit = XX_N([0, 1], TOTAL_TIME, GATE_TIME)
        assert len(circuit) > 0

    def test_syncopated_for_adjacent_even_odd_labels(self):
        circuit = XX_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=2)
        assert len(circuit) == 13

    def test_not_syncopated_when_all_labels_share_parity(self):
        # Both qubits are odd-labeled: XX_N should agree with XY_N
        # that these are not a syncopatable even/odd pair, regardless
        # of their position in the qubits list.
        circuit = XX_N([1, 3], TOTAL_TIME, GATE_TIME, n_pulses=2)
        single = XX_N([1], TOTAL_TIME, GATE_TIME, n_pulses=2)
        assert len(circuit) == len(single)

    def test_single_qubit_is_not_syncopated(self):
        circuit = XX_N([0], TOTAL_TIME, GATE_TIME, n_pulses=2)
        assert len(circuit) == 8

    def test_scales_linearly_with_n_pulses(self):
        c2 = XX_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=2)
        c4 = XX_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=4)
        assert len(c4) == 2 * len(c2)

    def test_includes_all_requested_qubits(self):
        circuit = XX_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=2)
        assert circuit._qubits == {0, 1}

    def test_ends_with_a_barrier_over_all_qubits(self):
        circuit = XX_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=2)
        last_cycle = circuit[-1]
        assert isinstance(last_cycle, Barrier)
        assert set(last_cycle.qubits) == {0, 1}

    def test_only_uses_x90_and_idle_gates(self):
        circuit = XX_N([0, 1], TOTAL_TIME, GATE_TIME, n_pulses=2)
        assert _gate_names(circuit) <= {'X90', 'Idle'}

    def test_subspace_is_propagated_to_gates(self):
        circuit = XX_N(
            [0, 1], TOTAL_TIME, GATE_TIME, n_pulses=2, subspace='EF'
        )
        pulse_gates = [
            gate for cycle in circuit if isinstance(cycle, Cycle)
            for gate in cycle if gate.name == 'X90'
        ]
        assert pulse_gates
        assert all(gate.subspace == 'EF' for gate in pulse_gates)

    def test_warns_when_total_time_is_too_small_syncopated(self, caplog):
        with caplog.at_level('WARNING'):
            XX_N([0, 1], 10 * ns, GATE_TIME, n_pulses=2)
        assert 'Syncopated XX2' in caplog.text

    def test_warns_when_total_time_is_too_small_non_syncopated(self, caplog):
        with caplog.at_level('WARNING'):
            XX_N([0], 10 * ns, GATE_TIME, n_pulses=2)
        assert 'XX2' in caplog.text


class TestDdSequencesRegistry:

    def test_xy_n_is_registered(self):
        assert DD_SEQUENCES['XY_N'] is XY_N

    def test_xx_n_is_registered(self):
        assert DD_SEQUENCES['XX_N'] is XX_N

    def test_unsupported_sequence_returns_sentinel_string(self):
        assert DD_SEQUENCES['not_a_real_sequence'] == (
            'DD sequence not currently supported!'
        )
