"""Unit tests for qcal.post_processing.utils."""
import numpy as np
import pandas as pd
import pytest

from qcal.circuit import Barrier, Circuit, Cycle
from qcal.config import Config
from qcal.gates.single_qubit import MCM, X90, Meas, X
from qcal.gates.two_qubit import CZ
from qcal.post_processing.utils import (
    circuit_measurement_cycles,
    construct_dataframes,
    find_herald_idx,
    reshape_results,
)

EXAMPLE_CONFIG_PATH = 'examples/config/config.yaml'


class TestConstructDataframes:

    def _results(self):
        return [{
            0: np.array([[0, 1], [1, 0]]),
            1: np.array([[1, 1], [0, 0]]),
        }]

    def test_one_dataframe_per_read_from_start_idx(self):
        dfs = construct_dataframes(self._results(), start_idx=0)

        assert len(dfs) == 1  # one circuit
        assert len(dfs[0]) == 2  # two reads

    def test_dataframe_columns_match_per_shot_qubit_values(self):
        dfs = construct_dataframes(self._results(), start_idx=0)

        pd.testing.assert_frame_equal(
            dfs[0][0], pd.DataFrame({0: [0, 1], 1: [1, 0]})
        )
        pd.testing.assert_frame_equal(
            dfs[0][1], pd.DataFrame({0: [1, 0], 1: [1, 0]})
        )

    def test_start_idx_skips_earlier_reads(self):
        dfs = construct_dataframes(self._results(), start_idx=1)

        assert len(dfs[0]) == 1
        pd.testing.assert_frame_equal(
            dfs[0][0], pd.DataFrame({0: [1, 0], 1: [1, 0]})
        )

    def test_multiple_circuits_are_independent(self):
        results = [
            {0: np.array([[0, 1]]), 1: np.array([[1, 0]])},
            {0: np.array([[1, 1]]), 1: np.array([[0, 0]])},
        ]

        dfs = construct_dataframes(results, start_idx=0)

        assert len(dfs) == 2
        pd.testing.assert_frame_equal(
            dfs[1][0], pd.DataFrame({0: [1], 1: [0]})
        )


class TestFindHeraldIdx:

    def test_zero_when_active_reset_disabled(self):
        config = Config(EXAMPLE_CONFIG_PATH)
        assert config.parameters['reset']['active']['enable'] is False

        assert find_herald_idx(config) == 0

    def test_offsets_by_n_resets_when_active_reset_enabled(self):
        config = Config(EXAMPLE_CONFIG_PATH)
        config.parameters['reset']['active']['enable'] = True
        config.parameters['reset']['active']['n_resets'] = 3

        assert find_herald_idx(config) == 3


class TestReshapeResults:

    def test_reshapes_into_one_dict_per_circuit(self):
        results = {
            0: [np.array([[0], [1]]), np.array([[1], [0]])],
            1: [np.array([[1], [1]]), np.array([[0], [1]])],
        }

        reshaped = reshape_results(results)

        assert len(reshaped) == 2
        np.testing.assert_array_equal(reshaped[0][0], results[0][0])
        np.testing.assert_array_equal(reshaped[0][1], results[1][0])
        np.testing.assert_array_equal(reshaped[1][0], results[0][1])
        np.testing.assert_array_equal(reshaped[1][1], results[1][1])

    def test_excludes_qubits_with_none_for_a_given_circuit(self):
        results = {
            0: [np.array([[0], [1]]), np.array([[1], [0]])],
            1: [np.array([[1], [1]]), None],
        }

        reshaped = reshape_results(results)

        assert set(reshaped[0].keys()) == {0, 1}
        assert set(reshaped[1].keys()) == {0}


class TestCircuitMeasurementCycles:

    def test_no_measurements(self):
        circuit = Circuit([Cycle({X(0)}), Cycle({CZ(0, 1)})])
        assert circuit_measurement_cycles(circuit) == []

    def test_single_final_measurement_cycle(self):
        circuit = Circuit([
            Cycle({X90(0), X90(1)}),
            Cycle({Meas(0), Meas(1)}),
        ])
        assert circuit_measurement_cycles(circuit) == [{0: 0, 1: 0}]

    def test_ancilla_measured_mid_circuit_and_data_at_end(self):
        # Matches the example in the docstring.
        circuit = Circuit([
            Cycle({X(0)}),
            Cycle({MCM(1)}),
            Cycle({MCM(1)}),
            Cycle({Meas(0)}),
        ])
        assert circuit_measurement_cycles(circuit) == [
            {1: 0}, {1: 1}, {0: 0}
        ]

    def test_read_indices_count_per_qubit(self):
        circuit = Circuit([
            Cycle({MCM(0)}),
            Cycle({MCM((0, 1))}),
            Cycle({MCM(1)}),
            Cycle({Meas(0), Meas(1), Meas(2)}),
        ])
        assert circuit_measurement_cycles(circuit) == [
            {0: 0},
            {0: 1, 1: 0},
            {1: 1},
            {0: 2, 1: 2, 2: 0},
        ]

    def test_barriers_and_unitary_cycles_are_skipped(self):
        circuit = Circuit([
            Cycle({MCM(1)}),
            Barrier((0, 1)),
            Cycle({X(0), X(1)}),
            Barrier(),
            Cycle({Meas(0), Meas(1)}),
        ])
        assert circuit_measurement_cycles(circuit) == [
            {1: 0}, {0: 0, 1: 1}
        ]

    def test_measure_appends_a_final_cycle_on_all_qubits(self):
        circuit = Circuit([Cycle({CZ(0, 2)}), Cycle({MCM(2)})])
        circuit.measure()
        assert circuit_measurement_cycles(circuit) == [
            {2: 0}, {0: 0, 2: 1}
        ]

    def test_read_counts_match_number_of_measurement_gates(self):
        n_rounds = 5
        round_ = [Cycle({CZ(0, 1), CZ(2, 3)}), Cycle({MCM((1, 3))})]
        circuit = Circuit(round_ * n_rounds)
        circuit.measure()

        cycles = circuit_measurement_cycles(circuit)
        assert len(cycles) == n_rounds + 1
        assert [c[1] for c in cycles] == list(range(n_rounds + 1))
        assert [c[3] for c in cycles] == list(range(n_rounds + 1))
        assert cycles[-1][0] == 0 and cycles[-1][2] == 0

    def test_aligns_construct_dataframes(self):
        # Ancilla 1: herald + 2 MCMs; data 0: herald + final read.
        circuit = Circuit([
            Cycle({MCM(1)}),
            Cycle({MCM(1)}),
            Cycle({Meas(0)}),
        ])
        results = [{
            0: np.array([[0, 1], [0, 0]]),
            1: np.array([[0, 1, 0], [0, 0, 1]]),
        }]
        dfs = construct_dataframes(
            results, 1, [circuit_measurement_cycles(circuit)]
        )[0]

        assert [list(df.columns) for df in dfs] == [[1], [1], [0]]
        assert dfs[0][1].tolist() == [1, 0]
        assert dfs[1][1].tolist() == [0, 1]
        assert dfs[2][0].tolist() == [1, 0]

    def test_mismatched_reads_raise_in_construct_dataframes(self):
        circuit = Circuit([Cycle({MCM(1)}), Cycle({Meas(1)})])
        results = [{1: np.array([[0]])}]  # one read short
        with pytest.raises(ValueError, match='measurement cycles'):
            construct_dataframes(
                results, 0, [circuit_measurement_cycles(circuit)]
            )
