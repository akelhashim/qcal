"""Unit tests for qcal.post_processing.utils."""
import numpy as np
import pandas as pd

from qcal.config import Config
from qcal.post_processing.utils import (
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
