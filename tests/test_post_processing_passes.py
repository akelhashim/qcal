"""Unit tests for qcal.post_processing.passes.

Each pass is exercised in isolation by constructing a PostProcessor
with passes=[] (so run() would do nothing), manually calling
_construct_dataframes(), then invoking the pass function directly on
the resulting state.
"""
import numpy as np
import pandas as pd
import pytest

from qcal.config import Config
from qcal.post_processing.passes import (
    compute_conditional_counts,
    compute_counts,
    discard_heralded_shots,
    relabel_esp,
)
from qcal.post_processing.post_process import PostProcessor

EXAMPLE_CONFIG_PATH = 'examples/config/config.yaml'


def _pp(results, herald=False, esp=False):
    config = Config(EXAMPLE_CONFIG_PATH)
    config['readout/herald'] = herald
    config['readout/esp/enable'] = esp
    pp = PostProcessor(config=config, results=results, passes=[])
    pp._construct_dataframes()
    return pp


class TestComputeCounts:

    def test_counts_unique_bitstrings_per_read(self):
        results = [{
            0: np.array([[0], [1], [0]]),
            1: np.array([[1], [1], [0]]),
        }]
        pp = _pp(results)

        compute_counts(pp)

        assert pp._circuit_results == [[{'01': 1, '11': 1, '00': 1}]]

    def test_duplicate_bitstrings_are_aggregated(self):
        results = [{
            0: np.array([[0], [0], [1]]),
            1: np.array([[0], [0], [1]]),
        }]
        pp = _pp(results)

        compute_counts(pp)

        assert pp._circuit_results == [[{'00': 2, '11': 1}]]


class TestDiscardHeraldedShots:

    def test_keeps_only_shots_where_all_qubits_heralded_to_zero(self):
        results = [{
            0: np.array([[0, 1], [0, 0], [1, 1]]),
            1: np.array([[0, 0], [0, 1], [0, 0]]),
        }]
        pp = _pp(results, herald=True)
        before = pp._circuit_results[0][0].copy()

        discard_heralded_shots(pp)

        after = pp._circuit_results[0][0]
        # Shot 2 heralds q0=1 -> discarded; shots 0, 1 herald all-zero.
        assert len(after) == 2
        assert list(after.index) == [0, 1]
        pd.testing.assert_frame_equal(after, before.iloc[[0, 1]])

    def test_no_op_when_heralding_disabled(self):
        results = [{
            0: np.array([[0, 1], [1, 0]]),
            1: np.array([[0, 0], [0, 1]]),
        }]
        pp = _pp(results, herald=False)
        before = [df.copy() for df in pp._circuit_results[0]]

        discard_heralded_shots(pp)

        for df_before, df_after in zip(
            before, pp._circuit_results[0], strict=True
        ):
            pd.testing.assert_frame_equal(df_before, df_after)


class TestRelabelEsp:

    def test_relabels_2s_as_1s_when_enabled(self):
        results = [{
            0: np.array([[0], [2], [1]]),
            1: np.array([[2], [1], [0]]),
        }]
        pp = _pp(results, esp=True)

        relabel_esp(pp)

        df = pp._circuit_results[0][0]
        assert list(df[0]) == [0, 1, 1]
        assert list(df[1]) == [1, 1, 0]

    def test_no_op_when_esp_disabled(self):
        results = [{
            0: np.array([[0], [2], [1]]),
            1: np.array([[2], [1], [0]]),
        }]
        pp = _pp(results, esp=False)

        relabel_esp(pp)

        df = pp._circuit_results[0][0]
        assert list(df[0]) == [0, 2, 1]
        assert list(df[1]) == [2, 1, 0]


class TestComputeConditionalCounts:

    def test_combines_mid_circuit_and_terminal_outcomes(self):
        results = [{
            0: np.array([[0, 1, 1], [1, 0, 0]]),
            1: np.array([[1, 0, 1], [0, 1, 0]]),
        }]
        pp = _pp(results, herald=False)

        compute_conditional_counts(pp)

        assert pp._circuit_results == [[{
            'p0p1:p1p0:11': 1,
            'p1p0:p0p1:00': 1,
        }]]

    def test_relabels_qutrit_mid_circuit_outcomes(self):
        # Regression test: the replace() dict used to mix int keys
        # (0, 1) with string keys ('2', '3'), so outcomes of 2 or 3
        # silently failed to relabel to 'p2'/'p3' and stayed as raw
        # ints in the combined bitstring key.
        results = [{
            0: np.array([[2, 1]]),
            1: np.array([[3, 0]]),
        }]
        pp = _pp(results, herald=False)

        compute_conditional_counts(pp)

        assert pp._circuit_results == [[{'p2p3:10': 1}]]

    def test_unequal_row_counts_raises(self):
        results = [{
            0: np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]),
            1: np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]]),
        }]
        pp = _pp(results, herald=False)
        # Artificially truncate one of the per-read DataFrames.
        pp._circuit_results[0][0] = pp._circuit_results[0][0].iloc[:2]

        with pytest.raises(ValueError, match='same number of rows'):
            compute_conditional_counts(pp)
