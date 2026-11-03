"""Unit tests for qcal.post_processing.post_process.PostProcessor."""
import numpy as np

from qcal.config import Config
from qcal.post_processing.post_process import PostProcessor

EXAMPLE_CONFIG_PATH = 'examples/config/config.yaml'


def _config(herald=True, esp=False):
    config = Config(EXAMPLE_CONFIG_PATH)
    config['readout/herald'] = herald
    config['readout/esp/enable'] = esp
    return config


class TestConstruction:

    def test_qubits_are_sorted_and_deduplicated(self):
        results = [
            {1: np.zeros((2, 1)), 0: np.zeros((2, 1))},
            {0: np.zeros((2, 1)), 2: np.zeros((2, 1))},
        ]
        pp = PostProcessor(config=_config(), results=results)

        assert pp.qubits == [0, 1, 2]

    def test_herald_idx_from_config(self):
        config = _config()
        config.parameters['reset']['active']['enable'] = True
        config.parameters['reset']['active']['n_resets'] = 2

        pp = PostProcessor(config=config, results=[])

        assert pp.herald_idx == 2

    def test_default_passes(self):
        pp = PostProcessor(config=_config(), results=[])

        names = [p.__name__ for p in pp.passes]
        assert names == [
            'discard_heralded_shots', 'relabel_esp', 'compute_counts',
        ]


class TestRun:

    def test_herald_discards_failing_shots_before_counting(self):
        # Herald idx 0; shot heralds pass only if all qubits read 0.
        results = [{
            0: np.array([[0, 1], [0, 0], [1, 1], [0, 1]]),
            1: np.array([[0, 0], [0, 1], [0, 0], [1, 0]]),
        }]
        pp = PostProcessor(config=_config(herald=True), results=results)

        pp.run()

        # Shots 0, 1 pass heralding; their terminal values are
        # (1, 0) and (0, 1) respectively.
        assert pp.tm_results == [{'10': 1, '01': 1}]
        assert pp.mcm_results == [[]]

    def test_esp_relabels_before_counting(self):
        results = [{
            0: np.array([[0], [2], [1]]),
            1: np.array([[0], [1], [2]]),
        }]
        pp = PostProcessor(
            config=_config(herald=False, esp=True), results=results
        )

        pp.run()

        assert pp.tm_results == [{'00': 1, '11': 2}]

    def test_mid_circuit_and_terminal_measurements_are_split(self):
        # 3 reads: two mid-circuit measurements, then a terminal one.
        results = [{
            0: np.array([[0, 1, 0], [1, 0, 1]]),
            1: np.array([[1, 0, 0], [0, 1, 1]]),
        }]
        pp = PostProcessor(
            config=_config(herald=False, esp=False), results=results
        )

        pp.run()

        assert pp.mcm_results == [[
            {'01': 1, '10': 1}, {'10': 1, '01': 1},
        ]]
        assert pp.tm_results == [{'00': 1, '11': 1}]

    def test_multiple_circuits_are_processed_independently(self):
        results = [
            {0: np.array([[0, 1]]), 1: np.array([[0, 0]])},
            {0: np.array([[0, 0]]), 1: np.array([[0, 1]])},
        ]
        pp = PostProcessor(config=_config(herald=True), results=results)

        pp.run()

        assert pp.tm_results == [{'10': 1}, {'01': 1}]
