"""Unit tests for qcal.calibration.cz.calculate_parametric_cz_frequency.

This function writes directly into the passed-in Config object and
calls config.save() when Settings.save_data is enabled, matching the
calibration classes' own final()/save() gating. Settings.save_data
defaults to False, so this wouldn't touch disk in these tests anyway,
but every test here still patches Config.save to a no-op as a belt-
and-suspenders guard against ever overwriting the real example
config.yaml on disk (e.g. if save_data is left enabled by an earlier
test). Config uses __slots__, so the patch has to go on the class, not
the instance (see test_calibration.py's TestFinalSaving for the same
pattern).
"""
import pytest

import qcal.settings as settings
from qcal.calibration.cz import calculate_parametric_cz_frequency
from qcal.config import Config


@pytest.fixture(autouse=True)
def _no_config_save(monkeypatch):
    monkeypatch.setattr(Config, 'save', lambda self, *a, **k: None)


@pytest.fixture(autouse=True)
def _restore_save_data():
    original = settings.Settings.save_data
    yield
    settings.Settings.save_data = original


class TestCalculateParametricCzFrequency:

    def test_invalid_intermediate_state_raises(self, config):
        with pytest.raises(ValueError):
            calculate_parametric_cz_frequency(
                config, [(0, 1)], intermediate_state='bogus'
            )

    def test_02_intermediate_state_computes_correct_frequency(
        self, config
    ):
        pair = (0, 1)
        freq_11 = (
            config['single_qubit/0/GE/freq']
            + config['single_qubit/1/GE/freq']
        )
        freq_02 = (
            config['single_qubit/1/GE/freq']
            + config['single_qubit/1/EF/freq']
        )
        calculate_parametric_cz_frequency(
            config, [pair], intermediate_state='02'
        )
        assert config[f'two_qubit/{pair}/CZ/freq'] == pytest.approx(
            abs(freq_02 - freq_11)
        )

    def test_20_intermediate_state_computes_correct_frequency(
        self, config
    ):
        pair = (0, 1)
        freq_11 = (
            config['single_qubit/0/GE/freq']
            + config['single_qubit/1/GE/freq']
        )
        freq_20 = (
            config['single_qubit/0/GE/freq']
            + config['single_qubit/0/EF/freq']
        )
        calculate_parametric_cz_frequency(
            config, [pair], intermediate_state='20'
        )
        assert config[f'two_qubit/{pair}/CZ/freq'] == pytest.approx(
            abs(freq_20 - freq_11)
        )

    def test_default_intermediate_state_is_02(self, config):
        pair = (0, 1)
        calculate_parametric_cz_frequency(config, [pair])
        default_freq = config[f'two_qubit/{pair}/CZ/freq']

        calculate_parametric_cz_frequency(
            config, [pair], intermediate_state='02'
        )
        explicit_02_freq = config[f'two_qubit/{pair}/CZ/freq']

        assert default_freq == pytest.approx(explicit_02_freq)

    def test_handles_multiple_qubit_pairs(self, config):
        pairs = [(0, 1), (2, 3)]
        calculate_parametric_cz_frequency(config, pairs)
        for pair in pairs:
            assert config[f'two_qubit/{pair}/CZ/freq'] is not None


class TestSaveGating:

    def test_saves_config_when_save_data_enabled(self, config, monkeypatch):
        saved = []
        monkeypatch.setattr(
            Config, 'save', lambda self, *a, **k: saved.append(True)
        )
        settings.Settings.save_data = True
        calculate_parametric_cz_frequency(config, [(0, 1)])
        assert saved == [True]

    def test_does_not_save_config_when_save_data_disabled(
        self, config, monkeypatch
    ):
        saved = []
        monkeypatch.setattr(
            Config, 'save', lambda self, *a, **k: saved.append(True)
        )
        settings.Settings.save_data = False
        calculate_parametric_cz_frequency(config, [(0, 1)])
        assert saved == []
