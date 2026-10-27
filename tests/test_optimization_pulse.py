"""Unit tests for qcal.optimization.pulse.optimize_FAST_DRAG.

There's no other caller of optimize_FAST_DRAG anywhere in the
codebase to anchor the expected pulse_envelope_func calling convention
against, so this fake envelope function is built directly from how
optimize_FAST_DRAG itself calls it: pulse_envelope_func(config,
config[pulse_param]). It returns a baseline envelope plus a synthetic
leakage term at the EF transition frequency whose amplitude is a known
function of N and weights[0], so the N/weight values that minimize it
are known in advance and can be checked exactly.
"""
import shutil

import numpy as np
import pytest

import qcal.settings as settings
from qcal.config import Config
from qcal.optimization.pulse import optimize_FAST_DRAG

EXAMPLE_CONFIG_PATH = 'examples/config/config.yaml'
PULSE_PARAM = 'single_qubit/0/GE/FAST_DRAG'
TARGET_N = 6
TARGET_W0 = 3.0


@pytest.fixture
def config(tmp_path):
    path = tmp_path / 'config.yaml'
    shutil.copy(EXAMPLE_CONFIG_PATH, path)
    cfg = Config(str(path))
    cfg._parameters['single_qubit'][0]['GE']['FAST_DRAG'] = {
        'length': 100e-9,
        'kwargs': {'N': 2, 'weights': [1.0]},
    }
    return cfg


@pytest.fixture(autouse=True)
def restore_save_data():
    original = settings.Settings.save_data
    yield
    settings.Settings.save_data = original


def fake_pulse_envelope(config, pulse_dict):
    freq_GE = config['single_qubit/0/GE/freq']
    freq_EF = config['single_qubit/0/EF/freq']
    length = pulse_dict['length']
    N = pulse_dict['kwargs']['N']
    weights = pulse_dict['kwargs']['weights']

    t = np.linspace(0, length, 200)
    baseline = np.exp(-((t - length / 2) ** 2) / (2 * (length / 6) ** 2))
    leak_amp = 0.01 * (abs(N - TARGET_N) + abs(weights[0] - TARGET_W0))
    leak_freq = freq_EF - freq_GE
    leakage = leak_amp * np.exp(1j * 2 * np.pi * leak_freq * t)

    return (baseline + leakage).astype(np.complex64)


class TestOptimizeFastDrag:

    def test_selects_n_and_weight_that_minimize_ef_leakage(self, config):
        optimize_FAST_DRAG(config, PULSE_PARAM, fake_pulse_envelope)

        assert config[f'{PULSE_PARAM}/kwargs/N'] == TARGET_N
        assert config[f'{PULSE_PARAM}/kwargs/weights'] == [TARGET_W0]

    def test_does_not_save_config_by_default(self, config, monkeypatch):
        settings.Settings.save_data = False
        save_calls = []
        monkeypatch.setattr(
            Config, 'save', lambda self, *a, **kw: save_calls.append(1)
        )

        optimize_FAST_DRAG(config, PULSE_PARAM, fake_pulse_envelope)

        assert save_calls == []

    def test_saves_config_when_save_data_is_enabled(self, config, monkeypatch):
        settings.Settings.save_data = True
        save_calls = []
        monkeypatch.setattr(
            Config, 'save', lambda self, *a, **kw: save_calls.append(1)
        )

        optimize_FAST_DRAG(config, PULSE_PARAM, fake_pulse_envelope)

        assert save_calls == [1]

    def test_custom_param_sweep_is_respected(self, config):
        # Restrict the N sweep so the designed target (6) is excluded;
        # the function should fall back to the best value within the
        # custom sweep range instead.
        optimize_FAST_DRAG(
            config, PULSE_PARAM, fake_pulse_envelope,
            param_sweep={
                'N': np.arange(2, 5).astype(int),
                'w': [0.1, 0.3, 1.0, 2.0, 3.0],
            },
        )

        assert config[f'{PULSE_PARAM}/kwargs/N'] == 4
        assert config[f'{PULSE_PARAM}/kwargs/weights'] == [TARGET_W0]
