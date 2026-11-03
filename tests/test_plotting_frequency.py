"""Unit tests for qcal.plotting.frequency.plot_freq_spectrum."""
import plotly.graph_objects as go
import pytest

from qcal.config import Config
from qcal.plotting.frequency import plot_freq_spectrum
from qcal.units import GHz

EXAMPLE_CONFIG_PATH = 'examples/config/config.yaml'


@pytest.fixture
def config():
    return Config(EXAMPLE_CONFIG_PATH)


@pytest.fixture
def captured_figure(monkeypatch):
    captured = []
    monkeypatch.setattr(
        go.Figure, 'show', lambda self, *a, **kw: captured.append(self)
    )
    return captured


def _trace_x0(fig, name):
    for tr in fig.data:
        if tr.name == name:
            return tr.x[0]
    raise AssertionError(f'no trace named {name!r}')


class TestPlotFreqSpectrum:

    def test_ge_frequencies_are_converted_to_ghz(self, config, captured_figure):
        plot_freq_spectrum(config, qubits=(0, 1), plot_GE=True)

        fig = captured_figure[0]
        assert _trace_x0(fig, 'Q0 GE') == pytest.approx(
            config['single_qubit/0/GE/freq'] / GHz
        )
        assert _trace_x0(fig, 'Q1 GE') == pytest.approx(
            config['single_qubit/1/GE/freq'] / GHz
        )

    def test_ef_frequencies_plotted_only_when_requested(
        self, config, captured_figure
    ):
        plot_freq_spectrum(config, qubits=(0,), plot_GE=True, plot_EF=True)

        fig = captured_figure[0]
        names = {tr.name for tr in fig.data}
        assert 'Q0 EF' in names
        assert _trace_x0(fig, 'Q0 EF') == pytest.approx(
            config['single_qubit/0/EF/freq'] / GHz
        )

    def test_ef_not_plotted_by_default(self, config, captured_figure):
        plot_freq_spectrum(config, qubits=(0,), plot_GE=True)

        names = {tr.name for tr in captured_figure[0].data}
        assert 'Q0 EF' not in names

    def test_readout_frequencies(self, config, captured_figure):
        plot_freq_spectrum(
            config, qubits=(0,), plot_GE=True, plot_readout=True
        )

        fig = captured_figure[0]
        assert _trace_x0(fig, 'R0') == pytest.approx(
            config['readout/0/freq'] / GHz
        )

    def test_two_qubit_frequencies(self, config, captured_figure):
        plot_freq_spectrum(
            config, qubits=(0, 1), qubit_pairs=[(0, 1)],
            plot_GE=True, plot_two_qubit=True,
        )

        fig = captured_figure[0]
        assert _trace_x0(fig, 'CZ (0, 1)') == pytest.approx(
            config['two_qubit/(0, 1)/CZ/freq'] / GHz
        )

    def test_defaults_to_all_config_qubits(self, config, captured_figure):
        plot_freq_spectrum(config, plot_GE=True)

        names = {tr.name for tr in captured_figure[0].data}
        for q in config.qubits:
            assert f'Q{q} GE' in names
