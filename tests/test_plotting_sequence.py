"""Unit tests for qcal.plotting.sequence.{plot_pulse,plot_mock_sequence}."""
import numpy as np
import plotly.graph_objects as go
import pytest

from qcal.circuit import Circuit, Cycle
from qcal.config import Config
from qcal.gates.single_qubit import X90, Meas, Rz
from qcal.gates.two_qubit import CZ
from qcal.plotting.sequence import plot_mock_sequence, plot_pulse
from qcal.units import ns

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


class TestPlotPulse:

    def _envelope(self, duration, n=50):
        t = np.linspace(0, duration, n)
        return (
            np.cos(2 * np.pi * t / duration)
            + 1j * 0.5 * np.sin(2 * np.pi * t / duration)
        ).astype(np.complex64)

    def test_i_and_q_traces_match_envelope(self, config, captured_figure):
        duration = 1e-7
        envelope = self._envelope(duration)

        plot_pulse(config, pulse_duration=duration, pulse_envelope=envelope)

        fig = captured_figure[0]
        i_trace, q_trace = fig.data[0], fig.data[1]
        np.testing.assert_allclose(
            np.array(i_trace.y), np.real(envelope), atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(q_trace.y), np.imag(envelope), atol=1e-6
        )

    def test_time_axis_is_in_nanoseconds(self, config, captured_figure):
        duration = 1e-7
        envelope = self._envelope(duration)
        t = np.linspace(0, duration, len(envelope))

        plot_pulse(config, pulse_duration=duration, pulse_envelope=envelope)

        np.testing.assert_allclose(
            np.array(captured_figure[0].data[0].x), t / ns, atol=1e-6
        )

    def test_no_qubit_lines_without_pulse_qubit(self, config, captured_figure):
        envelope = self._envelope(1e-7)

        plot_pulse(config, pulse_duration=1e-7, pulse_envelope=envelope)

        assert len(captured_figure[0].layout.shapes) == 0

    def test_pulse_qubit_and_neighbors_add_frequency_lines(
        self, config, captured_figure
    ):
        envelope = self._envelope(1e-7)

        plot_pulse(
            config, pulse_duration=1e-7, pulse_envelope=envelope,
            pulse_qubit=0, neighbor_qubits=[1],
        )

        # GE + EF lines for the pulse qubit and each neighbor qubit.
        assert len(captured_figure[0].layout.shapes) == 4


class TestPlotMockSequence:

    def _circuit(self):
        return Circuit([
            Cycle({X90(0), Rz(1, 0.5)}),
            Cycle({CZ(0, 1)}),
            Cycle({Meas(0), Meas(1)}),
        ])

    def test_virtual_z_gate_produces_no_trace(self, captured_figure):
        plot_mock_sequence(self._circuit())

        names = [tr.name for tr in captured_figure[0].data]
        assert not any(n.startswith('Rz') for n in names)

    def test_two_qubit_gate_appears_on_both_qubit_rows(self, captured_figure):
        plot_mock_sequence(self._circuit())

        cz_traces = [
            tr for tr in captured_figure[0].data if tr.name == 'CZ(0, 1)'
        ]
        assert len(cz_traces) == 2
        assert {tr.yaxis for tr in cz_traces} == {'y', 'y2'}

    def test_one_subplot_title_per_qudit(self, captured_figure):
        plot_mock_sequence(self._circuit())

        titles = [a.text for a in captured_figure[0].layout.annotations]
        assert titles == ['Q0', 'Q1']
