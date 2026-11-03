"""Unit tests for qcal.plotting.error_rates.plot_error_rates."""
import plotly.graph_objects as go
import pytest

from qcal.plotting.error_rates import plot_error_rates


@pytest.fixture
def captured_figure(monkeypatch):
    captured = []
    monkeypatch.setattr(
        go.Figure, 'show', lambda self, *a, **kw: captured.append(self)
    )
    return captured


class TestPlotErrorRates:

    def test_trace_reflects_values_and_uncertainties(self, captured_figure):
        plot_error_rates({0: 0.01, 1: 0.02}, {0: 0.001, 1: 0.002})

        trace = captured_figure[0].data[0]
        assert trace.x == ('0', '1')
        assert trace.y == (0.01, 0.02)
        assert trace.error_y.array == (0.001, 0.002)

    def test_qubit_labels_are_sorted_regardless_of_dict_order(
        self, captured_figure
    ):
        plot_error_rates({1: 0.02, 0: 0.01}, {1: 0.002, 0: 0.001})

        trace = captured_figure[0].data[0]
        assert trace.x == ('0', '1')
        assert trace.y == (0.01, 0.02)

    def test_custom_ylabel_is_applied(self, captured_figure):
        plot_error_rates({0: 0.01}, {0: 0.001}, ylabel='Infidelity')

        assert captured_figure[0].layout.yaxis.title.text == 'Infidelity'

    def test_save_path_writes_three_file_formats(
        self, captured_figure, tmp_path
    ):
        save_path = str(tmp_path) + '/'

        plot_error_rates({0: 0.01}, {0: 0.001}, save_path=save_path)

        assert (tmp_path / 'error_rates.png').exists()
        assert (tmp_path / 'error_rates.pdf').exists()
        assert (tmp_path / 'error_rates.svg').exists()
