"""Unit tests for qcal.plotting.cdf.CDF.

CDF() builds a plotly figure and calls fig.show(), returning None, so
these tests monkeypatch plotly.graph_objects.Figure.show to intercept
and inspect the constructed figure instead of actually rendering it.
"""
import pandas as pd
import plotly.graph_objects as go
import pytest

from qcal.plotting.cdf import CDF


@pytest.fixture
def captured_figure(monkeypatch):
    captured = []
    monkeypatch.setattr(
        go.Figure, 'show', lambda self, *a, **kw: captured.append(self)
    )
    return captured


class TestCDF:

    def test_builds_a_scatter_and_histogram_trace_per_group(
        self, captured_figure
    ):
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 2, 3],
            'group': ['a', 'a', 'a', 'b', 'b', 'b', 'b'],
        })

        CDF(df, x='value', color='group')

        fig = captured_figure[0]
        trace_types = [tr.type for tr in fig.data]
        assert trace_types.count('scatter') == 2
        assert trace_types.count('histogram') == 2

    def test_yaxis_labeled_cumulative_probability(self, captured_figure):
        df = pd.DataFrame({'value': [1, 2, 3], 'group': ['a', 'a', 'a']})

        CDF(df, x='value', color='group')

        fig = captured_figure[0]
        assert fig.layout.yaxis.title.text == 'Cumulative Probability'
