"""Unit tests for qcal.plotting.graphs."""
import networkx as nx
import plotly.graph_objects as go
import pytest

from qcal.circuit import Barrier, Circuit, Cycle
from qcal.config import Config
from qcal.gates.single_qubit import X90, Meas
from qcal.gates.two_qubit import CZ
from qcal.plotting.graphs import (
    draw_circuit,
    draw_DAG,
    draw_qpu,
    format_gate_text,
    format_qubit_text,
)

EXAMPLE_CONFIG_PATH = 'examples/config/config.yaml'


@pytest.fixture
def captured_figure(monkeypatch):
    captured = []
    monkeypatch.setattr(
        go.Figure, 'show', lambda self, *a, **kw: captured.append(self)
    )
    return captured


class TestFormatGateText:

    def test_includes_name_qubits_and_subspace(self):
        text = format_gate_text(X90(0))

        assert 'Name: X90' in text
        assert 'Qubits: (0,)' in text
        assert 'Subspace: GE' in text

    def test_uses_unitary_when_available(self):
        text = format_gate_text(X90(0))
        assert 'Unitary:' in text

    def test_falls_back_to_matrix_without_unitary(self):
        # Meas has no `unitary` property, only `matrix`.
        text = format_gate_text(Meas(0))
        assert 'Matrix:' in text
        assert 'Unitary:' not in text

    def test_includes_gate_params(self):
        text = format_gate_text(X90(0))
        assert 'angle:' in text
        assert 'axis:' in text


class TestFormatQubitText:

    def test_flat_keys_are_capitalized(self):
        text = format_qubit_text({'freq': 5e9})
        assert text == 'Freq: 5000000000.0<br>'

    def test_nested_dict_is_flattened_one_level(self):
        text = format_qubit_text({'GE': {'freq': 5e9, 'T1': 5e-5}})
        assert text == 'GE:<br>  Freq: 5000000000.0<br>  T1: 5e-05<br>'

    def test_lists_and_doubly_nested_dicts_are_skipped(self):
        text = format_qubit_text(
            {'GE': {'tags': ['a', 'b'], 'nested': {'x': 1}}}
        )
        assert text == 'GE:<br>'


class TestDrawCircuit:

    def _circuit(self):
        return Circuit([
            Cycle({X90(0), X90(1)}),
            Cycle({CZ(0, 1)}),
            Barrier((0, 1)),
            Cycle({Meas(0), Meas(1)}),
        ])

    def test_show_false_returns_figure_instead_of_displaying(
        self, captured_figure
    ):
        fig = draw_circuit(self._circuit(), show=False)

        assert fig is not None
        assert captured_figure == []

    def test_node_positions_match_cycle_and_qubit_index(self):
        fig = draw_circuit(self._circuit(), show=False)

        node_trace = fig.data[-1]
        assert node_trace.x == (0, 0, 1, 1, 2, 2)
        assert node_trace.y == (0, 1, 0, 1, 0, 1)

    def test_barrier_shifts_later_cycles_and_adds_a_vline(self):
        fig = draw_circuit(self._circuit(), show=False)

        node_trace = fig.data[-1]
        # Meas cycle comes after the barrier, so its column index is
        # shifted down by one (2, not 3).
        assert node_trace.x[-2:] == (2, 2)
        assert len(fig.layout.shapes) == 1

    def test_gate_symbols_reflect_gate_type(self):
        fig = draw_circuit(self._circuit(), show=False)

        node_trace = fig.data[-1]
        assert node_trace.marker.symbol == (
            'square', 'square', 'circle', 'circle',
            'triangle-right', 'triangle-right',
        )

    def test_show_true_calls_figure_show(self, captured_figure):
        draw_circuit(self._circuit(), show=True)

        assert len(captured_figure) == 1


class TestDrawDAG:

    def test_node_text_includes_dependency_count(self, captured_figure):
        G = nx.DiGraph()
        G.add_edge('a', 'b')
        G.add_edge('a', 'c')

        draw_DAG(G)

        node_trace = captured_figure[0].data[1]
        assert node_trace.text == (
            'a<br>Dependencies: 2',
            'b<br>Dependencies: 0',
            'c<br>Dependencies: 0',
        )

    def test_node_labels_trace_lists_all_nodes(self, captured_figure):
        G = nx.DiGraph()
        G.add_edge('a', 'b')

        draw_DAG(G)

        labels_trace = captured_figure[0].data[2]
        assert set(labels_trace.text) == {'a', 'b'}


class TestDrawQpu:

    @pytest.fixture
    def config(self):
        return Config(EXAMPLE_CONFIG_PATH)

    def test_one_label_per_qubit(self, config, captured_figure):
        draw_qpu(config)

        labels_trace = captured_figure[0].data[2]
        assert labels_trace.text == tuple(
            f'Q{q}' for q in config.qubits
        )

    def test_node_hover_text_includes_qubit_params(
        self, config, captured_figure
    ):
        draw_qpu(config)

        node_trace = captured_figure[0].data[1]
        assert 'GE:' in node_trace.text[0]
        assert f'Freq: {config["single_qubit/0/GE/freq"]}' in (
            node_trace.text[0]
        )

    def test_middle_node_hover_text_includes_pair_params(
        self, config, captured_figure
    ):
        draw_qpu(config)

        middle_trace = captured_figure[0].data[3]
        assert len(middle_trace.text) == len(config.qubit_pairs)
        assert 'CZ:' in middle_trace.text[0]
