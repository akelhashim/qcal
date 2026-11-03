"""Unit tests for qcal.plotting.graphs."""
import networkx as nx
import plotly.colors as pc
import plotly.graph_objects as go
import pytest

from qcal.circuit import Barrier, Circuit, Cycle
from qcal.compilation.merge import merge_gates
from qcal.config import Config
from qcal.gates.single_qubit import MCM, X90, Meas, Reset, X, Y
from qcal.gates.two_qubit import CZ
from qcal.plotting.graphs import (
    MAX_LABELED_NODES,
    draw_circuit,
    draw_DAG,
    draw_qpu,
    draw_qpu_heatmap,
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


class TestFormatGateTextIncludeMatrix:

    def test_include_matrix_true_is_default(self):
        assert format_gate_text(X90(0), include_matrix=True) == (
            format_gate_text(X90(0))
        )

    def test_include_matrix_false_omits_matrix(self):
        text = format_gate_text(X90(0), include_matrix=False)

        assert 'Unitary:' not in text
        assert 'Matrix:' not in text
        assert 'Name: X90' in text
        assert 'Subspace: GE' in text

    def test_include_matrix_false_omits_non_unitary_matrix(self):
        text = format_gate_text(Meas(0), include_matrix=False)
        assert 'Matrix:' not in text


class TestDrawCircuitPerQubitGates:
    """`is_per_qubit` (local to draw_circuit): Meas/MCM/Reset are drawn
    independently on each qubit, even when spanning several qubits.
    """

    def test_multi_qubit_mcm_is_drawn_on_each_qubit(self):
        circuit = Circuit([
            Cycle({X90(0), X90(1)}),
            Cycle({MCM((0, 1))}),
        ])
        fig = draw_circuit(circuit, show=False)

        node_trace = fig.data[-1]
        assert node_trace.x[-2:] == (1, 1)
        assert sorted(node_trace.y[-2:]) == [0, 1]
        assert node_trace.marker.symbol[-2:] == (
            'triangle-right', 'triangle-right'
        )
        labels = {
            (a['x'], a['y']): a['text'] for a in fig.layout.annotations
            if 'x' in a and a['x'] == 1
        }
        assert labels == {(1, 0): 'M', (1, 1): 'M'}

    def test_multi_qubit_mcm_adds_no_connecting_line(self):
        circuit = Circuit([
            Cycle({X90(0), X90(1)}),
            Cycle({MCM((0, 1))}),
        ])
        fig = draw_circuit(circuit, show=False)

        # One wire per qubit, then the node trace; nothing else.
        assert len(fig.data) == len(circuit.qubits) + 1

    def test_two_qubit_gate_adds_connecting_line(self):
        circuit = Circuit([
            Cycle({X90(0), X90(1)}),
            Cycle({CZ(0, 1)}),
        ])
        fig = draw_circuit(circuit, show=False)

        assert len(fig.data) == len(circuit.qubits) + 2
        assert fig.data[2].x == (1, 1)
        assert sorted(fig.data[2].y) == [0, 1]

    def test_reset_is_drawn_per_qubit_as_bowtie(self):
        circuit = Circuit([Cycle({X90(0)}), Cycle({Reset(0)})])
        fig = draw_circuit(circuit, show=False)

        node_trace = fig.data[-1]
        assert node_trace.marker.symbol[-1] == 'bowtie'
        assert len(fig.data) == len(circuit.qubits) + 1


class TestDrawCircuitLargeCircuits:

    def test_small_circuit_has_gate_labels(self):
        circuit = Circuit([Cycle({X90(0)}), Cycle({Meas(0)})])
        fig = draw_circuit(circuit, show=False)

        texts = [a['text'] for a in fig.layout.annotations]
        assert 'X90' in texts
        assert 'M' in texts

    def test_labels_omitted_above_max_labeled_nodes(self):
        n_cycles = MAX_LABELED_NODES // 2 + 1
        circuit = Circuit(
            [Cycle({X90(0), X90(1)}) for _ in range(n_cycles)]
        )
        assert circuit.circuit_depth * 2 > MAX_LABELED_NODES

        fig = draw_circuit(circuit, show=False)

        texts = [a['text'] for a in fig.layout.annotations]
        assert texts == ['Quantum Circuit']
        assert len(fig.data[-1].x) == 2 * n_cycles


class TestDrawCircuitHoverText:

    def test_merged_gates_on_same_qubit_get_own_hover_text(self):
        # Merged gates are all generic Gates named 'Gate' on qubit 0;
        # only their matrices differ.
        circuit = Circuit([
            Cycle({merge_gates(X(0), Y(0))}),
            Cycle({merge_gates(X90(0), X90(0))}),
        ])
        fig = draw_circuit(circuit, show=False)

        node_text = fig.data[-1].text
        assert node_text[0] == format_gate_text(
            circuit.cycles[0].gates[0], include_matrix=True
        )
        assert node_text[1] == format_gate_text(
            circuit.cycles[1].gates[0], include_matrix=True
        )
        assert node_text[0] != node_text[1]


class TestDrawQpuHeatmap:

    @pytest.fixture
    def config(self):
        return Config(EXAMPLE_CONFIG_PATH)

    @staticmethod
    def _color(frac):
        return pc.sample_colorscale('Tealgrn', frac)[0]

    def test_show_false_returns_figure(self, config, captured_figure):
        fig = draw_qpu_heatmap(config, {0: 1.0}, show=False)

        assert isinstance(fig, go.Figure)
        assert captured_figure == []

    def test_show_true_displays_and_returns_none(
        self, config, captured_figure
    ):
        assert draw_qpu_heatmap(config, {0: 1.0}) is None
        assert len(captured_figure) == 1

    def test_traces_for_values_only(self, config):
        fig = draw_qpu_heatmap(
            config, {0: 1.0, 1: 3.0, (0, 1): 2.0}, show=False
        )

        # 1 edge + nodes + labels + edge midpoints + colorbar.
        assert len(fig.data) == 5
        assert fig.data[2].text == ('Q0', 'Q1')

    def test_colors_scale_between_min_and_max(self, config):
        fig = draw_qpu_heatmap(
            config, {0: 1.0, 1: 3.0, (0, 1): 2.0}, show=False
        )

        edge_trace, node_trace = fig.data[0], fig.data[1]
        assert node_trace.marker.color == (
            self._color(0.0), self._color(1.0)
        )
        assert edge_trace.line.color == self._color(0.5)

    def test_colorbar_range_and_titles(self, config):
        fig = draw_qpu_heatmap(
            config, {0: 1.0, 1: 3.0}, label='My QPU',
            cbar_label='T1 (us)', show=False,
        )

        colorbar = fig.data[-1].marker
        assert (colorbar.cmin, colorbar.cmax) == (1.0, 3.0)
        assert colorbar.colorbar.title.text == 'T1 (us)'
        assert fig.layout.annotations[0].text == 'My QPU'

    def test_constant_values_widen_range(self, config):
        fig = draw_qpu_heatmap(config, {0: 2.0, 1: 2.0}, show=False)

        colorbar = fig.data[-1].marker
        assert (colorbar.cmin, colorbar.cmax) == (1.5, 2.5)
        assert fig.data[1].marker.color == (
            self._color(0.5), self._color(0.5)
        )

    def test_connecting_config_edge_without_value_is_missing(
        self, config
    ):
        fig = draw_qpu_heatmap(
            config, {0: 1.0, 1: 3.0}, missing_color='#123456',
            show=False,
        )

        # Edge (0, 1) comes from config.qubit_pairs, not from values.
        assert fig.data[0].line.color == '#123456'
        assert fig.data[3].text == ('(0, 1)<br>Value: N/A<br>',)

    def test_reversed_edge_key_is_found(self, config):
        fig = draw_qpu_heatmap(
            config, {0: 1.0, 1: 3.0, (1, 0): 2.0}, show=False
        )

        assert fig.data[0].line.color == self._color(0.5)
        assert 'Value: 2.0' in fig.data[3].text[0]

    def test_edge_key_adds_its_qubits(self, config):
        fig = draw_qpu_heatmap(config, {(2, 3): 0.5}, show=False)

        assert fig.data[2].text == ('Q2', 'Q3')
        assert all(
            text.endswith(': N/A<br>') for text in fig.data[1].text
        )

    def test_full_qpu_draws_every_qubit_and_pair(self, config):
        fig = draw_qpu_heatmap(
            config, {0: 1.0, 1: 3.0}, plot_full_qpu=True,
            missing_color='#123456', show=False,
        )

        n_edges = len(config.qubit_pairs)
        assert len(fig.data) == n_edges + 4
        labels = fig.data[n_edges + 1]
        assert labels.text == tuple(f'Q{q}' for q in config.qubits)
        node_colors = fig.data[n_edges].marker.color
        assert node_colors[:2] == (self._color(0.0), self._color(1.0))
        assert set(node_colors[2:]) == {'#123456'}

    def test_empty_values_raise(self, config):
        with pytest.raises(ValueError, match='at least one entry'):
            draw_qpu_heatmap(config, {}, show=False)

    def test_node_hover_text_uses_value_label(self, config):
        fig = draw_qpu_heatmap(
            config, {0: 1.0, (0, 1): 2.0}, label='My QPU',
            cbar_label='T1 (us)', show=False,
        )

        assert fig.data[3].text == ('(0, 1)<br>T1 (us): 2.0<br>',)
        assert fig.data[1].text[0] == 'Q0<br>T1 (us): 1.0<br>'
