"""Submodule for plotting networkx style graphs using Plotly.
"""
import logging
from collections import defaultdict
from typing import Dict

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.graph_objs.scatter import Marker

from qcal.circuit import Circuit
from qcal.config import Config
from qcal.gate.gate import Gate

logger = logging.getLogger(__name__)


def format_gate_text(gate: Gate):
    """Format Gate text using the gate properties metadata.

    Args:
        gate (Gate): qcal Gate object.

    Returns:
        str: text formatted for Plotly plot.
    """
    text =  f'Name: {gate.name}<br>'
    if gate.alias is not None:
        text += f'Alias: {gate.alias}<br>'
    text += f'Qubits: {gate.qubits}<br>'
    text += f'Dim: {gate.dim}<br>'
    text += (
        'Matrix: <br>  '
        + np.array_str(np.around(gate.matrix, 3)).replace("\n ", "<br>" + '   ')
        + '<br>'
    )
    if gate.locally_equivalent is not None:
        text += f'Locally Equivalent: {gate.locally_equivalent}<br>'
    text += f'Subspace: {gate.subspace}<br>'
    if bool(gate.properties['params']):
        for key, value in gate.properties['params'].items():
            if isinstance(value, float):
                if value != 0 and abs(value) < 1e-2:
                    value = '{:.2E}'.format(value)
                else:
                    value = '{:.2f}'.format(value)
            text += f'{key}: {value}<br>'

    return text


def draw_circuit(circuit: Circuit, show: bool = True):
    """Draw a circuit.

    Args:
        circuit (Circuit): qcal Circuit object.
        show (bool):       whether to plot the circuit. Defaults to True. If
            False, the figure object is returned.

    """
    pio.templates.default = 'plotly'

    # https://plotly.com/python/marker-style/
    symbol_map = defaultdict(lambda: ['square', 'square'],
        {'Meas':   ['triangle-right'],
         'MCM':    ['triangle-right'],
         'Reset':  ['bowtie'],
         'CH':     ['circle', 'square'],
         'CNOT':   ['circle', 'circle-cross'],
         'CPhase': ['circle', 'square'],
         'CRot':   ['circle', 'square'],
         'CV':     ['circle', 'square'],
         'CX':     ['circle', 'circle-cross'],
         'CY':     ['circle', 'square'],
         'CZ':     ['circle', 'circle'],
         'SWAP':   ['x-thin', 'x-thin'],
         'iSWAP':  ['square-x', 'square-x']}
    )
    # color_map = defaultdict(lambda: '#3366CC', {'Meas': '#B82E2E'})
    color_map = defaultdict(lambda: 'white',
        {'Meas': 'black',
         'MCM':  'white',
         'Reset': 'white'
        }
    )

    node_x = []
    node_y = []
    node_text = []
    node_symbols = []
    gate_names = []
    marker_colors = []
    barrier_locs = []
    qnode_text = defaultdict(lambda: False, {})
    n_barriers = 0
    for c, cycle in enumerate(circuit.cycles):
        if cycle.is_barrier:
            barrier_locs.append(c - n_barriers - 0.5)
            n_barriers += 1
        else:
            c -= n_barriers
            for gate in cycle.gates:
                if gate.is_single_qubit:
                    for q in gate.qubits:
                        node_x.append(c)
                        node_y.append(circuit.qubits.index(q))
                    name = gate.name
                    if 'phase' in gate.properties['params'].keys():
                        name += str(gate.properties['params']['phase'])
                    text = qnode_text[
                        f'{name}{gate.subspace}:{gate.qubits}'
                    ]
                    if not text:
                        text = format_gate_text(gate)
                        qnode_text[
                            f'{gate.name}{gate.subspace}:{gate.qubits}'
                        ] = text
                    node_text.extend(
                        [text] * len(gate.qubits)
                    )
                    node_symbols.extend(
                        [symbol_map[gate.name][0]] * len(gate.qubits)
                    )
                    gate_names.append(
                        {'x': c,
                         'y': circuit.qubits.index(q),
                         'text': 'M' if gate.name in ('Meas', 'MCM') else (
                             gate.name if len(gate.name) < 3 else gate.name[:3]
                         ),
                         'font': {'color': 'black', 'size': 13},
                         'showarrow': False
                        }
                    )
                    marker_colors.extend(
                        [color_map[gate.name]] * len(gate.qubits)
                    )
                elif gate.is_multi_qubit:
                    for q in gate.qubits:
                        node_x.append(c)
                        node_y.append(circuit.qubits.index(q))
                    text = qnode_text[
                        f'{gate.name}{gate.subspace}:{gate.qubits}'
                    ]
                    if not text:
                        text = format_gate_text(gate)
                        qnode_text[
                            f'{gate.name}{gate.subspace}:{gate.qubits}'
                        ] = text
                    node_text.extend([text] * 2)
                    node_symbols.extend(symbol_map[gate.name])
                    gate_names.append(
                        {
                            'x': c,
                            'y': circuit.qubits.index(q),
                            'text': '',
                            'font': {'color': 'black', 'size': 13},
                            'showarrow': False
                        },
                    )
                    marker_colors.extend(['white', 'white'])

    # ms_scale = 200
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker_symbol=node_symbols,
        marker_line_color="black",
        marker={
            'showscale': False,
            'reversescale': False,
            'color': marker_colors,
            'size': 30,
            'line_width': 2
        },
    )
    node_trace.text = node_text

    edge_traces = []
    for i, _ in enumerate(circuit.qubits):
        edge_x = []
        for j in range(1, circuit.circuit_depth):
            edge_x.extend([j-1, j])
        edge_y = [i]*(2*(circuit.circuit_depth - 2) + 2)
        edge_traces.append(
            go.Scatter(
                x=edge_x, y=edge_y,
                line={'width': 2, 'color': '#888'},
                hoverinfo='none',
                mode='lines'
        ))

    n_barriers = 0
    for c, cycle in enumerate(circuit.cycles):
        if cycle.is_barrier:
            n_barriers += 1
        else:
            c -= n_barriers
            for gate in cycle.gates:
                edge_x_mq = []
                edge_y_mq = []
                if gate.is_multi_qubit:
                    for q in gate.qubits:
                        edge_x_mq.append(c)
                        edge_y_mq.append(circuit.qubits.index(q))
                    edge_traces.append(
                        go.Scatter(
                        x=edge_x_mq, y=edge_y_mq,
                        line={'width': 2, 'color': '#888'},
                        hoverinfo='none',
                        mode='lines'
                        )
                    )

    gate_names.append(
        {
            'text': "Quantum Circuit",
            'showarrow': False,
            'xref': "paper",
            'yref': "paper",
            'x': -0.001,
            'y': 1.05
        }
    )

    fig = go.Figure(
        data= edge_traces + [node_trace],
        layout=go.Layout(
            title={'font': {'size': 16}},
            showlegend=False,
            hovermode='closest',
            margin={'b': 20, 'l': 10, 'r': 5, 't': 50},
            annotations=gate_names,
            xaxis={'tick0': 0, 'dtick': 1, 'showgrid': False, 'zeroline': False,
                'showticklabels': True},
            yaxis={'tickmode': 'array',
                'tickvals': list(range(len(circuit.qubits))),
                'ticktext': circuit.qubits,
                'showgrid': False, 'zeroline': False, 'showticklabels': True}
        )
    )

    fig['layout']['yaxis']['autorange'] = "reversed"
    for loc in barrier_locs:
        fig.add_vline(x=loc, line_width=3,
                      line_dash="dash", line_color="black")

    fig.update_xaxes(range=[-0.75, circuit.circuit_depth - 0.5])
    # fig.update_yaxes(range=[-2, circuit.circuit_width + 2])
    fig.update_layout(
        autosize=False,
        width=75 * (circuit.circuit_depth + 0.5) if circuit.circuit_depth > 1
              else 150,
        height=75 * (circuit.circuit_width + 0.5) if circuit.circuit_width > 1
               else 150,
    )

    save_properties = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'quantum_circuit',
            # 'height': 500,
            # 'width': 1000,
            'scale': 10 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }

    if show:
        fig.show(config=save_properties)
    else:
        return fig


def draw_DAG(G: nx.Graph | nx.DiGraph):
    """Draw a Directed Acyclic Graph (DAG).

    Args:
        G (nx.Graph | nx.DiGraph): networkx graph object.
    """
    pio.templates.default = 'plotly'

    pos = nx.kamada_kawai_layout(G)
    # pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker={
            'showscale': True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            # colorscale='YlGnBu',
            'colorscale': 'Tealgrn',
            'reversescale': True,
            'color': [],
            'size': 40,
            'colorbar': {
                'thickness': 15,
                'title': {
                    'text': "Number of Dependencies",
                    'font': {
                        'size': 20,
                        # family="Arial",
                        # color="darkblue"
                    }
                },
                'tickfont': {
                    'size': 15,
                    # family="Arial",
                    # color="darkred"
                },
                'xanchor': 'left',
                'titleside': 'right'
            },
            'line_width': 2
        }
    )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line={'width': 0.5, 'color': '#888'},
        hoverinfo='none',
        mode='lines'
    )

    node_text = []
    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        # node_text[node] = 'Connectivity: ' + str(len(adjacencies[1]))
        node_text.append(adjacencies[0] + '<br>')
        node_text[node] += 'Dependencies: ' + str(len(adjacencies[1]))
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    node_labels = go.Scatter(
        x=node_x, y=node_y,
        text=list(G.nodes),
        mode='text',
        hoverinfo='text',
        marker={'color': '#5D69B1', 'size': 0.01}
    )

    fig = go.Figure(
        data=[edge_trace, node_trace, node_labels],
        layout=go.Layout(
            title={'font': {'size': 16}},
            showlegend=False,
            hovermode='closest',
            margin={'b': 20,'l': 5,'r': 5,'t': 40},
            annotations=[ {
                'text': "DAG",
                'showarrow': False,
                'xref': "paper", 'yref': "paper",
                'x': 0.005, 'y': -0.002,
                'font': {'size': 25}
            } ],
            xaxis={
                'showgrid': False, 'zeroline': False, 'showticklabels': False
            },
            yaxis={
                'showgrid': False, 'zeroline': False, 'showticklabels': False
            }
        )
    )

    save_properties = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'quantum_circuit',
            # 'height': 500,
            # 'width': 1000,
            'scale': 10 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }

    fig.update_layout(
        autosize=False,
        width=1250,
        height=800,
    )

    fig.show(config=save_properties)


def format_qubit_text(qubit: Dict) -> str:
    """Format qubit node text in processor plot.

    Args:
        qubit (Dict): dictionary of qubit properties.

    Returns:
        str: text formatted for Plotly plot.
    """
    text = ''
    for key, value in qubit.items():
        if not isinstance(value, dict):
            text += f'{key.capitalize()}: {qubit[key]}<br>'
        else:
            text += f'{key}:<br>'
            for k, v in qubit[key].items():
                if not isinstance(v, dict) and not isinstance(v, list):
                    text += f'  {k.capitalize()}: {qubit[key][k]}<br>'
    return text


def draw_qpu(config: Config):
    """Draw a processor connectivity graph for a given config.

    Args:
        config (Config): qcal Config
    """
    G = nx.Graph()
    G.add_edges_from(config.qubit_pairs)
    pos = nx.spring_layout(G)

    node_x = []
    node_y = []
    for node in sorted(G.nodes()):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    edge_x = []
    edge_y = []
    middle_node_x = []
    middle_node_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        middle_node_x.append((x0+x1)/2)
        middle_node_y.append((y0+y1)/2)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker={
            'showscale': True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            'colorscale': 'Tealgrn',
            'reversescale': False,
            'opacity': 0.65,
            'color': [],
            'size': 30,
            'colorbar': {
                'thickness': 15,
                'title': {
                    'text': "Qubit Connectivity",
                    'side': 'right',
                    'font': {
                        'size': 20,
                        # family="Arial",
                        # color="darkblue"
                    }
                },
                'tickfont': {
                    'size': 15,
                    # family="Arial",
                    # color="darkred"
                },
                'xanchor': 'left',
            },
            'line_width': 2
        }
    )

    node_text = []
    for q in config.qubits:
        node_text.append(
            f'Q{q}:<br>' + format_qubit_text(
            config.parameters['single_qubit'][q]
            )
        )
    node_adjacencies = []
    for node, adjacencies in enumerate(sorted(G.adjacency())):
        node_adjacencies.append(len(adjacencies[1]))
        node_text[node] += 'Connectivity: '+str(len(adjacencies[1]))
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    qubit_labels = go.Scatter(
        x=node_x, y=node_y,
        text=[f'Q{q}' for q in config.qubits],
        mode='text',
        hoverinfo='text',
        marker={'color': '#5D69B1', 'size': 0.01}
    )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line={'width': 0.5, 'color': '#888'},
        hoverinfo='none',
        mode='lines')

    middle_node_text = []
    for pair in config.qubit_pairs:
        middle_node_text.append(
            format_qubit_text(
                config.parameters['two_qubit'][str(pair)]
            )
        )
    middle_node_trace = go.Scatter(
        x=middle_node_x,
        y=middle_node_y,
        text=middle_node_text,
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            opacity=0
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace, qubit_labels, middle_node_trace],
        layout=go.Layout(
            title={'font': {'size': 16}},
            showlegend=False,
            hovermode='closest',
            margin={'b': 20,'l': 5,'r': 5,'t': 40},
            annotations=[ {
                'text': "QPU",
                'showarrow': False,
                'xref': "paper", 'yref': "paper",
                'x': 0.005, 'y': -0.002,
                'font': {'size': 25}
            } ],
            xaxis={
                'showgrid': False, 'zeroline': False, 'showticklabels': False
            },
            yaxis={
                'showgrid': False, 'zeroline': False, 'showticklabels': False
            }
        )
    )

    save_properties = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'qpu_layout',
            # 'height': 500,
            # 'width': 1000,
            'scale': 10 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }

    fig.show(config=save_properties)
