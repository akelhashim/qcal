"""Submodule for plotting networkx style graphs using Plotly.
"""
from qcal.circuit import Circuit
from qcal.config import Config
from qcal.gate.gate import Gate

import networkx as nx
import numpy as np

from collections import defaultdict

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'colab'  # TODO: replace with settings

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
        + np.array_str(gate.matrix).replace("\n ", "<br>" + '   ') 
        + '<br>'
    )
    if gate.locally_equivalent is not None:
        text += f'Locally Equivalent: {gate.locally_equivalent}<br>'
    if bool(gate.properties['params']):
        for key, value in gate.properties['params'].items():
            text += f'{key}: {value}<br>'

    return text


def draw_circuit(circuit: Circuit, show: bool = True):
    """Draw a circuit.

    Args:
        circuit (Circuit): qcal Circuit object.
        show (bool): whether to plot the Fig. Defaults to True. If False, the 
            Fig object is returned instead.
    """
    # https://plotly.com/python/marker-style/
    symbol_map = defaultdict(lambda: ['square', 'square'],
        {'Meas':   ['triangle-left'],
         'CH':     ['circle', 'square'],
         'CNot':   ['circle', 'circle-cross'],
         'CPhase': ['circle', 'square'],
         'CRot':   ['circle', 'square'],
         'CV':     ['circle', 'square'],
         'CX':     ['circle', 'circle-cross'],
         'CY':     ['circle', 'square'],
         'CZ':     ['circle', 'circle'],
         'SWAP':   ['x-thin', 'x-thin']}
    )
    color_map = defaultdict(lambda: 'blue', {'Meas': 'red'})

    node_x = []
    node_y = []
    node_text = []
    node_symbols = []
    marker_colors = []
    for c, cycle in enumerate(circuit.cycles):
        for gate in cycle.gates:
            if gate.is_single_qubit:
                node_x.append(c)
                node_y.append(gate.qubits[0])
                node_text.append(format_gate_text(gate))
                node_symbols.append(symbol_map[gate.name][0])
                marker_colors.append(color_map[gate.name])
            elif gate.is_multi_qubit:
                for q in gate.qubits:
                    node_x.append(c)
                    node_y.append(q)
                node_text.extend([format_gate_text(gate)]*2)
                node_symbols.extend(symbol_map[gate.name])
                marker_colors.extend(['grey', 'grey'])

    ms_scale = 200
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker_symbol=node_symbols,
        marker_line_color="black",
        marker=dict(
            showscale=False,
            reversescale=False,
            color=marker_colors,
            size=ms_scale/circuit.circuit_width,
            line_width=2)
        )
    node_trace.text = node_text

    edge_traces = []
    for i in range(circuit.circuit_width):
        edge_x = []
        for j in range(1, circuit.circuit_depth):
            edge_x.extend([j-1, j])
        edge_y = [i]*(2*(circuit.circuit_depth - 2) + 2)
        edge_traces.append(
            go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
        ))
    for c, cycle in enumerate(circuit.cycles):
        for gate in cycle.gates:
            edge_x_mq = []
            edge_y_mq = []
            if gate.is_multi_qubit:
                for q in gate.qubits:
                    edge_x_mq.append(c)
                    edge_y_mq.append(q)
                edge_traces.append(
                    go.Scatter(
                    x=edge_x_mq, y=edge_y_mq,
                    line=dict(width=2, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                    )
                )

    fig = go.Figure(
        data= edge_traces + [node_trace],
        layout=go.Layout(
        # title='',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Quantum Circuit",
            showarrow=False,
            xref="paper", yref="paper",
            x=-0.001, y=1.0) ],
        xaxis=dict(tick0=0, dtick=1, showgrid=False, zeroline=False, 
                   showticklabels=True),
        yaxis=dict(tickmode ='array', tickvals = circuit.qubits, 
                   showgrid=False, zeroline=False, showticklabels=True))
    )
    fig['layout']['yaxis']['autorange'] = "reversed"
    # fig.update_layout(
    #     autosize=True)
    if show:
        fig.show()
    else:
        return fig


def draw_DAG(G: nx.Graph):
    """Draw a Directed Acyclic Graph (DAG).

    Args:
        G (nx.Graph): networkx graph object.
    """
    pos = nx.spring_layout(G)

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
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Dependencies',
                xanchor='left',
                titleside='right'),
            line_width=2))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of dependencies: ' + str(len(adjacencies[1] - 1)))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
        # title='',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="DAG",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002 ) ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    fig.show()


def draw_processor_connectivity(config: Config):
    """Draw a processor connectivity graph for a given config.

    Args:
        config (Config): qcal Config
    """
    G = nx.Graph()
    G.add_edges_from(config.qubit_pairs)
    pos = nx.spring_layout(G)

    node_x = []
    node_y = []
    for i, node in enumerate(G.nodes()):
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

    two_qubit_gates = []
    for pair in config.parameters['qubit_pairs']:
        two_qubit_gates.append(
            list(config.parameters['qubit_pairs'][pair]['gate'].keys())[0]
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=25,
            colorbar=dict(
                thickness=15,
                title='Connectivity',
                xanchor='left',
                titleside='right'),
            line_width=2)
    )
    
    qubit_labels = go.Scatter(
        x=node_x, y=node_y,
        text=[str(q) for q in config.qubits],
        mode='text',
        hoverinfo='text',
        marker=dict(color='#5D69B1', size=0.01)
    )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    two_qubit_gate_labels = go.Scatter(
        x=np.ediff1d(node_x, to_end=node_x[0] - node_x[-1]), 
        y=np.ediff1d(node_y, to_end=node_y[0] - node_y[-1]), 
        mode="markers+text",
        text=two_qubit_gates,
        textposition="top center",
        hoverinfo='text',
        marker=dict(color='#5D69B1', size=0.01)
    )
    
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(
            '# of nearest neighbors: ' + str(len(adjacencies[1]))
        )

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace, qubit_labels, two_qubit_gate_labels],
        layout=go.Layout(
            # title='',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Processor Layout",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    fig.show()