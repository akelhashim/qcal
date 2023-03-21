"""Submodule for plotting networkx style graphs using Plotly.
"""
from qcal.config import Config

import networkx as nx
import numpy as np

from typing import Dict

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'  # TODO: replace with settings

def draw_DAG(G):
    """Draw a Directed Acyclic Graph (DAG)

    Args:
        G (nx.Graph): networkx graph object
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
            line_width=2))
    
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
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    fig.show()