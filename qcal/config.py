"""Submodule for loading and saving experimental configuration files.

Basic example useage:

    cfg = Config('config.yaml')
    cfg.qubits       # Returns a list of the qubits on the processor
    cfg.qubit_pairs  # Returns a list of the qubit pairs on the processor
    cfg.parameters   # Returns a dictionary of the entire config
    cfg.processor()  # Plots the processor connectivity
"""

import copy
import io
import networkx as nx
import logging
import yaml

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'

logger = logging.getLogger(__name__)


class Config:
    """Class for loading and storing configuration files.
    
    Basic exam
    """

    __slots__ = ['_parameters', '_filename']

    def __init__(self, filename: str = None) -> None:

        self._filename = filename
        if filename is not None:
            self.load(filename)
        else:
            logger.warning('No configuration file was provided.')
            self._parameters = {}

    def __call__(self) -> dict:
        return self._parameters

    def __copy__(self, deep_copy=False) -> dict:
        """Copy the config dictionary.

        Args:
            deep_copy (bool, optional): deep copy of the config dictionary.
                Defaults to False.

        Returns:
            dict: copy of the config dictionary.
        """
        if deep_copy is True:
            return copy.deepycopy(self._parameters)
        else:
            return self._parameters.copy()

    def __iter__(self):
        return self._parameters.__iter__()

    def __len__(self) -> int:
        return len(self._parameters)

    def __next__(self):
        return self._parameters_dict.__next__()

    def __str__(self) -> str:
        return str(self._parameters_dict)

    def __repr__(self) -> str:
        return repr(self._parameters)

    @property
    def filename(self) -> str:
        """Returns the filename of the config.yaml file.

        Returns:
            str: filename of the config.yaml file.
        """
        return self._filename

    @property
    def parameters(self) -> dict:
        return self._parameters
    
    @property
    def qubits(self) -> list:

        return list(self.get('qubits').keys())
    
    @property
    def qubit_pairs(self) -> list:

        return [eval(key) for key in self.get('qubit_pairs').keys()]
    
    def get(self, param: str):

        try:
            return self._parameters[param]
        except Exception:
            logger.warning(f"Parameter '{param}' not found in the config!")
            return None
        
    def items(self) -> tuple:
        return self._parameters.items()

    def load(self, yaml_file):

        # TODO: add automatic saved backup of file after each load

        with open(yaml_file, "r") as config:
            try:
                self._parameters = yaml.load(config, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

    def processor(self):
        """Plot a graph displaying the connectivity of the quantum processor.
        """
        plot_connectivity_graph(self.qubit_pairs)

    def save(self, filename: str = None):

        # assert self._yaml_file is not None, ""

        with io.open(
                filename if filename is not None else self._yaml_file, 'w', 
                encoding='utf8'
            ) as yaml_file:
                yaml.dump(
                    self._parameters, yaml_file, default_flow_style=False, 
                    allow_unicode=True
                )


def plot_connectivity_graph(list_of_qubit_pairs: list[tuple]):
    """Plot a graph displaying the connectivity of the quantum processor.

    Args:
        list_of_qubit_pairs (list): List of tuples of qubit pairs.
    """
    G = nx.Graph()
    G.add_edges_from(list_of_qubit_pairs)
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
                title='Connectivity',
                xanchor='left',
                titleside='right'
            ),
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
        node_text.append('# of nearest neighbors: ' + str(len(adjacencies[1])))

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
            text="Processor Layout",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002 ) ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    fig.show()