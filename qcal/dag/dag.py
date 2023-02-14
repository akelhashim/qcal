"""Directed Acyclic Graph (DAG)
    
Module for constructing and running a DAG for calibration routines.
This module is a wrapper around the networkx.Graph class.
See: https://networkx.org/documentation/stable/tutorial.html
"""

import matplotlib.pyplot as plt
import networkx as nx

from typing import List


class DAG:
    """Directed Acyclic Graph"""

    __slots__ = '_graph'

    def __init__(self) -> None:
        
        self._graph = nx.DiGraph()

    @property
    def nodes(self) -> list:

        return self._graph.nodes
    
    @property
    def edges(self) -> list:

        return self._graph.edges

    def add_node(self, name: str, protocol: str, kwargs: dict):

        self._graph.add_node(name, protocol, kwargs)

    def draw(self):

        fig, ax = plt.subplots(figsize=(5,5))
        nx.draw(self._graph, with_labels=True, font_weight='bold')
        return ax

    def remove_node(self, node: str):

        self._graph.remove_node(node)

    def run(self, config, start: str, stop: str, skip: List[str]):
        """Run the DAG calibration.

        Args:
            config (_type_): _description_
            start (str): start node.
            stop (str): stop node (inclusive).
            skip (List[str]): nodes to skip.
        """
        pass
