"""Directed Acyclic Graph (DAG)
    
Module for constructing and running a DAG for calibration routines.
This module is a wrapper around the networkx.Graph class.
See: https://networkx.org/documentation/stable/tutorial.html
"""
import qcal.settings as settings

from qcal.plotting.graphs import draw_DAG

# import matplotlib.pyplot as plt
import logging
import networkx as nx
import time

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Callable, Any

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of a calibration node"""
    PENDING =   "PENDING"
    RUNNING =   "RUNNING"
    COMPLETED = "COMPLETED "
    FAILED =    "FAILED"
    SKIPPED =   "SKIPPED"


class DAG:
    """
    Directed Acyclic Graph for managing quantum workflows.
    
    This class allows you to:
    - Add (e.g., calibration) nodes with associated classes
    - Define dependencies between nodes
    - Execute calibration in dependency order
    - Track status of each calibration step
    """
    
    def __init__(self, name: str = "DAG"):
        self._name = name
        self._nodes: Dict[str, Node] = {}
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        self._execution_order: Optional[List[str]] = None
        self._graph = nx.DiGraph()

    @property
    def nodes(self) -> list:
        """DAG Nodes"""
        return self._graph.nodes
        # return self._nodes
    
    @property
    def edges(self) -> list:
        """DAG Edges"""
        return self._graph.edges

    def _has_cycle(self) -> bool:
        """Check if the DAG has any cycles using DFS.

        Returns:
            bool: whether it has cycles or not.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in self._nodes}
        
        def visit(node):
            if color[node] == GRAY:
                return True  # Back edge found - cycle detected
            if color[node] == BLACK:
                return False  # Already processed
            
            color[node] = GRAY
            for neighbor in self._dependencies.get(node, []):
                if visit(neighbor):
                    return True
            color[node] = BLACK
            return False
        
        for node in self._nodes:
            if color[node] == WHITE:
                if visit(node):
                    return True
        return False
    
    def _get_execution_range(self, 
            full_order: List[str],
            start_node: Optional[str],
            stop_node: Optional[str]
    ) -> List[str]:
        """
        Determine which nodes to execute based on start/stop/skip parameters.
        
        Args:
            full_order (List[str]): complete topological order of nodes.
            start_node (Optional[str]): node to start from.
            stop_node (Optional[str]): node to stop at.
            
        Returns:
            List[str]: nodes to execute in order.
        """
        # Find start and stop indices
        start_idx = 0
        stop_idx = len(full_order)
        
        if start_node:
            try:
                start_idx = full_order.index(start_node)
            except ValueError:
                raise ValueError(
                    f"Start node '{start_node}' not in execution order"
                )
        
        if stop_node:
            try:
                stop_idx = full_order.index(stop_node) + 1  # +1 to include stop node
            except ValueError:
                raise ValueError(
                    f"Stop node '{stop_node}' not in execution order"
                )
        
        # Validate that start comes before stop
        if start_idx >= stop_idx:
            raise ValueError(
                f"Start node '{start_node}' must come before stop node "
                f"'{stop_node}'"
            )
        
        # Get nodes in range, excluding skipped nodes
        execution_nodes = full_order[start_idx:stop_idx]
        
        return execution_nodes

    def _mark_nodes_before_start_as_completed(
            self, full_order: List[str], start_node: str
    ) -> None:
        """
        Mark all nodes before the start node as completed.
        This is useful when resuming a workflow.
        
        Args:
            full_order (List[str]): complete topological order.
            start_node (str): node to start from.
        """
        start_idx = full_order.index(start_node)
        for node_name in full_order[:start_idx]:
            if self._nodes[node_name].status == NodeStatus.PENDING:
                self._nodes[node_name].status = NodeStatus.COMPLETED
        
    def add_node(self, 
        name:         str, 
        module:       Callable,
        parameters:   Dict[str, Any],
        dependencies: List[str] = None
    ) -> None:
        """
        Add a module (i.e., node) to the DAG.
        
        Args:
            name (str): unique identifier for the node.
            module (Callable): class to execute. Must have a 
                module.run() method.
            parameters: args and kwargs to pass to the module.
            dependencies (List[str], optional): the nodes that must be completed 
                first.
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists in the DAG")
        
        self._nodes[name] = Node(
            name=name,
            module=module,
            parameters=parameters or {}
        )
        self._execution_order = None  # Reset cached execution order

        self._graph.add_node(
            name, 
            **{**{'module': module}, **parameters}
        )

        if dependencies:
            self.add_dependencies(name, dependencies)
        
    def add_dependency(self, node: str, depends_on: str) -> None:
        """
        Add a dependency relationship between nodes.
        
        Args:
            node (str): the node that has a dependency.
            depends_on (str): the node that must be completed first.
        """
        if node not in self._nodes:
            raise ValueError(f"Node '{node}' does not exist in the DAG")
        if depends_on not in self._nodes:
            raise ValueError(f"Node '{depends_on}' does not exist in the DAG")
        
        self._dependencies[node].add(depends_on)
        self._dependents[depends_on].add(node)
        
        # Check for cycles
        if self._has_cycle():
            # Revert the change
            self._dependencies[node].discard(depends_on)
            self._dependents[depends_on].discard(node)
            raise ValueError(
                f"Adding dependency {depends_on} -> {node} would create a cycle"
            )
        
        self._graph.add_edge(node, depends_on)
        
        self._execution_order = None  # Reset cached execution order
        
    def add_dependencies(self, node: str, depends_on: List[str]) -> None:
        """Add multiple dependencies for a node.

        Args:
            node (str): the node that has a dependency.
            depends_on (List[str]): the nodes that must be completed first.
        """
        for dep in depends_on:
            self.add_dependency(node, dep)

    def can_execute(self, node: str) -> bool:
        """
        Check if a node's dependencies are satisfied, considering skipped nodes.
        
        Args:
            node (str): node to check.
            
        Returns:
            bool: True if node can be executed.
        """
        for dep in self._dependencies.get(node, []):
            dep_status = self._nodes[dep].status
            # Allow execution if dependency is completed or skipped
            if dep_status not in [NodeStatus.COMPLETED, NodeStatus.SKIPPED]:
                return False
        
        return True
    
    def execute(self, 
            start_node: Optional[str] = None,
            stop_node:  Optional[str] = None,
            skip_nodes: Optional[List[str]] = None,
            assume_completed_before_start: bool = True
        ) -> None:
        """
        Execute the calibration workflow with optional start/stop nodes and 
        optional nodes to skip.
        
        Args:
            start_node: Node from which to start execution. Defaults to None, 
                which means start from beginning.
            stop_node:  Node to stop execution at (inclusive). Defaults to None,
                which means execute to end.
            skip_nodes: List of nodes to skip during execution. Defaults to 
                None, which means execute all nodes.
            assume_completed_before_start: If True, assume all nodes before 
                start_node are completed (for dependency checking). Defaults to
                True.
        """
        skip_nodes = skip_nodes or []
        
        # Validate inputs
        if start_node and start_node not in self._nodes:
            raise ValueError(
                f"Start node '{start_node}' does not exist in DAG!"
            )
        if stop_node and stop_node not in self._nodes:
            raise ValueError(
                f"Stop node '{stop_node}' does not exist in DAG!"
            )
        for skip_node in skip_nodes:
            if skip_node not in self._nodes:
                raise ValueError(
                    f"Skip node '{skip_node}' does not exist in DAG!"
                )
        
        # Get full execution order
        full_execution_order = self.get_execution_order()
        
        # Determine execution range
        execution_nodes = self._get_execution_range(
            full_execution_order, start_node, stop_node
        )
        
        # Mark nodes before start as completed if needed
        if start_node and assume_completed_before_start:
            self._mark_nodes_before_start_as_completed(
                full_execution_order, start_node
            )
        
        # Mark skip nodes
        for node_name in skip_nodes:
            if node_name in self._nodes:
                self._nodes[node_name].status = NodeStatus.SKIPPED
        
        logger.info(f" Starting workflow: {self._name}")
        if start_node:
            logger.info(f" Starting from node: {start_node}")
        if stop_node:
            logger.info(f" Stopping at node: {stop_node}")
        if skip_nodes:
            logger.info(f" Skipping nodes: {', '.join(skip_nodes)}")
        logger.info(f" Execution order: {' -> '.join(execution_nodes)}")
        logger.info(" -" * 50)
        
        for node_name in execution_nodes:
            if node_name in skip_nodes:
                logger.info(f"  ⊘ {node_name} skipped")
                continue
                
            try:
                self.execute_node(
                    node_name, 
                    check_dependencies=not assume_completed_before_start
                )
                logger.info(f"  ✓ {node_name} completed!")
                
                # Check if we should stop
                if node_name == stop_node:
                    logger.info(f" Stopping at {stop_node}.")
                    break
                    
            except Exception as e:
                logger.info(f"  ✗ {node_name} failed: {e}")
                raise RuntimeError(
                    f"Workflow failed at node '{node_name}': {e}!"
                )
        
        logger.info(" -" * 50)
        logger.info(" Workflow completed successfully!")

    def execute_node(
        self, node_name: str, check_dependencies: bool = True
    ) -> None:
        """
        Execute a specific node.
        
        Args:
            node_name (str): name of the node to execute.
            check_dependencies (bool): whether to check if dependencies are 
                satisfied.
        """
        if node_name not in self._nodes:
            raise ValueError(f"Node '{node_name}' does not exist!")
        
        node = self._nodes[node_name]
        
        if check_dependencies and not self.can_execute(node_name):
            raise RuntimeError(
                f"Cannot execute '{node_name}' - dependencies not satisfied!"
            )
        
        logger.info(f" Executing node: {node_name}")
        node.execute()

    def execute_subgraph(self, nodes: List[str]) -> None:
        """
        Execute only a specific subgraph of nodes.
        
        Args:
            nodes: List of specific nodes to execute
        """
        # Validate all nodes exist
        for node in nodes:
            if node not in self._nodes:
                raise ValueError(f"Node '{node}' does not exist in DAG")
        
        # Get the induced subgraph execution order
        full_order = self.get_execution_order()
        subgraph_order = [n for n in full_order if n in nodes]
        
        logger.info(f" Executing subgraph of {self._name}")
        logger.info(f" Nodes: {', '.join(subgraph_order)}")
        logger.info(" -" * 50)
        
        for node_name in subgraph_order:
            # Check if dependencies within the subgraph are satisfied
            deps_in_subgraph = [
                d for d in self._dependencies.get(node_name, []) 
                if d in nodes
            ]
            
            can_execute = all(
                self._nodes[dep].status == NodeStatus.COMPLETED 
                for dep in deps_in_subgraph
            )
            
            if not can_execute:
                logger.warning(
                    f"  ⚠ Skipping {node_name} - dependencies not satisfied!"
                )
                continue
            
            try:
                logger.info(f" Executing node: {node_name}")
                self._nodes[node_name].execute()
                logger.info(f"  ✓ {node_name} completed!")
            except Exception as e:
                logger.info(f"  ✗ {node_name} failed: {e}")
                raise RuntimeError(
                    f"Subgraph execution failed at '{node_name}': {e}!"
                )
        
        logger.info(" -" * 50)
        logger.info(" Subgraph execution completed!")
    
    def get_execution_order(self) -> List[str]:
        """Get the topological order for executing calibration nodes.
        
        Uses Kahn's algorithm for topological sorting.

        Raises:
            RuntimeError: if there exists a cyle in the DAG.

        Returns:
            List[str]: order of nodes to execute.
        """
        if self._execution_order is not None:
            return self._execution_order
        
        # Calculate in-degree for each node
        in_degree = {
            node: len(self._dependencies.get(node, set())) 
            for node in self._nodes
        }
        
        # Find all nodes with no dependencies
        queue = deque(
            [node for node, degree in in_degree.items() if degree == 0]
        )
        execution_order = []
        
        while queue:
            node = queue.popleft()
            execution_order.append(node)
            
            # Reduce in-degree for dependent nodes
            for dependent in self._dependents.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(execution_order) != len(self._nodes):
            raise RuntimeError(
                "Failed to determine execution order - possible cycle in DAG!"
            )
        
        self._execution_order = execution_order
        return execution_order
    
    def get_node_status(self) -> Dict[str, NodeStatus]:
        """Get the status of all nodes.

        Returns:
            Dict[str, NodeStatus]: status of each node.
        """
        return {name: node.status for name, node in self._nodes.items()}

    def get_reachable_nodes(self, from_node: str) -> Set[str]:
        """
        Get all nodes reachable from a given node (forward dependencies).
        
        Args:
            from_node (str): starting node name.
            
        Returns:
            Set: set of reachable node names.
        """
        if from_node not in self._nodes:
            raise ValueError(f"Node '{from_node}' does not exist")
        
        reachable = set()
        to_visit = [from_node]
        
        while to_visit:
            current = to_visit.pop()
            if current not in reachable:
                reachable.add(current)
                to_visit.extend(self._dependents.get(current, []))
        
        return reachable

    def get_required_nodes(self, for_node: str) -> Set[str]:
        """
        Get all nodes required for a given node (backward dependencies).
        
        Args:
            for_node (str): target node.
            
        Returns:
            Set: set of required node names.
        """
        if for_node not in self._nodes:
            raise ValueError(f"Node '{for_node}' does not exist")
        
        required = set()
        to_visit = [for_node]
        
        while to_visit:
            current = to_visit.pop()
            if current not in required:
                required.add(current)
                to_visit.extend(self._dependencies.get(current, []))
        
        return required
    
    def reset(self) -> None:
        """Reset all nodes to pending status"""
        for node in self._nodes.values():
            node.status = NodeStatus.PENDING
            node.error = None
            node.start_time = None
            node.end_time = None
    
    def visualize(self) -> None:
        """Generate a simple text representation of the DAG."""
        lines = [f"DAG: {self._name}"]
        lines.append("=" * 40)
        
        for node_name in self.get_execution_order():
            node = self._nodes[node_name]
            deps = list(self._dependencies.get(node_name, []))
            
            status_symbol = {
                NodeStatus.PENDING: "○",
                NodeStatus.RUNNING: "◐",
                NodeStatus.COMPLETED: "●",
                NodeStatus.FAILED: "✗",
                NodeStatus.SKIPPED: "○"
            }.get(node.status, "?")
            
            if deps:
                lines.append(
                    f"{status_symbol} {node_name} <- {', '.join(deps)}"
                )
            else:
                lines.append(f"{status_symbol} {node_name}")
        
        print("\n".join(lines))
    
    def draw(self):
        """Draw the DAG."""
        draw_DAG(self._graph)


@dataclass
class Node:
    """Represents a single node in the DAG"""
    name:       str
    module:     Callable
    parameters: Dict[str, Any]
    status:     NodeStatus = NodeStatus.PENDING
    error:      Optional[Exception] = None
    start_time: Optional[float] = None
    end_time:   Optional[float] = None

    @property
    def execution_time(self) -> Optional[float]:
        """Get the execution time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        
        return None
    
    def execute(self):
        """Execute the calibration function for this node."""
        self.status = NodeStatus.RUNNING
        self.start_time = time.time()

        # Load the config to pick up the current changes
        if settings.Settings.save_data:
            self.parameters['config'].load()
        
        try:
            self.module = self.module(**self.parameters)
            self.module.run()
            
            self.status = NodeStatus.COMPLETED
            self.end_time = time.time()
            logger.info(f' Execution time: {self.execution_time}')

        except Exception as e:
            self.status = NodeStatus.FAILED
            self.error = e
            self.end_time = time.time()
            raise
