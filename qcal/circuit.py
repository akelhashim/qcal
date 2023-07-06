"""Submodule for handling cycles, circuits, and sets of circuits.

A Cycle or Layer defines a single clock-cycle in a quantum circuit (i.e. a set
of gates defined in parallel on disjoin sets of qubits).

A Circuit is a collection of cycles or layers defining the set of operations
on quibts.

CircuitSet takes a list of circuits and loads them into a dataframe that can be
used to store other useful information about the circuits, enabling fast and
easy sorting of circuits by arbitrary variables.

Basic example useage:

    cs = CircuitSet([list of circuits])
"""
from __future__ import annotations

from qcal.gate.gate import Gate
from qcal.gate.single_qubit import basis_rotation, Meas
from qcal.results import Results

import copy
import pandas as pd

from collections import deque
from collections.abc import Iterable
from typing import Any, Dict, List, Tuple, Union

import plotly.io as pio
pio.renderers.default = 'colab'  # TODO: replace with settings


__all__ = ('Barrier', 'Cycle', 'Layer', 'Circuit', 'CircuitSet')


class Barrier:
    """Class defining a barrier in a circuit."""

    __slots__ = ('_qubits')

    def __init__(self, qubits: Tuple[int] = tuple()) -> None:
        """Initialize the Barrier class. 
        
        Barrier takes an option qubits kwarg. If qubits does not include every
        qubit in full quantum register, then only a partial barrier is
        implemented.

        Args:
            qubits (Tuple[int], optional): qubits to enforce a barrier between.
                Defaults to tuple().
        """
        self._qubits = tuple(qubits)

    def __copy__(self) -> Barrier:
        return Barrier(copy.deepcopy(self._qubits))

    def __repr__(self) -> str:
        """Draw the circuit as a string."""
        return self.name
    
    def __str__(self) -> str:
        """Draw the circuit as a string."""
        return self.name

    @property
    def is_barrier(self) -> bool:
        """Whether or not the cycle is a barrier.

        Returns:
            bool: True
        """
        return True
    
    @property
    def name(self) -> str:
        """Name of the barrier cycle.

        Returns:
            str: 'Barrier'
        """
        return 'Barrier'
    
    @property
    def qubits(self) -> Tuple:
        """Empty tuple of qubit labels.

        Returns:
            Tuple: empty tuple.
        """
        return self._qubits
    
    def copy(self) -> Barrier:
        """Deep copy of the Barrier.

        Returns:
            Barrier: copy of the Barrier.
        """
        return self.__copy__()

class Cycle:
    """Class defining a cycle in a circuit."""

    __slots__ = ('_gates', '_qubits')

    def __init__(self, gates: List[Gate] = []) -> None:
        """
        Args:
            gates (List[Gate], optional): list of gates. Defaults to [].
        """
        self._gates = gates
        qubits = tuple()
        for gate in gates:
            qubits += gate.qubits
        self._qubits = tuple(sorted(set(qubits)))

    def __getitem__(self, idx: int) -> Gate:
        """Index the gates in the cycle/layer.

        Args:
            idx (int): index of a gate.

        Returns:
            Gate: Gate object.
        """
        return self._gates[idx]
    
    def __call__(self) -> List:
        return self._gates
    
    def __copy__(self) -> List:
        return Cycle([copy.deepcopy(gate) for gate in self._gates])
    
    def __iter__(self):
        return iter(self._gates)
    
    def __repr__(self) -> str:
        """Draw the cycle/layer as a string."""
        return self.to_str()
    
    def __str__(self) -> str:
        """Draw the cycle/layer as a string."""
        return self.to_str()
    
    def _repr_html_(self):
        """Draw the circuit as an html string."""
        df = pd.DataFrame([
            [(pd.DataFrame(gate.matrix)
                .style
                .format(precision=3)
                .hide(axis='index')
                .hide(axis='columns')
                .set_table_attributes('class="matrix"')
                .to_html()
            ) for gate in self._gates]
        ], dtype="object").transpose()
        df.index = [gate.qubits for gate in self._gates]
        df.columns = ['Matrix']
        df.insert(0, 'Gate', [gate.name for gate in self._gates])

        df_styler = df.style.set_table_styles([
            {"selector": ".matrix", "props": "position: relative;"},
            {"selector": ".matrix:before, .matrix:after", 
             "props":  'content: ""; position: absolute; top: 0; border: 1px \
                solid #000; width: 6px; height: 100%;'
            },
            {"selector": ".matrix:before", 
             "props": "left: -0px; border-right: -0;"},
            {"selector": ".matrix:after",
              "props": "right: -0px; border-left: 0;"}
        ])

        return df_styler.to_html()
    
    @property
    def is_barrier(self) -> bool:
        """Whether or not the cycle is a barrier.

        Returns:
            bool: False
        """
        return False
    
    @property
    def n_gates(self) -> int:
        """The number of gates in the cycle/layer.

        Returns:
            int: number of gates.
        """
        return len(self._gates)

    @property
    def n_qubits(self) -> int:
        """The number of qubits in the cycle/layer.

        Returns:
            int: number of qubits.
        """
        return len(self._qubits)
    
    @property
    def gates(self) -> List:
        """The gates in the cycle/layer.

        Returns:
            List: gates in cycle/layer.
        """
        return [gate for gate in self._gates]
    
    @property
    def qubits(self) -> Tuple:
        """The qubit labels for the cycle/layer.

        Returns:
            Tuple: qubit labels.
        """
        return self._qubits
    
    def append(self, gate: Gate) -> None:
        """Appends a gate to the existing cycle/layer.

        Args:
            gate (Gate): qcal gate object.
        """
        assert isinstance(gate, Gate), "The gate must be a qcal Gate object!"
        self._gates.append(gate)

    def copy(self) -> Cycle | Layer:
        """Deep copy of the cycle/layer.

        Returns:
            Cycle | Layer: copy of the cycle/layer.
        """
        return self.__copy__()

    def to_matrix_table(self) -> pd.DataFrame:
        """Convert the cycle to a table of matrices acting on gate labels.

        Returns:
            pd.DataFrame: table of matrices.
        """
        df = pd.DataFrame(
            data=[[gate.matrix] for gate in self._gates], 
            columns=['Matrix'], 
            index=[gate.qubits for gate in self._gates]
        )
        return df
    
    def to_str(self) -> str:
        """Convert the cycle to a string.

        Returns:
            str: string representation of the cycle.
        """
        str_rep = 'Cycle'
        for gate in self._gates:
            str_rep += f' {gate.name}:{gate.qubits}'
        return str_rep


class Layer(Cycle):
    """Class defining a layer in a circuit."""

    __slots__ = ['_gates', '_qubits']

    def __init__(self, gates: List = []) -> None:
        super().__init__(gates)

    def __copy__(self) -> Layer:
        return Layer([copy.deepcopy(gate) for gate in self._gates])
    
    def to_str(self) -> str:
        """Convert the cycle to a string.

        Returns:
            str: string representation of the cycle.
        """
        str_rep = 'Layer'
        for gate in self._gates:
            str_rep += f' {gate.name}:{gate.qubits}'
        return str_rep
    

class Circuit:

    __slots__ = ('_cycles', '_qubits', '_results')

    def __init__(self,
            cycles_or_layers: List[Union[Cycle, Layer]] = []
        ) -> None:
        
        if cycles_or_layers:
            cycles_or_layers = [
                cycle if isinstance(cycle, (Cycle, Layer, Barrier)) else 
                Cycle(cycle) for cycle in cycles_or_layers
            ]
        self._cycles = deque(cycles_or_layers)
        qubits = tuple()
        for cycle in self._cycles:
            qubits += cycle.qubits
        self._qubits = tuple(sorted(set(qubits)))

        self._results = Results()

    def __copy__(self) -> Circuit:
        """Make a deep copy of the Circuit."""
        return Circuit([cycle.copy() for cycle in self._cycles])

    def __getitem__(self, idx: int) -> Cycle | Layer:
        """Index the Circuit object to obtain a given cycle/layer.

        Args:
            idx (int): index of a cycle/layer.

        Returns:
            Cycle | Layer: individual cycle or layer in a circuit.
        """
        return self._cycles[idx]

    def __iter__(self):
        return iter(self._cycles)
    
    def __len__(self) -> int:
        return len(self._cycles)
    
    def _repr_html_(self) -> str:  # TODO: sometimes causes notebook to crash
        """Draw the html formatted circuit."""
        from qcal.plotting.graphs import draw_circuit
        fig = draw_circuit(self, show=False)
        return pio.to_html(fig)

    @property
    def cycles(self) -> deque:
        """The cycles in the circuit.

        Returns:
            deque: list of cycles.
        """
        return [cycle for cycle in self._cycles]
    
    @property
    def layers(self) -> deque:
        """The layers in the circuit.

        Returns:
            deque: list of layers.
        """
        return [cycle for cycle in self._cycles]
    
    @property
    def circuit_depth(self) -> int:  # TODO: exclude measurement cycle?
        """The number of layers/cycles in the circuit.

        Returns:
            int: circuit depth in terms of layers/cycles.
        """
        return len([cycle for cycle in self._cycles if not cycle.is_barrier])
    
    @property
    def circuit_width(self) -> int:
        """The number of qubits in the circuit.

        Returns:
            int: number of qubits.
        """
        return len(self._qubits)
    
    @property
    def n_cycles(self) -> int:
        """The number of cycles in the circuit.

        Returns:
            int: number of cycles.
        """
        return self.circuit_depth
    
    @property
    def n_layers(self) -> int:
        """The number of layers in the circuit.

        Returns:
            int: number of layers.
        """
        return self.circuit_depth
    
    @property
    def n_qubits(self) -> int:
        """The number of qubits in the circuit.

        Returns:
            int: number of qubits.
        """
        return len(self._qubits)
    
    @property
    def results(self) -> Results:
        """Circuit results.

        Returns:
            Results: Results object.
        """
        return self._results
    
    @property
    def qubits(self) -> Tuple[int]:
        """The qubits in the circuit.

        Returns:
            Tuple: qubit labels.
        """
        return self._qubits
    
    @results.setter
    def results(self, results: Dict):
        """Write a dictionary of results to the circuit Results object.

        Args:
            results (Dict): dictionary of bitstring and counts.
        """
        self._results = Results(results)
    
    def _update_qubits(self) -> None:
        """Updates the qubits after mutating the cycles."""
        qubits = tuple()
        for cycle in self._cycles:
            qubits += cycle.qubits
        self._qubits = tuple(sorted(set(qubits)))
    
    def append(
            self, cycle_or_layer: List[Cycle | Layer] | Cycle | Layer
        ) -> None:
        """Appends a cycle/layer to the end of the circuit.

        Args:
            cycle_or_layer (List, Cycle, Layer): cycle/layer to append.
        """
        if isinstance(cycle_or_layer, List):
            cycle_or_layer = Cycle(cycle_or_layer)
        else:
            assert (
                isinstance(cycle_or_layer, Cycle) or 
                isinstance(cycle_or_layer, Layer) or
                isinstance(cycle_or_layer, Barrier)
            ), "cycle_or_layer must be a Cycle or Layer object!"
        self._cycles.append(cycle_or_layer)
        self._update_qubits()

    def copy(self) -> Circuit:
        """Deep copy of the Circuit.

        Returns:
            Circuit: copy of the Circuit.
        """
        return self.__copy__()

    def extend(self, circuit: Circuit | List[Cycle | Layer]) -> None:
        """Appends another circuit to the end of the current circuit.

        Args:
            circuit (Circuit | List[Cycle | Layer]): circuit to append.
        """
        if isinstance(circuit, List):
            circuit = Circuit(circuit)
        else:
            assert (
                isinstance(circuit, Circuit)
            ), "circuit must be a Circuit object!"
        self._cycles.extend(circuit._cycles)
        self._update_qubits()

    def draw(self) -> None:
        """Draw the circuit."""
        from qcal.plotting.graphs import draw_circuit
        draw_circuit(self)

    # TODO
    def get_index(self,
            cycle_or_layer: Union[List, Cycle, Layer], beg=0, end=-1
        ) -> int:
        """Returns the first index of the cycle_or_layer in the circuit.

        The search is performed starting from the `beg` index and ending with 
        the `end` index.

        Args:
            cycle_or_layer (List, Cycle, Layer): cycle/layer to index.
            beg (int, optional): beginning index. Defaults to 0.
            end (int, optional): ending index. Defaults to -1.

        Returns:
            int: first index where cycle_or_layer is found.
        """
        if isinstance(cycle_or_layer, List):
            cycle_or_layer = Cycle(cycle_or_layer)
        else:
            assert (
                isinstance(cycle_or_layer, Cycle) or 
                isinstance(cycle_or_layer, Layer)
            ), "cycle_or_layer must be a Cycle or Layer object!"
        return self._cycles.index(cycle_or_layer, beg, end)

    def insert(self,
            cycle_or_layer: Union[List, Cycle, Layer], idx: int
        ) -> None:
        """Inserts a cycle/layer at index `idx`.

        Args:
            cycle_or_layer (Union[List, Cycle, Layer]): qcal Cycle or Layer object.
            idx (int): index at which the cycle/layer is inserted.
        """
        if isinstance(cycle_or_layer, List):
            cycle_or_layer = Cycle(cycle_or_layer)
        else:
            assert (
                isinstance(cycle_or_layer, Cycle) or 
                isinstance(cycle_or_layer, Layer)
            ), "cycle_or_layer must be a Cycle or Layer object!"
        self._cycles.insert(idx, cycle_or_layer)
        self._update_qubits()

    def measure(self,
            qubits: Union[Tuple, List] = None,
            basis:  Union[Tuple, List] = None,
        ) -> None:
        """Appends a measurement cycle to the end of the circuit.

        Args:
            qubits (Union[Tuple, List], optional): qubits to measure. Defaults
                to None.
            basis (Union[Tuple, List], optional):  measurement basis for each
                qubit. Defaults to None.
        """
        if qubits is None:
            qubits = self._qubits
        if basis is None:
            basis = ('Z',) * len(qubits)

        meas_cycle = Cycle([Meas(q, b) for q, b in zip(qubits, basis)])
        if all([meas.properties['params']['basis'].upper() == 'Z' for meas in 
                meas_cycle]):
            self.append(Barrier(tuple(q for q in self.qubits)))
            self.append(meas_cycle)
        else:
            self.append(
                Cycle([basis_rotation(meas) for meas in meas_cycle])
            )
            self.append(Barrier(tuple(q for q in self.qubits)))
            self.append(meas_cycle)

    def pop(self) -> None:
        """Removes the last cycle/layer to the end of the circuit."""
        self._cycles.pop()
        self._update_qubits()

    def popleft(self) -> None:
        """Removes the first cycle/layer to the end of the circuit."""
        self._cycles.popleft()
        self._update_qubits()

    def prepend(self, cycle_or_layer: Union[List, Cycle, Layer]) -> None:
        """Prepends a cycle/layer to the end of the circuit.

        Args:
            cycle_or_layer (Union[List, Cycle, Layer]): cycle/layer to prepend.
        """
        if isinstance(cycle_or_layer, List):
            cycle_or_layer = Cycle(cycle_or_layer)
        else:
            assert (
                isinstance(cycle_or_layer, Cycle) or 
                isinstance(cycle_or_layer, Layer)
            ), "cycle_or_layer must be a Cycle or Layer object!"
        self._cycles.appendleft(cycle_or_layer)
        self._update_qubits()

    def prepend_circuit(self, circuit) -> None:
        """Prepends another circuit to the beginning of the current circuit..

        Args:
            circuit (List[Cycle], List[Layer], Circuit): circuit to prepend.
        """
        if isinstance(circuit, List):
            circuit = Circuit(circuit)
        else:
            assert (
                isinstance(circuit, Circuit)
            ), "circuit must be a Circuit object!"
        self._cycles.extendleft(circuit._cycles)
        self._update_qubits()

    # TODO
    def relable(self, map: Dict) -> None:
        pass
    
    # TODO
    def remove(self, cycle_or_layer: Union[List, Cycle, Layer]) -> None:
        """Removes the first instance of the cycle/layer found in the circuit.

        Args:
            cycle_or_layer (Union[List, Cycle, Layer]): cycle/layer to remove.
        """
        if isinstance(cycle_or_layer, List):
            cycle_or_layer = Cycle(cycle_or_layer)
        else:
            assert (
                isinstance(cycle_or_layer, Cycle) or 
                isinstance(cycle_or_layer, Layer)
            ), "cycle_or_layer must be a Cycle or Layer object!"
        self._cycles.remove(cycle_or_layer)
        self._update_qubits()

    # TODO    
    def replace(self, 
            old_cycle_or_layer: Union[List, Cycle, Layer],
            new_cycle_or_layer: Union[List, Cycle, Layer]
        ) -> None:
        """Replaces an old cycle/layer with a new one in the circuit.

        Args:
            old_cycle_or_layer (Union[List, Cycle, Layer]): cycle/layer to 
                replace.
            new_cycle_or_layer (Union[List, Cycle, Layer]): new cycle/layer.
        """
        if isinstance(old_cycle_or_layer, List):
            old_cycle_or_layer = Cycle(old_cycle_or_layer)
        else:
            assert (
                isinstance(old_cycle_or_layer, Cycle) or 
                isinstance(old_cycle_or_layer, Layer)
            ), "old_cycle_or_layer must be a Cycle or Layer object!"
        
        if isinstance(new_cycle_or_layer, List):
            new_cycle_or_layer = Cycle(new_cycle_or_layer)
        else:
            assert (
                isinstance(new_cycle_or_layer, Cycle) or 
                isinstance(new_cycle_or_layer, Layer)
            ), "new_cycle_or_layer must be a Cycle or Layer object!"
        
        for i, cycle in enumerate(self._cycles):
            if cycle == old_cycle_or_layer:
                self.remove(old_cycle_or_layer)
                self.insert(new_cycle_or_layer, i)

        self._update_qubits()

    def reverse(self) -> None:
        """Reverses the order of the cycles."""
        self._cycles.reverse()

    def shift(self, num_spaces: int) -> None:
        """Shifts the cycles in the circuit to the right.

        Args:
            num_spaces (int): number of spaces by which to shift all cycles.
        """
        self._cycles.rotate(num_spaces)


class CircuitSet:
    """Class for storing multiple circuits in a single set."""

    __slots__ = '_df'
    
    def __init__(self, 
                 circuits: List[Any] | None = None, 
                 index: List[int] | None = None
        ) -> None:
        """Initialize a CircuitSet.

        Args:
            circuits (List[Any] | None, optional): circuits to store in a 
                CircuitSet. Defaults to None.
            index (List[int] | None, optional): Indices for the circuits in the 
                DataFrame. Defaults to None.
        """
        self._df = pd.DataFrame(columns=['circuit'], dtype='object')

        if circuits is not None:
            if isinstance(circuits, Iterable):
                self._df['circuit'] = circuits
                # self._df['Results'] = [dict()] * len(circuits)
            else:
                self._df['circuit'] = [circuits]
                # self._df['Results'] = [dict()] * len([circuits])
        
        if index is not None:
            self._df = self._df.set_index(pd.Index(index))

    def __getitem__(self, idx_or_label: int | str) -> Circuit | pd.Series:
        """Index the CircuitSet dataframe.

        Args:
            idx_or_label (int | str): argument to index by.

        Returns:
            Circuit | pd.Series: circuit or data series for the given index.
        """
        if isinstance(idx_or_label, int):
            return self._df.iloc[idx_or_label].circuit
        else:
            return self._df[idx_or_label]
        
    def __setitem__(self, label: str, value: Any) -> None:
        """Assign a new column in the dataframe.

        Args:
            label (str): column header.
            value (Any): column data.
        """
        self._df[label] = value    

    def __call__(self) -> pd.DataFrame:
        return self._df

    def __copy__(self) -> CircuitSet:
        """Make a deep copy of the CircuitSet."""
        cs = CircuitSet()
        for col in self._df.columns:
            cs[col] = [copy.deepcopy(c) for c in self._df[col]]
        return cs
    
    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        return iter(self._df.circuit)

    def __repr__(self) -> str:
        return repr(self._df)
    
    def _repr_html_(self) -> str:
        return self._df.to_html()

    @property
    def n_circuits(self) -> int:
        """The total number of circuits in the CircuitSet.

        Returns:
            int: number of circuits
        """
        return self.__len__()

    @property
    def circuit(self) -> pd.Series:
        """The circuits in the CircuitSet.

        Returns:
            pd.Series: circuits
        """
        return self._df['circuit']
    
    @property
    def circuits(self) -> List[Circuit]:
        """A list of the circuits in the CircuitSet.

        Returns:
            List[Circuit]: list of circuits.
        """
        return self.circuit.to_list()

    @property
    def is_empty(self) -> bool:
        """Whether or not the CircuitSet has circuits.

        Returns:
            bool: whether or not the dataframe is empty.
        """
        return False if self.n_circuits > 0 else True
    
    def append(self, circuits, index=None):
        """Appends circuit(s) to the circuit collection."""
        if not isinstance(circuits, CircuitSet):
            circuits = CircuitSet(circuits, index)
        self._df = pd.concat([self._df, circuits._df],
                            ignore_index=True if index is None else False)
        return self._df

    def batch(self, batch_size: int):
        """Batch the circuits into smaller chunks.

        Args:
            batch_size (int): maximum number of circuits per total sequence.

        Yields:
            CircuitSet: CircuitSet of maximum size given by batch_size.
        """
        for i in range(0, len(self), batch_size):
            yield CircuitSet(
                self._df.iloc[i:i + batch_size]['circuit'].tolist()
            )

    def save(self, path: str):
        """Save the CircuitSet dataframe.

        This method pickles the CircuitSet dataframe.

        Args:
            path (str): save path.
        """
        self._df.to_pickle(path)

    # def subset(self, **kwargs) -> pd.DataFrame:
    #     """Subset of the full CircuitSet.

    #     Returns a subset of the full CircuitSet given by the keyword argument.

    #     Returns:
    #         pd.DataFrame: subset of the full CircuitSet
    #     """
    #     df = self._df.copy()  # TODO: deep copy here?
    #     for key in kwargs:
    #         assert key in self._df.columns, f'{key} is not a valid column name.'
    #         df = df.loc[self._df[key] == kwargs[key]]
    #     return df

    # def union_results(self, idx: int = None) -> Dict:
    #     """Compute the union of all of the results.

    #     This can take in an optional index, for which the results will only
    #     be unioned for columns of matching indices.

    #     Args:
    #         idx (int, optional): index over which to union the results.
    #             Defaults to None.

    #     Returns:
    #         Dict: unioned bit string results.
    #     """
    #     if idx is None:
    #         results_list = self._df['results'].to_list()

    #     # TODO: finish

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """Load a saved CircuitSet

        Args:
            path (str): filepath for saved CircuitSet.

        Returns:
            pd.DataFrame: CircuitSet
        """
        return pd.read_pickle(path)
