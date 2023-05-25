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
from qcal.gate.gate import Gate

import copy
# import itertools
import pandas as pd

from collections import Iterable, deque
from typing import Any, Dict, List, Tuple, Union
# from types import IntType, NoneType

__all__ = ['Cycle', 'Layer', 'Circuit', 'CircuitSet']


class Cycle:
    """Class defining a cycle in a circuit."""

    __slots__ = ['_gates', '_qubits']

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
    
    def __call__(self) -> List:
        return self._gates
    
    def __copy__(self) -> List:
        return Cycle(copy.deepcopy(self._gates))
    
    def __repr__(self) -> str:
        df = pd.DataFrame(
            data=[gate.name for gate in self._gates], 
            columns=['Cycle'], 
            index=[gate.qubits for gate in self._gates]
        )
        return repr(df)
    
    def __str__(self) -> str:
        df = pd.DataFrame(
            data=[gate.name for gate in self._gates], 
            columns=['Cycle'], 
            index=[gate.qubits for gate in self._gates]
        )
        return repr(df)
    
    def _repr_html_(self):
        df = pd.DataFrame([
            [(pd.DataFrame(gate.matrix)
                .style
                .format(precision=2)
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
        return [gate.name for gate in self._gates]
    
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

    def to_matrix_table(self) -> pd.DataFrame:
        """Convert the cycle to a table of matrices acting on gate labels.

        Returns:
            pd.DataFrame: table of matrices.
        """
        df = pd.DataFrame(
            data=[gate.matrix.tolist() for gate in self._gates], 
            columns=['Matrix'], 
            index=[gate.qubits for gate in self._gates]
        )
        return df


class Layer(Cycle):
    """Class defining a layer in a circuit."""

    __slots__ = ['_gates', '_qubits']

    def __init__(self, gates: List = []) -> None:
        super().__init__(gates)

    def __copy__(self) -> List:
        return Layer(copy.deepcopy(self._gates))
    

class Circuit:

    __slots__ = ['_cycles']

    def __init__(self,
            cycles_or_layers: List[Union[Cycle, Layer]] = []
        ) -> None:
        
        self._cycles = deque(cycles_or_layers)
        qubits = tuple()
        for cycle in self._cycles:
            qubits += cycle.qubits
        self._qubits = tuple(sorted(set(qubits)))

    def __len__(self) -> int:
        return len(self._cycles)

    @property
    def cycles(self) -> deque:
        """The cycles in the circuit.

        Returns:
            deque: list of cycles.
        """
        return self._cycles
    
    @property
    def layers(self) -> deque:
        """The layers in the circuit.

        Returns:
            deque: list of layers.
        """
        return self._cycles
    
    @property
    def circuit_depth(self) -> int:
        """The number of layers/cycles in the circuit.

        Returns:
            int: circuit depth in terms of layers/cycles.
        """
        return len(self._cycles)
    
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
        return len(self._cycles)
    
    @property
    def n_layers(self) -> int:
        """The number of layers in the circuit.

        Returns:
            int: number of layers.
        """
        return len(self._cycles)
    
    @property
    def n_qubits(self) -> int:
        """The number of qubits in the circuit.

        Returns:
            int: number of qubits.
        """
        return len(self._qubits)
    
    def _update_qubits(self) -> None:
        """Updates the qubits after mutating the cycles."""
        qubits = tuple()
        for cycle in self._cycles:
            qubits += cycle.qubits
        self._qubits = tuple(sorted(set(qubits)))
    
    def append(self, cycle_or_layer: Union[List, Cycle, Layer]) -> None:
        """Appends a cycle/layer to the end of the circuit.

        Args:
            cycle_or_layer (List, Cycle, Layer): cycle/layer to append.
        """
        if isinstance(cycle_or_layer, List):
            cycle_or_layer = Cycle(cycle_or_layer)
        else:
            assert (
                isinstance(cycle_or_layer, Cycle) or 
                isinstance(cycle_or_layer, Layer)
            ), "cycle_or_layer must be a Cycle or Layer object!"
        self._cycles.append(cycle_or_layer)
        self._update_qubits()

    def append_circuit(self, circuit) -> None:
        """Appends another circuit to the end of the current circuit.

        Args:
            circuit (List[Cycle], List[Layer], Circuit): circuit to append.
        """
        if isinstance(circuit, List):
            circuit = Circuit(circuit)
        else:
            assert (
                isinstance(circuit, Circuit)
            ), "circuit must be a Circuit object!"
        self._cycles.extend(circuit._cycles)
        self._update_qubits()

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
            cycle_or_layer (List, Cycle, Layer): cycle/layer to prepend.
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
    
    def __init__(self, circuits: List[Any] = None, index: List[int] = None):
        """
        Args:
            circuits (List[Any], optional): Circuits to store in the 
                CircuitSet. Defaults to None.
            index (list[int], optional): Indices for the circuits in the 
                DataFrame. Defaults to None.
        """

        self._df = pd.DataFrame(columns=['Circuits', 'Results'])

        if circuits is not None:  # TODO: how to handle circuits that already have results
            if isinstance(circuits, Iterable):
                self._df['Circuits'] = circuits
                self._df['Results'] = [dict()] * len(circuits)
            else:
                self._df['Circuits'] = [circuits]
                self._df['Results'] = [dict()] * len([circuits])
        
        if index is not None:
            self._df = self._df.set_index(pd.Index(index))

    def __call__(self) -> pd.DataFrame:
        return self._df

    def __copy__(self):  # -> CircuitSet:
        """Make a deep copy of the CircuitSet."""
        cs = CircuitSet()
        for col in self._df.columns:
            cs[col] = [copy.deepcopy(c) for c in self._df[col]]
        return cs
    
    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        return iter(self._df)

    def __repr__(self) -> str:
        return repr(self._df)

    @property
    def n_circuits(self) -> int:
        """The total number of circuits in the CircuitSet.

        Returns:
            int: number of circuits
        """
        return self.__len__()

    @property
    def circuits(self) -> pd.Series:
        """The circuits in the CircuitSet.

        Returns:
            pd.Series: circuits
        """
        return self._df['Circuits']

    @property
    def results(self) -> pd.Series:
        return self._df['Results']
    
    def append(self, circuits, index=None):
        """Appends circuit(s) to the circuit collection."""
        if not isinstance(circuits, CircuitSet):
            circuits = CircuitSet(circuits, index)
        self._df = pd.concat([self._df, circuits.df],
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
                self._df.iloc[i:i + batch_size]['Circuits'].tolist()
            )

    def save(self, path: str):
        """Save the CircuitSet dataframe.

        This method pickles the CircuitSet dataframe.

        Args:
            path (str): save path.
        """
        self._df.to_pickle(path)

    def subset(self, **kwargs) -> pd.DataFrame:
        """Subset of the full CircuitSet.

        Returns a subset of the full CircuitSet given by the keyword argument.

        Returns:
            pd.DataFrame: subset of the full CircuitSet
        """
        df = self._df.copy()  # TODO: deep copy here?
        for key in kwargs:
            assert key in self._df.columns, f'{key} is not a valid column name.'
            df = df.loc[self._df[key] == kwargs[key]]
        return df

    def union_results(self, idx: int = None) -> Dict:
        """Compute the union of all of the results.

        This can take in an optional index, for which the results will only
        be unioned for columns of matching indices.

        Args:
            idx (int, optional): index over which to union the results.
                Defaults to None.

        Returns:
            Dict: unioned bit string results.
        """
        if idx is None:
            results_list = self._df['Results'].to_list()

        # TODO: finish

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """Load a saved CircuitSet

        Args:
            path (str): filepath for saved CircuitSet.

        Returns:
            pd.DataFrame: CircuitSet
        """
        return pd.read_pickle(path)
