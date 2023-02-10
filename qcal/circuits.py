""" 
Module for handling many circuits in CircuitSet
"""

import copy
import pandas as pd

from collections.abc import Iterable
from typing import Any, Dict, List, Optional

__all__ = ['CircuitSet']


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

        if circuits is not None:
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
        """Returns a deep copy of the CircuitSet."""
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
        return self.__len__()

    @property
    def circuits(self) -> pd.Series:
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
            batch_size (int): _description_

        Yields:
            CircuitSet: CircuitSet of maximum size given by batch_size.
        """
        for i in range(0, len(self), batch_size):
            yield CircuitSet(
                self._df.iloc[i:i + batch_size]['Circuits'].tolist()
            )

    def union_results(self, idx=None) -> Dict:
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

    def save(self):
        pass

