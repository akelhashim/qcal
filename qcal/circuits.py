"""Submodule for handling many circuits in CircuitSet.

CircuitSet takes a list of circuits and loads them into a dataframe that can be
used to store other useful information about the circuits, enabling fast and
easy sorting of circuits by arbitrary variables.

Basic example useage:

    cs = CircuitSet([list of circuits])
"""

import copy
import pandas as pd

from collections.abc import Iterable
from typing import Any, Dict, List, Optional

__all__ = ['load', 'CircuitSet']


def load(path: str) -> pd.DataFrame:
    """Load a saved CircuitSet

    Args:
        path (str): filepath for saved CircuitSet.

    Returns:
        pd.DataFrame: CircuitSet
    """
    return pd.read_pickle(path)


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
        """Returns the total number of circuits in the CircuitSet.

        Returns:
            int: number of circuits
        """
        return self.__len__()

    @property
    def circuits(self) -> pd.Series:
        """Returns the circuits in the CircuitSet.

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
