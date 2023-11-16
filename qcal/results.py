"""Submodule for storing and handling bitstring results.

All results are stored in a Results object.
"""
from __future__ import annotations

import itertools
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from collections import defaultdict
from typing import Dict, Tuple
from pandas import DataFrame

logger = logging.getLogger(__name__)


__all__ = ('readout_correction', 'Results')


def readout_correction(results: Results, confusion_matrix: DataFrame) -> Dict:
    """Perform readout correction using a measured confusion matrix.

    This readout correction is performed individually on each qubit.
    Therefore, it is scalable, but will not correct correlated readout
    errors.

    Args:
        results (Results): Results object.
        confusion_matrix (DataFrame): confusion matrix measured using the
            ```qcal.benchmarking.fidelity.ReadoutFidelity``` module.

    Returns:
        Dict: corrected results dictionary.
    """
    if confusion_matrix.columns[0][0] == 'Meas State':
        cmats = [confusion_matrix.to_numpy().astype(float).T]
    else:
        cmats = []
        qubits = set()
        for col in confusion_matrix:
            qubits.add(col[0])
        qubits = tuple(sorted(qubits))
        for q in qubits:
            cmats.append(
                confusion_matrix[q].to_numpy().astype(float).T
            )

    if len(cmats) != len(tuple(results._results.keys())[0]):
        logger.warning(
            ' The number of confusion matrices does not match the length'
            ' of the provided bitstrings. No readout correction will be'
            ' performed!'
        )
        return results

    else:
        corrected_results = {}
        cmats_inv = [np.linalg.inv(cmat) for cmat in cmats]
        btstrs = [
            ''.join(i) for i in itertools.product(
                [str(j) for j in results.levels], 
                repeat=len(results.states[0])
            )
        ]
        # Observable, i.e. measured bitstrings (e.g. '00', '10', etc.)
        for obsrv in results.states:

            ideal_dists = []  # List of ideal distributions
            for o in obsrv:
                # print(o)
                dist = np.zeros(results.dim)
                # print(dist)
                dist[int(o)] = 1.
                ideal_dists.append(dist)

            corrected_value = 0.
            for btstr in btstrs:  # 00, 01, 10, 11
                coeff = 1.  # Coefficient for capturing readout errors
                for q in range(len(results.states[0])):  # Qubit index
                    coeff *= np.dot(
                        ideal_dists[q], cmats_inv[q][:, int(btstr[q])]
                    )
                corrected_value += coeff * results[btstr].counts

            corrected_results[obsrv] = max(0, int(corrected_value))

        return corrected_results


# TODO: add fidelity and TVD
class Results:
    """Results class.

    This class should be passed a dictionary which maps bitstrings to counts.
    
    Basic example useage:
    ```results = Results({'000': 200, '010': 10, '100': 12, '111': 200})```
    """

    def __init__(self,
            results: Dict | None = None, 
            confusion_matrix: DataFrame | None = None
        ) -> None:
        """Initialize a Results object.

        Args:
            results (Dict | None, optional): dictionary of bitstring results.
                Defaults to None.
            confusion_matrix (DataFrame | None, optional): confusion matrix 
                used for readout correction. Defaults to None.
        """
        if results is not None:
            results = dict(sorted(results.items()))
        else:
            results = dict()
        
        self._results = results
        self._df = pd.DataFrame([results], index=['counts'], dtype='object')
        self._df = pd.concat(
            [self._df,
             pd.DataFrame([self.populations], index=['probabilities'])
            ],
            join='inner'
        )

        if confusion_matrix is not None:
            self.apply_readout_correction(confusion_matrix)

    def __getitem__(self, item: str) -> pd.Series:
        """Index the dataframe by bitstring.

        Args:
            item (str): bitstring label.

        Returns:
            pd.Series: dataseries of counts and probabilities for a given
                bistring.
        """
        # assert item in self._results.keys(), f'{item} is not a valid bitstring!'
        if item not in self._results.keys():
            return pd.Series(
                data=[0, 0.], 
                index=['counts', 'probabilities'], 
                name=item, 
                dtype='object'
            )
        else:
            return self._df[item]

    def __repr__(self) -> str:
        return str(self._df)

    def __str__(self) -> str:
        return str(self._df)

    def _repr_html_(self):
        return self._df.to_html()

    @property
    def counts(self) -> pd.Series:
        """Counts for each bitstring.

        Returns:
            pd.Series: integer counts.
        """
        return self._df.loc['counts']

    @property
    def dim(self) -> int:
        """Dimension of the results (e.g. 2 for qubits, 3 for qutrits, etc.).

        Returns:
            int: dimension.
        """
        return len(self.levels)

    @property
    def df(self) -> pd.DataFrame:
        """DataFrame of counts and probabilities for each bitstring.

        Returns:
            pd.DataFrame: DataFrame of results.
        """
        return self._df

    @property
    def dictionary(self) -> defaultdict:
        """Dictionary of bitstrings and counts.

        Returns:
            defaultdict: dictionary of results.
        """
        return self._results

    @property
    def n_shots(self) -> int:
        """Total number of shots.

        Returns:
            int: number of shots.
        """
        return self.counts.sum()

    @property
    def populations(self) ->  Dict:
        """Populations of each bitstring.

        Returns:
            Dict: populations.
        """
        pop = defaultdict(lambda: 0.)
        for state in self.states:
            pop[state] = self._results[state] / self.n_shots
        return pop

    @property
    def probabilities(self) -> pd.Series:
        """Probabilities of each bitstring.

        This is the same as self.populations, but stored in a DataFrame.

        Returns:
            pd.Series: _description_
        """
        return self._df.loc['probabilities']

    @property
    def levels(self) -> Tuple:
        """Energy levels in the results (e.g. (1, 2, 3) for qutrit results).

        Returns:
            Tuple: energy levels.
        """
        levels = set()
        for key in self._results.keys():
            for i in key:
                levels.add(int(i))
        return tuple(sorted(levels))

    @property
    def states(self) -> Tuple:
        """Distinct bitstrings in the results.

        Returns:
            Tuple: unique bitstrings.
        """
        return tuple(sorted(self._results.keys()))
    
    def apply_readout_correction(self, confusion_matrix: DataFrame) -> None:
        """Apply local readout correction using a measured confusion matrix.

        Args:
            confusion_matrix (DataFrame): confusion matrix measured using the
                ```qcal.benchmarking.fidelity.ReadoutFidelity``` module.
        """
        self._raw_results = self._results.copy()
        self._confusion_matrix = confusion_matrix
        self._results = readout_correction(self, confusion_matrix)
        self._df = pd.DataFrame(
            [self._results], index=['counts'], dtype='object'
        )
        self._df = pd.concat(
            [self._df,
             pd.DataFrame([self.populations], index=['probabilities'])
            ],
            join='inner'
        )

    def marginalize(self, idx: int | Tuple[int]):
        """Marginalize the results over a given bistring index.

        This method excepts a single index (e.g. 0) or a tuple of indicies
        (e.g. (0, 2)). The bitstring results will be marginalized over these
        indices. For example, for idx = (0, 2), the bitstring '012' will be
        marginalized to '02', etc.

        Args:
            idx (int | Tuple[int]): bitstring indices to marginalize over.

        Returns:
            Results: marginalized results.
        """
        idx = (idx,) if isinstance(idx, int) else idx
        marg_states = set()
        for s in self.states:
            marg_state = ''
            for i in idx:
                marg_state += s[i]
            marg_states.add(marg_state)
        marg_states = tuple(sorted(marg_states))

        marg_results = {state: 0 for state in marg_states}
        for btstr, counts in self._results.items():
            marg_state = ''
            for i in idx:
                marg_state += btstr[i]
            marg_results[marg_state] += counts

        return Results(marg_results)

    def plot(self, normalize: bool = False) -> None:
        """Plot the results in a histogram.

        Args:
            normalize (bool, optional): whether to plot the normalized counts. 
                Defaults to False.
        """
        fig = go.Figure()
        if normalize:
            title = 'Probability'
            fig.add_trace(go.Bar(y=self.probabilities))
        else:
            title = 'Counts'
            fig.add_trace(go.Bar(y=self.counts))

        fig.update_traces(marker_color='blue')
        fig.update_layout(
            autosize=False,
            width=175 * len(self.states),
            height=400,
            xaxis=dict(
                tickvals=[i for i in range(len(self.states))],
                ticktext=self.states,
                title='Bit String',
                titlefont_size=20,
                tickfont_size=15
            ),
            yaxis=dict(
                title=title,
                titlefont_size=20,
                tickfont_size=15
            )
        )
        if len(self.states[0]) > 5:
            fig.update_xaxes(tickangle=-45)
        fig.show()
                