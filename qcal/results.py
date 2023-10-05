"""Submodule for storing and handling bitstring results.

All results are stored in a Results object.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from collections import defaultdict
from typing import Dict, Tuple

__all__ = ('Results')

# TODO: add fidelity and TVD
class Results:
    """Results class.

    This class should be passed a dictionary which maps bitstrings to counts.
    
    Basic example useage:

        results = Results({'000': 200, '010': 10, '100': 12, '111': 200})
    """

    def __init__(self, results: dict = {}) -> None:
        """Initialize a Results object.

        Args:
            results (dict, optional): dictionary of bitstring results. 
                Defaults to {}.
        """
        results = dict(sorted(results.items()))
        self._dict = results
        self._df = pd.DataFrame([results], index=['counts'], dtype='object')
        self._df = pd.concat(
            [self._df,
             pd.DataFrame([self.populations], index=['probabilities'])],
            join='inner'
        )

    def __getitem__(self, item: str) -> pd.Series:
        """Index the dataframe by bitstring.

        Args:
            item (str): bitstring label.

        Returns:
            pd.Series: dataseries of counts and probabilities for a given
                bistring.
        """
        assert item in self._dict.keys(), f'{item} is not a valid bitstring!'
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
        return self._dict

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
            pop[state] = self._dict[state] / self.n_shots
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
        for key in self._dict.keys():
            for i in key:
                levels.add(int(i))
        return tuple(sorted(levels))

    @property
    def states(self) -> Tuple:
        """Distinct bitstrings in the results.

        Returns:
            Tuple: unique bitstrings.
        """
        return tuple(sorted(self._dict.keys()))

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
        for btstr, counts in self._dict.items():
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
            width=100 * len(self.states),
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