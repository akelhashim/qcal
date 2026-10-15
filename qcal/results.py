"""Submodule for storing and handling bitstring results.

All results are stored in a Results object.
"""
from __future__ import annotations

import copy
import itertools
import logging
from collections import defaultdict
from functools import reduce
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame

from qcal.math.entropy import shannon_entropy
from qcal.math.probability import total_variation_distance

logger = logging.getLogger(__name__)


__all__ = ('qp_readout_correction', 'readout_correction', 'Results')


# TODO: add fidelity and TVD
class Results:
    """Results class.

    This class should be passed a dictionary which maps bitstrings to counts.

    Basic example useage:
    ```results = Results({'000': 200, '010': 10, '100': 12, '111': 200})```
    """

    def __init__(self,
        results:          Dict | None = None,
        confusion_matrix: DataFrame | None = None,
        rr_char_results:  Results | Dict | None = None,
    ) -> None:
        """Initialize a Results object.

        Args:
            results (Dict | None, optional): dictionary of bitstring results.
                Defaults to None.
            confusion_matrix (DataFrame | None, optional): confusion matrix
                used for readout correction. Defaults to None.
            rr_char_results (Results | Dict | None, optional): results of
                measuring the all-zero state with randomized readout, used
                for quasi-probabilistic readout correction (with default
                settings). Defaults to None.

        Raises:
            ValueError: if both ``confusion_matrix`` and ``rr_char_results``
                are provided.
        """
        if confusion_matrix is not None and rr_char_results is not None:
            raise ValueError(
                'Only one of confusion_matrix or rr_char_results can be'
                ' provided!'
            )

        if results is not None:
            results = dict(sorted(results.items()))
        else:
            results = {}

        self._results = results
        self.__raw_results = copy.deepcopy(results)
        self._confusion_matrix = None
        self._rr_char_results = None
        self.__df = None

        if confusion_matrix is not None:
            self.apply_readout_correction(confusion_matrix)
        elif rr_char_results is not None:
            self.apply_qp_readout_correction(rr_char_results)

    def _rebuild_df(self) -> None:
        """(Re)build the counts/probabilities DataFrame from self._results.

        The counts row is assigned first (not via the `_df` property) so
        that `self.populations` (via `n_shots`/`counts`) reads the
        already-cached counts row instead of recursing back into `_df`.
        """
        self.__df = pd.DataFrame(
            [self._results], index=['counts'], dtype='object'
        )
        self.__df = pd.concat(
            [self.__df,
             pd.DataFrame([self.populations], index=['probabilities'])
            ],
            join='inner'
        )

    @property
    def _df(self) -> pd.DataFrame:
        """Lazily-built counts/probabilities DataFrame.

        Deferred until first access: most Circuit objects are constructed
        without results ever being assigned to them (e.g. transient
        sub-circuits during circuit generation), so building this eagerly
        in `__init__` wastes a pandas DataFrame + concat on every one of
        them.
        """
        if self.__df is None:
            self._rebuild_df()
        return self.__df

    def __getitem__(self, item: str) -> pd.Series:
        """Index the dataframe by bitstring.

        Args:
            item (str): bitstring label.

        Returns:
            pd.Series: dataseries of counts and probabilities for a given
                bistring.
        """
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
    def confusion_matrix(self) -> DataFrame | None:
        """Confusion matrix used to correct the results.

        Returns:
            DataFrame | None: confusion matrix, or None if the results have
                not been corrected using a confusion matrix.
        """
        return self._confusion_matrix

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
    def dict(self) -> defaultdict:
        """Dictionary of bitstrings and counts.

        Returns:
            defaultdict: dictionary of results.
        """
        return self._results

    @property
    def entropy(self) -> float:
        """Shannon entropy of the results.

        Returns:
            float: entropy.
        """
        return shannon_entropy(self)

    @property
    def ev(self) -> float:
        """Expectation value of the measured observable.

        Returns:
            float: expectation value.
        """
        counts_p = 0
        counts_m = 0
        for state in self.states:
            if state.count('1') % 2 == 1:
                counts_m += self._results[state]
            else:
                counts_p += self._results[state]
        return (counts_p - counts_m) / self.n_shots

    @property
    def n_shots(self) -> int:
        """Total number of shots.

        Returns:
            int: number of shots.
        """
        return self.counts.sum()

    @property
    def n_qudits(self) -> int:
        """Total number of qudits.

        Returns:
            int: number of qudits.
        """
        return len(self.states[0])

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
    def raw_results(self) -> Dict:
        """Dictionary of uncorrected bitstrings and counts.

        A copy is returned so that the raw results, from which any readout
        correction is computed, cannot be modified.

        Returns:
            Dict: raw results.
        """
        return dict(self.__raw_results)

    @property
    def rr_char_results(self) -> Results | Dict | None:
        """Randomized readout characterization results used to correct the
        results.

        Returns:
            Results | Dict | None: results of measuring the all-zero state
                with randomized readout, or None if the results have not
                been corrected using quasi-probabilistic readout correction.
        """
        return self._rr_char_results

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
        # Always correct the raw results so that repeated calls do not
        # compound the correction
        self._results = copy.deepcopy(self.__raw_results)
        self.__df = None
        self._confusion_matrix = None
        self._rr_char_results = None
        try:
            corrected = readout_correction(self, confusion_matrix)
            # readout_correction returns self._results unchanged if the
            # confusion matrix does not match the bitstring length
            if corrected is not self._results:
                self._results = corrected
                self._confusion_matrix = confusion_matrix
        except Exception as e:
            logger.warning(
                f' Readout correction failed: {e} No readout correction'
                ' will be performed!'
            )
            self._results = copy.deepcopy(self.__raw_results)
        self.__df = None

    def apply_qp_readout_correction(
        self,
        rr_char_results: Results | Dict,
        order:           int = 2,
        clip_to_zero:    bool = True,
        threshold:       float = 1e-8,
    ) -> None:
        """Apply quasi-probabilistic readout correction (QPRC).

        See ``qp_readout_correction`` for details. The results must have
        been obtained using randomized readout.

        Args:
            rr_char_results (Results | Dict): results of measuring the
                all-zero state with randomized readout.
            order (int, optional): order k of the quasi-probabilistic
                inverse. Defaults to 2.
            clip_to_zero (bool, optional): whether to set negative corrected
                counts to zero. Defaults to True.
            threshold (float, optional): quasi-probabilities with magnitude
                less than or equal to this value are discarded when
                constructing the inverse. Defaults to 1e-8.
        """
        # Always correct the raw results so that repeated calls do not
        # compound the correction
        self._results = copy.deepcopy(self.__raw_results)
        self.__df = None
        self._confusion_matrix = None
        self._rr_char_results = None
        try:
            self._results = dict(sorted(qp_readout_correction(
                self,
                rr_char_results,
                order=order,
                clip_to_zero=clip_to_zero,
                threshold=threshold
            ).items()))
            self._rr_char_results = rr_char_results
        except Exception as e:
            logger.warning(
                f' QP readout correction failed: {e} No readout correction'
                ' will be performed!'
            )
            self._results = copy.deepcopy(self.__raw_results)
        self.__df = None

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

        marg_results = dict.fromkeys(marg_states, 0)
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

        # Scale the width with the number of states, but keep it within
        # reasonable bounds so that large dit string sets do not produce an
        # arbitrarily wide figure
        n_states = len(self.states)
        width = min(max(100 * n_states, 400), 1000)

        # Rotate the tick labels if they are wider than the space allotted
        # to each bar (~9 px per character at font size 15, ~150 px margins)
        px_per_bar = (width - 150) / n_states
        rotate = 9 * len(self.states[0]) > px_per_bar

        fig.update_traces(marker_color='blue')
        fig.update_layout(
            autosize=False,
            width=width,
            height=400,
            xaxis={
                'tickvals': list(range(len(self.states))),
                'ticktext': self.states,
                'title': 'Dit String',
                'titlefont_size': 20,
                'tickfont_size': 15,
                'tickangle': -45 if rotate else 0
            },
            yaxis={
                'title': title,
                'titlefont_size': 20,
                'tickfont_size': 15
            }
        )

        save_properties = {
            'toImageButtonOptions': {
                'format': 'png', # one of png, svg, jpeg, webp
                'filename': 'results',
                'height': 500,
                'width': 1000,
                'scale': 10 # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        fig.show(config=save_properties)

    def tvd(self, results: Results) -> float:
        """Total Variation Distance

        Args:
            results (Results): other results.

        Returns:
            float: tvd.
        """
        return total_variation_distance(self, results)


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
        cmats = [confusion_matrix.to_numpy().astype(float)]
    else:
        cmats = []
        qubits = set()
        for col in confusion_matrix:
            qubits.add(col[0])
        qubits = tuple(sorted(qubits, key=lambda x: int(x[1:])))
        for q in qubits:
            cmats.append(
                confusion_matrix[q].to_numpy().astype(float)
            )

    if len(cmats) != len(tuple(results._results.keys())[0]):
        logger.warning(
            ' The number of confusion matrices does not match the length'
            ' of the provided bitstrings. No readout correction will be'
            ' performed!'
        )
        return results.dict

    else:
        # Tensor product of confusion matricies
        cmat_inv = reduce(np.kron,
            [np.linalg.inv(cmat) for cmat in cmats]
        ).astype(float)

        # Generate all possible dit strings
        dtstrs = [
            ''.join(i) for i in itertools.product(
                [str(j) for j in results.levels],
                repeat=len(results.states[0])
            )
        ]

        # Probability distribution over all possible dit strings
        probs = {
            **dict.fromkeys(dtstrs, 0.0),
            **dict(results.probabilities.astype(float))
        }
        probs = np.array(list(probs.values()))#.reshape(-1, 1)

        # Perform the readout correction
        corr_probs = np.array(
            # probs @ np.linalg.inv(cmat) * results.n_shots
            probs @ cmat_inv * results.n_shots
        ).astype(int)

        corrected_results = dict(zip(dtstrs, corr_probs, strict=False))

    return corrected_results


def qp_readout_correction(
    results:          Results | Dict,
    rr_char_results:  Results | Dict,
    order:            int = 2,
    clip_to_zero:     bool = True,
    threshold:        float = 1e-8,
) -> Dict:
    """Quasi-probabilistic readout correction (QPRC).

    Implements the readout correction scheme of
    https://arxiv.org/abs/2312.14139. Under measurement randomized compiling
    (MRC), readout noise is tailored into a stochastic bit-flip channel
    described by a probability distribution p over error bitstrings x. This
    distribution is obtained by measuring the all-zero state with randomized
    readout (``rr_char_results``). Writing e = sum_{x != 0} p_x x, the order-k
    quasi-probabilistic inverse is

        q^(k) = c * sum_{j=0}^{2k-1} (-e / p_0)^j,

    where powers denote convolution over bitwise addition mod 2 and c
    normalizes q^(k) to unit sum. For k = 1, this reduces to
    q = (p_0 0 - e) / (2 p_0 - 1). Residual readout errors are suppressed
    from O(1 - p_0) to O((1 - p_0)^(2k)). The corrected counts are given by
    the convolution of the measured counts with q^(k).

    Bitstrings are encoded as integers and convolved in a vectorized,
    chunked manner, so the cost scales with the number of observed
    bitstrings rather than with 2^n.

    Args:
        results (Results | Dict): measured results to correct. These must be
            obtained using randomized readout.
        rr_char_results (Results | Dict): characterization results of
            measuring the all-zero state with randomized readout.
        order (int, optional): order k of the quasi-probabilistic inverse.
            Defaults to 2.
        clip_to_zero (bool, optional): whether to set negative corrected
            counts to zero. Defaults to True.
        threshold (float, optional): quasi-probabilities with magnitude
            less than or equal to this value are discarded when constructing
            the inverse. This bounds the growth of the support for
            ``order > 1``. Defaults to 1e-8.

    Returns:
        Dict: corrected results dictionary.

    Raises:
        ValueError: if the bitstrings are not binary, do not have consistent
            lengths, if ``order < 1``, or if p_0 <= 0.5 (in which case the
            inverse is undefined).
    """
    if order < 1:
        raise ValueError('order must be >= 1!')

    # Encode each bitstring as an integer so that bitwise addition mod 2
    # becomes a vectorized XOR. Every distribution below is stored sparsely
    # as a pair of arrays: (keys, vals) = (bitstrings, weights).
    b_keys, b_vals, n = _to_xor_arrays(results)
    p_keys, p_vals, n_rr_char = _to_xor_arrays(rr_char_results)
    if n != n_rr_char:
        raise ValueError(
            f'Bitstring length of the results ({n}) does not match that of'
            f' the RR char results ({n_rr_char})!'
        )

    # Under RR, measuring the all-zero state yields the error bitstring
    # directly, so the normalized RR results are the error distribution p.
    # p0 is the probability of no readout error. The inverse requires
    # p0 > 0.5, since otherwise the normalization (2 p0 - 1 for k = 1)
    # vanishes or changes sign.
    p_vals = p_vals / p_vals.sum()
    is_zero = p_keys == 0
    p0 = p_vals[is_zero].sum()
    if p0 <= 0.5:
        raise ValueError(
            f'The all-zero probability ({p0:.3f}) must be greater than 0.5'
            ' for the readout inverse to exist!'
        )

    # Build q^(k) = c * sum_{j=0}^{2k-1} (-e / p0)^j, i.e. the Neumann
    # series of p^-1 = (1 / p0) (0 + e / p0)^-1 truncated after 2k terms.
    # e_vals holds -e / p0: the non-zero error strings, rescaled and negated.
    e_keys, e_vals = p_keys[~is_zero], -p_vals[~is_zero] / p0
    # The j = 0 term is the identity for XOR convolution: all weight on the
    # all-zero bitstring. q accumulates the series; t holds the current
    # term (-e / p0)^j.
    q_keys = t_keys = np.zeros(1, dtype=p_keys.dtype)
    q_vals = t_vals = np.ones(1)
    # If no errors were observed, p is the identity and so is its inverse
    if len(e_keys) > 0:
        for _ in range(2 * order - 1):
            # Next term: (-e / p0)^j = (-e / p0)^(j-1) * (-e / p0), dropping
            # entries with magnitude <= threshold to bound the support
            t_keys, t_vals = _xor_convolve(
                t_keys, t_vals, e_keys, e_vals, threshold
            )
            # Add the term to the running sum, merging duplicate bitstrings
            q_keys, q_vals = _xor_reduce(
                np.concatenate([q_keys, t_keys]),
                np.concatenate([q_vals, t_vals]),
                threshold
            )
    # Without truncation, the sum is (p0^2k - (1 - p0)^2k) / p0^(2k-1)
    q_vals = q_vals / q_vals.sum()

    keys, vals = _xor_convolve(b_keys, b_vals, q_keys, q_vals)
    if clip_to_zero:
        vals = np.clip(vals, 0., None)
    counts = np.rint(vals).astype(np.int64)

    return {
        format(int(k), f'0{n}b'): int(c)
        for k, c in zip(keys, counts, strict=True) if c != 0
    }


def _to_xor_arrays(results: Results | Dict) -> Tuple[np.ndarray, ...]:
    """Convert bitstring results to integer keys and float values.

    Bitstrings are encoded as integers so that bitwise addition mod 2 can be
    performed with ``np.bitwise_xor``. Bitstrings longer than 64 bits fall
    back to arbitrary-precision Python integers.

    Args:
        results (Results | Dict): results object or dictionary mapping
            bitstrings to counts.

    Returns:
        Tuple[np.ndarray, ...]: integer keys, float values, and bitstring
            length.
    """
    results = results.dict if isinstance(results, Results) else results
    lengths = {len(s) for s in results}
    if len(lengths) != 1:
        raise ValueError('All bitstrings must have the same length!')
    n = lengths.pop()
    if any(set(s) - {'0', '1'} for s in results):
        raise ValueError('Bitstrings must be binary!')

    dtype = np.uint64 if n <= 64 else object
    keys = np.array([int(s, 2) for s in results], dtype=dtype)
    vals = np.array(list(results.values()), dtype=float)
    return keys, vals, n


def _xor_reduce(
    keys: np.ndarray, vals: np.ndarray, threshold: float = 0.
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum values with identical keys and discard negligible entries.

    Args:
        keys (np.ndarray): integer-encoded bitstrings (may repeat).
        vals (np.ndarray): (quasi-)probabilities or counts.
        threshold (float, optional): entries with magnitude less than or
            equal to this value are discarded. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: unique keys and summed values.
    """
    uniq, inv = np.unique(keys, return_inverse=True)
    summed = np.bincount(inv.ravel(), weights=vals, minlength=len(uniq))
    mask = np.abs(summed) > threshold
    return uniq[mask], summed[mask]


def _xor_convolve(
    keys1: np.ndarray,
    vals1: np.ndarray,
    keys2: np.ndarray,
    vals2: np.ndarray,
    threshold: float = 0.,
    max_size: int = 2**22,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convolve two sparse distributions over bitwise addition mod 2.

    Computes r(z) = sum_{x ^ y = z} a(x) b(y). The outer product is evaluated
    in chunks of at most ``max_size`` elements to bound memory usage.

    Args:
        keys1 (np.ndarray): integer-encoded bitstrings of the first
            distribution.
        vals1 (np.ndarray): values of the first distribution.
        keys2 (np.ndarray): integer-encoded bitstrings of the second
            distribution.
        vals2 (np.ndarray): values of the second distribution.
        threshold (float, optional): entries with magnitude less than or
            equal to this value are discarded. Defaults to 0.
        max_size (int, optional): maximum number of elements in each chunk
            of the outer product. Defaults to 2**22.

    Returns:
        Tuple[np.ndarray, np.ndarray]: keys and values of the convolution.
    """
    rows = max(1, max_size // max(1, len(keys2)))
    chunks = [
        _xor_reduce(
            np.bitwise_xor.outer(keys1[i:i + rows], keys2).ravel(),
            np.multiply.outer(vals1[i:i + rows], vals2).ravel(),
        )
        for i in range(0, len(keys1), rows)
    ]
    return _xor_reduce(
        np.concatenate([k for k, _ in chunks]),
        np.concatenate([v for _, v in chunks]),
        threshold
    )
