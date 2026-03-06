"""Submodule for post-processing passes.

This submodule contains a collection of passes for post-processing measurement
results. Each pass is a function that takes a PostProcessor object as an
argument and performs some operation on the results.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from qcal.post_processing.post_process import PostProcessor

logger = logging.getLogger(__name__)


__all__ = [
    'compute_counts',
    'compute_conditional_counts',
    'discard_heralded_shots',
    'relabel_esp',
]


def compute_counts(p: PostProcessor) -> None:
    """Compute the counts for all unique combinations of 0s, 1s, etc.

    Args:
        p (PostProcessor): post-processor object.
    """
    p._circuit_results = [
        [result.value_counts().rename('counts').reset_index()
         for result in p._circuit_results[i]
        ]
        for i in range(len(p._circuit_results))
    ]

    for circuit_results in p._circuit_results:
        for result in circuit_results:
            result['bitstring'] = result[list(p._qubits)].apply(
                lambda row: ''.join(row.values.astype(str)), axis=1
            )

    p._circuit_results = [
        [dict(zip(result.bitstring, result.counts, strict=False))
         for result in p._circuit_results[i]
        ]
        for i in range(len(p._circuit_results))
    ]


def compute_conditional_counts(p: PostProcessor) -> None:
    """Compute the conditional counts for all unique combinations of 0s, 1s, ...

    This pass takes in sequential dataframes of mid-circuit and terminating
    measurements an computes a dictionary mapping mapping 'b0:b1:...:bk-1' ->
    count for each circuit, where b_i is the bitstring of the i-th measurement,
    and b_i:b_j denotes the bitstring of the j-th measurement that is
    conditioned on the i-th measurement, etc. This is extended to the case where
    there are multiple qubits measured in each mid-circuit measurement, which
    would yield an outcome of the form '<mid0>:<mid1>:...:<final>', where <mid>
    is the combined bitstring of the mid-circuit measurement and <final> is the
    bitstring of the final measurement (e.g., '010:110').

    Args:
        p (PostProcessor): post-processor object.
    """
    conditional_counts = []
    for circuit_results in p._circuit_results:
        n = len(circuit_results[0])
        if any(len(df) != n for df in circuit_results):
            raise ValueError(
                "All dataframes must have the same number of rows (shots)!"
            )
        if any(
            not isinstance(df, pd.DataFrame) or df.shape[1] < 1
            for df in circuit_results
        ):
            raise ValueError(
                "Each element of dfs must be a DataFrame with >= 1 column!"
            )

        outcomes = []

        # Mid-circuit measurements
        for df in circuit_results[:-1]:
            mapped = df.astype(int).replace(
                {0: 'p0', 1: 'p1', '2': 'p2', '3': 'p3'}
            ).astype(str)
            outcomes.append(  # e.g. "p0p1p0"
                mapped.agg("".join, axis=1).reset_index(drop=True)
            )

        # Terminating measurement, e.g. "010"
        final = circuit_results[-1].astype(int).astype(str).agg(
            "".join, axis=1
        ).reset_index(drop=True)
        outcomes.append(final)

        combined = (":".join(parts) for parts in zip(*outcomes, strict=True))
        conditional_counts.append([dict(Counter(combined))])

    p._circuit_results = conditional_counts


def discard_heralded_shots(
    p: PostProcessor
) -> None:
    """Discard heralded shots for each circuit.

    Args:
        p (PostProcessor): post-processor object.
        herald_idx (int): measurement index for heralding.
    """
    if p.config['readout/herald']:
        # Boolean series
        pass_herald = [
            pd.DataFrame( # only columns with all True
                {q: p.results[i][q][:,p._herald_idx] for q in p._qubits}
        ).isin([0]).all(1) for i in range(len(p._circuit_results))
        ]
        p._circuit_results = [
            [result[pass_herald[i]] for result in p._circuit_results[i]]
            for i in range(len(p._circuit_results))
        ]


def relabel_esp(p: PostProcessor) -> None:
    """Relable all 2s as 1s for excited state promotion.

    Args:
        p (PostProcessor): post-processor object.
    """
    if p.config['readout/esp/enable']:
        p._circuit_results = [
            [result.replace(2, 1) for result in p._circuit_results[i]]
            for i in range(len(p._circuit_results))
        ]
