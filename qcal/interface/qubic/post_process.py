"""Submodule for post-processing results measured on QubiC.

"""
from qcal.circuit import CircuitSet
from qcal.config import Config

import logging
import numpy as np
import pandas as pd

from typing import Any, List

logger = logging.getLogger(__name__)


__all__ = ('post_process')


def find_herald_idx(config: Config) -> int:
    """Finds the herald index for the number of reads in a measurement.

    The herald index is taken to be the 0th read unless there is active reset.

    Args:
        config (Config): qcal Config object.

    Returns:
        int: read index for the herald measurement.
    """
    idx = 0
    if config.parameters['reset']['active']['enable']:
        idx += config.parameters['reset']['active']['n_resets']
    return idx


def post_process(config: Config, measurements: List, circuits: Any) -> None:
    """Post-process measurement results from QubiC.

    This post-processing routine assumes that all circuits have the same number
    of measurements. This function will write a results attribute to each
    circuit, which is a dictionary mapping all n-qubit bitstrings to their
    measured counts.

    Args:
        config (Config): qcal Config object.
        measurements (List): list of batched measurements.
        circuits (Any): any collection or set of circuits.
    """
    meas_qubits = set()
    for meas in measurements:
        meas_qubits |= set(meas['shots'].keys())

    meas = {
        q: np.vstack([
            meas['shots'][q] for meas in measurements
        ]) for q in meas_qubits
    }
    # meas[q].shape = (n circuits, n shots, n reads)

    all_results = []
    for i, circuit in enumerate(circuits):
        print
        # This assumes that the measurement results are at the very end
        results = pd.DataFrame(
            {q: meas[q][i][:,-1] for q in meas_qubits}
        )

        # Subsort shots by those which pass heralding
        if config.parameters['readout']['herald']:
            herald_idx = find_herald_idx(config)
            pass_herald = pd.DataFrame(      # Boolean series
                {q: meas[q][i][:,herald_idx] for q in meas_qubits}
            ).isin([0]).all(1)  # only columns with all True
            results = results[pass_herald]

        # Relable all 2s as 1s for excited state promotion
        if config.parameters['readout']['esp']['enable']:
            results = results.replace(2, 1)

        # Compute the counts for all unique combinations of 0s and 1s
        results = results.value_counts().rename('counts').reset_index()
        
        # Compute the bitstring for each set of counts
        results['bitstring'] = results[[q for q in meas_qubits]].apply(
            lambda row: ''.join(row.values.astype(str)), axis=1
        )

        # Write results to a dictionary mapping bitstrings to counts
        results = dict(zip(results.bitstring, results.counts))

        # Assign results as a circuit attribute
        try:
            circuit.results = results
        except Exception:
            logger.warning(f'Cannot write results to type {type(circuit)}!')

        all_results.append(results)

    if isinstance(circuits, CircuitSet):
        circuits['results'] = all_results