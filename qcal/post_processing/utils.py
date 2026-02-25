"""Helper functions for post-processing measurement results.

"""
import logging
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qcal.config import Config

logger = logging.getLogger(__name__)


def construct_dataframes(
    results: List[dict[int, NDArray[np.float64]]],
    start_idx: int
) -> List[List[pd.DataFrame]]:
    """Construct all circuit results as a list of a list of DataFrames.

    This method takes the existing results (which is a list of a
    dictionary mapping qubit labels to an array of measurement results) and
    reshapes it to be a list of a list of DataFrames, where each DataFrame
    contains the results on all qubits for a single circuit. The outer list is
    over the number of circuits, and the inner list is over the number of
    measurements per shot for each circuit.

    Args:
        results (List[dict[int, Array2D]]): all circuit results.

    Returns:
        List[List[pd.DataFrame]]: reshaped all circuit results.
    """
    return [[
        pd.DataFrame({q: results[i][q][:, j] for q in results[i].keys()})
        for j in range(
            start_idx, results[i][list(results[i].keys())[0]].shape[1]
        )
    ] for i in range(len(results))]


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


def reshape_results(
    results: dict[int, List[NDArray[np.float64]]]
) -> list[dict[int, NDArray[np.float64]]]:
    """Reshape measurement results.

    This function takes the existing results (which is a dictionary mapping
    qubit labels to a list of measurement results) and reshapes it to be a list
    of dictionaries, where each dictionary maps qubit labels to a the circuit
    results.

    Args:
        results (dict[int, List[Array2D]]): measurement results.

    Returns:
        list[dict[int, Array2D]]: reshaped measurement results.
    """
    result = [
        dict(zip(results.keys(), values, strict=True))
        for values in zip(*results.values(), strict=True)
    ]
    return result
