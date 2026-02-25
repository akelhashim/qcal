"""Submodule for post-processing measurement results.

"""
from __future__ import annotations  # noqa: I001

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
from numpy.typing import NDArray

from qcal.config import Config

from .passes import (
    compute_counts, discard_heralded_shots, relabel_esp
)
from .utils import construct_dataframes, find_herald_idx

logger = logging.getLogger(__name__)


__all__ = 'PostProcessor'


# Array2D has shape (n_shots, n_meas)
Array2D = NDArray[np.float64]

# Define a Pass type for post-processing methods
Pass = Callable[["PostProcessor"], None]


@dataclass
class PostProcessor:
    """Class for post-processing measurement results.

    Attributes:
        config:  qcal Config object.
        results: Dictionary mapping qubit labels to measurement results.
        passes:  List of post-processing passes.
    """
    config:  Config = field(default_factory=Config)
    results: List[dict[int, Array2D]] = field(default_factory=list)
    passes:  List[Pass] = field(
        default_factory=lambda: [
            discard_heralded_shots,
            relabel_esp,
            compute_counts
        ]
    )

    def __post_init__(self):
        self._circuit_results = None
        self._mcm_results = None # Mid-circuit measurement results
        self._tm_results = None  # Terminating measurement results
        self._qubits = sorted(set(self.results[0].keys()))
        self._herald_idx = find_herald_idx(self.config)

    @property
    def circuit_results(self) -> List[Any]:
        """Get the results for all circuits.

        Returns:
            List[Any]: results for all circuits.
        """
        return self._circuit_results

    @property
    def herald_idx(self) -> int:
        """Get the index of the herald measurement.

        Returns:
            int: herald index.
        """
        return self._herald_idx

    @property
    def mcm_results(self) -> List[List[Dict]]:
        """Get the mid-circuit measurement results.

        Returns:
            List[List[Dict]]: mid-circuit measurement results for all circuits.
                Each inner list contains the results for a single circuit with
                arbitrary number of mid-circuit measurements.
        """
        return self._mcm_results

    @property
    def tm_results(self) -> List[Dict]:
        """Get the terminating measurement results.

        Returns:
            List[Dict]: terminating measurement results.
        """
        return self._tm_results

    @property
    def qubits(self) -> List:
        """Get the labels of the measured qubits.

        Returns:
            List: qubit labels.
        """
        return self._qubits

    def _construct_dataframes(self) -> None:
        """Reshape the measurement results into a list of lists of DataFrames.
        """
        if self.config['readout/herald']:
            start_idx = self._herald_idx + 1
        else:
            start_idx = 0

        # This assumes that all qubits have the same number of measurements
        self._circuit_results = construct_dataframes(self.results, start_idx)

    def _apply_passes(self) -> None:
        """Apply the post-processing passes to the measurement results."""
        for pass_ in self.passes:
            pass_(self)

    def run(self) -> None:
        """Run the post-processing routine."""
        self._construct_dataframes()
        self._apply_passes()
        self._mcm_results = [result[:-1] for result in self._circuit_results]
        self._tm_results = [result[-1] for result in self._circuit_results]
