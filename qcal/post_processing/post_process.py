"""Submodule for post-processing measurement results.

"""
from __future__ import annotations  # noqa: I001

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
from numpy.typing import NDArray

from qcal.config import Config
from qcal.results import PerShotResults

from .passes import (
    compute_conditional_counts,
    compute_counts,
    discard_heralded_shots,
    relabel_esp,
)
from .utils import MeasurementCycles, construct_dataframes, find_herald_idx

logger = logging.getLogger(__name__)


__all__ = 'PostProcessor'


# Array2D has shape (n_shots, n_meas)
Array2D = NDArray[np.float64]

# Define a Pass type for post-processing methods
Pass = Callable[["PostProcessor"], None]

# Passes that aggregate per-shot outcomes into counts
_AGGREGATION_PASSES = (compute_counts, compute_conditional_counts)


@dataclass
class PostProcessor:
    """Class for post-processing measurement results.

    Post-processing runs in two stages. First, ``passes`` operate on the
    per-shot outcomes of each measurement layer (e.g. discarding heralded
    shots, or decoding). Then ``aggregate`` turns them into counts.

    Attributes:
        config (Config): qcal Config object.
        results (List[dict[int, Array2D]]): list of dictionaries mapping
            qubit labels to measurement results, one per circuit. Each array
            has shape ``(n_shots, n_reads)``.
        passes (List[Pass]): list of per-shot post-processing passes.
        aggregate (Pass): pass aggregating per-shot outcomes into counts.
            Defaults to ``compute_counts``.
        measurement_cycles (List[MeasurementCycles] | None): measurement
            cycles of each circuit (see ``circuit_measurement_cycles``).
            Required if qubits in a circuit are measured a different number
            of times. Defaults to None.
        keep_per_shot_results (bool): whether to keep the per-shot outcomes
            of every measurement cycle (e.g. for decoding) in
            ``per_shot_results``. These can be large, so they are only built
            on request. Defaults to False.
    """
    config:  Config = field(default_factory=Config)
    results: List[dict[int, Array2D]] = field(default_factory=list)
    passes:  List[Pass] = field(
        default_factory=lambda: [
            discard_heralded_shots,
            relabel_esp,
        ]
    )
    aggregate: Pass = compute_counts
    measurement_cycles: List[MeasurementCycles] | None = None
    keep_per_shot_results: bool = False

    def __post_init__(self):
        # Backwards compatibility: passes used to end with the aggregation
        self.passes = list(self.passes)
        if self.passes and self.passes[-1] in _AGGREGATION_PASSES:
            self.aggregate = self.passes.pop()
        if any(pass_ in _AGGREGATION_PASSES for pass_ in self.passes):
            raise ValueError(
                "Aggregation passes must be given as `aggregate`, not in "
                "`passes`."
            )

        self._circuit_results = None
        self._per_shot_results = None
        self._mcm_results = None # Mid-circuit measurement results
        self._tm_results = None  # Terminating measurement results
        self._qubits = tuple(
            sorted({q for r in self.results for q in r.keys()})
        )
        self._herald_idx = find_herald_idx(self.config)

    @property
    def circuit_results(self) -> List[Any]:
        """Get the results for all circuits.

        Returns:
            List[Any]: results for all circuits.
        """
        return self._circuit_results

    @property
    def per_shot_results(self) -> List[PerShotResults] | None:
        """Get the per-shot outcomes for all circuits.

        These are the outcomes after ``passes`` have been applied (e.g.
        with heralded shots discarded), before they are aggregated into
        counts. Shots are identified by their original shot number, even
        after others have been discarded.

        Returns:
            List[PerShotResults] | None: per-shot outcomes of each circuit,
                or None if they weren't kept.
        """
        return self._per_shot_results

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

        self._circuit_results = construct_dataframes(
            self.results, start_idx, self.measurement_cycles
        )

    def _apply_passes(self) -> None:
        """Apply the post-processing passes to the measurement results."""
        for pass_ in self.passes:
            pass_(self)

    def run(self) -> None:
        """Run the post-processing routine."""
        self._construct_dataframes()
        self._apply_passes()
        if self.keep_per_shot_results:
            self._per_shot_results = [
                PerShotResults.from_cycles(
                    cycles,
                    n_shots=next(
                        (reads.shape[0] for reads in result.values()), None
                    ),
                )
                for cycles, result in zip(
                    self._circuit_results, self.results, strict=True
                )
            ]
        self.aggregate(self)
        self._mcm_results = [result[:-1] for result in self._circuit_results]
        self._tm_results = [result[-1] for result in self._circuit_results]
