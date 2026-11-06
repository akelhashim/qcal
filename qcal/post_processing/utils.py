"""Helper functions for post-processing measurement results.

"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qcal.config import Config

if TYPE_CHECKING:
    from qcal.circuit import Circuit

logger = logging.getLogger(__name__)


# One {qubit: read index} per measurement cycle of a circuit, in circuit order
MeasurementCycles = List[Dict[int, int]]


def circuit_measurement_cycles(circuit: Circuit) -> MeasurementCycles:
    """Find which read of each qubit is made in each cycle of a circuit.

    Every cycle containing at least one measurement (``Meas`` or ``MCM``)
    is a measurement cycle. Read indices count each qubit's reads from the
    start of the circuit, so they exclude any reads prepended to it (e.g.
    heralding or active reset).

    For example, if ancilla qubit 1 is measured mid-circuit twice and data
    qubit 0 is only measured at the end, the measurement cycles are
    ``[{1: 0}, {1: 1}, {0: 0}]``.

    Args:
        circuit (Circuit): qcal circuit.

    Returns:
        MeasurementCycles: one mapping per measurement cycle, from each
            qubit measured in that cycle to the index of that read among the
            qubit's reads.
    """
    n_reads = defaultdict(int)
    meas_cycles = []
    for cycle in circuit.cycles:
        if cycle.is_barrier:
            continue
        reads = {}
        for gate in cycle:
            if gate.is_measurement:
                for q in gate.qubits:
                    reads[q] = n_reads[q]
                    n_reads[q] += 1
        if reads:
            meas_cycles.append(reads)
    return meas_cycles


def construct_dataframes(
    results:            List[dict[int, NDArray[np.float64]]],
    start_idx:          int,
    measurement_cycles: List[MeasurementCycles] | None = None,
) -> List[List[pd.DataFrame]]:
    """Construct all circuit results as a list of a list of DataFrames.

    This function takes the existing results (which is a list of a
    dictionary mapping qubit labels to an array of measurement results) and
    reshapes it to be a list of a list of DataFrames. The outer list is over
    the circuits, and the inner list is over the measurement cycles of each
    circuit, so that each DataFrame contains the results of the qubits
    measured in a single cycle.

    Without ``measurement_cycles``, the n-th read of every qubit is taken to be
    in the n-th cycle, which requires every qubit in a circuit to have the same
    number of reads. With ``measurement_cycles`` (see
    ``circuit_measurement_cycles``), qubits can be measured a different number
    of times (e.g. ancillas measured mid-circuit but not at the end), and each
    DataFrame only contains the qubits measured in that cycle. Cycles measuring
    none of the qubits in ``results`` are dropped.

    Args:
        results (List[dict[int, Array2D]]): all circuit results. Each array
            has shape ``(n_shots, n_reads)``.
        start_idx (int): number of reads preceding the circuit (e.g.
            heralding), which are skipped.
        measurement_cycles (List[MeasurementCycles] | None, optional):
            measurement cycles of each circuit. Defaults to None.

    Returns:
        List[List[pd.DataFrame]]: reshaped all circuit results.

    Raises:
        ValueError: if qubits have a different number of reads and no
            ``measurement_cycles`` are given, or if a qubit's number of reads
            doesn't match its ``measurement_cycles``.
    """
    if measurement_cycles is None:
        circuit_results = []
        for i, result in enumerate(results):
            n_reads = {q: reads.shape[1] for q, reads in result.items()}
            if len(set(n_reads.values())) > 1:
                raise ValueError(
                    f"Qubits in circuit {i} have a different number of "
                    f"reads {n_reads}; pass `measurement_cycles` to align them."
                )
            n = next(iter(n_reads.values()), start_idx)
            circuit_results.append([
                pd.DataFrame({q: reads[:, j] for q, reads in result.items()})
                for j in range(start_idx, n)
            ])
        return circuit_results

    if len(measurement_cycles) != len(results):
        raise ValueError(
            f"Got measurement cycles for {len(measurement_cycles)} circuits, "
            f"but results for {len(results)} circuits."
        )

    circuit_results = []
    for i, (result, cycles) in enumerate(
        zip(results, measurement_cycles, strict=True)
    ):
        n_cycle_reads = defaultdict(int)
        for cycle in cycles:
            for q in cycle:
                n_cycle_reads[q] += 1
        for q, reads in result.items():
            if reads.shape[1] != start_idx + n_cycle_reads[q]:
                raise ValueError(
                    f"Qubit {q} in circuit {i} has {reads.shape[1]} reads, "
                    f"but {start_idx} preceding reads and "
                    f"{n_cycle_reads[q]} measurement cycles."
                )

        dfs = []
        for cycle in cycles:
            measured = {
                q: result[q][:, start_idx + j]
                for q, j in cycle.items() if q in result
            }
            if measured:
                dfs.append(pd.DataFrame(measured))
        circuit_results.append(dfs)
    return circuit_results


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
        {
            k: v for k, v in zip(results.keys(), values, strict=True)
            if not np.all(v == None)  # noqa: E711, NOTE: don't remove
        }
        for values in zip(*results.values(), strict=True)
    ]
    return result
