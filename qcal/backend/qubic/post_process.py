"""Submodule for post-processing results measured on QubiC.

"""
from qcal.circuit import CircuitSet
from qcal.config import Config
from qcal.managers.classification_manager import ClassificationManager

import logging
import numpy as np
import pandas as pd

from typing import Any, List

logger = logging.getLogger(__name__)


__all__ = ('post_process')


def calculate_n_reads(config: Config) -> int:
    """Calculate the number of reads per circuit from a config.

    The number of reads will depend on:
    1) number of active resets
    2) heralding
    3) readout at the end of the circuit

    This function assumes that there is no mid-circuit measurement, and that
    there is always a readout at the end of a circuit.

    Args:
        config (Config): config object.

    Returns:
        int: number of reads per circuit.
    """
    n_reads = 1  # Measurement at the end of the circuit
    
    if config.parameters['reset']['active']['enable']:
        n_reads += config.parameters['reset']['active']['n_resets']
    
    if config.parameters['readout']['herald']:
        n_reads += 1

    return n_reads


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


def post_process(
        config:          Config, 
        measurements:    List,
        measure_qubits:  List[str] | None,
        classifier:      ClassificationManager | None, 
        circuits:        Any,
        raster_circuits: bool = False
    ) -> None:
    """Post-process measurement results from QubiC.

    This post-processing routine assumes that all circuits have the same number
    of measurements. This function will write a results attribute to each
    circuit, which is a dictionary mapping all n-qubit bitstrings to their
    measured counts.

    Args:
        config (Config): qcal Config object.
        measurements (List): list of batched measurements.
        measure_qubits (List[str] | None, optional): list of qubit labels 
            for post-processing measurements. If None, these will be extracted
            from the measurements.
        classifier (ClassificationManager, None): manager used for classifying
            raw data.
        circuits (Any): any collection or set of circuits.
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.
    """
    outputs = list(measurements[0].keys())

    chanmap = {}
    chanmap_r = {}
    for q, ch in config.readout.loc['channel'].items():
        chanmap[str(int(ch))] = f'Q{q}'
        chanmap_r[f'Q{q}'] = str(int(ch))

    meas_qubits = set()
    if measure_qubits is not None:
        for q in measure_qubits:
            meas_qubits.add(q)
    else:
        for meas in measurements:
            if 'shots' in outputs:
                meas_qubits |= set(meas['shots'].keys())
            elif 's11' in outputs:
                for ch in meas['s11'].keys():
                    meas_qubits.add(chanmap[ch])
    meas_qubits = sorted(meas_qubits)

    if 's11' in outputs:  # Might not work with rastering
        # {'Q0': np.array([[...],...,[...]])} of shape 
        # (n circuits, n shots, n reads)}
        raw_iq = { 
            q: np.vstack([
                meas['s11'][chanmap_r[q]] for meas in measurements
            ]) for q in meas_qubits
        }

        if raster_circuits:
            n_reads = calculate_n_reads(config)
            for q in meas_qubits:
                reorg_raw_iqs = []
                for i in range(raw_iq[q].shape[0]):
                    reorg_raw_iq = np.vstack([
                        raw_iq[q][i, :, j] for j in range(
                            0, raw_iq[q].shape[-1], n_reads
                        )
                    ])
                    reorg_raw_iqs.append(reorg_raw_iq)
                raw_iq[q] = np.vstack([
                    iq for iq in reorg_raw_iqs
                ])
        # print(raw_iq['Q7'].shape)
        # print(raw_iq['Q7'])

        if isinstance(circuits, CircuitSet):
            for q, meas in raw_iq.items():
                circuits[f'{q}: iq_data'] = [
                    m[:, -1].reshape(-1, 1) for m in meas
                ]

    # meas[q]['s11'].shape = (n circuits, n shots, n reads)
    if ('s11' in outputs) and (classifier is not None):
        measurement = {}
        for q, meas in raw_iq.items():
            if classifier[int(q[1:])].is_fitted:

                circ_results = []
                for circ in meas:

                    circ_reads = []
                    for r in range(circ.shape[-1]):  # n reads
                        X = np.hstack([
                            np.real(circ[:, r]).reshape(-1, 1), 
                            np.imag(circ[:, r]).reshape(-1, 1)
                        ])
                        y = classifier[int(q[1:])].predict(X).reshape(-1, 1)
                        circ_reads.append(y)
                    circ_results.append(np.hstack(circ_reads))
                
                measurement[q] = np.array(circ_results)
    
    elif 'shots' in outputs:
        measurement = {
            q: np.vstack([
                meas['shots'][q] for meas in measurements
            ]) for q in meas_qubits
        }

        if raster_circuits:
            n_reads = calculate_n_reads(config)
            for q in meas_qubits:
                reorg_measurements = []
                for i in range(measurement[q].shape[0]):
                    reorg_measurement = np.vstack([
                        measurement[q][i, :, j] for j in range(
                            0, measurement[q].shape[-1], n_reads
                        )
                    ])
                    reorg_measurements.append(reorg_measurement)
                measurement[q] = np.vstack([
                    meas for meas in reorg_measurements
                ])

    if measurement:
        all_results = []
        for i, circuit in enumerate(circuits):
            # This assumes that the measurement results are at the very end
            results = pd.DataFrame(
                {q: measurement[q][i][:,-1] for q in meas_qubits}
            )

            # Subsort shots by those which pass heralding
            if config.parameters['readout']['herald']:
                herald_idx = find_herald_idx(config)
                pass_herald = pd.DataFrame(      # Boolean series
                    {q: measurement[q][i][:,herald_idx] for q in meas_qubits}
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
                if i == 0:
                    logger.warning(
                        f'Cannot write results to type {type(circuit)}!'
                    )

            all_results.append(results)

        if isinstance(circuits, CircuitSet):
            circuits['results'] = all_results