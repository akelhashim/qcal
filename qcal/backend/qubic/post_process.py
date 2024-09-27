"""Submodule for post-processing results measured on QubiC.

"""
from .utils import calculate_n_reads
from qcal.circuit import CircuitSet
from qcal.config import Config
from qcal.managers.classification_manager import ClassificationManager

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


def post_process(
        config:           Config, 
        measurements:     List,
        circuits:         Any,
        measure_qubits:   List[str] | None = None,
        n_reads_per_shot: int | dict | None = None,
        classifier:       ClassificationManager | None = None, 
        raster_circuits:  bool = False,
        rcorr_cmat:       pd.DataFrame | None = None,
        save_raw_data:    bool = False
    ) -> None:
    """Post-process measurement results from QubiC.

    This post-processing routine assumes that all circuits have the same number
    of measurements. This function will write a results attribute to each
    circuit, which is a dictionary mapping all n-qubit bitstrings to their
    measured counts.

    Args:
        config (Config): qcal Config object.
        measurements (List): list of batched measurements.
        circuits (Any): any collection or set of circuits.
        measure_qubits (List[str] | None, optional): list of qubit labels 
            for post-processing measurements. Defaults to None. If None, these 
            will be extracted from the measurements.
         n_reads_per_shot (int | dict | None, optional): number of reads per 
                shot per circuit. Defaults to None.
        classifier (ClassificationManager, None): manager used for classifying
            raw data. Defaults to None.
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.
        rcorr_cmat (pd.DataFrame | None, optional): confusion matrix for
            readout correction. Defaults to None. If passed, the readout
            correction will be applied to the raw bit strings in 
            post-processing.
        save_raw_data (bool, optional): whether to save raw IQ data for each
                qubit in the CircuitSet. Defaults to False.
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
            assert n_reads_per_shot is not None
            for q in meas_qubits:
                if isinstance(n_reads_per_shot, dict):
                    n_reads = int(
                        n_reads_per_shot[chanmap_r[q]] / len(circuits)
                    )
                elif isinstance(n_reads_per_shot, int):
                    n_reads = int(n_reads_per_shot / len(circuits))

                reorg_raw_iqs = []
                for i in range(raw_iq[q].shape[0]):
                    reorg_raw_iq = np.vstack([
                        [raw_iq[q][i, :, j:j+n_reads] for j in range(
                            0, raw_iq[q].shape[-1], n_reads
                        )]
                    ])
                    reorg_raw_iqs.append(reorg_raw_iq)
                raw_iq[q] = np.vstack([
                    iq for iq in reorg_raw_iqs
                ])

        if isinstance(circuits, CircuitSet) and save_raw_data:
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
            assert n_reads_per_shot is not None
            for q in meas_qubits:
                if isinstance(n_reads_per_shot, dict):
                    n_reads = int(
                        n_reads_per_shot[chanmap_r[q]] / len(circuits)
                    )
                elif isinstance(n_reads_per_shot, int):
                    n_reads = int(n_reads_per_shot / len(circuits))
                
                reorg_measurements = []
                for i in range(measurement[q].shape[0]):
                    reorg_measurement = np.vstack([
                        [measurement[q][i, :, j:j+n_reads] for j in range(
                            0, measurement[q].shape[-1], n_reads
                        )]
                    ])
                    reorg_measurements.append(reorg_measurement)
                measurement[q] = np.vstack([
                    meas for meas in reorg_measurements
                ])

    else:
        measurement = {}

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
                if rcorr_cmat is not None:
                    try:
                        circuit.results.apply_readout_correction(rcorr_cmat)
                    except Exception:
                        if i == 0:
                            logger.warning(
                                f'Cannot perform readout correction!'
                            )

            except Exception:
                if i == 0:
                    logger.warning(
                        f'Cannot write results to type {type(circuit)}!'
                    )

            all_results.append(results)

        if isinstance(circuits, CircuitSet):
            circuits['results'] = all_results