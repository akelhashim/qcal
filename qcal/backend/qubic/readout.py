"""Submodule for performing readout ML using QubiC.

"""
from qcal.circuit import CircuitSet
from qcal.managers.classification_manager import ClassificationManager
from qcal.utils import save_to_pickle

import logging
import numpy as np

from typing import Dict

logger = logging.getLogger(__name__)


def train_readout_ml(
        circuits: CircuitSet,
        classifier: ClassificationManager,
        savedir: str | None = None
    ) -> Dict:
    """Train the readout ML model.

    Args:
        circuits (CircuitSet): readout calibration circuits.
        classifier (ClassificationManager): classifier for terminating 
            measurements.
        savedir (str | None, optional): save directory. Defaults to None.

    Returns:
        Dict: dictionary mapping qubit label to trained parameters.
    """
    try:
       from qubicml.pipeline.train_pipeline import Trainer
    except ImportError:
        logger.warning(' Unable to import qubicml!')

    qubits = circuits[0].qubits
    n_levels = len(circuits)

    data = {}
    for q in qubits:
        state = np.concatenate([
            classifier.predict(q,
                    np.hstack((
                        np.real(circuits[f'Q{q}: iq_data'][n]), 
                        np.imag(circuits[f'Q{q}: iq_data'][n])
                    )) 
                )
            for n in range(n_levels)
        ])
        
        data[f'Q{q}'] = {
            'data': np.concatenate([
                circuits[f'Q{q}: iq_data'][n].reshape(-1) for n in 
                range(n_levels)
            ]),
            'state': state
        }

    sd_param = {}
    for qid in data:
        qcml = Trainer(
            qid=qid, data=data[qid]['data'], target=data[qid]['state']
        )
        sd_param[qid] = qcml.start()

    if savedir:
        save_to_pickle(sd_param, f'{savedir}sd_param.pkl')

    return sd_param
