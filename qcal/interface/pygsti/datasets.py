"""Submodule for handling pyGSTi reports.

"""
from qcal.circuit import CircuitSet

import pandas as pd
import logging

logger = logging.getLogger(__name__)


__all__ = 'generate_pygsti_report'


def generate_pygsti_dataset(circuits: CircuitSet, save_path: str):
    """Generate a pyGSTi dataset from circuit results.

    Args:
        circuits (CircuitSet): measured circuits
        save_path (str): location where to save the results.
    """
    results_df = pd.DataFrame(
        [circuits['results'].loc[i] for i in range(len(circuits))]
    )
    results_df.index = circuits['pygsti_circuit'].values
    results_df = results_df.fillna(0).astype(int).rename(
        columns=lambda col: col + ' count'
    )

    logger.info(" Saving the pyGSTi results...")
    with open(f'{save_path}dataset.txt', 'w') as f:  # Write the header
        f.write('## Columns = ' + ', '.join(results_df.columns) + "\n")
        f.close()
    results_df.to_csv(
        f'{save_path}dataset.txt', sep=' ', mode='a', header=False
    )