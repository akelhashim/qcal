"""Submodule for handling pyGSTi reports.

"""
from qcal.circuit import CircuitSet

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def generate_pygsti_report(circuits: CircuitSet, fileloc: str):
    """Generate a pyGSTi report from circuit results.

    Args:
        circuits (CircuitSet): measured circuits
        fileloc (str): location where to save the results.
    """
    results_dfs = []
    for i, circuit in enumerate(circuits):
        results_dfs.append(
            pd.DataFrame(
                [circuit.results.dict], 
                index=[circuits['pygsti_circuit'][i]]
            )
        )
    results_df = pd.concat(results_dfs)
    results_df = results_df.fillna(0).astype(int).rename(
        columns=lambda col: col + ' count'
    )

    logger.info(" Saving the pyGSTi results...")
    with open(f'{fileloc}_dataset.txt', 'w') as f:  # Write the header
        f.write('## Columns = ' + ', '.join(results_df.columns) + "\n")
        f.close()
    results_df.to_csv(
        f'{fileloc}_dataset.txt', sep=' ', mode='a', header=False
    )