"""Submodule for handling pyGSTi reports.

"""
import logging

# import pandas as pd
from pygsti.data import DataSet
from pygsti.io import write_dataset

from qcal.circuit import CircuitSet
from qcal.results import Results

logger = logging.getLogger(__name__)


__all__ = 'generate_pygsti_report'


def generate_pygsti_dataset(
    circuits: CircuitSet, save_path: str | None = None
) -> DataSet:
    """Generate a pyGSTi dataset from circuit results.

    Args:
        circuits (CircuitSet): measured circuits
        save_path (str): location where to save the results. Defaults to None.

    Returns:
        DataSet: pyGSTi dataset.
    """
    if 'results' not in circuits._df.columns:
        raise ValueError("CircuitSet must have a 'results' column!")

    # Old way
    # results_df = pd.DataFrame(
    #     [circuits['results'].loc[i] for i in range(len(circuits))]
    # )
    # results_df.index = circuits['pygsti_circuit'].values
    # results_df = results_df.fillna(0).astype(int).rename(
    #     columns=lambda col: col + ' count'
    # )

    # logger.info(" Saving the pyGSTi results...")
    # with open(f'{save_path}dataset.txt', 'w') as f:  # Write the header
    #     f.write('## Columns = ' + ', '.join(results_df.columns) + "\n")
    #     f.close()
    # results_df.to_csv(
    #     f'{save_path}dataset.txt', sep=' ', mode='a', header=False
    # )

    # New way
    states = set()
    for result in circuits['results']:
        res = Results(result)
        states.update(res.states)

    dataset = DataSet(outcome_labels=list(states))
    for i, circuit in enumerate(circuits):
        dataset[circuit] = circuits['results'].iloc[i]

    if save_path:
        write_dataset(f'{save_path}dataset.txt', dataset)

    return dataset
