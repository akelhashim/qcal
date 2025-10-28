"""Submodule for analyzing leakage.

"""
import qcal.settings as settings

from qcal.fitting.fit import FitExponential
from qcal.math.utils import round_to_order_error
from qcal.results import Results
from qcal.utils import save_to_pickle

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmfit import Parameters
from typing import Any, Dict

logger = logging.getLogger(__name__)


def analyze_leakage(circuits: Any, filename: str | None = None) -> Dict:
    """Analyze leakage results for a set of circuits.

    Args:
        circuits (Any): set of circuits.
        filename (str | None, optional): filename where to save the results. 
            Defaults to None.

    Returns:
        Dict: dictionary of leakage counts per circuit depth for each qubit.
    """
    qubits = circuits[0].labels
    q_index = [qubits.index(q) for q in qubits]

    circuit_depths = set()
    label = {q: f'Q{q}' for q in qubits}
    leakage = {q: {} for q in qubits}
    fit = {q: FitExponential() for q in qubits}

    # Gather the leakage counts
    for circuit in circuits:
        circuit_depth = len(circuit)
        circuit_depths.add(circuit_depth)
        
        if not isinstance(circuit.results, Results):
            results = Results(circuit.results)
        else:
            results = circuit.results

        for q, i in zip(qubits, q_index):
            if circuit_depth in leakage[q].keys():
                leakage[q][circuit_depth].append(
                    results.marginalize(i).populations['2']
                )
            else:
                leakage[q][circuit_depth] = [
                    results.marginalize(i).populations['2']
                ]

    circuit_depths = np.array(sorted(circuit_depths))

    # Fit the data
    for q in qubits:
        y = np.array([np.mean(leakage[q][d]) for d in circuit_depths])
        a = y.min() - y.max()
        params = Parameters()
        params.add('a', value=a, min=-10, max=0)
        params.add('b', value=1/np.mean(circuit_depths), min=0.0, max=1.0)
        params.add('c', value=y.min() - a, min=0, max=1.0)
        fit[q].fit(circuit_depths, y, params=params)

        if fit[q].fit_success:
            r = abs(fit[q].fit_params['b'].value)
            err = fit[q].fit_params['b'].stderr
            r, err = round_to_order_error(r, err, 2)
            label[q] += f": r = {r:.3f} ({err:.3f})"
            print(label[q])

    # Plot the data
    fig = plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(qubits)))

    for i, q in enumerate(qubits):
        plt.plot(
            leakage[q].keys(), leakage[q].values(), 
            'o', c=colors[i], ms=3.0, alpha=0.25
        )
        plt.plot(
            leakage[q].keys(), np.array(list(leakage[q].values())).mean(axis=1), 
            'o', ms=10.0, c=colors[i], label=label[q]
        )
        if fit[q].fit_success:
            x = np.linspace(min(circuit_depths), max(circuit_depths), 100)
            plt.plot(x, fit[q]._result.eval(x=x), c=colors[i], ls='--')

    plt.xlabel('Circuit Depth', fontsize=15)
    plt.ylabel(r'$|2\rangle$ State Population', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')

    if settings.Settings.save_data and filename is not None:         
        fig.savefig(filename + 'leakage.png', dpi=300)
        fig.savefig(filename + 'leakage.pdf')
        fig.savefig(filename + 'leakage.svg')
        save_to_pickle(
            pd.DataFrame(leakage), filename=filename + 'leakage'
        )

    plt.plot()

    return leakage
