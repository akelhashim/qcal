"""Helper functions for benchmarking protocols.

"""
import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_error_rates(
    error_rates: Dict,
    uncertainties: Dict,
    ylabel: str = 'Error Rate',
    save_path: str | None = None
) -> None:
    """Plot error rates for randomized benchmarks.

    Args:
        error_rates (Dict): dictionary mapping qubit label to error rate.
        uncertainties (Dict): dictionary mapping qubit label to uncertainty.
        ylabel (str, optional): y-axis label. Defaults to 'Error Rate'.
        save_path (str | None, optional): save path for figure. Defaults to
            None.
    """
    qlabels = sorted(error_rates.keys())
    error_rates = [error_rates[ql] for ql in qlabels]
    uncertainties = [uncertainties[ql] for ql in qlabels]

    fig = plt.figure(figsize=(min(3*len(qlabels), 10), 4))
    ms = 7

    plt.errorbar(
        np.arange(len(qlabels)),
        error_rates,
        yerr=uncertainties,
        fmt='o', ms=ms, color='blue'
    )

    plt.xlabel('Qubit Label', fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(np.arange(len(qlabels)), qlabels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.yscale('log')

    fig.set_tight_layout(True)
    if save_path:
        fig.savefig(
            save_path + 'error_rates.png',
            dpi=600,
            bbox_inches='tight',
            # pad_inches=0
        )
        fig.savefig(
            save_path + 'error_rates.pdf',
            bbox_inches='tight',
            # pad_inches=0
        )
        fig.savefig(
            save_path + 'error_rates.svg',
            bbox_inches='tight',
            # pad_inches=0
        )
    plt.plot()
