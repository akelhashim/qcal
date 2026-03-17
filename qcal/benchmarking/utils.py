"""Helper functions for benchmarking protocols.

"""
import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

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

    ms = 7
    x = np.arange(len(qlabels))

    if save_path:
        # Matplotlib figure (for saving)
        fig = plt.figure(figsize=(min(3*len(qlabels), 10), 4))
        plt.errorbar(
            x,
            error_rates,
            yerr=uncertainties,
            fmt='o', ms=ms, color='blue'
        )
        plt.xlabel('Qubit Label', fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.xticks(x, qlabels, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.yscale('log')
        fig.set_tight_layout(True)
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
        plt.close(fig)

    # Plotly figure (for displaying)
    pfig = go.Figure(
        data=[
            go.Scatter(
                x=[str(ql) for ql in qlabels],
                y=error_rates,
                mode='markers',
                marker={'color': '#1f77b4', 'size': 10},
                error_y={
                    'type': 'data',
                    'array': uncertainties,
                    'visible': True,
                    'thickness': 1,
                    'width': 6,
                },
                showlegend=False,
            )
        ]
    )
    pfig.update_layout(
        height=450,
        width=min(200 * len(qlabels), 1000),
        margin={'t': 40, 'r': 20, 'b': 60, 'l': 80},
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='#fbfbfd',
    )
    pfig.update_xaxes(
        title_text='Qubit Label',
        type='category',
        tickmode='array',
        tickvals=[str(ql) for ql in qlabels],
        ticktext=[str(ql) for ql in qlabels],
        title_standoff=10,
        automargin=True,
        showgrid=True,
    )
    pfig.update_yaxes(
        title_text=ylabel,
        type='log',
        title_standoff=10,
        automargin=True,
        showgrid=True,
    )
    pfig.update_xaxes(
        showline=True,
        mirror=True,
        linecolor='#c7c7c7',
        linewidth=1,
        gridcolor='#e5e7eb',
        zeroline=False,
        ticks='outside',
    )
    pfig.update_yaxes(
        showline=True,
        mirror=True,
        linecolor='#c7c7c7',
        linewidth=1,
        gridcolor='#e5e7eb',
        zeroline=False,
        ticks='outside',
    )
    save_properties = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'error_rates',
            'scale': 10,
        }
    }
    pfig.show(config=save_properties)
