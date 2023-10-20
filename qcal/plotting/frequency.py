"""Submodule for frequency plots.

"""
from qcal.config import Config

import logging
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from typing import List, Tuple
# from plotly.colors import n_colors

logger = logging.getLogger(__name__)


def plot_freq_spectrum(
        config:         Config, 
        qubits:         List | Tuple | None = None,
        qubit_pairs:    List | Tuple | None = None, 
        plot_GE:        bool = True,
        plot_EF:        bool = False,
        plot_readout:   bool = False,
        plot_two_qubit: bool = False,
    ) -> None:
    """Plot the frequency spectrum for the frequencies in the config.

    Args:
        config (Config): qcal config object.
        qubits (List | Tuple | None, optional): qubits to plot. Defaults to 
            None.
        qubit_pairs (List | Tuple | None, optional): qubit pairs to plot. 
            Defaults to None.
        plot_GE (bool, optional): plot GE frequencies. Defaults to True.
        plot_EF (bool, optional): plot EF frequencies. Defaults to False.
        plot_readout (bool, optional): plot readout frequencies. Defaults to 
            False.
        plot_two_qubit (bool, optional): plot two-qubit gate frequencies. 
            Defaults to False.
    """

    # TODO: add support for qubit_LO and readout_LO
    fig = go.Figure(
        layout=go.Layout(
        title={
            'text': 'Frequency Spectrum',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        titlefont_size=16,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        yaxis=dict(showticklabels=False))
    )

    qubits = config.qubits if qubits == None else qubits
    qubit_pairs = config.qubit_pairs if qubit_pairs == None else qubit_pairs

    # https://plotly.com/python/builtin-colorscales/
    colors = px.colors.sample_colorscale(
        'Bluered' if len(qubits) <= 2 else 'jet', 
        [n/(len(qubits) - 1) for n in range(len(qubits))]
    )

    n_points = 0
    for qp in config.qubit_pairs:
        n_points += len(config.basis_gates['two_qubit'][qp])
    colors_tq = px.colors.sample_colorscale(
        'RdBu',
        [n/(n_points - 1) for n in range(n_points)]
    )

    x_mins = []
    x_maxs = []
    if plot_GE:
        for i, q in enumerate(qubits):
            fig.add_trace(
                go.Scatter(
                    x=[config[f'single_qubit/{q}/GE/freq'] / 1e9,
                       config[f'single_qubit/{q}/GE/freq'] / 1e9],
                    y=[1, 0],
                    line_width=5,
                    mode='lines',
                    name=f'Q{q} GE',
                    line=dict(color=colors[i])
                )
            )
        x_mins.append(
            min([config[f'single_qubit/{q}/GE/freq'] / 1e9 for q in qubits])
        )
        x_maxs.append(
            max([config[f'single_qubit/{q}/GE/freq'] / 1e9 for q in qubits])
        )

    if plot_EF:
        for i, q in enumerate(qubits):
            fig.add_trace(
                go.Scatter(
                    x=[config[f'single_qubit/{q}/EF/freq'] / 1e9,
                       config[f'single_qubit/{q}/EF/freq'] / 1e9],
                    y=[1, 0],
                    line_width=5,
                    mode='lines',
                    name=f'Q{q} EF',
                    line=dict(color=colors[i], dash='dash')
                )
            )
        x_mins.append(
            min([config[f'single_qubit/{q}/EF/freq'] / 1e9 for q in qubits])
        )
        x_maxs.append(
            max([config[f'single_qubit/{q}/EF/freq'] / 1e9 for q in qubits])
        )

    if plot_readout:
        for i, q in enumerate(qubits):
            fig.add_trace(
                go.Scatter(
                    x=[config[f'readout/{q}/freq'] / 1e9,
                       config[f'readout/{q}/freq'] / 1e9],
                    y=[1, 0],
                    line_width=5,
                    mode='lines',
                    name=f'R{q}',
                    line=dict(color=colors[i], dash='dot')
                )
            )

        x_mins.append(
            min([config[f'readout/{q}/freq'] / 1e9 for q in qubits])
        )
        x_maxs.append(
            max([config[f'readout/{q}/freq'] / 1e9 for q in qubits])
        )

    if plot_two_qubit:
        i = -1
        for qp in qubit_pairs:
            for gate in config.basis_gates['two_qubit'][qp]:
                i += 1
                fig.add_trace(
                    go.Scatter(
                        x=[config[f'two_qubit/{qp}/{gate}/freq'] / 1e9,
                           config[f'two_qubit/{qp}/{gate}/freq'] / 1e9],
                        y=[1, 0],
                        line_width=5,
                        mode='lines',
                        name=f'{gate} {qp}',
                        line=dict(color=colors_tq[i])
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=np.linspace(min(x_mins) - 0.05, max(x_maxs) + 0.05, 1000),
            y=np.zeros(1000),
            line_width=0,
            mode='lines',
            name=''
        )
    )

    fig.update_layout(
        hovermode='x unified',
        xaxis_title='Frequency (GHz)',
        xaxis_range=[min(x_mins) - 0.05, max(x_maxs) + 0.05],
        yaxis_range=[0, 1],
        yaxis_hoverformat='',
    )

    fig.show()
