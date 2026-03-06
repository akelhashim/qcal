"""Submodule for frequency plots.

"""
import logging
from typing import List, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# from plotly.colors import n_colors
import plotly.io as pio

from qcal.config import Config
from qcal.units import GHz

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
    pio.templates.default = 'plotly'

    # TODO: add support for qubit_LO and readout_LO
    fig = go.Figure(
        layout=go.Layout(
        # title={
        #     'text': 'Frequency Spectrum',
        #     'x': 0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top'
        # },
        title={'font': {'size': 25}},
        showlegend=True,
        hovermode='closest',
        margin={'b': 20,'l': 5,'r': 5,'t': 40},
        yaxis={'showticklabels': False})
    )

    qubits = config.qubits if qubits is None else qubits
    qubit_pairs = config.qubit_pairs if qubit_pairs is None else qubit_pairs

    # https://plotly.com/python/builtin-colorscales/
    colors = px.colors.sample_colorscale(
        'Bluered' if len(qubits) <= 2 else 'jet',
        [n/(len(qubits) - 1) for n in range(len(qubits))]
    )

    n_points = 0
    for qp in config.qubit_pairs:
        n_points += len(config.native_gates['two_qubit'][qp])
    colors_tq = px.colors.sample_colorscale(
        'RdBu',
        [n/(n_points - 1) for n in range(n_points)]
    )

    x_mins = []
    x_maxs = []
    if plot_GE:
        x_mins_ge = []
        x_maxs_ge = []
        for i, q in enumerate(qubits):
            if config[f'single_qubit/{q}/GE/freq'] is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[config[f'single_qubit/{q}/GE/freq'] / GHz,
                       config[f'single_qubit/{q}/GE/freq'] / GHz],
                    y=[1, 0],
                    line_width=5,
                    mode='lines',
                    name=f'Q{q} GE',
                    line={'color': colors[i]}
                )
            )
            x_mins_ge.append(config[f'single_qubit/{q}/GE/freq'] / GHz)
            x_maxs_ge.append(config[f'single_qubit/{q}/GE/freq'] / GHz)
        x_mins.append(min(x_mins_ge))
        x_maxs.append(max(x_maxs_ge))

    if plot_EF:
        x_mins_ef = []
        x_maxs_ef = []
        for i, q in enumerate(qubits):
            if config[f'single_qubit/{q}/EF/freq'] is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[config[f'single_qubit/{q}/EF/freq'] / GHz,
                       config[f'single_qubit/{q}/EF/freq'] / GHz],
                    y=[1, 0],
                    line_width=5,
                    mode='lines',
                    name=f'Q{q} EF',
                    line={'color': colors[i], 'dash': 'dash'}
                )
            )
            x_mins_ef.append(config[f'single_qubit/{q}/EF/freq'] / GHz)
            x_maxs_ef.append(config[f'single_qubit/{q}/EF/freq'] / GHz)
        x_mins.append(min(x_mins_ef))
        x_maxs.append(max(x_maxs_ef))

    if plot_readout:
        x_mins_ro = []
        x_maxs_ro = []
        for i, q in enumerate(qubits):
            if config[f'readout/{q}/freq'] is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[config[f'readout/{q}/freq'] / GHz,
                       config[f'readout/{q}/freq'] / GHz],
                    y=[1, 0],
                    line_width=5,
                    mode='lines',
                    name=f'R{q}',
                    line={'color': colors[i], 'dash': 'dashdot'}
                )
            )
            x_mins_ro.append(config[f'readout/{q}/freq'] / GHz)
            x_maxs_ro.append(config[f'readout/{q}/freq'] / GHz)
        x_mins.append(min(x_mins_ro))
        x_maxs.append(max(x_maxs_ro))

    if plot_two_qubit:
        i = -1
        x_mins_tq = []
        x_maxs_tq = []
        for qp in qubit_pairs:
            for gate in config.native_gates['two_qubit'][qp]:
                if config[f'two_qubit/{qp}/{gate}/freq'] is None:
                    continue
                i += 1
                fig.add_trace(
                    go.Scatter(
                        x=[config[f'two_qubit/{qp}/{gate}/freq'] / GHz,
                           config[f'two_qubit/{qp}/{gate}/freq'] / GHz],
                        y=[1, 0],
                        line_width=5,
                        mode='lines',
                        name=f'{gate} {qp}',
                        line={'color': colors_tq[i], 'dash': 'dot'}
                    )
                )
                x_mins_tq.append(config[f'two_qubit/{qp}/{gate}/freq'] / GHz)
                x_maxs_tq.append(config[f'two_qubit/{qp}/{gate}/freq'] / GHz)
        x_mins.append(min(x_mins_tq))
        x_maxs.append(max(x_maxs_tq))

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
        font={'size': 20},
        legend={'font': {'size': 15}}
    )

    save_properties = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'frequency_plot',
            'height': 500,
            'width': 1000,
            'scale': 10 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }

    fig.show(config=save_properties)
