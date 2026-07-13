"""Submodule for analyzing leakage.

"""
import logging
from math import gamma
from typing import Any, Dict
from unittest import result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from lmfit import Parameters

import qcal.settings as settings
from qcal.fitting.fit import FitExponential
from qcal.math.utils import round_to_order_error, uncertainty_of_product
from qcal.results import Results
from qcal.utils import save_to_pickle

logger = logging.getLogger(__name__)


def analyze_leakage(circuits: Any, filename: str | None = None) -> Dict:
    """Analyze leakage results for a set of circuits.

    Relevant paper:
    - https://arxiv.org/abs/1509.05470

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

        for q, i in zip(qubits, q_index, strict=False):
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
            gamma = fit[q].fit_params['b'].value # Γ = γ↑ + γ↓
            p_inf = fit[q].fit_params['c'].value # asymptote
            gamma_up = gamma * p_inf             # γ↑ = leakage rate per gate

            stderr_b = fit[q].result.params['b'].stderr
            stderr_c = fit[q].result.params['c'].stderr
            idx_b = fit[q].result.var_names.index('b')
            idx_c = fit[q].result.var_names.index('c')

            full_cov = fit[q].result.covar
            vals = np.array([gamma, p_inf])
            if full_cov is not None:
                cov = full_cov[np.ix_([idx_b, idx_c], [idx_b, idx_c])]
                gamma_up_sigma = uncertainty_of_product(vals, cov=cov)
            else:
                if stderr_b is not None and stderr_c is not None:
                    gamma_up_sigma = uncertainty_of_product(
                        vals, stds=np.array([stderr_b, stderr_c])
                    )
                else:
                    gamma_up_sigma = None

            if stderr_b is not None:
                g_disp, g_sig_disp = round_to_order_error(gamma, stderr_b, 2)
                gamma_str = f"Γ = {g_disp:.1e} ({g_sig_disp:.1e})"
            else:
                gamma_str = f"Γ = {gamma:.1e} (NaN)"

            if gamma_up_sigma is not None:
                gu_disp, gu_sig_disp = round_to_order_error(
                    gamma_up, gamma_up_sigma, 2
                )
                gamma_up_str = f"γ↑ = {gu_disp:.1e} ({gu_sig_disp:.1e})"
            else:
                gamma_up_str = f"γ↑ = {gamma_up:.1e} (NaN)"

            label[q] += f": {gamma_str}, {gamma_up_str}"
            print(label[q])


    mpl_colors = plt.cm.viridis(np.linspace(0, 1, max(len(qubits), 2)))
    plotly_colors = pc.sample_colorscale(
        'jet', np.linspace(0, 1, max(len(qubits), 2))
    )

    # Save with matplotlib
    if settings.Settings.save_data and filename is not None:
        fig = plt.figure(figsize=(10, 6))

        for i, q in enumerate(qubits):
            for depth, pops in leakage[q].items():
                plt.plot(
                    [depth] * len(pops), pops,
                    'o', c=mpl_colors[i], ms=3.0, alpha=0.25
                )
            plt.plot(
                leakage[q].keys(), [np.mean(v) for v in leakage[q].values()],
                'o', ms=10.0, c=mpl_colors[i], label=label[q]
            )
            if fit[q].fit_success:
                x = np.linspace(min(circuit_depths), max(circuit_depths), 100)
                plt.plot(x, fit[q]._result.eval(x=x), c=mpl_colors[i], ls='--')

        plt.xlabel('Circuit Depth', fontsize=15)
        plt.ylabel(r'$|2\rangle$ State Population', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')

        fig.savefig(filename + 'leakage.png', dpi=300)
        fig.savefig(filename + 'leakage.pdf')
        fig.savefig(filename + 'leakage.svg')
        save_to_pickle(
            pd.DataFrame(leakage), filename=filename + 'leakage'
        )
        plt.close('all')

    # Display interactively with plotly
    pfig = go.Figure()

    for i, q in enumerate(qubits):
        color = plotly_colors[i]
        for depth, pops in leakage[q].items():
            pfig.add_trace(go.Scatter(
                x=[depth] * len(pops),
                y=pops,
                mode='markers',
                marker={'size': 5, 'color': color, 'opacity': 0.35},
                showlegend=False,
            ))
        pfig.add_trace(go.Scatter(
            x=list(leakage[q].keys()),
            y=[np.mean(v) for v in leakage[q].values()],
            mode='markers',
            marker={'size': 10, 'color': color},
            name=label[q],
        ))
        if fit[q].fit_success:
            x = np.linspace(min(circuit_depths), max(circuit_depths), 100)
            pfig.add_trace(go.Scatter(
                x=x,
                y=fit[q]._result.eval(x=x),
                mode='lines',
                line={'color': color, 'width': 2, 'dash': 'dash'},
                showlegend=False,
            ))

    pfig.update_layout(
        xaxis_title='Circuit Depth',
        yaxis_title='|2⟩ State Population',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02},
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='#fbfbfd',
    )
    pfig.update_xaxes(
        automargin=True,
        showline=True,
        mirror=True,
        linecolor='#c7c7c7',
        linewidth=1,
        gridcolor='#e5e7eb',
        zeroline=False,
        ticks='outside',
    )
    pfig.update_yaxes(
        automargin=True,
        showline=True,
        mirror=True,
        linecolor='#c7c7c7',
        linewidth=1,
        gridcolor='#e5e7eb',
        zeroline=False,
        ticks='outside',
    )
    pfig.show(config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'leakage',
            'scale': 10,
        }
    })

    return leakage
