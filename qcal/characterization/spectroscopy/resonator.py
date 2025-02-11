"""Submodule for resonator spectroscopy.

"""
import qcal.settings as settings

from qcal.characterization.characterize import Characterize
from qcal.circuit import Cycle, Circuit, CircuitSet
from qcal.config import Config
from qcal.fitting.fit import FitLinear
from qcal.gate.single_qubit import Meas
from qcal.qpu.qpu import QPU

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from collections.abc import Iterable
from IPython.display import clear_output
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from typing import Callable, Dict

logger = logging.getLogger(__name__)


def find_inflection_points(x: NDArray, y: NDArray, sigma=10) -> NDArray:
    """Find inflection points for resonator fitting.

    Args:
        x (NDArray): independent variable.
        y (NDArray): dependent variable.
        sigma (int, optional): standard deviation for Gaussian kernel. Defaults 
            to 10.

    Returns:
        NDArray: indices of inflection points.
    """
    y_smoothed = gaussian_filter1d(y, sigma=sigma)
    dy_dx = np.gradient(y_smoothed, x)
    d2y_dx2 = np.gradient(dy_dx, x)
    inflection_idxs = np.where(np.diff(np.sign(d2y_dx2)))[0]
    
    return inflection_idxs
    


def Resonator(             
        qpu:    QPU,
        config: Config,
        qubits: Iterable[int],
        freqs:  NDArray | Dict[int, NDArray],
        params: Dict | None = None,
        **kwargs
    ) -> Callable:
    """Resonator Spectroscopy.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List[int] | Tuple[int]): qubits to measure.
        freqs (NDArray | Dict[int, NDArray]): frequencies to sweep over for each
            qubit resonator.
        params (Dict | None, optional): dictionary mapping qubit label to sweep
            parameter in the config. Defaults to None.

    Returns:
        Callable: Resonator
    """
    
    class Resonator(qpu, Characterize):

        def __init__(self, 
                config: Config,
                qubits: Iterable[int],
                freqs:  NDArray | Dict[int, NDArray],
                params: Dict | None = None,
                **kwargs
            ) -> None:
            """
            Instantiate the Resonator Spectroscopy class within the function.
            """
            qpu.__init__(self, config=config, **kwargs)
            Characterize.__init__(self, config)

            self._qubits = qubits

            if not isinstance(freqs, dict):
                self._freqs = {q: freqs for q in qubits}
            else:
                self._freqs = freqs
            self._param_sweep = self._freqs

            if not params:
                self._params = {q: f'readout/{q}/freq' for q in qubits}
            else:
                self._params = params

            self._iq = {}
            self._mag = {}
            self._phase = {}
            self._linear_fit = {q: FitLinear() for q in qubits}

        def generate_circuits(self) -> None:
            """Generate spectroscopy circuits."""
            logger.info(' Generating circuits...')

            circuit = Circuit([
                Cycle({Meas(q) for q in self._qubits})
            ])

            self._circuits = CircuitSet()
            for _ in range(len(list(self._freqs.values())[0])):
                self._circuits.append(circuit.copy())

            for q in self._qubits:
                self._circuits[f'param: {self._params[q]}'] = self._freqs[q]

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            for q in self._qubits:
                iq = np.array(list(
                    self._circuits[f'Q{q}: iq_data']
                ))
                self._iq[q] = iq.sum(axis=1).flatten()
                self._mag[q] = np.abs(self._iq[q])
                self._phase[q] = np.unwrap(np.angle(self._iq[q]))
                
                # Fit the first 1/3 of the unwrapped phase to extract the slope
                n_elements = self._freqs[q].shape[0]
                self._linear_fit[q].fit(
                    self._freqs[q][:int(n_elements/10)],
                    self._phase[q][:int(n_elements/10)]
                )
                if self._linear_fit[q].fit_success:
                    m = self._linear_fit[q].fit_params[0]
                    self._iq[q] *= np.exp(-1j * self._freqs[q] * m)
                    self._phase[q] *= np.unwrap(np.angle(self._iq[q]))

                # Fit the phase by finding the inflection point
                try:
                    sigma = 10 
                    inflection_idxs = find_inflection_points(
                        self._freqs[q], self._phase[q], sigma=sigma
                    )
                    while len(inflection_idxs) > 1:
                        sigma += 2
                        inflection_idxs = find_inflection_points(
                            self._freqs[q], self._phase[q], sigma=sigma
                        )
                        if sigma == 50:
                            raise Exception("Fitting error!")
                    self._char_values[q] = self._freqs[q][inflection_idxs][0]
                except Exception as e:
                    self._char_values[q] = None
                    logger.warning(
                        f' Unable to find the resonator frequency for Q{q}: {e}'
                    )

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_res_spectroscopy_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._char_values]), 'frequencies'
                )

        def plot(self, interactive=False) -> None:
            """Plot the results."""
            if interactive:
                fig = make_subplots(rows=2, cols=1)
                colors = px.colors.sample_colorscale(
                    "turbo", 
                    [n / (
                         len(self._qubits) + 2 - 1) 
                         for n in range(len(self._qubits) + 2
                        )
                    ]
                )
                for i, q in enumerate(self._qubits):
                    fig.add_trace(
                        go.Scatter(
                            x=self._freqs[q], 
                            y=np.log(self._mag[q]), 
                            name=f'R{q}',
                            line=dict(width=4, color=colors[i + 1]),
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=self._freqs[q], 
                            y=self._phase[q], 
                            name=f'R{q}',
                            line=dict(width=4, color=colors[i + 1]),
                            showlegend=False
                        ),
                        row=2, col=1
                    )

                fig.update_layout(
                    height=800, title_text="Resonator Spectroscopy"
                )
                fig['layout']['xaxis']['title'] = 'Frequency (Hz)'
                fig['layout']['xaxis2']['title'] = 'Frequency (Hz)'
                fig['layout']['yaxis']['title'] = 'Log(Mag) (a.u.)'
                fig['layout']['yaxis2']['title'] = 'Unwrapped Phase (rad.)'

                save_properties = {
                    'toImageButtonOptions': {
                        'format': 'png', # one of png, svg, jpeg, webp
                        'filename': 'resonator_spectroscopy',
                        'height': 800,
                        'width': 1000,
                        'scale': 10 # Multiply all sizes by this factor
                    }
                }

                fig.show(config=save_properties)

            else:
                nrows = len(self._qubits)
                figsize = (12, 4 * nrows)
                fig, axes = plt.subplots(
                    nrows, 3, figsize=figsize, layout='constrained'
                )

                for i, q in enumerate(self._qubits):
                    if len(self._qubits) == 1:
                        ax = axes
                    else:
                        ax = axes[i]

                    ax[0].plot(
                        np.real(self._iq[q]), 
                        np.imag(self._iq[q]), 
                        'b-',
                        label=f'Meas, R{q}'
                    )
                    ax[1].plot(
                        self._freqs[q], 
                        np.log(self._mag[q]), 
                        'b-',
                        label=f'Meas, R{q}'
                    )
                    ax[2].plot(
                        self._freqs[q], 
                        self._phase[q], 
                        'b-',
                        label=f'Meas, R{q}'
                    )

                    if self._char_values[q]:
                        ax[1].axvline(
                            self._char_values[q], 
                            ls='--', c='k', label='Freq. fit'
                        )
                        ax[2].axvline(
                            self._char_values[q], 
                            ls='--', c='k', label='Freq. fit'
                        )
                    
                    
                    ax[0].set_xlabel('Re (I)', fontsize=15)
                    ax[1].set_xlabel('Frequency (Hz)', fontsize=15)
                    ax[2].set_xlabel('Frequency (Hz)', fontsize=15)
                    ax[0].set_ylabel('Im (Q)', fontsize=15)
                    ax[1].set_ylabel('Log(Mag) (a.u.)', fontsize=15)
                    ax[2].set_ylabel('Unwrapped Phase (rad.)', fontsize=15)
                    ax[0].tick_params(axis='both', which='major', labelsize=12)
                    ax[1].tick_params(axis='both', which='major', labelsize=12)
                    ax[2].tick_params(axis='both', which='major', labelsize=12)
                    ax[0].grid(True)
                    ax[1].grid(True)
                    ax[2].grid(True)
                    ax[0].legend()
                    ax[1].legend()
                    ax[2].legend()

                fig.set_tight_layout(True)
                if settings.Settings.save_data:
                    fig.savefig(
                        self._data_manager._save_path + 
                        'resonator_spectroscopy.png', 
                        dpi=300
                    )
                    fig.savefig(
                        self._data_manager._save_path + 
                        'resonator_spectroscopy.pdf'
                    )
                    fig.savefig(
                        self._data_manager._save_path + 
                        'resonator_spectroscopy.svg'
                    )
                plt.show()
            
        def final(self) -> None:
            """Final calibration method."""
            Characterize.final(self)
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return Resonator(
        config=config,
        qubits=qubits,
        freqs=freqs,
        params=params,
        **kwargs
    )