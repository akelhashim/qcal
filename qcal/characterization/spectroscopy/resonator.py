"""Submodule for resonator spectroscopy.

"""
import qcal.settings as settings

from qcal.characterization.characterize import Characterize
from qcal.circuit import Cycle, Circuit, CircuitSet
from qcal.config import Config
from qcal.fitting.fit import FitLinear
from qcal.gate.single_qubit import Meas
from qcal.qpu.qpu import QPU
# from qcal.characterization.spectroscopy.utils import find_inflection_points
from qcal.units import GHz

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from collections.abc import Iterable
from IPython.display import clear_output
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from typing import Callable, Dict

logger = logging.getLogger(__name__)


def Punchout(             
        qpu:    QPU,
        config: Config,
        qubits: Iterable[int],
        amps:   NDArray | Dict[int, NDArray],
        freqs:  NDArray | Dict[int, NDArray],
        params: Dict | None = None,
        **kwargs
    ) -> Callable:
    """Resonator Punchout.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List[int] | Tuple[int]): qubits to measure.
        amps (NDArray | Dict[int, NDArray]): amps to sweep over for each qubit
            resonator.
        freqs (NDArray | Dict[int, NDArray]): frequencies to sweep over for each
            qubit resonator.
        params (Dict | None, optional): dictionary mapping qubit label to sweep
            parameter in the config. Defaults to None.

    Returns:
        Callable: Punchout
    """
    
    class Punchout(qpu, Characterize):

        def __init__(self, 
                config: Config,
                qubits: Iterable[int],
                amps:   NDArray | Dict[int, NDArray],
                freqs:  NDArray | Dict[int, NDArray],
                params: Dict | None = None,
                **kwargs
            ) -> None:
            """
            Instantiate the Punchout class within the function.
            """
            qpu.__init__(self, config=config, **kwargs)
            Characterize.__init__(self, config)

            self._qubits = qubits

            if not isinstance(amps, dict):
                self._amps = {q: amps for q in qubits}
            else:
                self._amps = amps

            if not isinstance(freqs, dict):
                self._freqs = {q: freqs for q in qubits}
            else:
                self._freqs = freqs

            for q in qubits:
                self._param_sweep[q] = (self._amps[q], self._freqs[q])

            if not params:
                self._params = {}
                for q in qubits:
                    self._params[q] = (
                        f'readout/{q}/amp', f'readout/{q}/freq'
                    ) 
            else:
                self._params = params

            self._iq = {q: [] for q in qubits}
            self._mag = {q: [] for q in qubits}
            self._phase = {q: [] for q in qubits}
            self._linear_fit = {q: FitLinear() for q in qubits}

        @property
        def amps(self) -> Dict:
            """Amp sweep.

            Returns:
                Dict: amp sweep for each qubit.
            """
            return self._amps
        
        @property
        def frequencies(self) -> Dict:
            """Frequency sweep.

            Returns:
                Dict: frequency sweep for each qubit.
            """
            return self._freqs
        
        @property
        def iq(self) -> Dict:
            """IQ data.

            Returns:
                Dict: IQ data for each qubit.
            """
            return self._iq
        
        @property
        def mag(self) -> Dict:
            """Magnitude data.

            Returns:
                Dict: magnitude data for each qubit.
            """
            return self._mag
        
        @property
        def phase(self) -> Dict:
            """Phase data.

            Returns:
                Dict: phase data for each qubit.
            """
            return self._phase

        def generate_circuits(self) -> None:
            """Generate spectroscopy circuits."""
            logger.info(' Generating circuits...')

            circuit = Circuit([
                Cycle({Meas(q) for q in self._qubits})
            ])

            self._circuits = CircuitSet()
            for _ in range(len(list(self._amps.values())[0])):
                for _ in range(len(list(self._freqs.values())[0])):
                    self._circuits.append(circuit.copy())

            for q in self._qubits:
                idx_amp = [
                    i for i, param in enumerate(self._params[q]) if 'amp' in 
                    param
                ]
                idx_freq = [
                    i for i, param in enumerate(self._params[q]) if 'freq' in 
                    param
                ]

                amps = []
                for amp in self._amps[q]:
                    amps.extend([amp] * self._freqs[q].size)
                
                self._circuits[f'param: {self._params[q][idx_amp[0]]}'] = (
                    amps
                )
                self._circuits[f'param: {self._params[q][idx_freq[0]]}'] = list(
                    self._freqs[q]
                ) * self._amps[q].size

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            for q in self._qubits:
                for val in self._param_sweep[q][0]:
                    circuits = self._circuits._df.loc[
                        self._circuits[f'param: {self._params[q][0]}'] == val
                    ]
                    iq = np.squeeze(
                        np.array(list(circuits[f'Q{q}: iq_data'])).mean(axis=1)
                    )
                    self._iq[q].append(iq)
                    self._mag[q].append(np.abs(iq))
                    self._phase[q].append(np.unwrap(np.angle(iq)))
                
                self._iq[q] = np.array(self._iq[q])
                self._mag[q] = np.array(self._mag[q])
                self._phase[q] = np.array(self._phase[q])

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_res_punchout_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)

        # def plot(self, interactive=False) -> None:
            # """Plot the sweep results.

            # Args:
            #     interactive (bool, optional): whether to plot the 2D sweep 
            #         using interactive plotting with Plotly. Defaults to False.
            # """
        def plot(self) -> None:
            """Plot the sweep results."""
            nrows, ncols = len(self._qubits), 2
            # if interactive:  # TODO: colorbar bug when ploting multiple rows
            #     fig = make_subplots(
            #         rows=nrows, cols=ncols, start_cell="top-left"
            #     )
            #     for i, q in enumerate(self._qubits):
            #         fig.add_trace(
            #             go.Heatmap(
            #                 z=20 * np.log10(self._mag[q]),
            #                 x=np.around(self._freqs[q] / GHz, 3),
            #                 y=np.around(self._amps[q] , 3),
            #                 colorscale='Viridis',
            #                 colorbar=dict(title="Log(mag) (dB)"),
            #                 colorbar_x=0.45
            #             ),
            #             row=i+1, 
            #             col=1
            #         )

            #         fig.add_trace(
            #             go.Heatmap(
            #                 z=self._phase[q],
            #                 x=np.around(self._freqs[q] / GHz, 3),
            #                 y=np.around(self._amps[q] , 3),
            #                 colorscale='Viridis',
            #                 colorbar=dict(title="Unwrapped Phase (rad.)")
            #             ),
            #             row=i+1, 
            #             col=2
            #         )

            #         fig.update_layout(
            #             height=400 * len(self._qubits),
            #             title_text="Resonator Punchout"
            #         )
            #         fig['layout']['xaxis']['title'] = 'Frequency (GHz)'
            #         fig['layout']['xaxis2']['title'] = 'Frequency (GHz)'
            #         fig['layout']['yaxis']['title'] = 'Amplitude (a.u.)'
            #         fig['layout']['yaxis2']['title'] = 'Amplitude (a.u.)'

            #         save_properties = {
            #             'toImageButtonOptions': {
            #                 'format': 'png', # one of png, svg, jpeg, webp
            #                 'filename': 'resonator_punchout',
            #                 'height': 400 * len(self._qubits),
            #                 'width': 1000,
            #                 'scale': 10 # Multiply all sizes by this factor
            #             }
            #         }
            #         fig.show(config=save_properties)
            
            # else:
            
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            for i, q in enumerate(self._qubits):
                if len(self._qubits) == 1:
                    ax = axes
                else:
                    ax = axes[i]
            
                p = sns.heatmap(
                    20 * np.log10(self._mag[q]), 
                    cmap='viridis', 
                    cbar=True,
                    ax=ax[0],
                )
                xtick_values = [
                    int(tick.get_text()) for tick in ax[0].get_xticklabels()
                ]
                ytick_values = [
                    int(tick.get_text()) for tick in ax[0].get_yticklabels()
                ]
                cbar = p.collections[0].colorbar
                cbar.ax.set_ylabel("Log(Mag) (dB)", fontsize=10)
                cbar.ax.tick_params(labelsize=10)
                # ax[0].set_title(f'R{q}', fontsize=12)
                ax[0].text(
                        0.05, 0.9, f'R{q}', size=12, 
                        transform=ax[0].transAxes
                    )
                ax[0].set_xlabel('Frequency (GHz)', fontsize=12)
                ax[0].set_ylabel('Amplitude (a.u.)', fontsize=12)
                ax[0].set_xticklabels(
                    np.around(self._freqs[q][xtick_values] / GHz, 3)
                )
                ax[0].set_yticklabels(
                    np.around(self._amps[q][ytick_values], 3)
                )
                ax[0].tick_params(
                    axis='x', which='major', labelsize=10, labelrotation=45
                )
                ax[0].tick_params(
                    axis='y', which='major', labelsize=10, labelrotation=0
                )
                ax[0].invert_yaxis()

                p = sns.heatmap(
                    self._phase[q], 
                    cmap='viridis', 
                    cbar=True,
                    ax=ax[1],
                )
                xtick_values = [
                    int(tick.get_text()) for tick in ax[1].get_xticklabels()
                ]
                ytick_values = [
                    int(tick.get_text()) for tick in ax[1].get_yticklabels()
                ]
                cbar = p.collections[0].colorbar
                cbar.ax.set_ylabel("Unwrapped Phase (rad.)", fontsize=10)
                cbar.ax.tick_params(labelsize=10)
                # ax[1].set_title(f'R{q}', fontsize=12)
                ax[1].text(
                        0.05, 0.9, f'R{q}', size=12, 
                        transform=ax[1].transAxes
                    )
                ax[1].set_xlabel('Frequency (GHz)', fontsize=12)
                ax[1].set_ylabel('Amplitude (a.u.)', fontsize=12)
                ax[1].set_xticklabels(
                    np.around(self._freqs[q][xtick_values] / GHz, 3)
                )
                ax[1].set_yticklabels(
                    np.around(self._amps[q][ytick_values], 3)
                )
                ax[1].tick_params(
                    axis='x', which='major', labelsize=10, labelrotation=45
                )
                ax[1].tick_params(
                    axis='y', which='major', labelsize=10, labelrotation=0
                )
                ax[1].invert_yaxis()

            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'resonator_punchout.png', 
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'resonator_punchout.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'resonator_punchout.svg'
                )
            plt.show()
            
        def final(self) -> None:
            """Final calibration method."""
            # Characterize.final(self)
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return Punchout(
        config=config,
        qubits=qubits,
        amps=amps,
        freqs=freqs,
        params=params,
        **kwargs
    )


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

        @property
        def frequencies(self) -> Dict:
            """Frequency sweep.

            Returns:
                Dict: frequency sweep for each qubit.
            """
            return self._freqs
        
        @property
        def iq(self) -> Dict:
            """IQ data.

            Returns:
                Dict: IQ data for each qubit.
            """
            return self._iq
        
        @property
        def mag(self) -> Dict:
            """Magnitude data.

            Returns:
                Dict: magnitude data for each qubit.
            """
            return self._mag
        
        @property
        def phase(self) -> Dict:
            """Phase data.

            Returns:
                Dict: phase data for each qubit.
            """
            return self._phase

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
                self._iq[q] = np.squeeze(iq.mean(axis=1))
                self._mag[q] = np.abs(self._iq[q])
                self._phase[q] = np.unwrap(np.angle(self._iq[q]))
                
                # Fit the first 1/3 of the unwrapped phase to extract the slope
                n_elements = self._freqs[q].shape[0]
                self._linear_fit[q].fit(
                    self._freqs[q][:int(n_elements/4)],
                    self._phase[q][:int(n_elements/4)]
                )
                if self._linear_fit[q].fit_success:
                    m = self._linear_fit[q].fit_params[0]
                    self._iq[q] *= np.exp(-1j * self._freqs[q] * m)
                    self._phase[q] = np.unwrap(np.angle(self._iq[q]))

                # Fit the phase by finding the minimum magnitude
                try:
                    x = 20 * np.log10(self._mag[q])
                    x_s = gaussian_filter1d(x, sigma=3)
                    threshold = x.mean() + (x[x.argmin()] - x.mean()) / 2
                    peaks, _ = find_peaks(-x_s, height=-threshold)
                    if len(peaks) == 0:
                        self._char_values[q] = self._freqs[q][
                            np.argmin(self._mag[q])
                        ]
                    elif len(peaks) == 1:
                        self._char_values[q] = self._freqs[q][peaks][0]
                    else:
                        raise Exception("Fitting error! Too many peaks.")
                    
                    # sigma = 10 
                    # inflection_idxs = find_inflection_points(
                    #     self._freqs[q], self._phase[q], sigma=sigma
                    # )
                    # while len(inflection_idxs) > 1:
                    #     sigma += 2
                    #     inflection_idxs = find_inflection_points(
                    #         self._freqs[q], self._phase[q], sigma=sigma
                    #     )
                    #     if sigma == 50:
                    #         raise Exception("Fitting error!")
                    # self._char_values[q] = self._freqs[q][inflection_idxs][0]
                
                except Exception as e:
                    self._char_values[q] = None
                    logger.warning(
                        f' Unable to find the resonator frequency for R{q}: {e}'
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
            """Plot the results.
            
            Args:
                interactive (bool, optional): whether to plot the spectroscopy
                    results using interactive plotting with Plotly. Defaults to 
                    False.
            """
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
                            y=20 * np.log10(self._mag[q]), 
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
                fig['layout']['yaxis']['title'] = 'Log(Mag) (dB)'
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
                        20 * np.log10(self._mag[q]), 
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
                    ax[1].set_ylabel('Log(Mag) (dB)', fontsize=15)
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