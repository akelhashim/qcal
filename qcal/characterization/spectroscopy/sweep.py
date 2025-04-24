"""Submodule for qubit spectroscopy.

"""
import qcal.settings as settings

from qcal.characterization.characterize import Characterize
from qcal.circuit import Cycle, Circuit, CircuitSet
from qcal.config import Config
from qcal.fitting.fit import FitLinear
from qcal.gate.single_qubit import Meas
from qcal.qpu.qpu import QPU
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
from matplotlib.ticker import ScalarFormatter
from numpy.typing import NDArray
from typing import Callable, Dict

logger = logging.getLogger(__name__)


def Sweep2D(             
        qpu:      QPU,
        config:   Config,
        qubits:   Iterable[int],
        sweep1:   NDArray | Dict[int, NDArray],
        sweep2:   NDArray | Dict[int, NDArray],
        params:   Dict[int, Iterable[str]],
        prepulse: Cycle | Circuit = None,
        xlabel:   str = '',
        ylabel:   str = '',
        **kwargs
    ) -> Callable:
    """2D sweep for qubit spectroscopy.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List[int] | Tuple[int]): qubits to measure.
        sweep1 (NDArray | Dict[int, NDArray]): first value to sweep over for 
            each qubit.
        sweep2 (NDArray | Dict[int, NDArray]): second value to sweep over for 
            each qubit.
        params (Dict[int, Iterable[str]]): dictionary mapping qubit label to 
            sweep parameters in the config. Defaults to None.
        prepulse (Cycle | Circuit, optional): optional pre-pulse to add before
            the normal spectroscopy circuit. Defaults to None.

    Returns:
        Callable: Sweep2D
    """
    
    class Sweep2D(qpu, Characterize):

        def __init__(self, 
                config:   Config,
                qubits:   Iterable[int],
                sweep1:   NDArray | Dict[int, NDArray],
                sweep2:   NDArray | Dict[int, NDArray],
                params:   Dict[int, Iterable[str]],
                prepulse: Cycle | Circuit = None,
                xlabel:   str = '',
                ylabel:   str = '',
                **kwargs
            ) -> None:
            """
            Instantiate the Sweep2D class within the function.
            """
            qpu.__init__(self, config=config, **kwargs)
            Characterize.__init__(self, config)

            self._qubits = qubits

            if not isinstance(sweep1, dict):
                self._sweep1 = {q: sweep1 for q in qubits}
            else:
                self._sweep1 = sweep1

            if not isinstance(sweep2, dict):
                self._sweep2 = {q: sweep2 for q in qubits}
            else:
                self._sweep2 = sweep2

            for q in qubits:
                self._param_sweep[q] = (self._sweep1[q], self._sweep2[q])

            self._params = params
            self._prepulse = prepulse
            self._xlabel = xlabel
            self._ylabel = ylabel

            self._iq = {q: [] for q in qubits}
            self._mag = {q: [] for q in qubits}
            self._phase = {q: [] for q in qubits}
            self._linear_fit = {q: FitLinear() for q in qubits}

        @property
        def sweep1(self) -> Dict:
            """Sweep 1.

            Returns:
                Dict: first sweep for each qubit.
            """
            return self._sweep1
        
        @property
        def sweep2(self) -> Dict:
            """Sweep 2.

            Returns:
                Dict: second sweep for each qubit.
            """
            return self._sweep2
        
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
            if self._prepulse:
                if isinstance(self._prepulse, Cycle):
                    circuit.prepend(self._prepulse)
                elif isinstance(self._prepulse, Circuit):
                    circuit.prepend_circuit(self._prepulse)

            self._circuits = CircuitSet()
            for _ in range(len(list(self._sweep1.values())[0])):
                for _ in range(len(list(self._sweep2.values())[0])):
                    self._circuits.append(circuit.copy())

            for q in self._qubits:
                sweep1 = []
                for s in self._sweep1[q]:
                    sweep1.extend([s] * self._sweep2[q].size)
                
                self._circuits[f'param: {self._params[q][0]}'] = (
                    sweep1
                )
                self._circuits[f'param: {self._params[q][1]}'] = list(
                    self._sweep2[q]
                ) * self._sweep1[q].size

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            for q in self._qubits:
                for i, param in enumerate(self._params[q]):
                    if 'freq' in param:
                        idx = i
                        break
                for val in self._param_sweep[q][idx]:
                    circuits = self._circuits._df.loc[
                        self._circuits[f'param: {self._params[q][idx]}'] == val
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
                f'_2D_Sweep_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)

        def plot(self) -> None:
            """Plot the sweep results."""
            nrows, ncols = len(self._qubits), 2
            
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            for i, q in enumerate(self._qubits):
                if len(self._qubits) == 1:
                    ax = axes
                else:
                    ax = axes[i]

                for j, param in enumerate(self._params[q]):
                    if 'freq' in param:
                        idx = j
                        break
            
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
                        0.05, 0.9, f'Q{q}', size=12, 
                        transform=ax[0].transAxes
                    )
                ax[0].set_xlabel(f'{self._xlabel}', fontsize=12)
                ax[0].set_ylabel(f'{self._ylabel}', fontsize=12)
                ax[0].set_xticklabels(
                    np.around(self._sweep1[q][xtick_values] / GHz, 3) if idx==0
                    else np.around(self._sweep1[q][xtick_values], 3)
                )
                ax[0].set_yticklabels(
                    np.around(self._sweep2[q][ytick_values] / GHz, 3) if idx==1
                    else np.around(self._sweep2[q][ytick_values], 3)
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
                        0.05, 0.9, f'Q{q}', size=12, 
                        transform=ax[1].transAxes
                    )
                ax[1].set_xlabel(f'{self._xlabel}', fontsize=12)
                ax[1].set_ylabel(f'{self._ylabel}', fontsize=12)
                ax[1].set_xticklabels(
                    np.around(self._sweep1[q][xtick_values] / GHz, 3) if idx==0
                    else np.around(self._sweep1[q][xtick_values], 3)
                )
                ax[1].set_yticklabels(
                    np.around(self._sweep2[q][ytick_values] / GHz, 3) if idx==1
                    else np.around(self._sweep2[q][ytick_values], 3)
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
                    self._data_manager._save_path + '2D_sweep.png', 
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + '2D_sweep.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + '2D_sweep.svg'
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

    return Sweep2D(
        config=config,
        qubits=qubits,
        sweep1=sweep1,
        sweep2=sweep2,
        params=params,
        prepulse=prepulse,
        xlabel=xlabel,
        ylabel=ylabel,
        **kwargs
    )
