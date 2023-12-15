"""Submodule for two-qubit CZ gate calibrations.

"""
from __future__ import annotations

import qcal.settings as settings

from .calibration import Calibration
from .utils import find_pulse_index, in_range
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.fitting.fit import (
    # FitAbsoluteValue, FitDecayingCosine, 
    FitCosine, FitParabola
)
from qcal.managers.classification_manager import ClassificationManager
from qcal.math.utils import wrap_phase
from qcal.gate.single_qubit import Meas, VirtualZ, X90
from qcal.gate.two_qubit import CZ
from qcal.plotting.utils import calculate_nrows_ncols
from qcal.qpu.qpu import QPU
from qcal.units import GHz

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import plotly.express as px
import plotly.graph_objects as go

# from collections.abc import Iterable
from IPython.display import clear_output
from itertools import chain
from typing import Any, Callable, Dict, List, Tuple
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def tomography_circuits(
        qubit_pairs: List[Tuple], n_elements: int, n_gates: int = 1
    ) -> CircuitSet:
    """Partial tomography circuits for CZ sweeps.

    Args:
        qubit_pairs (List[Tuple]): list of qubit pairs.
        n_elements (int):          size of parameter sweep.
        n_gates (int, optional): number of gates for pulse repetition.
            Defaults to 1.

    Returns:
        CircuitSet: partial tomography CZ circuits.
    """
    qubits = list(set(chain.from_iterable(qubit_pairs)))

    circuit_C0_X = Circuit([
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        Barrier(qubits)
    ])
    for _ in range(n_gates):
        circuit_C0_X.extend([
            Cycle({CZ(pair) for pair in qubit_pairs}),
            Barrier(qubits),
        ])
    circuit_C0_X.extend([  
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
    ])
    circuit_C0_X.measure()

    circuit_C1_X = Circuit([
        # X on control qubit
        Cycle({X90(p[0]) for p in qubit_pairs}),
        Cycle({X90(p[0]) for p in qubit_pairs}),
        Barrier(qubits),
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        Barrier(qubits)
    ])
    for _ in range(n_gates):
        circuit_C1_X.extend([
            Cycle({CZ(pair) for pair in qubit_pairs}),
            Barrier(qubits),
        ])
    circuit_C1_X.extend([
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
    ])
    circuit_C1_X.measure()

    circuit_C0_Y = Circuit([
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        Barrier(qubits)
    ])
    for _ in range(n_gates):
        circuit_C0_Y.extend([
            Cycle({CZ(pair) for pair in qubit_pairs}),
            Barrier(qubits),
        ])
    circuit_C0_Y.extend([
        # X90 on target qubit
        Cycle({X90(p[1]) for p in qubit_pairs}),
    ])
    circuit_C0_Y.measure()

    circuit_C1_Y = Circuit([
        # X on control qubit
        Cycle({X90(p[0]) for p in qubit_pairs}),
        Cycle({X90(p[0]) for p in qubit_pairs}),
        Barrier(qubits),
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        Barrier(qubits)
    ])
    for _ in range(n_gates):
        circuit_C1_Y.extend([
            Cycle({CZ(pair) for pair in qubit_pairs}),
            Barrier(qubits),
        ])
    circuit_C1_Y.extend([
        # X90 on target qubit
        Cycle({X90(p[1]) for p in qubit_pairs}),
    ])
    circuit_C1_Y.measure()

    circuits = list()
    circuits.extend([circuit_C0_X.copy() for _ in range(n_elements)])
    circuits.extend([circuit_C1_X.copy() for _ in range(n_elements)])
    circuits.extend([circuit_C0_Y.copy() for _ in range(n_elements)])
    circuits.extend([circuit_C1_Y.copy() for _ in range(n_elements)])
    
    cs = CircuitSet(circuits)
    cs['sequence'] = (
        ['C0_X'] * n_elements + ['C1_X'] * n_elements +
        ['C0_Y'] * n_elements + ['C1_Y'] * n_elements 
    )

    return cs


def AmpFreqSweep(
        qpu:             QPU,
        config:          Config,
        qubit_pairs:     List[Tuple],
        amplitudes:      ArrayLike | NDArray | Dict[ArrayLike | NDArray],
        frequencies:     ArrayLike | NDArray | Dict[ArrayLike | NDArray],
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        n_gates:         int = 1,
        raster_circuits: bool = False,
        **kwargs
    ) -> Callable:
    """Amplitude & Frequency sweep for CZ gate.

    This sweep searches a 2D lanscape to find the where the conditionality is 
    a maximum.

    Basic example useage for initial calibration:

    ```
    amplitudes = np.linspace(0.0, 0.5, 21)
    frequencies = np.linspace(5.5, 5.6, 21) * GHz
    sweep = AmpFreqSweep(
        CustomQPU, 
        config, 
        qubit_pairs=[(0, 1), (2, 3)],
        amplitudes=amplitudes,
        frequencies=frequencies)
    sweep.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_pairs (List[Tuple]): pairs of qubit labels for the two-qubit gate
            calibration.
        amplitudes (ArrayLike | NDArray | Dict[ArrayLike | NDArray]): array of
            amplitudes to sweep over for calibrating the two-qubit CZ gate. 
            These amplitudes are swept over both the control and target qubit
            lines. If calibrating multiple gates at the same time, this should 
            be a dictionary mapping two arrays to each qubit pair label.
        frequencies (ArrayLike | NDArray | Dict[ArrayLike | NDArray]): array of
            frequencies to sweep over for calibrating the two-qubit CZ gate. 
            If calibrating multiple gates at the same time, this should be a 
            dictionary mapping an array to each qubit pair label.
        compiler (Any | Compiler | None, optional): custom compiler to
            compile the experimental circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to 
            transpile the experimental circuits. Defaults to None.
        classifier (ClassificationManager, optional): manager used for 
            classifying raw data. Defaults to None.
        n_shots (int, optional): number of measurements per circuit. 
            Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. 
            Defaults to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that
            can be measured per sequence. Defaults to 1.
        n_levels (int, optional): number of energy levels to be measured. 
            Defaults to 2. If n_levels = 3, this assumes that the
            measurement supports qutrit classification.
        n_gates (int, optional): number of gates for pulse repetition.
            Defaults to 1.
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.

    Returns:
        Callable: AmpFreqSweep class.
    """

    class AmpFreqSweep(qpu, Calibration):
        """AmpFreqSweep class.
        
        This class inherits a custom QPU from the AmpFreqSweep function.
        """

        def __init__(self, 
                config:          Config,
                qubit_pairs:     List[Tuple],
                amplitudes:      ArrayLike | Dict[ArrayLike],
                frequencies:     ArrayLike | Dict[ArrayLike],
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2,
                n_gates:         int = 1,
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            """Initialize the AmpFreqSweep within the function."""
            qpu.__init__(self,
                config=config, 
                compiler=compiler, 
                transpiler=transpiler,
                classifier=classifier,
                n_shots=n_shots, 
                n_batches=n_batches, 
                n_circs_per_seq=n_circs_per_seq, 
                n_levels=n_levels,
                raster_circuits=raster_circuits,
                **kwargs
            )
            Calibration.__init__(self, config)

            self._qubits = qubit_pairs
            self._n_gates = n_gates

            self._params = {}
            for pair in qubit_pairs:
                idx = find_pulse_index(config, f'two_qubit/{pair}/CZ/pulse')
                self._params[pair] = (
                    f'two_qubit/{pair}/CZ/pulse/{idx}/kwargs/amp',
                    f'two_qubit/{pair}/CZ/pulse/{idx+1}/kwargs/amp',
                    f'two_qubit/{pair}/CZ/freq'
                )

            self._amplitudes = {}
            self._frequencies = {}
            for pair in qubit_pairs:
                if isinstance(amplitudes, dict):
                    self._amplitudes[pair] = (
                        amplitudes[pair], amplitudes[pair]
                    )
                else:
                    self._amplitudes[pair] = (amplitudes, amplitudes)

                if isinstance(frequencies, dict):
                    self._frequencies[pair] = frequencies[pair]
                else:
                    self._frequencies[pair] = frequencies
                self._param_sweep[pair] = (
                    self._amplitudes[pair], self._frequencies[pair]
                )

            self._R = {}

        @property
        def amplitudes(self) -> Dict:
            """Amp sweep for each qubit.

            Returns:
                Dict: qubit to array map.
            """
            return self._amplitudes
        
        @property
        def frequencies(self) -> Dict:
            """Frequency sweep for each qubit pair.

            Returns:
                Dict: qubit to array map.
            """
            return self._frequencies
        
        @property
        def qubit_pairs(self) -> List[Tuple]:
            """Qubit pair labels.

            Returns:
                List[Tuple]: qubit pairs.
            """
            return self._qubits
        
        @property
        def conditionality(self) -> Dict[Tuple]:
            """Conditionality computed at each phase value.

            Returns:
                Dict[Tuple]: conditionality for each pair.
            """
            return self._R

        def generate_circuits(self) -> None:
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            self._circuits = tomography_circuits(
                self._qubits,
                (self._amplitudes[self._qubits[0]][0].size * 
                 self._frequencies[self._qubits[0]].size
                )
            )

            for pair in self._qubits:
                freqs = []
                for freq in self._frequencies[pair]:
                    freqs.extend([freq] * self._amplitudes[pair][0].size)
                freqs *= 4
                self._circuits[f'param: {self._params[pair][0]}'] = list(
                    self._amplitudes[pair][0]
                ) * 4 * self._frequencies[pair].size
                self._circuits[f'param: {self._params[pair][1]}'] = list(
                    self._amplitudes[pair][1]
                ) * 4 * self._frequencies[pair].size
                self._circuits[f'param: {self._params[pair][2]}'] = freqs

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            # Compute the conditionality for each amp/freq
            qubits = list(set(chain.from_iterable(self._qubits)))
            for pair in self._qubits:
                self._R[pair] = np.zeros((
                    self._frequencies[pair].size,
                    self._amplitudes[pair][0].size
                ))
                for f, freq in enumerate(self._frequencies[pair]):
                    i = qubits.index(pair[1])

                    prob_C0_X = []
                    for circuit in self._circuits._df.loc[
                        (self._circuits['sequence'] == 'C0_X') & 
                        (self._circuits[f'param: two_qubit/{pair}/CZ/freq'
                            ] == freq)
                        ].circuit:
                        prob_C0_X.append(
                            circuit.results.marginalize(i).populations['0']
                        )

                    prob_C1_X = []
                    for circuit in self._circuits._df.loc[
                        (self._circuits['sequence'] == 'C1_X') & 
                        (self._circuits[f'param: two_qubit/{pair}/CZ/freq'
                            ] == freq)
                        ].circuit:
                        prob_C1_X.append(
                            circuit.results.marginalize(i).populations['0']
                        )

                    prob_C0_Y = []
                    for circuit in self._circuits._df.loc[
                        (self._circuits['sequence'] == 'C0_Y') & 
                        (self._circuits[f'param: two_qubit/{pair}/CZ/freq'
                            ] == freq)
                        ].circuit:
                        prob_C0_Y.append(
                            circuit.results.marginalize(i).populations['0']
                        )

                    prob_C1_Y = []
                    for circuit in self._circuits._df.loc[
                        (self._circuits['sequence'] == 'C0_X') & 
                        (self._circuits[f'param: two_qubit/{pair}/CZ/freq'
                            ] == freq)
                        ].circuit:
                        prob_C1_Y.append(
                            circuit.results.marginalize(i).populations['0']
                        )

                    self._R[pair][f] = np.sqrt(
                        (np.array(prob_C1_X) - np.array(prob_C0_X)) ** 2 + 
                        (np.array(prob_C1_Y) - np.array(prob_C0_Y)) ** 2
                    )

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CZ_Amp_Freq_Sweep_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)

        def plot(self, interactive=False) -> None:
            """Plot the sweep results.

            Args:
                interactive (bool, optional): whether to plot the 2D sweep 
                    using interactive plotting with Plotly. Defaults to False.
            """
            import seaborn as sns
            nrows, ncols = calculate_nrows_ncols(len(self._qubits), 2)
            figsize = (10 * ncols, 8 * nrows)
            if interactive:
                fig = make_subplots(
                    rows=nrows, cols=ncols, start_cell="top-left"
                )
            else:
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=figsize, layout='constrained'
                )

            k = -1
            for i in range(nrows):
                for j in range(ncols):
                    k += 1

                    if not interactive:
                        if len(self._qubits) == 1:
                            ax = axes
                        elif axes.ndim == 1:
                            ax = axes[j]
                        elif axes.ndim == 2:
                            ax = axes[i,j]

                    if k < len(self._qubits):
                        q = self._qubits[k]

                        if interactive:
                            fig.add_trace(go.Heatmap(
                                    z=self._R[q],
                                    x=np.around(self._amplitudes[q][1], 3),
                                    y=np.around(self._frequencies[q] / GHz, 3),
                                    colorscale='Viridis'
                                ),
                                row=i+1, 
                                col=j+1
                            ).update_layout(
                                xaxis_title="Amplitude (a.u.)", 
                                yaxis_title="Frequency (GHz)"
                            )

                        else:
                            sns.heatmap(
                                self._R[q], 
                                cmap='viridis', 
                                cbar_kws={'label': 'R'},
                                ax=ax,
                                yticklabels=np.around(
                                    self._frequencies[q] / GHz, 3
                                ),
                                xticklabels=np.around(
                                    self._amplitudes[q][1], 3
                                )
                            )
                            ax.set_xlabel('Amplitude (a.u.)', fontsize=15)
                            ax.set_ylabel('Frequency (GHz)', fontsize=15)
                            ax.tick_params(
                                axis='both', which='major', labelsize=12
                            )
                            ax.invert_yaxis()

                    else:
                        if not interactive:
                            ax.axis('off')
            
            if interactive:
                fig.show()
            else:
                if settings.Settings.save_data:
                    fig.savefig(
                        self._data_manager._save_path + 'amp_freq_sweep.png', 
                        dpi=300
                    )
                plt.show()

        def final(self) -> None:
            """Final calibration method."""
            Calibration.final(self)
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()
            

    return AmpFreqSweep(
        config,
        qubit_pairs,
        amplitudes,
        frequencies,
        compiler, 
        transpiler,
        classifier,
        n_shots, 
        n_batches, 
        n_circs_per_seq,
        n_levels,
        n_gates,
        raster_circuits,
        **kwargs
    )


def Amplitude(
        qpu:             QPU,
        config:          Config,
        qubit_pairs:     List[Tuple],
        amplitudes:      ArrayLike | NDArray | Dict[ArrayLike | NDArray],
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        n_gates:         int = 1,
        relative_amp:    bool = False,
        raster_circuits: bool = False,
        **kwargs
    ) -> Callable:
    """Amplitude calibration for CZ gate.

    This calibration finds the amplitude for both drive lines where the
    conditionality is a maximum.

    Basic example useage for initial calibration:

    ```
    amplitudes = np.linspace(0.0, 0.5, 21)
    cal = Amplitude(
        CustomQPU, 
        config, 
        qubit_pairs=[(0, 1), (2, 3)],
        amplitudes=amplitudes)
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_pairs (List[Tuple]): pairs of qubit labels for the two-qubit gate
            calibration.
        amplitudes (ArrayLike | NDArray | Dict[ArrayLike | NDArray]): array of
            amplitudes to sweep over for calibrating the two-qubit CZ gate. 
            These amplitudes are swept over both the control and target qubit
            lines. If calibrating multiple gates at the same time, this should 
            be a dictionary mapping two arrays to each qubit pair label.
        compiler (Any | Compiler | None, optional): custom compiler to
            compile the experimental circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to 
            transpile the experimental circuits. Defaults to None.
        classifier (ClassificationManager, optional): manager used for 
            classifying raw data. Defaults to None.
        n_shots (int, optional): number of measurements per circuit. 
            Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. 
            Defaults to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that
            can be measured per sequence. Defaults to 1.
        n_levels (int, optional): number of energy levels to be measured. 
            Defaults to 2. If n_levels = 3, this assumes that the
            measurement supports qutrit classification.
        n_gates (int, optional): number of gates for pulse repetition.
            Defaults to 1.
        relative_amp (bool, optional): whether or not the amplitudes argument
            is defined relative to the existing pulse amplitude. Defaults to
            False. If True, the amplitudes are swept over the current amplitude
            times the amplitudes argument.
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.

    Returns:
        Callable: Amplitude calibration class.
    """

    class Amplitude(qpu, Calibration):
        """Amplitude calibration class.
        
        This class inherits a custom QPU from the Amplitude calibration
        function.
        """

        def __init__(self, 
                config:          Config,
                qubit_pairs:     List[Tuple],
                amplitudes:      ArrayLike | Dict[ArrayLike],
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2,
                n_gates:         int = 1,
                relative_amp:    bool = False,
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            """Initialize the Amplitude class within the function."""
            qpu.__init__(self,
                config=config, 
                compiler=compiler, 
                transpiler=transpiler,
                classifier=classifier,
                n_shots=n_shots, 
                n_batches=n_batches, 
                n_circs_per_seq=n_circs_per_seq, 
                n_levels=n_levels,
                raster_circuits=raster_circuits,
                **kwargs
            )
            Calibration.__init__(self, config)

            self._qubits = qubit_pairs
            self._n_gates = n_gates

            self._params = {}
            for pair in qubit_pairs:
                idx = find_pulse_index(config, f'two_qubit/{pair}/CZ/pulse')
                self._params[pair] = (
                    f'two_qubit/{pair}/CZ/pulse/{idx}/kwargs/amp',
                    f'two_qubit/{pair}/CZ/pulse/{idx+1}/kwargs/amp'
                )

            self._amplitudes = {}
            for pair in qubit_pairs:
                if isinstance(amplitudes, dict):
                    amplitude = amplitudes[pair]
                else:
                    amplitude = amplitudes
                
                if relative_amp:
                    self._amplitudes[pair] = (
                        self._config[self._params[pair][0]] * amplitude,
                        self._config[self._params[pair][1]] * amplitude,
                    ) 
                else:
                    self._amplitudes[pair] = (amplitude, amplitude)
                self._param_sweep[pair] = self._amplitudes[pair]
                
            self._R = {}
            self._fit = {
                pair: (FitParabola(), FitParabola()) for pair in qubit_pairs
            }

        @property
        def amplitudes(self) -> Dict:
            """Amp sweep for each qubit.

            Returns:
                Dict: qubit to array map.
            """
            return self._amplitudes
        
        @property
        def qubit_pairs(self) -> List[Tuple]:
            """Qubit pair labels.

            Returns:
                List[Tuple]: qubit pairs.
            """
            return self._qubits
        
        @property
        def conditionality(self) -> Dict[Tuple]:
            """Conditionality computed at each phase value.

            Returns:
                Dict[Tuple]: conditionality for each pair.
            """
            return self._R

        def generate_circuits(self) -> None:
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            self._circuits = tomography_circuits(
                self._qubits,
                self._amplitudes[self._qubits[0]][0].size
            )

            for pair in self._qubits:
                self._circuits[f'param: {self._params[pair][0]}'] = list(
                    self._amplitudes[pair][0]
                ) * 4
                self._circuits[f'param: {self._params[pair][1]}'] = list(
                    self._amplitudes[pair][1]
                ) * 4

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            # Compute the conditionality and fit to a parabola
            qubits = list(set(chain.from_iterable(self._qubits)))
            for pair in self._qubits:
                i = qubits.index(pair[1])

                prob_C0_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_X'].circuit:
                    prob_C0_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C1_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_X'].circuit:
                    prob_C1_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C0_Y = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_Y'].circuit:
                    prob_C0_Y.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C1_Y = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_Y'].circuit:
                    prob_C1_Y.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                self._circuits[f'{pair}: Prob(0)'] = (
                    prob_C0_X + prob_C1_X + prob_C0_Y + prob_C1_Y
                )
                self._R[pair] = np.sqrt(
                    (np.array(prob_C1_X) - np.array(prob_C0_X)) ** 2 + 
                    (np.array(prob_C1_Y) - np.array(prob_C0_Y)) ** 2
                )
                self._sweep_results[pair] = self._R[pair]

                self._fit[pair][0].fit(
                    self._amplitudes[pair][0], self._R[pair]
                )
                self._fit[pair][1].fit(
                    self._amplitudes[pair][1], self._R[pair]
                )
                # If the fit was successful, find the new amplitude
                self._cal_values[pair] = []
                for j in range(2):
                    if self._fit[pair][j].fit_success:
                        a, b, _ = self._fit[pair][j].fit_params
                        newvalue = -b / (2 * a)  # Assume c = 0
                        if a > 0:
                            logger.warning(
                              f'Fit failed for {pair[j]} (positive curvature)!'
                            )
                            self._fit[pair][j]._fit_success = False
                        elif not in_range(newvalue, self._amplitudes[pair]):
                            logger.warning(
                                f'Fit failed for {pair[j]} (out of range)!'
                            )
                            self._fit[pair][j]._fit_success = False
                        else:
                            self._cal_values[pair].append(newvalue)

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CZ_Amp_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._sweep_results]), 'sweep_results'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._cal_values]), 'calibrated_values'
                )

        def plot(self) -> None:
            """Plot the amplitude sweep and fit results."""
            nrows, ncols = calculate_nrows_ncols(len(self._qubits))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            k = -1
            for i in range(nrows):
                for j in range(ncols):
                    k += 1

                    if len(self._qubits) == 1:
                        ax = axes
                    elif axes.ndim == 1:
                        ax = axes[j]
                    elif axes.ndim == 2:
                        ax = axes[i,j]

                    if k < len(self._qubits):
                        q = self._qubits[k]

                        ax.set_xlabel('Target Amplitude (a.u.)', fontsize=15)
                        ax.set_ylabel('Conditionality', fontsize=15)
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )
                        ax.grid(True)

                        ax.plot(
                            self._param_sweep[q][1], self._sweep_results[q],
                            'o', c='blue', label=f'Meas, {q}'
                        )

                        if self._fit[q][1].fit_success:
                            x = np.linspace(
                                self._param_sweep[q][1][0],
                                self._param_sweep[q][1][-1], 
                                100
                            )
                            ax.plot(
                                x, self._fit[q][1].predict(x),
                                '-', c='orange', label='Fit'
                            )
                            ax.axvline(
                                self._cal_values[q][1],  
                                ls='--', c='k', label='Fit value'
                            )

                        ax.legend(loc=0, fontsize=12)

                        # Add a second x-axis on top of plot
                        # ax2 = ax.twiny()
                        # ax2.set_xlabel('Control Amplitude (a.u.)', fontsize=15)
                        # ax.plot(
                        #     self._param_sweep[q][0], self._sweep_results[q],
                        #     'o', c='blue'
                        # )
                        # ax2.tick_params(
                        #     axis='x', which='major', labelsize=12
                        # )
                        # # ax2.cla()

                    else:
                        ax.axis('off')
                
            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'calibration_results.png', 
                    dpi=300
                )
            plt.show()

        def final(self) -> None:
            """Final calibration method."""
            Calibration.final(self)
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return Amplitude(
        config,
        qubit_pairs,
        amplitudes,
        compiler, 
        transpiler,
        classifier,
        n_shots, 
        n_batches, 
        n_circs_per_seq,
        n_levels,
        n_gates,
        relative_amp,
        raster_circuits,
        **kwargs
    )


def Frequency(
        qpu:             QPU,
        config:          Config,
        qubit_pairs:     List[Tuple],
        frequencies:     ArrayLike | NDArray | Dict[ArrayLike | NDArray],
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        n_gates:         int = 1,
        relative_freq:   bool = False,
        raster_circuits: bool = False,
        **kwargs
    ) -> Callable:
    """Frequency calibration for CZ gate.

    This calibration finds the frequency where the conditionality is a maximum.

    Basic example useage for initial calibration:

    ```
    frequencies = np.linspace(5.0, 5.2, 21) * GHz
    cal = Frequency(
        CustomQPU, 
        config, 
        qubit_pairs=[(0, 1), (2, 3)],
        frequencies=frequencies)
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_pairs (List[Tuple]): pairs of qubit labels for the two-qubit gate
            calibration.
        frequencies (ArrayLike | NDArray | Dict[ArrayLike | NDArray]): array of
            frequencies to sweep over for calibrating the two-qubit CZ gate. 
            If calibrating multiple gates at the same time, this should be a 
            dictionary mapping an array to each qubit pair label.
        compiler (Any | Compiler | None, optional): custom compiler to
            compile the experimental circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to 
            transpile the experimental circuits. Defaults to None.
        classifier (ClassificationManager, optional): manager used for 
            classifying raw data. Defaults to None.
        n_shots (int, optional): number of measurements per circuit. 
            Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. 
            Defaults to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that
            can be measured per sequence. Defaults to 1.
        n_levels (int, optional): number of energy levels to be measured. 
            Defaults to 2. If n_levels = 3, this assumes that the
            measurement supports qutrit classification.
        n_gates (int, optional): number of gates for pulse repetition.
            Defaults to 1.
        relative_freq (bool, optional): whether or not the frequencies argument
            is defined relative to the existing pulse frequencies. Defaults to
            False. If True, the frequencies are swept over the current 
            frequency plus/minus the frequencies argument.
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.

    Returns:
        Callable: Frequency calibration class.
    """

    class Frequency(qpu, Calibration):
        """Frequency calibration class.
        
        This class inherits a custom QPU from the Frequency calibration
        function.
        """

        def __init__(self, 
                config:          Config,
                qubit_pairs:     List[Tuple],
                frequencies:     ArrayLike | Dict[ArrayLike],
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2,
                n_gates:         int = 1,
                relative_freq:   bool = False,
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            """Initialize the Amplitude class within the function."""
            qpu.__init__(self,
                config=config, 
                compiler=compiler, 
                transpiler=transpiler,
                classifier=classifier,
                n_shots=n_shots, 
                n_batches=n_batches, 
                n_circs_per_seq=n_circs_per_seq, 
                n_levels=n_levels,
                raster_circuits=raster_circuits,
                **kwargs
            )
            Calibration.__init__(self, config)

            self._qubits = qubit_pairs
            self._n_gates = n_gates

            self._params = {}

            for pair in qubit_pairs:
                self._params[pair] = f'two_qubit/{pair}/CZ/freq'         

            self._frequencies = {}
            for pair in qubit_pairs:
                if isinstance(frequencies, dict):
                    frequency = frequencies[pair]
                else:
                    frequency = frequencies

                if relative_freq:
                    self._frequencies[pair] = (
                        self._config[self._params[pair]] + frequency
                    )
                else:
                    self._frequencies[pair] = frequency
                self._param_sweep[pair] = self._frequencies[pair]
                
            self._R = {}
            self._fit = {
                pair: FitParabola() for pair in qubit_pairs
            }

        @property
        def frequencies(self) -> Dict:
            """Frequency sweep for each qubit pair.

            Returns:
                Dict: qubit to array map.
            """
            return self._frequencies
        
        @property
        def qubit_pairs(self) -> List[Tuple]:
            """Qubit pair labels.

            Returns:
                List[Tuple]: qubit pairs.
            """
            return self._qubits
        
        @property
        def conditionality(self) -> Dict[Tuple]:
            """Conditionality computed at each phase value.

            Returns:
                Dict[Tuple]: conditionality for each pair.
            """
            return self._R

        def generate_circuits(self) -> None:
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            self._circuits = tomography_circuits(
                self._qubits,
                self._frequencies[self._qubits[0]].size
            )

            for pair in self._qubits:
                self._circuits[f'param: {self._params[pair]}'] = list(
                    self._frequencies[pair]
                ) * 4

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            # Compute the conditionality and fit to a parabola
            qubits = list(set(chain.from_iterable(self._qubits)))
            for pair in self._qubits:
                i = qubits.index(pair[1])

                prob_C0_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_X'].circuit:
                    prob_C0_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C1_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_X'].circuit:
                    prob_C1_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C0_Y = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_Y'].circuit:
                    prob_C0_Y.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C1_Y = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_Y'].circuit:
                    prob_C1_Y.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                self._circuits[f'{pair}: Prob(0)'] = (
                    prob_C0_X + prob_C1_X + prob_C0_Y + prob_C1_Y
                )
                self._R[pair] = np.sqrt(
                    (np.array(prob_C1_X) - np.array(prob_C0_X)) ** 2 + 
                    (np.array(prob_C1_Y) - np.array(prob_C0_Y)) ** 2
                )
                self._sweep_results[pair] = self._R[pair]

                self._fit[pair].fit(
                    self._frequencies[pair][0], self._R[pair]
                )

                # If the fit was successful, find the new frequency
                if self._fit[pair].fit_success:
                    a, b, _ = self._fit[pair].fit_params
                    newvalue = -b / (2 * a)  # Assume c = 0
                    if a > 0:
                        logger.warning(
                            f'Fit failed for {pair} (positive curvature)!'
                        )
                        self._fit[pair]._fit_success = False
                    elif not in_range(newvalue, self._frequencies[pair]):
                        logger.warning(
                            f'Fit failed for {pair} (out of range)!'
                        )
                        self._fit[pair]._fit_success = False
                    else:
                        self._cal_values[pair].append(newvalue)

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CZ_Freq_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._sweep_results]), 'sweep_results'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._cal_values]), 'calibrated_values'
                )

        def plot(self) -> None:
            """Plot the amplitude sweep and fit results."""
            nrows, ncols = calculate_nrows_ncols(len(self._qubits))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            k = -1
            for i in range(nrows):
                for j in range(ncols):
                    k += 1

                    if len(self._qubits) == 1:
                        ax = axes
                    elif axes.ndim == 1:
                        ax = axes[j]
                    elif axes.ndim == 2:
                        ax = axes[i,j]

                    if k < len(self._qubits):
                        q = self._qubits[k]

                        ax.set_xlabel('Frequency (a.u.)', fontsize=15)
                        ax.set_ylabel('Conditionality', fontsize=15)
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )
                        ax.grid(True)

                        ax.plot(
                            self._param_sweep[q], self._sweep_results[q],
                            'o', c='blue', label=f'Meas, {q}'
                        )

                        if self._fit[q].fit_success:
                            x = np.linspace(
                                self._param_sweep[q][0],
                                self._param_sweep[q][-1], 
                                100
                            )
                            ax.plot(
                                x, self._fit[q].predict(x),
                                '-', c='orange', label='Fit'
                            )
                            ax.axvline(
                                self._cal_values[q],
                                ls='--', c='k', label='Fit value'
                            )

                        ax.legend(loc=0, fontsize=12)

                    else:
                        ax.axis('off')
                
            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'calibration_results.png', 
                    dpi=300
                )
            plt.show()

        def final(self) -> None:
            """Final calibration method."""
            Calibration.final(self)
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return Frequency(
        config,
        qubit_pairs,
        frequencies,
        compiler, 
        transpiler,
        classifier,
        n_shots, 
        n_batches, 
        n_circs_per_seq,
        n_levels,
        n_gates,
        relative_freq,
        raster_circuits,
        **kwargs
    )


def RelativeAmp(
        qpu:             QPU,
        config:          Config,
        qubit_pairs:     List[Tuple],
        amplitudes:      ArrayLike | NDArray | Dict[ArrayLike | NDArray],
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        raster_circuits: bool = False,
        **kwargs
    ) -> Callable:
    """Relative amplitude calibration for CZ gate.

    This calibration finds a relative amplitude between the drive lines where
    the conditionality is a maximum.

    Basic example useage for initial calibration:

    ```
    amplitudes = np.linspace(0.8, 1.2, 21)
    cal = RelativeAmp(
        CustomQPU, 
        config, 
        qubit_pairs=[(0, 1), (2, 3)],
        amplitudes=amplitudes)
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_pairs (List[Tuple]): pairs of qubit labels for the two-qubit gate
            calibration.
        amplitudes (ArrayLike | NDArray | Dict[ArrayLike | NDArray]): array of
            relative amps to sweep over for calibrating the two-qubit CZ gate. 
            These amplitudes are swept over the target qubit line and are 
            relative to the amplitude on the control qubit line. If calibrating 
            multiple gates at the same time, this should be a dictionary  
            mapping an array to a qubit pair label.
        compiler (Any | Compiler | None, optional): custom compiler to
            compile the experimental circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to 
            transpile the experimental circuits. Defaults to None.
        classifier (ClassificationManager, optional): manager used for 
            classifying raw data. Defaults to None.
        n_shots (int, optional): number of measurements per circuit. 
            Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. 
            Defaults to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that
            can be measured per sequence. Defaults to 1.
        n_levels (int, optional): number of energy levels to be measured. 
            Defaults to 2. If n_levels = 3, this assumes that the
            measurement supports qutrit classification.
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.

    Returns:
        Callable: RelativeAmp calibration class.
    """

    class RelativeAmp(qpu, Calibration):
        """Relative amplitude calibration class.
        
        This class inherits a custom QPU from the RelativeAmp calibration
        function.
        """

        def __init__(self, 
                config:          Config,
                qubit_pairs:     List[Tuple],
                amplitudes:      ArrayLike | Dict[ArrayLike],
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2, 
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            """Initialize the RelativePhase class within the function."""
            qpu.__init__(self,
                config=config, 
                compiler=compiler, 
                transpiler=transpiler,
                classifier=classifier,
                n_shots=n_shots, 
                n_batches=n_batches, 
                n_circs_per_seq=n_circs_per_seq, 
                n_levels=n_levels,
                raster_circuits=raster_circuits,
                **kwargs
            )
            Calibration.__init__(self)

            self._qubits = qubit_pairs

            self._params = {}
            for pair in qubit_pairs:
                idx = find_pulse_index(config, f'two_qubit/{pair}/CZ/pulse')
                self._params[pair] = (
                    f'two_qubit/{pair}/CZ/pulse/{idx+1}/kwargs/amp'
                )

            self._amplitudes = {}
            if not isinstance(amplitudes, dict):
                for pair in qubit_pairs:
                    idx = find_pulse_index(
                        config, f'two_qubit/{pair}/CZ/pulse'
                    )
                    self._amplitudes[pair] = (
                        config[f'two_qubit/{pair}/CZ/pulse/{idx}/kwargs/amp'] 
                        * amplitudes
                    )
            else:
                for pair, amps_ in amplitudes.items():
                    idx = find_pulse_index(
                        config, f'two_qubit/{pair}/CZ/pulse'
                    )
                    self._amplitudes[pair] = (
                        config[f'two_qubit/{pair}/CZ/pulse/{idx}/kwargs/amp'] 
                        * amps_
                    )
            self._param_sweep = self._amplitudes
                
            self._R = {}
            self._fit = {pair: FitParabola() for pair in qubit_pairs}

        @property
        def amplitudes(self) -> Dict:
            """Amp sweep for each qubit.

            Returns:
                Dict: qubit to array map.
            """
            return self._amplitudes
        
        @property
        def qubit_pairs(self) -> List[Tuple]:
            """Qubit pair labels.

            Returns:
                List[Tuple]: qubit pairs.
            """
            return self._qubits
        
        @property
        def conditionality(self) -> Dict[Tuple]:
            """Conditionality computed at each phase value.

            Returns:
                Dict[Tuple]: conditionality for each pair.
            """
            return self._R

        def generate_circuits(self) -> None:
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            self._circuits = tomography_circuits(
                self._qubits,
                self._amplitudes[self._qubits[0]].size
            )

            for pair in self._qubits:
                self._circuits[f'param: {self._params[pair]}'] = list(
                    self._amplitudes[pair]
                ) * 4

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            # Compute the conditionality and fit to a parabola
            qubits = list(set(chain.from_iterable(self._qubits)))
            for pair in self._qubits:
                i = qubits.index(pair[1])

                prob_C0_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_X'].circuit:
                    prob_C0_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C1_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_X'].circuit:
                    prob_C1_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C0_Y = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_Y'].circuit:
                    prob_C0_Y.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C1_Y = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_Y'].circuit:
                    prob_C1_Y.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                self._circuits[f'{pair}: Prob(0)'] = (
                    prob_C0_X + prob_C1_X + prob_C0_Y + prob_C1_Y
                )
                self._R[pair] = np.sqrt(
                    (np.array(prob_C1_X) - np.array(prob_C0_X)) ** 2 + 
                    (np.array(prob_C1_Y) - np.array(prob_C0_Y)) ** 2
                )
                self._sweep_results[pair] = self._R[pair]

                self._fit[pair].fit(self._amplitudes[pair], self._R[pair])
                # If the fit was successful, find the new phase
                if self._fit[pair].fit_success:
                    a, b, _ = self._fit[pair].fit_params
                    newvalue = -b / (2 * a)  # Assume c = 0
                    if a > 0:
                        logger.warning(
                            f'Fit failed for {pair} (positive curvature)!'
                        )
                        self._fit[pair]._fit_success = False
                    elif not in_range(newvalue, self._amplitudes[pair]):
                        logger.warning(
                            f'Fit failed for {pair} (out of range)!'
                        )
                        self._fit[pair]._fit_success = False
                    else:
                        self._cal_values[pair] = newvalue

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CZ_RelAmp_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._sweep_results]), 'sweep_results'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._cal_values]), 'calibrated_values'
                )

        def plot(self):
            """Plot the sweep results."""
            Calibration.plot(self,
                xlabel='Amp. (a.u.)',
                ylabel='Conditionality',
                save_path=self._data_manager._save_path
            )

        def final(self) -> None:
            """Final calibration method."""
            Calibration.final(self)
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return RelativeAmp(
        config,
        qubit_pairs,
        amplitudes,
        compiler, 
        transpiler,
        classifier,
        n_shots, 
        n_batches, 
        n_circs_per_seq,
        n_levels,
        raster_circuits,
        **kwargs
    )


def RelativePhase(
        qpu:             QPU,
        config:          Config,
        qubit_pairs:     List[Tuple],
        phases:          ArrayLike | NDArray | Dict[ArrayLike | NDArray],
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        raster_circuits: bool = False,
        **kwargs
    ) -> Callable:
    """Relative phase calibration for CZ gate.

    This calibration finds a relative phase between the drive lines where the
    conditionality is a maximum.

    Basic example useage for initial calibration:

    ```
    phases = np.pi * np.linspace(-0.5, 0.5, 21)
    cal = RelativePhase(
        CustomQPU, 
        config, 
        qubit_pairs=[(0, 1), (2, 3)],
        phases=phases)
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_pairs (List[Tuple]): pairs of qubit labels for the two-qubit gate
            calibration.
        phases (ArrayLike | NDArray | Dict[ArrayLike | NDArray]): array of
            phases to sweep over for calibrating the two-qubit gate 
            phases. If calibrating multiple gates at the same time, this 
            should be a dictionary mapping an array to a qubit pair label.
        compiler (Any | Compiler | None, optional): custom compiler to
            compile the experimental circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to 
            transpile the experimental circuits. Defaults to None.
        classifier (ClassificationManager, optional): manager used for 
            classifying raw data. Defaults to None.
        n_shots (int, optional): number of measurements per circuit. 
            Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. 
            Defaults to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that
            can be measured per sequence. Defaults to 1.
        n_levels (int, optional): number of energy levels to be measured. 
            Defaults to 2. If n_levels = 3, this assumes that the
            measurement supports qutrit classification.
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.

    Returns:
        Callable: RelativePhase calibration calss.
    """

    class RelativePhase(qpu, Calibration):
        """Relative phase calibration class.
        
        This class inherits a custom QPU from the RelativePhase calibration
        function.
        """

        def __init__(self, 
                config:          Config,
                qubit_pairs:     List[Tuple],
                phases:          ArrayLike | Dict[ArrayLike],
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2, 
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            """Initialize the RelativePhase class within the function."""
            qpu.__init__(self,
                config=config, 
                compiler=compiler, 
                transpiler=transpiler,
                classifier=classifier,
                n_shots=n_shots, 
                n_batches=n_batches, 
                n_circs_per_seq=n_circs_per_seq, 
                n_levels=n_levels,
                raster_circuits=raster_circuits,
                **kwargs
            )
            Calibration.__init__(self, config)

            self._qubits = qubit_pairs

            if not isinstance(phases, dict):
                self._phases = {pair: phases for pair in qubit_pairs}
            else:
                self._phases = phases
            self._param_sweep = self._phases

            self._params = {}
            for pair in qubit_pairs:
                idx = find_pulse_index(config, f'two_qubit/{pair}/CZ/pulse')
                self._params[pair] = (
                    f'two_qubit/{pair}/CZ/pulse/{idx+1}/kwargs/phase'
                )
                
            self._R = {}
            self._fit = {pair: FitParabola() for pair in qubit_pairs}

        @property
        def phases(self) -> Dict:
            """Phase sweep for each qubit.

            Returns:
                Dict: qubit to array map.
            """
            return self._phases
        
        @property
        def qubit_pairs(self) -> List[Tuple]:
            """Qubit pair labels.

            Returns:
                List[Tuple]: qubit pairs.
            """
            return self._qubits
        
        @property
        def conditionality(self) -> Dict[Tuple]:
            """Conditionality computed at each phase value.

            Returns:
                Dict[Tuple]: conditionality for each pair.
            """
            return self._R

        def generate_circuits(self) -> None:
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            self._circuits = tomography_circuits(
                self._qubits,
                self._phases[self._qubits[0]].size
            )

            for pair in self._qubits:
                self._circuits[f'param: {self._params[pair]}'] = list(
                    self._phases[pair]
                ) * 4

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            # Compute the conditionality and fit to a parabola
            qubits = list(set(chain.from_iterable(self._qubits)))
            for pair in self._qubits:
                i = qubits.index(pair[1])

                prob_C0_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_X'].circuit:
                    prob_C0_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C1_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_X'].circuit:
                    prob_C1_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C0_Y = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_Y'].circuit:
                    prob_C0_Y.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C1_Y = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_Y'].circuit:
                    prob_C1_Y.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                self._circuits[f'{pair}: Prob(0)'] = (
                    prob_C0_X + prob_C1_X + prob_C0_Y + prob_C1_Y
                )
                self._R[pair] = np.sqrt(
                    (np.array(prob_C1_X) - np.array(prob_C0_X)) ** 2 + 
                    (np.array(prob_C1_Y) - np.array(prob_C0_Y)) ** 2
                )
                self._sweep_results[pair] = self._R[pair]

                self._fit[pair].fit(self._phases[pair], self._R[pair])
                # If the fit was successful, write find the new phase
                if self._fit[pair].fit_success:
                    a, b, _ = self._fit[pair].fit_params
                    newvalue = -b / (2 * a)  # Assume c = 0
                    if a > 0:
                        logger.warning(
                            f'Fit failed for {pair} (positive curvature)!'
                        )
                        self._fit[pair]._fit_success = False
                    elif not in_range(newvalue, self._phases[pair]):
                        logger.warning(
                            f'Fit failed for {pair} (out of range)!'
                        )
                        self._fit[pair]._fit_success = False
                    else:
                        self._cal_values[pair] = newvalue

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CZ_Phase_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._sweep_results]), 'sweep_results'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._cal_values]), 'calibrated_values'
                )

        def plot(self):
            """Plot the sweep results."""
            Calibration.plot(self,
                xlabel='Phase (rad.)',
                ylabel='Conditionality',
                save_path=self._data_manager._save_path
            )

        def final(self) -> None:
            """Final calibration method."""
            Calibration.final(self)
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return RelativePhase(
        config,
        qubit_pairs,
        phases,
        compiler, 
        transpiler,
        classifier,
        n_shots, 
        n_batches, 
        n_circs_per_seq,
        n_levels, 
        raster_circuits,
        **kwargs
    )


def LocalPhases(
        qpu:             QPU,
        config:          Config,
        qubit_pairs:     List[Tuple],
        phases:          NDArray | Dict = np.pi * np.linspace(-1, 1, 21),
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        raster_circuits: bool = False,
        **kwargs
    ) -> Callable:
    """Relative phase calibration for CZ gate.

    This calibration finds a relative phase between the drive lines where the
    conditionality is a maximum.

    Basic example useage for initial calibration:

    ```
    cal = LocalPhases(
        CustomQPU, 
        config, 
        qubit_pairs=[(0, 1), (2, 3)])
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_pairs (List[Tuple]): pairs of qubit labels for the two-qubit gate
            calibration.
        phases (NDArray | Dict, optional): array of phases to sweep over for 
            calibrating the local phases of the gate. If calibrating multiple 
            gates at the same time, this should be a dictionary mapping an 
            array to a qubit pair label.
        compiler (Any | Compiler | None, optional): custom compiler to
            compile the experimental circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to 
            transpile the experimental circuits. Defaults to None.
        classifier (ClassificationManager, optional): manager used for 
            classifying raw data. Defaults to None.
        n_shots (int, optional): number of measurements per circuit. 
            Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. 
            Defaults to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that
            can be measured per sequence. Defaults to 1.
        n_levels (int, optional): number of energy levels to be measured. 
            Defaults to 2. If n_levels = 3, this assumes that the
            measurement supports qutrit classification.
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.

    Returns:
        Callable: LocalPhases calibration calss.
    """

    class LocalPhases(qpu, Calibration):
        """Local phase calibration class.
        
        This class inherits a custom QPU from the LocalPhase calibration
        function.
        """

        def __init__(self, 
                config:          Config,
                qubit_pairs:     List[Tuple],
                phases:          NDArray | Dict = np.pi*np.linspace(-1, 1, 21),
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2, 
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            """Initialize the RelativePhase class within the function."""
            qpu.__init__(self,
                config=config, 
                compiler=compiler, 
                transpiler=transpiler,
                classifier=classifier,
                n_shots=n_shots, 
                n_batches=n_batches, 
                n_circs_per_seq=n_circs_per_seq, 
                n_levels=n_levels,
                raster_circuits=raster_circuits,
                **kwargs
            )
            Calibration.__init__(self, config)

            self._qubits = qubit_pairs

            if not isinstance(phases, dict):
                self._phases = {pair: phases for pair in qubit_pairs}
            else:
                self._phases = phases
            self._param_sweep = self._phases

            self._params = {}
            for pair in qubit_pairs:
                idx = find_pulse_index(config, f'two_qubit/{pair}/CZ/pulse')
                self._params[pair] = (
                    f'two_qubit/{pair}/CZ/pulse/{idx+2}/kwargs/phase',
                    f'two_qubit/{pair}/CZ/pulse/{idx+3}/kwargs/phase'
                )
                
            self._fit = {
                pair: {
                    'X_C0': FitCosine(), 
                    'X_C1': FitCosine(), 
                    'X_T0': FitCosine(), 
                    'X_T1': FitCosine()
                } for pair in qubit_pairs
            }
            
            self._phase_C0 = {}
            self._phase_C1 = {}
            self._phase_T0 = {}
            self._phase_T1 = {}
            self._cal_values = {pair: [0., 0.] for pair in qubit_pairs}

        @property
        def phases(self) -> Dict:
            """Phase sweep for each qubit.

            Returns:
                Dict: qubit to array map.
            """
            return self._phases
        
        @property
        def qubit_pairs(self) -> List[Tuple]:
            """Qubit pair labels.

            Returns:
                List[Tuple]: qubit pairs.
            """
            return self._qubits
        
        @property
        def conditionality(self) -> Dict[Tuple]:
            """Conditionality computed at each phase value.

            Returns:
                Dict[Tuple]: conditionality for each pair.
            """
            return self._R

        def generate_circuits(self) -> None:
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')
            qubits = list(set(chain.from_iterable(self._qubits)))

            circuit_C0_X = Circuit([
                # Y90 on target qubit
                Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
                Cycle({X90(p[1]) for p in qubit_pairs}),
                Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
                # CZ
                Barrier(qubits),
                Cycle({CZ(pair) for pair in qubit_pairs}),
                Barrier(qubits),
                # Y-90 on target qubit
                Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
                Cycle({X90(p[1]) for p in qubit_pairs}),
                Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
            ])
            circuit_C0_X.measure()

            circuit_C1_X = Circuit([
                # X on control qubit
                Cycle({X90(p[0]) for p in qubit_pairs}),
                Cycle({X90(p[0]) for p in qubit_pairs}),
                Barrier(qubits),
                # Y90 on target qubit
                Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
                Cycle({X90(p[1]) for p in qubit_pairs}),
                Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
                # CZ
                Barrier(qubits),
                Cycle({CZ(pair) for pair in qubit_pairs}),
                Barrier(qubits),
                # Y-90 on target qubit
                Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
                Cycle({X90(p[1]) for p in qubit_pairs}),
                Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
            ])
            circuit_C1_X.measure()

            circuit_T0_X = Circuit([
                # Y90 on control qubit
                Cycle({VirtualZ(np.pi/2, p[0]) for p in qubit_pairs}),
                Cycle({X90(p[0]) for p in qubit_pairs}),
                Cycle({VirtualZ(-np.pi/2, p[0]) for p in qubit_pairs}),
                # CZ
                Barrier(qubits),
                Cycle({CZ(pair) for pair in qubit_pairs}),
                Barrier(qubits),
                # Y-90 on control qubit
                Cycle({VirtualZ(-np.pi/2, p[0]) for p in qubit_pairs}),
                Cycle({X90(p[0]) for p in qubit_pairs}),
                Cycle({VirtualZ(np.pi/2, p[0]) for p in qubit_pairs}),
                # Measure
                Cycle(Meas(q) for q in qubits)
            ])
            circuit_T0_X.measure()

            circuit_T1_X = Circuit([
                # X on target qubit
                Cycle({X90(p[1]) for p in qubit_pairs}),
                Cycle({X90(p[1]) for p in qubit_pairs}),
                Barrier(qubits),
                # Y90 on control qubit
                Cycle({VirtualZ(np.pi/2, p[0]) for p in qubit_pairs}),
                Cycle({X90(p[0]) for p in qubit_pairs}),
                Cycle({VirtualZ(-np.pi/2, p[0]) for p in qubit_pairs}),
                # CZ
                Barrier(qubits),
                Cycle({CZ(pair) for pair in qubit_pairs}),
                Barrier(qubits),
                # Y-90 on control qubit
                Cycle({VirtualZ(-np.pi/2, p[0]) for p in qubit_pairs}),
                Cycle({X90(p[0]) for p in qubit_pairs}),
                Cycle({VirtualZ(np.pi/2, p[0]) for p in qubit_pairs}),
            ])
            circuit_T1_X.measure()

            n_elements = self._phases[self._qubits[0]].size
            circuits = list()
            circuits.extend([circuit_C0_X.copy() for _ in range(n_elements)])
            circuits.extend([circuit_C1_X.copy() for _ in range(n_elements)])
            circuits.extend([circuit_T0_X.copy() for _ in range(n_elements)])
            circuits.extend([circuit_T1_X.copy() for _ in range(n_elements)])
            
            self._circuits = CircuitSet(circuits=circuits)
            self._circuits['sequence'] = (
                ['C0_X'] * n_elements + ['C1_X'] * n_elements +
                ['T0_X'] * n_elements + ['T1_X'] * n_elements 
            )

            for pair in self._qubits:
                self._circuits[f'param: {self._params[pair][0]}'] = (
                    [0.0] * 2 * self._phases[pair].size + 
                    list(self._phases[pair]) * 2
                )
                self._circuits[f'param: {self._params[pair][1]}'] = (
                    list(self._phases[pair]) * 2 + 
                    [0.0] * 2 * self._phases[pair].size
                )                

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            # Compute the conditionality and fit to a parabola
            qubits = list(set(chain.from_iterable(self._qubits)))
            for pair in self._qubits:
                c = qubits.index(pair[0])
                t = qubits.index(pair[1])

                prob_C0_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_X'].circuit:
                    prob_C0_X.append(
                        circuit.results.marginalize(t).populations['0']
                    )

                prob_C1_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_X'].circuit:
                    prob_C1_X.append(
                        circuit.results.marginalize(t).populations['0']
                    )

                prob_T0_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'T0_X'].circuit:
                    prob_T0_X.append(
                        circuit.results.marginalize(c).populations['0']
                    )

                prob_T1_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'T1_X'].circuit:
                    prob_T1_X.append(
                        circuit.results.marginalize(c).populations['0']
                    )

                self._circuits[f'{pair}: Prob(0)'] = (
                    prob_C0_X + prob_C1_X + prob_T0_X + prob_T1_X
                )
                
                contrast_C0 = max(prob_C0_X) - min(prob_C0_X)
                contrast_C1 = max(prob_C1_X) - min(prob_C1_X)
                contrast_T0 = max(prob_T0_X) - min(prob_T0_X)
                contrast_T1 = max(prob_T1_X) - min(prob_T1_X)

                X_C0 = 2 * (
                    np.array(prob_C0_X) - min(prob_C0_X)
                ) / contrast_C0 - 1
                X_C1 = 2 * (
                    np.array(prob_C1_X) - min(prob_C1_X)
                ) / contrast_C1 - 1
                X_T0 = 2 * (
                    np.array(prob_T0_X) - min(prob_T0_X)
                ) / contrast_T0 - 1
                X_T1 = 2 * (
                    np.array(prob_T1_X) - min(prob_T1_X)
                ) / contrast_T1 - 1

                self._sweep_results[pair] = {
                    'X_C0': X_C0, 'X_C1': X_C1, 'X_T0': X_T0, 'X_T1': X_T1, 
                }

                self._fit[pair]['X_C0'].fit(
                    self._phases[pair], X_C0, p0=(1, 1/(2*np.pi), 0, 0),
                    bounds=(
                        [0., 0., -np.pi, -0.1], 
                        [1., np.inf, np.pi, 0.1]
                    )
                )
                self._fit[pair]['X_C1'].fit(
                    self._phases[pair], X_C1, p0=(-1, 1/(2*np.pi), 0, 0),
                    bounds=(
                        [-1., 0., -np.pi, -0.1], 
                        [0., np.inf, np.pi, 0.1]
                    )
                )
                self._fit[pair]['X_T0'].fit(
                    self._phases[pair], X_T0, p0=(1, 1/(2*np.pi), 0, 0),
                    bounds=(
                        [0., 0., -np.pi, -0.1], 
                        [1., np.inf, np.pi, 0.1]
                    )
                )
                self._fit[pair]['X_T1'].fit(
                    self._phases[pair], X_T1, p0=(-1, 1/(2*np.pi), 0, 0),
                    bounds=(
                        [-1., 0., -np.pi, -0.1], 
                        [0., np.inf, np.pi, 0.1]
                    )
                )

                if (self._fit[pair]['X_C0'].fit_success and
                    self._fit[pair]['X_C1'].fit_success):
                    _, freq, phase, _ = self._fit[pair]['X_C0'].fit_params
                    self._phase_C0[pair] = wrap_phase(
                        -phase / (2 * np.pi * freq)
                    )

                    _, freq, phase, _ = self._fit[pair]['X_C1'].fit_params
                    self._phase_C1[pair] = wrap_phase(
                        -phase / (2 * np.pi * freq)
                    )

                    newvalue = np.mean([
                        self._phase_C0[pair], self._phase_C1[pair]
                    ])
                    self._cal_values[pair][1] = newvalue  # Target qubit phase

                if (self._fit[pair]['X_T0'].fit_success and
                    self._fit[pair]['X_T1'].fit_success):
                    _, freq, phase, _ = self._fit[pair]['X_T0'].fit_params
                    self._phase_T0[pair] = wrap_phase(
                        -phase / (2 * np.pi * freq)
                    )

                    _, freq, phase, _ = self._fit[pair]['X_T1'].fit_params
                    self._phase_T1[pair] = wrap_phase(
                        -phase / (2 * np.pi * freq)
                    )

                    newvalue = np.mean([
                        self._phase_T0[pair], self._phase_T1[pair]
                    ])
                    self._cal_values[pair][0] = newvalue  # Control qubit phase

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CZ_LocalPhases_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._sweep_results]), 'sweep_results'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._cal_values]), 'calibrated_values'
                )

        def plot(self) -> None:
            """Plot the frequency sweep and fit results."""
            nrows = len(self._qubits)
            figsize = (12, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, 2, figsize=figsize, layout='constrained'
            )

            for i, pair in enumerate(self._qubits):
                if len(self._qubits) == 1:
                    ax = axes
                else:
                    ax = axes[i]

                ax[0].set_xlabel(f'Phase Q{pair[0]} (rad.)', fontsize=15)
                ax[1].set_xlabel(f'Phase Q{pair[1]} (rad.)', fontsize=15)
                ax[0].set_ylabel(r'$\langle X \rangle$', fontsize=15)
                ax[1].set_ylabel(r'$\langle X \rangle$', fontsize=15)
                ax[0].tick_params(axis='both', which='major', labelsize=12)
                ax[1].tick_params(axis='both', which='major', labelsize=12)
                ax[0].grid(True)
                ax[1].grid(True)

                # Control phase sweep
                ax[0].plot(
                    self._phases[pair], self._sweep_results[pair]['X_T0'], 'bo'
                )
                ax[0].plot(
                   self._phases[pair], self._sweep_results[pair]['X_T1'], 'ro'
                )
                if self._fit[pair]['X_T0'].fit_success:
                    x = np.linspace(
                        self._phases[pair][0], self._phases[pair][-1], 100
                    )
                    ax[0].plot(x, self._fit[pair]['X_T0'].predict(x), 'b-')
                    ax[0].axvline(self._phase_T0[pair], ls='-', c='b',
                        label=(
                         rf'Q{pair[1]} $|0\rangle$: '
                          f'{round(self._phase_T0[pair], 2)}'
                        )
                    )
                if self._fit[pair]['X_T1'].fit_success:
                    x = np.linspace(
                        self._phases[pair][0], self._phases[pair][-1], 100
                    )
                    ax[0].plot(x, self._fit[pair]['X_T1'].predict(x), 'r-')
                    ax[0].axvline(self._phase_T1[pair], ls='-', c='r',
                        label=(
                         rf'Q{pair[1]} $|1\rangle$: '
                          f'{round(self._phase_T1[pair], 2)}'
                        )
                    )

                # Target phase sweep
                ax[1].plot(
                    self._phases[pair], self._sweep_results[pair]['X_C0'], 'bo'
                )
                ax[1].plot(
                    self._phases[pair], self._sweep_results[pair]['X_C1'], 'ro'
                )
                if self._fit[pair]['X_C0'].fit_success:
                    x = np.linspace(
                        self._phases[pair][0], self._phases[pair][-1], 100
                    )
                    ax[1].plot(x, self._fit[pair]['X_C0'].predict(x), 'b-')
                    ax[1].axvline(self._phase_C0[pair], ls='-', c='b',
                        label=(
                         rf'Q{pair[0]} $|0\rangle$: '
                          f'{round(self._phase_C0[pair], 2)}'
                        )
                    )
                if self._fit[pair]['X_C1'].fit_success:
                    x = np.linspace(
                        self._phases[pair][0], self._phases[pair][-1], 100
                    )
                    ax[1].plot(x, self._fit[pair]['X_C1'].predict(x), 'r-')
                    ax[1].axvline(self._phase_C1[pair], ls='-', c='r', 
                        label=(
                            rf'Q{pair[0]} $|1\rangle$: '
                             f'{round(self._phase_C1[pair], 2)}'
                        )
                    )

                ax[0].legend(loc=0, fontsize=12)
                ax[1].legend(loc=0, fontsize=12)

            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'freq_calibration.png', 
                    dpi=300
                )
            plt.show()

        def final(self) -> None:
            """Final calibration method."""
            Calibration.final(self)
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return LocalPhases(
        config,
        qubit_pairs,
        phases,
        compiler, 
        transpiler,
        classifier,
        n_shots, 
        n_batches, 
        n_circs_per_seq,
        n_levels, 
        raster_circuits,
        **kwargs
    )