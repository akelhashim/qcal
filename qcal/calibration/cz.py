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
# from qcal.units import MHz, us

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from collections.abc import Iterable
from IPython.display import clear_output
from itertools import chain
from typing import Any, Callable, Dict, List, Tuple
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


def tomography_circuits(
        qubit_pairs: List[Tuple], n_elements: int
    ) -> CircuitSet:
    """Partial tomography circuits for CZ sweeps.

    Args:
        qubit_pairs (List[Tuple]): list of qubit pairs.
        n_elements (int):          size of parameter sweep.

    Returns:
        CircuitSet: partial tomography CZ circuits.
    """
    qubits = list(set(chain.from_iterable(qubit_pairs)))

    circuit_C0_X = Circuit([
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        # CZ
        Barrier(qubits),
        Cycle({CZ(pair) for pair in qubit_pairs}),
        Barrier(qubits),
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        # Measure
        Cycle(Meas(q) for q in qubits)
    ])

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
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
    ])

    circuit_C0_Y = Circuit([
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        # CZ
        Barrier(qubits),
        Cycle({CZ(pair) for pair in qubit_pairs}),
        Barrier(qubits),
        # X90 on target qubit
        Cycle({X90(p[1]) for p in qubit_pairs}),
    ])

    circuit_C1_Y = Circuit([
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
        # X90 on target qubit
        Cycle({X90(p[1]) for p in qubit_pairs}),
    ])

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
        relative_amp:    bool = False,
        esp:             bool = False,
        heralding:       bool = True,
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
        relative_amp (bool, optional): whether or not the amplitudes argument
            is defined relative to the existing pulse amplitude. Defaults to
            False. If True, the amplitudes are swept over the current amplitude
            times the amplitudes argument.
        esp (bool, optional): whether to enable excited state promotion for 
            the calibration. Defaults to False.
        heralding (bool, optional): whether to enable heralding for the 
            calibraion. Defaults to True.

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
                relative_amp:    bool = False,
                esp:             bool = False,
                heralding:       bool = True,
                **kwargs
            ) -> None:
            """Initialize the Amplitude class within the function."""
            qpu.__init__(self,
                config, 
                compiler, 
                transpiler,
                classifier,
                n_shots, 
                n_batches, 
                n_circs_per_seq, 
                n_levels,
                **kwargs
            )
            Calibration.__init__(self, 
                config, 
                esp=esp,
                heralding=heralding
            )

            self._qubits = qubit_pairs

            self._params = {}
            for pair in qubit_pairs:
                idx = find_pulse_index(config, f'two_qubit/{pair}/CZ/pulse')
                self._params[pair] = (
                    f'two_qubit/{pair}/CZ/pulse/{idx}/kwargs/amp',
                    f'two_qubit/{pair}/CZ/pulse/{idx+1}/kwargs/amp'
                )

                if not isinstance(amplitudes, dict):
                    if relative_amp:
                        self._amplitudes = {pair: (
                        config[f'two_qubit/{pair}/CZ/pulse/{idx}/kwargs/amp']
                         * amplitudes,
                        config[f'two_qubit/{pair}/CZ/pulse/{idx+1}/kwargs/amp'] 
                         * amplitudes
                        )
                    }
                    else:
                        self._amplitudes = {pair: (amplitudes, amplitudes)
                    }
                else:
                    self._amplitudes = amplitudes
            self._param_sweep = self._amplitudes
                
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
                            self._cal_values[pair][j] = newvalue

        def save(self):
            """Save all circuits and data."""
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
                        ax2 = ax.twiny()
                        ax2.set_xlabel('Control Amplitude (a.u.)', fontsize=15)
                        ax.plot(
                            self._param_sweep[q][0], self._sweep_results[q],
                            'o', c='blue'
                        )
                        ax2.cla()

                    else:
                        ax.axis('off')
                
            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'calibration_results.png', 
                    dpi=300
                )
            plt.show()

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CZ_Amp_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                self.save()
            self.plot()
            self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

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
        relative_amp,
        esp,
        heralding,
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
        esp:             bool = False,
        heralding:       bool = True,
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
        esp (bool, optional): whether to enable excited state promotion for 
            the calibration. Defaults to False.
        heralding (bool, optional): whether to enable heralding for the 
            calibraion. Defaults to True.

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
                esp:             bool = False,
                heralding:       bool = True,
                **kwargs
            ) -> None:
            """Initialize the RelativePhase class within the function."""
            qpu.__init__(self,
                config, 
                compiler, 
                transpiler,
                classifier,
                n_shots, 
                n_batches, 
                n_circs_per_seq, 
                n_levels,
                **kwargs
            )
            Calibration.__init__(self, 
                config, 
                esp=esp,
                heralding=heralding
            )

            self._qubits = qubit_pairs

            self._params = {}
            self._amplitudes = {}
            for pair in qubit_pairs:
                idx = find_pulse_index(config, f'two_qubit/{pair}/CZ/pulse')
                self._params[pair] = (
                    f'two_qubit/{pair}/CZ/pulse/{idx+1}/kwargs/amp'
                )

                if not isinstance(amplitudes, dict):
                    self._amplitudes = {
                        pair: (
                          config[f'two_qubit/{pair}/CZ/pulse/{idx}/kwargs/amp'] 
                          * amplitudes
                        )
                    }
                else:
                    for pair, amps_ in amplitudes.items():
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
            qpu.save(self)
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._sweep_results]), 'sweep_results'
            )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._cal_values]), 'calibrated_values'
            )

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CZ_RelAmp_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                self.save()
            self.plot(
                xlabel='Phase (rad.)',
                ylabel='Conditionality',
                save_path=self._data_manager._save_path
            )
            self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

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
        esp,
        heralding,
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
        esp:             bool = False,
        heralding:       bool = True,
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
        esp (bool, optional): whether to enable excited state promotion for 
            the calibration. Defaults to False.
        heralding (bool, optional): whether to enable heralding for the 
            calibraion. Defaults to True.

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
                esp:             bool = False,
                heralding:       bool = True,
                **kwargs
            ) -> None:
            """Initialize the RelativePhase class within the function."""
            qpu.__init__(self,
                config, 
                compiler, 
                transpiler,
                classifier,
                n_shots, 
                n_batches, 
                n_circs_per_seq, 
                n_levels,
                **kwargs
            )
            Calibration.__init__(self, 
                config, 
                esp=esp,
                heralding=heralding
            )

            self._qubits = qubit_pairs
            # self._qubits = list(set(chain.from_iterable(qubit_pairs)))

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
            qpu.save(self)
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._sweep_results]), 'sweep_results'
            )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._cal_values]), 'calibrated_values'
            )

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CZ_Phase_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                self.save()
            self.plot(
                xlabel='Phase (rad.)',
                ylabel='Conditionality',
                save_path=self._data_manager._save_path
            )
            self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

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
        esp,
        heralding,
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
        esp:             bool = False,
        heralding:       bool = True,
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
        esp (bool, optional): whether to enable excited state promotion for 
            the calibration. Defaults to False.
        heralding (bool, optional): whether to enable heralding for the 
            calibraion. Defaults to True.

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
                esp:             bool = False,
                heralding:       bool = True,
                **kwargs
            ) -> None:
            """Initialize the RelativePhase class within the function."""
            qpu.__init__(self,
                config, 
                compiler, 
                transpiler,
                classifier,
                n_shots, 
                n_batches, 
                n_circs_per_seq, 
                n_levels,
                **kwargs
            )
            Calibration.__init__(self, 
                config, 
                esp=esp,
                heralding=heralding
            )

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
            
            self._phase_C0 = None
            self._phase_C1 = None
            self._phase_T0 = None
            self._phase_T1 = None
            self._cal_values = {pair: tuple() for pair in qubit_pairs}

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
                # Measure
                Cycle(Meas(q) for q in qubits)
            ])

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
                self._circuits[f'param: {self._params[pair]}'] = list(
                    self._phases[pair]
                ) * 4

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            # Compute the conditionality and fit to a parabola
            qubits = list(set(chain.from_iterable(self._qubits)))
            for pair in self._qubits:
                i = qubits.index(pair[0])

                prob_C0_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_X'].circuit:
                    prob_C0_X.append(
                        circuit.results.marginalize(i+1).populations['0']
                    )

                prob_C1_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_X'].circuit:
                    prob_C1_X.append(
                        circuit.results.marginalize(i+1).populations['0']
                    )

                prob_T0_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'T0_X'].circuit:
                    prob_T0_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_T1_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'T1_X'].circuit:
                    prob_T1_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                self._circuits[f'{pair}: Prob(0)'] = (
                    prob_C0_X + prob_C1_X + prob_T0_X + prob_T1_X
                )
                
                contrast_C0 = max(prob_C0_X) - min(prob_C0_X)
                contrast_C1 = max(prob_C1_X) - min(prob_C1_X)
                contrast_T0 = max(prob_T0_X) - min(prob_T0_X)
                contrast_T1 = max(prob_T1_X) - min(prob_T1_X)

                X_C0 = 2 * (prob_C0_X - min(prob_C0_X)) / contrast_C0 - 1
                X_C1 = 2 * (prob_C1_X - min(prob_C1_X)) / contrast_C1 - 1
                X_T0 = 2 * (prob_T0_X - min(prob_T0_X)) / contrast_T0 - 1
                X_T1 = 2 * (prob_T1_X - min(prob_T1_X)) / contrast_T1 - 1

                self._sweep_results[pair] = {
                    'X_C0': X_C0, 'X_C1': X_C1, 'X_T0': X_T0, 'X_T1': X_T1, 
                }

                self._fit[pair]['X_C0'].fit(self._phases[pair], X_C0)
                self._fit[pair]['X_C1'].fit(self._phases[pair], -X_C1)
                self._fit[pair]['X_T0'].fit(self._phases[pair], X_T0)
                self._fit[pair]['X_T1'].fit(self._phases[pair], -X_T1)

                if (self._fit[pair]['X_C0'].fit_success and
                    self._fit[pair]['X_C1'].fit_success):
                    _, freq, phase, _ = self._fit[pair]['X_C0'].fit_params
                    self._phase_C0 = wrap_phase(-phase / (2 * np.pi * freq))

                    _, freq, phase, _ = self._fit[pair]['X_C1'].fit_params
                    self._phase_C1 = wrap_phase(-phase / (2 * np.pi * freq))

                    newvalue = np.mean([self._phase_C0, self._phase_C1])
                    self._cal_values[pair][1] = newvalue  # Target qubit phase

                if (self._fit[pair]['X_T0'].fit_success and
                    self._fit[pair]['X_T1'].fit_success):
                    _, freq, phase, _ = self._fit[pair]['X_T0'].fit_params
                    self._phase_T0 = wrap_phase(-phase / (2 * np.pi * freq))

                    _, freq, phase, _ = self._fit[pair]['X_T1'].fit_params
                    self._phase_T1 = wrap_phase(-phase / (2 * np.pi * freq))

                    newvalue = np.mean([self._phase_T0, self._phase_T1])
                    self._cal_values[pair][0] = newvalue  # Control qubit phase

        def save(self):
            """Save all circuits and data."""
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
            fig, ax = plt.subplots(
                nrows, 2, figsize=figsize, layout='constrained'
            )

            for i, pair in enumerate(self._qubits):
                if len(self._qubits) == 1:
                    ax = ax
                else:
                    ax = ax[i]

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
                    self._phases[pair], self._sweep_results['X_T0'], 'bo'
                )
                ax[0].plot(
                    self._phases[pair], self._sweep_results['X_T1'], 'ro'
                )
                if self._fit[pair]['X_T0'].fit_success:
                    x = np.linspace(
                        self._phases[pair][0], self._phases[pair][-1], 100
                    )
                    ax[0].plot(x, self._fit[pair]['X_T0'].predict(x), 'b-')
                    ax[0].axvline(self._phase_T0, ls='-', c='b',
                        label=(
                         rf'Q{pair[1]} $|0\rangle$: {round(self._phase_T0, 2)}'
                        )
                    )
                if self._fit[pair]['X_T1'].fit_success:
                    x = np.linspace(
                        self._phases[pair][0], self._phases[pair][-1], 100
                    )
                    ax[0].plot(x, -self._fit[pair]['X_T1'].predict(x), 'r-')
                    ax[0].axvline(self._phase_T1, ls='-', c='b',
                        label=(
                         rf'Q{pair[1]} $|1\rangle$: {round(self._phase_T1, 2)}'
                        )
                    )

                # Target phase sweep
                ax[1].plot(
                    self._phases[pair], self._sweep_results['X_C0'], 'bo'
                )
                ax[1].plot(
                    self._phases[pair], self._sweep_results['X_C1'], 'ro'
                )
                if self._fit[pair]['X_C0'].fit_success:
                    x = np.linspace(
                        self._phases[pair][0], self._phases[pair][-1], 100
                    )
                    ax[1].plot(x, self._fit[pair]['X_C0'].predict(x), 'b-')
                    ax[1].axvline(self._phase_C0, ls='-', c='b',
                        label=(
                         rf'Q{pair[0]} $|0\rangle$: {round(self._phase_C0, 2)}'
                        )
                    )
                if self._fit[pair]['X_C1'].fit_success:
                    x = np.linspace(
                        self._phases[pair][0], self._phases[pair][-1], 100
                    )
                    ax[1].plot(x, -self._fit[pair]['X_C1'].predict(x), 'r-')
                    ax[1].axvline(self._phase_C1, ls='-', c='r', 
                        label=(
                         rf'Q{pair[0]} $|1\rangle$: {round(self._phase_C1, 2)}'
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

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CZ_LocalPhases_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                self.save()
            self.plot()
            self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

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
        esp,
        heralding,
        **kwargs
    )