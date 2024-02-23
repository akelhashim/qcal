"""Submodule for single-qubit gate calibration.

"""
from __future__ import annotations

import qcal.settings as settings

from .calibration import Calibration
from .utils import find_pulse_index, in_range
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.fitting.fit import (
    FitAbsoluteValue, FitCosine, FitDecayingCosine, FitParabola
)
from qcal.managers.classification_manager import ClassificationManager
from qcal.math.utils import round_to_order_error
from qcal.gate.single_qubit import Idle, VirtualZ, X, X90
from qcal.plotting.utils import calculate_nrows_ncols
from qcal.qpu.qpu import QPU
from qcal.units import MHz, us

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.display import clear_output
from typing import Any, Callable, Dict, List, Tuple
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


def Amplitude(
        qpu:             QPU,
        config:          Config,
        qubits:          List | Tuple,
        amplitudes:      ArrayLike | NDArray | Dict[ArrayLike | NDArray],
        gate:            str = 'X90',
        subspace:        str = 'GE',
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1, 
        n_gates:         int = 1,
        relative_amp:    bool = False,
        raster_circuits: bool = False,
        **kwargs
    ) -> Callable:
    """Amplitude calibration for single-qubit gates.

    Basic example useage for initial calibration:

    ```
    amps = np.linspace(0, 1.0, 21)
    cal = Amplitude(
        CustomQPU, 
        config, 
        qubits=[0, 1, 2],
        amplitudes=amps)
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List | Tuple): qubits to calibrate.
        amplitudes (ArrayLike | NDArray | Dict[ArrayLike | NDArray]): array of 
            gate amplitudes to sweep over for calibrating the single-qubit gate 
            amplitude. If calibrating multiple qubits at the same time, this 
            should be a dictionary mapping an array to a qubit label.
        gate (str, optional): native gate to calibrate. Defaults 
            to 'X90'.
        subspace (str, optional): qubit subspace for the defined gate.
            Defaults to 'GE'.
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
                qubits:          List | Tuple,
                amplitudes:      ArrayLike | NDArray | Dict,
                gate:            str = 'X90',
                subspace:        str = 'GE',
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1, 
                n_gates:         int = 1,
                relative_amp:    bool = False,
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            """Initialize the Amplitude calibration class within the function.
            """
            n_levels = 3 if subspace == 'EF' else 2
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

            self._qubits = qubits
            
            assert gate in ('X90', 'X'), (
                "'gate' must be one of 'X90' or 'X'!"
            )
            self._gate = gate

            assert subspace in ('GE', 'EF'), (
                "'subspace' must be one of 'GE' or 'EF'!"
            )
            self._subspace = subspace

            if not isinstance(amplitudes, dict):
                self._amplitudes = {q: amplitudes for q in qubits}
            else:
                self._amplitudes = amplitudes
            self._param_sweep = self._amplitudes

            self._params = {}
            for q in qubits:
                param = f'single_qubit/{q}/{subspace}/{gate}/pulse'
                param += f'/{find_pulse_index(self._config, param)}/'
                param += 'kwargs/amp'
                self._params[q] = param

            if relative_amp:
                for q in qubits:
                    self._amplitudes[q] = (
                        self._config[self._params[q]] * self._amplitudes[q]
                    )
            self._relative_amp = relative_amp

            if n_gates > 1 and gate == 'X90':
                assert n_gates % 4 == 0, 'n_gates must be a multiple of 4!'
            elif n_gates > 1 and gate == 'X':
                assert n_gates % 2 == 0, 'n_gates must be a multiple of 2!'
            self._n_gates = n_gates

            if n_gates == 1:
                self._fit = {q: FitCosine() for q in qubits}
            elif n_gates > 1:
                self._fit = {q: FitParabola() for q in qubits}

        @property
        def amplitudes(self) -> Dict:
            """Amplitude sweep for each qubit.

            Returns:
                Dict: qubit to array map.
            """
            return self._amplitudes

        def generate_circuits(self):
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            gate = {'X90': X90, 'X': X}
            level = {2: 'GE', 3: 'EF'}
            circuit = Circuit()

            # Prepulse for EF calibration
            if self._subspace == 'EF':
                
                if self._gate == 'X90':
                    circuit.extend([
                        Cycle({
                            gate[self._gate](q, subspace='GE') 
                            for q in self._qubits
                        }),
                        Barrier(self._qubits),
                        Cycle({
                            gate[self._gate](q, subspace='GE') 
                            for q in self._qubits
                        }),
                        Barrier(self._qubits)
                    ])

                elif self._gate == 'X':
                    circuit.extend([
                        Cycle({
                            gate[self._gate](q, subspace='GE') 
                            for q in self._qubits
                        }),
                        Barrier(self._qubits)
                    ])

            for _ in range(self._n_gates):
                circuit.extend([
                    Cycle({
                        gate[self._gate](q, subspace=level[self._n_levels]) 
                        for q in self._qubits
                    }),
                    Barrier(self._qubits)
                ])
            circuit.measure()
            
            circuits = []
            for _ in range(self._amplitudes[self._qubits[0]].size):
                circuits.append(circuit.copy())

            self._circuits = CircuitSet(circuits=circuits)
            for q in self._qubits:
                self._circuits[
                        f'param: {self._params[q]}'
                    ] = self._amplitudes[q]
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            # Fit the probability of being in 0 from the gate amplitude sweep 
            # to a cosine or parabola
            level = {'GE': '0', 'EF': '1'}
            for i, q in enumerate(self._qubits):
                prob = []
                for circuit in self._circuits:
                    prob.append(
                        circuit.results.marginalize(i).populations[
                                level[self._subspace]
                            ]
                    )
                self._sweep_results[q] = prob
                self._circuits[f'Q{q}: Prob({level[self._subspace]})'] = prob

                self._fit[q].fit(self._amplitudes[q], prob)

                # If the fit was successful, write find the new amp
                if self._fit[q].fit_success:

                    period_frac = {'X90': 0.25, 'X': 0.5}
                    if self._n_gates == 1:  # Cosine fit
                        _, freq, _, _ = self._fit[q].fit_params
                        newvalue = 1 / freq * period_frac[self._gate] 
                        if not in_range(newvalue, self._amplitudes[q]):
                            logger.warning(
                              f'Fit failed for qubit {q} (out of range)!'
                            )
                            self._fit[q]._fit_success = False
                        else:
                            self._cal_values[q] = newvalue

                    elif self._n_gates > 1:  # Quadratic fit
                        a, b, _ = self._fit[q].fit_params
                        newvalue = -b / (2 * a)  # Assume c = 0
                        if a > 0:
                            logger.warning(
                              f'Fit failed for qubit {q} (positive curvature)!'
                            )
                            self._fit[q]._fit_success = False
                        elif not in_range(newvalue, self._amplitudes[q]):
                            logger.warning(
                              f'Fit failed for qubit {q} (out of range)!'
                            )
                            self._fit[q]._fit_success = False
                        else:
                            self._cal_values[q] = newvalue

        def save(self):
            """Save all circuits and data."""
            qpu.save(self)
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
                f'_amp_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                self.save()
            self.plot(
                xlabel='Amplitude (a.u.)',
                ylabel=(
                    r'$|0\rangle$ Population' if self._subspace == 'GE' else 
                    r'$|1\rangle$ Population'
                ),
                save_path=self._data_manager._save_path
            )
            self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

    return Amplitude(
        config,
        qubits,
        amplitudes,
        gate,
        subspace,
        compiler, 
        transpiler,
        classifier,
        n_shots, 
        n_batches, 
        n_circs_per_seq, 
        n_gates,
        relative_amp,
        raster_circuits,
        **kwargs
    )


def Frequency(
        qpu:             QPU,
        config:          Config,
        qubits:          List | Tuple,
        t_max:           float = 2*us,
        detunings:       NDArray = np.array([-2, -1, 1, 2])*MHz,
        subspace:        str = 'GE',
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_elements:      int = 30,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1, 
        raster_circuits: bool = False,
        **kwargs
    ) -> Callable:
    """Frequency calibration for single qubits.

    Basic example useage:

    ```
    cal = Frequency(
        CustomQPU, 
        config, 
        qubits=[0],
        detunings=np.array([-2, -1, 1, 2])*MHz)
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List | Tuple): qubits to measure.
        t_max (float, optional): maximum Ramsey time. Defaults to 2 us.
        detunings (float, optional): artificial detunings from the actual qubit
            frequency. Defaults to np.array([-2, -1, 1, 2])*MHz.
        subspace (str, optional): qubit subspace for frequency calibration.
            Defaults to 'GE'.
        compiler (Any | Compiler | None, optional): custom compiler to
            compile the experimental circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to 
            transpile the experimental circuits. Defaults to None.
        classifier (ClassificationManager, optional): manager used for 
            classifying raw data. Defaults to None.
        n_elements (int, optional): number of delays starting from 0 to t_max.
            Defaults to 30.
        n_shots (int, optional): number of measurements per circuit. 
            Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. 
            Defaults to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that
            can be measured per sequence. Defaults to 1.
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
                qubits:          List | Tuple,
                t_max:           float = 2*us,
                detunings:       NDArray = np.array([-2, -1, 1, 2])*MHz,
                subspace:        str = 'GE',
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_elements:      int = 30,
                n_shots:         int = 1024, 
                n_batches:       int = 1,
                n_circs_per_seq: int = 1,
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            """Initialize the Frequency experiment class within the function.
            """
            n_levels = 3 if subspace == 'EF' else 2
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

            self._qubits = qubits
            self._detunings = detunings

            assert subspace in ('GE', 'EF'), (
                "'subspace' must be one of 'GE' or 'EF'!"
            )
            self._subspace = subspace

            self._times = {
                q: np.linspace(0., t_max, n_elements) for q in qubits
            }
            self._param_sweep = self._times

            self._params = {
                q: f'single_qubit/{q}/{subspace}/freq' for q in qubits
            }
            self._sweep_results = {
                q: list() for q in qubits
            }
            self._fit = {
                q: FitAbsoluteValue() for q in qubits
            }
            self._freq_fit = {
                q: [FitDecayingCosine() for _ in range(detunings.size)]
                for q in qubits
            }

            self._freqs = {q: list() for q in qubits}
            self._fit_detunings = {q: list() for q in self._qubits}
            
        @property
        def detunings(self) -> NDArray:
            """Frequency detunings.

            Returns:
                NDArray: frequency detunings..
            """
            return self._detunings

        @property
        def times(self) -> Dict:
            """Time sweep for each qubit.

            Returns:
                Dict: qubit to time array map.
            """
            return self._times

        def generate_circuits(self):
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            circuits = []
            times = []
            detunings = []
            for detuning in self._detunings:
                for t in self._times[self._qubits[0]]:
                    times.append(t)
                    detunings.append(detuning)
                    phase = 2. * np.pi * detuning * t
                    
                    circuit = Circuit()

                    # State prepration
                    circuit.extend([
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                        Barrier(self._qubits)
                    ])
                    if self._subspace == 'EF':
                        circuit.extend([
                          Cycle({X90(q, subspace='GE') for q in self._qubits}),
                          Barrier(self._qubits),
                          Cycle({X90(q, subspace='EF') for q in self._qubits}),
                          Barrier(self._qubits)
                        ])
                    
                    # Ramsey experiment
                    circuit.extend([
                        Cycle({Idle(q, duration=t) for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({VirtualZ(q, phase, subspace=self._subspace) 
                              for q in self._qubits}),
                        Barrier(self._qubits),
                    ])
                    
                    # Basis preparation
                    circuit.append(
                        Cycle({X90(q, subspace=self._subspace) 
                            for q in self._qubits})
                    )
                    
                    circuit.measure()
                    circuits.append(circuit)
            
            self._circuits = CircuitSet(circuits=circuits)
            self._circuits['detuning'] = detunings
            self._circuits['time'] = times
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            level = {'GE': '1', 'EF': '2'}
            # Fit the measured frequency for each detuning
            probs = {q: list() for q in self._qubits}
            for i, detuning in enumerate(self._detunings):
                for j, q in enumerate(self._qubits):
                    prob = []
                    for circuit in self._circuits[
                        self._circuits['detuning'] == detuning].circuit:
                        prob.append(
                            circuit.results.marginalize(j).populations[
                                level[self._subspace]
                            ]
                        )
                    probs[q] += prob
                    self._sweep_results[q].append(prob)

                    e = np.array(prob).min()
                    a = np.array(prob).max() - e
                    b = -np.mean( np.diff(prob) / np.diff(self._times[q]) ) / a
                    c = detuning
                    d = 0.
                    self._freq_fit[q][i].fit(
                        self._times[q], prob, p0=(a, b, c, d, e)
                    )

                    # If the fit was successful, grab the frequency
                    if self._freq_fit[q][i].fit_success:
                        self._freqs[q].append(
                            abs(self._freq_fit[q][i].fit_params[2])
                        )
                        self._fit_detunings[q].append(detuning)
                
            for q in self._qubits:
                self._circuits[f'Q{q}: Prob({level[self._subspace]})'] = (
                    probs[q]
                )
            
            # Fit the characterized frequencies to an absolute value curve
            for i, q in enumerate(self._qubits):
                self._fit[q].fit(self._detunings, self._freqs[q], p0=(1, 0, 0)) 

                if self._fit[q].fit_success:
                    a, _, _ = self._fit[q].fit_params
                    newval, err = round_to_order_error(
                        self._fit[q].fit_params[1],
                        self._fit[q].error[1],
                        2
                    )

                    if a < 0:
                        logger.warning(
                            f'Fit failed for qubit {q} (negative curvature)!'
                        )
                        self._fit[q]._fit_success = False
                    if not in_range(newval, self._detunings):
                        logger.warning(
                            f'Fit failed for qubit {q} (out of range)!'
                        )
                        self._fit[q]._fit_success = False
                    else:
                        self._cal_values[q] = (
                            self._config[self._params[q]] + newval
                        )
                        self._errors[q] = err

        def save(self) -> None:
            """Save all circuits and data."""
            qpu.save(self)
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._cal_values]), 'freq_values'
            )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._errors]), 'freq_fit_errors'
            )

        def plot(self) -> None:
            """Plot the frequency sweep and fit results."""
            nrows = len(self._qubits)
            figsize = (10, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, 2, figsize=figsize, layout='constrained'
            )
            colors = plt.get_cmap('viridis', self._detunings.size)
            level = {'GE': '1', 'EF': '2'}

            for i, q in enumerate(self._qubits):
                if len(self._qubits) == 1:
                    ax = axes
                else:
                    ax = axes[i]

                ax[0].set_xlabel(r'Time ($\mu$s)', fontsize=15)
                ax[1].set_xlabel('Detuning (MHz)', fontsize=15)
                ax[0].set_ylabel(
                    rf'$|{level[self._subspace]}\rangle$ Population', 
                    fontsize=15
                )
                ax[1].set_ylabel('Measured Detuning (MHz)', fontsize=15)
                ax[0].tick_params(axis='both', which='major', labelsize=12)
                ax[1].tick_params(axis='both', which='major', labelsize=12)
                ax[0].grid(True)
                ax[1].grid(True)

                # Plot the Ramsey sweeps
                for j, det in enumerate(self._detunings):
                    ax[0].plot(
                        self._times[q],
                        np.array(self._sweep_results[q][j]) + j,
                        'o', 
                        c=colors(j), 
                        label=f'{round(det / MHz, 1)} MHz'
                    )

                    if self._freq_fit[q][j].fit_success:
                        x = np.linspace(
                            self._times[q][0], self._times[q][-1], 100
                        )
                        ax[0].plot(x, 
                            self._freq_fit[q][j].predict(x) + j,
                            ls='-', 
                            c=colors(j)
                        )

                # Plot the frequency fit
                ax[1].plot(
                    self._fit_detunings[q], 
                    self._freqs[q],
                    'o',
                    c='blue', 
                    label=f'Meas, Q{q}'
                )
                if self._fit[q].fit_success:
                    x = np.linspace(
                            self._detunings[0], 
                            self._detunings[-1], 
                            100
                        )
                    ax[1].plot(x,
                        self._fit[q].predict(x),
                        ls='-',
                        c='orange',
                        label='Fit'
                    )
                    df = round(self._fit[q].fit_params[1] / MHz, 3)
                    ax[1].axvline(
                        round(self._fit[q].fit_params[1], 2),  
                        ls='--', c='k', label=rf'$\Delta f$ = {df} MHz'
                    )

                ax[0].legend(loc=0, fontsize=12)
                ax[1].legend(loc=0, fontsize=12)
                ax[0].xaxis.set_major_formatter(
                        lambda x, pos: round(x / 1e-6, 1)
                    )
                ax[1].xaxis.set_major_formatter(
                        lambda x, pos: round(x / MHz, 1)
                    )
                ax[1].yaxis.set_major_formatter(
                        lambda x, pos: round(x / MHz, 1)
                    )

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
                f'_freq_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                self.save()
            self.plot()
            self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

    return Frequency(
        config,
        qubits,
        t_max,
        detunings,
        subspace,
        compiler, 
        transpiler,
        classifier,
        n_elements, 
        n_shots, 
        n_batches, 
        n_circs_per_seq, 
        raster_circuits,
        **kwargs
    )


def Phase(
        qpu:             QPU,
        config:          Config,
        qubits:          List | Tuple,
        phases:          ArrayLike | NDArray | Dict[ArrayLike | NDArray],
        subspace:        str = 'GE',
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_shots:         int = 1024, 
        n_batches:       int = 1,
        n_circs_per_seq: int = 1,
        raster_circuits: bool = False,
        **kwargs
    ) -> Callable:
    """Phase calibration for single-qubit gates.

    This calibration corrects for phases errors that occur due to stark shifts
    during single-qubit gates.

    Basic example useage for initial calibration:

    ```
    phases = np.pi * np.linspace(-1, 1, 21)
    cal = Phase(
        CustomQPU, 
        config, 
        qubits=[0, 1, 2],
        phases=phases)
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List | Tuple): qubits to calibrate.
        phases (ArrayLike | NDArray | Dict[ArrayLike | NDArray]): array of
            phases to sweep over for calibrating the single-qubit gate 
            phases. If calibrating multiple qubits at the same time, this 
            should be a dictionary mapping an array to a qubit label.
        subspace (str, optional): qubit subspace for the defined gate.
            Defaults to 'GE'.
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
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.

    Returns:
        Callable: Phases calibration class.
    """

    class Phase(qpu, Calibration):
        """Phase calibration class.
        
        This class inherits a custom QPU from the Phase calibration
        function.
        """

        def __init__(self, 
                config:          Config,
                qubits:          List | Tuple,
                phases:          ArrayLike | NDArray | Dict,
                subspace:        str = 'GE',
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            """Initialize the Phase calibration class within the function."""
            n_levels = 3 if subspace == 'EF' else 2
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

            self._qubits = qubits

            assert subspace in ('GE', 'EF'), (
                "'subspace' must be one of 'GE' or 'EF'!"
            )
            self._subspace = subspace

            if not isinstance(phases, dict):
                self._phases = {q: phases for q in qubits}
            else:
                self._phases = phases
            self._param_sweep = self._phases

            self._params = {}
            for q in qubits:
                self._params[q] = [
                    f'single_qubit/{q}/{subspace}/X90/pulse/0/kwargs/phase',
                    f'single_qubit/{q}/{subspace}/X90/pulse/2/kwargs/phase'
                ]

            self._fit = {q: FitParabola() for q in qubits}

        @property
        def phases(self) -> Dict:
            """Phase sweep for each qubit.

            Returns:
                Dict: qubit to array map.
            """
            return self._phases

        def generate_circuits(self):
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            level = {2: 'GE', 3: 'EF'}
            circuit_Y180_X90 = Circuit()
            circuit_X180_Y90 = Circuit()

            # Prepulse for EF calibration
            if self._subspace == 'EF':
                
                circuit_Y180_X90.extend([
                    Cycle({X90(q, subspace='GE') for q in self._qubits}),
                    Barrier(self._qubits),
                    Cycle({X90(q, subspace='GE') for q in self._qubits}),
                    Barrier(self._qubits)
                ])

                circuit_X180_Y90.extend([
                    Cycle({X90(q, subspace='GE') for q in self._qubits}),
                    Barrier(self._qubits),
                    Cycle({X90(q, subspace='GE') for q in self._qubits}),
                    Barrier(self._qubits)
                ])

            # Y180_X90
            circuit_Y180_X90.extend([
                Cycle({
                    VirtualZ(q, np.pi/2, subspace=level[self._n_levels])
                    for q in self._qubits
                }),
                Cycle({
                    X90(q, subspace=level[self._n_levels]) 
                    for q in self._qubits
                }),
                Barrier(self._qubits),
                Cycle({
                    X90(q, subspace=level[self._n_levels]) 
                    for q in self._qubits
                }),
                Cycle({
                    VirtualZ(q, -np.pi/2, subspace=level[self._n_levels])
                    for q in self._qubits
                }),
                Barrier(self._qubits),
                Cycle({
                    X90(q, subspace=level[self._n_levels]) 
                    for q in self._qubits
                })
            ])
            circuit_Y180_X90.measure()

            # X180_Y90
            circuit_X180_Y90.extend([
                Cycle({
                    X90(q, subspace=level[self._n_levels]) 
                    for q in self._qubits
                }),
                Barrier(self._qubits),
                Cycle({
                    X90(q, subspace=level[self._n_levels]) 
                    for q in self._qubits
                }),
                Barrier(self._qubits),
                Cycle({
                    VirtualZ(q, np.pi/2, subspace=level[self._n_levels])
                    for q in self._qubits
                }),
                Cycle({
                    X90(q, subspace=level[self._n_levels]) 
                    for q in self._qubits
                }),
                Cycle({
                    VirtualZ(q, -np.pi/2, subspace=level[self._n_levels])
                    for q in self._qubits
                }),
            ])
            circuit_X180_Y90.measure()
            
            circuits = []
            for _ in range(self._phases[self._qubits[0]].size):
                circuits.append(circuit_Y180_X90.copy())
                circuits.append(circuit_X180_Y90.copy())

            self._circuits = CircuitSet(circuits=circuits)
            self._circuits['sequence'] = [
                    'Y180_X90', 'X180_Y90'
                ] * int(self._circuits.n_circuits / 2)
            for q in self._qubits:
                _phases = []
                for p in self._phases[q]:
                    _phases.extend([p, p])
                self._circuits[
                        f'param: {self._params[q][0]}'
                    ] = _phases
                self._circuits[
                        f'param: {self._params[q][1]}'
                    ] = _phases
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            level = {'GE': '1', 'EF': '2'}
            for i, q in enumerate(self._qubits):
                pop0 = []  # circuit_Y180_X90
                pop1 = []  # circuit_X180_Y90
                pops = []
                for j, circuit in enumerate(self._circuits):
                    pop = circuit.results.marginalize(i).populations[
                        level[self._subspace]
                    ]
                    pops.append(pop)
                    
                    # 'Y180_X90'
                    if j % 2 == 0:
                        pop0.append(pop)
                    # 'X180_Y90'
                    elif j % 2 == 1:
                        pop1.append(pop)
        
                self._circuits[f'Q{q}: Prob({level[self._subspace]})'] = pops
                self._sweep_results[q] = (np.array(pop0) - np.array(pop1))**2
                
                self._fit[q].fit(self._phases[q], self._sweep_results[q])

                # If the fit was successful, write find the new phase
                if self._fit[q].fit_success:
                    a, b, _ = self._fit[q].fit_params
                    newvalue = -b / (2 * a)  # Assume c = 0
                    if a < 0:
                        logger.warning(
                            f'Fit failed for qubit {q} (negative curvature)!'
                        )
                        self._fit[q]._fit_success = False
                    elif not in_range(newvalue, self._phases[q]):
                        logger.warning(
                            f'Fit failed for qubit {q} (out of range)!'
                        )
                        self._fit[q]._fit_success = False
                    else:
                        self._cal_values[q] = newvalue

        def save(self):
            """Save all circuits and data."""
            qpu.save(self)
            # self._data_manager.save_to_csv(
            #      pd.DataFrame([self._param_sweep]), 'param_sweep'
            # )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._sweep_results]), 'sweep_results'
            )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._cal_values]), 'calibrated_values'
            )

        def plot(self) -> None:
            """Plot the phase sweep and fit results."""
            level = {'GE': '1', 'EF': '2'}

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

                        ax.set_xlabel('Phase (rad.)', fontsize=15)
                        ax.set_ylabel(
                            r'$|2\rangle$ Population' if self._subspace == 'EF' 
                            else r'$|1\rangle$ Population',
                            fontsize=15
                        )
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )
                        ax.grid(True)

                        ax.plot(
                            self._param_sweep[q], 
                            self._circuits[
                                self._circuits['sequence'] == 'Y180_X90'
                            ][f'Q{q}: Prob({level[self._subspace]})'],
                            '-o', c='blue', label='Y180_X90'
                        )
                        ax.plot(
                            self._param_sweep[q], 
                            self._circuits[
                                self._circuits['sequence'] == 'X180_Y90'
                            ][f'Q{q}: Prob({level[self._subspace]})'],
                            '-o', c='blueviolet', label='X180_Y90'
                        )
                        ax.plot(
                            self._param_sweep[q], self._sweep_results[q],
                            'o', c='k', label=f'Meas, Q{q}'
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

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_phase_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                self.save()
            self.plot()
            self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

    return Phase(
        config,
        qubits,
        phases,
        subspace,
        compiler, 
        transpiler,
        classifier,
        n_shots, 
        n_batches, 
        n_circs_per_seq,
        raster_circuits,
        **kwargs
    )