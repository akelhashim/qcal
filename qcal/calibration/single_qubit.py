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
from qcal.math.utils import reciprocal_uncertainty, round_to_order_error
from qcal.gate.single_qubit import Idle, VirtualZ, X90, X
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
        qpu:               QPU,
        config:            Config,
        qubits:            List | Tuple,
        amplitudes:        ArrayLike | NDArray | Dict,
        gate:              str = 'X90',
        subspace:          str = 'GE',
        compiler:          Any | Compiler | None = None, 
        transpiler:        Any | None = None, 
        n_shots:           int = 1024, 
        n_batches:         int = 1, 
        n_circs_per_seq:   int = 1, 
        # n_levels:          int = 2,
        n_gates:           int = 1,
        relative_amp:      bool = False,
        disable_esp:       bool = True,
        disable_heralding: bool = False,
        **kwargs
    ) -> Callable:
    """Function which passes a custom QPU to the Amplitude class.

    Basic example useage for initial calibration:

        cal = Amplitude(
            CustomQPU, 
            config, 
            qubits=[0, 1, 2],
            amps)
        cal.run()

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List | Tuple): qubits to calibrate.
        amplitudes (ArrayLike | NDArray | Dict): array of gate amplitudes to 
            sweep over for calibrating the single-qubit gate amplitude. If
            calibrating multiple qubits at the same time, this should be a
            dictionary mapping an array to a qubit label.
        gate (str, optional): native gate to calibrate. Defaults 
            to 'X90'.
        subspace (str, optional): qubit subspace for the defined gate.
            Defaults to 'GE'.
        compiler (Any | Compiler | None, optional): custom compiler to
            compile the experimental circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to 
            transpile the experimental circuits. Defaults to None.
        n_shots (int, optional): number of measurements per circuit. 
            Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. 
            Defaults to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that
            can be measured per sequence. Defaults to 1.
        # n_levels (int, optional): number of energy levels to be measured. 
        #     Defaults to 2. If n_levels = 3, this assumes that the
        #     measurement supports qutrit classification.
        n_gates (int, optional): number of gates for pulse repetition.
            Defaults to 1.
        relative_amp (bool, optional): whether or not the amplitudes argument
            is defined relative to the existing pulse amplitude. Defaults to
            False. If true, the amplitudes are swept over the current amplitude
            times the amplitudes argument.
        disable_esp (bool, optional): whether to disable excited state
                promotion for the calibration. Defaults to True.
        disable_heralding (bool, optional): whether to disable heralding
            for the calibraion. Defaults to False.

    Returns:
        Callable: Amplitude calibration class.
    """

    class Amplitude(qpu, Calibration):
        """Amplitude calibration class.
        
        This class inherits a custom QPU from the Amplitude calibration
        function.
        """

        def __init__(self, 
                config:            Config,
                qubits:            List | Tuple,
                amplitudes:        ArrayLike | NDArray | Dict,
                gate:              str = 'X90',
                subspace:          str = 'GE',
                compiler:          Any | Compiler | None = None, 
                transpiler:        Any | None = None, 
                n_shots:           int = 1024, 
                n_batches:         int = 1, 
                n_circs_per_seq:   int = 1, 
                # n_levels:          int = 2,
                n_gates:           int = 1,
                relative_amp:      bool = False,
                disable_esp:       bool = True,
                disable_heralding: bool = False,
                **kwargs
            ) -> None:
            """Initialize the Amplitude calibration class within the function.

            """
            # if subspace == 'EF':
            #     assert n_levels == 3, 'n_levels must be 3 for EF calibration!'
            n_levels = 3 if subspace == 'EF' else 2
            qpu.__init__(self,
                config, 
                compiler, 
                transpiler, 
                n_shots, 
                n_batches, 
                n_circs_per_seq, 
                n_levels,
                **kwargs
            )
            Calibration.__init__(self, 
                config, 
                disable_esp=disable_esp,
                disable_heralding=disable_heralding
            )

            self._qubits = qubits
            
            assert gate in ('X90', 'X'), 'gate must be an X90 or X!'
            self._gate = gate

            assert subspace in ('GE', 'EF'), 'subspace must be GE or EF!'
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
                        Cycle([
                            gate[self._gate](q, subspace='GE') 
                            for q in self._qubits
                        ]),
                        Barrier(self._qubits),
                        Cycle([
                            gate[self._gate](q, subspace='GE') 
                            for q in self._qubits
                        ]),
                        Barrier(self._qubits)
                    ])

                elif self._gate == 'X':
                    circuit.extend([
                        Cycle([
                            gate[self._gate](q, subspace='GE') 
                            for q in self._qubits
                        ]),
                        Barrier(self._qubits)
                    ])

            for _ in range(self._n_gates):
                circuit.extend([
                    Cycle([
                        gate[self._gate](q, subspace=level[self._n_levels]) 
                        for q in self._qubits
                    ]),
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
            for i, q in enumerate(self._qubits):
                prob0 = []
                for circuit in self._circuits:
                    prob0.append(
                        circuit.results.marginalize(i)['0']['probabilities']
                    )
                self._sweep_results[q] = prob0
                self._fit[q].fit(self._amplitudes[q], prob0)

                # If the fit was successful, write find the new amp
                if self._fit[q].fit_success:

                    period_frac = {'X90': 0.25, 'X': 0.5}
                    if self._n_gates == 1:  # Cosine fit
                        _, freq, _, _ = self._fit[q].fit_params
                        newvalue = 1 / freq * period_frac[self._gate] 
                        self._cal_values[q] = newvalue

                    elif self._n_gates > 1:  # Quadratic fit
                        a, b, _ = self._fit[q].fit_params
                        newvalue = -b / (2*a)  # Assume c = 0
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
                 pd.DataFrame([self._param_sweep]), 'param_sweep'
            )
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
            self._data_manager._exp_id += '_amp_calibration'
            if settings.Settings.save_data:
                self.save()
            self.plot(
                xlabel='Amplitude (a.u.)',
                ylabel=(r'$|0\rangle$ Population'),
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
        n_shots, 
        n_batches, 
        n_circs_per_seq, 
        n_gates,
        relative_amp,
        disable_esp,
        disable_heralding
    )


def Frequency(
        qpu:               QPU,
        config:            Config,
        qubits:            List | Tuple,
        t_max:             float = 2*us,
        detunings:         NDArray = np.array([-2, -1, 1, 2])*MHz,
        subspace:          str = 'GE',
        compiler:          Any | Compiler | None = None, 
        transpiler:        Any | None = None,
        n_elements:        int = 30,
        n_shots:           int = 1024, 
        n_batches:         int = 1, 
        n_circs_per_seq:   int = 1, 
        disable_esp:       bool = True,
        disable_heralding: bool = False,
        **kwargs
    ) -> Callable:
    """Function which passes a custom QPU to the Frequency class.

    Basic example useage:

        freq = Frequency(
            CustomQPU, 
            config, 
            qubits=[0],
            np.array([-2, -1, 1, 2])*MHz)
        freq.run()

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
        n_elements (int, optional): number of delays starting from 0 to t_max.
            Defaults to 30.
        n_shots (int, optional): number of measurements per circuit. 
            Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. 
            Defaults to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that
            can be measured per sequence. Defaults to 1.
        disable_esp (bool, optional): whether to disable excited state
                promotion for the calibration. Defaults to True.
        disable_heralding (bool, optional): whether to disable heralding
            for the calibraion. Defaults to False.

    Returns:
        Callable: Frequency calibration class.
    """

    class Frequency(qpu, Calibration):
        """Frequency calibration class.
        
        This class inherits a custom QPU from the Frequency calibration
        function.
        """

        def __init__(self, 
                config:            Config,
                qubits:            List | Tuple,
                t_max:             float = 2*us,
                detunings:         NDArray = np.array([-2, -1, 1, 2])*MHz,
                subspace:          str = 'GE',
                compiler:          Any | Compiler | None = None, 
                transpiler:        Any | None = None,
                n_elements:        int = 30,
                n_shots:           int = 1024, 
                n_batches:         int = 1, 
                n_circs_per_seq:   int = 1, 
                disable_esp:       bool = True,
                disable_heralding: bool = False,
                **kwargs
            ) -> None:
            """Initialize the Frequency experiment class within the function.
            """

            n_levels = 3 if subspace == 'EF' else 2
            qpu.__init__(self,
                config, 
                compiler, 
                transpiler, 
                n_shots, 
                n_batches, 
                n_circs_per_seq, 
                n_levels,
                **kwargs
            )
            Calibration.__init__(self, 
                config, 
                disable_esp=disable_esp,
                disable_heralding=disable_heralding
            )

            self._qubits = qubits
            self._detunings = detunings
            self._gate = 'X90'

            assert subspace in ('GE', 'EF'), 'subspace must be GE or EF!'
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
                # q: [FitDecayingCosine() for _ in range(detunings.size)]
                 q: [FitCosine() for _ in range(detunings.size)] 
                for q in qubits
            }

            self._freqs = {q: list() for q in qubits}
            
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
                        Cycle([X90(q, subspace='GE') for q in self._qubits]),
                        Barrier(self._qubits)
                    ])
                    if self._subspace == 'EF':
                        circuit.extend([
                          Cycle([X90(q, subspace='GE') for q in self._qubits]),
                          Barrier(self._qubits),
                          Cycle([X90(q, subspace='EF') for q in self._qubits]),
                          Barrier(self._qubits)
                        ])
                    
                    # Ramsey experiment
                    circuit.extend([
                        Cycle([Idle(q, duration=t) for q in self._qubits]),
                        Barrier(self._qubits),
                        Cycle([VirtualZ(phase, q, subspace=self._subspace) 
                            for q in self._qubits]),
                        Barrier(self._qubits),
                    ])
                    
                    # Basis preparation
                    circuit.append(
                        Cycle([X90(q, subspace=self._subspace) 
                            for q in self._qubits])
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
            for i, detuning in enumerate(self._detunings):
                for j, q in enumerate(self._qubits):
                    pop = []
                    for circuit in self._circuits[
                        self._circuits['detuning'] == detuning].circuit:
                        pop.append(
                            circuit.results.marginalize(j).populations[
                                level[self._subspace]
                            ]
                        )
                    self._sweep_results[q].append(pop)

                    # e = np.array(pop).min()
                    # a = np.array(pop).max() - e
                    # b = -np.mean( np.diff(pop) / np.diff(self._times[q]) ) / a
                    # c = detuning
                    # d = 0.
                    # self._freq_fit[q][i].fit(
                    #     self._times[q], pop, p0=(a, b, c, d, e)
                    # )
                    d = np.array(pop).min()
                    a = np.array(pop).max() - d
                    b = np.abs(detuning)
                    c = 0.
                    self._freq_fit[q][i].fit(
                        self._times[q], pop, p0=(a, b, c, d)
                    )

                    # If the fit was successful, grab the frequency
                    if self._freq_fit[q][i].fit_success:
                        self._freqs[q].append(
                            self._freq_fit[q][i].fit_params[1] ## 2 for FtiDecaying
                        )
            
            # Fit the characterized frequencies to an absolute value curve
            for i, q in enumerate(self._qubits):
                self._fit[q].fit(self._detunings, self._freqs[q], p0=(1, 0, 0))

                if self._fit[q].fit_success:
                    val, err = round_to_order_error(
                        self._fit[q].fit_params[1],
                        self._fit[q].error[1]
                    )
                    self._cal_values[q] = self._config[self._params[q]] + val
                    self._errors[q] = err

        def save(self) -> None:
            """Save all circuits and data."""
            qpu.save(self)
            # self._data_manager.save_to_csv(
            #      pd.DataFrame([self._param_sweep]), 'param_sweep'
            # )
            # self._data_manager.save_to_csv(
            #      pd.DataFrame([self._sweep_results]), 'sweep_results'
            # )
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
            fig, ax = plt.subplots(
                nrows, 2, figsize=figsize, layout='constrained'
            )
            colors = plt.get_cmap('viridis', self._detunings.size)

            for i, q in enumerate(self._qubits):
                if len(self._qubits) == 1:
                    ax = ax
                else:
                    ax = ax[i]

                ax[0].set_xlabel(r'Time ($\mu$s)', fontsize=15)
                ax[1].set_xlabel('Detuning (MHz)', fontsize=15)
                ax[0].set_ylabel('Ramsey', fontsize=15)
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
                    self._detunings, 
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
                    df = round(self._fit[q].fit_params[1] / MHz, 2)
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
            # if settings.Settings.save_data:
            #     fig.savefig(
            #         self._data_manager._save_path + 'freq_calibration.png', 
            #         dpi=300
            #     )
            plt.show()

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            clear_output(wait=True)
            self._data_manager._exp_id += '_freq_calibration'
            # if settings.Settings.save_data:
            #     self.save()
            self.plot()
            # self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

    return Frequency(
        config,
        qubits,
        t_max,
        detunings,
        subspace,
        compiler, 
        transpiler,
        n_elements, 
        n_shots, 
        n_batches, 
        n_circs_per_seq, 
        disable_esp,
        disable_heralding
    )