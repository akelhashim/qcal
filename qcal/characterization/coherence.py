""""Submodule for qubit coherence experiments.

"""
import qcal.settings as settings

from qcal.characterization.characterize import Characterize
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.fitting.fit import FitDecayingCosine, FitExponential
from qcal.gate.single_qubit import Idle, X90, X, VirtualZ
from qcal.math.utils import reciprocal_uncertainty, round_to_order_error
from qcal.qpu.qpu import QPU
from qcal.units import MHz, us

import logging
import numpy as np
import pandas as pd

from IPython.display import clear_output
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


def T1(qpu:               QPU,
       config:            Config,
       qubits:            List | Tuple,
       t_max:             float = 500*us,
       gate:              str = 'X90',
       subspace:          str = 'GE',
       compiler:          Any | Compiler | None = None, 
       transpiler:        Any | None = None,
       n_elements:        int = 50,
       n_shots:           int = 1024, 
       n_batches:         int = 1, 
       n_circs_per_seq:   int = 1, 
       disable_esp:       bool = True,
       disable_heralding: bool = False,
       **kwargs
    ) -> Callable:
    """Function which passes a custom QPU to the Amplitude class.

    Basic example useage:

        exp = T1(
            CustomQPU, 
            config, 
            qubits=[0, 1, 2],
            t_max=5e-4)
        exp.run()

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List | Tuple): qubits to measure.
        t_max (float, option): maximum wait time. Defaults to 500 us.
        gate (str, optional): native gate used for state preparation. Defaults 
            to 'X90'.
        subspace (str, optional): qubit subspace for T1 measurement.
            Defaults to 'GE'.
        compiler (Any | Compiler | None, optional): custom compiler to
            compile the experimental circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to 
            transpile the experimental circuits. Defaults to None.
        n_elements (int, optional): number of delays starting from 0 to t_max.
            Defaults to 50.
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
        Callable: T1 class.
    """

    class T1(qpu, Characterize):
        """T1 characterization class.
        
        This class inherits a custom QPU from the T1 characterization
        function.
        """

        def __init__(self, 
                config:            Config,
                qubits:            List | Tuple,
                t_max:             float = 500*us,
                gate:              str = 'X90',
                subspace:          str = 'GE',
                compiler:          Any | Compiler | None = None, 
                transpiler:        Any | None = None,
                n_elements:        int = 50,
                n_shots:           int = 1024, 
                n_batches:         int = 1, 
                n_circs_per_seq:   int = 1, 
                disable_esp:       bool = True,
                disable_heralding: bool = False,
                **kwargs
            ) -> None:
            """Initialize the T1 experiment class within the function."""

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
            Characterize.__init__(self, 
                config, 
                disable_esp=disable_esp,
                disable_heralding=disable_heralding
            )

            self._qubits = qubits
            
            assert gate in ('X90', 'X'), 'gate must be an X90 or X!'
            self._gate = gate

            assert subspace in ('GE', 'EF'), 'subspace must be GE or EF!'
            self._subspace = subspace

            self._times = {
                q: np.linspace(0., t_max, n_elements) for q in qubits
            }
            self._param_sweep = self._times

            self._params = {
                q: f'single_qubit/{q}/{subspace}/T1' for q in qubits
            }
            self._fit = {q: FitExponential() for q in qubits}

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
            for t in self._times[self._qubits[0]]:
                circuit = Circuit()

                # State prepration
                if self._gate == 'X90':
                    circuit.extend([
                        Cycle([X90(q, subspace='GE') for q in self._qubits]),
                        Barrier(self._qubits),
                        Cycle([X90(q, subspace='GE') for q in self._qubits]),
                        Barrier(self._qubits)
                    ])

                    if self._subspace == 'EF':
                        circuit.extend([
                            Cycle([X90(q, subspace='EF') 
                                   for q in self._qubits]),
                            Barrier(self._qubits),
                            Cycle([X90(q, subspace='EF') 
                                   for q in self._qubits]),
                            Barrier(self._qubits)
                        ])

                elif self._gate == 'X':
                    circuit.extend([
                        Cycle([X(q, subspace='GE') for q in self._qubits]),
                        Barrier(self._qubits)
                    ])

                    if self._subspace == 'EF':
                        circuit.extend([
                            Cycle([X(q, subspace='EF') for q in self._qubits]),
                            Barrier(self._qubits),
                        ])

                # T1 delay
                circuit.append(
                    Cycle([Idle(q, duration=t) for q in self._qubits]),
                )
                circuit.measure()

                circuits.append(circuit)
            
            self._circuits = CircuitSet(circuits=circuits)
            self._circuits['time'] = self._times[self._qubits[0]]
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            pop = {'GE': '1', 'EF': '2'}
            # Fit the probability of being in 1 from the time sweep to an 
            # exponential
            for i, q in enumerate(self._qubits):
                prob1 = []
                for circuit in self._circuits:
                    prob1.append(
                        circuit.results.marginalize(i).populations[
                            pop[self._subspace]
                        ]
                    )
                self._results[q] = prob1

                # Add initial guesses to fit
                c = np.array(prob1).min()
                a = np.array(prob1).max() - c
                b = -np.mean( np.diff(prob1) / np.diff(self._times[q]) ) / a
                self._fit[q].fit(self._times[q], prob1, p0=(a, b, c))

                # If the fit was successful, write to the config
                if self._fit[q].fit_success:
                    val, err = round_to_order_error(
                        *reciprocal_uncertainty(
                            self._fit[q].fit_params[1], self._fit[q].error[1]
                        )
                    )
                    self._char_values[q] = val
                    self._errors[q] = err
                    # self._char_values[q] = 1 / self._fit[q].fit_params[1]
                    # self._errors[q] = 1 / self._fit[q].error[1]

        def save(self):
            """Save all circuits and data."""
            qpu.save(self)
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._param_sweep]), 'param_sweep'
            )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._results]), 'sweep_results'
            )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._char_values]), 'T1_values'
            )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._errors]), 'T1_errors'
            )

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            clear_output(wait=True)
            if settings.Settings.save_data:
                self._data_manager._exp_id += (
                    f'_T1_Q{"".join(str(q) for q in self._qubits)}'
                )
                self.save()
                self.plot(
                    xlabel=r'Time ($\mu$s)',
                    ylabel=(
                       r'$|2\rangle$ Population' if self._subspace == 'EF' else
                       r'$|1\rangle$ Population'
                    ),
                    flabel=r'$T_1$',
                    save_path=self._data_manager._save_path
                )
            self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

    return T1(
        config,
        qubits,
        t_max,
        gate,
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


def T2(qpu:               QPU,
       config:            Config,
       qubits:            List | Tuple,
       t_max:             float = 250*us,
       detuning:          float = 0.05 * MHz,
       echo:              bool = False,
       subspace:          str = 'GE',
       compiler:          Any | Compiler | None = None, 
       transpiler:        Any | None = None,
       n_elements:        int = 50,
       n_shots:           int = 1024, 
       n_batches:         int = 1, 
       n_circs_per_seq:   int = 1, 
       disable_esp:       bool = True,
       disable_heralding: bool = False,
       **kwargs
    ) -> Callable:
    """Function which passes a custom QPU to the Amplitude class.

    Basic example useage:

        exp = T2(
            CustomQPU, 
            config, 
            qubits=[0, 1, 2],
            t_max=250e-4,
            detuning=0.05e6,
            echo=True)
        exp.run()

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List | Tuple): qubits to measure.
        t_max (float, optional): maximum wait time. Defaults to 250 us.
        detuning (float, optional): artificial detuning from the actual qubit
            frequency. Defaults to 0.05 MHz.
        echo (bool, optional): whether to echo the qubit in the middle. 
            Defaults to False.
        subspace (str, optional): qubit subspace for T2 measurement.
            Defaults to 'GE'.
        compiler (Any | Compiler | None, optional): custom compiler to
            compile the experimental circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to 
            transpile the experimental circuits. Defaults to None.
        n_elements (int, optional): number of delays starting from 0 to t_max.
            Defaults to 50.
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
        Callable: T2 class.
    """

    class T2(qpu, Characterize):
        """T2 characterization class.
        
        This class inherits a custom QPU from the T2 characterization
        function.
        """

        def __init__(self, 
                config:            Config,
                qubits:            List | Tuple,
                t_max:             float = 250*us,
                detuning:          float = 0.05 * MHz,
                echo:              bool = False,
                subspace:          str = 'GE',
                compiler:          Any | Compiler | None = None, 
                transpiler:        Any | None = None,
                n_elements:        int = 50,
                n_shots:           int = 1024, 
                n_batches:         int = 1, 
                n_circs_per_seq:   int = 1, 
                disable_esp:       bool = True,
                disable_heralding: bool = False,
                **kwargs
            ) -> None:
            """Initialize the T2 experiment class within the function."""

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
            Characterize.__init__(self, 
                config, 
                disable_esp=disable_esp,
                disable_heralding=disable_heralding
            )

            self._qubits = qubits
            self._echo = echo
            self._detuning = detuning
            self._gate = 'X90'

            assert subspace in ('GE', 'EF'), 'subspace must be GE or EF!'
            self._subspace = subspace

            self._times = {
                q: np.linspace(0., t_max, n_elements) for q in qubits
            }
            self._param_sweep = self._times

            if not echo:
                self._params = {
                    q: f'single_qubit/{q}/{subspace}/T2*' for q in qubits
                }
                self._fit = {q: FitDecayingCosine() for q in qubits}
            elif echo:
                self._params = {
                    q: f'single_qubit/{q}/{subspace}/T2e' for q in qubits
                }
                self._fit = {q: FitExponential() for q in qubits}
            
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
            for t in self._times[self._qubits[0]]:
                phase = 2. * np.pi * self._detuning * t  # theta = 2pi*freq*t
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

                # T2 experiment
                if not self._echo:
                    circuit.extend([
                        Cycle([Idle(q, duration=t) for q in self._qubits]),
                        Barrier(self._qubits),
                        Cycle([VirtualZ(phase, q, subspace=self._subspace) 
                               for q in self._qubits]),
                        Barrier(self._qubits),
                    ])
                elif self._echo:
                    circuit.extend([
                        Cycle([Idle(q, duration=t/2) for q in self._qubits]),
                        Barrier(self._qubits),
                        Cycle([VirtualZ(phase/2, q, subspace=self._subspace) 
                               for q in self._qubits]),
                        Barrier(self._qubits),
                        Cycle([X90(q, subspace=self._subspace) 
                               for q in self._qubits]),
                        Barrier(self._qubits),
                        Cycle([X90(q, subspace=self._subspace) 
                               for q in self._qubits]),
                        Barrier(self._qubits),
                        Cycle([Idle(q, duration=t/2) for q in self._qubits]),
                        Barrier(self._qubits),
                        Cycle([VirtualZ(phase/2, q, subspace=self._subspace) 
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
            self._circuits['time'] = self._times[self._qubits[0]]
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            pop = {'GE': '1', 'EF': '2'}
            # Fit the probability of being in 1 from the time sweep to an 
            # exponential
            for i, q in enumerate(self._qubits):
                prob1 = []
                for circuit in self._circuits:
                    prob1.append(
                        circuit.results.marginalize(i).populations[
                            pop[self._subspace]
                        ]
                    )
                self._results[q] = prob1

                # Add initial guesses to fit
                if self._echo:
                    c = np.array(prob1).min()
                    a = np.array(prob1).max() - c
                    b = np.mean( np.diff(prob1) / np.diff(self._times[q]) ) / a
                    self._fit[q].fit(self._times[q], prob1, p0=(-a, b, c))
                else:
                    e = np.array(prob1).min()
                    a = np.array(prob1).max() - e
                    b = -np.mean( np.diff(prob1) / np.diff(self._times[q]) )/a
                    c = self._detuning
                    d = 0.
                    self._fit[q].fit(self._times[q], prob1, p0=(a, b, c, d, e))

                # If the fit was successful, write to the config
                if self._fit[q].fit_success:
                    val, err = round_to_order_error(
                        *reciprocal_uncertainty(
                            self._fit[q].fit_params[1], self._fit[q].error[1]
                        )
                    )
                    self._char_values[q] = val
                    self._errors[q] = err

        def save(self):
            """Save all circuits and data."""
            qpu.save(self)
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._param_sweep]), 'param_sweep'
            )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._results]), 'sweep_results'
            )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._char_values]), 'T2_values'
            )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._errors]), 'T2_errors'
            )

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            clear_output(wait=True)
            if settings.Settings.save_data:
                self._data_manager._exp_id += (
                    f'_T2_Q{"".join(str(q) for q in self._qubits)}'
                )
                self.save()
                self.plot(
                    xlabel=r'Time ($\mu$s)',
                    ylabel=(
                    r'$|2\rangle$ Population' if self._subspace == 'EF' else
                    r'$|1\rangle$ Population'
                    ),
                    flabel=r'$T_{2E}$' if self._echo else r'$T_2$',
                    save_path=self._data_manager._save_path
                )
            self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

    return T2(
        config,
        qubits,
        t_max,
        detuning,
        echo,
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