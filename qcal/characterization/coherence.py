""""Submodule for qubit coherence experiments.

"""
import qcal.settings as settings

from qcal.characterization.characterize import Characterize
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.config import Config
from qcal.fitting.fit import FitCosine, FitDecayingCosine, FitExponential
from qcal.gate.single_qubit import Idle, Rz, VirtualZ, X90, X, Z
from qcal.math.utils import (
    uncertainty_of_sum, reciprocal_uncertainty, round_to_order_error
)
from qcal.sequence.dynamical_decoupling import XY
from qcal.qpu.qpu import QPU
from qcal.units import MHz, us

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.display import clear_output
from lmfit import Parameters
from numpy.typing import NDArray
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


__all__ = ['T1', 'T2', 'T2XY', 'ParityOscillations']


def T1(qpu:        QPU,
       config:     Config,
       qubits:     List[int] | Tuple[int],
       t_max:      float = 500*us,
       gate:       str = 'X90',
       subspace:   str = 'GE',
       n_elements: int = 50,
       **kwargs
    ) -> Callable:
    """T1 coherence characterization.

    Basic example useage:
    ```
    exp = T1(
        CustomQPU, 
        config, 
        qubits=[0, 1, 2],
        t_max=5e-4)
    exp.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List[int] | Tuple[int]): qubits to measure.
        t_max (float, option): maximum wait time. Defaults to 500 us.
        gate (str, optional): native gate used for state preparation. Defaults 
            to 'X90'.
        subspace (str, optional): qubit subspace for T1 measurement.
            Defaults to 'GE'.
        n_elements (int, optional): number of delays starting from 0 to t_max.
            Defaults to 50.

    Returns:
        Callable: T1 class.
    """

    class T1(qpu, Characterize):
        """T1 characterization class.
        
        This class inherits a custom QPU from the T1 characterization
        function.
        """

        def __init__(self, 
                config:     Config,
                qubits:     List | Tuple,
                t_max:      float = 500*us,
                gate:       str = 'X90',
                subspace:   str = 'GE',
                n_elements: int = 50,
                **kwargs
            ) -> None:
            """Initialize the T1 experiment class within the function."""

            n_levels = 3 if subspace == 'EF' else 2
            qpu.__init__(self, config=config, n_levels=n_levels, **kwargs)
            Characterize.__init__(self, config)

            self._qubits = qubits
            
            assert gate in ('X90', 'X'), (
                "'gate' must be one of 'X90' or 'X'!"
            )
            self._gate = gate

            assert subspace in ('GE', 'EF'), (
                "'subspace' must be one of 'GE' or 'EF'!"
            )
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
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                        Barrier(self._qubits)
                    ])

                    if self._subspace == 'EF':
                        circuit.extend([
                            Cycle({X90(q, subspace='EF') 
                                   for q in self._qubits}),
                            Barrier(self._qubits),
                            Cycle({X90(q, subspace='EF') 
                                   for q in self._qubits}),
                            Barrier(self._qubits)
                        ])

                elif self._gate == 'X':
                    circuit.extend([
                        Cycle({X(q, subspace='GE') for q in self._qubits}),
                        Barrier(self._qubits)
                    ])

                    if self._subspace == 'EF':
                        circuit.extend([
                            Cycle({X(q, subspace='EF') for q in self._qubits}),
                            Barrier(self._qubits),
                        ])

                # T1 delay
                circuit.append(
                    Cycle({Idle(q, duration=t) for q in self._qubits}),
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
                self._circuits[f'Q{q}: Prob(1)'] = prob1

                # Add initial guesses to fit
                c = np.array(prob1).min()
                a = np.array(prob1).max() - c
                b = -np.mean( np.diff(prob1) / np.diff(self._times[q]) ) / a
                params = Parameters()
                params.add('a', value=a)  
                params.add('b', value=b)
                params.add('c', value=c)
                self._fit[q].fit(self._times[q], prob1, params=params)

                # If the fit was successful, write to the config
                if self._fit[q].fit_success:
                    val, err = round_to_order_error(
                        *reciprocal_uncertainty(
                            self._fit[q].fit_params['b'].value, 
                            self._fit[q].fit_params['b'].stderr
                        )
                    )
                    self._char_values[q] = val
                    self._errors[q] = err

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_T1_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._char_values]), 'T1_values'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._errors]), 'T1_errors'
                )

        def plot(self):
            """Plot the data."""
            Characterize.plot(self,
                xlabel=r'Time ($\mu$s)',
                ylabel=(
                    r'$|2\rangle$ Population' if self._subspace == 'EF' else
                    r'$|1\rangle$ Population'
                ),
                flabel=r'$T_1$',
                save_path=self._data_manager._save_path
            )

        def final(self):
            """Final experimental method."""
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

    return T1(
        config,
        qubits,
        t_max,
        gate,
        subspace,
        n_elements, 
        **kwargs
    )


def T2(qpu:        QPU,
       config:     Config,
       qubits:     List | Tuple,
       t_max:      float = 250*us,
       detuning:   float = 0.05 * MHz,
       echo:       bool = False,
       subspace:   str = 'GE',
       n_elements: int = 50,
       **kwargs
    ) -> Callable:
    """T2 coherence characterization.

    Basic example useage:

    ```
    exp = T2(
        CustomQPU, 
        config, 
        qubits=[0, 1, 2],
        t_max=250e-4,
        detuning=0.05e6,
        echo=True)
    exp.run()
    ```

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
        n_elements (int, optional): number of delays starting from 0 to t_max.
            Defaults to 50.

    Returns:
        Callable: T2 class.
    """

    class T2(qpu, Characterize):
        """T2 characterization class.
        
        This class inherits a custom QPU from the T2 characterization
        function.
        """

        def __init__(self, 
                config:     Config,
                qubits:     List | Tuple,
                t_max:      float = 250*us,
                detuning:   float = 0.05 * MHz,
                echo:       bool = False,
                subspace:   str = 'GE',
                n_elements: int = 50,
                **kwargs
            ) -> None:
            """Initialize the T2 experiment class within the function."""

            n_levels = 3 if subspace == 'EF' else 2
            qpu.__init__(self, config=config, n_levels=n_levels, **kwargs)
            Characterize.__init__(self, config)

            self._qubits = qubits
            self._echo = echo
            self._detuning = detuning

            assert subspace in ('GE', 'EF'), (
                "'subspace' must be one of 'GE' or 'EF'!"
            )
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

                # T2 experiment
                if not self._echo:
                    circuit.extend([
                        Cycle({Idle(q, duration=t) for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({VirtualZ(q, phase, subspace=self._subspace) 
                               for q in self._qubits}),
                        Barrier(self._qubits),
                    ])
                elif self._echo:
                    circuit.extend([
                        Cycle({Idle(q, duration=t/2) for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({VirtualZ(q, phase/2, subspace=self._subspace) 
                               for q in self._qubits}),
                        Barrier(self._qubits),
                        # Cycle({X(q, subspace=self._subspace) 
                        #        for q in self._qubits}),
                        Cycle({X90(q, subspace=self._subspace) 
                               for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({X90(q, subspace=self._subspace) 
                               for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({Idle(q, duration=t/2) for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({VirtualZ(q, phase/2, subspace=self._subspace) 
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
            self._circuits['time'] = self._times[self._qubits[0]]
            self._circuits['phase'] = (
                2. * np.pi * self._detuning * self._times[self._qubits[0]]
            )
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            pop = {'GE': '0', 'EF': '1'}
            # Fit the probability of being in 1 from the time sweep to an 
            # exponential
            for i, q in enumerate(self._qubits):
                prob0 = []
                for circuit in self._circuits:
                    prob0.append(
                        circuit.results.marginalize(i).populations[
                            pop[self._subspace]
                        ]
                    )
                self._results[q] = prob0
                self._circuits[f'Q{q}: Prob(0)'] = prob0

                # Add initial guesses to fit
                if self._echo:  # a * np.exp(-b * x) + c
                    c = np.array(prob0).min()
                    a = np.array(prob0).max() - c
                    b = -np.mean( np.diff(prob0) / np.diff(self._times[q]) ) / a
                    params = Parameters()
                    params.add('a', value=a)  
                    params.add('b', value=b)
                    params.add('c', value=c)
                else:  # a * np.exp(-b * x) * np.cos(2 * np.pi * c * x + d) + e
                    e = np.array(prob0).min()
                    a = np.array(prob0).max() - e
                    b = np.mean( np.diff(prob0) / np.diff(self._times[q]) ) / a
                    params = Parameters()
                    params.add('a', value=a)  
                    params.add('b', value=b)
                    params.add('c', value=self._detuning)
                    params.add('d', value=0.)
                    params.add('e', value=e)
                
                self._fit[q].fit(self._times[q], prob0, params=params)

                # If the fit was successful, write to the config
                if self._fit[q].fit_success:
                    val, err = round_to_order_error(
                        *reciprocal_uncertainty(
                            self._fit[q].fit_params['b'].value, 
                            self._fit[q].fit_params['b'].stderr
                        )
                    )
                    self._char_values[q] = val
                    self._errors[q] = err

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_T2_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._char_values]), 'T2_values'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._errors]), 'T2_errors'
                )

        def plot(self):
            """Plot the data."""
            Characterize.plot(self,
                xlabel=r'Time ($\mu$s)',
                ylabel=(
                    r'$|1\rangle$ Population' if self._subspace == 'EF' else
                    r'$|0\rangle$ Population'
                ),
                flabel=r'$T_{2E}$' if self._echo else r'$T_2$',
                save_path=self._data_manager._save_path
            )

        def final(self):
            """Final experimental method."""
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

    return T2(
        config,
        qubits,
        t_max,
        detuning,
        echo,
        subspace,
        n_elements, 
        **kwargs
    )


def T2XY(
        qpu:        QPU,
        config:     Config,
        qubits:     List | Tuple,
        t_max:      float = 250*us,
        detuning:   float = 0.05 * MHz,
        subspace:   str = 'GE',
        n_elements: int = 50,
        n_pulses:   int | None = None,
        **kwargs
    ) -> Callable:
    """T2 XY coherence characterization.

    Basic example useage:
    ```
    exp = T2XY(
        CustomQPU, 
        config, 
        qubits=[0, 1, 2],
        t_max=250e-4,
        detuning=0.05e6
    )
    exp.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List | Tuple): qubits to measure.
        t_max (float, optional): maximum wait time. Defaults to 250 us.
        detuning (float, optional): artificial detuning from the actual qubit
            frequency. Defaults to 0.05 MHz.
        subspace (str, optional): qubit subspace for T2 measurement.
            Defaults to 'GE'.
        n_elements (int, optional): number of delays starting from 0 to t_max.
            Defaults to 50.
        n_pulses (int | None, optional): number of pulses. Defaults to None. 
            For example, for n_pulses = 4, this performs an XY4 DD sequence. If
            None, the number of pulses will be determined by the total idle
            time.

    Returns:
        Callable: T2XY class.
    """

    class T2XY(qpu, Characterize):
        """T2XY characterization class.
        
        This class inherits a custom QPU from the T2XY characterization
        function.
        """

        def __init__(self, 
                config:     Config,
                qubits:     List | Tuple,
                t_max:      float = 250*us,
                detuning:   float = 0.05 * MHz,
                echo:       bool = False,
                subspace:   str = 'GE',
                n_elements: int = 50,
                n_pulses:   int | None = None,
                **kwargs
            ) -> None:
            """Initialize the T2 XY experiment class within the function."""

            n_levels = 3 if subspace == 'EF' else 2
            qpu.__init__(self, config=config, n_levels=n_levels, **kwargs)
            Characterize.__init__(self, config)

            self._qubits = qubits
            self._echo = echo
            self._detuning = detuning
            self._n_pulses = n_pulses

            assert subspace in ('GE', 'EF'), (
                "'subspace' must be one of 'GE' or 'EF'!"
            )
            self._subspace = subspace

            self._times = {
                q: np.linspace(0., t_max, n_elements) for q in qubits
            }
            self._param_sweep = self._times

            self._params = {
                q: f'single_qubit/{q}/{subspace}/T2DD' for q in qubits
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

                # Add the XY DD sequence
                DD_sequence = XY(
                    config=self._config, 
                    qubits=self._qubits, 
                    time=t, 
                    n_pulses=self._n_pulses, 
                    phase=phase,
                    subspace=self._subspace
                )
                circuit.extend(DD_sequence)

                # if self._n_pulses is None:
                #     n_pulses = int(DD_sequence.circuit_depth / 13) * 4
                # else:
                #     n_pulses = self._n_pulses

                # # Add in the virtual Z gates
                # # Pre-pulse + first idle
                # i = 3 if self._subspace == 'GE' else 5  # depth incrementer
                # for n in range(2 * n_pulses):
                #     circuit.insert(
                #         Cycle({
                #             VirtualZ(
                #                 q, 
                #                 phase/(2*n_pulses), 
                #                 subspace=self._subspace
                #             ) for q in self._qubits
                #         }),
                #         idx=i+n
                #     )
                #     i = (i + 2) if (n % 2 == 0) else i + 1
                
                # Basis preparation
                circuit.extend([
                    Cycle({
                        Z(q, subspace=self._subspace) for q in self._qubits
                    }),
                    Cycle({
                        X90(q, subspace=self._subspace) for q in self._qubits
                    })
                ])
                
                circuit.measure()
                circuits.append(circuit)
            
            self._circuits = CircuitSet(circuits=circuits)
            self._circuits['time'] = self._times[self._qubits[0]]
            self._circuits['phase'] = (
                2. * np.pi * self._detuning * self._times[self._qubits[0]]
            )
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            pop = {'GE': '0', 'EF': '1'}
            # Fit the probability of being in 0 from the time sweep to an 
            # exponential
            for i, q in enumerate(self._qubits):
                prob0 = []
                for circuit in self._circuits:
                    prob0.append(
                        circuit.results.marginalize(i).populations[
                            pop[self._subspace]
                        ]
                    )
                self._results[q] = prob0
                self._circuits[f'Q{q}: Prob(0)'] = prob0

                c = np.array(prob0).min()
                a = np.array(prob0).max() - c
                b = -np.mean( np.diff(prob0) / np.diff(self._times[q]) ) / a
                params = Parameters()
                params.add('a', value=-a)  
                params.add('b', value=b)
                params.add('c', value=c)
                self._fit[q].fit(self._times[q], prob0, params=params)

                # If the fit was successful, write to the config
                if self._fit[q].fit_success:
                    val, err = round_to_order_error(
                        *reciprocal_uncertainty(
                            self._fit[q].fit_params['b'].value, 
                            self._fit[q].fit_params['b'].stderr
                        )
                    )
                    self._char_values[q] = val
                    self._errors[q] = err

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_T2XY_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._char_values]), 'T2XY_values'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._errors]), 'T2XY_errors'
                )

        def plot(self):
            """Plot the data."""
            Characterize.plot(self,
                xlabel=r'Time ($\mu$s)',
                ylabel=(
                r'$|2\rangle$ Population' if self._subspace == 'EF' else
                r'$|1\rangle$ Population'
                ),
                flabel=r'$T_{2XY}$',
                save_path=self._data_manager._save_path
            )

        def final(self):
            """Final experimental method."""
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

    return T2XY(
        config=config,
        qubits=qubits,
        t_max=t_max,
        detuning=detuning,
        subspace=subspace,
        n_elements=n_elements, 
        n_pulses=n_pulses,
        **kwargs
    )


def ParityOscillations(
        qpu:        QPU,
        config:     Config,
        circuit:    Circuit,
        qubits:     List[int] | Tuple[int] = None,
        n_elements: int = 31,
        **kwargs
    ) -> Callable:
    """Parity oscillations coherence characterization.

    See: https://arxiv.org/abs/2112.14589

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        cicuit (Circuit): qcal Circuit.
        qubits (List[int] | Tuple[int]): qubits to measure. Defaults to None.
        n_elements (int, optional): number of phases between 0 and pi.
            Defaults to 31.

    Returns:
        Callable: ParityOscillations class.
    """

    class ParityOscillations(qpu, Characterize):
        """Parity oscillations characterization class.
        
        This class inherits a custom QPU from the ParityOscillations 
        characterization function.
        """

        def __init__(self, 
                config:     Config,
                circuit:    Circuit,
                qubits:     List[int] | Tuple[int] = None,
                n_elements: int = 31,
                **kwargs
            ) -> None:
            """Initialize the ParityOscillations class within the function."""

            qpu.__init__(self, config=config, **kwargs)
            Characterize.__init__(self, config)

            self._circuit = circuit
            self._qubits = qubits if qubits is not None else circuit.qubits

            self._circuits = CircuitSet()
            self._phases = np.linspace(0, np.pi, n_elements)
            self._evs = []
            self._fidelity = None
            self._fit = FitCosine()

        @property
        def evs(self) -> List:
            """Expectation values for each phase.

            Returns:
                List: expectation values.
            """
            return self._evs
        
        @property
        def fidelity(self) -> Dict:
            """Fidelity of the state.

            The fidelity is determined from the populations of the |0^n> and 
            |1^n> states, as well as the coherence of the state, which is
            determined from the amplitude of the parity oscillations. The error
            in the fidelity is determined from the shot noise for the
            populations and the error in the amplitude fit for the parity 
            oscillations.

            Returns:
                Dict: value and error (uncertainty) of the estimated fidelity.
            """
            return self._fidelity

        @property
        def phases(self) -> NDArray:
            """Phase sweep.

            Returns:
                NDArray: phases.
            """
            return self._phases
            
        def generate_circuits(self):
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            # Measure the state in the computational basis
            circuit = self._circuit.copy()
            circuit.measure(self._qubits)
            self._circuits.append(circuit)
            
            # Parity oscillations
            for phase in self._phases:
                circuit = self._circuit.copy()
                circuit.append(Barrier(self._qubits))
                circuit.append(Cycle({Rz(q, phase) for q in self._qubits}))
                circuit.append(Cycle({X90(q) for q in self._qubits}))
                circuit.measure(self._qubits)

                self._circuits.append(circuit)

            self._circuits['phase'] = [np.nan] + list(self._phases)
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            q_index = tuple([
                self._circuit.qubits.index(q) for q in self._qubits
            ])

            # Populations
            pop0 = self._circuits[0].results.marginalize(q_index).populations[
                '0' * len(self._qubits)
            ]
            pop1 = self._circuits[0].results.marginalize(q_index).populations[
                '1' * len(self._qubits)
            ]
            errors = [
                1/np.sqrt(
                    self._circuits[0].results.marginalize(q_index).n_shots
                )
            ] * 2

            # Parity
            for circuit in self._circuits[1:].circuit:
                results = circuit.results.marginalize(q_index)
                self._evs.append(results.ev)

            self._fit.fit(
                self._phases, 
                self._evs, 
                p0=(max(self._evs), 1.0/((len(self._qubits)-1) * np.pi), 0, 0)
            )
            assert self._fit.fit_success, 'Cosine fit was unsuccessful!'
            errors.append(self._fit.error[0])

            fidelity = (pop0 + pop1 + abs(self._fit.fit_params[0])) / 2
            error = uncertainty_of_sum(errors)
            fidelity, error = round_to_order_error(fidelity, error)
            self._fidelity = {
                'val': fidelity, 'err': error
            }

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
              f'_ParityOscillations_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._evs]), 'parity'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._fidelity]), 'fidelity'
                )

        def plot(self):
            """Plot the parity oscillations."""

            q_index = tuple([
                self._circuit.qubits.index(q) for q in self._qubits
            ])

            fig, ax = plt.subplots(1, 2, figsize=(10,4))

            ax[0].bar(
                [0, 1, 2, 3],
                list(
                   self._circuits[0].results.marginalize(q_index).probabilities
                ), 
                color='blue'
            )
            ax[0].set_xticks([0, 1, 2, 3])
            ax[0].set_xticklabels(
                self._circuits[0].results.marginalize(q_index).states
            )
            ax[0].set_ylabel('Probability', fontsize=15)
            ax[0].tick_params(axis='both', which='major', labelsize=12)

            ax[1].plot(self._phases, self._evs, 'o', ms=6, color='blue')
            ax[1].plot(
                self._phases, self._fit.predict(self._phases), color='k'
            )
            ax[1].set_ylabel('Parity', fontsize=15)
            ax[1].set_xlabel('Phase (rad.)', fontsize=15)
            ax[1].set_ylim((-1.1, 1.1))
            ax[1].set_yticks([-1, -0.5, 0, 0.5, 1.0])
            ax[1].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax[1].set_xticklabels(
                ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
            )
            ax[1].grid()
            ax[1].tick_params(axis='both', which='major', labelsize=12)

            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'parity_oscillations.png', 
                    dpi=300
                )
                fig.savefig(
                    self._data_manager._save_path + 'parity_oscillations.pdf', 
                )
                fig.savefig(
                    self._data_manager._save_path + 'parity_oscillations.svg', 
                )
            plt.show()

        def final(self):
            """Final experimental method."""
            fidelity = self._fidelity['val']
            error = self._fidelity['err']
            print(f'\nFidelity = {fidelity} ({error})')
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return ParityOscillations(
        config,
        circuit,
        qubits,
        n_elements,
        **kwargs
    )