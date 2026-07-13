""""Submodule for qubit coherence experiments.

"""
import logging
from typing import Callable, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from lmfit import Parameters
from numpy.typing import NDArray

import qcal.settings as settings
from qcal.calibration.utils import find_pulse_index
from qcal.characterization.characterize import Characterize
from qcal.circuit import Barrier, Circuit, CircuitSet, Cycle
from qcal.config import Config
from qcal.fitting.fit import (
    FitCosine,
    FitDecayingCosine,
    FitDecayingCosineExponential,
    FitExponential,
)
from qcal.gate.single_qubit import X90, Idle, Rz, X, Z
from qcal.math.utils import (
    reciprocal_uncertainty,
    round_to_order_error,
    uncertainty_of_sum,
)
from qcal.qpu.qpu import QPU
from qcal.sequence.dynamical_decoupling import DD_SEQUENCES
from qcal.units import kHz, us

logger = logging.getLogger(__name__)


__all__ = ['T1', 'T2', 'T2DD', 'ParityOscillations']


def T1(
    qpu:        QPU,
    config:     Config,
    qubits:     Sequence[int],
    t_max:      float = 500*us,
    gate:       str = 'X90',
    subspace:   str = 'GE',
    n_elements: int = 50,
    mapto0:     bool = True,
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
        qubits (Sequence[int]): qubits to measure.
        t_max (float, option): maximum wait time. Defaults to 500 us.
        gate (str, optional): native gate used for state preparation.
            Defaults to 'X90'.
        subspace (str, optional): qubit subspace for T1 measurement.
            Defaults to 'GE'.
        n_elements (int, optional): number of delays starting from 0
            to t_max. Defaults to 50.
        mapto0 (bool, optional): map EF measurements back to the
            computational subspace by appending EF and GE pi pulses,
            so that a |2> outcome is read as |0>. Defaults to True.

    Returns:
        Callable: T1 class.
    """

    class T1(qpu, Characterize):
        """T1 characterization class.

        This class inherits a custom QPU from the T1 characterization
        function.
        """

        def __init__(
            self,
            config:     Config,
            qubits:     Sequence[int],
            t_max:      float = 500*us,
            gate:       str = 'X90',
            subspace:   str = 'GE',
            n_elements: int = 50,
            mapto0:     bool = True,
            **kwargs
        ) -> None:
            """Initialize the T1 experiment class within the function."""

            n_levels = 3 if subspace == 'EF' else 2
            qpu.__init__(self, config=config, n_levels=n_levels, **kwargs)
            Characterize.__init__(self, config)

            self._qubits = qubits

            if gate not in ('X90', 'X'):
                raise ValueError(
                    f"'gate' must be one of 'X90' or 'X', not {gate}!"
                )
            self._gate = gate

            if subspace not in ('GE', 'EF'):
                raise ValueError(
                    f"'subspace' must be one of 'GE' or 'EF', not {subspace}!"
                )
            self._subspace = subspace
            self._mapto0 = mapto0

            self._times = {
                q: np.concatenate([
                    [0.], np.geomspace(
                        t_max/n_elements/2, t_max, n_elements - 1
                    )
                ]) for q in qubits
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

                if self._subspace == 'GE' and self._mapto0:
                    circuit.extend([
                        Barrier(self._qubits),
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                    ])
                elif self._subspace == 'EF' and self._mapto0:
                    circuit.extend([
                        Barrier(self._qubits),
                        Cycle({X90(q, subspace='EF') for q in self._qubits}),
                        Cycle({X90(q, subspace='EF') for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                    ])

                circuit.measure()

                circuits.append(circuit)

            self._circuits = CircuitSet(circuits=circuits)
            self._circuits['time'] = self._times[self._qubits[0]]

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            ge_pop = '0' if self._mapto0 else '1'
            ef_pop = '0' if self._mapto0 else '2'
            pop = {'GE': ge_pop, 'EF': ef_pop}
            # Fit the probability of being in 0 (2) from the time sweep to an
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
                self._circuits[f'Q{q}: Prob({pop[self._subspace]})'] = prob1

                # Add initial guesses to fit
                c = np.array(prob1).min()
                a = np.array(prob1).max() - c
                b = -np.mean( np.diff(prob1) / np.diff(self._times[q]) ) / a
                params = Parameters()
                params.add('a', value=a)
                params.add('b', value=b)
                params.add('c', value=c)
                weights = np.full(
                    len(self._times[q]), 1.0 / np.sqrt(self._n_shots),
                )
                self._fit[q].fit(
                    self._times[q], prob1, params=params, weights=weights
                )

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
            if self._subspace == 'GE':
                ylabel = (
                    r'$|0\rangle$ Population' if self._mapto0
                    else r'$|1\rangle$ Population'
                )
            elif self._subspace == 'EF':
                ylabel = (
                    r'$|0\rangle$ Population' if self._mapto0
                    else r'$|2\rangle$ Population'
                )

            Characterize.plot(self,
                xlabel=r'Time ($\mu$s)',
                ylabel=ylabel,
                flabel=rf'$T_{{1,{self._subspace}}}$',
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
        config=config,
        qubits=qubits,
        t_max=t_max,
        gate=gate,
        subspace=subspace,
        n_elements=n_elements,
        mapto0=mapto0,
        **kwargs
    )


def T2(
    qpu:        QPU,
    config:     Config,
    qubits:     Sequence[int],
    t_max:      float = 250*us,
    detuning:   float = 100 * kHz,
    echo:       bool = False,
    subspace:   str = 'GE',
    n_elements: int = 50,
    mapto0:     bool = True,
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
        qubits (Sequence[int]): qubits to measure.
        t_max (float, optional): maximum wait time. Defaults to 250 us.
        detuning (float, optional): artificial detuning from the actual
            qubit frequency. Defaults to 100 * kHz.
        echo (bool, optional): whether to echo the qubit in the middle.
            Defaults to False.
        subspace (str, optional): qubit subspace for T2 measurement.
            Defaults to 'GE'.
        n_elements (int, optional): number of delays starting from 0
            to t_max. Defaults to 50.
        mapto0 (bool, optional): map EF measurements back to the
            computational subspace by appending EF and GE pi pulses,
            so that a |2> outcome is read as |0>. Defaults to True.

    Returns:
        Callable: T2 class.
    """

    class T2(qpu, Characterize):
        """T2 characterization class.

        This class inherits a custom QPU from the T2 characterization
        function.
        """

        def __init__(
            self,
            config:     Config,
            qubits:     Sequence[int],
            t_max:      float = 250*us,
            detuning:   float = 100 * kHz,
            echo:       bool = False,
            subspace:   str = 'GE',
            n_elements: int = 50,
            mapto0:     bool = True,
            **kwargs
        ) -> None:
            """Initialize the T2 experiment class within the function."""

            n_levels = 3 if subspace == 'EF' else 2
            qpu.__init__(self, config=config, n_levels=n_levels, **kwargs)
            Characterize.__init__(self, config)

            self._qubits = qubits
            self._echo = echo
            self._detuning = detuning

            if subspace not in ('GE', 'EF'):
                raise ValueError(
                    f"'subspace' must be one of 'GE' or 'EF', not {subspace}!"
                )
            self._subspace = subspace
            self._mapto0 = mapto0

            if echo:
                t_start = t_max / n_elements / 2
            else:
                t_start = 1 / (8 * detuning)
            self._times = {
                q: np.concatenate([
                    [0.], np.geomspace(t_start, t_max, n_elements - 1)
                ]) for q in qubits
            }
            self._param_sweep = self._times

            if not echo:
                self._params = {
                    q: f'single_qubit/{q}/{subspace}/T2*' for q in qubits
                }
                self._fit = {
                    q: (
                        FitDecayingCosineExponential()
                        if subspace == 'EF' else FitDecayingCosine()
                    )
                    for q in qubits
                }
            elif echo: # TODO: fit double exponential for EF subspace
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
                        Cycle({Rz(q, phase, subspace=self._subspace)
                               for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle(  # maps t=0 back to |0> (|1>) for GE (EF)
                            {Z(q, subspace=self._subspace)
                             for q in self._qubits}
                        ),
                    ])
                elif self._echo:
                    circuit.extend([
                        Cycle({Idle(q, duration=t/2) for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({X90(q, subspace=self._subspace)
                               for q in self._qubits}),
                        Cycle({X90(q, subspace=self._subspace)
                               for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({Idle(q, duration=t/2) for q in self._qubits}),
                        Barrier(self._qubits),
                    ])

                # Basis preparation
                circuit.extend([
                    Cycle(
                        {X90(q, subspace=self._subspace) for q in self._qubits}
                    ),
                    Barrier(self._qubits)
                ])

                if self._subspace == 'EF' and self._mapto0:
                    circuit.extend([
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                        Barrier(self._qubits),
                    ])

                circuit.measure()
                circuits.append(circuit)

            self._circuits = CircuitSet(circuits=circuits)
            self._circuits['time'] = self._times[self._qubits[0]]
            if not self._echo:
                self._circuits['phase'] = (
                    2. * np.pi * self._detuning * self._times[self._qubits[0]]
                )

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            ef_pop = '0' if self._mapto0 else '1'
            pop = {'GE': '0', 'EF': ef_pop}
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
                self._circuits[f'Q{q}: Prob({pop[self._subspace]})'] = prob0

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
                    if self._subspace == 'GE':
                        e = np.array(prob0).min()
                        a = np.array(prob0).max() - e
                        b = np.mean(
                            np.diff(prob0) / np.diff(self._times[q])
                        ) / a
                        params = Parameters()
                        params.add('a', value=a)
                        params.add('b', value=b)
                        params.add('c', value=self._detuning)
                        params.add('d', value=0.)
                        params.add('e', value=e)
                    elif self._subspace == 'EF':
                        # a*exp(-b*x)*cos(2pi*c*x+d) + e*exp(-f*x) + g
                        g = np.array(prob0).min()
                        a = (np.array(prob0).max() - g) / 2
                        b = np.mean(
                            np.diff(prob0) / np.diff(self._times[q])
                        ) / a
                        e = np.array(prob0).max() - a - g
                        params = Parameters()
                        params.add('a', value=a)
                        params.add('b', value=b)
                        params.add('c', value=self._detuning)
                        params.add('d', value=0.)
                        params.add('e', value=e)
                        params.add('f', value=b)
                        params.add('g', value=g)

                weights = np.full(
                    len(self._times[q]), 1.0 / np.sqrt(self._n_shots),
                )
                self._fit[q].fit(
                    self._times[q], prob0, params=params, weights=weights
                )

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
            if self._subspace == 'EF' and not self._mapto0:
                ylabel = r'$|1\rangle$ Population'
            else:
                ylabel = r'$|0\rangle$ Population'

            Characterize.plot(self,
                xlabel=r'Time ($\mu$s)',
                ylabel=ylabel,
                flabel=(
                    rf'$T_{{2E,{self._subspace}}}$' if self._echo
                    else rf'$T_{{2,{self._subspace}}}$'
                ),
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
        config=config,
        qubits=qubits,
        t_max=t_max,
        detuning=detuning,
        echo=echo,
        subspace=subspace,
        n_elements=n_elements,
        mapto0=mapto0,
        **kwargs
    )


def T2DD(
    qpu:        QPU,
    config:     Config,
    qubits:     Sequence[int],
    t_max:      float = 250*us,
    gate_time:  float | None = None,
    subspace:   str = 'GE',
    dd_method:  str = 'XY_N',
    n_elements: int = 50,
    n_pulses:   int | None = None,
    mapto0:     bool = True,
    **kwargs
) -> Callable:
    """T2 dynamical decoupling coherence characterization.

    Basic example useage:
    ```
    exp = T2DD(
        CustomQPU,
        config,
        qubits=[0, 1, 2]
    )
    exp.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (Sequence[int]): qubits to measure.
        t_max (float, optional): maximum wait time. Defaults to 250 us.
        gate_time (float | None, optional): X90 gate time used to
            compute DD idle intervals. If None, the value is looked up
            from the config. Defaults to None.
        subspace (str, optional): qubit subspace for T2 measurement.
            Defaults to 'GE'.
        dd_method (str, optional): the DD method to use.
            Defaults to 'XY_N'.
        n_elements (int, optional): number of delays starting from 0
            to t_max. Defaults to 50.
        n_pulses (int | None, optional): number of pulses. Defaults to
            None. For example, for n_pulses = 4, this performs an XY4
            DD sequence. If None, the number of pulses will be
            determined by the total idle time.
        mapto0 (bool, optional): map EF measurements back to the
            computational subspace by appending EF and GE pi pulses,
            so that a |2> outcome is read as |0>. Defaults to True.

    Returns:
        Callable: T2DD class.
    """

    class T2DD(qpu, Characterize):
        """T2DD characterization class.

        This class inherits a custom QPU from the T2DD characterization
        function.
        """

        def __init__(
            self,
            config:     Config,
            qubits:     Sequence[int],
            t_max:      float = 250*us,
            gate_time:  float | None = None,
            echo:       bool = False,
            subspace:   str = 'GE',
            dd_method:  str = 'XY_N',
            n_elements: int = 50,
            n_pulses:   int | None = None,
            mapto0:     bool = True,
            **kwargs
        ) -> None:
            """Initialize the T2 XY experiment class within the function."""

            n_levels = 3 if subspace == 'EF' else 2
            qpu.__init__(self, config=config, n_levels=n_levels, **kwargs)
            Characterize.__init__(self, config)

            self._qubits = qubits
            self._echo = echo
            self._n_pulses = n_pulses
            self._gate_time = gate_time

            if subspace not in ('GE', 'EF'):
                raise ValueError(
                    f"'subspace' must be one of 'GE' or 'EF', not {subspace}!"
                )
            self._subspace = subspace
            self._mapto0 = mapto0

            self._dd_method = dd_method

            self._times = {
                q: np.concatenate([
                    [0.], np.geomspace(
                        t_max/n_elements/2, t_max, n_elements - 1
                    )
                ]) for q in qubits
            }
            self._param_sweep = self._times

            self._params = {
                q: f'single_qubit/{q}/{subspace}/T2DD' for q in qubits
            }
            # TODO: fit double exponential for EF subspace
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

            if self._gate_time is not None:
                gate_time = self._gate_time
            else:
                def _gate_time(q):
                    prefix = f'single_qubit/{q}/{self._subspace}/X90/pulse'
                    idx = find_pulse_index(self._config, prefix)
                    return self._config[f'{prefix}/{idx}/time']

                gate_time = max(_gate_time(q) for q in self._qubits)

            circuits = []
            for t in self._times[self._qubits[0]]:
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
                DD_sequence = DD_SEQUENCES[self._dd_method](
                    qubits=self._qubits,
                    total_time=t,
                    gate_time=gate_time,
                    n_pulses=self._n_pulses,
                    subspace=self._subspace,
                )
                circuit.extend(DD_sequence)

                # Basis preparation
                circuit.extend([
                    Cycle({
                        Z(q, subspace=self._subspace) for q in self._qubits
                    }),
                    Cycle({
                        X90(q, subspace=self._subspace) for q in self._qubits
                    }),
                    Barrier(self._qubits),
                ])

                if self._subspace == 'EF' and self._mapto0:
                    circuit.extend([
                        # Cycle({X90(q, subspace='EF') for q in self._qubits}),
                        # Cycle({X90(q, subspace='EF') for q in self._qubits}),
                        # Barrier(self._qubits),
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                        Barrier(self._qubits),
                    ])

                circuit.measure()
                circuits.append(circuit)

            self._circuits = CircuitSet(circuits=circuits)
            self._circuits['time'] = self._times[self._qubits[0]]

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            ef_pop = '0' if self._mapto0 else '1'
            pop = {'GE': '0', 'EF': ef_pop}
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
                self._circuits[f'Q{q}: Prob({pop[self._subspace]})'] = prob0

                c = np.array(prob0).min()
                a = np.array(prob0).max() - c
                b = -np.mean( np.diff(prob0) / np.diff(self._times[q]) ) / a
                params = Parameters()
                params.add('a', value=-a)
                params.add('b', value=b)
                params.add('c', value=c)
                weights = np.full(
                    len(self._times[q]), 1.0 / np.sqrt(self._n_shots),
                )
                self._fit[q].fit(
                    self._times[q], prob0, params=params, weights=weights
                )

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
                f'_T2DD_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._char_values]), 'T2DD_values'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._errors]), 'T2DD_errors'
                )

        def plot(self):
            """Plot the data."""
            if self._subspace == 'EF' and not self._mapto0:
                ylabel = r'$|1\rangle$ Population'
            else:
                ylabel = r'$|0\rangle$ Population'

            Characterize.plot(self,
                xlabel=r'Time ($\mu$s)',
                ylabel=ylabel,
                flabel=rf'$T_{{2DD,{self._subspace}}}$',
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

    return T2DD(
        config=config,
        qubits=qubits,
        t_max=t_max,
        gate_time=gate_time,
        subspace=subspace,
        dd_method=dd_method,
        n_elements=n_elements,
        n_pulses=n_pulses,
        mapto0=mapto0,
        **kwargs
    )


def ParityOscillations(
    qpu:        QPU,
    config:     Config,
    circuit:    Circuit,
    qubits:     Sequence[int] | None = None,
    n_elements: int = 31,
    **kwargs
) -> Callable:
    """Parity oscillations coherence characterization.

    See: https://arxiv.org/abs/2112.14589

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        cicuit (Circuit): qcal Circuit.
        qubits (Sequence[int] | None): qubits to measure. Defaults to None.
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

        def __init__(
            self,
            config:     Config,
            circuit:    Circuit,
            qubits:     Sequence[int] | None = None,
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
        config=config,
        circuit=circuit,
        qubits=qubits,
        n_elements=n_elements,
        **kwargs
    )
