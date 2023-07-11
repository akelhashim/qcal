"""Submodule for single-qubit gate calibration.

"""
from __future__ import annotations

import qcal.settings as settings

from .calibration import Calibration
from .utils import find_pulse_index
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.fitting.fit import CosineFit, ParabolaFit
from qcal.gate.single_qubit import X90, X
from qcal.qpu.qpu import QPU

import logging
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
        n_levels:          int = 2,
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
        n_levels (int, optional): number of energy levels to be measured. 
            Defaults to 2. If n_levels = 3, this assumes that the
            measurement supports qutrit classification.
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
        Callable: Amplitude class.
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
                n_levels:          int = 2,
                n_gates:           int = 1,
                relative_amp:      bool = False,
                disable_esp:       bool = True,
                disable_heralding: bool = False,
                **kwargs
            ) -> None:
            """Initialize the Amplitude calibration class within the function.

            """
            if subspace == 'EF':
                assert n_levels == 3, 'n_levels must be 3 for EF calibration!'
            
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
                self._fit = {q: CosineFit() for q in qubits}
            elif n_gates > 1:
                self._fit = {q: ParabolaFit() for q in qubits}

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
                self._results[q] = prob0
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
                        if a > 0:
                            logger.warning(
                              f'Fit failed for qubit {q} (positive curvature)!'
                            )
                            self._fit[q]._fit_success = False
                        else:
                            newvalue = -b / (2*a)  # Assume c = 0
                            self._cal_values[q] = newvalue

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
                 pd.DataFrame([self._cal_values]), 'calibrated_values'
            )

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            clear_output(wait=True)
            if settings.Settings.save_data:
                self._data_manager._exp_id += '_amp_calibration'
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
        n_levels,
        n_gates,
        relative_amp,
        disable_esp,
        disable_heralding
    )