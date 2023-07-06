"""Submodule for single-qubit gate calibration.

"""
from __future__ import annotations

import qcal.settings as settings
from qcal.calibration.utils import find_pulse_index
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.gate.single_qubit import X90, X
from qcal.qpu.qpu import QPU

import logging
import pandas as pd

from typing import Any, List, Tuple
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


def Amplitude(
        qpu:             QPU,
        config:          Config,
        qubits:          List | Tuple,
        amplitudes:      ArrayLike | NDArray,
        gate:            str = 'X90',
        subspace:        str = 'GE',
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None, 
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1, 
        n_levels:        int = 2,
        n_gates:         int = 1,
        relative_amp:    bool = False
    ) -> Amplitude:
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
        amplitudes (ArrayLike | NDArray): array of gate amplitudes to sweep
            over for calibrating the single-qubit gate amplitude.
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

    Returns:
        Amplitude: Amplitude class.
    """

    class Amplitude(qpu):
        """Amplitude calibration class.
        
        This class inherits a custom QPU from the Amplitude calibration
        function.
        """

        def __init__(self, 
                config:          Config,
                qubits:          List | Tuple,
                amplitudes:      ArrayLike | NDArray,
                gate:            str = 'X90',
                subspace:        str = 'GE',
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None, 
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1, 
                n_levels:        int = 2,
                n_gates:         int = 1,
                relative_amp:    bool = False
            ) -> None:
            """Initialize the Amplitude calibration class within the function.

            Args:
                config (Config): qcal Config object.
                qubits (List | Tuple): qubits to calibrate.
                amplitudes (ArrayLike | NDArray): array of gate amplitudes to 
                    sweep over for calibrating the single-qubit gate amplitude.
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
                n_circs_per_seq (int, optional): maximum number of circuits 
                    that can be measured per sequence. Defaults to 1.
                n_levels (int, optional): number of energy levels to be  
                    measured. Defaults to 2. If n_levels = 3, this assumes that 
                    the measurement supports qutrit classification.
                n_gates (int, optional): number of gates for pulse repetition.
                    Defaults to 1.
                relative_amp (bool, optional): whether or not the amplitudes 
                    argument is defined relative to the existing pulse 
                    amplitude. Defaults to False. If true, the amplitudes are 
                    swept over the current amplitude times the amplitudes 
                    argument.
            """
            if subspace == 'EF':
                assert n_levels == 3, 'n_levels must be 3 for EF calibration!'
            
            super().__init__(
                config, 
                compiler, 
                transpiler, 
                n_shots, 
                n_batches, 
                n_circs_per_seq, 
                n_levels
            )
            self._qubits = qubits
            
            assert gate in ('X90', 'X'), 'gate must be an X90 or X!'
            self._gate = gate

            assert subspace in ('GE', 'EF'), 'subspace must be GE or EF!'
            self._subspace = subspace

            self._amplitudes = amplitudes
            self._relative_amp = relative_amp

            if n_gates > 1 and gate == 'X90':
                assert n_gates % 4 == 0, 'n_gates must be a multiple of 4!'
            elif n_gates > 1 and gate == 'X':
                assert n_gates % 2 == 0, 'n_gates must be a multiple of 2!'
            self._n_gates = n_gates

        def generate_circuits(self):
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            gate = {'X90': X90, 'X': X}
            circuit = Circuit()
            for _ in range(self._n_gates):
                circuit.extend([
                    Cycle([gate[self._gate](q) for q in self._qubits]),
                    Barrier(self._qubits)
                ])
            circuit.measure()
            
            circuits = []
            for _ in range(self._amplitudes.size):
                circuits.append(circuit.copy())

            self._circuits = CircuitSet(circuits=circuits)
            for q in self._qubits:
                param = f'single_qubit/{q}/{self._subspace}/{self._gate}/pulse'
                param += f'/{find_pulse_index(self._config, param)}/'
                param += 'kwargs/amp'
                self._circuits[f'param: {param}'] = self._amplitudes

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            super().run(self._circuits)
            # self.analyze()
            # if settings.Settings.save_data:
            #     self._data_manager._exp_id += '_readout_fidelity'
            #     self.save()


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
        relative_amp
    )