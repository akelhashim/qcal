"""Submodule for readout benchmarking routines.

"""
import qcal.settings as settings

from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.gate.single_qubit import Id, Meas, X90, X
from qcal.qpu.qpu import QPU

import logging
import pandas as pd

from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


def ReadoutFidelity(
        qpu:             QPU,
        config:          Config,
        qubits:          List | Tuple,
        gate:            str = 'X90',
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None, 
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1, 
        n_levels:        int = 2
    ):
    """Function which passes a custom QPU to the ReadoutFidelity class.

    Basic example useage:

        ro = ReadoutFidelity(CustomQPU, cfg, [0, 1, 2])
        ro.run()

    Args:
        qpu (QPU): custom QPU class.
        config (Config): qcal config object.
        qubits (List | Tuple): qubits to measure.
        gate (str, optional): native gate used for state preparation. Defaults 
            to 'X90'.
        compiler (Any | Compiler | None, optional): a custom compiler to
            compile the experimental circuits. Defaults to None.
        transpiler (Any | None, optional): a custom transpiler to 
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

    Returns:
        class: ReadoutFidelity class.
    """

    class ReadoutFidelity(qpu):
        """ReadoutFidelity class.
        
        This class inherits a custom QPU from the ReadoutFidelity function.
        """

        def __init__(self, 
                config:          Config,
                qubits:          List | Tuple,
                gate:            str = 'X90',
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None, 
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1, 
                n_levels:        int = 2
            ) -> None:
            """Initialize the ReadoutFidelity class within the function.

            Args:
                config (Config): qcal config object.
                qubits (List | Tuple): qubits to measure.
                gate (str, optional): native gate used for state preparation.  
                    Defaults to 'X90'.
                compiler (Any | Compiler | None, optional): a custom compiler 
                    to compile the experimental circuits. Defaults to None.
                transpiler (Any | None, optional): a custom transpiler to 
                    transpile the experimental circuits. Defaults to None.
                n_shots (int, optional): number of measurements per circuit. 
                    Defaults to 1024.
                n_batches (int, optional): number of batches of measurements. 
                    Defaults to 1.
                n_circs_per_seq (int, optional): maximum number of circuits 
                    that can be measured per sequence. Defaults to 1.
                n_levels (int, optional): number of energy levels to be 
                    measured. Defaults to 2. If n_levels = 3, this assumes 
                    that the measurement supports qutrit classification.
            """
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
            assert gate in ('X90', 'X'), (
                'gate must be an X90 or X!'
            )
            self._gate = gate
            self._confusion_mat = None

        @property
        def confusion_matrix(self):
            """Confusion matrix for each qubit."""
            return self._confusion_mat

        def generate_circuits(self):
            """Generate the readout calibration circuits."""
            logger.info(' Generating circuits...')

            circuits = [
                Circuit([
                    Cycle([Id(q) for q in self._qubits]),
                    Barrier(),
                    Cycle([Meas(q) for q in qubits])
                ])
            ]
            
            level = {1: 'GE', 2: 'EF'}
            for n in range(1, self._n_levels):

                if self._gate == 'X90':
                    circuits.append(
                        Circuit([
                            Cycle(
                              [X90(q, subspace=level[n]) for q in self._qubits]
                            ),
                            Barrier(),
                            Cycle(
                              [X90(q, subspace=level[n]) for q in self._qubits]
                            ),
                            Barrier(),
                            Cycle([Meas(q) for q in qubits])
                        ])
                    )

                elif self._gate == 'X':
                    circuits.append(
                        Circuit([
                            Cycle(
                                [X(q, subspace=level[n]) for q in self._qubits]
                            ),
                            Barrier(),
                            Cycle([Meas(q) for q in qubits])
                        ])
                    )

                self._circuits = CircuitSet(circuits)
                self._circuits._df['Prep state'] = [
                    n for n in range(self._n_levels)
                ]

        def analyze(self):
            """Analyze the data and generate confusion matrices."""
            logger.info(' Analyzing the data...')

            index = [
                ['Prep State'] * self._n_levels,
                [n for n in range(self._n_levels)]
            ]
            columns = [[], [], []]
            for q in self._qubits:
                columns[0].extend([f'Q{q}'] * self._n_levels)
                columns[1].extend(['Meas State'] * self._n_levels)
                columns[2].extend([n for n in range(self._n_levels)])

            self._confusion_mat = pd.DataFrame(
                columns=columns, index=index
            )

            for i, circuit in enumerate(self._circuits):  # i = prep state
                for j, q in enumerate(self._qubits):  # j, q = idx, qubit
                    self._confusion_mat.iloc[i].loc[f'Q{q}'][
                        'Meas State'] = (
                        circuit.results.marginalize(j).probabilities
                    )

        def save(self):
            """Save all circuits and data."""
            super().save()
            self._data_manager.save(
                self._confusion_mat, 'confusion_matrix'
            )

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            super().run(self._circuits)
            self.analyze()
            if settings.Settings.save_data:
                self.save()


    return ReadoutFidelity(
        config,
        qubits,
        gate,
        compiler, 
        transpiler, 
        n_shots, 
        n_batches, 
        n_circs_per_seq, 
        n_levels
    )