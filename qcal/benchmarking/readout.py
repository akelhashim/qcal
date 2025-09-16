"""Submodule for readout benchmarking routines.

"""
from __future__ import annotations

import qcal.settings as settings
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.config import Config
from qcal.gate.single_qubit import Id, Meas, X90, X
from qcal.qpu.qpu import QPU

import logging
import pandas as pd

from IPython.display import clear_output
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)


def ReadoutFidelity(
        qpu:    QPU,
        config: Config,
        qubits: List | Tuple,
        gate:   str = 'X90',
        **kwargs
    ) -> Callable:
    """Function which passes a custom QPU to the ReadoutFidelity class.

    Basic example useage:

    ```
    ro = ReadoutFidelity(CustomQPU, config, [0, 1, 2])
    ro.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List | Tuple): qubits to measure.
        gate (str, optional): native gate used for state preparation. Defaults 
            to 'X90'.

    Returns:
        Callable: ReadoutFidelity class.
    """


    class ReadoutFidelity(qpu):
        """ReadoutFidelity class.
        
        This class inherits a custom QPU from the ReadoutFidelity function.
        """

        def __init__(self, 
                config: Config,
                qubits: List | Tuple,
                gate:   str = 'X90',
                **kwargs
            ) -> None:
            """Initialize the ReadoutFidelity class within the function.

            """
            qpu.__init__(self,
                config=config, 
                **kwargs
            )
            self._qubits = sorted(qubits)
            assert gate in ('X90', 'X'), 'gate must be an X90 or X!'
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
                    Barrier(self._qubits),
                    Cycle([Meas(q) for q in self._qubits])
                ])
            ]
            
            level = {1: 'GE', 2: 'EF'}
            for m in range(1, self._n_levels):
                circuit = Circuit()
                for n in range(1, m+1):

                    if self._gate == 'X90':
                        circuit.extend([
                            Cycle(
                              [X90(q, subspace=level[n]) for q in self._qubits]
                            ),
                            Barrier(self._qubits),
                            Cycle(
                              [X90(q, subspace=level[n]) for q in self._qubits]
                            ),
                            Barrier(self._qubits)
                        ])

                    elif self._gate == 'X':
                        circuit.extend([
                            Cycle(
                                [X(q, subspace=level[n]) for q in self._qubits]
                            ),
                            Barrier(self._qubits)
                        ])

                circuit.measure()
                circuits.append(circuit)

            self._circuits = CircuitSet(circuits)
            self._circuits['prep state'] = [n for n in range(self._n_levels)]

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
                    self._confusion_mat.loc[
                        ('Prep State', i), (f'Q{q}', 'Meas State')
                    ] = ([
                        circuit.results.marginalize(j).populations[f'{n}']
                        for n in range(self._n_levels)
                    ])

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_readout_fidelity_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    self._confusion_mat, 'confusion_matrix'
                )

        def final(self):
            """Final calibration method."""
            print(f"Runtime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.final()

    return ReadoutFidelity(
        config=config,
        qubits=qubits,
        gate=gate,
        **kwargs
    )