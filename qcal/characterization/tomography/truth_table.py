"""Submodule for Truth Table Tomography.

"""
import qcal.settings as settings

from qcal.characterization.characterize import Characterize
from qcal.circuit import Cycle, Circuit, CircuitSet
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.gate.single_qubit import Id, X90
from qcal.managers.classification_manager import ClassificationManager
from qcal.math.utils import (
    uncertainty_of_sum, round_to_order_error
)
from qcal.qpu.qpu import QPU

import itertools
import logging
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.display import clear_output
from numpy.typing import NDArray
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


def TruthTable(
        qpu:             QPU,
        config:          Config,
        circuit:         Circuit,
        qubits:          List | Tuple | None = None,
        ideal_unitary:   NDArray | None = None,
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        raster_circuits: bool = False,
        **kwargs
    ) -> Callable:
    """Truth Table tomography.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        cicuit (Circuit): qcal Circuit.
        qubits (List | Tuple | None): qubits to measure. Defaults to None.
        ideal_unitary (NDArray| None): unitary for ideal gate. Defaults to None.
            If not None, it will be used to compute the truth table fidelity.
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
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.

    Returns:
        Callable: TruthTable class.
    """

    class TruthTable(qpu, Characterize):
        """Truth Table tomography characterization class.
        
        This class inherits a custom QPU from the TruthTable
        characterization function.
        """

        def __init__(self, 
                config:          Config,
                circuit:         Circuit,
                qubits:          List | Tuple = None,
                ideal_unitary:   NDArray | None = None,
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2,
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            """Initialize the TruthTable class within the function."""

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
            Characterize.__init__(self, config)

            self._circuit = circuit
            self._qubits = qubits if qubits is not None else circuit.qubits
            self._ideal_unitary = ideal_unitary

            self._circuits = CircuitSet()
            self._fidelity = None
            self._states = None
            self._truth_table = None
        
        @property
        def fidelity(self) -> Dict:
            """Truth table fidelity.

            F = Tr[U_exp^T U] / 2^n

            Returns:
                Dict: value and error (uncertainty) of the estimated fidelity.
            """
            return self._fidelity
        
        @property
        def truth_table(self) -> NDArray:
            """Truth table.

            Returns:
                NDArray: truth table.
            """
            return self._truth_table
            
        def generate_circuits(self):
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            mapper = {'0': [Id, Id], '1': [X90, X90]}
            self._states  = [
                ''.join(i) for i in itertools.product(
                    [str(j) for j in [0, 1]], 
                    repeat=len(self._qubits)
                )
            ]

            for state in self._states:
                circuit = self._circuit.copy()
                circuit.prepend(
                    Cycle(
                        {mapper[s][1](q) for s, q in zip(state, self._qubits)}
                    )
                )
                circuit.prepend(
                    Cycle(
                        {mapper[s][0](q) for s, q in zip(state, self._qubits)}
                    )
                )
                circuit.measure(self._qubits)
                self._circuits.append(circuit)

            self._circuits['input state'] = self._states
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            q_index = tuple([
                self._qubits.index(q) for q in self._circuit.qubits
            ])

            probs = []
            for circuit in self._circuits:
                probs.append(
                    [circuit.results.marginalize(q_index).populations[state] 
                     for state in self._states
                    ]
                )
            self._truth_table = np.array(probs)

            if self._ideal_unitary is not None:
                fidelity = np.trace(
                    np.matmul(self._truth_table.T, self._ideal_unitary)
                ) / 2**len(self._qubits)
                error = uncertainty_of_sum(
                    [1 / np.sqrt(circuit.results.n_shots) 
                     for circuit in self._circuits
                    ]
                ) / 2**len(self._qubits)
                fidelity, error = round_to_order_error(fidelity, error)
                self._fidelity = {
                    'val': fidelity, 'err': error
                }

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_TruthTableTomography _Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                np.save(
                    self._data_manager._save_path + 'truth_table', 
                    self._truth_table
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._fidelity]), 'fidelity'
                )

        def plot(self):
            """Plot the parity oscillations."""

            bg = colors.LinearSegmentedColormap.from_list(
                'custom grey', [(1, 1, 1), '#d3d3d3'], N=256
            )

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')

            xpos = list(
                range(self._truth_table.shape[0])
            ) * self._truth_table.shape[0]
            ypos = sorted(xpos.copy())
            zpos = np.zeros(self._truth_table.size)
            dx = [0.85] * self._truth_table.size
            dy = [0.85] * self._truth_table.size
            dz = self._truth_table.real.flatten()
            offset = dz + np.abs(dz.min())
            fracs = offset.astype(float) / offset.max()
            norm = colors.Normalize(fracs.min(), fracs.max())
            cmap = bg(norm(fracs))

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cmap, alpha=0.85)
            ax.view_init(35, 45)

            ax.set_xticks(np.arange(self._truth_table.shape[0]) + 0.4)
            ax.set_yticks(np.arange(self._truth_table.shape[1]) + 0.4)
            ax.set_zticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticklabels(self._states, fontsize=12)
            ax.set_yticklabels(self._states, fontsize=12)
            ax.set_zticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
            ax.set_xlabel('Measured State', fontsize=15, labelpad=10)
            ax.set_ylabel('Prepared State', fontsize=15, labelpad=10)
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel('Probability', fontsize=15, rotation=90, labelpad=10)
            ax.set_box_aspect(aspect=None, zoom=0.8)

            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'parity_oscillations.png', 
                    dpi=600,
                    bbox_inches='tight', 
                    pad_inches=0
                )
                fig.savefig(
                    self._data_manager._save_path + 'parity_oscillations.pdf',
                    bbox_inches='tight', 
                    pad_inches=0
                )
                fig.savefig(
                    self._data_manager._save_path + 'parity_oscillations.svg',
                    bbox_inches='tight', 
                    pad_inches=0
                )
            plt.show()

        def final(self):
            """Final experimental method."""
            if self._fidelity is not None:
                fidelity = self._fidelity['val']
                error = self._fidelity['err']
                print(f'\nTruth Table Fidelity = {fidelity} ({error})')
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return TruthTable(
        config,
        circuit,
        qubits,
        ideal_unitary,
        compiler, 
        transpiler,
        classifier,
        n_shots, 
        n_batches,
        n_circs_per_seq, 
        n_levels,
        raster_circuits,
        **kwargs
    )