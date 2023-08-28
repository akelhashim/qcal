"""Submodule for readout calibration.

"""
import qcal.settings as settings

from .calibration import Calibration
from .utils import find_pulse_index, in_range
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.machine_learning.clustering import GaussianMixture
from qcal.managers.classification_manager import ClassificationManager

from qcal.gate.single_qubit import Id, X, X90
from qcal.plotting.utils import calculate_nrows_ncols
from qcal.qpu.qpu import QPU

import inspect
import logging
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from IPython.display import clear_output
from typing import Any, Callable, Dict, List, Tuple
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


_classifiers = {
    'gmm': GaussianMixture
}


def ReadoutCalibration(
        qpu:             QPU,
        config:          Config,
        qubits:          List | Tuple,
        method:          str = 'pi_pulse',
        gate:            str = 'X90',
        classifier:      str = 'gmm',
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None, 
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1, 
        n_levels:        int = 2,
        esp:             bool = False,
        heralding:       bool = False,
        **kwargs
    ) -> Callable:
    """Readout calibration

    # Basic example useage for initial calibration:
        
        # ```
    #     amps = np.linspace(0, 1.0, 21)
    #     cal = Amplitude(
    #         CustomQPU, 
    #         config, 
    #         qubits=[0, 1, 2],
    #         amps=amps)
    #     cal.run()
        # ```

    # Args:
    #     qpu (QPU): custom QPU object.
    #     config (Config): qcal Config object.
    #     qubits (List | Tuple): qubits to calibrate.
        # method (str, optional): calibration method. Must be one of ('pi_pulse',
        #     'rabi').
    #     gate (str, optional): native gate to calibrate. Defaults 
    #         to 'X90'.
    #     subspace (str, optional): qubit subspace for the defined gate.
    #         Defaults to 'GE'.
    #     compiler (Any | Compiler | None, optional): custom compiler to
    #         compile the experimental circuits. Defaults to None.
    #     transpiler (Any | None, optional): custom transpiler to 
    #         transpile the experimental circuits. Defaults to None.
    #     n_shots (int, optional): number of measurements per circuit. 
    #         Defaults to 1024.
    #     n_batches (int, optional): number of batches of measurements. 
    #         Defaults to 1.
    #     n_circs_per_seq (int, optional): maximum number of circuits that
    #         can be measured per sequence. Defaults to 1.
        #  n_levels (int, optional): number of energy levels to classify. 
        #     Defaults to 2.
    #     esp (bool, optional): whether to enable excited state promotion for 
    #         the calibration. Defaults to False.
    #     heralding (bool, optional): whether to enable heralding for the 
    #         calibraion. Defaults to False.

    Returns:
        Callable: ReadoutCalibration class.
    """

    class ReadoutCalibration(qpu, Calibration):
        """ReadoutCalibration class.
        
        This class inherits a custom QPU from the ReadoutCalibration function.
        """

        def __init__(self, 
                config:          Config,
                qubits:          List | Tuple,
                method:          str = 'pi_pulse',
                gate:            str = 'X90',
                classifier:      str = 'gmm',
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None, 
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1, 
                n_levels:        int = 2,
                esp:             bool = False,
                heralding:       bool = False,
                **kwargs
            ) -> None:
            """Initialize the Amplitude calibration class within the function.
            """
            qpu_args = list(inspect.signature(qpu).parameters)
            qpu_kwargs = {
                k: kwargs.pop(k) for k in dict(kwargs) if k in qpu_args
            }
            cm_args = list(
                inspect.signature(_classifiers[classifier]).parameters
            )
            cm_kwargs = {
                k: kwargs.pop(k) for k in dict(kwargs) if k in cm_args
            }

            assert heralding is False, (
                'Heralding must be disabled for readout calibration!'
            )
            
            qpu.__init__(self,
                config, 
                compiler, 
                transpiler, 
                n_shots, 
                n_batches, 
                n_circs_per_seq, 
                n_levels,
                **qpu_kwargs
            )
            Calibration.__init__(self, 
                config, 
                esp=esp,
                heralding=heralding
            )

            assert method in ('pi_pulse', 'rabi'), (
                "'method' must be one of 'pi_pulse' or 'rabi'!"
            )

            self._qubits = qubits
            
            assert gate in ('X90', 'X'), (
                "'gate' must be one of 'X90' or 'X'!"
            )
            self._gate = gate

            self._classifier = ClassificationManager(
                qubits=qubits, n_levels=n_levels, classifier=classifier,
                **cm_kwargs 
            )

        @property
        def classifier(self) -> ClassificationManager:
            """Classification manager.

            Returns:
                ClassificationManager: classifier.
            """
            return self._classifier

        def generate_circuits(self):
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')
            
            circuit0 = Circuit([
                Cycle({Id(q) for q in self._qubits}),
            ])
            circuit0.measure()

            circuit1 = Circuit()
            if self._gate == 'X90':
                circuit1.extend([
                    Cycle({X90(q, subspace='GE') for q in self._qubits}),
                    Barrier(self._qubits),
                    Cycle({X90(q, subspace='GE') for q in self._qubits}),
                ])
            elif self._gate == 'X':
                circuit1.extend([
                    Cycle({X(q, subspace='GE') for q in self._qubits}),
                ])
            circuit1.measure()

            circuits = [circuit0, circuit1]

            if self._n_levels == 3:
                circuit2 = Circuit()
                if self._gate == 'X90':
                    circuit2.extend([
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({X90(q, subspace='GE') for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({X90(q, subspace='EF') for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({X90(q, subspace='EF') for q in self._qubits}),
                    ])
                elif self._gate == 'X':
                    circuit2.extend([
                        Cycle({X(q, subspace='GE') for q in self._qubits}),
                        Barrier(self._qubits),
                        Cycle({X(q, subspace='EF') for q in self._qubits}),
                    ])
                circuit2.measure()
                circuits.append(circuit2)

            self._circuits = CircuitSet(circuits=circuits)
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            for q in self._qubits:
                iq_0 = self._circuits[f'Q{q}: iq_data'][0]
                iq_1 = self._circuits[f'Q{q}: iq_data'][1]

                X = np.vstack([
                        np.hstack([np.real(iq_0), np.imag(iq_0)]),
                        np.hstack([np.real(iq_1), np.imag(iq_1)]),
                ])
                y = [0] * self._n_shots + [1] * self._n_shots

                if self._n_levels == 3:
                    iq_2 = self._circuits[f'Q{q}: iq_data'][2]
                    X = np.vstack([
                        X,
                        np.hstack([np.real(iq_2), np.imag(iq_2)]),
                    ])
                    y += [2] * self._n_shots
                
                y = np.array(y)
                self._classifier.fit(q, X, y)

        def save(self) -> None:
            """Save all circuits and data."""
            qpu.save(self)
            self._data_manager.save_to_pickle(
                self._classifier, 
                os.path.join(
                    os.path.dirname(self._config.filename), 
                    'ClassificationManager'
                )
            )

        def plot(self) -> None:
            """Plot the readout calibration results."""
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

                        ax.set_xlabel('I', fontsize=15)
                        ax.set_ylabel('Q', fontsize=15)
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )
                        ax.grid(False)

                        # Create a mesh plot
                        h = 0.01
                        x_min, x_max = (
                            self._classifier[q].X[:, 0].min() - 1, 
                            self._classifier[q].X[:, 0].max() + 1
                        )
                        y_min, y_max =(
                            self._classifier[q].X[:, 1].min() - 1, 
                            self._classifier[q].X[:, 1].max() + 1
                        )
                        xx, yy = np.meshgrid(
                            np.arange(x_min, x_max, h), 
                            np.arange(y_min, y_max, h)
                        )

                        # Plot the decision boundary by assigning a color to 
                        # each point in the mesh [x_min, x_max]x[y_min, y_max].
                        Z = self._classifier[q].predict(
                            np.c_[xx.ravel(), yy.ravel()]
                        )
                        Z = Z.reshape(xx.shape)
                        ax.contourf(xx, yy, Z, cmap='viridis', alpha=0.1)
                            
                        # Plot the raw data
                        sc = ax.scatter(
                            self._classifier[q].X[:, 0], 
                            self._classifier[q].X[:, 1], 
                            c=self._classifier[q].predict(
                                self._classifier[q].X
                            ), 
                            cmap='viridis', alpha=0.15
                        )

                        ax.legend(
                            handles=sc.legend_elements()[0], 
                            labels=range(0, self._classifier[q].n_components), 
                            fontsize=12
                        )

                    else:
                        ax.axis('off')
                
            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'readout_calibration.png', 
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
                f'_RCal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                self.save()
            self.plot()
            self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

    return ReadoutCalibration(
        config,
        qubits,
        method,
        gate,
        compiler, 
        transpiler, 
        n_shots, 
        n_batches, 
        n_circs_per_seq, 
        n_levels,
        esp,
        heralding,
        **kwargs
    )