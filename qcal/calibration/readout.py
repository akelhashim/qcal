"""Submodule for readout calibration.

"""
import qcal.settings as settings

from .calibration import Calibration
from qcal.benchmarking.readout import ReadoutFidelity
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.config import Config
from qcal.machine_learning.clustering import GaussianMixture
from qcal.managers.classification_manager import ClassificationManager

from qcal.gate.single_qubit import Id, X, X90
from qcal.plotting.utils import calculate_nrows_ncols
from qcal.qpu.qpu import QPU
from qcal.utils import save_to_pickle

import inspect
import logging
import matplotlib.pyplot as plt
import os
import numpy as np

from IPython.display import clear_output
from numpy.typing import ArrayLike
from sklearn.inspection import DecisionBoundaryDisplay

from typing import Callable, Dict, List, Tuple
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


logger = logging.getLogger(__name__)


_classifiers = {
    'gmm': GaussianMixture
}


def ReadoutCalibration(
        qpu:        QPU,
        config:     Config,
        qubits:     List | Tuple,
        method:     str = 'pi_pulse',
        gate:       str = 'X90',
        model:      str = 'gmm',
        classifier: ClassificationManager = None,
        n_levels:   int = 2,
        **kwargs
    ) -> Callable:
    """Readout calibration

    Basic example useage for initial calibration:
        
        ```
        cal = ReadoutCalibration(
            CustomQPU, 
            config, 
            qubits=[0, 1, 2])
        cal.run()
        ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List | Tuple): qubits to calibrate.
        method (str, optional): calibration method. Must be one of ('pi_pulse',
            'rabi').
        gate (str, optional): native gate to calibrate. Defaults 
            to 'X90'.
        model (str, optional): classification algorithm. Defaults to 'gmm'.
        classifier (ClassificationManager, optional): manager used for 
            classifying raw data. Defaults to None.
        n_levels (int, optional): number of energy levels to classify. 
            Defaults to 2.

    Returns:
        Callable: ReadoutCalibration class.
    """

    class ReadoutCalibration(qpu, Calibration):
        """ReadoutCalibration class.
        
        This class inherits a custom QPU from the ReadoutCalibration function.
        """

        def __init__(self, 
                config:     Config,
                qubits:     List | Tuple,
                method:     str = 'pi_pulse',
                gate:       str = 'X90',
                model:      str = 'gmm',
                classifier: ClassificationManager = None,
                n_levels:   int = 2,
                **kwargs
            ) -> None:
            """Initialize the readout calibration class within the function.
            """
            qpu_args = list(inspect.signature(qpu).parameters)
            qpu_kwargs = {
                k: kwargs.pop(k) for k in dict(kwargs) if k in qpu_args
            }
            cm_args = list(
                inspect.signature(_classifiers[model]).parameters
            )
            cm_kwargs = {
                k: kwargs.pop(k) for k in dict(kwargs) if k in cm_args
            }
            
            qpu.__init__(self,
                config=config, 
                classifier=None,
                n_levels=n_levels,
                **qpu_kwargs
            )
            Calibration.__init__(self, config)

            if self._config['readout/esp/enable']:
                self.set_param('readout/esp/enable', False)
                self._enable_esp = True
            else:
                self._enable_esp = False

            self._qubits = qubits

            assert method in ('pi_pulse', 'rabi'), (
                "'method' must be one of 'pi_pulse' or 'rabi'!"
            )
            self._method = method
            
            assert gate in ('X90', 'X'), (
                "'gate' must be one of 'X90' or 'X'!"
            )
            self._gate = gate

            if classifier:
                self._classifier = classifier
            else:
                self._classifier = ClassificationManager(
                    qubits=qubits, n_levels=n_levels, model=model,
                    **cm_kwargs 
                )
            self._X = {}
            self._y = {}

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

                xy_0 = np.hstack([np.real(iq_0), np.imag(iq_0)])
                xy_1 = np.hstack([np.real(iq_1), np.imag(iq_1)])
                means_init = np.vstack([
                    np.mean(xy_0, axis=0), np.mean(xy_1, axis=0)
                ])
                X = np.vstack([xy_0, xy_1])
                y = [0] * self._n_shots + [1] * self._n_shots

                if self._n_levels == 3:
                    iq_2 = self._circuits[f'Q{q}: iq_data'][2]
                    xy_2 = np.hstack([np.real(iq_2), np.imag(iq_2)])
                    means_init = np.vstack([
                        means_init, np.mean(xy_2, axis=0)
                    ])
                    X = np.vstack([X, xy_2])
                    y += [2] * self._n_shots
                
                y = np.array(y)
                self._X[q] = X
                self._y[q] = y

                self._classifier[q].means_init = means_init
                self._classifier.fit(q, X, y)

        def save(self) -> None:
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_RCal_Q{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                save_to_pickle(
                    self._classifier, 
                    os.path.join(
                        os.path.dirname(self._config.filename), 
                        'ClassificationManager'
                    )
                )

        def plot(self, raw=False) -> None:
            """Plot the readout calibration results.

            Args:
                raw (bool, optional): plot the raw data. Defaults to False.
            """
            nrows, ncols = calculate_nrows_ncols(len(self._qubits))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            colors = [
                (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                (1.0, 0.4980392156862745, 0.054901960784313725),
                (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
            ]
            cmap = ListedColormap(colors[:self._n_levels])

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

                        if raw:
                            sc = ax.scatter(
                                self._X[q][:, 0], self._X[q][:, 1], 
                                c=self._y[q], cmap=cmap, alpha=0.03
                            )
                        else:
                            ax.hexbin(
                                self._X[q][:, 0], self._X[q][:, 1], 
                                cmap='Greys', gridsize=75
                            )

                        DecisionBoundaryDisplay.from_estimator(
                            self._classifier[q], 
                            self._X[q], 
                            response_method="predict",
                            alpha=0.15,
                            ax=ax,
                            grid_resolution=50,
                            cmap=cmap
                        )

                        # Create a mesh plot
                        x_min, x_max = (
                            self._X[q][:, 0].min() - 10, 
                            self._X[q][:, 0].max() + 10
                        )
                        y_min, y_max =(
                            self._X[q][:, 1].min() - 10, 
                            self._X[q][:, 1].max() + 10
                        )
                        # h = int(min([abs(x_min), abs(y_min)]) * 0.025)
                        # xx, yy = np.meshgrid(
                        #     np.arange(x_min, x_max, h), 
                        #     np.arange(y_min, y_max, h)
                        # )

                        # # Plot the decision boundary by assigning a color to 
                        # # each point in the mesh [x_min, x_max]x[y_min, y_max].
                        # Z = self._classifier[q].predict(
                        #     np.c_[xx.ravel(), yy.ravel()]
                        # )
                        # Z = Z.reshape(xx.shape)
                        # if raw:
                        #     ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.15)
                        # else:
                        #     cs = ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.15)
                        #     self._cs = cs
                            
                        ax.set_xlim([x_min, x_max])
                        ax.set_ylim([y_min, y_max])
                        ax.ticklabel_format(
                            axis='both', style='sci', scilimits=(0,0)
                        )
                        ax.text(
                            0.05, 0.9, f'R{q}', size=15, 
                            transform=ax.transAxes
                        )

                        if raw:
                            leg = ax.legend(
                                handles=sc.legend_elements()[0], 
                                labels=range(0, self._n_levels), 
                                fontsize=12,
                                loc=0
                            )
                        else:
                        #     handles = []
                        #     for l in range(self._n_levels):
                        #         if l*3 < len(cs.legend_elements()[0]):
                        #             handles.append(cs.legend_elements()[0][l*3])
                        #     leg = ax.legend(
                        #         handles=handles,
                        #         labels=range(0, self._n_levels), 
                        #         fontsize=12,
                        #         loc=0
                        #     )
                            handles = [
                                Patch(color=cmap(i/(self._n_levels-1)), alpha=1) 
                                for i in range(self._n_levels)
                            ]
                            leg = ax.legend(
                                handles=handles,
                                labels=range(0, self._n_levels), 
                                fontsize=12,
                                loc=0
                            )
                        for lh in leg.legend_handles:
                            lh.set_alpha(1)

                    else:
                        ax.axis('off')
                
            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                if raw:
                    fig.savefig(
                        self._data_manager._save_path + 
                        'readout_calibration_raw.png', 
                        dpi=300
                    )
                    fig.savefig(
                        self._data_manager._save_path + 
                        'readout_calibration_raw.pdf'
                    )
                    fig.savefig(
                        self._data_manager._save_path + 
                        'readout_calibration_raw.svg'
                    )
                else:
                    fig.savefig(
                        self._data_manager._save_path + 
                        'readout_calibration.png', 
                        dpi=300
                    )
                    fig.savefig(
                        self._data_manager._save_path + 
                        'readout_calibration.pdf'
                    )
                    fig.savefig(
                        self._data_manager._save_path + 
                        'readout_calibration.svg'
                    )
            plt.show()

        def final(self):
            """Final calibration methods."""
            if self._enable_esp:
                self.set_param('readout/esp/enable', True)
            
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self, plot: bool = True):
            """Run all experimental methods and analyze results.

            Args:
                plot (bool, optional): whether to plot the readout results.
                    Defaults to True. Plotting the results can take time due to
                    meshgrid. It is best to set this to False if doing a sweep.
            """
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            if plot:
                self.plot()
            self.final()

    return ReadoutCalibration(
        config,
        qubits,
        method,
        gate,
        model,
        classifier, 
        n_levels,
        **kwargs
    )


def Fidelity(
        qpu:         QPU,
        config:      Config,
        qubits:      List | Tuple,
        params:      Dict[int, str],
        param_sweep: ArrayLike | Dict[int, ArrayLike],
        gate:        str = 'X90',
        n_levels:    int = 2,
        **kwargs
    ) -> Callable:
    """Fidelity calibration.

    Basic example useage:
        
    ```
    qubits = [0, 1, 2, 3, 4, 5, 6, 7]
    params = {q: f'readout/{q}/amp' for q in qubits}
    param_sweep = {
        q: np.linspace(-0.005, 0.005, 21)
        + cfg[f'readout/{q}/amp'] for q in qubits
    }

    cal = Fidelity(
        CustomQPU,
        config,
        qubits=qubits,
        params=params,
        param_sweep=param_sweep
    )
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List | Tuple): qubits to calibrate.
        params (Dict[int, str]): dictionary mapping qubit label to param to 
            sweep over.
        param_sweep (ArrayLike | Dict[int, ArrayLike]): value sweep for each 
            qubit.
        gate (str, optional): native gate to used to prepare states. Defaults 
            to 'X90'.
        n_levels (int, optional): number of energy levels to classify. 
            Defaults to 2.

    Returns:
        Callable: Fidelity class.
    """

    class Fidelity(qpu, Calibration):
        """Fidelity class.
        
        This class inherits a custom QPU from the Fidelity function.
        """

        def __init__(self, 
                config:      Config,
                qubits:      List | Tuple,
                params:      Dict[int, str],
                param_sweep: ArrayLike | Dict[int, ArrayLike],
                gate:        str = 'X90',
                n_levels:    int = 2,
                **kwargs
            ) -> None:
            """Initialize the Fidelity calibration class within the function.
            """
            self._rcal = ReadoutFidelity(
                qpu=qpu,
                config=config,
                qubits=qubits,
                gate=gate,
                n_levels=n_levels,
                **kwargs
            )

            Calibration.__init__(self, config)

            assert len(qubits) == len(params), (
                "The number of qubits must be equal to the number of params!"
            )
            self._qubits = qubits

            self._params = params
            self._param_sweep = (
                param_sweep if isinstance(param_sweep, Dict) else {
                    q: param_sweep for q in qubits
                }
            )

            self._n_levels = n_levels
            self._fid = {q: {
                    level: [] for level in range(n_levels)
                } for q in qubits
            }

        @property
        def fidelity(self) -> dict:
            """Fidelity.

            Returns:
                dict: fidelity of states for each qubit for the param sweep.
            """
            return self._fid
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')
            
            # Find the maximum separation for 
            for q in self._qubits:
                fid = np.array(self._fid[q][0])
                for n in range(self._n_levels)[1:]:
                    fid += np.array(self._fid[q][n])
                max_fid_idx = np.argmax(fid)
                self._cal_values[q] = self._param_sweep[q][max_fid_idx]

        def plot(self) -> None:
            """Plot the fidelity calibration results."""
            nrows, ncols = calculate_nrows_ncols(len(self._qubits))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            colors = [
                (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                (1.0, 0.4980392156862745, 0.054901960784313725),
                (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
            ]

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

                        ax.set_xlabel('Parameter Sweep', fontsize=15)
                        ax.set_ylabel('Fidelity', fontsize=15)
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )

                        for n in range(self._n_levels):
                            ax.plot(
                                self._param_sweep[q], 
                                self._fid[q][n],
                                'o-',
                                c=colors[n],
                                label=rf'$|{n}\rangle$'
                            )

                        ax.axvline(
                            self._cal_values[q],  
                            ls='--', c='k', label='Max fid.'
                        )
                        ax.text(
                            0.05, 0.9, f'R{q}', size=15, 
                            transform=ax.transAxes
                        )
                        ax.legend(loc=0, fontsize=12)

                    else:
                        ax.axis('off')
                
            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                   self._rcal._data_manager._save_path + 'fid_calibration.png', 
                   dpi=300
                )
            plt.show()

        def final(self) -> None:
            """Save and load the config after changing parameters."""
            for q in self._qubits:
                self._config[self._params[q]] = self._cal_values[q]

            self._config.save()
            self._config.load()

        def run(self) -> None:
            """Run the experiment."""
            for i in range(len(self._param_sweep[self._qubits[0]])):
                for q, param in self._params.items():
                    self._rcal._config[param] = self._param_sweep[q][i]

                self._rcal._measurements = []
                self._rcal.run()

                for q in self._qubits:
                    for n in range(self._n_levels):
                        self._fid[q][n].append(
                            self._rcal.confusion_matrix[f'Q{q}'].loc[
                                    'Prep State', 'Meas State'
                                ].loc[n][n]
                        )

            self.analyze()
            clear_output(wait=True)
            print(f"\nRuntime: {repr(self._rcal._runtime)[8:]}\n")
            self.plot()
            self.final()

    return Fidelity(
        config=config,
        qubits=qubits,
        params=params,
        param_sweep=param_sweep,
        gate=gate,
        n_levels=n_levels,
        **kwargs
    )


def Separation(
        qpu:         QPU,
        config:      Config,
        qubits:      List | Tuple,
        params:      Dict[int, str],
        param_sweep: ArrayLike | Dict[int, ArrayLike],
        method:      str = 'pi_pulse',
        gate:        str = 'X90',
        model:       str = 'gmm',
        n_levels:    int = 2,
        **kwargs
    ) -> Callable:
    """Separation calibration.

    Basic example useage:
        
    ```
    qubits = [0, 1, 2, 3, 4, 5, 6, 7]
    params = {q: f'readout/{q}/freq' for q in qubits}
    # +/- 500 kHz sweep around the current frequency
    param_sweep = { # +/- 500 kHz sweep around the current frequency
        q: np.linspace(-0.5, 0.5, 31) * MHz
        + cfg[f'readout/{q}/freq'] for q in qubits
    }

    cal = Separation(
        CustomQPU,
        config,
        qubits=qubits,
        params=params,
        param_sweep=param_sweep
    )
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (List | Tuple): qubits to calibrate.
        params (Dict[int, str]): dictionary mapping qubit label to param to 
            sweep over.
        param_sweep (ArrayLike | Dict[int, ArrayLike]): value sweep for each 
            qubit.
        method (str, optional): calibration method. Must be one of ('pi_pulse',
            'rabi').
        gate (str, optional): native gate used for state preparation. Defaults 
            to 'X90'.
        model (str, optional): classification algorithm. Defaults to 'gmm'.
        n_levels (int, optional): number of energy levels to classify. 
            Defaults to 2.

    Returns:
        Callable: Separation class.
    """

    class Separation(qpu, Calibration):
        """Separation class.
        
        This class inherits a custom QPU from the Separation function.
        """

        def __init__(self, 
                config:      Config,
                qubits:      List | Tuple,
                params:      Dict[int, str],
                param_sweep: ArrayLike | Dict[int, ArrayLike],
                method:      str = 'pi_pulse',
                gate:        str = 'X90',
                model:       str = 'gmm',
                n_levels:    int = 2,
                **kwargs
            ) -> None:
            """Initialize the Amplitude calibration class within the function.
            """
            self._rcal = ReadoutCalibration(
                qpu=qpu,
                config=config,
                qubits=qubits,
                method=method,
                gate=gate,
                model=model, 
                n_levels=n_levels,
                **kwargs
            )

            Calibration.__init__(self, config)

            assert len(qubits) == len(params), (
                "The number of qubits must be equal to the number of params!"
            )
            self._qubits = qubits

            self._params = params
            self._param_sweep = (
                param_sweep if isinstance(param_sweep, Dict) else {
                    q: param_sweep for q in qubits
                }
            )

            self._groupings = []
            for groupings in np.array(np.triu_indices(n_levels, 1)).T:
                self._groupings.append("{0}{1}".format(*groupings))
            self._sep = {q: {
                    grouping: [] for grouping in self._groupings
                } for q in qubits
            }

        @property
        def separation(self) -> dict:
            """Separation.

            Returns:
                dict: separation of states for each qubit for the param sweep.
            """
            return self._sep
                
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')
            
            # Find the maximum separation for 
            for q in self._qubits:
                sep = np.array(self._sep[q][self._groupings[0]])
                for g in self._groupings[1:]:
                    sep += np.array(self._sep[q][g])
                max_sep_idx = np.argmax(sep)
                self._cal_values[q] = self._param_sweep[q][max_sep_idx]

        def plot(self) -> None:
            """Plot the calibration results for the maximum separation."""
            nrows, ncols = calculate_nrows_ncols(len(self._qubits))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            colors = [
                (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                (1.0, 0.4980392156862745, 0.054901960784313725),
                (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
            ]

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

                        ax.set_xlabel('Parameter Sweep', fontsize=15)
                        ax.set_ylabel('Separation (SNR)', fontsize=15)
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )
                        ax.text(
                            0.05, 0.9, f'R{q}', size=15, 
                            transform=ax.transAxes
                        )

                        for m, g in enumerate(self._groupings):
                            ax.plot(
                                self._param_sweep[q], 
                                self._sep[q][g],
                                'o-',
                                c=colors[m],
                                label=g
                            )

                        ax.axvline(
                            self._cal_values[q],  
                            ls='--', c='k', label='Max sep.'
                        )

                        ax.legend(loc=0, fontsize=12)

                    else:
                        ax.axis('off')
                
            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                   self._rcal._data_manager._save_path + 'sep_calibration.png', 
                   dpi=300
                )
                fig.savefig(
                   self._rcal._data_manager._save_path + 'sep_calibration.pdf'
                )
                fig.savefig(
                   self._rcal._data_manager._save_path + 'sep_calibration.svg'
                )
            plt.show()

        def final(self) -> None:
            """Save and load the config after changing parameters."""
            for q in self._qubits:
                self._config[self._params[q]] = self._cal_values[q]

            self._config.save()
            self._config.load()

        def run(self) -> None:
            """Run the experiment."""
            for i in range(len(self._param_sweep[self._qubits[0]])):
                for q, param in self._params.items():
                    self._rcal._config[param] = self._param_sweep[q][i]

                self._rcal._measurements = []
                self._rcal.run(plot=False)

                for q in self._qubits:
                    for g in self._groupings:
                        self._sep[q][g].append(self._rcal.classifier[q].snr[g])

            self.analyze()
            clear_output(wait=True)
            print(f"\nRuntime: {repr(self._rcal._runtime)[8:]}\n")
            self.plot()
            self.final()

    return Separation(
        config=config,
        qubits=qubits,
        params=params,
        param_sweep=param_sweep,
        method=method,
        gate=gate,
        model=model,
        n_levels=n_levels,
        **kwargs
    )