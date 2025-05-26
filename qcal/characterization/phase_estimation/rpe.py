""""Submodule for RPE experiments.

See:
https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.062315
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.190502

Relevant pyRPE code: https://gitlab.com/quapack/pyrpe
"""
import qcal.settings as settings

from qcal.characterization.phase_estimation.analysis import (
    analyze_idle, analyze_x90, analyze_cz, analyze_zz
)
from qcal.characterization.phase_estimation.circuits import (
    make_idle_circuits, make_x90_circuits, make_cz_circuits
)
from qcal.config import Config
from qcal.math.utils import round_to_order_error
from qcal.qpu.qpu import QPU
from qcal.plotting.utils import calculate_nrows_ncols
from qcal.utils import save_init

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from collections.abc import Iterable
from IPython.display import clear_output
from matplotlib.lines import Line2D
from numpy.typing import NDArray, ArrayLike
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


def X90(theta):
    """Definition of an X gate.

    Args:
        theta (angle): angle of rotation.

    Returns:
        pygsti.unitary_to_pauligate: X gate.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    H = theta/2 * pygsti.sigmax
    U = scipy.linalg.expm(-1j * H)
    return pygsti.unitary_to_pauligate(U)


def CZ(theta_iz: float, theta_zi: float, theta_zz: float) -> NDArray:
    """Definition of a CZ gate.

    diag([1,1,1,-1]) == CZ(np.pi/2, np.pi/2, -np.pi/2) (up to phase).

    Args:
        theta_iz (float): IZ angle.
        theta_zi (float): ZI angle
        theta_zz (float): ZZ angle.

    Returns:
        NDArray: matrix exponential of a CZ gate.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    return scipy.linalg.expm(
        -1j / 2 * (
            theta_iz * pygsti.sigmaiz + 
            theta_zi * pygsti.sigmazi + 
            theta_zz * pygsti.sigmazz
        )
    )


def plot_signal(
        signal:         Dict, 
        circuit_depths: ArrayLike,
        ax:             plt.axes = None, 
        title:          str = None
    ) -> None:
    """Plot RPE signal decay.

    Args:
        signal (Dict): signal for each RPE experiment.
        circuit_depths (ArrayLike): circuit depths.
        ax (plt.axes, optional): plot axes. Defaults to None.
        title (str, optional): plot title. Defaults to None.
    """
    if not ax:
        fig, ax = plt.subplots(1, figsize=(5, 4))

    mt = list(Line2D.markers.keys())[2:]
    # Plot the signals on the complex plane with a colormap for the depth
    n = 0
    for label, data in signal.items():
        for i, d in enumerate(circuit_depths):
            ax.scatter(
                data[i].real, 
                data[i].imag,
                marker=mt[n], 
                color=plt.cm.viridis(i / (len(circuit_depths) - 1)),
                label=label if i == 0 else None
            )
        n += 1
    ax.set_title(title)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_aspect('equal')
    ax.grid()
    ax.legend(loc=1)

    # Add colorbar 
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, 
        norm=plt.Normalize(vmin=0, vmax=len(circuit_depths))
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Depth index')

    # Draw the unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    ax.add_artist(circle)

    # Set the axis limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)


def RPE(qpu:            QPU,
        config:         Config,
        qubit_labels:   Iterable[int],
        gate:           str,
        circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
        gate_layer:     List = None,
        loss_angle:     str | List[str] | None = None,
        **kwargs
    ) -> Callable:
    """Robust Phase Estimation.

    This protocol requires a valid pyGSTi and pyRPE installation.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[int]): a list specifying sets of system labels 
            on which to perform RPE for a given gate.
        gate (str): gate on which to perform RPE.
        circuit_depths (List[int], optional): a list of positive integers 
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32, 
            64, 128, 256]```.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.
        loss_angle (str | list[str] | None, optional): a string or list of 
            strings specifying which angle to use for calculating the loss from 
            RPE. Defaults to None. If None, all angles are used. For example, 
            for the X90 gate, the possible options are `'X'` or `'Z'`.

    Returns:
        Callable: RPE class instance.
    """

    class RPE(qpu):
        """pyRPE protocol."""
        
        @save_init
        def __init__(self,
                config:         Config,
                qubit_labels:   Iterable[int],
                gate:           str,
                circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
                gate_layer:     List = None,
                loss_angle:     str | None = None,
                **kwargs
            ) -> None:
            from qcal.interface.pygsti.transpiler import PyGSTiTranspiler

            try:
                import pygsti
                from pygsti.modelpacks import smq2Q_XXYYII
                from pygsti.modelpacks import smq2Q_XYICPHASE
                logger.info(f" pyGSTi version: {pygsti.__version__}\n")
            except ImportError:
                logger.warning(' Unable to import pyGSTi!')

            assert gate.upper() in ('I', 'X90', 'CZ', 'ZZ'), (
                'Only I, X90, and CZ gates are currently supported!'
            )
            self._qubit_labels = qubit_labels
            self._gate = gate.upper()
            self._circuit_depths = circuit_depths
            self._loss_angle = loss_angle
            self._gate_layer = gate_layer

            qubits = []
            for q in qubit_labels:
                if isinstance(q, Iterable):
                    qubits.extend(q)
                else:
                    qubits.append(q)
            self._qubits = sorted(qubits)

            self._gate_model = {
                'I':   None,
                'X90': X90,
                'CZ':  CZ,
                'ZZ': None
            }
            self._target_model = {
                'I':   None,
                'X90': smq2Q_XXYYII.target_model(),
                'CZ':  smq2Q_XYICPHASE.target_model(),
                'I':   None
            }
            self._make_circuits = {
                'I':   make_idle_circuits,
                'X90': make_x90_circuits,
                'CZ':  make_cz_circuits,
                'ZZ':  make_cz_circuits
            }
            self._analyze_results = {
                'I':   analyze_idle,
                'X90': analyze_x90,
                'CZ':  analyze_cz,
                'ZZ':  analyze_zz
            }

            self._circuits = None
            self._datasets = {}
            self._angle_estimates = {}
            self._angle_errors = {}
            self._last_good_idx = {}
            self._signal = {}
            self._loss = {ql: {} for ql in qubit_labels}
            self._trusted_angle_est = {ql: {} for ql in qubit_labels}
            self._trusted_err_est = {ql: {} for ql in qubit_labels}

            transpiler = kwargs.get('transpiler', PyGSTiTranspiler())
            kwargs.pop('transpiler', None)
            qpu.__init__(self, config=config, transpiler=transpiler, **kwargs)

        @property
        def angle_estimates(self) -> Dict:
            """Angle estimates for each qubit/qubit pair.

            Returns:
                Dict: angle estimates.
            """
            return self._angle_estimates

        @property
        def angle_errors(self) -> Dict:
            """Angle errors for each qubit/qubit pair.

            Returns:
                Dict: angle errors.
            """
            return self._angle_errors

        @property
        def gate(self) -> str:
            """Gate being characterized.

            Returns:
                str: name of gate.
            """
            return self._gate

        @property
        def last_good_index(self) -> Dict:
            """Last good index for each qubit/qubit pair.

            Returns:
                Dict: last good index.
            """
            return self._last_good_idx
        
        @property
        def loss(self) -> Dict:
            """Loss for each qubit using the most trusted RPE estimates.

            This property can be used for parameter optimization.

            Returns:
                Dict: loss for each qubit.
            """
            if self._loss_angle is None:
                loss_angle = list(
                    self._angle_errors[self._qubit_labels[0]].keys()
                )
            else:
                loss_angle = (
                    self._loss_angle if isinstance(self._loss_angle, list) else
                    [self._loss_angle]
                )
            loss = {}
            for ql, errors in self._loss.items():
                vals = []
                for err, val in errors.items():
                    if err in loss_angle:
                        vals.append(val)
                loss[ql] = vals

            return loss
        
        @property
        def signal(self) -> Dict:
            """Signal decay.

            signal = 1 - 2p(I) + i(1 - 2p(Q))

            Returns:
                Dict: signal as a function of cirucit depth for each experiment.
            """
            return self._signal
        
        @property
        def trusted_angle_est(self) -> Dict:
            """Most trusted angle estimates.

            Returns:
                Dict: trusted angle estimates and uncertainties for each qubit 
                    label and error type.
            """
            return self._trusted_angle_est
        
        @property
        def trusted_error_est(self) -> Dict:
            """Most trusted error estimates.

            Returns:
                Dict: trusted error estimates and uncertainties for each qubit 
                    label and error type.
            """
            return self._trusted_err_est

        def generate_circuits(self):
            """Generate all RPE circuits."""
            from qcal.interface.pygsti.circuits import load_circuits

            logger.info(' Generating circuits from pyGSTi...')
            circuits = self._make_circuits[self._gate](
                self._circuit_depths, 
                self._qubit_labels, 
                self._gate_layer
            )
            self._circuits = load_circuits(circuits)

        def save(self):
            """Save all circuits and data."""
            self._data_manager._exp_id += (
                f'_RPE_Q{"".join(str(q) for q in self._qubit_labels)}'
            )

            if settings.Settings.save_data:
                logger.info(' Saving the circuits...')
                qpu.save(self)
                
        def generate_pygsti_dataset(self):
            """Generate a pyGSTi dataset for each qubit label."""
            logger.info(' Generating pyGSTi reports...')
            
            circuits = self._transpiled_circuits
            fileloc = self.data_manager.save_path
            for ql in self._qubit_labels:
                if isinstance(ql, Iterable):
                    q_index = tuple([self._qubits.index(q) for q in ql])
                    qs = ''.join(str(q) for q in ql)
                else:
                    q_index = tuple([self._qubits.index(ql)])
                    qs = str(ql)

                results_dfs = []
                for i, circuit in enumerate(circuits):
                    results_dfs.append(
                        pd.DataFrame(
                            [circuit.results.marginalize(q_index).dict], 
                            index=[circuits['pygsti_circuit'][i]]
                        )
                    )
                results_df = pd.concat(results_dfs)
                results_df = results_df.fillna(0).astype(int).rename(
                    columns=lambda col: col + ' count'
                )
                self._df = results_df

                with open(f'{fileloc}dataset_{qs}.txt', 'w') as f:
                    f.write(
                        '## Columns = ' + ', '.join(results_df.columns) + "\n"
                    )
                    f.close()
                results_df.to_csv(
                    f'{fileloc}dataset_{qs}.txt', 
                    sep=' ', 
                    mode='a', 
                    header=False
                )

        def analyze(self):
            """Analyze the RPE results."""
            logger.info(' Analyzing the results...')
            import pygsti
            
            clear_output(wait=True)
            for ql in self._qubit_labels:
                if isinstance(ql, Iterable):
                    qs = ''.join(str(q) for q in ql)
                else:
                    qs = str(ql)
                
                dataset = pygsti.io.read_dataset(
                    self.data_manager.save_path + f'dataset_{qs}.txt'
                )
                self._datasets[ql] = dataset

                results = self._analyze_results[self._gate](
                    dataset, 
                    self._qubit_labels, 
                    self._circuit_depths, 
                    self._gate_layer
                )
                angle_estimates, angle_errors, last_good_idx, signal = results
                self._angle_estimates[ql] = angle_estimates
                self._angle_errors[ql] = angle_errors
                self._last_good_idx[ql] = last_good_idx
                self._signal[ql] = signal

                for angle, estimates in self._angle_estimates[ql].items():
                    est = estimates[self._last_good_idx[ql]]
                    unc = np.pi / (2 * 2**self._last_good_idx[ql])
                    est, unc = round_to_order_error(est, unc)
                    self._trusted_angle_est[ql][angle] = {
                        'val': est, 'err': unc
                    }

            for ql in self._qubit_labels:
                if isinstance(ql, Iterable):
                    print(f'\nQubit pair: {ql}')
                else:
                    print(f'\nQubit: {ql}')

                print(f'Last good depth: L = {2**self._last_good_idx[ql]}')
                for angle, errors in self._angle_errors[ql].items():
                    error = errors[self._last_good_idx[ql]]
                    self._loss[ql][angle] = error
                    unc = np.pi / (2 * 2**self._last_good_idx[ql])
                    error, unc = round_to_order_error(error, unc)
                    error_deg, unc_deg = round_to_order_error(
                        error * 180 / np.pi, unc * 180 / np.pi
                    )
                    self._trusted_err_est[ql][angle] = {
                        'val': error, 'err': unc
                    }
                    print(
                        f'{angle} error = {error} ({unc}) rad., '
                        f'{error_deg} ({unc_deg}) deg.'
                    )

        def plot(self) -> None:
            """Plot the RPE results."""
            nrows, ncols = calculate_nrows_ncols(len(self._qubit_labels))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            k = -1
            for i in range(nrows):
                for j in range(ncols):
                    k += 1

                    if len(self._qubit_labels) == 1:
                        ax = axes
                    elif axes.ndim == 1:
                        ax = axes[j]
                    elif axes.ndim == 2:
                        ax = axes[i,j]

                    if k < len(self._qubit_labels):
                        ql = self._qubit_labels[k]
                        for angle, errors in self._angle_errors[ql].items():
                            ax.errorbar(
                                self._circuit_depths[:len(errors)], 
                                errors,
                                yerr=np.pi/(2*np.array(
                                    self._circuit_depths[:len(errors)]
                                )),
                                fmt='o-',
                                elinewidth=0.75,
                                capsize=7,
                                label=angle
                            )
                        ax.axvline(
                            2**self._last_good_idx[ql], 
                            ls='--',
                            c='k',
                            label='Last good depth',
                        )

                        maxval = np.nanmax(np.abs(np.concatenate(
                            [err for err in self._angle_errors[ql].values()]
                        )))
                        ax.set_ylim((-1.1 * maxval, 1.1 * maxval))

                        ax.set_title(f'Q{ql}', fontsize=20)
                        ax.set_xlabel('Circuit Depth', fontsize=15)
                        ax.set_ylabel('Angle Error (rad.)', fontsize=15)
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )
                        ax.set_xscale('log')
                        # ax.set_yscale('log')
                        ax.legend(prop=dict(size=12))
                        ax.grid(True)

                    else:
                        ax.axis('off')
                
            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'RPE.png', 
                    dpi=300
                )
                fig.savefig(
                    self._data_manager._save_path + 'RPE.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'RPE.svg'
                )
            plt.show()

        def plot_signal(self) -> None:
            """Plot signal decay."""
            nrows, ncols = calculate_nrows_ncols(len(self._qubit_labels))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            k = -1
            for i in range(nrows):
                for j in range(ncols):
                    k += 1

                    if len(self._qubit_labels) == 1:
                        ax = axes
                    elif axes.ndim == 1:
                        ax = axes[j]
                    elif axes.ndim == 2:
                        ax = axes[i,j]

                    if k < len(self._qubit_labels):
                        ql = self._qubit_labels[k]

                        plot_signal(
                            signal=self._signal[ql], 
                            circuit_depths=self._circuit_depths,
                            ax=ax,
                            title=f'Q{ql}'
                        )

                    else:
                        ax.axis('off')

            # fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'RPE_signal.png', 
                    dpi=300
                )
                fig.savefig(
                    self._data_manager._save_path + 'RPE_signal.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'RPE_signal.svg'
                )
            plt.show()

        def final(self) -> None:
            """Final method."""
            if settings.Settings.save_data:
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._angle_estimates]), 'angle_estimates'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._angle_errors]), 'angle_errors'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._last_good_idx]), 'last_good_idx'
                )
            
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.save()
            self.generate_pygsti_dataset()
            self.analyze()
            self.plot()
            self.final()

    return RPE(
        config=config,
        qubit_labels=qubit_labels,
        gate=gate,   
        circuit_depths=circuit_depths,
        gate_layer=gate_layer,
        loss_angle=loss_angle,
        **kwargs
    )
