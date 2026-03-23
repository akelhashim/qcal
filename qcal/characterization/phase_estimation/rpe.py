""""Submodule for RPE experiments.

See:
https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.062315
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.190502

Relevant pyRPE code: https://gitlab.com/quapack/pyrpe
"""
import logging
from collections.abc import Iterable
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pygsti
import scipy
from IPython.display import clear_output
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from pygsti.data import DataSet
from pygsti.io import write_dataset
from pygsti.modelpacks import smq2Q_XXYYII, smq2Q_XYICPHASE

from qcal.characterization.phase_estimation.analysis import (
    analyze_cz,
    analyze_idle,
    analyze_x90,
    analyze_zz,
)
from qcal.characterization.phase_estimation.circuits import (
    make_cz_circuits,
    make_idle_circuits,
    make_x90_circuits,
)
from qcal.config import Config
from qcal.interface.pygsti.circuits import load_circuits
from qcal.interface.pygsti.transpiler import PyGSTiTranspiler
from qcal.math.utils import round_to_order_error
from qcal.plotting.utils import calculate_nrows_ncols
from qcal.qpu.qpu import QPU
from qcal.results import Results
from qcal.settings import Settings
from qcal.utils import save_init

logger = logging.getLogger(__name__)


def X90(theta):
    """Definition of an X gate.

    Args:
        theta (angle): angle of rotation.

    Returns:
        pygsti.unitary_to_pauligate: X gate.
    """
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
        for i, _d in enumerate(circuit_depths):
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


def RPE(
    qpu:            QPU,
    config:         Config,
    qubit_labels:   Iterable[int],
    gate:           str,
    circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
    gate_layer:     List = None,
    loss_angle:     str | List[str] | None = None,
    **kwargs
) -> Callable:
    """Robust Phase Estimation.

    This is the parent class for all RPE experiments. For gate-specific RPE,
    use the subclasses: IdleRPE, X90RPE, CZRPE, or ZZRPE.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[int]): a list specifying sets of system labels
            on which to perform RPE for a given gate.
        gate (str): gate on which to perform RPE. Must be one of 'I', 'X90',
            'CZ', or 'ZZ'.
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
    if gate not in ['I', 'X90', 'CZ', 'ZZ']:
        raise ValueError(
            f"Gate must be one of 'I', 'X90', 'CZ', or 'ZZ'. Got {gate}."
        )

    gate_model_map = {
        'I':   None,
        'X90': X90,
        'CZ':  CZ,
        'ZZ':  None
    }
    target_model_map = {
        'I':   None,
        'X90': smq2Q_XXYYII.target_model(),
        'CZ':  smq2Q_XYICPHASE.target_model(),
        'ZZ':  None
    }
    make_circuits_map = {
        'I':   make_idle_circuits,
        'X90': make_x90_circuits,
        'CZ':  make_cz_circuits,
        'ZZ':  make_cz_circuits
    }
    analyze_results_map = {
        'I':   analyze_idle,
        'X90': analyze_x90,
        'CZ':  analyze_cz,
        'ZZ':  analyze_zz
    }

    _gate_name = gate.upper()
    _gate_model_func = gate_model_map[_gate_name]
    _target_model_obj = target_model_map[_gate_name]
    _make_circuits_func = make_circuits_map[_gate_name]
    _analyze_results_func = analyze_results_map[_gate_name]

    class RPE(qpu):
        """pyRPE protocol."""

        @save_init
        def __init__(
            self,
            config:         Config,
            qubit_labels:   Iterable[int],
            circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
            gate_layer:     List = None,
            loss_angle:     str | None = None,
            **kwargs
        ) -> None:
            logger.info(f" pyGSTi version: {pygsti.__version__}\n")

            self._qubit_labels = qubit_labels
            self._gate = _gate_name
            self._gate_model_func = _gate_model_func
            self._target_model_obj = _target_model_obj
            self._make_circuits_func = _make_circuits_func
            self._analyze_results_func = _analyze_results_func
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
        def datasets(self) -> Dict[int | Tuple[int], DataSet]:
            """Datasets for each qubit/qubit pair.

            Returns:
                Dict[int | Tuple[int], DataSet]: datasets.
            """
            return self._datasets

        @property
        def gate(self) -> str:
            """Gate being characterized.

            Returns:
                str: name of gate.
            """
            return self._gate

        @property
        def gate_model(self):
            """Gate model function for this RPE experiment.

            Returns:
                Callable: gate model function.
            """
            return self._gate_model_func

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
        def target_model(self):
            """Target model for this RPE experiment.

            Returns:
                Any: pyGSTi target model.
            """
            return self._target_model_obj

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
            logger.info(' Generating circuits from pyGSTi...')
            circuits = self._make_circuits_func(
                self._circuit_depths,
                self._qubit_labels,
                self._gate_layer
            )
            self._circuits = load_circuits(circuits)

        def save(self):
            """Save all circuits and data."""
            self._data_manager._exp_id += (
                f'_RPE_{"".join("Q" + str(q) for q in self._qubit_labels)}'
            )

            if Settings.save_data:
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
                    qs = '_'.join(str(q) for q in ql)
                else:
                    q_index = (self._qubits.index(ql),)
                    qs = str(ql)

                states = set()
                all_results = []
                for result in circuits['results']:
                    res = Results(result).marginalize(q_index)
                    states.update(res.states)
                    all_results.append(res.dict)

                self._datasets[ql] = DataSet(outcome_labels=list(states))
                for i, result in enumerate(all_results):
                    self._datasets[ql][self._circuits[i]] = result
                self._datasets[ql].done_adding_data()

                if Settings.save_data is True:
                    write_dataset(
                        f'{fileloc}dataset_{qs}.txt',
                        self._datasets[ql],
                    )

        def analyze(self):
            """Analyze the RPE results."""
            logger.info(' Analyzing the results...')

            clear_output(wait=True)
            for ql in self._qubit_labels:
                results = self._analyze_results_func(
                    self._datasets[ql],
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

            mpl_colors = ['tab:blue', 'tab:orange', 'tab:green']
            plotly_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            pfig_height = 350 * nrows
            pfig_width = 350 * ncols + 50
            pfig_margin = {'t': 50, 'b': 50, 'l': 50, 'r': 50}
            pfig_gap_px = 60
            vertical_spacing = (
                0.0 if nrows <= 1 else min(0.2, pfig_gap_px / pfig_height)
            )
            horizontal_spacing = (
                0.0 if ncols <= 1 else min(0.2, pfig_gap_px / pfig_width)
            )
            pfig = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=[f"Q{ql}" for ql in self._qubit_labels],
                vertical_spacing=vertical_spacing,
                horizontal_spacing=horizontal_spacing,
            )
            pfig.update_annotations(font_size=12)

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
                        for idx, angle in enumerate(
                            sorted(self._angle_errors[ql])
                        ):
                            errors = self._angle_errors[ql][angle]
                            depths = self._circuit_depths[:len(errors)]
                            yerr = np.pi / (2 * np.array(depths))
                            mpl_color = mpl_colors[idx % len(mpl_colors)]
                            plotly_color = plotly_colors[
                                idx % len(plotly_colors)
                            ]

                            ax.errorbar(
                                depths,
                                errors,
                                yerr=yerr,
                                fmt='o-',
                                elinewidth=0.75,
                                capsize=7,
                                color=mpl_color,
                                ecolor=mpl_color,
                                label=angle
                            )
                            pfig.add_trace(
                                go.Scatter(
                                    x=depths,
                                    y=errors,
                                    mode='lines+markers',
                                    name=str(angle),
                                    line={'color': plotly_color},
                                    marker={'color': plotly_color},
                                    error_y={
                                        'type': 'data',
                                        'array': yerr,
                                        'visible': True,
                                        'thickness': 1,
                                        'width': 6,
                                    },
                                    showlegend=(k == 0),
                                ),
                                row=i + 1,
                                col=j + 1,
                            )

                        # Add vertical line for last good depth
                        last_good_depth = 2**self._last_good_idx[ql]
                        ax.axvline(
                            last_good_depth,
                            ls='--',
                            c='k',
                            label='Last good depth',
                        )
                        pfig.add_vline(
                            x=last_good_depth,
                            line_dash='dash',
                            line_color='black',
                            name='Last good depth',
                            showlegend=(k == 0),
                            row=i + 1,
                            col=j + 1,
                        )

                        maxval = np.nanmax(np.abs(np.concatenate(
                            list(self._angle_errors[ql].values())
                        )))
                        ax.set_ylim((-1.1 * maxval, 1.1 * maxval))
                        ax.set_title(f'Q{ql}', fontsize=20)
                        ax.set_xlabel('Circuit Depth', fontsize=15)
                        ax.set_ylabel('Angle Error (rad.)', fontsize=15)
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )
                        ax.set_xscale('log')
                        ax.legend(prop={'size': 12})
                        ax.grid(True)

                        pfig.update_xaxes(
                            type='log',
                            title_text='Circuit Depth' if i == nrows-1 else '',
                            title_standoff=10,
                            automargin=True,
                            showgrid=True,
                            row=i + 1,
                            col=j + 1,
                        )
                        pfig.update_yaxes(
                            title_text='Angle Error (rad.)' if j == 0 else '',
                            title_standoff=10,
                            automargin=True,
                            showgrid=True,
                            range=[-1.1 * maxval, 1.1 * maxval],
                            row=i + 1,
                            col=j + 1,
                        )

                    else:
                        ax.axis('off')
                        pfig.update_xaxes(visible=False, row=i + 1, col=j + 1)
                        pfig.update_yaxes(visible=False, row=i + 1, col=j + 1)

            pfig.update_layout(
                height=pfig_height,
                width=pfig_width,
                margin=pfig_margin,
                legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02},
                template='plotly_white',
                paper_bgcolor='white',
                plot_bgcolor='#fbfbfd',
            )

            pfig.update_xaxes(
                automargin=True,
                showline=True,
                mirror=True,
                linecolor='#c7c7c7',
                linewidth=1,
                gridcolor='#e5e7eb',
                zeroline=False,
                ticks='outside',
            )
            pfig.update_yaxes(
                automargin=True,
                showline=True,
                mirror=True,
                linecolor='#c7c7c7',
                linewidth=1,
                gridcolor='#e5e7eb',
                zeroline=False,
                ticks='outside',
            )
            save_properties = {
                'toImageButtonOptions': {
                    'format': 'svg', # one of png, svg, jpeg, webp
                    'filename': 'RPE',
                    # 'height': 500,
                    # 'width': 1000,
                    'scale': 10
                }
            }
            pfig.show(config=save_properties)

            fig.set_tight_layout(True)
            if Settings.save_data:
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
            plt.close(fig)

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
            if Settings.save_data:
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
            if Settings.save_data:
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
        circuit_depths=circuit_depths,
        gate_layer=gate_layer,
        loss_angle=loss_angle,
        **kwargs
    )


def IdleRPE(
    qpu:            QPU,
    config:         Config,
    qubit_labels:   Iterable[int],
    circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
    gate_layer:     List = None,
    loss_angle:     str | List[str] | None = None,
    **kwargs
) -> Callable:
    """Robust Phase Estimation for the Idle (I) gate.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[int]): a list specifying sets of system labels
            on which to perform RPE for the idle gate.
        circuit_depths (List[int], optional): a list of positive integers
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32,
            64, 128, 256]```.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.
        loss_angle (str | list[str] | None, optional): a string or list of
            strings specifying which angle to use for calculating the loss from
            RPE. Defaults to None. If None, all angles are used.

    Returns:
        Callable: IdleRPE class instance.
    """
    rpe = type(RPE(
        qpu=qpu,
        config=config,
        qubit_labels=qubit_labels,
        gate='I',
        circuit_depths=circuit_depths,
        gate_layer=gate_layer,
        loss_angle=loss_angle,
        **kwargs
    ))

    class IdleRPE(rpe):
        """RPE protocol for the Idle (I) gate."""

        @save_init
        def __init__(
            self,
            config:         Config,
            qubit_labels:   Iterable[int],
            circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
            gate_layer:     List = None,
            loss_angle:     str | None = None,
            **kwargs
        ) -> None:
            rpe.__init__(
                self,
                config=config,
                qubit_labels=qubit_labels,
                circuit_depths=circuit_depths,
                gate_layer=gate_layer,
                loss_angle=loss_angle,
                **kwargs
            )

    return IdleRPE(
        config=config,
        qubit_labels=qubit_labels,
        circuit_depths=circuit_depths,
        gate_layer=gate_layer,
        loss_angle=loss_angle,
        **kwargs
    )


def X90RPE(
    qpu:            QPU,
    config:         Config,
    qubit_labels:   Iterable[int],
    circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
    gate_layer:     List = None,
    loss_angle:     str | List[str] | None = None,
    **kwargs
) -> Callable:
    """Robust Phase Estimation for the X90 gate.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[int]): a list specifying sets of system labels
            on which to perform RPE for the X90 gate.
        circuit_depths (List[int], optional): a list of positive integers
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32,
            64, 128, 256]```.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.
        loss_angle (str | list[str] | None, optional): a string or list of
            strings specifying which angle to use for calculating the loss from
            RPE. Defaults to None. If None, all angles are used. Possible
            options are `'X'` or `'Z'`.

    Returns:
        Callable: X90RPE class instance.
    """
    rpe = type(RPE(
        qpu=qpu,
        config=config,
        qubit_labels=qubit_labels,
        gate='X90',
        circuit_depths=circuit_depths,
        gate_layer=gate_layer,
        loss_angle=loss_angle,
        **kwargs
    ))

    class X90RPE(rpe):
        """RPE protocol for the X90 gate."""

        @save_init
        def __init__(
            self,
            config:         Config,
            qubit_labels:   Iterable[int],
            circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
            gate_layer:     List = None,
            loss_angle:     str | None = None,
            **kwargs
        ) -> None:
            rpe.__init__(
                self,
                config=config,
                qubit_labels=qubit_labels,
                circuit_depths=circuit_depths,
                gate_layer=gate_layer,
                loss_angle=loss_angle,
                **kwargs
            )

    return X90RPE(
        config=config,
        qubit_labels=qubit_labels,
        circuit_depths=circuit_depths,
        gate_layer=gate_layer,
        loss_angle=loss_angle,
        **kwargs
    )


def CZRPE(
    qpu:            QPU,
    config:         Config,
    qubit_labels:   Iterable[Tuple[int]],
    circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
    gate_layer:     List = None,
    loss_angle:     str | List[str] | None = None,
    **kwargs
) -> Callable:
    """Robust Phase Estimation for the CZ gate.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[Tuple[int]]): a list specifying sets of system labels
            on which to perform RPE for the CZ gate.
        circuit_depths (List[int], optional): a list of positive integers
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32,
            64, 128, 256]```.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.
        loss_angle (str | list[str] | None, optional): a string or list of
            strings specifying which angle to use for calculating the loss from
            RPE. Defaults to None. If None, all angles are used.

    Returns:
        Callable: CZRPE class instance.
    """
    rpe = type(RPE(
        qpu=qpu,
        config=config,
        qubit_labels=qubit_labels,
        gate='CZ',
        circuit_depths=circuit_depths,
        gate_layer=gate_layer,
        loss_angle=loss_angle,
        **kwargs
    ))

    class CZRPE(rpe):
        """RPE protocol for the CZ gate."""

        @save_init
        def __init__(
            self,
            config:         Config,
            qubit_labels:   Iterable[Tuple[int]],
            circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
            gate_layer:     List = None,
            loss_angle:     str | None = None,
            **kwargs
        ) -> None:
            rpe.__init__(
                self,
                config=config,
                qubit_labels=qubit_labels,
                circuit_depths=circuit_depths,
                gate_layer=gate_layer,
                loss_angle=loss_angle,
                **kwargs
            )

    return CZRPE(
        config=config,
        qubit_labels=qubit_labels,
        circuit_depths=circuit_depths,
        gate_layer=gate_layer,
        loss_angle=loss_angle,
        **kwargs
    )


def ZZRPE(
    qpu:            QPU,
    config:         Config,
    qubit_labels:   Iterable[Tuple[int]],
    circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
    gate_layer:     List = None,
    loss_angle:     str | List[str] | None = None,
    **kwargs
) -> Callable:
    """Robust Phase Estimation for the ZZ gate.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[Tuple[int]]): a list specifying sets of system
            labels on which to perform RPE for the ZZ gate.
        circuit_depths (List[int], optional): a list of positive integers
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32,
            64, 128, 256]```.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.
        loss_angle (str | list[str] | None, optional): a string or list of
            strings specifying which angle to use for calculating the loss from
            RPE. Defaults to None. If None, all angles are used.

    Returns:
        Callable: ZZRPE class instance.
    """
    rpe = type(RPE(
        qpu=qpu,
        config=config,
        qubit_labels=qubit_labels,
        gate='ZZ',
        circuit_depths=circuit_depths,
        gate_layer=gate_layer,
        loss_angle=loss_angle,
        **kwargs
    ))

    class ZZRPE(rpe):
        """RPE protocol for the ZZ gate."""

        @save_init
        def __init__(
            self,
            config:         Config,
            qubit_labels:   Iterable[Tuple[int]],
            circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
            gate_layer:     List = None,
            loss_angle:     str | None = None,
            **kwargs
        ) -> None:
            rpe.__init__(
                self,
                config=config,
                qubit_labels=qubit_labels,
                circuit_depths=circuit_depths,
                gate_layer=gate_layer,
                loss_angle=loss_angle,
                **kwargs
            )

    return ZZRPE(
        config=config,
        qubit_labels=qubit_labels,
        circuit_depths=circuit_depths,
        gate_layer=gate_layer,
        loss_angle=loss_angle,
        **kwargs
    )
