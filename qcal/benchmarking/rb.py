"""Submodule for RB routines.

For CRB, see:
https://github.com/sandialabs/pyGSTi/blob/master/jupyter_notebooks/Tutorials/algorithms/RB-CliffordRB.ipynb

For SRB, see:
https://trueq.quantumbenchmark.com/guides/error_diagnostics/srb.html
"""
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pygsti
from IPython.display import clear_output
from plotly.subplots import make_subplots
from pygsti.processors import CliffordCompilationRules as CCR
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import (
    RB,
    CliffordRBDesign,
    DefaultRunner,
    SimultaneousExperimentDesign,
)
from pygsti.protocols.protocol import ProtocolData

from qcal.analysis.leakage import analyze_leakage
from qcal.benchmarking.utils import plot_error_rates
from qcal.circuit import CircuitSet
from qcal.config import Config
from qcal.fitting.fit_functions import base_exponential
from qcal.interface.pygsti.datasets import generate_pygsti_dataset
from qcal.interface.pygsti.processor_spec import pygsti_pspec
from qcal.interface.pygsti.transpiler import PyGSTiTranspiler
from qcal.plotting.utils import calculate_nrows_ncols
from qcal.qpu.qpu import QPU
from qcal.settings import Settings
from qcal.utils import flatten

logger = logging.getLogger(__name__)


def _build_crb_edesign_for_qubit_label(
    ql: int | Tuple[int],
    pspec: QubitProcessorSpec,
    compilations: Any,
    circuit_depths: List[int] | Tuple[int],
    n_circuits: int,
    randomizeout: bool,
    citerations: int,
) -> CliffordRBDesign:
    """
    Build a CRB experiment design for a given qubit label.

    Args:
        ql (int | Tuple[int]): Qubit label (int or tuple of ints).
        pspec (QubitProcessorSpec): PyGSTi processor specification.
        compilations (Any): Clifford compilation rules.
        circuit_depths (List[int] | Tuple[int]): List of circuit depths to test.
        n_circuits (int): Number of circuits per depth.
        randomizeout (bool): Whether to randomize output.
        citerations (int): Number of iterations.

    Returns:
        CliffordRBDesign: CRB experiment design.
    """
    qubits = list(flatten([ql]))
    return CliffordRBDesign(
        pspec=pspec,
        clifford_compilations=compilations,
        depths=circuit_depths,
        circuits_per_depth=n_circuits,
        qubit_labels=[f'Q{q}' for q in qubits],
        randomizeout=randomizeout,
        citerations=citerations,
        add_default_protocol=True,
    )


def CRB(
    qpu:            QPU,
    config:         Config,
    qubit_labels:   List[int | Tuple[int]],
    circuit_depths: List[int] | Tuple[int],
    n_circuits:     int = 30,
    native_gates:   List[str] | None = None,
    pspec:          Any | None = None,
    randomizeout:   bool = True,
    citerations:    int = 5,
    **kwargs
) -> Callable:
    """Clifford Randomized Benchmarking.

    This is a pyGSTi protocol and requires pyGSTi to be installed.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (List[int | Tuple[int]]): a list specifying sets of
            qubit labels to be twirled together. For example, [0, 1, (2, 3)]
            would perform single-qubit RB on 0 and 1, and two-qubit RB on
            (2, 3).
        circuit_depths (List[int] | Tuple[int]):  a list of integers >= 0. The
            CRB depth is the number of Cliffords in the circuit - 2 before
            each Clifford is compiled into the native gate-set.
        n_circuits (int, optional): The number of (possibly) different CRB
            circuits sampled at each depth. Defaults to 30.
        native_gates (List[str] | None, optional): a list of the native gates.
            Example: ['X90', 'Y90']. All Cliffords will be decomposed in terms
            of these gates. If a custom pspec is passed, this list will be
            ignored.
        pspec (QubitProcessorSpec | None, optional): pyGSTi qubit processor
            spec. Defaults to None. If None, a processor spec will be
            automatically generated based on the native_gates and the qubit
            labels.
        randomizeout (bool, optional): whether or not a random Pauli is compiled
            into the output layer of the circuit. Defaults to True. If False,
            the ideal output of the circuits (the "success" or "survival"
            outcome) is always the all-zeros bit string. If True, the ideal
            output a circuit is randomized to a uniformly random bit-string.
        citerations (int, optional): some of the Clifford compilation algorithms
            in pyGSTi (including the default algorithm) are randomized, and the
            lowest-cost circuit is chosen from all the circuit generated in the
            iterations of the algorithm. This is the number of iterations used.
            Defaults to 5. The time required to generate a CRB circuit is
            linear in `citerations * (CRB length + 2)`. Lower-depth / lower
            2-qubit gate count compilations of the Cliffords are important in
            order to successfully implement CRB on more qubits.

    Returns:
        Callable: CRB class instance.
    """

    class CRB(qpu):
        """pyGSTi Clifford RB protocol."""

        def __init__(
            self,
            config:         Config,
            qubit_labels:   List[int | Tuple[int]],
            circuit_depths: List[int] | Tuple[int],
            n_circuits:     int = 30,
            native_gates:   List[str] | None = None,
            pspec:          Any | None = None,
            randomizeout:   bool = True,
            citerations:    int = 5,
            **kwargs
        ) -> None:
            logger.info(f" pyGSTi version: {pygsti.__version__}\n")

            self._qubit_labels = qubit_labels
            self._qubits = list(flatten(qubit_labels))
            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._pspec = pspec
            self._randomizeout = randomizeout
            self._citerations = citerations

            self._pspec = {}
            self._compilations = {}

            self._qtups = []
            for ql in self._qubit_labels:
                if isinstance(ql, (list, tuple)):
                    qtup = (f'Q{q}' for q in ql)
                else:
                    qtup = (f'Q{ql}',)
                self._qtups.append(qtup)

                qubits = list(flatten([ql]))
                if not native_gates:
                    ql_native_gates = ['X90', 'Z90']
                else:
                    ql_native_gates = native_gates

                if isinstance(ql, (list, tuple)):  # Multi-qubit gates
                    if not native_gates:
                        for i in range(len(ql) - 1):  # E.g., (1, 2, 3)
                            ql_native_gates.extend(   # Assumes linear connect.
                                config.native_gates['two_qubit'][
                                    (ql[i], ql[i+1])
                                ]
                            )
                    if pspec is None:
                        self._pspec[ql] = pygsti_pspec(
                            config, qubits, ql_native_gates
                        )
                    self._compilations[ql] = {
                        'absolute': CCR.create_standard(
                            self._pspec[ql], 'absolute',
                            ('paulis', '1Qcliffords'),
                            verbosity=0
                        ),
                        'paulieq': CCR.create_standard(
                            self._pspec[ql], 'paulieq',
                            ('1Qcliffords', 'allcnots'),
                            verbosity=0
                        )
                    }

                else:
                    if pspec is None:
                        self._pspec[ql] = pygsti_pspec(
                            config, qubits, ql_native_gates
                        )
                    self._compilations[ql] = {
                        'absolute': CCR.create_standard(
                            self._pspec[ql], 'absolute',
                            ('paulis', '1Qcliffords'),
                            verbosity=0
                        ),
                        'paulieq': CCR.create_standard(
                            self._pspec[ql], 'paulieq',
                            ('1Qcliffords',),
                            verbosity=0
                        )
                    }

            transpiler = kwargs.get('transpiler', PyGSTiTranspiler())
            kwargs.pop('transpiler', None)
            qpu.__init__(self, config=config, transpiler=transpiler, **kwargs)

            self._sim_RB = True if len(self._qubit_labels) > 1 else False
            self._protocol = (
                DefaultRunner() if self._sim_RB else RB()
            )
            self._edesign = None
            self._data = None
            self._dataset = None
            self._results = None

            self._error_rates = {}
            self._fit_params = {}
            self._success_probabilities = {}
            self._uncertainties = {}

        @property
        def pspec(self):
            """pyGSTi processor spec."""
            return self._pspec

        @property
        def protocol(self):
            """pyGSTi protocol."""
            return self._protocol

        @property
        def edesign(self):
            """pyGSTi edesign."""
            return self._edesign

        @property
        def data(self):
            """pyGSTi data object."""
            return self._data

        @property
        def dataset(self):
            """pyGSTi dataset."""
            return self._dataset

        @property
        def fit_params(self):
            """CRB fit parameters."""
            return self._fit_params

        @property
        def process_infidelity(self):
            """CRB process infidelity."""
            process_infidelity = {}
            for ql, error_rate in self._error_rates.items():
                process_infidelity[ql] = {
                    'val': error_rate,
                    'err': self._uncertainties[ql]
                }
            return process_infidelity

        @property
        def success_probabilities(self):
            """Success probabilities."""
            return self._success_probabilities

        @property
        def results(self):
            """pyGSTi results object."""
            return self._results

        def generate_circuits(self):
            """Generate all pyGSTi clifford RB circuits."""
            logger.info(' Generating circuits from pyGSTi...')

            if not self._sim_RB:
                ql = self._qubit_labels[0]
                self._edesign = CliffordRBDesign(
                        pspec=self._pspec[ql],
                        clifford_compilations=self._compilations[ql],
                        depths=self._circuit_depths,
                        circuits_per_depth=self._n_circuits,
                        qubit_labels=[f'Q{q}' for q in self._qubits],
                        randomizeout=self._randomizeout,
                        citerations=self._citerations,
                    )

            elif self._sim_RB:
                max_workers = min(
                    len(self._qubit_labels), max(1, mp.cpu_count() - 1)
                )
                with ThreadPoolExecutor(
                    max_workers=max_workers
                ) as ex:
                    futures = [
                        ex.submit(
                            _build_crb_edesign_for_qubit_label,
                            ql,
                            self._pspec[ql],
                            self._compilations[ql],
                            self._circuit_depths,
                            self._n_circuits,
                            self._randomizeout,
                            self._citerations,
                        )
                        for ql in self._qubit_labels
                    ]
                    edesigns = [f.result() for f in futures]
                self._edesign = SimultaneousExperimentDesign(
                    edesigns
                )

            self._circuits = CircuitSet(self._edesign.all_circuits_needing_data)
            self._circuits['pygsti_circuit'] = [
                circ.str for circ in self._edesign.all_circuits_needing_data
            ]

            self._data_manager._exp_id += (
                f'_CRB_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if Settings.save_data:
                self._data_manager.create_data_path()
                pygsti.io.write_empty_protocol_data(
                    self._data_manager._save_path,
                    self._edesign,
                    sparse=True,
                    clobber_ok=True
                )

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            if Settings.save_data:
                qpu.save(self, create_data_path=False)

        def analyze(self):
            """Analyze the CRB results."""
            logger.info(' Analyzing the results...')
            self._dataset = generate_pygsti_dataset(
                self._circuits,
                save_path=self._data_manager._save_path + 'data/'
                if Settings.save_data else None
            )
            self._data = ProtocolData(self._edesign, self._dataset)

            if not self._sim_RB:
                try:
                    self._results = self._protocol.run(self._data)
                    self._success_probabilities[self._qubit_labels[0]] = (
                        self._results.data.cache['success_probabilities']
                    )
                    r = self._results.fits['full'].estimates['r']
                    rstd = self._results.fits['full'].stds['r']
                    rA = self._results.fits['A-fixed'].estimates['r']
                    rAstd = self._results.fits['A-fixed'].stds['r']
                    if rAstd < rstd:
                        self._error_rates[self._qubit_labels[0]] = rA
                        self._uncertainties[self._qubit_labels[0]] = rAstd
                        self._fit_params[self._qubit_labels[0]] = {
                            'base': self._results.fits['A-fixed'].estimates['p'],
                            'a': self._results.fits['A-fixed'].estimates['b'],
                            'b': 1,
                            'c': self._results.fits['A-fixed'].estimates['a']
                        }

                    else:
                        self._error_rates[self._qubit_labels[0]] = r
                        self._uncertainties[self._qubit_labels[0]] = rstd
                        self._fit_params[self._qubit_labels[0]] = {
                            'base': self._results.fits['full'].estimates['p'],
                            'a': self._results.fits['full'].estimates['b'],
                            'b': 1,
                            'c': self._results.fits['full'].estimates['a']
                        }

                    print(f'\n{self._qubit_labels[0]}:')
                    print(
                        f"Process infidelity: r = {r:1.2e} ({rstd:1.2e}) "
                        "(fit with a free asymptote)"
                    )
                    print(
                        f"Process infidelity: r = {rA:1.2e} ({rAstd:1.2e}) "
                        "(fit with the asymptote fixed to 1/2^n)"
                    )
                except Exception:
                    logger.warning(' Unable to fit the RB data!')

            elif self._sim_RB:
                results = {}
                fit_success = False
                try:
                    self._results = self._protocol.run(self._data)
                    fit_success = True
                    for qtup in self._qtups:
                        results[qtup] = self._results[qtup].for_protocol['RB']
                except Exception:
                    logger.warning(' Unable to fit the data simultaneously!')

                if not fit_success:
                    for qtup in self._qtups:
                        try:
                            results[qtup] = RB().run(self._data[qtup])
                        except Exception:
                            logger.warning(
                                f' Unable to fit the data for {qtup}!'
                            )
                            results[qtup] = None

                self._results = results
                for i, qtup in enumerate(self._qtups):

                    if self._results[qtup] is not None:
                        self._success_probabilities[self._qubit_labels[i]] = (
                            self._results[qtup].data.cache[
                                'success_probabilities'
                            ]
                        )
                        r = self._results[qtup].fits['full'].estimates['r']
                        rstd = self._results[qtup].fits['full'].stds['r']
                        rA = self._results[qtup].fits['A-fixed'].estimates['r']
                        rAstd = self._results[qtup].fits['A-fixed'].stds['r']
                        if rAstd < rstd:
                            self._error_rates[self._qubit_labels[i]] = rA
                            self._uncertainties[self._qubit_labels[i]] = rAstd
                            self._fit_params[self._qubit_labels[i]] = {
                                'base': self._results[qtup].fits[
                                    'A-fixed'
                                ].estimates['p'],
                                'a': self._results[qtup].fits[
                                    'A-fixed'
                                ].estimates['b'],
                                'b': 1,
                                'c': self._results[qtup].fits[
                                    'A-fixed'
                                ].estimates['a']
                            }
                        else:
                            self._error_rates[self._qubit_labels[i]] = r
                            self._uncertainties[self._qubit_labels[i]] = rstd
                            self._fit_params[self._qubit_labels[i]] = {
                                'base': self._results[qtup].fits[
                                    'full'
                                ].estimates['p'],
                                'a': self._results[qtup].fits[
                                    'full'
                                ].estimates['b'],
                                'b': 1,
                                'c': self._results[qtup].fits[
                                    'full'
                                ].estimates['a']
                            }

                        print(f'\n{self._qubit_labels[i]}:')
                        print(
                            f"Process infidelity: r = {r:1.2e} ({rstd:1.2e}) "
                            "(fit with a free asymptote)"
                        )
                        print(
                            f"Process infidelity: r = {rA:1.2e} ({rAstd:1.2e}) "
                            "(fit with the asymptote fixed to 1/2^n)"
                        )

            if Settings.save_data:
                # self._data_manager.save_to_pickle(self._results, 'CRB_results')
                self._data_manager.save_to_csv(
                    pd.DataFrame(self._success_probabilities),
                    'CRB_success_probabilities'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame(self.process_infidelity),
                    'CRB_process_infidelity'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame(self._fit_params), 'CRB_fit_params'
                )

        def plot(self) -> None:
            """Plot the CRB fit results."""
            if not self._sim_RB:
                if self._results is not None and Settings.save_data:
                    self._results.plot(
                        figpath=self._data_manager._save_path +
                        'CRB_decay.png' if Settings.save_data
                        else None
                    )
                    plt.close("all")

            elif self._sim_RB:
                for qtup in self._qtups:
                    if self._results[qtup] is not None and Settings.save_data:
                        self._results[qtup].plot(
                            figpath=self._data_manager._save_path +
                            f'{"".join(qtup)}_CRB_decay.png'
                            if Settings.save_data else None
                        )
                        plt.close("all")

            if self._results is not None:
                if self._sim_RB:
                    qubit_labels_to_plot = [
                        self._qubit_labels[i]
                        for i, qtup in enumerate(self._qtups)
                        if self._results.get(qtup, None) is not None
                    ]
                else:
                    qubit_labels_to_plot = [self._qubit_labels[0]]

                # qubit_labels_to_plot = [
                #     ql for ql in qubit_labels_to_plot
                #     if ql in self._success_probabilities
                # ]

                if len(qubit_labels_to_plot) > 0:
                    nrows, ncols = calculate_nrows_ncols(
                        len(qubit_labels_to_plot)
                    )

                    pfig_height = 350 * nrows
                    pfig_width = 300 * ncols + 50
                    pfig_margin = {'t': 50, 'b': 50, 'l': 50, 'r': 50}
                    pfig_gap_px = 60
                    vertical_spacing = (
                        0.0 if nrows <= 1 else min(
                            0.2, pfig_gap_px / pfig_height
                        )
                    )
                    horizontal_spacing = (
                        0.0 if ncols <= 1 else min(
                            0.2, pfig_gap_px / pfig_width
                        )
                    )

                    subplot_titles = []
                    for ql in qubit_labels_to_plot:
                        if isinstance(ql, (list, tuple)):
                            ql_str = "".join([f"Q{q}" for q in ql])
                        else:
                            ql_str = f"Q{ql}"

                        er = self._error_rates.get(ql, None)
                        un = self._uncertainties.get(ql, None)
                        if er is not None and un is not None:
                            ql_str = f"{ql_str}<br>r={er:1.2e} ({un:1.2e})"
                        subplot_titles.append(ql_str)

                    pfig = make_subplots(
                        rows=nrows,
                        cols=ncols,
                        subplot_titles=subplot_titles,
                        vertical_spacing=vertical_spacing,
                        horizontal_spacing=horizontal_spacing,
                    )
                    pfig.update_annotations(font_size=12)

                    plotly_blue = '#1f77b4'

                    for k, ql in enumerate(qubit_labels_to_plot):
                        row = (k // ncols) + 1
                        col = (k % ncols) + 1

                        depth_to_probs = self._success_probabilities[ql]
                        depths = sorted(depth_to_probs.keys())
                        means = []

                        for depth in depths:
                            probs = np.asarray(
                                depth_to_probs[depth], dtype=float
                            )
                            probs = probs[np.isfinite(probs)]
                            if probs.size == 0:
                                means.append(np.nan)
                                continue

                            xvals = [depth] * int(probs.size)
                            pfig.add_trace(
                                go.Scatter(
                                    x=xvals,
                                    y=probs,
                                    mode='markers',
                                    marker={
                                        'size': 6,
                                        'opacity': 0.2,
                                        'color': plotly_blue,
                                    },
                                    showlegend=False,
                                ),
                                row=row,
                                col=col,
                            )
                            pfig.add_trace(
                                go.Violin(
                                    x=xvals,
                                    y=probs,
                                    name=str(depth),
                                    showlegend=False,
                                    points=False,
                                    spanmode='hard',
                                    line={'width': 1, 'color': plotly_blue},
                                    opacity=0.25,
                                    fillcolor=plotly_blue,
                                ),
                                row=row,
                                col=col,
                            )
                            means.append(float(np.mean(probs)))

                        depths_arr = np.asarray(depths, dtype=float)
                        means_arr = np.asarray(means, dtype=float)
                        finite_mask = (
                            np.isfinite(depths_arr) & np.isfinite(means_arr)
                        )

                        pfig.add_trace(
                            go.Scatter(
                                x=depths_arr[finite_mask],
                                y=means_arr[finite_mask],
                                mode='markers',
                                marker={'size': 11, 'color': plotly_blue},
                                showlegend=False,
                            ),
                            row=row,
                            col=col,
                        )

                        if ql in self._fit_params and np.any(finite_mask):
                            max_depth = float(np.max(depths_arr[finite_mask]))
                            xfit = np.linspace(
                                0.0,
                                max_depth,
                                200,
                            )
                            yfit = base_exponential(
                                xfit,
                                **self._fit_params[ql],
                            )
                            pfig.add_trace(
                                go.Scatter(
                                    x=xfit,
                                    y=yfit,
                                    mode='lines',
                                    line={'color': plotly_blue, 'width': 2},
                                    name='Fit',
                                    showlegend=(k == 0),
                                ),
                                row=row,
                                col=col,
                            )

                        pfig.update_xaxes(
                            title_text='Circuit Depth' if row == nrows else '',
                            automargin=True,
                            showgrid=True,
                            row=row,
                            col=col,
                        )
                        pfig.update_yaxes(
                            title_text='Success Probability' if col == 1 else '',
                            automargin=True,
                            showgrid=True,
                            row=row,
                            col=col,
                        )

                    pfig.update_layout(
                        height=pfig_height,
                        width=pfig_width,
                        margin=pfig_margin,
                        legend={
                            'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02
                        },
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

            if len(self._error_rates) > 0:
                plot_error_rates(
                    self._error_rates,
                    self._uncertainties,
                    ylabel='Process Infidelity',
                    save_path=self._data_manager._save_path
                    if Settings.save_data else None
                )

            # if any(circ.results.dim == 3 for circ in self._transpiled_circuits):
            #     analyze_leakage(
            #         self._transpiled_circuits,
            #         filename=self._data_manager._save_path
            #     )

        def final(self) -> None:
            """Final benchmarking method."""
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.save()
            self.analyze()
            self.plot()
            self.final()

    return CRB(
        config,
        qubit_labels,
        circuit_depths,
        n_circuits,
        native_gates,
        pspec,
        randomizeout,
        citerations,
        **kwargs
    )


def SRB(
    qpu:            QPU,
    config:         Config,
    qubit_labels:   List[int | Tuple[int]],
    circuit_depths: List[int] | Tuple[int],
    n_circuits:     int = 30,
    tq_config:      str | Any = None,
    compiled_pauli: bool = True,
    include_rcal:   bool = False,
    **kwargs
) -> Callable:
    """Streamlined Randomized Benchmarking.

    This is a True-Q protocol and requires a valid True-Q license.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (List[int | Tuple[int]]): a list specifying sets of
            system labels to be twirled together by Clifford gates in each
            circuit. For example, [0, 1, (2, 3)] would perform single-qubit RB
            on 0 and 1, and two-qubit RB on (2, 3).
        circuit_depths (List[int] | Tuple[int]): a list of positive integers
            specifying how many cycles of random Clifford gates to generate for
            RB, for example, [4, 64, 256].
        n_circuits (int, optional): the number of circuits for each circuit
            depth. Defaults to 30.
        tq_config (str | Any, optional): True-Q config yaml file or config
            object. Defaults to None.
        compiled_pauli (bool, optional): whether or not to compile a random
            Pauli gate for each qubit in the cycle preceding a measurement
            operation. Defaults to True.
        include_rcal (bool, optional): whether to measure RCAL circuits in the
            same circuit collection as the SRB circuit. Defaults to False. If
            True, readout correction will be apply to the fit results
            automatically.

    Returns:
        Callable: SRB class instance.
    """

    class SRB(qpu):
        """True-Q SRB protocol."""
        import trueq as tq

        def __init__(
            self,
            config:          Config,
            qubit_labels:    List[int | Tuple[int]],
            circuit_depths:  List[int] | Tuple[int],
            n_circuits:      int = 30,
            tq_config:       str | tq.Config = None,
            compiled_pauli:  bool = True,
            include_rcal:    bool = False,
            **kwargs
        ) -> None:
            try:
                import trueq as tq

                from qcal.interface.trueq.compiler import TrueqCompiler
                from qcal.interface.trueq.transpiler import TrueqTranspiler
                logger.info(f" True-Q version: {tq.__version__}")
            except ImportError:
                logger.warning(' Unable to import trueq!')

            self._qubit_labels = qubit_labels
            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._compiled_pauli = compiled_pauli
            self._include_rcal = include_rcal

            compiler = kwargs.get(
                'compiler',
                TrueqCompiler(config if tq_config is None else tq_config)
            )
            kwargs.pop('compiler', None)

            transpiler = kwargs.get('transpiler', TrueqTranspiler())
            kwargs.pop('transpiler', None)

            qpu.__init__(self,
                config=config,
                compiler=compiler,
                transpiler=transpiler,
                **kwargs
            )

        def generate_circuits(self):
            """Generate all True-Q SRB circuits."""
            logger.info(' Generating circuits from True-Q...')
            import trueq as tq

            self._circuits = tq.make_srb(
                self._qubit_labels,
                self._circuit_depths,
                self._n_circuits,
                self._compiled_pauli
            )

            if self._include_rcal:
                self._circuits += tq.make_rcal(self._circuits.labels)

            self._circuits.shuffle()

        def analyze(self):
            """Analyze the SRB results."""
            logger.info(' Analyzing the results...')
            print('')
            try:
                print(self._circuits.fit(analyze_dim=2))
            except Exception:
                logger.warning(' Unable to fit the estimate collection!')

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_SRB_{"".join("Q" + str(q) for q in self._circuits.labels)}'
            )
            if Settings.save_data:
                qpu.save(self)

        def plot(self) -> None:
            """Plot the SRB fit results."""

            # Plot the raw curves
            nrows, ncols = calculate_nrows_ncols(len(self._qubit_labels))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            if isinstance(axes, np.ndarray):
                self._circuits.plot.raw(axes=axes.ravel())
            else:
                self._circuits.plot.raw(axes=axes)

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
                        ax.set_title(ax.get_title(), fontsize=20)
                        ax.xaxis.get_label().set_fontsize(15)
                        ax.yaxis.get_label().set_fontsize(15)
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )
                        ax.legend(prop={"size": 12})
                        ax.grid(True)

                    else:
                        ax.axis('off')

            fig.set_tight_layout(True)
            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'SRB_decays.png',
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'SRB_decays.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'SRB_decays.svg'
                )
            plt.show()

            # Plot the RB infidelities
            figsize = (5 * ncols, 4)
            fig, ax = plt.subplots(figsize=figsize, layout='constrained')

            self._circuits.plot.compare_rb(axes=ax)
            ax.set_title(ax.get_title(), fontsize=18)
            ax.xaxis.get_label().set_fontsize(15)
            ax.yaxis.get_label().set_fontsize(15)
            ax.tick_params(
                axis='both', which='major', labelsize=12
            )
            ax.legend(prop={"size": 12})
            ax.grid(True)

            fig.set_tight_layout(True)
            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'SRB_infidelities.png',
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'SRB_infidelities.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'SRB_infidelities.svg'
                )
            plt.show()

            if any(res.dim == 3 for res in self._circuits.results):
                analyze_leakage(
                    self._circuits, filename=self._data_manager._save_path
                )

        def final(self) -> None:
            """Final benchmarking method."""
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.save()
            self.analyze()
            self.plot()
            self.final()

    return SRB(
        config,
        qubit_labels,
        circuit_depths,
        n_circuits,
        tq_config,
        compiled_pauli,
        include_rcal,
        **kwargs
    )
