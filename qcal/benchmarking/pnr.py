"""Submodule for Pauli Noise Reconstruction/Pauli Noise Learning.

"""
import logging
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygsti
import pygsti.algorithms.randomcircuit as random_circuit
from IPython.display import clear_output
from pygsti.baseobjs.label import Label
from pygsti.circuits import Circuit
from pygsti.processors import QubitProcessorSpec
from pygsti.processors.compilationrules import CliffordCompilationRules as CCR
from pygsti.tools import symplectic
from scipy.optimize import curve_fit

from qcal.circuit import CircuitSet
from qcal.config import Config
from qcal.gate.two_qubit import TWO_QUBIT_GATES
from qcal.interface.pygsti.compiler import pauli_randomize_clifford_circuit
from qcal.interface.pygsti.processor_spec import pygsti_pspec
from qcal.interface.pygsti.transpiler import PyGSTiTranspiler
from qcal.qpu.qpu import QPU
from qcal.settings import Settings

from ..benchmarking.cb import _generate_rc_circuits
from .utils import (
    generate_n_qubit_pauli_measurement_map,
    generate_n_qubit_paulis_up_to_weight_k,
)

logger = logging.getLogger(__name__)


__all__ = [
    'KNR',
    'PNLMCM'
]


def KNR(
    qpu:                  QPU,
    config:               Config,
    cycle:                dict,
    circuit_depths:       Iterable[int],
    tq_config:            str | Any = None,
    n_circuits:           int = 30,
    n_subsystems:         int = 2,
    twirl:                str = 'P',
    propogate_correction: bool = False,
    compiled_pauli:       bool = True,
    include_rcal:         bool = False,
    **kwargs
) -> Callable:
    """K-body Noise Reconstruction.

    This is a True-Q protocol and requires a valid True-Q license.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        cycle (dict, tq.Cycle): cycle (or subcircuit) to benchmark.
        circuit_depths (Iterable[int]): iterable of positive integers
            specifying how many interleaved cycles of the target cycle and
            random Pauli operators to generate, for example, [4, 16, 64].
        tq_config (str | Any, optional): True-Q config yaml file or config
            object. Defaults to None.
        n_circuits (int, optional): the number of circuits for each circuit
            depth. Defaults to 30.
        n_subsystems (int, optional): a positive integer specifying the number
            of body errors to estimate on the subset of qubits in the cycle.
            Defaults to 2.
        twirl (tq.Twirl, str, optional): The Twirl to use in this protocol.
            Defaults to 'P'. You can also specify a twirling group that will be
            used to automatically instantiate a twirl based on the labels in
            the given cycles.
        propagate_correction (bool, optional): whether to propagate correction
            gates to the end of the circuit or compile them into neighbouring
            cycles. Defaults to False. Warning: this can result in arbitrary
            multi-qubit gates at the end of the circuit!
        compiled_pauli (bool, optional): whether or not to compile a random
            Pauli gate for each qubit in the cycle preceding a measurement
            operation. Defaults to True.
        include_rcal (bool, optional): whether to measure RCAL circuits in the
            same circuit collection as the SRB circuit. Defaults to False. If
            True, readout correction will be apply to the fit results
            automatically.

    Returns:
        Callable: KNR class instance.
    """

    class KNR(qpu):
        """True-Q KNR protocol."""

        def __init__(
            self,
            config:               Config,
            cycle:                dict,
            circuit_depths:       Iterable[int],
            tq_config:            str | Any = None,
            n_circuits:           int = 30,
            n_subsystems:         int = 2,
            twirl:                str = 'P',
            propogate_correction: bool = False,
            compiled_pauli:       bool = True,
            include_rcal:         bool = False,
            **kwargs
        ) -> None:
            try:
                import trueq as tq

                from qcal.interface.trueq.compiler import TrueqCompiler
                from qcal.interface.trueq.transpiler import TrueqTranspiler
                logger.info(f" True-Q version: {tq.__version__}")
            except ImportError:
                logger.warning(' Unable to import trueq!')

            self._cycle = cycle
            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._n_subsystems = n_subsystems
            self._twirl = twirl
            self._propagate_correction = propogate_correction
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
            """Generate all True-Q KNR circuits."""
            logger.info(' Generating circuits from True-Q...')
            import trueq as tq

            self._circuits = tq.make_knr(
                cycles=self._cycle,
                n_random_cycles=self._circuit_depths,
                n_circuits=self._n_circuits,
                subsystems=self._n_subsystems,
                twirl=self._twirl,
                propagate_correction=self._propagate_correction,
                compiled_pauli=self._compiled_pauli
            )

            if self._include_rcal:
                self._circuits += tq.make_rcal(self._circuits.labels)

            self._circuits.shuffle()

        def analyze(self):
            """Analyze the KNR results."""
            logger.info(' Analyzing the results...')
            print('')
            try:
                for fit in self._circuits.fit(analyze_dim=2):
                    print(fit)
            except Exception:
                logger.warning(' Unable to fit the estimate collection!')

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_KNR_Q{"".join(str(q) for q in self._circuits.labels)}'
            )
            if Settings.save_data:
                qpu.save(self)

        def plot(self) -> None:
            """Plot the KNR fit results."""
            # Plot the raw curves
            self._circuits.plot.raw()
            fig = plt.gcf()
            for ax in fig.axes:
                ax.set_title(ax.get_title(), fontsize=20)
                ax.xaxis.get_label().set_fontsize(15)
                ax.yaxis.get_label().set_fontsize(15)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.grid(True)

            fig.set_tight_layout(True)
            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'KNR_decays.png',
                    dpi=300
                )
            plt.show()

            # Plot the KNR infidelities
            self._circuits.plot.compare_pauli_infidelities()
            fig = plt.gcf()
            for ax in fig.axes:
                ax.set_title(ax.get_title(), fontsize=20)
                ax.xaxis.get_label().set_fontsize(15)
                ax.yaxis.get_label().set_fontsize(15)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.grid(True)

            fig.set_tight_layout(True)
            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'KNR_infidelities.png',
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'KNR_infidelities.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'KNR_infidelities.svg'
                )
            plt.show()

            # Plot the KNR heatmap
            self._circuits.plot.knr_heatmap()
            fig = plt.gcf()
            fig.set_size_inches((8, 8))

            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'KNR_heatmap.png',
                    dpi=300
                )
                fig.savefig(
                    self._data_manager._save_path + 'KNR_heatmap.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'KNR_heatmap.svg'
                )
            plt.show()

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

    return KNR(
        config=config,
        cycle=cycle,
        circuit_depths=circuit_depths,
        tq_config=tq_config,
        n_circuits=n_circuits,
        n_subsystems=n_subsystems,
        twirl=twirl,
        propogate_correction=propogate_correction,
        compiled_pauli=compiled_pauli,
        include_rcal=include_rcal,
        **kwargs
    )


def PNLMCM(
    qpu:            QPU,
    config:         Config,
    # cycle:          Circuit,
    mcm_qubits:     Sequence[int],
    idle_qubits:    Sequence[int],
    circuit_depths: Iterable[int],
    n_circuits:     int = 30,
    max_weight:     int | None = None,
    pspec:          QubitProcessorSpec | None = None,
    **kwargs
) -> Callable:
    """Pauli Noise Learning of Mid-Circuit Measurements.

    This is a pyGSTi protocol.

    Relevant paper:
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.020602

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        # cycle (Circuit): MCM cycle to be benchmarked.
        mcm_qubits (Sequence[int]): qubits measured in the MCM cycle.
        idle_qubits (Sequence[int]): qubits that are idle in the MCM
            cycle (i.e., have no gates applied).
        circuit_depths (Iterable[int]):  a list of integers >= 0
            specifying the CB circuit depths.
        n_circuits (int, optional): The number of different CB circuits sampled
            at each depth. Defaults to 30.
        pspec (QubitProcessorSpec | None, optional): pyGSTi qubit processor
            spec. Defaults to None. If None, a processor spec will be
            automatically generated based on the native_gates and the qubit
            labels.

    Returns:
        Callable: PNLMCM class instance.
    """

    class PNLMCM(qpu):
        """pyGSTi PNL of MCM protocol."""

        def __init__(
            self,
            config:         Config,
            # cycle:          Circuit,
            mcm_qubits:     Sequence[int],
            idle_qubits:    Sequence[int],
            circuit_depths: Iterable[int],
            n_circuits:     int = 30,
            max_weight:     int | None = None,
            pspec:          QubitProcessorSpec | None = None,
            **kwargs
        ) -> None:
            logger.info(f" pyGSTi version: {pygsti.__version__}\n")

            self._qubits = sorted(set(mcm_qubits) | set(idle_qubits))
            self._mcm_qubits = mcm_qubits
            self._idle_qubits = idle_qubits

            cycle_gates = []
            for q in mcm_qubits:
                cycle_gates.append([Label('Iz', f'Q{q}')])
            for q in idle_qubits:
                cycle_gates.append([Label('Gc0', f'Q{q}')])
            self._cycle = pygsti.circuits.Circuit(cycle_gates) # TODO [cycle_gates]
            self._qubit_labels = self._cycle.line_labels

            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._max_weight = max_weight if max_weight is not None else (
                min(2, len(self._qubits))
            )

            gate_set = [f'Gc{i}' for i in range(24)]
            for gate in config.native_gates['set']:
                if gate in TWO_QUBIT_GATES:
                    gate_set.append(gate)
            self._pspec = pspec if pspec is not None else pygsti_pspec(
                config, self._qubits, gate_set
            )

            self._compilations = {
                'absolute': CCR.create_standard(
                    self._pspec, 'absolute',
                    ('paulis', '1Qcliffords'),
                    verbosity=0
                ),
                'paulieq': CCR.create_standard(
                    self._pspec, 'paulieq',
                    ('1Qcliffords', 'allcnots'),
                    verbosity=0
                )
            }

            transpiler = kwargs.get('transpiler', PyGSTiTranspiler())
            kwargs.pop('transpiler', None)
            qpu.__init__(self, config=config, transpiler=transpiler, **kwargs)

            self._pauli_measurements = None
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

        # @property
        # def fit_params(self):
        #     """CRB fit parameters."""
        #     return self._fit_params

        # @property
        # def process_infidelity(self):
        #     """CRB process infidelity."""
        #     process_infidelity = {}
        #     for ql, error_rate in self._error_rates.items():
        #         process_infidelity[ql] = {
        #             'val': error_rate,
        #             'err': self._uncertainties[ql]
        #         }
        #     return process_infidelity

        # @property
        # def success_probabilities(self):
        #     """Success probabilities."""
        #     return self._success_probabilities

        @property
        def results(self):
            """pyGSTi results object."""
            return self._results

        def generate_circuits(self):
            """Generate all pyGSTi circuits."""
            logger.info(' Generating circuits from pyGSTi...')

            paulis = generate_n_qubit_paulis_up_to_weight_k(
                qubits=self._qubits,
                measured_qubits=self._mcm_qubits,
                weight_k=self._max_weight,
                connectivity=self._config.qubit_pairs
            )

            # Remove the all-identity Pauli if it is included
            while tuple(['I'] * len(self._qubits)) in paulis:
                paulis.remove(tuple(['I'] * len(self._qubits)))

            self._pauli_groups = generate_n_qubit_pauli_measurement_map(paulis)
            cs_by_pauli, signs_by_pauli, tbs_by_pauli = _generate_rc_circuits(
                cycle=self._cycle,
                circuit_depths=self._circuit_depths,
                pauli_measurements=self._pauli_groups.keys(),
                compilations=self._compilations,
                n_randomizations=self._n_circuits,
            )

            edesigns = {}
            for pauli, clists in cs_by_pauli.items():
                print(pauli)
                edesigns[pauli] = pygsti.protocols.ByDepthDesign(
                    self._circuit_depths, clists
                )
            self._edesign = pygsti.protocols.CombinedExperimentDesign(edesigns)

            self._circuits = CircuitSet(self._edesign.all_circuits_needing_data)
            self._circuits['pygsti_circuit'] = [
                circ.str for circ in self._edesign.all_circuits_needing_data
            ]

            self._data_manager._exp_id += (
                f'_PNLMCM_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if Settings.save_data:
                self._data_manager.create_data_path()
                pygsti.io.write_empty_protocol_data(
                    self._data_manager._save_path,
                    self._edesign,
                    sparse=True,
                    clobber_ok=True
                )

        # def save(self):
        #     """Save all circuits and data."""
        #     clear_output(wait=True)
        #     if Settings.save_data:
        #         qpu.save(self, create_data_path=False)

        # def analyze(self):
        #     """Analyze the CRB results."""
        #     logger.info(' Analyzing the results...')
        #     # self._dataset = generate_pygsti_dataset(
        #     #     self._circuits,
        #     #     save_path=self._data_manager._save_path + 'data/'
        #     #     if Settings.save_data else None
        #     # )
        #     # self._data = ProtocolData(self._edesign, self._dataset)

        #     # if Settings.save_data:
        #     #     # self._data_manager.save_to_pickle(self._results, 'CRB_results')
        #     #     self._data_manager.save_to_csv(
        #     #         pd.DataFrame(self._success_probabilities),
        #     #         'CRB_success_probabilities'
        #     #     )
        #     #     self._data_manager.save_to_csv(
        #     #         pd.DataFrame(self.process_infidelity),
        #     #         'CRB_process_infidelity'
        #     #     )
        #     #     self._data_manager.save_to_csv(
        #     #         pd.DataFrame(self._fit_params), 'CRB_fit_params'
        #     #     )

        # def plot(self) -> None:
        #     """Plot the CRB fit results."""
        #     if not self._sim_RB:
        #         if self._results is not None and Settings.save_data:
        #             self._results.plot(
        #                 figpath=self._data_manager._save_path +
        #                 'CRB_decay.png' if Settings.save_data
        #                 else None
        #             )
        #             plt.close("all")

        #     elif self._sim_RB:
        #         for qtup in self._qtups:
        #             if self._results[qtup] is not None and Settings.save_data:
        #                 self._results[qtup].plot(
        #                     figpath=self._data_manager._save_path +
        #                     f'{"".join(qtup)}_CRB_decay.png'
        #                     if Settings.save_data else None
        #                 )
        #                 plt.close("all")

        #     if self._results is not None:
        #         if self._sim_RB:
        #             qubit_labels_to_plot = [
        #                 self._qubit_labels[i]
        #                 for i, qtup in enumerate(self._qtups)
        #                 if self._results.get(qtup, None) is not None
        #             ]
        #         else:
        #             qubit_labels_to_plot = [self._qubit_labels[0]]

        #         # qubit_labels_to_plot = [
        #         #     ql for ql in qubit_labels_to_plot
        #         #     if ql in self._success_probabilities
        #         # ]

        #         if len(qubit_labels_to_plot) > 0:
        #             nrows, ncols = calculate_nrows_ncols(
        #                 len(qubit_labels_to_plot)
        #             )

        #             pfig_height = 350 * nrows
        #             pfig_width = 300 * ncols + 50
        #             pfig_margin = {'t': 50, 'b': 50, 'l': 50, 'r': 50}
        #             pfig_gap_px = 60
        #             vertical_spacing = (
        #                 0.0 if nrows <= 1 else min(
        #                     0.2, pfig_gap_px / pfig_height
        #                 )
        #             )
        #             horizontal_spacing = (
        #                 0.0 if ncols <= 1 else min(
        #                     0.2, pfig_gap_px / pfig_width
        #                 )
        #             )

        #             subplot_titles = []
        #             for ql in qubit_labels_to_plot:
        #                 if isinstance(ql, (list, tuple)):
        #                     ql_str = "".join([f"Q{q}" for q in ql])
        #                 else:
        #                     ql_str = f"Q{ql}"

        #                 er = self._error_rates.get(ql, None)
        #                 un = self._uncertainties.get(ql, None)
        #                 if er is not None and un is not None:
        #                     ql_str = f"{ql_str}<br>r={er:1.2e} ({un:1.2e})"
        #                 subplot_titles.append(ql_str)

        #             pfig = make_subplots(
        #                 rows=nrows,
        #                 cols=ncols,
        #                 subplot_titles=subplot_titles,
        #                 vertical_spacing=vertical_spacing,
        #                 horizontal_spacing=horizontal_spacing,
        #             )
        #             pfig.update_annotations(font_size=12)

        #             plotly_blue = '#1f77b4'

        #             for k, ql in enumerate(qubit_labels_to_plot):
        #                 row = (k // ncols) + 1
        #                 col = (k % ncols) + 1

        #                 depth_to_probs = self._success_probabilities[ql]
        #                 depths = sorted(depth_to_probs.keys())
        #                 means = []

        #                 for depth in depths:
        #                     probs = np.asarray(
        #                         depth_to_probs[depth], dtype=float
        #                     )
        #                     probs = probs[np.isfinite(probs)]
        #                     if probs.size == 0:
        #                         means.append(np.nan)
        #                         continue

        #                     xvals = [depth] * int(probs.size)
        #                     pfig.add_trace(
        #                         go.Scatter(
        #                             x=xvals,
        #                             y=probs,
        #                             mode='markers',
        #                             marker={
        #                                 'size': 6,
        #                                 'opacity': 0.2,
        #                                 'color': plotly_blue,
        #                             },
        #                             showlegend=False,
        #                         ),
        #                         row=row,
        #                         col=col,
        #                     )
        #                     pfig.add_trace(
        #                         go.Violin(
        #                             x=xvals,
        #                             y=probs,
        #                             name=str(depth),
        #                             showlegend=False,
        #                             points=False,
        #                             spanmode='hard',
        #                             line={'width': 1, 'color': plotly_blue},
        #                             opacity=0.25,
        #                             fillcolor=plotly_blue,
        #                         ),
        #                         row=row,
        #                         col=col,
        #                     )
        #                     means.append(float(np.mean(probs)))

        #                 depths_arr = np.asarray(depths, dtype=float)
        #                 means_arr = np.asarray(means, dtype=float)
        #                 finite_mask = (
        #                     np.isfinite(depths_arr) & np.isfinite(means_arr)
        #                 )

        #                 pfig.add_trace(
        #                     go.Scatter(
        #                         x=depths_arr[finite_mask],
        #                         y=means_arr[finite_mask],
        #                         mode='markers',
        #                         marker={'size': 11, 'color': plotly_blue},
        #                         showlegend=False,
        #                     ),
        #                     row=row,
        #                     col=col,
        #                 )

        #                 if ql in self._fit_params and np.any(finite_mask):
        #                     max_depth = float(np.max(depths_arr[finite_mask]))
        #                     xfit = np.linspace(
        #                         0.0,
        #                         max_depth,
        #                         200,
        #                     )
        #                     yfit = base_exponential(
        #                         xfit,
        #                         **self._fit_params[ql],
        #                     )
        #                     pfig.add_trace(
        #                         go.Scatter(
        #                             x=xfit,
        #                             y=yfit,
        #                             mode='lines',
        #                             line={'color': plotly_blue, 'width': 2},
        #                             name='Fit',
        #                             showlegend=(k == 0),
        #                         ),
        #                         row=row,
        #                         col=col,
        #                     )

        #                 pfig.update_xaxes(
        #                     title_text='Circuit Depth' if row == nrows else '',
        #                     automargin=True,
        #                     showgrid=True,
        #                     row=row,
        #                     col=col,
        #                 )
        #                 pfig.update_yaxes(
        #                     title_text='Success Probability' if col == 1 else '',
        #                     automargin=True,
        #                     showgrid=True,
        #                     row=row,
        #                     col=col,
        #                 )

        #             pfig.update_layout(
        #                 height=pfig_height,
        #                 width=pfig_width,
        #                 margin=pfig_margin,
        #                 legend={
        #                     'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02
        #                 },
        #                 template='plotly_white',
        #                 paper_bgcolor='white',
        #                 plot_bgcolor='#fbfbfd',
        #             )

        #             pfig.update_xaxes(
        #                 automargin=True,
        #                 showline=True,
        #                 mirror=True,
        #                 linecolor='#c7c7c7',
        #                 linewidth=1,
        #                 gridcolor='#e5e7eb',
        #                 zeroline=False,
        #                 ticks='outside',
        #             )
        #             pfig.update_yaxes(
        #                 automargin=True,
        #                 showline=True,
        #                 mirror=True,
        #                 linecolor='#c7c7c7',
        #                 linewidth=1,
        #                 gridcolor='#e5e7eb',
        #                 zeroline=False,
        #                 ticks='outside',
        #             )

        #             save_properties = {
        #                 'toImageButtonOptions': {
        #                     'format': 'svg', # one of png, svg, jpeg, webp
        #                     'filename': 'RPE',
        #                     # 'height': 500,
        #                     # 'width': 1000,
        #                     'scale': 10
        #                 }
        #             }
        #             pfig.show(config=save_properties)

        #     if len(self._error_rates) > 0:
        #         plot_error_rates(
        #             self._error_rates,
        #             self._uncertainties,
        #             ylabel='Process Infidelity',
        #             save_path=self._data_manager._save_path
        #             if Settings.save_data else None
        #         )

        #     # if any(circ.results.dim == 3 for circ in self._transpiled_circuits):
        #     #     analyze_leakage(
        #     #         self._transpiled_circuits,
        #     #         filename=self._data_manager._save_path
        #     #     )

        # def final(self) -> None:
        #     """Final benchmarking method."""
        #     print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        # def run(self):
        #     """Run all experimental methods and analyze results."""
        #     self.generate_circuits()
        #     qpu.run(self, self._circuits, save=False)
        #     self.save()
        #     self.analyze()
        #     self.plot()
        #     self.final()

    return PNLMCM(
        config=config,
        # cycle=cycle,
        mcm_qubits=mcm_qubits,
        idle_qubits=idle_qubits,
        circuit_depths=circuit_depths,
        n_circuits=n_circuits,
        max_weight=max_weight,
        pspec=pspec,
        **kwargs
    )


def _avg_energy(
    cd: Dict[Tuple[str], int], measurement: str, sign: int, tbs: str
) -> float:
    """Compute average energy from a counts dictionary.

    Args:
        cd (Dict[Tuple[str], int]): counts dictionary whose keys are tuples with
            the outcome bitstring as the first element.
        measurement (str): Pauli measurement string (e.g., "ZZI").
        sign (int): sign (+1 or -1) associated with the stabilizer state.
        tbs (str): twirl bit-string.

    Returns:
        float: average signed energy.
    """
    energy = 0
    total = sum(cd.values())
    for key, count in cd.items():
        out_eng = _outcome_energy(key[0], measurement, sign, tbs)
        energy += count * out_eng

    return energy / total


def _avg_energy_sign_mod(
    cd: Dict[Tuple[Any, ...], int],
    measurement: str,
    sign: int,
    tbs: str,
    measured_qs: List[int] | Tuple[int, ...] = (),
    toggled_qs: List[int] | Tuple[int, ...] = (),
) -> float:
    """Average energy with an additional sign flip computed from MCM outcomes.

    This is a variant of :func:`_avg_energy` for experiments that include
    mid-circuit measurement (MCM) registers in the result key.

    Args:
        cd (Dict[Tuple[Any, ...], int]): counts dictionary. Keys may include
            MCM register entries followed by the final measurement outcome.
        measurement (str): Pauli measurement string (e.g., "ZZI").
        sign (int): stabilizer sign (+1 or -1).
        tbs (str): twirl bit-string.
        measured_qs (List[int] | Tuple[int, ...]): qubits measured in the MCM
            registers (used to interpret which bits to include in the sign).
        toggled_qs (List[int] | Tuple[int, ...]): subset of ``measured_qs`` that
            contribute to sign toggling.

    Returns:
        float: average signed energy.
    """
    energy = 0
    total = sum(cd.values())
    if total == 0:
        return 0.0

    measured_qs_list = list(measured_qs)
    toggled_qs_set = set(toggled_qs)

    for key, count in cd.items():
        this_sign = 1
        if (
            len(key) > 1
            and len(toggled_qs_set) > 0
            and len(measured_qs_list) > 0
        ):
            mcm_results = [0 if b == 'p0' else 1 for b in key[:-1]]
            counted_mcm_results = [
                1 if q in toggled_qs_set else 0 for q in measured_qs_list
            ] * (len(mcm_results) // len(measured_qs_list))
            mcm_tbs = [
                0 if tbs[j] == '0' else 1 for j in range(len(mcm_results))
            ]
            mcm_adjusted_results = np.dot(
                np.logical_xor(mcm_results, mcm_tbs), counted_mcm_results
            )
            this_sign = (-1) ** (mcm_adjusted_results % 2)

        out_eng = (
            _outcome_energy(key[-1], measurement, sign, tbs[-len(measurement):])
            * this_sign
        )
        energy += count * out_eng

    return energy / total


def _bootstrap_error(
    data_by_depth: List[Any],
    depths: List[int] | Tuple[int, ...],
    bootstrap_samples: int = 100,
    n_shots: int = 10000,
) -> float:
    """Estimate uncertainty on the fitted decay parameter via bootstrapping.

    Args:
        data_by_depth (List[Any]): list of energies (or list of energies) per
            depth.
        depths (List[int] | Tuple[int, ...]): circuit depths corresponding to
            ``data_by_depth``.
        bootstrap_samples (int): number of bootstrap resamples.
        n_shots (int): number of binomial shots used for each bootstrap
            replicate.

    Returns:
        float: standard deviation of the bootstrapped decay parameter ``b``.
    """
    rs = []
    for _ in range(bootstrap_samples):
        new_ps: Dict[int, List[float]] = {}
        for energies, d in zip(data_by_depth, depths, strict=True):
            if type(energies) is not list:
                energies = [energies]
            probs = [(e + 1) / 2 for e in energies]
            sampled_probs = np.random.choice(probs, len(energies))

            new_ps[int(d)] = []
            for p in sampled_probs:
                outcomes = np.random.binomial(1, p, size=n_shots)
                new_p = float(sum(outcomes) / len(outcomes))
                new_ps[int(d)].append(new_p)

        mean_sps = [2 * np.mean(new_ps[int(d)]) - 1 for d in depths]
        result = curve_fit(_decay_form, list(depths), mean_sps, p0=[1, 0.99])[0]
        rs.append(result[-1])

    return float(np.std(rs))


def _compute_eigenvalue_decays(
    data_by_pauli: Dict[Tuple[str, ...], Any],
    cs_by_pauli: Dict[Tuple[str, ...], Any],
    signs_by_pauli: Dict[Tuple[str, ...], Any],
    tbs_by_pauli: Dict[Tuple[str, ...], Any],
) -> Tuple[
    Dict[Tuple[str, ...], List[float]],
    Dict[Tuple[str, ...], List[List[float]]],
]:
    """Compute per-Pauli decay curves from raw per-circuit data.

    Args:
        data_by_pauli (Dict[Tuple[str, ...], Any]): mapping from Pauli string to
            datasets by depth.
        cs_by_pauli (Dict[Tuple[str, ...], Any]): Mapping from Pauli string to
            compiled circuits by depth.
        signs_by_pauli (Dict[Tuple[str, ...], Any]): Mapping from Pauli string
            to stabilizer signs.
        tbs_by_pauli (Dict[Tuple[str, ...], Any]): Mapping from Pauli string to
            twirl bit-strings.

    Returns:
        Tuple of:
            - energies_by_pauli: average energy vs depth for each Pauli.
            - circuit_energies_by_pauli: per-circuit energies for each depth.
    """
    energies_by_pauli: Dict[Tuple[str, ...], List[float]] = {}
    circuit_energies_by_pauli: Dict[Tuple[str, ...], List[List[float]]] = {}

    for pauli, ds_by_d in data_by_pauli.items():
        circuits = cs_by_pauli[pauli]
        signs = signs_by_pauli[pauli]
        tbss = tbs_by_pauli[pauli]

        meas_pauli = ''.join([p if p in ['I', 'Z'] else 'Z' for p in pauli])
        circuit_energies_by_pauli[pauli] = []
        avg_energies: List[float] = []

        for clist, signlist, tbslist, ds in zip(
            circuits, signs, tbss, ds_by_d, strict=True
        ):
            circuit_energies: List[float] = []
            for c, sgn, tbs in zip(clist, signlist, tbslist, strict=True):
                dsrow = ds[c]
                cd = _ignore_mcm_results(dsrow.to_dict())
                circuit_energies.append(_avg_energy(cd, meas_pauli, sgn, tbs))

            avg_energies.append(float(np.mean(circuit_energies)))
            circuit_energies_by_pauli[pauli].append(circuit_energies)

        energies_by_pauli[pauli] = avg_energies

    return energies_by_pauli, circuit_energies_by_pauli


def _compute_toggle_decays(
    data_by_pauli: Dict[Tuple[str, ...], Any],
    cs_by_pauli: Dict[Tuple[str, ...], Any],
    signs_by_pauli: Dict[Tuple[str, ...], Any],
    tbs_by_pauli: Dict[Tuple[str, ...], Any],
    measured_qs: List[int] | Tuple[int, ...] = (),
    toggled_qs: List[int] | Tuple[int, ...] = (),
) -> Tuple[
    Dict[Tuple[str, ...], List[float]],
    Dict[Tuple[str, ...], List[List[float]]],
]:
    """Compute decay curves that incorporate MCM-derived sign toggles.

    Args:
        data_by_pauli (Dict[Tuple[str, ...], Any]): mapping from Pauli string to
            datasets by depth.
        cs_by_pauli (Dict[Tuple[str, ...], Any]): mapping from Pauli string to
            compiled circuits by depth.
        signs_by_pauli (Dict[Tuple[str, ...], Any]): mapping from Pauli string
            to stabilizer signs.
        tbs_by_pauli (Dict[Tuple[str, ...], Any]): mapping from Pauli string to
            twirl bit-strings.
        measured_qs (List[int] | Tuple[int, ...]): qubits that appear in MCM
            registers.
        toggled_qs (List[int] | Tuple[int, ...]): subset of ``measured_qs`` that
            contribute to sign toggling.

    Returns:
        Tuple of:
            - energies_by_pauli: average energy vs depth for each Pauli.
            - circuit_energies_by_pauli: per-circuit energies for each depth.
    """
    energies_by_pauli: Dict[Tuple[str, ...], List[float]] = {}
    circuit_energies_by_pauli: Dict[Tuple[str, ...], List[List[float]]] = {}

    for pauli, ds_by_d in data_by_pauli.items():
        circuits = cs_by_pauli[pauli]
        signs = signs_by_pauli[pauli]
        tbss = tbs_by_pauli[pauli]

        circuit_energies_by_pauli[pauli] = []
        meas_pauli = ''.join([p if p in ['I', 'Z'] else 'Z' for p in pauli])
        avg_energies: List[float] = []

        for clist, signlist, tbslist, ds in zip(
            circuits, signs, tbss, ds_by_d, strict=False
        ):
            circuit_energies: List[float] = []
            for c, sgn, tbs in zip(clist, signlist, tbslist, strict=False):
                dsrow = ds[c]
                circuit_energies.append(
                    _avg_energy_sign_mod(
                        dsrow.to_dict(),
                        meas_pauli,
                        sgn,
                        tbs,
                        measured_qs=measured_qs,
                        toggled_qs=toggled_qs,
                    )
                )

            avg_energies.append(float(np.mean(circuit_energies)))
            circuit_energies_by_pauli[pauli].append(circuit_energies)

        energies_by_pauli[pauli] = avg_energies

    return energies_by_pauli, circuit_energies_by_pauli


def _decay_form(x: np.ndarray, A: float, b: float) -> np.ndarray:
    """Exponential decay model used for fitting.

    Args:
        x (np.ndarray): depths / independent variable.
        A (float): amplitude.
        b (float): base of exponential decay.

    Returns:
        np.ndarray: model values ``A * (b ** x)``.
    """
    return A * (b**x)


def _determine_new_sign(
    circuit: Circuit, tbs: str, pspec: QubitProcessorSpec, meas_pauli: str
) -> Tuple[int, str]:
    """
    Determine the new sign and twirl bit-string for a given circuit.

    Args:
        circuit (Circuit): the circuit to determine the new sign and twirl
            bit-string for.
        tbs (str): the twirl bit-string.
        pspec (QubitProcessorSpec): the qubit processor specification.
        meas_pauli (str): the measurement Pauli string.

    Returns:
        Tuple[int, str]: the new sign and twirl bit-string.
    """
    n = circuit.width

    # Remove measured qubit
    rest_c = circuit.copy(editable=True)
    rest_c = rest_c.replace_gatename_with_idle('Iz')
    init_stab, init_phase = symplectic.prep_stabilizer_state(
        len(rest_c.line_labels)
    )
    s_c, p_c = symplectic.symplectic_rep_of_clifford_circuit(
        rest_c,
        pspec=pspec.subset(
            gate_names_to_include='all',
            qubit_labels_to_keep=rest_c.line_labels,
        ),
    )
    s_state, p_state = symplectic.compose_cliffords(
        s1=init_stab, p1=init_phase, s2=s_c, p2=p_c
    )

    # Measure at end, compute the new sign
    meas = random_circuit._measure(s_state, p_state)  # list of 0 and 1 outcomes
    meas_to_count = [0 if p == 'I' else 1 for p in meas_pauli]
    new_sign = (-1) ** (np.dot(meas, meas_to_count))
    new_tbs = tbs[:-n] + '0' * n

    return new_sign, new_tbs


def _ignore_mcm_results(
    count_dict: Dict[Tuple[Any, ...], int]
) -> Dict[Tuple[Any], int]:
    """Reduce count keys by keeping only the final element.

    This is useful when results include mid-circuit measurement (MCM) registers
    in addition to the final measurement outcome.

    Args:
        count_dict (Dict[Tuple[Any, ...], int]): counts dictionary.

    Returns:
        Dict[Tuple[Any], int]: counts dictionary keyed only by the last element
            of the original key tuple.
    """
    new_dict: Dict[Tuple[Any], int] = {}
    for key, count in count_dict.items():
        new_key = (key[-1],)
        if new_key in new_dict:
            new_dict[new_key] += count
        else:
            new_dict[new_key] = count

    return new_dict


def _outcome_energy(outcome: str, measurement: str, sign: int, tbs: str) -> int:
    """Compute the signed energy contribution of a single outcome.

    Args:
        outcome (str): measurement outcome bitstring.
        measurement (str): Pauli measurement string (e.g., "ZZI").
        sign (int): sign (+1 or -1) associated with the stabilizer state.
        tbs (str): twirl bit-string.

    Returns:
        int: signed energy (+1 or -1).
    """
    energy = 1
    for out_bit, meas_axis, twirl_bit in zip(
        outcome, measurement, tbs, strict=True
    ):
        if meas_axis == 'Z' and (out_bit != twirl_bit):
            energy = -1 * energy

    return sign * energy
