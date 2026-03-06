"""Submodule for performing Gate Set Tomography (GST)

Relevant literature:
- https://quantum-journal.org/papers/q-2021-10-05-557/

Relevant code repos:
- https://www.pygsti.info/
- https://github.com/sandialabs/pyGSTi
"""
import logging
import multiprocessing as mp
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import plotly.graph_objects as go
import pygsti
import pygsti.report.reportables as metrics
from IPython.display import clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from pygsti.algorithms.fiducialpairreduction import (
    find_sufficient_fiducial_pairs_per_germ_greedy,
)
from pygsti.algorithms.fiducialselection import find_fiducials
from pygsti.algorithms.germselection import find_germs
from pygsti.circuits.circuit import Circuit
from pygsti.data import DataSet
from pygsti.io import read_dataset, write_dataset, write_empty_protocol_data
from pygsti.modelmembers.instruments import Instrument
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYCPHASE
from pygsti.models.explicitmodel import ExplicitOpModel
from pygsti.models.modelconstruction import create_explicit_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import StandardGST, StandardGSTDesign
from pygsti.protocols.gst import ModelEstimateResults
from pygsti.protocols.protocol import ProtocolData
from pygsti.tools.internalgates import standard_gatename_unitaries

from qcal.circuit import CircuitSet
from qcal.config import Config
from qcal.interface.pygsti.datasets import generate_pygsti_dataset
from qcal.interface.pygsti.transpiler import PyGSTiTranspiler
from qcal.plotting.utils import calculate_nrows_ncols
from qcal.post_processing.passes import (
    compute_conditional_counts,
    discard_heralded_shots,
    relabel_esp,
)
from qcal.post_processing.post_process import PostProcessor
from qcal.qpu.qpu import QPU
from qcal.results import Results
from qcal.settings import Settings
from qcal.utils import flatten, load_from_pickle, save_init, save_to_pickle

logger = logging.getLogger(__name__)


def GST(
    qpu:            QPU,
    config:         Config,
    qubit_labels:   Iterable[int | Tuple[int]],
    pspec:          Any | None = None,
    target_model:   Any | None = None,
    prep_fiducials: Any | None = None,
    meas_fiducials: Any | None = None,
    germs:          Any | None = None,
    circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
    modes:          Tuple[str] = ('full TP','CPTPLND','Target','H+S','S'),  # noqa: B006
    fpr:            bool = False,
    **kwargs
) -> Callable:
    """Gate Set Tomography.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[int | Tuple[int]]): a list specifying sets of
            system labels on which to perform GST.
        pspec (Any | Dict[int, Any] | None, optional): a pyGSTi ProcessorSpec
            object. Defaults to None.
        target_model (Any | Dict[int, Any] | None, optional): a pyGSTi Model
            object. Defaults to None.
        prep_fiducials (Any | Dict[int, Any] | None, optional): a list of pyGSTi
            fiducial circuits. Defaults to None.
        meas_fiducials (Any | Dict[int, Any] | None, optional): a list of pyGSTi
            fiducial circuits. Defaults to None.
        germs (Any | Dict[int, Any] | None, optional): a list of pyGSTi germ
            circuits. Defaults to None.
        circuit_depths (List[int], optional): a list of positive integers
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32,
            64, 128, 256]```.
        modes (Tuple[str], optional): a tuple of strings specifying the modes
            to be used in the GST protocol. Defaults to ```('full TP',
            'CPTPLND', 'Target', 'H+S', 'S')```. These correspond to different
            types of parameterizations/constraints to apply to the estimated
            model. Allowed values are:
            - 'full': full (completely unconstrained)
            - 'TP': TP-constrained
            - 'CPTPLND': Lindbladian CPTP-constrained
            - 'H+S': Only Hamiltonian + Stochastic errors allowed (CPTP)
            - 'S': Only Stochastic errors allowed (CPTP)
            - 'Target': use the target (ideal) gates as the estimate
            - <model>: any key in the models_to_test argument
        fpr (bool, optional): whether to use Fiducial Pair Reduction (FPR).
            Defaults to False.

    Returns:
        Callable: GST class instance.
    """

    class GST(qpu):
        """GST protocol."""

        @save_init
        def __init__(
            self,
            config:         Config,
            qubit_labels:   Iterable[int | Tuple[int]],
            pspec:          Any | None = None,
            target_model:   Any | None = None,
            prep_fiducials: Any | None = None,
            meas_fiducials: Any | None = None,
            germs:          Any | None = None,
            circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
            modes:          Tuple[str] = (
                'full TP', 'CPTPLND', 'Target', 'H+S', 'S'
            ),
            fpr:            bool = False,
            **kwargs
        ) -> None:
            logger.info(f" pyGSTi version: {pygsti.__version__}\n")

            self._qubit_labels = qubit_labels
            self._qubits = sorted(flatten(qubit_labels))
            self._pspec = pspec
            self._target_model = target_model
            self._prep_fiducials = prep_fiducials
            self._meas_fiducials = meas_fiducials
            self._germs = germs
            self._circuit_depths = circuit_depths
            self._modes = modes
            self._fpr = fpr

            self._protocol = None
            self._edesign = None
            self._data = None
            self._dataset = None
            self._results = None
            self._circuits = None
            self._report = None

            transpiler = kwargs.get('transpiler', PyGSTiTranspiler())
            kwargs.pop('transpiler', None)
            qpu.__init__(self, config=config, transpiler=transpiler, **kwargs)

        @property
        def avg_gate_infidelity(self) -> Dict[str, Dict[str, float]]:
            """Average gate infidelity for the gates in the gate set.

            Returns:
                Dict[str, Dict[str, float]]: average gate infidelity for each
                    gate in the gate set for each model.
            """
            infidelity = {}
            for mode, model in self.models.items():
                infidelity[mode] = {
                    str(gate): metrics.avg_gate_infidelity(
                        self.target_model.operations[gate].to_dense(),
                        op.to_dense(),
                        'pp'
                    )
                    for gate, op in model.operations.items()
                }
            return infidelity

        @property
        def circuit_depths(self) -> List[int]:
            """GST circuit depths.

            Returns:
                List[int]: GST max circuit depths.
            """
            return self._circuit_depths

        @property
        def data(self) -> ProtocolData:
            """pyGSTi data object.

            Returns:
                ProtocolData: pyGSTi data object.
            """
            return self._data

        @property
        def dataset(self) -> DataSet:
            """pyGSTi dataset object.

            Returns:
                DataSet: pyGSTi dataset object.
            """
            return self._dataset

        @property
        def diamond_norm(self) -> Dict[str, Dict[str, float]]:
            """Diamond norm for the gates in the gate set.

            Returns:
                Dict[str, Dict[str, float]]: diamond norm for each gate in the
                    gate set for each model.
            """
            diamondnorm = {}
            for mode, model in self.models.items():
                dn_ops = {
                    str(gate): metrics.half_diamond_norm(
                        self.target_model.operations[gate].to_dense(),
                        op.to_dense(),
                        'pp'
                    )
                    for gate, op in model.operations.items()
                }

                instruments = getattr(model, 'instruments', {})
                if instruments and hasattr(self.target_model, 'instruments'):
                    dn_instr = {
                        f'{instr}': metrics.half_diamond_norm(
                            sum(
                                effect.to_dense() for effect in effects.values()
                            ),
                            sum(
                                self.target_model.instruments[
                                    instr
                                ][p].to_dense()
                                for p in effects.keys()
                            ),
                            'pp'
                        )
                        for instr, effects in instruments.items()
                    }
                    diamondnorm[mode] = dn_ops | dn_instr
                else:
                    diamondnorm[mode] = dn_ops

            return diamondnorm

        @property
        def edesign(self) -> StandardGSTDesign:
            """pyGSTi edesign.

            Returns:
                StandardGSTDesign: pyGSTi edesign.
            """
            return self._edesign

        @property
        def eigenvalue_avg_gate_infidelity(self) -> Dict[str, Dict[str, float]]:
            """Eigenvalue average gate infidelity for the gates in the gate set.

            Returns:
                Dict[str, Dict[str, float]]: eigenvalue average gate infidelity
                    for each gate in the gate set for each model.
            """
            infidelity = {}
            for mode, model in self.models.items():
                infidelity[mode] = {
                    str(gate): metrics.eigenvalue_avg_gate_infidelity(
                        self.target_model.operations[gate].to_dense(),
                        op.to_dense(),
                        'pp'
                    )
                    for gate, op in model.operations.items()
                }
            return infidelity

        @property
        def eigenvalue_diamond_norm(self) -> Dict[str, Dict[str, float]]:
            """Eigenvalue diamond norm for the gates in the gate set.

            Returns:
                Dict[str, Dict[str, float]]: eigenvalue diamond norm for each
                    gate in the gate set for each model.
            """
            diamondnorm = {}
            for mode, model in self.models.items():
                diamondnorm[mode] = {
                    str(gate): metrics.eigenvalue_diamondnorm(
                        self.target_model.operations[gate].to_dense(),
                        op.to_dense(),
                        'pp'
                    )
                    for gate, op in model.operations.items()
                }
            return diamondnorm

        @property
        def eigenvalue_entanglement_infidelity(self) -> Dict[str, Dict[str, float]]:
            """Eigenvalue entanglement infidelity for the gates in the gate set.

            Returns:
                Dict[str, Dict[str, float]]: eigenvalue entanglement infidelity
                    for each gate in the gate set for each model.
            """
            infidelity = {}
            for mode, model in self.models.items():
                infidelity[mode] = {
                    str(gate): metrics.eigenvalue_entanglement_infidelity(
                        self.target_model.operations[gate].to_dense(),
                        op.to_dense(),
                        'pp'
                    )
                    for gate, op in model.operations.items()
                }
            return infidelity

        @property
        def entanglement_infidelity(self) -> Dict[str, Dict[str, float]]:
            """Entanglement infidelity for the gates in the gate set.

            Returns:
                Dict[str, Dict[str, float]]: entanglement infidelity for each
                    gate in the gate set for each model.
            """
            infidelity = {}
            for mode, model in self.models.items():
                ei_ops = {
                    str(gate): metrics.entanglement_infidelity(
                        self.target_model.operations[gate].to_dense(),
                        op.to_dense(),
                        'pp'
                    )
                    for gate, op in model.operations.items()
                }

                instruments = getattr(model, 'instruments', {})
                if instruments and hasattr(self.target_model, 'instruments'):
                    ei_instr = {
                        f'{instr}': metrics.entanglement_infidelity(
                            sum(
                                effect.to_dense() for effect in effects.values()
                            ),
                            sum(
                                self.target_model.instruments[instr][p].to_dense()
                                for p in effects.keys()
                            ),
                            'pp'
                        )
                        for instr, effects in instruments.items()
                    }
                    infidelity[mode] = ei_ops | ei_instr
                else:
                    infidelity[mode] = ei_ops

            return infidelity

        @property
        def germs(self) -> List[Circuit]:
            """GST germs.

            Returns:
                List[Circuit]: GST germs.
            """
            return self._germs

        @property
        def jtrace_diff(self) -> Dict[str, Dict[str, float]]:
            """Jamiolkowski trace distance for the gates in the gate set.

            Returns:
                Dict[str, Dict[str, float]]: Jamiolkowski trace distance for
                    each gate in the gate set for each model.
            """
            jtrace_diff = {}
            for mode, model in self.models.items():
                jtrace_diff[mode] = {
                    str(gate): metrics.jtrace_diff(
                        self.target_model.operations[gate].to_dense(),
                        op.to_dense(),
                        'pp'
                    )
                    for gate, op in model.operations.items()
                }
            return jtrace_diff

        @property
        def meas_fiducials(self) -> List[Circuit]:
            """GST measurement fiducials.

            Returns:
                List[Circuit]: GST measurement fiducials.
            """
            return self._meas_fiducials

        @property
        def models(self) -> Dict[str, ExplicitOpModel]:
            """pyGSTi models object.

            Returns:
                Dict[str, ExplicitOpModel]: pyGSTi models object.
            """
            return {
                mode: self._results.estimates[mode].models['stdgaugeopt']
                for mode in self._modes
            }

        @property
        def modes(self) -> Tuple[str]:
            """GST model types.

            Returns:
                Tuple[str]: GST modes.
            """
            return self._modes

        @property
        def POVM(self) -> Dict[str, Dict[str, NDArray]]:
            """Estimated Hilbert-Schmidt vectors for the POVM effects.

            Returns:
                Dict[str, Dict[str, NDArray]]: POVM effects.
            """
            return {
                mode: {
                    effect: model.effects[effect].to_dense()
                    for effect in model.effects
                }
                for mode, model in self.models.items()
            }

        @property
        def prep_fiducials(self) -> List[Circuit]:
            """GST preparation fiducials.

            Returns:
                List[Circuit]: GST preparation fiducials.
            """
            return self._prep_fiducials

        @property
        def process_infidelity(self) -> Dict[str, Dict[str, float]]:
            """Process infidelity for the gates in the gate set.

            Returns:
                Dict[str, Dict[str, float]]: process infidelity for each
                    gate in the gate set for each model.
            """
            return self.entanglement_infidelity

        @property
        def protocol(self) -> Any:
            """pyGSTi protocol."""
            return self._protocol

        @property
        def pspec(self) -> QubitProcessorSpec:
            """pyGSTi processor spec.

            Returns:
                QubitProcessorSpec: pyGSTi processor spec.
            """
            return self._pspec

        @property
        def ptm(self) -> Dict[str, Dict[str, NDArray]]:
            """Pauli Transfer Matrices for each gate in the gate set.

            Returns:
                Dict[str, Dict[str, NDArray]]: PTM for each gate in the gate
                    set for each fitted model.
            """
            ptm = {}
            for mode, model in self.models.items():
                ptm_ops = {
                    str(gate): op.to_dense()
                    for gate, op in model.operations.items()
                }

                instruments = getattr(model, 'instruments', {})
                if instruments:
                    ptm_instr = {
                        f'{instr}.{p}': effect.to_dense()
                        for instr, effects in instruments.items()
                        for p, effect in effects.items()
                    }
                    ptm[mode] = ptm_ops | ptm_instr
                else:
                    ptm[mode] = ptm_ops

            return ptm

        @property
        def qubits(self) -> List[int]:
            """All qubits used in the GST experiment."""
            return self._qubits

        @property
        def qubit_labels(self) -> List[int | Tuple[int, int]]:
            """Qubit labels used in the GST experiment."""
            return self._qubit_labels

        @property
        def results(self) -> ModelEstimateResults:
            """pyGSTi results object.

            Returns:
                ModelEstimateResults: pyGSTi results object.
            """
            return self._results

        @property
        def state_prep(self) -> Dict[str, NDArray]:
            """Estimated Hilbert-Schmidt vector for the initial state rho_0.

            Returns:
                Dict[str, NDArray]: rho_0.
            """
            return {
                mode: model.prep.to_dense()
                for mode, model in self.models.items()
            }

        @property
        def state_prep_fidelity(self) -> Dict[str, float]:
            """State fidelity for the initial state rho_0.

            Returns:
                Dict[str, float]: state fidelity for each model.
            """
            from pygsti.tools import ppvec_to_stdmx
            from pygsti.tools.optools import fidelity
            return {
                mode: fidelity(
                    ppvec_to_stdmx(self._target_model.prep.to_dense()),
                    ppvec_to_stdmx(model.prep.to_dense()),
                )
                for mode, model in self.models.items()
            }

        @property
        def target_model(self) -> ExplicitOpModel:
            """GST target model.

            Returns:
                ExplicitOpModel: GST target model.
            """
            return self._target_model

        @property
        def unitarity(self) -> Dict[str, float]:
            """Unitarity for each gate in the gate set.

            Returns:
                Dict[str, float]: unitarity for each gate in the gate set for
                    each model.
            """
            from pygsti.tools.optools import unitarity
            uni = {}
            for mode, model in self.models.items():
                uni[mode] = {
                    str(gate): unitarity(op.to_dense())
                    for gate, op in model.operations.items()
                }
            return uni

        def generate_circuits(self):
            """Generate all GST circuits."""
            print("Prep fiducials:\n", self._prep_fiducials)
            print("Meas fiducials:\n", self._meas_fiducials)
            print("Germs:\n", self._germs)

            self._protocol = StandardGST(
                modes=self._modes,
                target_model=self._target_model
            )

            if self._fpr:
                fiducial_pairs = find_sufficient_fiducial_pairs_per_germ_greedy(
                    target_model=self._target_model,
                    prep_fiducials=self._prep_fiducials,
                    meas_fiducials=self._meas_fiducials,
                    germs=self._germs,
                    prep_povm_tuples="first",
                    constrain_to_tp=True,
                    inv_trace_tol= 10,
                    initial_seed_mode='greedy',
                    evd_tol=1e-5,
                    sensitivity_threshold=1e-5,
                    # seed=1222022,
                    verbosity=1,
                    check_complete_fid_set=False
                )
            else:
                fiducial_pairs = None

            self._edesign = StandardGSTDesign(
                processorspec_filename_or_obj=self._pspec,
                # target_model=self._target_model,
                prep_fiducial_list_or_filename=self._prep_fiducials,
                meas_fiducial_list_or_filename=self._meas_fiducials,
                germ_list_or_filename=self._germs,
                max_lengths=self._circuit_depths,
                fiducial_pairs=fiducial_pairs
            )
            print(
                'Number of circuits: ',
                len(self._edesign.all_circuits_needing_data)
            )

            self._circuits = CircuitSet(
                list(self._edesign.all_circuits_needing_data)
            )
            self._circuits['pygsti_circuit'] = [
                circ.str for circ in self._edesign.all_circuits_needing_data
            ]

            self._data_manager._exp_id += (
                f'_GST_{"".join("Q" + str(q) for q in self._qubits)}'
            )
            if Settings.save_data:
                self._data_manager.create_data_path()
                write_empty_protocol_data(
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
            """Analyze the GST results."""
            logger.info(' Analyzing the results...')
            self._dataset = (
                generate_pygsti_dataset(
                    self._circuits,
                    save_path=self._data_manager._save_path + 'data/'
                    if Settings.save_data else None
                ) if self._dataset is None else self._dataset
            )
            self._data = ProtocolData(self._edesign, self._dataset)

            self._results = self._protocol.run(
                self._data,
                disable_checkpointing=True if not Settings.save_data else False,
                checkpoint_path=self._data_manager._save_path
                + 'gst_checkpoints/checkpoint' if Settings.save_data else None
            )
            if Settings.save_data:
                save_to_pickle(
                    self._results, self._data_manager._save_path + 'results'
                )

                self._report = pygsti.report.construct_standard_report(
                    self._results,
                    title="GST Report",
                    verbosity=2
                )
                self._report.write_html(
                    self._data_manager._save_path, verbosity=2
                )

                save_to_pickle(
                    self.ptm, self._data_manager._save_path + 'PTMs'
                )

            clear_output(wait=True)

        def plot(self) -> None:
            """Plot the PTMs and their associated performance metrics."""
            save_path = self._data_manager._save_path
            basis_state = {
                4: ['I', 'X', 'Y', 'Z'],
                16: [
                    'II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
                    'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ'
                ]
            }
            max_ncols = {4: 3, 16: 2} # PTM shape: number of columns
            size = {4: 5, 16: 7.5}    # PTM shape: size of each row/column

            for model, gateset in self.ptm.items():
                gates = list(gateset.keys())
                ptms = list(gateset.values())
                nrows, ncols = calculate_nrows_ncols(
                    len(gateset), max_ncols=max_ncols[ptms[0].shape[0]]
                )

                # Matplotlib plot
                figsize = (
                    size[ptms[0].shape[0]] * ncols,
                    size[ptms[0].shape[0]] * nrows
                )
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=figsize, layout='constrained'
                )
                fig.suptitle(model, fontsize=15)

                # Plotly plot
                ei = self.entanglement_infidelity[model]
                dn = self.diamond_norm[model]
                pfig = make_subplots(
                    rows=nrows,
                    cols=ncols,
                    # title=model,
                    vertical_spacing=0.15,
                    horizontal_spacing=0.05,
                    subplot_titles=[
                        f"{gate}<br>Proc. Inf. = {ei[gate.split('.')[0]]:.3f}, "
                        f"Diam. Norm = {dn[gate.split('.')[0]]:.3f}"
                        for gate in gates
                    ]
                )
                pfig.update_annotations(font_size=12, yshift=5)
                pfig.update_layout(margin={'t': 100})

                k = -1
                for i in range(nrows):
                    for j in range(ncols):
                        k += 1

                        if len(gates) == 1:
                            ax = axes
                        elif axes.ndim == 1:
                            ax = axes[j]
                        elif axes.ndim == 2:
                            ax = axes[i,j]

                        if k < len(gates):
                            gate = gates[k]
                            ptm = ptms[k]

                            # Matplotlib plot
                            im = ax.imshow(ptm, cmap='RdBu', vmin=-1, vmax=1)
                            ax.set_xticks(
                                range(ptm.shape[0]), basis_state[ptm.shape[0]],
                                fontsize=12
                            )
                            ax.set_yticks(
                                range(ptm.shape[1]), basis_state[ptm.shape[1]],
                                fontsize=12
                            )
                            # fig.colorbar(im, ax=ax)
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes(
                                "right", size="3%", pad=0.05
                            )
                            cb = fig.colorbar(im, cax=cax)
                            cb.set_ticks([-1, 0, 1])
                            cb.set_ticklabels(['-1', '0', '1'])
                            # Add text annotations
                            for m in range(ptm.shape[0]):
                                for n in range(ptm.shape[1]):
                                    value = ptm[m, n]
                                    ax.text(
                                        n, m,
                                        f"{value:.2f}",
                                        ha='center',
                                        va='center',
                                        color='w' if abs(value) > 0.5 else 'k'
                                    )
                            inf = self.entanglement_infidelity[model][
                                gate.split('.')[0]
                            ]
                            dn = self.diamond_norm[model][gate.split('.')[0]]
                            ax.set_title(
                                f'{gate}\n'
                                f'Proc. Inf. = {inf:.3f}, '
                                f'Diam. Norm = {dn:.3f}'
                            )

                            # Plotly plot
                            pfig.add_trace(
                                go.Heatmap(
                                    z=ptm,
                                    x=basis_state[ptm.shape[0]],
                                    y=basis_state[ptm.shape[1]],
                                    coloraxis="coloraxis",
                                ),
                                row=i + 1,
                                col=j + 1
                            )
                            pfig.update_yaxes(
                                categoryorder="array",
                                categoryarray=list(
                                    reversed(basis_state[ptm.shape[1]])
                                ),
                                row=i + 1,
                                col=j + 1
                            )

                        else:
                            ax.axis('off')
                            pfig.update_xaxes(
                                visible=False, row=i + 1, col=j + 1
                            )
                            pfig.update_yaxes(
                                visible=False, row=i + 1, col=j + 1
                            )

                pfig.update_layout(
                    coloraxis={
                        'colorscale': 'RdBu',
                        'cmin': -1,
                        'cmax': 1,
                        'colorbar': {'title': 'Value'}
                    },
                    height=350 * nrows + 50,
                    width=350 * ncols
                )
                pfig.update_layout(
                    title={'text': model, 'pad': {'t': 10, 'b': 10}},
                    margin={'t': 100}
                )
                save_properties = {
                    'toImageButtonOptions': {
                        'format': 'svg', # one of png, svg, jpeg, webp
                        'filename': 'qpu_layout',
                        # 'height': 500,
                        # 'width': 1000,
                        'scale': 10
                    }
                }
                pfig.show(config=save_properties)
                # pfig.show()

                if Settings.save_data:
                    fig.savefig(
                        save_path + f"{model.replace(' ', '')}.png",
                        dpi=600
                    )
                    fig.savefig(save_path + f"{model.replace(' ', '')}.pdf")
                    fig.savefig(save_path + f"{model.replace(' ', '')}.svg")
                plt.close(fig)

        def final(self) -> None:
            """Final method."""
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.save()
            self.analyze()
            self.plot()
            self.final()

    return GST(
        config=config,
        qubit_labels=qubit_labels,
        pspec=pspec,
        target_model=target_model,
        prep_fiducials=prep_fiducials,
        meas_fiducials=meas_fiducials,
        germs=germs,
        circuit_depths=circuit_depths,
        modes=modes,
        fpr=fpr,
        **kwargs
    )


def SimultaneousGST(
    qpu:            QPU,
    config:         Config,
    qubit_labels:   Iterable[int | Tuple[int]],
    pspec:          Dict[int | Tuple[int], Any] | None = None,
    target_model:   Dict[int | Tuple[int], Any] | None = None,
    prep_fiducials: Dict[int | Tuple[int], Any] | None = None,
    meas_fiducials: Dict[int | Tuple[int], Any] | None = None,
    germs:          Dict[int | Tuple[int], Any] | None = None,
    circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
    modes:          Tuple[str] = ('full TP','CPTPLND','Target','H+S','S'),  # noqa: B006
    fpr:            bool = False,
    **kwargs
) -> Callable:
    """Simultaneous Gate Set Tomography.

    This protocol requires a valid pyGSTi installation.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[int | Tuple[int]]): a list specifying sets of
            system labels on which to perform GST.
        pspec (Dict[int | Tuple[int], Any] | None, optional): a dictionary of
            pyGSTi ProcessorSpec objects. Defaults to None.
        target_model (Dict[int | Tuple[int], Any] | None, optional): a
            dictionary of pyGSTi Model objects. Defaults to None.
        prep_fiducials (Dict[int | Tuple[int], Any] | None, optional): a
            dictionary of lists of pyGSTi fiducial circuits. Defaults to None.
        meas_fiducials (Dict[int | Tuple[int], Any] | None, optional): a
            dictionary of lists of pyGSTi fiducial circuits. Defaults to None.
        germs (Dict[int | Tuple[int], Any] | None, optional): a dictionary of
            lists of pyGSTi germ circuits. Defaults to None.
        circuit_depths (List[int], optional): a list of positive integers
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32,
            64, 128, 256]```.
        modes (Tuple[str], optional): a tuple of strings specifying the modes
            to be used in the GST protocol. Defaults to ```('full TP',
            'CPTPLND', 'Target', 'H+S', 'S')```. These correspond to different
            types of parameterizations/constraints to apply to the estimated
            model. Allowed values are:
            - 'full': full (completely unconstrained)
            - 'TP': TP-constrained
            - 'CPTPLND': Lindbladian CPTP-constrained
            - 'H+S': Only Hamiltonian + Stochastic errors allowed (CPTP)
            - 'S': Only Stochastic errors allowed (CPTP)
            - 'Target': use the target (ideal) gates as the estimate
            - <model>: any key in the models_to_test argument
        fpr (bool, optional): whether to use Fiducial Pair Reduction (FPR).
            Defaults to False.

    Returns:
        Callable: SimultaneousGST class instance.
    """

    class SimultaneousGST(qpu):
        """Simultanous GST protocol."""

        @save_init
        def __init__(
            self,
            config:         Config,
            qubit_labels:   Iterable[int | Tuple[int]],
            pspec:          Dict[int | Tuple[int], Any] | None = None,
            target_model:   Dict[int | Tuple[int], Any] | None = None,
            prep_fiducials: Dict[int | Tuple[int], Any] | None = None,
            meas_fiducials: Dict[int | Tuple[int], Any] | None = None,
            germs:          Dict[int | Tuple[int], Any] | None = None,
            circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
            modes:          Tuple[str] = (
                'full TP', 'CPTPLND', 'Target', 'H+S', 'S'
            ),
            fpr:            bool = False,
            **kwargs
        ) -> None:
            self._qubit_labels = qubit_labels
            self._qubits = sorted(flatten(qubit_labels))
            self._pspec = (
                pspec if pspec is not None else dict.fromkeys(qubit_labels)
            )
            self._target_model = (
                target_model if target_model is not None
                else dict.fromkeys(qubit_labels)
            )
            self._prep_fiducials = (
                prep_fiducials if prep_fiducials is not None
                else dict.fromkeys(qubit_labels)
            )
            self._meas_fiducials = (
                meas_fiducials if meas_fiducials is not None
                else dict.fromkeys(qubit_labels)
            )
            self._germs = (
                germs if germs is not None
                else dict.fromkeys(qubit_labels)
            )
            self._circuit_depths = circuit_depths
            self._modes = modes
            self._fpr = fpr

            self._gst = {}
            for ql in self._qubit_labels:
                self._gst[ql] = GST(
                    qpu=qpu,
                    config=config,
                    qubit_labels=[ql],
                    pspec=self._pspec[ql],
                    target_model=self._target_model[ql],
                    prep_fiducials=self._prep_fiducials[ql],
                    meas_fiducials=self._meas_fiducials[ql],
                    germs=self._germs[ql],
                    circuit_depths=self._circuit_depths,
                    modes=self._modes,
                    fpr=self._fpr,
                    **kwargs
                )

            # Don't need to pass a transpiler because the circuits are already
            # in qcal format.
            kwargs.pop('transpiler', None)
            qpu.__init__(self, config=config, transpiler=None, **kwargs)

        def __getitem__(self, ql: int | Tuple[int]) -> GST:
            """Get the GST object for a given qubit label."""
            return self._gst[ql]

        def _gst_property_by_qubit_label(
            self, property_name: str
        ) -> Dict[Any, Any]:
            """Get a property of the GST object for a given qubit label."""
            return {
                ql: getattr(self[ql], property_name)
                for ql in self._qubit_labels
            }

        @property
        def avg_gate_infidelity(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('avg_gate_infidelity')

        @property
        def circuit_depths(self) -> List[int]:
            return self._circuit_depths

        @property
        def data(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('data')

        @property
        def diamond_norm(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('diamond_norm')

        @property
        def edesign(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('edesign')

        @property
        def eigenvalue_avg_gate_infidelity(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label(
                'eigenvalue_avg_gate_infidelity'
            )

        @property
        def eigenvalue_diamond_norm(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('eigenvalue_diamond_norm')

        @property
        def eigenvalue_entanglement_infidelity(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label(
                'eigenvalue_entanglement_infidelity'
            )

        @property
        def entanglement_infidelity(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('entanglement_infidelity')

        @property
        def germs(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('germs')

        @property
        def jtrace_diff(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('jtrace_diff')

        @property
        def meas_fiducials(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('meas_fiducials')

        @property
        def models(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('models')

        @property
        def modes(self) -> Tuple[str]:
            return self._modes

        @property
        def POVM(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('POVM')

        @property
        def prep_fiducials(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('prep_fiducials')

        @property
        def process_infidelity(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('process_infidelity')

        @property
        def protocol(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('protocol')

        @property
        def pspec(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('pspec')

        @property
        def ptm(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('ptm')

        @property
        def results(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('results')

        @property
        def state_prep(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('state_prep')

        @property
        def state_prep_fidelity(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('state_prep_fidelity')

        @property
        def target_model(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('target_model')

        @property
        def unitarity(self) -> Dict[Any, Any]:
            return self._gst_property_by_qubit_label('unitarity')

        def generate_circuits(self):
            """Generate all GST circuits."""
            transpiler = PyGSTiTranspiler()

            max_workers = min(
                len(self._qubit_labels), max(1, mp.cpu_count() - 1)
            )

            def _generate_one(ql):
                self._gst[ql].generate_circuits()
                return ql

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_generate_one, ql)
                    for ql in self._qubit_labels
                ]
                for f in as_completed(futures):
                    f.result()

                csets = [self._gst[ql].circuits for ql in self._qubit_labels]

                tcsets = list(executor.map(transpiler.transpile, csets))

            lengths = [len(cs) for cs in tcsets]
            if len(set(lengths)) != 1:
                msg = (
                    'Inconsistent number of circuits across qubit labels: '
                    f'{dict(zip(self._qubit_labels, lengths, strict=False))}. '
                    'This breaks simultaneous circuit joining.'
                )
                if self._fpr:
                    msg += ' Try setting fpr=False.'
                raise ValueError(msg)

            # Merge per-index circuits into one CircuitSet
            self._circuits = CircuitSet()
            for circuits_i in zip(*tcsets, strict=False):
                base = circuits_i[0].copy()
                for c in circuits_i[1:]:
                    base.join(c, how="outer")
                self._circuits.append(base)

            self._data_manager._save_path = (
                self[self._qubit_labels[0]].data_manager.save_path
            )

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)

            def _save_one(ql):
                if isinstance(ql, Iterable):
                    qidx = tuple([self._qubits.index(q) for q in ql])
                else:
                    qidx = (self._qubits.index(ql),)

                marginalized_results = []
                for result in self.circuits.results:
                    marginalized_results.append(
                        Results(result).marginalize(qidx).dict
                    )

                self._gst[ql]._circuits.results = marginalized_results
                self._gst[ql]._runtime = self._runtime
                self._gst[ql].save()
                return ql

            max_workers = min(
                len(self._qubit_labels), max(1, mp.cpu_count() - 1)
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_save_one, ql) for ql in self._qubit_labels
                ]
                for f in as_completed(futures):
                    f.result()

        def analyze(self):
            """Analyze the GST results."""

            def _analyze_one(ql):
                self._gst[ql].analyze()
                return ql

            max_workers = min(
                len(self._qubit_labels), max(1, mp.cpu_count() - 1)
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_analyze_one, ql)
                    for ql in self._qubit_labels
                ]
                for f in as_completed(futures):
                    f.result()

        def plot(self):
            """Plot the GST results."""
            for ql in self._qubit_labels:
                print('-------------------------------------------------------')
                print(f'Qubit Label: {ql}')
                print('-------------------------------------------------------')
                self._gst[ql].plot()

        def final(self) -> None:
            """Final method."""
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.save()
            self.analyze()
            self.plot()
            self.final()

    return SimultaneousGST(
        config=config,
        qubit_labels=qubit_labels,
        pspec=pspec,
        target_model=target_model,
        prep_fiducials=prep_fiducials,
        meas_fiducials=meas_fiducials,
        germs=germs,
        circuit_depths=circuit_depths,
        modes=modes,
        fpr=fpr,
        **kwargs
    )


def SingleQubitGST(
    qpu:            QPU,
    config:         Config,
    qubits:         Iterable[int],
    pspec:          Any | Dict[int, Any] | None = None,
    target_model:   Any | Dict[int, Any] | None = None,
    prep_fiducials: Any | Dict[int, Any] | None = None,
    meas_fiducials: Any | Dict[int, Any] | None = None,
    germs:          Any | Dict[int, Any] | None = None,
    circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
    modes:          Tuple[str] = ('full TP','CPTPLND','Target','H+S','S'),  # noqa: B006
    fpr:            bool = False,
    **kwargs
) -> Callable:
    """Single-Qubit Gate Set Tomography.

    This protocol requires a valid pyGSTi installation.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (Iterable[int]): a list specifying the qubits on which to perform
            GST.
        pspec (Any | Dict[int, Any] | None, optional): a pyGSTi ProcessorSpec
            object or a dictionary of such objects. Defaults to None.
        target_model (Any | Dict[int, Any] | None, optional): a pyGSTi Model
            object or a dictionary of such objects. Defaults to None.
        prep_fiducials (Any | Dict[int, Any] | None, optional): a list of pyGSTi
            fiducial circuits or a dictionary of such objects. Defaults to None.
        meas_fiducials (Any | Dict[int, Any] | None, optional): a list of pyGSTi
            fiducial circuits or a dictionary of such objects. Defaults to None.
        germs (Any | Dict[int, Any] | None, optional): a list of pyGSTi germ
            circuits or a dictionary of such objects. Defaults to None.
        circuit_depths (List[int], optional): a list of positive integers
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32,
            64, 128, 256]```.
        modes (Tuple[str], optional): a tuple of strings specifying the modes
            to be used in the GST protocol. Defaults to ```('full TP',
            'CPTPLND', 'Target', 'H+S', 'S')```. These correspond to different
            types of parameterizations/constraints to apply to the estimated
            model. Allowed values are:
            - 'full': full (completely unconstrained)
            - 'TP': TP-constrained
            - 'CPTPLND': Lindbladian CPTP-constrained
            - 'H+S': Only Hamiltonian + Stochastic errors allowed (CPTP)
            - 'S': Only Stochastic errors allowed (CPTP)
            - 'Target': use the target (ideal) gates as the estimate
            - <model>: any key in the models_to_test argument
        fpr (bool, optional): whether to use Fiducial Pair Reduction (FPR).
            Defaults to False.

    Returns:
        Callable: SingleQubitGST class instance.
    """
    if len(qubits) <= 2:
        gst = type(GST(
            qpu=qpu,
            config=config,
            qubit_labels=qubits,
            pspec=pspec,
            target_model=target_model,
            prep_fiducials=prep_fiducials,
            meas_fiducials=meas_fiducials,
            germs=germs,
            circuit_depths=circuit_depths,
            modes=modes,
            fpr=fpr,
            **kwargs
        ))
    else:
        gst = type(SimultaneousGST(
            qpu=qpu,
            config=config,
            qubit_labels=qubits,
            pspec=pspec,
            target_model=target_model,
            prep_fiducials=prep_fiducials,
            meas_fiducials=meas_fiducials,
            germs=germs,
            circuit_depths=circuit_depths,
            modes=modes,
            fpr=fpr,
            **kwargs
        ))
    class SingleQubitGST(gst):
        """GST protocol."""

        @save_init
        def __init__(
            self,
            config:         Config,
            qubits:         Iterable[int],
            pspec:          Any | Dict[int, Any] | None = None,
            target_model:   Any | Dict[int, Any] | None = None,
            prep_fiducials: Any | Dict[int, Any] | None = None,
            meas_fiducials: Any | Dict[int, Any] | None = None,
            germs:          Any | Dict[int, Any] | None = None,
            circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
            modes:          Tuple[str] = (
                'full TP', 'CPTPLND', 'Target', 'H+S', 'S'
            ),
            fpr:            bool = False,
            **kwargs
        ) -> None:
            if len(qubits) == 1:
                pspec = (
                    smq1Q_XYI.processor_spec(qubits) if pspec is None
                    else pspec
                )

                target_model = (
                    smq1Q_XYI.target_model(qubit_labels=qubits) if target_model
                    is None else target_model
                )

                prep_fiducials = (
                    smq1Q_XYI.prep_fiducials(qubits) if prep_fiducials is None
                    else prep_fiducials
                )

                meas_fiducials = (
                    smq1Q_XYI.meas_fiducials(qubits) if meas_fiducials is None
                    else meas_fiducials
                )

                germs = smq1Q_XYI.germs(qubits) if germs is None else germs

            elif len(qubits) == 2:
                if pspec is None:
                    gate_names = [
                        'Gxpi2', 'Gypi2', 'Gii', 'Gxx', 'Gxy','Gyx', 'Gyy'
                    ]

                    # Define a global 2-qubit idle
                    global_idle = np.eye(4)

                    # Define the unitaries for the parallel gates
                    standard_gate_unitaries = standard_gatename_unitaries()
                    gxpi2 = standard_gate_unitaries['Gxpi2']
                    gypi2 = standard_gate_unitaries['Gypi2']

                    gxx = np.kron(gxpi2, gxpi2)
                    gxy = np.kron(gxpi2, gypi2)
                    gyx = np.kron(gypi2, gxpi2)
                    gyy = np.kron(gypi2, gypi2)

                    nonstd_gate_unitaries = {
                        'Gii': global_idle,
                        'Gxx': gxx,
                        'Gxy': gxy,
                        'Gyx': gyx,
                        'Gyy': gyy
                    }

                    pspec = QubitProcessorSpec(
                        num_qubits=len(qubits),
                        gate_names=gate_names,
                        nonstd_gate_unitaries=nonstd_gate_unitaries,
                        prep_names=['rho0'],
                        povm_names=['Mdefault'],
                        availability={
                            'Gxpi2': [(q,) for q in qubits],
                            'Gypi2': [(q,) for q in qubits],
                            'Gii':   [tuple(qubits)],
                            'Gxx':   [tuple(qubits)],
                            'Gxy':   [tuple(qubits)],
                            'Gyx':   [tuple(qubits)],
                            'Gyy':   [tuple(qubits)]
                        },
                        qubit_labels=qubits
                    )

                if target_model is None:
                    target_model = create_explicit_model(
                        pspec,
                        # ideal_gate_type='full TP',
                        # ideal_spam_type='full TP',
                        # basis='pp',
                    )

                if prep_fiducials is None and meas_fiducials is None:
                    prep_fiducials, meas_fiducials = find_fiducials(
                        target_model,
                        candidate_fid_counts={3: 'all upto'},
                        assume_clifford=True,
                        verbosity=2
                    )

                if germs is None:
                    germs = find_germs(
                        target_model,
                        randomize=False,
                        algorithm='greedy',
                        assume_real=True,
                        mode='compactEVD',
                        float_type=np.double,
                        candidate_germ_counts={4:'all upto'},
                        verbosity=2
                    )

            elif len(qubits) > 2:
                if pspec is None:
                    pspec = {
                        q: smq1Q_XYI.processor_spec((q,)) for q in qubits
                    }

                if target_model is None:
                    target_model = {
                        q: smq1Q_XYI.target_model(qubit_labels=(q,))
                        for q in qubits
                }

                if prep_fiducials is None:
                    prep_fiducials = {
                        q: smq1Q_XYI.prep_fiducials((q,)) for q in qubits
                    }

                if meas_fiducials is None:
                    meas_fiducials = {
                        q: smq1Q_XYI.meas_fiducials((q,)) for q in qubits
                    }

                if germs is None:
                    germs = {
                        q: smq1Q_XYI.germs((q,)) for q in qubits
                    }

            gst.__init__(self,
                config=config,
                qubit_labels=qubits,
                pspec=pspec,
                target_model=target_model,
                prep_fiducials=prep_fiducials,
                meas_fiducials=meas_fiducials,
                germs=germs,
                circuit_depths=circuit_depths,
                modes=modes,
                fpr=fpr,
                **kwargs
            )

    return SingleQubitGST(
        config=config,
        qubits=qubits,
        pspec=pspec,
        target_model=target_model,
        prep_fiducials=prep_fiducials,
        meas_fiducials=meas_fiducials,
        germs=germs,
        circuit_depths=circuit_depths,
        modes=modes,
        fpr=fpr,
        **kwargs
    )


def TwoQubitGST(
    qpu:            QPU,
    config:         Config,
    qubit_labels:   Iterable[Tuple[int]],
    pspec:          Any | Dict[Tuple[int], Any] | None = None,
    target_model:   Any | Dict[Tuple[int], Any] | None = None,
    prep_fiducials: Any | Dict[Tuple[int], Any] | None = None,
    meas_fiducials: Any | Dict[Tuple[int], Any] | None = None,
    germs:          Any | Dict[Tuple[int], Any] | None = None,
    circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128],  # noqa: B006
    modes:          Tuple[str] = ('full TP','CPTPLND','Target','H+S','S'),  # noqa: B006
    fpr:            bool = False,
    **kwargs
) -> Callable:
    """Two-Qubit Gate Set Tomography.

    This protocol requires a valid pyGSTi installation.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[Tuple[int]]): a list of tuples of ints specifying
            sets of qubit labels on which to perform two-qubit GST.
        pspec (Any | Dict[int, Any] | None, optional): a pyGSTi ProcessorSpec
            object or a dictionary of such objects. Defaults to None.
        target_model (Any | Dict[int, Any] | None, optional): a pyGSTi Model
            object or a dictionary of such objects. Defaults to None.
        prep_fiducials (Any | Dict[int, Any] | None, optional): a list of pyGSTi
            fiducial circuits or a dictionary of such objects. Defaults to None.
        meas_fiducials (Any | Dict[int, Any] | None, optional): a list of pyGSTi
            fiducial circuits or a dictionary of such objects. Defaults to None.
        germs (Any | Dict[int, Any] | None, optional): a list of pyGSTi germ
            circuits or a dictionary of such objects. Defaults to None.
        circuit_depths (List[int], optional): a list of positive integers
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32,
            64, 128, 256]```.
        modes (Tuple[str], optional): a tuple of strings specifying the modes
            to be used in the GST protocol. Defaults to ```('full TP',
            'CPTPLND', 'Target', 'H+S', 'S')```. These correspond to different
            types of parameterizations/constraints to apply to the estimated
            model. Allowed values are:
            - 'full': full (completely unconstrained)
            - 'TP': TP-constrained
            - 'CPTPLND': Lindbladian CPTP-constrained
            - 'H+S': Only Hamiltonian + Stochastic errors allowed (CPTP)
            - 'S': Only Stochastic errors allowed (CPTP)
            - 'Target': use the target (ideal) gates as the estimate
            - <model>: any key in the models_to_test argument
        fpr (bool, optional): whether to use Fiducial Pair Reduction (FPR).
            Defaults to False.

    Returns:
        Callable: TwoQubitGST class instance.
    """
    if len(qubit_labels) == 1:
        gst = type(GST(
            qpu=qpu,
            config=config,
            qubit_labels=qubit_labels,
            pspec=pspec,
            target_model=target_model,
            prep_fiducials=prep_fiducials,
            meas_fiducials=meas_fiducials,
            germs=germs,
            circuit_depths=circuit_depths,
            modes=modes,
            fpr=fpr,
            **kwargs
        ))
    elif len(qubit_labels) > 1:
        gst = type(SimultaneousGST(
            qpu=qpu,
            config=config,
            qubit_labels=qubit_labels,
            pspec=pspec,
            target_model=target_model,
            prep_fiducials=prep_fiducials,
            meas_fiducials=meas_fiducials,
            germs=germs,
            circuit_depths=circuit_depths,
            modes=modes,
            fpr=fpr,
            **kwargs
        ))

    class TwoQubitGST(gst):
        """GST protocol."""

        @save_init
        def __init__(
            self,
            config:         Config,
            qubit_labels:   Iterable[Tuple[int]],
            pspec:          Any | Dict[Tuple[int], Any] | None = None,
            target_model:   Any | Dict[Tuple[int], Any] | None = None,
            prep_fiducials: Any | Dict[Tuple[int], Any] | None = None,
            meas_fiducials: Any | Dict[Tuple[int], Any] | None = None,
            germs:          Any | Dict[Tuple[int], Any] | None = None,
            circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],  # noqa: B006
            modes:          Tuple[str] = (
                'full TP', 'CPTPLND', 'Target', 'H+S', 'S'
            ),
            fpr:            bool = False,
            **kwargs
        ) -> None:
            if len(qubit_labels) == 1:
                pspec = (
                    smq2Q_XYCPHASE.processor_spec(qubit_labels[0])
                    if pspec is None else pspec
                )

                target_model = (
                    smq2Q_XYCPHASE.target_model(qubit_labels=qubit_labels[0])
                    if target_model is None else target_model
                )

                prep_fiducials = (
                    smq2Q_XYCPHASE.prep_fiducials(qubit_labels[0])
                    if prep_fiducials is None else prep_fiducials
                )

                meas_fiducials = (
                    smq2Q_XYCPHASE.meas_fiducials(qubit_labels[0])
                    if meas_fiducials is None else meas_fiducials
                )

                germs = (
                    smq2Q_XYCPHASE.germs(qubit_labels[0])
                    if germs is None else germs
                )

            elif len(qubit_labels) > 1:
                pspec = {
                    ql: smq2Q_XYCPHASE.processor_spec(ql)
                    for ql in qubit_labels
                } if pspec is None else pspec

                target_model = {
                    ql: smq2Q_XYCPHASE.target_model(qubit_labels=ql)
                    for ql in qubit_labels
                } if target_model is None else target_model

                prep_fiducials = {
                    ql: smq2Q_XYCPHASE.prep_fiducials(ql)
                    for ql in qubit_labels
                } if prep_fiducials is None else prep_fiducials

                meas_fiducials = {
                    ql: smq2Q_XYCPHASE.meas_fiducials(ql)
                    for ql in qubit_labels
                } if meas_fiducials is None else meas_fiducials

                germs = {
                    ql: smq2Q_XYCPHASE.germs(ql)
                    for ql in qubit_labels
                } if germs is None else germs

            gst.__init__(self,
                config=config,
                qubit_labels=qubit_labels,
                pspec=pspec,
                target_model=target_model,
                prep_fiducials=prep_fiducials,
                meas_fiducials=meas_fiducials,
                germs=germs,
                circuit_depths=circuit_depths,
                modes=modes,
                fpr=fpr,
                **kwargs
            )

    return TwoQubitGST(
        config=config,
        qubit_labels=qubit_labels,
        pspec=pspec,
        target_model=target_model,
        prep_fiducials=prep_fiducials,
        meas_fiducials=meas_fiducials,
        germs=germs,
        circuit_depths=circuit_depths,
        modes=modes,
        fpr=fpr,
        **kwargs
    )


def QuantumInstrumentGST(
    qpu:            QPU,
    config:         Config,
    qubits:         Iterable[int],
    pspec:          Any | Dict[int, Any] | None = None,
    target_model:   Any | Dict[int, Any] | None = None,
    prep_fiducials: Any | Dict[int, Any] | None = None,
    meas_fiducials: Any | Dict[int, Any] | None = None,
    germs:          Any | Dict[int, Any] | None = None,
    circuit_depths: List[int] = [1],  # noqa: B006
    modes:          Tuple[str] = ('Target', 'full TP'),  # noqa: B006
    # fpr:            bool = False,
    **kwargs
) -> Callable:
    """Quantum Instrument Gate Set Tomography.

    This protocol requires a valid pyGSTi installation.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (Iterable[int]): a list specifying the qubits on which to perform
            GST.
        pspec (Any | Dict[int, Any] | None, optional): a pyGSTi ProcessorSpec
            object or a dictionary of such objects. Defaults to None.
        target_model (Any | Dict[int, Any] | None, optional): a pyGSTi Model
            object or a dictionary of such objects. Defaults to None.
        prep_fiducials (Any | Dict[int, Any] | None, optional): a list of pyGSTi
            fiducial circuits or a dictionary of such objects. Defaults to None.
        meas_fiducials (Any | Dict[int, Any] | None, optional): a list of pyGSTi
            fiducial circuits or a dictionary of such objects. Defaults to None.
        germs (Any | Dict[int, Any] | None, optional): a list of pyGSTi germ
            circuits or a dictionary of such objects. Defaults to None.
        circuit_depths (List[int], optional): a list of positive integers
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32,
            64, 128, 256]```.
        modes (Tuple[str], optional): a tuple of strings specifying the modes
            to be used in the GST protocol. Defaults to ```('full TP',
            'CPTPLND', 'Target', 'H+S', 'S')```. These correspond to different
            types of parameterizations/constraints to apply to the estimated
            model. Allowed values are:
            - 'full': full (completely unconstrained)
            - 'TP': TP-constrained
            - 'CPTPLND': Lindbladian CPTP-constrained
            - 'H+S': Only Hamiltonian + Stochastic errors allowed (CPTP)
            - 'S': Only Stochastic errors allowed (CPTP)
            - 'Target': use the target (ideal) gates as the estimate
            - <model>: any key in the models_to_test argument
        fpr (bool, optional): whether to use Fiducial Pair Reduction (FPR).
            Defaults to False.
    """
    if len(qubits) == 1:
        gst = type(GST(
            qpu=qpu,
            config=config,
            qubit_labels=qubits,
            pspec=pspec,
            target_model=target_model,
            prep_fiducials=prep_fiducials,
            meas_fiducials=meas_fiducials,
            germs=germs,
            circuit_depths=circuit_depths,
            modes=modes,
            # fpr=fpr,
            **kwargs
        ))
    else:
        gst = type(SimultaneousGST(
            qpu=qpu,
            config=config,
            qubit_labels=qubits,
            pspec=pspec,
            target_model=target_model,
            prep_fiducials=prep_fiducials,
            meas_fiducials=meas_fiducials,
            germs=germs,
            circuit_depths=circuit_depths,
            modes=modes,
            # fpr=fpr,
            **kwargs
        ))

    class QuantumInstrumentGST(gst):
        """QI GST protocol."""

        @save_init
        def __init__(
            self,
            config:         Config,
            qubits:         Iterable[int],
            pspec:          Any | Dict[int, Any] | None = None,
            target_model:   Any | Dict[int, Any] | None = None,
            prep_fiducials: Any | Dict[int, Any] | None = None,
            meas_fiducials: Any | Dict[int, Any] | None = None,
            germs:          Any | Dict[int, Any] | None = None,
            circuit_depths: List[int] = [1],  # noqa: B006
            modes:          Tuple[str] = ('Target', 'full TP'),  # noqa: B006
            # fpr:            bool = False,
            **kwargs
        ) -> None:
            if len(circuit_depths) > 1 or circuit_depths[0] != 1:
                raise ValueError(
                    'Long-sequence GST is not currently supported for '
                    'Quantum Instrument GST!'
                )

            if len(modes) > 2 or 'full TP' not in modes:
                raise ValueError(
                    "Only 'Target' and 'full TP' modes are currently supported "
                    "for Quantum Instrument GST!"
                )

            if len(qubits) == 1:
                pspec = (
                    smq1Q_XYI.processor_spec(qubits) if pspec is None
                    else pspec
                )

                target_model = (
                    smq1Q_XYI.target_model(qubit_labels=qubits) if target_model
                    is None else target_model
                )
                target_model.set_all_parameterizations("full TP") # TODO: CPTP
                # Create and add the ideal quantum instrument to the target model
                E0 = target_model.effects['0']
                E1 = target_model.effects['1']
                target_model[('Iz', qubits[0])] = Instrument(
                    {'p0': np.dot(E0, E0.T), 'p1': np.dot(E1, E1.T)}
                )

                prep_fiducials = (
                    smq1Q_XYI.prep_fiducials(qubits) if prep_fiducials is None
                    else prep_fiducials
                )

                meas_fiducials = (
                    smq1Q_XYI.meas_fiducials(qubits) if meas_fiducials is None
                    else meas_fiducials
                )

                germs = smq1Q_XYI.germs(qubits) if germs is None else germs
                germs += [ # Add the instrument as a germ
                    pygsti.circuits.Circuit([('Iz', qubits[0])])
                ]

            elif len(qubits) > 1:
                if pspec is None:
                    pspec = {
                        q: smq1Q_XYI.processor_spec((q,)) for q in qubits
                    }

                if target_model is None:
                    target_model = {
                        q: smq1Q_XYI.target_model(qubit_labels=(q,))
                        for q in qubits
                    }
                    for q in qubits: # TODO: CPTP
                        target_model[q].set_all_parameterizations("full TP")
                        E0 = target_model[q].effects['0']
                        E1 = target_model[q].effects['1']
                        target_model[q][('Iz', q)] = Instrument(
                            {'p0': np.dot(E0, E0.T), 'p1': np.dot(E1, E1.T)}
                        )

                if prep_fiducials is None:
                    prep_fiducials = {
                        q: smq1Q_XYI.prep_fiducials((q,)) for q in qubits
                    }

                if meas_fiducials is None:
                    meas_fiducials = {
                        q: smq1Q_XYI.meas_fiducials((q,)) for q in qubits
                    }

                if germs is None:
                    germs = {
                        q: smq1Q_XYI.germs((q,)) for q in qubits
                    }
                    for q in qubits:
                        germs[q] += [ # Add the instrument as a germ
                            pygsti.circuits.Circuit([('Iz', q)])
                        ]

            gst.__init__(self,
                config=config,
                qubit_labels=qubits,
                pspec=pspec,
                target_model=target_model,
                prep_fiducials=prep_fiducials,
                meas_fiducials=meas_fiducials,
                germs=germs,
                circuit_depths=circuit_depths,
                modes=modes,
                # fpr=fpr,
                **kwargs
            )

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)

            if self._classified_results is None:
                raise ValueError(
                    "No classified results found. Cannot post-process the data!"
                )

            def _run_post_processing(gst_obj, results):
                pp = PostProcessor(
                    self._config,
                    results,
                    passes=[
                        discard_heralded_shots,
                        relabel_esp,
                        compute_conditional_counts
                    ]
                )
                pp.run()

                # Old way
                # results_df = pd.DataFrame(pp.tm_results)
                # results_df.index = gst_obj._circuits['pygsti_circuit'].values
                # results_df = results_df.apply(
                #     lambda col: col.round().astype('Int64')
                # ).astype(object)
                # results_df = results_df.fillna('--')
                # results_df = results_df.rename(
                #     columns=lambda col: col + ' count'
                # )
                # logger.info(" Saving the pyGSTi results...")
                # filepath = gst_obj.data_manager.save_path + 'data/dataset.txt'
                # with open(filepath, 'w') as f:
                #     f.write(
                #         '## Columns = ' + ', '.join(results_df.columns) + '\n'
                #     )
                #     f.close()
                # results_df.to_csv(filepath, sep=' ', mode='a', header=False)

                # New way
                gst_obj._dataset = DataSet(
                    outcome_labels=[
                        ('0',), ('1',),
                        ('p0', '0'), ('p0', '1'),
                        ('p1', '0'), ('p1', '1')
                    ]
                )
                for i, circuit in enumerate(
                    gst_obj._circuits['pygsti_circuit'].values
                ):
                    results = pp.tm_results[i]
                    results = {
                        tuple(str(k).split(':')): v for k, v in results.items()
                    }
                    gst_obj._dataset[circuit] = results

                if Settings.save_data:
                    logger.info(" Saving the pyGSTi results...")
                    filepath = (
                        gst_obj.data_manager.save_path + 'data/dataset.txt'
                    )
                    write_dataset(filepath, gst_obj._dataset)

                gst_obj._runtime = self._runtime
                if Settings.save_data:
                    qpu.save(gst_obj, create_data_path=False)

            if hasattr(self, '_gst'):
                def _process_qubit_label(ql):
                    gst_obj = self[ql]
                    qset = set(ql) if isinstance(ql, tuple) else {ql}
                    filtered_results = [
                        {q: res[q] for q in qset}
                        for res in self._classified_results
                    ]
                    _run_post_processing(gst_obj, filtered_results)

                max_workers = min(
                    len(self._qubit_labels), max(1, mp.cpu_count() - 1)
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_process_qubit_label, ql): ql
                        for ql in self._qubit_labels
                    }
                    for future in as_completed(futures):
                        ql = futures[future]
                        try:
                            future.result()
                        except Exception:
                            logger.exception(
                                " Post-processing failed for qubit label %s!",
                                ql
                            )
                            raise

            else:
                _run_post_processing(self, self._classified_results)

    return QuantumInstrumentGST(
        config=config,
        qubits=qubits,
        pspec=pspec,
        target_model=target_model,
        prep_fiducials=prep_fiducials,
        meas_fiducials=meas_fiducials,
        germs=germs,
        circuit_depths=circuit_depths,
        modes=modes,
        # fpr=fpr,
        **kwargs
    )
