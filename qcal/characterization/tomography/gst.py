"""Submodule for performing Gate Set Tomography (GST)

Relevant literature:
- https://quantum-journal.org/papers/q-2021-10-05-557/

Relevant code repos:
- https://www.pygsti.info/
- https://github.com/sandialabs/pyGSTi
"""
import logging
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import clear_output
from numpy.typing import NDArray
from plotly.subplots import make_subplots

import qcal.settings as settings
from qcal.config import Config
from qcal.plotting.utils import calculate_nrows_ncols
from qcal.post_processing.passes import (
    compute_conditional_counts,
    discard_heralded_shots,
    relabel_esp,
)
from qcal.post_processing.post_process import PostProcessor
from qcal.qpu.qpu import QPU
from qcal.utils import flatten, load_from_pickle, save_init, save_to_pickle

logger = logging.getLogger(__name__)


def GST(qpu:            QPU,
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

    This protocol requires a valid pyGSTi installation.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[int | Tuple[int]]): a list specifying sets of
            system labels on which to perform GST.
        pspec (Any | None, optional): processor spec. Defaults to None.
        target_model (Any | None, optional): target model. Defaults to None.
        prep_fiducials (Any | None, optional): prep fiducials. Defaults to None.
        meas_fiducials (Any | None, optional): meas fiducials. Defaults to None.
        germs (Any | None, optional): germs. Defaults to None.
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
    try:
        import pygsti
        import pygsti.report.reportables as metrics
        from pygsti.circuits.circuit import Circuit
        from pygsti.models.explicitmodel import ExplicitOpModel
        from pygsti.processors.processorspec import QubitProcessorSpec
        from pygsti.protocols.gst import ModelEstimateResults, StandardGSTDesign
        from pygsti.protocols.protocol import ProtocolData
        logger.info(f" pyGSTi version: {pygsti.__version__}\n")
    except ImportError:
        logger.warning(' Unable to import pyGSTi!')

    class GST(qpu):
        """GST protocol."""

        @save_init
        def __init__(self,
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
            from qcal.interface.pygsti.transpiler import PyGSTiTranspiler

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
            self._results = None
            self._circuits = None
            self._datasets = {}
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
        def diamond_norm(self) -> Dict[str, Dict[str, float]]:
            """Diamond norm for the gates in the gate set.

            Returns:
                Dict[str, Dict[str, float]]: diamond norm for each gate in the
                    gate set for each model.
            """
            diamondnorm = {}
            for mode, model in self.models.items():
                diamondnorm[mode] = {
                    str(gate): metrics.half_diamond_norm(
                        self.target_model.operations[gate].to_dense(),
                        op.to_dense(),
                        'pp'
                    )
                    for gate, op in model.operations.items()
                }
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
                infidelity[mode] = {
                    str(gate): metrics.entanglement_infidelity(
                        self.target_model.operations[gate].to_dense(),
                        op.to_dense(),
                        'pp'
                    )
                    for gate, op in model.operations.items()
                }
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
                ptm[mode] = {
                    str(gate): op.to_dense()
                    for gate, op in model.operations.items()
                }
            return ptm

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
            from pygsti.algorithms.fiducialpairreduction import (
                find_sufficient_fiducial_pairs_per_germ_greedy,
            )
            from pygsti.io import write_empty_protocol_data
            from pygsti.protocols import StandardGST, StandardGSTDesign

            from qcal.interface.pygsti.circuits import load_circuits

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

            # Save an empty dataset file of all the circuits
            self._data_manager._exp_id += (
                f'_GST_{"".join("Q" + str(q) for q in self._qubits)}'
            )
            self._data_manager.create_data_path()
            write_empty_protocol_data(
                self._data_manager._save_path,
                self._edesign,
                sparse=True,
                clobber_ok=True
            )

            self._circuits = load_circuits(
                self._data_manager._save_path + 'data/dataset.txt'
            )

        def save(self):
            """Save all circuits and data."""
            from qcal.interface.pygsti.datasets import generate_pygsti_dataset
            clear_output(wait=True)
            generate_pygsti_dataset(
                self._circuits,
                self._data_manager._save_path + 'data/'
            )
            if settings.Settings.save_data:
                qpu.save(self, create_data_path=False)

        def analyze(self):
            """Analyze the GST results."""
            logger.info(' Analyzing the results...')
            import pygsti

            self._data = pygsti.io.read_data_from_dir(
                self._data_manager._save_path
            )

            self._results = self._protocol.run(
                self._data,
                checkpoint_path=self._data_manager._save_path
                + 'gst_checkpoints/checkpoint'
            )
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

            # Print the models
            clear_output(wait=True)
            for key, model in self.models.items():
                print('-------------------------------------------------------')
                print(f'Model: {key}\n')
                print(model)

            if settings.Settings.save_data:
                save_to_pickle(
                    self.ptm, self._data_manager._save_path + 'PTMs'
                )

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

            for model, gateset in self.ptm.items():
                gates = list(gateset.keys())
                ptms = list(gateset.values())
                nrows, ncols = calculate_nrows_ncols(len(gateset), max_ncols=3)

                # Matplotlib plot
                figsize = (5 * ncols, 5 * nrows)
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
                    subplot_titles=[
                        f"{gate}<br>Proc. Inf. = {ei[gate.split('.')[0]]:.3f}, "
                        f"Diam. Norm = {dn[gate.split('.')[0]]:.3f}"
                        for gate in gates
                    ]
                )

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
                            fig.colorbar(im, ax=ax)
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
                    height=400 * nrows + 50,
                    width=400 * ncols
                )
                pfig.update_layout(
                    title_text=model
                )
                pfig.show()

                if settings.Settings.save_data:
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


def SingleQubitGST(
        qpu:            QPU,
        config:         Config,
        qubits:         Iterable[int],
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
    """Single-Qubit Gate Set Tomography.

    This protocol requires a valid pyGSTi installation.


    """
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
        fpr=fpr,
        **kwargs
    ))

    class SingleQubitGST(gst):
        """GST protocol."""

        @save_init
        def __init__(self,
                config:         Config,
                qubits:         Iterable[int],
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
            try:
                import pygsti
                from pygsti.modelpacks import smq1Q_XYI
                logger.info(f" pyGSTi version: {pygsti.__version__}\n")
            except ImportError:
                logger.warning(' Unable to import pyGSTi!')

            if len(qubits) > 2:
                raise ValueError(
                    'Single-qubit GST is not currently supported for more than '
                    '2 qubits!'
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
                from pygsti.algorithms.fiducialselection import find_fiducials
                from pygsti.algorithms.germselection import find_germs
                from pygsti.models.modelconstruction import create_explicit_model
                from pygsti.processors import QubitProcessorSpec
                from pygsti.tools.internalgates import standard_gatename_unitaries

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
        qubits:         Iterable[int],
        pspec:          Any | None = None,
        target_model:   Any | None = None,
        prep_fiducials: Any | None = None,
        meas_fiducials: Any | None = None,
        germs:          Any | None = None,
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
        qubits (Iterable[int]): a list specifying sets of system labels
            on which to perform GST.
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

    class TwoQubitGST(gst):
        """GST protocol."""

        @save_init
        def __init__(self,
                config:         Config,
                qubits:         Iterable[int],
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
            try:
                import pygsti
                from pygsti.modelpacks import smq2Q_XYCPHASE
                logger.info(f" pyGSTi version: {pygsti.__version__}\n")
            except ImportError:
                logger.warning(' Unable to import pyGSTi!')

            if len(qubits) != 2:
                raise ValueError(
                    'Two-qubit GST is only support for 2 qubits!'
                )

            pspec = (
                smq2Q_XYCPHASE.processor_spec(qubits) if pspec is None
                else pspec
            )

            target_model = (
                smq2Q_XYCPHASE.target_model(qubit_labels=qubits) if
                target_model is None else target_model
            )

            prep_fiducials = (
                smq2Q_XYCPHASE.prep_fiducials(qubits) if prep_fiducials is None
                else prep_fiducials
            )

            meas_fiducials = (
                smq2Q_XYCPHASE.meas_fiducials(qubits) if meas_fiducials is None
                else meas_fiducials
            )

            germs = smq2Q_XYCPHASE.germs(qubits) if germs is None else germs

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

    return TwoQubitGST(
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


def QuantumInstrumentGST(
        qpu:            QPU,
        config:         Config,
        qubits:         Iterable[int],
        pspec:          Any | None = None,
        target_model:   Any | None = None,
        prep_fiducials: Any | None = None,
        meas_fiducials: Any | None = None,
        germs:          Any | None = None,
        circuit_depths: List[int] = [1],  # noqa: B006
        modes:          Tuple[str] = ('Target', 'full TP'),  # noqa: B006
        **kwargs
    ) -> Callable:
    """Quantum Instrument Gate Set Tomography.

    This protocol requires a valid pyGSTi installation.
    """
    try:
        import pygsti
        import pygsti.report.reportables as metrics
        logger.info(f" pyGSTi version: {pygsti.__version__}\n")
    except ImportError:
        logger.warning(' Unable to import pyGSTi!')

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
        **kwargs
    ))

    class QuantumInstrumentGST(gst):
        """QI GST protocol."""

        @save_init
        def __init__(self,
                config:         Config,
                qubits:         Iterable[int],
                pspec:          Any | None = None,
                target_model:   Any | None = None,
                prep_fiducials: Any | None = None,
                meas_fiducials: Any | None = None,
                germs:          Any | None = None,
                circuit_depths: List[int] = [1],  # noqa: B006
                modes:          Tuple[str] = ('Target', 'full TP'),  # noqa: B006
                **kwargs
            ) -> None:
            import pygsti
            from pygsti.modelmembers.instruments import Instrument
            from pygsti.modelpacks import smq1Q_XYI

            if len(qubits) > 2:
                raise ValueError(
                    'Quantum Instrument GST is not currently supported for '
                    'more than 2 qubits!'
                )

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
                **kwargs
            )

        @property
        def diamond_norm(self) -> Dict[str, Dict[str, float]]:
            """Diamond norm for the gates in the gate set.

            Returns:
                Dict[str, Dict[str, float]]: diamond norm for each gate in the
                    gate set for each model.
            """
            diamondnorm = {}
            for mode, model in self.models.items():
                diamondnorm[mode] = {
                    str(gate): metrics.half_diamond_norm(
                        self.target_model.operations[gate].to_dense(),
                        op.to_dense(),
                        'pp'
                    )
                    for gate, op in model.operations.items()
                } | { # TODO: discrepancy between this value and reported value
                    f'{instr}': metrics.half_diamond_norm(
                        sum(effect.to_dense() for effect in effects.values()),
                        sum(
                            self.target_model.instruments[instr][p].to_dense()
                            for p in effects.keys()
                        ),
                        'pp'
                    )
                    for instr, effects in model.instruments.items()
                }
            return diamondnorm

        @property
        def entanglement_infidelity(self) -> Dict[str, Dict[str, float]]:
            """Entanglement infidelity for the gates in the gate set.

            Returns:
                Dict[str, Dict[str, float]]: entanglement infidelity for each
                    gate in the gate set for each model.
            """
            infidelity = {}
            for mode, model in self.models.items():
                infidelity[mode] = {
                    str(gate): metrics.entanglement_infidelity(
                        self.target_model.operations[gate].to_dense(),
                        op.to_dense(),
                        'pp'
                    )
                    for gate, op in model.operations.items()
                } | { # TODO: discrepancy between this value and reported value
                    f'{instr}': metrics.entanglement_infidelity(
                        sum(effect.to_dense() for effect in effects.values()),
                        sum(
                            self.target_model.instruments[instr][p].to_dense()
                            for p in effects.keys()
                        ),
                        'pp'
                    )
                    for instr, effects in model.instruments.items()
                }
            return infidelity

        @property
        def ptm(self) -> Dict[str, Dict[str, NDArray]]:
            """Pauli Transfer Matrices for each gate in the gate set.

            Returns:
                Dict[str, Dict[str, NDArray]]: PTM for each gate in the gate
                    set for each fitted model.
            """
            ptm = {}
            for mode, model in self.models.items():
                ptm[mode] = {
                    str(gate): op.to_dense()
                    for gate, op in model.operations.items()
                } | {
                    f'{instr}.{p}': effect.to_dense()
                    for instr, effects in model.instruments.items()
                    for p, effect in effects.items()
                }
            return ptm

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)

            classified_results = load_from_pickle(
                self.data_manager.save_path + 'classified_results.pkl'
            )
            pp = PostProcessor(
                self._config,
                classified_results,
                passes=[
                    discard_heralded_shots,
                    relabel_esp,
                    compute_conditional_counts
                ]
            )
            pp.run()

            results_df = pd.DataFrame(pp.tm_results)
            results_df.index = self._circuits['pygsti_circuit'].values
            results_df = results_df.apply(
                lambda col: col.round().astype('Int64')
            ).astype(object)
            results_df = results_df.fillna('--')
            results_df = results_df.rename(columns=lambda col: col + ' count')

            logger.info(" Saving the pyGSTi results...")
            filepath = self.data_manager.save_path + 'data/dataset.txt'
            with open(filepath, 'w') as f:
                f.write('## Columns = ' + ', '.join(results_df.columns) + "\n")
                f.close()
            results_df.to_csv(filepath, sep=' ', mode='a', header=False)

            if settings.Settings.save_data:
                qpu.save(self, create_data_path=False)

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
        **kwargs
    )
