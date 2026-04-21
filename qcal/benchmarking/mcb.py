"""Submodule for Mirror Circuit Benchmarking.

Relevant papers:
- https://www.nature.com/articles/s41567-021-01409-7,
- https://arxiv.org/abs/2008.11294
"""
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import plotly.graph_objects as go
import pygsti
from IPython.display import clear_output
from plotly.subplots import make_subplots
from pygsti.processors import CliffordCompilationRules as CCR
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import ByDepthSummaryStatistics as SummaryStats
from pygsti.protocols import (
    CombinedExperimentDesign,
    MirrorRBDesign,
    PeriodicMirrorCircuitDesign,
    SimpleRunner,
)
from pygsti.protocols.protocol import ProtocolData

from qcal.circuit import CircuitSet
from qcal.config import Config
from qcal.gate.two_qubit import TWO_QUBIT_GATES
from qcal.interface.pygsti.datasets import generate_pygsti_dataset
from qcal.interface.pygsti.processor_spec import pygsti_pspec
from qcal.interface.pygsti.transpiler import PyGSTiTranspiler
from qcal.qpu.qpu import QPU
from qcal.settings import Settings

logger = logging.getLogger(__name__)


ESTIMATED_QUBIT_ERROR_RATE = 0.005
TARGET_POLARIZATION = 0.01


def _build_mcb_edesigns_for_qubit_subset(
    qs: tuple,
    depths: list,
    pspec: QubitProcessorSpec,
    compilations: dict,
    n_circuits: int,
    two_qubit_gate_density: float,
) -> dict:
    """Build RMCS and PMCS experiment designs for a single qubit subset.

    Args:
        qs (tuple): qubit subset.
        depths (list): circuit depths for this subset.
        pspec (QubitProcessorSpec): pyGSTi processor spec.
        compilations (dict): Clifford compilation rules.
        n_circuits (int): number of circuits per depth.
        two_qubit_gate_density (float): density of two-qubit gates.

    Returns:
        dict: mapping of (qs, circuit_type) to experiment design.
    """
    qubit_labels = tuple(f'Q{q}' for q in qs)

    rmcs = MirrorRBDesign(
        pspec=pspec,
        depths=depths,
        circuits_per_depth=n_circuits,
        clifford_compilations=compilations,
        qubit_labels=qubit_labels,
        sampler='edgegrab',
        samplerargs=[2 * two_qubit_gate_density],
    )
    rmcs_density = np.mean(
        [
            [
                (2 * c.two_q_gate_count() / c.size) if c.size > 0 else 0
                for c in cl
            ] for cl in rmcs.circuit_lists
        ][1:]
    )
    logger.info(f' Interacting qubit density for {qs} RMCs: {rmcs_density:.3f}')

    pmcs = PeriodicMirrorCircuitDesign(
        pspec=pspec,
        depths=depths,
        circuits_per_depth=n_circuits,
        clifford_compilations=compilations,
        qubit_labels=qubit_labels,
        sampler='edgegrab',
        samplerargs=[two_qubit_gate_density],
    )
    pmcs_density = np.mean(
        [
            [
                (2 * c.two_q_gate_count() / c.size) if c.size > 0 else 0
                for c in cl
            ] for cl in pmcs.circuit_lists
        ][1:]
    )
    logger.info(f' Interacting qubit density for {qs} PMCs: {pmcs_density:.3f}')

    return {(qs, 'RMCS'): rmcs, (qs, 'PMCS'): pmcs}


def _trim_depths(
    depths: list,
    width: int,
    estimated_qubit_error_rate: float = ESTIMATED_QUBIT_ERROR_RATE,
    target_polarization: float = TARGET_POLARIZATION,
) -> list:
    """Heuristic function for automatically removing depths that are too long.

    This function can be used to trim MCB circuit depths so that they are not
    too long. If the circuit depths are too long, you will not get useful data
    and the runtime will be unnecessarily long.

    Args:
        depths (List): list of circuit depths to trim
        width (int): circuit width
        estimated_qubit_error_rate (float): estimated per-qubit error rate.
            Defaults to ESTIMATED_QUBIT_ERROR_RATE.
        target_polarization (float): target polarization at which to cut off
            depths. Defaults to TARGET_POLARIZATION.

    Returns:
        list: trimmed circuit depths
    """
    max_depth = np.log(target_polarization) / (
            width * np.log(1 - estimated_qubit_error_rate)
        )
    trimmed_depths = [d for d in depths if d < max_depth]
    n_depths = len(trimmed_depths)
    if n_depths < len(depths) and trimmed_depths[-1] < max_depth:
        trimmed_depths.append(depths[n_depths])

    return trimmed_depths


def _vb_to_heatmap_data(vb_data) -> tuple[list, list, np.ndarray]:
    """Convert a pyGSTi VBData dict to (widths, depths, z-matrix) for Plotly."""
    items = {}
    for (w, d), v in vb_data.items():
        try:
            fv = float(v)
            if not np.isnan(fv):
                items[(w, d)] = fv
        except (TypeError, ValueError):
            pass
    if not items:
        return [], [], np.array([[]])
    widths = sorted({w for w, _ in items})
    depths = sorted({d for _, d in items})
    matrix = np.full((len(widths), len(depths)), np.nan)
    for (w, d), val in items.items():
        matrix[widths.index(w), depths.index(d)] = val
    return widths, depths, matrix


def MCB(
    qpu:                     QPU,
    config:                  Config,
    qubits:                  Sequence[int] | None = None,
    circuit_depths:          Sequence[int] | None = None,
    circuit_widths:          Sequence[int] | None = None,
    n_circuits:              int = 20,
    two_qubit_gate_density:  float = 0.125,
    est_qubit_error_rate:    float = ESTIMATED_QUBIT_ERROR_RATE,
    target_polarization:     float = TARGET_POLARIZATION,
    qubit_subsets_per_width: dict[int, Sequence[int]] | None = None,
    pspec:                   QubitProcessorSpec | None = None,
    **kwargs
) -> Callable:
    """Mirror Circuit Benchmarking.

    This is a pyGSTi protocol.

    Relevant papers:
    - https://www.nature.com/articles/s41567-021-01409-7,
    - https://arxiv.org/abs/2008.11294

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (Sequence[int] | None, optional): sequence of qubits to
            benchmark. Defaults to None, in which case all qubits in the config
            are used.
        circuit_depths (Sequence[int] | None, optional): sequence of circuit
            depths. Defaults to None, in which case depths are automatically
            generated and trimmed based on the estimated qubit error rate.
        circuit_widths (Sequence[int] | None, optional): sequence of circuit
            widths. Defaults to None, in which case all widths from 1 to the
            total number of qubits are used.
        n_circuits (int, optional): number of circuits per (depth, width)
            combination. Defaults to 20.
        qubit_subsets_per_width (dict[int, Sequence[int]] | None, optional):
            dictionary mapping each width to a sequence of qubit subsets to
            benchmark at that width. Defaults to None, in which case a single
            subset of the first `width` qubits is used for each width.
        two_qubit_gate_density (float, optional): density of two-qubit gates in
            the random mirror circuits. Defaults to 0.125 (1/8).
        estimated_qubit_error_rate (float, optional): estimated per-qubit error
            rate used to trim depths. Defaults to ESTIMATED_QUBIT_ERROR_RATE
            (0.005).
        target_polarization (float, optional): target polarization at which to
            cut off depths. Defaults to TARGET_POLARIZATION (0.01).
        pspec (QubitProcessorSpec | None, optional): pyGSTi processor spec.
            Defaults to None, in which case a processor spec is automatically
            generated from the config.

    Returns:
        Callable: MCB class instance.
    """

    class MCB(qpu):
        """pyGSTi Mirror Circuit Benchmarking protocol."""

        def __init__(
            self,
            config:                  Config,
            qubits:                  Sequence[int] | None = None,
            circuit_depths:          Sequence[int] | None = None,
            circuit_widths:          Sequence[int] | None = None,
            n_circuits:              int = 20,
            two_qubit_gate_density:  float = 0.125,
            est_qubit_error_rate:    float = ESTIMATED_QUBIT_ERROR_RATE,
            target_polarization:     float = TARGET_POLARIZATION,
            qubit_subsets_per_width: dict[int, Sequence[int]] | None = None,
            pspec:                   QubitProcessorSpec | None = None,
            **kwargs
        ) -> None:
            logger.info(f" pyGSTi version: {pygsti.__version__}\n")

            if qubits is not None:
                self._qubits = sorted(qubits)
            elif qubit_subsets_per_width is not None:
                self._qubits = sorted(
                    {q for subsets in qubit_subsets_per_width.values()
                     for subset in subsets for q in subset}
                )
            else:
                self._qubits = config.qubits
            self._n_circuits = n_circuits
            self._two_qubit_gate_density = two_qubit_gate_density

            if circuit_widths is not None:
                if any(w < 1 or w > len(self._qubits) for w in circuit_widths):
                    raise ValueError(
                        f"Widths must be between 1 and {len(self._qubits)}!"
                    )
                else:
                    self._circuit_widths = sorted(circuit_widths)
            elif qubit_subsets_per_width is not None:
                self._circuit_widths = sorted(qubit_subsets_per_width.keys())
            else:
                self._circuit_widths = list(range(1, len(self._qubits) + 1))

            if circuit_depths is not None:
                if 1 in circuit_depths:
                    raise ValueError("A depth of 1 is not allowed in MCB!")
                else:
                    self._circuit_depths = sorted(circuit_depths)
                    self._circuit_depths_per_width = {
                        width: _trim_depths(
                            self._circuit_depths,
                            width,
                            est_qubit_error_rate,
                            target_polarization,
                        ) for width in self._circuit_widths
                    }
            else:
                self._circuit_depths = [0] + [
                    int(d) for d in 2**np.arange(1, 10)
                ]
                self._circuit_depths_per_width = {
                    width: _trim_depths(
                        self._circuit_depths,
                        width,
                        est_qubit_error_rate,
                        target_polarization,
                    ) for width in self._circuit_widths
                }

            if qubit_subsets_per_width is not None:
                for width, subsets in qubit_subsets_per_width.items():
                    if width not in self._circuit_widths:
                        raise ValueError(
                            f"Width {width} in qubit_subsets_per_width is not "
                            f"in circuit_widths!"
                        )
                    for subset in subsets:
                        if len(subset) != width:
                            raise ValueError(
                                f"Subset {subset} does not have the correct "
                                f"number of qubits for width {width}!"
                            )
                        if any(q not in self._qubits for q in subset):
                            raise ValueError(
                                f"Subset {subset} contains qubits that are "
                                f"not in the list of qubits to benchmark!"
                            )
                self._qubit_subsets_per_width = qubit_subsets_per_width
            else:
                self._qubit_subsets_per_width = {}
                for width in self._circuit_widths:
                    self._qubit_subsets_per_width[width] = [
                        tuple(self._qubits[:width])
                    ]

            if pspec is not None:
                self._pspec = pspec
            else:
                gate_set = [f'Gc{i}' for i in range(24)]
                for gate in config.native_gates['set']:
                    if gate in TWO_QUBIT_GATES:
                        gate_set.append(gate)
                self._pspec = pspec if pspec is not None else pygsti_pspec(
                    config, self._qubits, gate_set
                )

            self._compilations = {
                'absolute': CCR.create_standard(self._pspec, verbosity=0)
            }

            transpiler = kwargs.get('transpiler', PyGSTiTranspiler())
            kwargs.pop('transpiler', None)
            qpu.__init__(self, config=config, transpiler=transpiler, **kwargs)

            self._data = None
            self._dataset = None
            self._edesigns = {}
            self._edesign = None
            self._results = None
            self._summary = None

        @property
        def circuit_depths(self) -> list:
            """Circuit depths."""
            return self._circuit_depths

        @property
        def circuit_widths(self) -> list:
            """Circuit widths."""
            return self._circuit_widths

        @property
        def circuit_depths_per_width(self) -> dict[int, list]:
            """Circuit depths per width."""
            return self._circuit_depths_per_width

        @property
        def edesign(self) -> CombinedExperimentDesign | None:
            """Experiment design."""
            return self._edesign

        @property
        def results(self) -> SummaryStats | None:
            """Summary statistics results."""
            return self._results

        @property
        def qubits(self) -> list:
            """Qubits to benchmark."""
            return self._qubits

        @property
        def qubit_subsets_per_width(self) -> dict[int, list]:
            """Qubit subsets per width."""
            return self._qubit_subsets_per_width

        def generate_circuits(self):
            """Generate all pyGSTi mirror circuit benchmarking circuits."""
            logger.info(' Generating circuits from pyGSTi...')

            tasks = [
                (qs, self._circuit_depths_per_width[width])
                for width, subsets in self._qubit_subsets_per_width.items()
                for qs in subsets
            ]

            max_workers = len(tasks)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        _build_mcb_edesigns_for_qubit_subset,
                        qs,
                        depths,
                        self._pspec,
                        self._compilations,
                        self._n_circuits,
                        self._two_qubit_gate_density,
                    )
                    for qs, depths in tasks
                ]
                results = [f.result() for f in futures]

            edesigns = {}
            for result in results:
                edesigns.update(result)

            self._edesign = CombinedExperimentDesign(edesigns)
            self._circuits = CircuitSet(self._edesign.all_circuits_needing_data)
            self._circuits['pygsti_circuit'] = [
                circ.str for circ in self._edesign.all_circuits_needing_data
            ]

            self._data_manager._exp_id += (
                f'_MCB_{"".join("Q"+str(q) for q in self._qubits)}'
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

            # The statistics to compute for each circuit.
            statistics = [
                'polarization', 'success_probabilities', 'success_counts',
                'total_counts', 'two_q_gate_count'
            ]
            stats_generator = SimpleRunner(
                SummaryStats(statistics_to_compute=statistics)
            )

            # Computes the stats
            self._results = stats_generator.run(self._data)

            # Turns this "summary" data into a DataFrame
            self._summary = self._results.to_dataframe(
                'ValueName', drop_columns=['ProtocolName', 'ProtocolType']
            )

            # Adds a row that tells us which type of circuit the row is for.
            # Will not work if the `keys` in the
            # edesign are changed to not include `RMCs` or `PMCs`.
            self._summary['CircuitType'] = [
               'RMC' if 'RMCS' in p[0] else 'PMC' for p in self._summary['Path']
            ]

            # Redefines "depth" as twice what is in the Depth column, because
            # the circuit generation code currently
            # uses a different convention to that used in arXiv:2008.11294.
            self._summary['Depth'] = 2*self._summary['Depth']

        def plot(self) -> None:
            """Plot the results."""
            # Puts the DataFrame into VBDataFrame object that can be used to
            # create VB plots
            vbdf = pygsti.protocols.VBDataFrame(self._summary)
            fig, ax = pygsti.report.capability_region_plot(
                vbdf, figsize=(6, 8), scale=2
            )
            fig.show()
            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'capability_regions.png',
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'capability_regions.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'capability_regions.svg'
                )

            # Extracts the data for a plot like Fig. 2a of arXiv:2008.11294.
            vb_min = {}
            for circuit_type in ('RMC', 'PMC'):
                vbdf1 = vbdf.select_column_value('CircuitType', circuit_type)
                vb_min[circuit_type] = vbdf1.vb_data(
                    metric='polarization',
                    no_data_action='min',
                    statistic='monotonic_min',
                )

            # Creates the plot like those in Fig. 2a of arXiv:2008.11294. The
            # inner squares are the randomized mirror circuits, and the outer
            # squares are the periodic mirror circuits.
            # spectral = pygsti.report.spectral
            fig, ax = pygsti.report.volumetric_plot(
                vb_min['PMC'], scale=1.9, cmap='Spectral', figsize=(5.5,8)
            )
            fig, ax = pygsti.report.volumetric_plot(
                vb_min['RMC'], scale=0.4, cmap='Spectral',
                fig=fig, ax=ax, linescale=0.
            )
            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'volumetric.png',
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'volumetric.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'volumetric.svg'
                )

            # Creates a plot like those in Fig. 1d of arXiv:2008.11294. But note
            # that these RMCs don't have the same sampling as those in Fig. 1d:
            # this is just the same type of plot from RMC data, not the same
            # type of RMCs. To get the same color map as in Fig. 1d, set
            # cmap=None
            vbdf2 = vbdf.select_column_value('CircuitType', 'RMC')
            fig, ax = pygsti.report.volumetric_distribution_plot(
                vbdf2, figsize=(5.5,8), cmap=None
            )
            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'frontier.png', dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'frontier.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'frontier.svg'
                )

            # Plotly: capability region (binary heatmap)
            # _THRESHOLD = 0.1
            # pfig_cap = make_subplots(
            #     rows=1, cols=2,
            #     subplot_titles=['PMC', 'RMC'],
            #     shared_yaxes=True,
            #     horizontal_spacing=0.1,
            # )
            # widths = []
            # for col, ct in enumerate(('PMC', 'RMC'), 1):
            #     widths, depths, matrix = _vb_to_heatmap_data(vb_min[ct])
            #     capable = np.where(
            #         np.isnan(matrix), np.nan,
            #         (matrix > _THRESHOLD).astype(float),
            #     )
            #     pfig_cap.add_trace(
            #         go.Heatmap(
            #             x=[str(d) for d in depths],
            #             y=widths,
            #             z=capable,
            #             colorscale=[[0, '#d62728'], [1, '#2ca02c']],
            #             zmin=0,
            #             zmax=1,
            #             showscale=False,
            #             customdata=matrix,
            #             hovertemplate=(
            #                 'Width: %{y}<br>Depth: %{x}'
            #                 '<br>Polarization: %{customdata:.3f}<extra></extra>'
            #             ),
            #         ),
            #         row=1, col=col,
            #     )
            # pfig_cap.update_xaxes(title_text='Depth', type='category')
            # pfig_cap.update_yaxes(title_text='Width', dtick=1, row=1, col=1)
            # pfig_cap.update_layout(
            #     title=f'Capability Region (threshold = {_THRESHOLD})',
            #     template='plotly_white',
            #     height=max(400, 80 * (len(widths) + 2)),
            #     width=750,
            # )
            # pfig_cap.show()
            # if Settings.save_data:
            #     pfig_cap.write_html(
            #         self._data_manager._save_path + 'capability_regions.html'
            #     )

            # # Plotly: volumetric polarization (continuous heatmap)
            # pfig_vol = make_subplots(
            #     rows=1, cols=2,
            #     subplot_titles=['PMC', 'RMC'],
            #     shared_yaxes=True,
            #     horizontal_spacing=0.1,
            # )
            # for col, ct in enumerate(('PMC', 'RMC'), 1):
            #     widths, depths, matrix = _vb_to_heatmap_data(vb_min[ct])
            #     pfig_vol.add_trace(
            #         go.Heatmap(
            #             x=[str(d) for d in depths],
            #             y=widths,
            #             z=matrix,
            #             colorscale='RdYlGn',
            #             zmin=0,
            #             zmax=1,
            #             colorbar={
            #                 'title': 'Polarization', 'thickness': 15,
            #                 'len': 0.9, 'x': 1.02,
            #             },
            #             showscale=(col == 2),
            #             hovertemplate=(
            #                 'Width: %{y}<br>Depth: %{x}'
            #                 '<br>Polarization: %{z:.3f}<extra></extra>'
            #             ),
            #         ),
            #         row=1, col=col,
            #     )
            # pfig_vol.update_xaxes(title_text='Depth', type='category')
            # pfig_vol.update_yaxes(title_text='Width', dtick=1, row=1, col=1)
            # pfig_vol.update_layout(
            #     title='Volumetric Polarization',
            #     template='plotly_white',
            #     height=max(400, 80 * (len(widths) + 2)),
            #     width=750,
            # )
            # pfig_vol.show()
            # if Settings.save_data:
            #     pfig_vol.write_html(
            #         self._data_manager._save_path + 'polarization.html'
            #     )

            # # Plotly: RMC frontier (polarization vs depth per width)
            # if vb_min['RMC']:
            #     rmc_widths = sorted({w for w, _ in vb_min['RMC']})
            #     pfig_front = go.Figure()
            #     for w in rmc_widths:
            #         wd_pairs = sorted(
            #             (d, v) for (wi, d), v in vb_min['RMC'].items() if wi == w
            #         )
            #         if wd_pairs:
            #             ds = [d for d, _ in wd_pairs]
            #             ps = [p for _, p in wd_pairs]
            #             pfig_front.add_trace(go.Scatter(
            #                 x=[str(d) for d in ds],
            #                 y=ps,
            #                 mode='lines+markers',
            #                 name=f'Width {w}',
            #             ))
            #     pfig_front.update_layout(
            #         title='RMC Polarization Frontier',
            #         xaxis_title='Depth',
            #         yaxis_title='Polarization',
            #         template='plotly_white',
            #         height=450,
            #         width=600,
            #     )
            #     pfig_front.show()
            #     if Settings.save_data:
            #         pfig_front.write_html(
            #             self._data_manager._save_path + 'frontier.html'
            #         )

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

    return MCB(
        config=config,
        qubits=qubits,
        circuit_depths=circuit_depths,
        circuit_widths=circuit_widths,
        n_circuits=n_circuits,
        qubit_subsets_per_width=qubit_subsets_per_width,
        two_qubit_gate_density=two_qubit_gate_density,
        est_qubit_error_rate=est_qubit_error_rate,
        target_polarization=target_polarization,
        pspec=pspec,
        **kwargs
    )
