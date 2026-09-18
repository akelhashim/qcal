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
import pandas as pd
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
from qcal.gates.two_qubit import TWO_QUBIT_GATES
from qcal.interface.pygsti.datasets import generate_pygsti_dataset
from qcal.interface.pygsti.processor_spec import pygsti_pspec
from qcal.interface.pygsti.transpiler import PyGSTiTranspiler
from qcal.qpu.qpu import QPU
from qcal.settings import Settings

logger = logging.getLogger(__name__)


ESTIMATED_QUBIT_ERROR_RATE = 0.005
TARGET_POLARIZATION = 0.01


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
        def summary(self) -> pd.DataFrame | None:
            """Summary statistics as a DataFrame."""
            return self._summary

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

            # Plotly: capability region -- discrete success /
            # indeterminate / fail squares, matching pyGSTi's own
            # capability_region_plot colors and Fig. 3 of
            # arXiv:2008.11294.
            _CAP_THRESHOLD = 1 / np.e
            _CAP_STYLE = {
                2: ('#33a02c', 'All circuits succeed'),
                1: ('#fdbf6f', 'Some circuits succeed'),
                0: ('#ffffff', 'No circuits succeed'),
            }
            pfig_cap = make_subplots(
                rows=1, cols=2,
                subplot_titles=['PMC', 'RMC'],
                shared_yaxes=True,
                horizontal_spacing=0.15,
            )
            cap_widths = []
            for col, ct in enumerate(('PMC', 'RMC'), 1):
                vbdf1 = vbdf.select_column_value('CircuitType', ct)
                cap_depths, cap_widths = vbdf1.x_values, vbdf1.y_values
                creg = vbdf1.capability_regions(
                    metric='polarization',
                    threshold=_CAP_THRESHOLD,
                    significance=0.05,
                    monotonic=True,
                )
                for level in (2, 1, 0):
                    color, label = _CAP_STYLE[level]
                    xs, ys, text = [], [], []
                    for (d, w), val in creg.items():
                        if val == level:
                            xs.append(cap_depths.index(d))
                            ys.append(cap_widths.index(w))
                            text.append(f'Width: {w}<br>Depth: {d}')
                    pfig_cap.add_trace(
                        go.Scatter(
                            x=xs, y=ys,
                            mode='markers',
                            marker={
                                'symbol': 'square',
                                'size': 26,
                                'color': color,
                                'line': {'color': 'black', 'width': 1},
                            },
                            name=label,
                            legendgroup=str(level),
                            showlegend=(col == 1),
                            text=text,
                            hovertemplate='%{text}<extra></extra>',
                        ),
                        row=1, col=col,
                    )
                _set_volumetric_axes(
                    pfig_cap, cap_depths, cap_widths, row=1, col=col
                )
            pfig_cap.update_layout(
                title=(
                    'Capability Region '
                    f'(threshold = 1/e = {_CAP_THRESHOLD:.3f})'
                ),
                template='plotly_white',
                height=max(260, 55 * len(cap_widths) + 230),
                width=800,
                margin={'b': 90},
                legend={
                    'orientation': 'h',
                    'yanchor': 'top',
                    'y': -0.18,
                    'xanchor': 'left',
                    'x': 0,
                },
            )
            pfig_cap.show()

            # Plotly: volumetric polarization -- continuous squares
            # colored on the same 'Spectral' scale pyGSTi uses for
            # this plot (see the matplotlib volumetric_plot() calls
            # above), matching Fig. 2a of arXiv:2008.11294.
            pfig_vol = make_subplots(
                rows=1, cols=2,
                subplot_titles=['PMC', 'RMC'],
                shared_yaxes=True,
                horizontal_spacing=0.15,
            )
            vol_widths = []
            for col, ct in enumerate(('PMC', 'RMC'), 1):
                vbdf1 = vbdf.select_column_value('CircuitType', ct)
                vol_depths, vol_widths = vbdf1.x_values, vbdf1.y_values
                xs, ys, zs, text = [], [], [], []
                for (d, w), v in vb_min[ct].items():
                    if not np.isnan(v):
                        xs.append(vol_depths.index(d))
                        ys.append(vol_widths.index(w))
                        zs.append(v)
                        text.append(f'Width: {w}<br>Depth: {d}')
                pfig_vol.add_trace(
                    go.Scatter(
                        x=xs, y=ys,
                        mode='markers',
                        marker={
                            'symbol': 'square',
                            'size': 26,
                            'color': zs,
                            'colorscale': 'Spectral',
                            'cmin': 0, 'cmax': 1,
                            'showscale': (col == 2),
                            'colorbar': (
                                {'title': 'Polarization'}
                                if col == 2 else None
                            ),
                            'line': {'color': 'black', 'width': 1},
                        },
                        text=text,
                        hovertemplate=(
                            '%{text}<br>Polarization: '
                            '%{marker.color:.3f}<extra></extra>'
                        ),
                        showlegend=False,
                    ),
                    row=1, col=col,
                )
                _set_volumetric_axes(
                    pfig_vol, vol_depths, vol_widths, row=1, col=col
                )
            pfig_vol.update_layout(
                title='Volumetric Polarization',
                template='plotly_white',
                height=max(260, 55 * len(vol_widths) + 170),
                width=800,
            )
            pfig_vol.show()

            # Plotly: RMC volumetric distribution plot, reproducing
            # Fig. 1d of arXiv:2008.11294 (and pyGSTi's own
            # volumetric_distribution_plot). Nested squares show the
            # min/mean/max polarization at each (depth, width), and
            # stepped boundary lines mark where each statistic
            # crosses the success threshold (1/e): green = max
            # (best-case capable), black = mean, red = min
            # (worst-case capable).
            _FRONTIER_THRESHOLD = 1 / np.e
            rmc_depths = vbdf2.x_values
            rmc_widths = vbdf2.y_values

            if rmc_depths and rmc_widths:
                vb_stat = {
                    stat: vbdf2.vb_data(
                        metric='polarization', statistic=stat,
                        no_data_action='discard',
                    ) for stat in ('min', 'mean', 'max')
                }
                capability = vbdf2.capability_regions(
                    metric='polarization',
                    threshold=_FRONTIER_THRESHOLD,
                    significance=0.05,
                    monotonic=True,
                )

                pfig_front = go.Figure()

                # Nested squares: min (outer/largest) -> mean ->
                # max (inner/smallest), each colored by polarization
                # on pyGSTi's default 'Blues' scale for this plot
                # (volumetric_distribution_plot's cmap=None).
                square_sizes = {'min': 34, 'mean': 20, 'max': 8}
                for stat in ('min', 'mean', 'max'):
                    xs, ys, zs, text = [], [], [], []
                    for (d, w), v in vb_stat[stat].items():
                        xs.append(rmc_depths.index(d))
                        ys.append(rmc_widths.index(w))
                        zs.append(v)
                        text.append(f'Width: {w}<br>Depth: {d}')
                    pfig_front.add_trace(go.Scatter(
                        x=xs, y=ys,
                        mode='markers',
                        marker={
                            'symbol': 'square',
                            'size': square_sizes[stat],
                            'color': zs,
                            'colorscale': 'Blues',
                            'cmin': 0, 'cmax': 1,
                            'showscale': (stat == 'min'),
                            'colorbar': (
                                {'title': 'Polarization'}
                                if stat == 'min' else None
                            ),
                            'line': {'color': 'black', 'width': 0.5},
                        },
                        name=stat,
                        text=text,
                        hovertemplate=(
                            '%{text}<br>' + stat.capitalize()
                            + ' polarization: %{marker.color:.3f}'
                            '<extra></extra>'
                        ),
                        showlegend=False,
                    ))

                # Boundary lines, using the paper's own naming (see
                # Fig. 1 of arXiv:2008.11294): green = Best Circuit
                # (max), black = Average Circuit (mean), red = Worst
                # Circuit (min) -- the same three statistics as the
                # nested squares above. Drawn with 'Average Circuit'
                # (solid, opaque) first so it sits behind the dashed/
                # dotted lines: when boundaries coincide (as they
                # often do on clean/low-shot data), the dash gaps
                # still let every color show through, rather than the
                # last trace fully occluding the others. `legendrank`
                # reorders the legend independently of this draw
                # order, so it can still read Best/Average/Worst.
                boundary_specs = [
                    ('Average Circuit', vb_stat['mean'],
                     _FRONTIER_THRESHOLD, '#000000', False, 6,
                     'solid', 2),
                    ('Best Circuit', capability, 0.99, '#2ecc71',
                     True, 5, 'dash', 1),
                    ('Worst Circuit', capability, 1.99, '#e74c3c',
                     True, 3, 'dot', 3),
                ]
                for label, data, thr, color, monotonic, lw, dash, rank in (
                    boundary_specs
                ):
                    xvals, yvals = _volumetric_boundary_steps(
                        data, rmc_depths, rmc_widths, thr,
                        monotonic=monotonic,
                    )
                    pfig_front.add_trace(go.Scatter(
                        x=xvals, y=yvals,
                        mode='lines',
                        line={'color': color, 'width': lw, 'dash': dash},
                        name=label,
                        legendrank=rank,
                        hoverinfo='skip',
                    ))

                _set_volumetric_axes(pfig_front, rmc_depths, rmc_widths)
                pfig_front.update_layout(
                    title=(
                        'RMC Volumetric Distribution '
                        f'(threshold = 1/e = {_FRONTIER_THRESHOLD:.3f})'
                    ),
                    template='plotly_white',
                    height=max(260, 55 * len(rmc_widths) + 230),
                    width=750,
                    margin={'b': 90},
                    legend={
                        'orientation': 'h',
                        'yanchor': 'top',
                        'y': -0.18,
                        'xanchor': 'left',
                        'x': 0,
                    },
                )
                pfig_front.show()

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


def _set_volumetric_axes(
    fig: go.Figure,
    depths: list,
    widths: list,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Style a Plotly figure's axes to match pyGSTi's VB plot grid.

    pyGSTi's volumetric-benchmarking plots position every
    (depth, width) square by *index* into the sorted depth/width
    lists, not by raw value. This applies that same convention to a
    Plotly figure (or one subplot of it) so index-positioned square
    markers and boundary lines line up correctly. Marker squares are
    sized in fixed pixels (not data units), so no data-unit aspect
    locking is needed for them to render as literal squares -- doing
    so would instead waste vertical space padding a plot with few
    widths but many depths.

    Args:
        fig (go.Figure): the figure (or subplot grid) to style.
        depths (list): sorted depths for the x-axis.
        widths (list): sorted widths for the y-axis.
        row (int | None, optional): subplot row, if ``fig`` was built
            with ``make_subplots``. Defaults to None.
        col (int | None, optional): subplot column, if ``fig`` was
            built with ``make_subplots``. Defaults to None.
    """
    target = {} if row is None else {'row': row, 'col': col}
    fig.update_xaxes(
        title_text='Depth',
        tickmode='array',
        tickvals=list(range(len(depths))),
        ticktext=[str(d) for d in depths],
        range=[-0.5, len(depths) - 0.5],
        **target,
    )
    fig.update_yaxes(
        title_text='Width',
        tickmode='array',
        tickvals=list(range(len(widths))),
        ticktext=[str(w) for w in widths],
        range=[-0.5, len(widths) - 0.5],
        **target,
    )

def _volumetric_boundary_steps(
    data: dict,
    x_values: list,
    y_values: list,
    threshold: float,
    monotonic: bool = True,
) -> tuple[list, list]:
    """Index-based staircase boundary for a volumetric-style dataset.

    A direct, non-matplotlib port of the ``missing_data_action=
    'hedge'`` branch of ``pygsti.report.volumetric_boundary_plot``
    (see arXiv:2008.11294, Fig. 1), which pyGSTi's own
    ``volumetric_distribution_plot`` uses internally. It reproduces
    the same statistics/hedging, only returning coordinates instead
    of drawing them, so this can be plotted with Plotly.

    Args:
        data (dict): mapping of (depth, width) to a scalar metric or
            classification, as returned by ``VBDataFrame.vb_data()``
            or ``VBDataFrame.capability_regions()``.
        x_values (list): sorted depths (the VBDataFrame's x-axis).
        y_values (list): sorted widths (the VBDataFrame's y-axis).
        threshold (float): the value ``data`` must meet or exceed to
            count as "capable" at a given (depth, width).
        monotonic (bool, optional): enforce a non-increasing boundary
            as depth increases. Defaults to True.

    Returns:
        tuple[list, list]: index-based (xvals, yvals) for a staircase
            line, in depth-index and width-index units (offset by 0.5
            so the line sits on the cell edges of an index-positioned
            grid of (depth, width) squares).
    """
    def _widest_capable_index(d: object) -> int:
        return max(
            [-1] + [
                y_values.index(w) for w in y_values
                if (d, w) in data and data[d, w] >= threshold
            ]
        )

    boundaries = [_widest_capable_index(x_values[0])]
    hedged = set()
    for d in x_values[1:]:
        max_width_at_d = max(
            [-1] + [w for w in y_values if (d, w) in data]
        )
        if max_width_at_d < boundaries[-1]:
            boundaries.append(boundaries[-1])
            hedged.add(d)
        else:
            boundaries.append(_widest_capable_index(d))

    xvals, yvals = [], []
    last_x = -0.5
    for i, d in enumerate(x_values):
        if d in hedged:
            if not all(dd in hedged for dd in x_values[i:]):
                xvals += [last_x, i]
                yvals += [boundaries[i] + 0.5, boundaries[i] + 0.5]
        else:
            xvals += [last_x, i + 0.5]
            yvals += [boundaries[i] + 0.5, boundaries[i] + 0.5]
        last_x = xvals[-1]

    if monotonic and yvals:
        mono = [yvals[0]]
        for y in yvals[1:]:
            mono.append(mono[-1] if y > mono[-1] else y)
        yvals = mono

    return xvals, yvals
