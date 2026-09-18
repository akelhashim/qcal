"""Unit tests for qcal.benchmarking.mcb, run against the Emulator.

Like CRB (see test_rb.py), MCB is a pyGSTi-backed protocol, so a full
.run() is closer to an integration test than a fast unit test. Unlike
CRB, there is no pre-generated circuit-design cache for MCB, and its
default depths run up to 512 -- so the "real run" test below uses a
small qubit count *and* small depths/n_circuits to stay fast, rather
than qubit count alone.
"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pygsti
import pytest
from plotly.subplots import make_subplots

from qcal.backend.emulator import Emulator
from qcal.benchmarking.mcb import (
    MCB,
    _set_volumetric_axes,
    _trim_depths,
    _volumetric_boundary_steps,
)
from qcal.config import Config
from qcal.simulation.error_models import DepolarizingNoise
from qcal.simulation.simulators import DensityMatrixSimulator

DEFAULT_DEPTHS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512]

EXAMPLE_CONFIG_PATH = str(
    Path(__file__).resolve().parents[1] / 'examples' / 'config' /
    'config.yaml'
)


class TestTrimDepths:
    """_trim_depths is a plain function -- no pyGSTi/emulator needed."""

    def test_narrow_width_keeps_full_range(self):
        # At the default error rate, a 1-qubit circuit doesn't decay
        # enough by depth 512 to trim anything.
        assert _trim_depths(DEFAULT_DEPTHS, width=1) == DEFAULT_DEPTHS

    def test_wide_width_trims_and_appends_one_boundary_depth(self):
        # At width 8, depths beyond 128 are trimmed, but exactly one
        # depth past the cutoff (128) is kept as a boundary point.
        assert _trim_depths(DEFAULT_DEPTHS, width=8) == (
            [0, 2, 4, 8, 16, 32, 64, 128]
        )

    def test_custom_thresholds(self):
        depths = [0, 2, 4, 8, 16, 32, 64]
        trimmed = _trim_depths(
            depths, width=5,
            estimated_qubit_error_rate=0.05, target_polarization=0.1,
        )
        assert trimmed == [0, 2, 4, 8, 16]


class TestMCBDefaults:

    def test_default_qubits_are_all_config_qubits(self, config):
        mcb = MCB(qpu=Emulator, config=config)
        assert list(mcb.qubits) == sorted(config.qubits)

    def test_default_circuit_widths_span_all_qubits(self, config):
        mcb = MCB(qpu=Emulator, config=config)
        assert mcb.circuit_widths == list(range(1, len(config.qubits) + 1))

    def test_custom_qubits_limit_widths_and_subsets(self, config):
        mcb = MCB(qpu=Emulator, config=config, qubits=[0, 1, 2])
        assert mcb.qubits == [0, 1, 2]
        assert mcb.circuit_widths == [1, 2, 3]
        assert mcb.qubit_subsets_per_width == {
            1: [(0,)], 2: [(0, 1)], 3: [(0, 1, 2)],
        }

    def test_depth_of_one_is_rejected(self, config):
        with pytest.raises(ValueError):
            MCB(
                qpu=Emulator, config=config, qubits=[0, 1, 2],
                circuit_depths=[1, 2],
            )

    def test_width_outside_qubit_range_is_rejected(self, config):
        with pytest.raises(ValueError):
            MCB(
                qpu=Emulator, config=config, qubits=[0, 1, 2],
                circuit_widths=[4],
            )

    def test_qubit_subset_of_wrong_size_is_rejected(self, config):
        with pytest.raises(ValueError):
            MCB(
                qpu=Emulator, config=config, qubits=[0, 1, 2],
                qubit_subsets_per_width={1: [(0, 1)]},
            )


class TestMCBRun:

    def test_small_three_qubit_run_produces_summary_data(self, config):
        # Small on every axis (qubits, depths, and circuit count), not
        # just qubit count, since MCB has no circuit-design cache and
        # defaults to depths up to 512.
        mcb = MCB(
            qpu=Emulator, config=config, qubits=[0, 1, 2],
            circuit_depths=[0, 2, 4], n_circuits=2,
        )
        mcb.run()

        # 3 widths x 3 depths x 2 circuits/depth x 2 circuit types
        # (RMC, PMC).
        assert mcb.circuits.n_circuits == 36

        summary = mcb.summary
        assert len(summary) == 36
        assert set(summary['Width']) == {1, 2, 3}
        assert set(summary['CircuitType']) == {'RMC', 'PMC'}
        # Stored depths are 2x what was passed in -- see the comment
        # in MCB.analyze() about the arXiv:2008.11294 convention.
        assert set(summary['Depth']) == {0, 4, 8}
        assert summary['polarization'].between(-1e-9, 1 + 1e-9).all()


class TestVolumetricBoundarySteps:
    """_volumetric_boundary_steps is a plain function -- no pyGSTi/
    emulator needed. Depths (x) and widths (y) are index-positioned,
    offset by 0.5 so a step sits on the edge of its (depth, width)
    cell -- see the docstring in mcb.py for the full convention.
    """

    def test_boundary_steps_down_as_depth_increases(self):
        x, y = [0, 2, 4], [1, 2, 3]
        data = {
            (0, 1): 1.0, (0, 2): 1.0, (0, 3): 1.0,
            (2, 1): 1.0, (2, 2): 1.0, (2, 3): 0.0,
            (4, 1): 1.0, (4, 2): 0.0, (4, 3): 0.0,
        }
        xvals, yvals = _volumetric_boundary_steps(
            data, x, y, threshold=1.0
        )
        assert xvals == [-0.5, 0.5, 0.5, 1.5, 1.5, 2.5]
        assert yvals == [2.5, 2.5, 1.5, 1.5, 0.5, 0.5]

    def test_monotonic_caps_a_boundary_that_would_otherwise_rise(self):
        # An isolated spike at depth=2 (width=3 capable there, but
        # not at the shallower depth=0) should be clipped by
        # monotonic=True, and left alone by monotonic=False.
        x, y = [0, 2, 4], [1, 2, 3]
        data = {
            (0, 1): 1.0, (0, 2): 1.0, (0, 3): 0.0,
            (2, 1): 1.0, (2, 2): 1.0, (2, 3): 1.0,
            (4, 1): 1.0, (4, 2): 0.0, (4, 3): 0.0,
        }
        mono_x, mono_y = _volumetric_boundary_steps(
            data, x, y, threshold=1.0, monotonic=True
        )
        raw_x, raw_y = _volumetric_boundary_steps(
            data, x, y, threshold=1.0, monotonic=False
        )
        assert mono_x == raw_x == [-0.5, 0.5, 0.5, 1.5, 1.5, 2.5]
        assert mono_y == [1.5, 1.5, 1.5, 1.5, 0.5, 0.5]
        assert raw_y == [1.5, 1.5, 2.5, 2.5, 0.5, 0.5]

    def test_missing_depth_hedges_forward_the_prior_boundary(self):
        # depth=2 has no data at all (untested shape); the boundary
        # should carry the depth=0 value forward across the gap
        # rather than dropping to "nothing capable".
        x, y = [0, 2, 4], [1, 2, 3]
        data = {
            (0, 1): 1.0, (0, 2): 1.0, (0, 3): 1.0,
            (4, 1): 1.0, (4, 2): 0.0, (4, 3): 0.0,
        }
        xvals, yvals = _volumetric_boundary_steps(
            data, x, y, threshold=1.0
        )
        assert xvals == [-0.5, 0.5, 0.5, 1, 1, 2.5]
        assert yvals == [2.5, 2.5, 2.5, 2.5, 0.5, 0.5]

    def test_no_data_sits_below_the_lowest_width(self):
        _, yvals = _volumetric_boundary_steps(
            {}, [0, 2], [1, 2], threshold=1.0
        )
        assert yvals == [-0.5, -0.5, -0.5, -0.5]


class TestSetVolumetricAxes:
    """_set_volumetric_axes is a plain function operating on a
    go.Figure -- no pyGSTi/emulator needed.
    """

    def test_full_figure_gets_index_positioned_ticks(self):
        fig = go.Figure()
        _set_volumetric_axes(fig, depths=[0, 4, 8], widths=[1, 2])

        assert fig.layout.xaxis.tickvals == (0, 1, 2)
        assert fig.layout.xaxis.ticktext == ('0', '4', '8')
        assert fig.layout.xaxis.range == (-0.5, 2.5)
        assert fig.layout.xaxis.title.text == 'Depth'

        assert fig.layout.yaxis.tickvals == (0, 1)
        assert fig.layout.yaxis.ticktext == ('1', '2')
        assert fig.layout.yaxis.range == (-0.5, 1.5)
        assert fig.layout.yaxis.title.text == 'Width'

    def test_subplot_targeting_only_touches_its_own_axes(self):
        fig = make_subplots(rows=1, cols=2)
        _set_volumetric_axes(
            fig, depths=[0, 4], widths=[1, 2, 3], row=1, col=2
        )

        assert fig.layout.xaxis2.tickvals == (0, 1)
        assert fig.layout.xaxis2.range == (-0.5, 1.5)
        assert fig.layout.yaxis2.tickvals == (0, 1, 2)
        assert fig.layout.yaxis2.range == (-0.5, 2.5)
        # Column 1's axes were never targeted.
        assert fig.layout.xaxis.tickvals is None


class TestMCBCapabilityRegions:
    """Integration test: MCB.run() through pyGSTi's own
    capability_regions() classification, using a noisier-than-default
    emulator so the circuits actually decay within a small depth
    range (the Emulator's default depolarizing rates are too gentle
    to see any "fail" classification within a fast test's depth
    budget).

    MirrorRBDesign/PeriodicMirrorCircuitDesign don't accept a seed
    from MCB, so circuit sampling isn't reproducible across separate
    runs -- confirmed empirically: the same config can classify a
    borderline (depth, width) shape as "success" in one run and
    "indeterminate" in another. So this asserts statistical
    invariants that must hold regardless of the random circuit draw,
    not one brittle exact classification grid.
    """

    @staticmethod
    @pytest.fixture(scope='class')
    def capability_regions():
        simulator = DensityMatrixSimulator(
            noise_model=DepolarizingNoise(
                single_qubit=0.01, two_qubit=0.08
            )
        )
        mcb = MCB(
            qpu=Emulator, config=Config(EXAMPLE_CONFIG_PATH),
            qubits=[0, 1, 2], circuit_depths=[0, 2, 4, 8, 16, 32],
            n_circuits=10, est_qubit_error_rate=0.02,
            simulator=simulator,
        )
        mcb.run()
        vbdf = pygsti.protocols.VBDataFrame(mcb.summary)
        vbdf1 = vbdf.select_column_value('CircuitType', 'RMC')
        creg = vbdf1.capability_regions(
            metric='polarization', threshold=1 / np.e,
            significance=0.05, monotonic=True,
        )
        return creg, sorted(vbdf1.x_values), sorted(vbdf1.y_values)

    def test_classifications_are_valid_values(self, capability_regions):
        creg, _, _ = capability_regions
        assert set(creg.values()) <= {0, 1, 2}

    def test_shallowest_depth_always_succeeds(self, capability_regions):
        creg, depths, widths = capability_regions
        assert all(creg[(depths[0], w)] == 2 for w in widths)

    def test_deepest_depth_always_fails(self, capability_regions):
        creg, depths, widths = capability_regions
        assert all(creg[(depths[-1], w)] == 0 for w in widths)

    def test_capability_is_monotonic_in_depth(self, capability_regions):
        # Classification can only get worse (or stay the same) as
        # depth increases, for a fixed width.
        creg, depths, widths = capability_regions
        for w in widths:
            values = [creg[(d, w)] for d in depths]
            assert all(a >= b for a, b in zip(values, values[1:], strict=False))

    def test_capability_is_monotonic_in_width(self, capability_regions):
        # Classification can only get worse (or stay the same) as
        # width increases, for a fixed depth.
        creg, depths, widths = capability_regions
        for d in depths:
            values = [creg[(d, w)] for w in widths]
            assert all(a >= b for a, b in zip(values, values[1:], strict=False))
