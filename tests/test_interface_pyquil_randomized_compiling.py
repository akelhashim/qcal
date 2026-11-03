"""Unit tests for qcal.interface.pyquil.randomized_compiling.

Most of this module is pure circuit bookkeeping (RC-layer detection,
base-cycle extraction, final-layer lookup, per-layer phase tracking)
that only touches pyquil at its very edges. Those edges --
`build_rc_configuration`'s `RandomizedCompilingConfiguration` and the
trackers' `emit_phase_gate` (`RZ`/`MemoryReference`) -- are exercised
here against small fake pyquil modules injected into `sys.modules`, so
the bookkeeping is tested whether or not pyquil is installed.
"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from qcal.circuit import Barrier, Circuit, Cycle
from qcal.compilation.decompositions import zxzxz_to_unitary
from qcal.gates.gate import Gate
from qcal.gates.single_qubit import MCM, X90, Meas, Rz, VirtualZ, x, y, z
from qcal.gates.two_qubit import CZ
from qcal.interface.pyquil.randomized_compiling import (
    _READOUT_PAULI_FLIPS,
    _READOUT_PAULI_PHASES,
    PHASE_GATES,
    PULSE_GATES,
    RCLayerTracker,
    ReadoutLayerTracker,
    _minimal_period,
    _U2Randomization,
    build_rc_configuration,
    cycle_to_base_cycle,
    effective_layer_period,
    final_layer_cycle_indices,
    has_phase_gate,
    has_trailing_single_qubit_layer,
    is_rc_layer,
)

_PYQUIL_MODULES = (
    'pyquil',
    'pyquil.gates',
    'pyquil.quil',
    'pyquil.quilatom',
    'pyquil.quilbase',
    'pyquil._qpu',
    'pyquil._qpu.randomized_compiling',
)


class _FakeRef:
    """Stand-in for a pyquil MemoryReference.

    `2 * np.pi * ref` evaluates `(2 * np.pi) * ref`, so `__rmul__`
    records the scale factor instead of building a pyquil expression.
    """

    def __init__(self, name, index):
        self.name = name
        self.index = index

    def __eq__(self, other):
        return (
            isinstance(other, _FakeRef)
            and (self.name, self.index) == (other.name, other.index)
        )

    def __rmul__(self, factor):
        return ('scaled', factor, self)


class _FakeRCConfiguration:
    """Records the kwargs `build_rc_configuration` passes through."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeRCVariables:

    def __init__(self, unitaries_prefix):
        self.unitaries_prefix = unitaries_prefix


@pytest.fixture
def fake_pyquil(monkeypatch):
    """Install minimal fake pyquil modules for the duration of a test.

    Returns:
        SimpleNamespace: the fake `RZ`/`MemoryReference` callables.
    """
    modules = {name: types.ModuleType(name) for name in _PYQUIL_MODULES}
    modules['pyquil.gates'].RZ = lambda angle, q: ('RZ', angle, q)
    modules['pyquil.quilbase'].MemoryReference = _FakeRef
    modules['pyquil.quilatom'].MemoryReference = _FakeRef
    rc = modules['pyquil._qpu.randomized_compiling']
    rc.RandomizedCompilingConfiguration = _FakeRCConfiguration
    rc.RandomizedCompilingVariables = _FakeRCVariables
    modules['pyquil._qpu'].randomized_compiling = rc
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(MemoryReference=_FakeRef)


@pytest.fixture
def no_pyquil(monkeypatch):
    """Make every pyquil import raise ImportError for a test."""
    for name in _PYQUIL_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


def _rc_variables():
    """Fake `RandomizedCompilingVariables` as seen by RCLayerTracker."""
    return SimpleNamespace(
        source_unitaries=lambda q: f'source_phases_q{q}',
        twirled_unitaries_ref=lambda q, layer, angle: _FakeRef(
            f'unitaries_q{q}', 3 * layer + angle
        ),
    )


def _rc_config(
    qubits=(0, 1),
    base_cycles=(((0, 1),),),
    reps=1,
    skip_final_layer=False,
):
    """Fake `RandomizedCompilingConfiguration` for RCLayerTracker."""
    return SimpleNamespace(
        qubits_sorted=list(qubits),
        base_cycles=base_cycles,
        base_cycle_repetitions=reps,
        skip_final_layer=skip_final_layer,
        variables=_rc_variables(),
    )


def _readout_config(qubits=(0, 1)):
    return _U2Randomization(
        qubits=list(qubits),
        destination_names=lambda q: f'dest_q{q}',
        seed_names=lambda q: f'seed_q{q}',
        source_names=lambda q: f'src_q{q}',
        source_unitary_phases=list(_READOUT_PAULI_PHASES),
    )


def _template(q=0):
    """A ZXZXZ template on qubit `q`, as 5 separate Cycles."""
    return [
        Cycle({Rz(q, 0.1)}),
        Cycle({X90(q)}),
        Cycle({Rz(q, 0.2)}),
        Cycle({X90(q)}),
        Cycle({Rz(q, 0.3)}),
    ]


def _single_qubit_circuit(n_cycles, q=0):
    """`n_cycles` X90 Cycles followed by a terminal measurement."""
    return Circuit(
        [Cycle({X90(q)}) for _ in range(n_cycles)] + [Cycle({Meas(q)})]
    )


def _equal_up_to_phase(a, b):
    return np.isclose(abs(np.trace(a.conj().T @ b)), a.shape[0])


class TestConstants:

    def test_phase_and_pulse_gates(self):
        assert PHASE_GATES == ('Rz', 'VirtualZ')
        assert PULSE_GATES == ('X90', 'SX')

    def test_readout_phases_are_four_zxzxz_triples(self):
        assert len(_READOUT_PAULI_PHASES) == 12

    @pytest.mark.parametrize('index, pauli', [
        (0, np.eye(2)), (1, x), (2, y), (3, z),
    ])
    def test_readout_phases_are_i_x_y_z(self, index, pauli):
        turns = _READOUT_PAULI_PHASES[3 * index:3 * index + 3]
        unitary = zxzxz_to_unitary(2 * np.pi * np.asarray(turns))
        assert _equal_up_to_phase(unitary, pauli)

    def test_readout_flips_match_anticommutation_with_z(self):
        # A Pauli flips a Z-basis outcome iff it anticommutes with Z.
        expected = [
            not np.allclose(p @ z, z @ p) for p in (np.eye(2), x, y, z)
        ]
        assert _READOUT_PAULI_FLIPS.tolist() == expected
        assert _READOUT_PAULI_FLIPS.tolist() == [False, True, True, False]


class TestHasPhaseGate:

    def test_rz_is_phase_gate(self):
        assert has_phase_gate(Cycle({Rz(0, 0.5)}))

    def test_virtualz_is_phase_gate(self):
        assert has_phase_gate(Cycle({VirtualZ(0, 0.5)}))

    def test_mixed_cycle_with_one_phase_gate(self):
        assert has_phase_gate(Cycle({X90(0), Rz(1, 0.5)}))

    def test_pulse_two_qubit_and_measurement_cycles(self):
        assert not has_phase_gate(Cycle({X90(0), X90(1)}))
        assert not has_phase_gate(Cycle({CZ(0, 1)}))
        assert not has_phase_gate(Cycle({Meas(0)}))


class TestIsRcLayer:

    def test_two_qubit_gate_is_boundary(self):
        assert is_rc_layer(Cycle({CZ(0, 1)}))

    def test_measurement_is_boundary(self):
        assert is_rc_layer(Cycle({Meas(0)}))

    def test_multi_qubit_mcm_is_boundary(self):
        assert is_rc_layer(Cycle({MCM((0, 1))}))

    def test_single_qubit_cycle_is_not_boundary_by_default(self):
        assert not is_rc_layer(Cycle({X90(0), Rz(1, 0.1)}))

    @pytest.mark.parametrize('index, expected', [
        (0, False), (3, False), (4, True), (5, False), (8, False),
        # The last interior cycle is never a positional boundary.
        (9, False),
    ])
    def test_positional_boundary_every_period(self, index, expected):
        cycle = Cycle({X90(0)})
        assert is_rc_layer(
            cycle, index=index, layer_period=5, n_cycles=10
        ) is expected

    def test_positional_boundary_at_last_index_when_more_follow(self):
        assert is_rc_layer(
            Cycle({X90(0)}), index=9, layer_period=5, n_cycles=15
        )

    @pytest.mark.parametrize('kwargs', [
        {'index': 4, 'layer_period': 5},
        {'index': 4, 'n_cycles': 10},
        {'layer_period': 5, 'n_cycles': 10},
    ])
    def test_positional_fallback_needs_all_three_args(self, kwargs):
        assert not is_rc_layer(Cycle({X90(0)}), **kwargs)


class TestEffectiveLayerPeriod:

    def test_none_stays_none(self):
        assert effective_layer_period(_single_qubit_circuit(4), None) is None

    def test_kept_for_single_qubit_only_interior(self):
        assert effective_layer_period(_single_qubit_circuit(10), 5) == 5

    def test_dropped_when_interior_has_two_qubit_gate(self):
        circuit = Circuit([
            Cycle({X90(0)}), Cycle({CZ(0, 1)}), Cycle({X90(0)}),
            Cycle({Meas(0), Meas(1)}),
        ])
        assert effective_layer_period(circuit, 5) is None

    def test_dropped_when_interior_has_mid_circuit_measurement(self):
        circuit = Circuit([
            Cycle({X90(0)}), Cycle({MCM(0)}), Cycle({X90(0)}),
            Cycle({Meas(0)}),
        ])
        assert effective_layer_period(circuit, 5) is None

    def test_terminal_measurement_does_not_count(self):
        # The only measurement is the terminal one, which is excluded.
        circuit = Circuit([Cycle({X90(0)}), Cycle({Meas(0)})])
        assert effective_layer_period(circuit, 3) == 3


class TestCycleToBaseCycle:

    def test_edge_is_sorted_and_idle_qubits_are_bare_ints(self):
        assert cycle_to_base_cycle(
            Cycle({CZ(1, 0)}), [0, 1, 2]
        ) == ((0, 1), 2)

    def test_two_edges_then_idle_qubits_in_register_order(self):
        result = cycle_to_base_cycle(
            Cycle({CZ(0, 1), CZ(3, 2)}), [4, 0, 1, 2, 3]
        )
        assert set(result[:2]) == {(0, 1), (2, 3)}
        assert result[2:] == (4,)

    def test_single_qubit_gates_are_treated_as_idle(self):
        assert cycle_to_base_cycle(
            Cycle({CZ(0, 1), X90(2)}), [0, 1, 2]
        ) == ((0, 1), 2)

    def test_mcm_layer_is_all_bare_ints(self):
        assert cycle_to_base_cycle(
            Cycle({MCM((0, 1))}), [0, 1, 2]
        ) == (0, 1, 2)

    def test_mcm_and_edge_layer(self):
        assert cycle_to_base_cycle(
            Cycle({MCM((0, 1)), CZ(2, 3)}), [0, 1, 2, 3]
        ) == ((2, 3), 0, 1)

    def test_three_qubit_gate_raises(self):
        ccz = Gate(np.diag([1] * 7 + [-1]), (0, 1, 2))
        with pytest.raises(ValueError, match='more than two qubits'):
            cycle_to_base_cycle(Cycle({ccz}), [0, 1, 2])


class TestMinimalPeriod:

    @pytest.mark.parametrize('layers, expected', [
        ((), 0),
        (('a',), 1),
        (('a', 'a', 'a', 'a'), 1),
        (('a', 'b', 'a', 'b'), 2),
        (('a', 'b', 'a', 'b', 'a', 'b'), 2),
        (('a', 'b', 'c', 'a', 'b', 'c'), 3),
        (('a', 'b', 'a'), 3),
        (('a', 'b', 'a', 'c'), 4),
    ])
    def test_minimal_period(self, layers, expected):
        assert _minimal_period(layers) == expected


class TestHasTrailingSingleQubitLayer:

    def test_template_after_last_boundary(self):
        circuit = Circuit([
            Cycle({X90(0)}), Cycle({CZ(0, 1)}), Cycle({X90(0)}),
            Cycle({Meas(0), Meas(1)}),
        ])
        assert has_trailing_single_qubit_layer(circuit)

    def test_boundary_is_last_interior_cycle(self):
        circuit = Circuit([
            Cycle({X90(0)}), Cycle({CZ(0, 1)}),
            Cycle({Meas(0), Meas(1)}),
        ])
        assert not has_trailing_single_qubit_layer(circuit)

    def test_barrier_after_last_boundary_is_not_content(self):
        circuit = Circuit([
            Cycle({X90(0)}), Cycle({CZ(0, 1)}), Barrier((0, 1)),
            Cycle({Meas(0), Meas(1)}),
        ])
        assert not has_trailing_single_qubit_layer(circuit)

    def test_no_boundaries_is_one_trailing_layer(self):
        assert has_trailing_single_qubit_layer(_single_qubit_circuit(3))

    def test_layer_period_never_closes_last_template(self):
        # Boundaries at non-barrier index 1 only (index 3 is the last).
        assert has_trailing_single_qubit_layer(
            _single_qubit_circuit(4), layer_period=2
        )


class TestFinalLayerCycleIndices:

    def test_template_after_two_qubit_gate(self):
        circuit = Circuit(
            _template() + [Cycle({CZ(0, 1)})] + _template()
            + [Cycle({Meas(0), Meas(1)})]
        )
        assert final_layer_cycle_indices(circuit) == frozenset(
            {6, 7, 8, 9, 10}
        )

    def test_barriers_are_skipped(self):
        circuit = Circuit([
            Cycle({CZ(0, 1)}),
            Cycle({Rz(0, 0.1)}),
            Barrier((0, 1)),
            Cycle({X90(0)}),
            Barrier((0, 1)),
            Cycle({Meas(0), Meas(1)}),
        ])
        assert final_layer_cycle_indices(circuit) == frozenset({1, 3})

    def test_no_single_qubit_content_before_measurement(self):
        circuit = Circuit([
            Cycle({X90(0)}), Cycle({CZ(0, 1)}),
            Cycle({Meas(0), Meas(1)}),
        ])
        assert final_layer_cycle_indices(circuit) == frozenset()

    def test_mcm_is_a_boundary(self):
        circuit = Circuit([
            Cycle({X90(0)}), Cycle({MCM(1)}), Cycle({X90(0)}),
            Cycle({Rz(0, 0.1)}), Cycle({Meas(0), Meas(1)}),
        ])
        assert final_layer_cycle_indices(circuit) == frozenset({2, 3})

    def test_without_boundaries_whole_interior_is_final(self):
        assert final_layer_cycle_indices(
            _single_qubit_circuit(4)
        ) == frozenset({0, 1, 2, 3})

    def test_layer_period_scopes_to_last_template(self):
        circuit = _single_qubit_circuit(10)
        assert final_layer_cycle_indices(
            circuit, layer_period=5
        ) == frozenset({5, 6, 7, 8, 9})

    def test_layer_period_numbering_ignores_barriers(self):
        # Non-barrier indices 0..3 sit at circuit indices 1, 2, 4, 5;
        # the only positional boundary is non-barrier index 1.
        circuit = Circuit([
            Barrier((0,)),
            Cycle({X90(0)}),
            Cycle({X90(0)}),
            Barrier((0,)),
            Cycle({X90(0)}),
            Cycle({X90(0)}),
            Cycle({Meas(0)}),
        ])
        assert final_layer_cycle_indices(
            circuit, layer_period=2
        ) == frozenset({4, 5})


class TestBuildRcConfiguration:

    def test_returns_none_without_pyquil(self, no_pyquil):
        circuit = Circuit([Cycle({CZ(0, 1)}), Cycle({Meas(0)})])
        assert build_rc_configuration(circuit, [0, 1]) is None

    def test_repeated_layers_collapse_to_one_base_cycle(self, fake_pyquil):
        circuit = Circuit([
            Cycle({X90(0)}), Cycle({CZ(0, 1)}),
            Cycle({X90(0)}), Cycle({CZ(0, 1)}),
            Cycle({X90(0)}), Cycle({Meas(0), Meas(1)}),
        ])
        config = build_rc_configuration(circuit, [0, 1])

        assert config.base_cycles == (((0, 1),),)
        assert config.base_cycle_repetitions == 2
        assert config.variables.unitaries_prefix == 'source_phases'

    def test_explicit_single_repetition_keeps_all_layers(
        self, fake_pyquil
    ):
        circuit = Circuit([
            Cycle({CZ(0, 1)}), Cycle({CZ(0, 1)}), Cycle({Meas(0)}),
        ])
        config = build_rc_configuration(
            circuit, [0, 1], base_cycle_repetitions=1
        )

        assert config.base_cycles == (((0, 1),), ((0, 1),))
        assert config.base_cycle_repetitions == 1

    def test_repetition_code_with_mcm(self, fake_pyquil):
        unit = [
            Cycle({CZ(0, 1)}), Cycle({CZ(1, 2)}), Cycle({MCM(1)}),
        ]
        circuit = Circuit(
            unit + unit + [Cycle({Meas(0), Meas(1), Meas(2)})]
        )
        config = build_rc_configuration(circuit, [0, 1, 2])

        assert config.base_cycles == (((0, 1), 2), ((1, 2), 0), (0, 1, 2))
        assert config.base_cycle_repetitions == 2

    def test_layer_period_on_single_qubit_circuit(self, fake_pyquil):
        # Boundaries at non-barrier indices 1 and 3 (not 5, the last).
        config = build_rc_configuration(
            _single_qubit_circuit(6), [0], layer_period=2
        )

        assert config.base_cycles == ((0,),)
        assert config.base_cycle_repetitions == 2

    def test_layer_period_ignored_with_two_qubit_content(
        self, fake_pyquil
    ):
        circuit = Circuit(
            [Cycle({X90(0)}) for _ in range(4)]
            + [Cycle({CZ(0, 1)})]
            + [Cycle({X90(0)}) for _ in range(4)]
            + [Cycle({Meas(0), Meas(1)})]
        )
        config = build_rc_configuration(circuit, [0, 1], layer_period=2)

        assert config.base_cycles == (((0, 1),),)
        assert config.base_cycle_repetitions == 1

    def test_rc_kwargs_are_forwarded(self, fake_pyquil):
        circuit = Circuit([Cycle({CZ(0, 1)}), Cycle({Meas(0)})])
        config = build_rc_configuration(
            circuit, [0, 1],
            invert_random_paulis=True, shots_per_randomization=4,
        )

        assert config.kwargs['invert_random_paulis'] is True
        assert config.kwargs['shots_per_randomization'] == 4

    def test_no_layers_raises(self, fake_pyquil):
        with pytest.raises(ValueError, match='nothing to twirl'):
            build_rc_configuration(_single_qubit_circuit(4), [0])

    def test_zero_repetitions_raises(self, fake_pyquil):
        circuit = Circuit([Cycle({CZ(0, 1)}), Cycle({Meas(0)})])
        with pytest.raises(ValueError, match='>= 1'):
            build_rc_configuration(
                circuit, [0, 1], base_cycle_repetitions=0
            )

    def test_non_dividing_repetitions_raises(self, fake_pyquil):
        circuit = Circuit([
            Cycle({CZ(0, 1)}), Cycle({CZ(0, 1)}), Cycle({Meas(0)}),
        ])
        with pytest.raises(ValueError, match='divide evenly'):
            build_rc_configuration(
                circuit, [0, 1], base_cycle_repetitions=3
            )

    def test_non_repeating_structure_raises(self, fake_pyquil):
        circuit = Circuit([
            Cycle({CZ(0, 1)}), Cycle({CZ(1, 2)}),
            Cycle({Meas(0), Meas(1), Meas(2)}),
        ])
        with pytest.raises(ValueError, match='not 2 repetitions'):
            build_rc_configuration(
                circuit, [0, 1, 2], base_cycle_repetitions=2
            )


class TestBuildRcConfigurationRealPyquil:

    def test_base_cycles(self):
        pytest.importorskip('pyquil._qpu.randomized_compiling')
        circuit = Circuit([
            Cycle({X90(0)}), Cycle({CZ(0, 1)}),
            Cycle({X90(0)}), Cycle({CZ(0, 1)}),
            Cycle({X90(0)}), Cycle({Meas(0), Meas(1)}),
        ])
        config = build_rc_configuration(circuit, [0, 1])

        assert tuple(config.base_cycles) == (((0, 1),),)
        assert config.base_cycle_repetitions == 2


class TestRCLayerTracker:

    def test_initial_state(self):
        tracker = RCLayerTracker(_rc_config(qubits=(2, 5)))

        assert tracker.layer_index == 0
        assert tracker.angle_index == {2: 0, 5: 0}
        assert set(tracker.source_phases) == {
            'source_phases_q2', 'source_phases_q5'
        }

    def test_register_reserves_final_layer(self):
        config = _rc_config(base_cycles=('a', 'b', 'c'), reps=2)
        tracker = RCLayerTracker(config)

        # (2 * 3 + 1) layers x 3 angles.
        assert tracker.source_phases['source_phases_q0'] == [0.0] * 21

    def test_register_without_reserved_final_layer(self):
        config = _rc_config(base_cycles=('a', 'b', 'c'), reps=2)
        tracker = RCLayerTracker(config, reserve_final_layer=False)

        assert len(tracker.source_phases['source_phases_q0']) == 18

    def test_skip_final_layer_drops_reservation(self):
        config = _rc_config(
            base_cycles=('a', 'b', 'c'), reps=2, skip_final_layer=True
        )
        tracker = RCLayerTracker(config, reserve_final_layer=True)

        assert len(tracker.source_phases['source_phases_q0']) == 18

    def test_close_layer_advances_and_resets_angles(self):
        tracker = RCLayerTracker(_rc_config())
        tracker.angle_index[0] = 2

        tracker.close_layer()

        assert tracker.layer_index == 1
        assert tracker.angle_index == {0: 0, 1: 0}

    def test_emit_records_phase_in_turns(self, fake_pyquil):
        tracker = RCLayerTracker(_rc_config())

        instruction = tracker.emit_phase_gate(Rz(0, np.pi))

        assert tracker.source_phases['source_phases_q0'][:3] == [
            0.5, 0.0, 0.0
        ]
        assert tracker.angle_index == {0: 1, 1: 0}
        assert instruction == (
            'RZ', ('scaled', 2 * np.pi, _FakeRef('unitaries_q0', 0)), 0
        )

    def test_emit_fills_slots_by_layer_and_angle(self, fake_pyquil):
        config = _rc_config(base_cycles=('a',), reps=1)
        tracker = RCLayerTracker(config)

        tracker.emit_phase_gate(Rz(1, np.pi / 2))
        tracker.emit_phase_gate(VirtualZ(1, -np.pi))
        tracker.close_layer()
        instruction = tracker.emit_phase_gate(Rz(1, -np.pi / 2))

        assert tracker.source_phases['source_phases_q1'] == [
            0.25, -0.5, 0.0, -0.25, 0.0, 0.0
        ]
        assert tracker.source_phases['source_phases_q0'] == [0.0] * 6
        assert instruction[2] == 1
        assert instruction[1][2] == _FakeRef('unitaries_q1', 3)

    def test_emit_with_readout_tracker_writes_readout_register(
        self, fake_pyquil
    ):
        tracker = RCLayerTracker(_rc_config())
        readout = ReadoutLayerTracker(_readout_config())

        tracker.emit_phase_gate(Rz(0, np.pi / 2))
        instruction = tracker.emit_phase_gate(
            Rz(0, np.pi), readout_tracker=readout
        )

        assert readout.source_phases[0] == [0.0, 0.5, 0.0]
        # Only the first (non-readout) phase reached the RC register.
        assert tracker.source_phases['source_phases_q0'][:3] == [
            0.25, 0.0, 0.0
        ]
        assert tracker.angle_index[0] == 2
        assert instruction == (
            'RZ', ('scaled', 2 * np.pi, _FakeRef('dest_q0', 1)), 0
        )


class TestReadoutLayerTracker:

    def test_initial_state(self):
        tracker = ReadoutLayerTracker(_readout_config(qubits=(3, 4)))

        assert tracker.angle_index == {3: 0, 4: 0}
        assert tracker.source_phases == {3: [0.0] * 3, 4: [0.0] * 3}

    def test_emit_records_phases_and_references_destination(
        self, fake_pyquil
    ):
        tracker = ReadoutLayerTracker(_readout_config())

        first = tracker.emit_phase_gate(Rz(1, np.pi / 2))
        second = tracker.emit_phase_gate(VirtualZ(1, -np.pi))

        assert tracker.source_phases[1] == [0.25, -0.5, 0.0]
        assert tracker.source_phases[0] == [0.0, 0.0, 0.0]
        assert tracker.angle_index == {0: 0, 1: 2}
        assert first == (
            'RZ', ('scaled', 2 * np.pi, _FakeRef('dest_q1', 0)), 1
        )
        assert second[1][2] == _FakeRef('dest_q1', 1)


class TestU2Randomization:

    def test_source_unitary_count(self):
        assert _readout_config().source_unitary_count == 4

    def test_generate_seeds_is_reproducible(self):
        config = _readout_config(qubits=(0, 1, 2))

        seeds = config.generate_seeds(np.random.default_rng(1234))
        again = config.generate_seeds(np.random.default_rng(1234))

        assert seeds == again
        assert list(seeds) == [0, 1, 2]
        assert all(type(s) is int for s in seeds.values())
        assert all(0 <= s < 2**47 - 1 for s in seeds.values())

    def test_generate_seeds_differs_per_qubit(self):
        seeds = _readout_config(qubits=(0, 1, 2)).generate_seeds(
            np.random.default_rng(0)
        )
        assert len(set(seeds.values())) == 3

    def test_build_quil_program_without_pyquil(self, no_pyquil):
        assert _readout_config().build_quil_program() is None
