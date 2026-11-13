"""Unit tests for qcal.interface.pyquil, exercising its behavior when
the optional `pyquil` dependency is NOT installed.

Like trueq/cirq_superstaq/qiskit_superstaq, `pyquil` is never declared
as a dependency anywhere in pyproject.toml -- every function/class
here lazily imports it inside a `try` block. Pyquil's own
code already degrades cleanly on a missing import:

- transpiler.py: every `try: import pyquil.* / except ImportError:
  logger.warning(...)` is immediately followed by `return`, so
  generator functions (the add_* gate builders) yield nothing and
  plain functions return None.
- utils.py: every `try: from pyquil.* / except ImportError as exc:`
  re-raises as `raise ImportError(...) from exc`, giving a clear,
  immediate signal instead of silently returning anything.

This file is a regression suite protecting both of those conventions,
not a bug report -- unlike test_interface_trueq.py/
test_interface_superstaq.py, which documented (and this session then
fixed) real crashes in those modules.

It only makes sense in an environment without pyquil installed.
"""
import sys
import types

import numpy as np
import pytest

import qcal.interface.pyquil.transpiler as transpiler_module
from qcal.circuit import Circuit, Cycle
from qcal.gates.single_qubit import (
    MCM,
    X90,
    Y90,
    H,
    Meas,
    Rz,
    SXdag,
    SYdag,
    X,
    Y,
)
from qcal.gates.two_qubit import CZ
from qcal.interface.pyquil.transpiler import (
    DEFAULT_GATEMAPPER,
    RX_F12,
    RZ_F12,
    PyQuilTranspiler,
    add_CNOT,
    add_CX,
    add_CZ,
    add_H,
    add_Idle,
    add_ISWAP,
    add_MCM,
    add_Measure,
    add_Rz,
    add_SXdag,
    add_SYdag,
    add_X,
    add_X90,
    add_Y,
    add_Y90,
    to_pyquil,
    transpile_circuit,
    transpile_cycle,
)
from qcal.interface.pyquil.utils import (
    _classify,
    _get_template_object,
    _is_identity,
    _make_invocation,
    _qubit_index,
    _set_template_object,
    add_active_reset,
    add_dd_sequence_during_operation,
    add_delay_after_measurements,
    get_defcal_param,
    prepend_delay_to_program,
    prepend_esp_to_measure,
    prepend_measure_to_program,
    replace_parallel_cycles,
    set_defcal_param,
    set_waveform,
)

try:
    import pyquil  # noqa: F401
    _HAS_PYQUIL = True
except ImportError:
    _HAS_PYQUIL = False

pytestmark = pytest.mark.skipif(
    _HAS_PYQUIL,
    reason=(
        "pyquil is installed; this file only covers the "
        "pyquil-not-installed code path"
    ),
)


class TestModuleImport:

    def test_transpiler_imports_successfully(self):
        import qcal.interface.pyquil.transpiler  # noqa: F401

    def test_utils_imports_successfully(self):
        import qcal.interface.pyquil.utils  # noqa: F401


class TestTranspilerGeneratorsYieldNothing:

    @pytest.mark.parametrize('fn, args, kwargs', [
        (add_CNOT, (0, 1), {}),
        (add_CX, (0, 1), {}),
        (add_CZ, (0, 1), {}),
        (add_H, (0,), {'subspace': 'GE'}),
        (add_Idle, (0, 1e-7), {}),
        (add_ISWAP, (0, 1), {}),
        (add_Measure, (0,), {}),
        (add_SXdag, (0,), {'subspace': 'GE'}),
        (add_SYdag, (0,), {'subspace': 'GE'}),
        (add_X, (0,), {'subspace': 'GE'}),
        (add_X90, (0,), {'subspace': 'GE'}),
        (add_Y, (0,), {'subspace': 'GE'}),
        (add_Y90, (0,), {'subspace': 'GE'}),
        (add_Rz, (0, 0.5), {'subspace': 'GE'}),
    ])
    def test_yields_nothing(self, fn, args, kwargs):
        assert list(fn(*args, **kwargs)) == []


class TestTranspilerFunctionsReturnNone:

    def test_rx_f12(self):
        assert RX_F12(0.5, 0) is None

    def test_rz_f12(self):
        assert RZ_F12(0.5, 0) is None

    def test_to_pyquil(self):
        assert to_pyquil(None, None) is None

    def test_transpile_circuit(self):
        assert transpile_circuit(None, None) is None

    def test_transpile_cycle(self):
        assert transpile_cycle(None, None, None) is None


class TestDefaultGateMapper:

    @pytest.mark.parametrize('name, fn', [
        ('H', add_H), ('SX', add_X90), ('SXdag', add_SXdag),
        ('SY', add_Y90), ('SYdag', add_SYdag),
    ])
    def test_sqrt_paulis_route_to_pulse_gates(self, name, fn):
        assert DEFAULT_GATEMAPPER[name] is fn


class TestPyQuilTranspilerConstruction:

    def test_does_not_crash(self):
        PyQuilTranspiler()


class TestUtilsFunctionsRaiseCleanImportError:
    """Every function here hits its `from pyquil... / except
    ImportError as exc: raise ImportError(...) from exc` before
    touching any of its arguments, so dummy/None args are safe.
    """

    @pytest.mark.parametrize('fn, args, kwargs', [
        (_classify, (None, frozenset()), {}),
        (_get_template_object, (None,), {}),
        (_is_identity, (None,), {}),
        (_make_invocation, ('X', [0]), {}),
        (_set_template_object, (None, None), {}),
        (_qubit_index, (0,), {}),
        (add_active_reset, (None,), {}),
        (
            add_dd_sequence_during_operation,
            (None, None, [0], 1.0, 1.0), {}
        ),
        (add_delay_after_measurements, (None, 1.0), {}),
        (get_defcal_param, (None, 'phase'), {}),
        (prepend_delay_to_program, (None, 1.0), {}),
        (prepend_esp_to_measure, (None,), {}),
        (prepend_measure_to_program, (None,), {}),
        (replace_parallel_cycles, (None,), {}),
        (set_defcal_param, (None, 'phase', 1.0), {}),
        (set_waveform, ('name', [1, 2, 3]), {}),
    ])
    def test_raises_import_error(self, fn, args, kwargs):
        with pytest.raises(ImportError):
            fn(*args, **kwargs)


# ---------------------------------------------------------------------
# Randomized-compiling/readout transpiler behavior, exercised against
# minimal fake pyquil modules injected into `sys.modules`. These only
# model the handful of names `transpiler.py` imports, so the logic of
# `add_MCM`, `transpile_cycle` and `transpile_circuit` can be checked
# without a pyquil installation.
# ---------------------------------------------------------------------

class _Inst:
    """Fake pyquil instruction: a name plus its positional args."""

    def __init__(self, name, *args):
        self.name = name
        self.args = args

    def __eq__(self, other):
        return (
            isinstance(other, _Inst)
            and (self.name, self.args) == (other.name, other.args)
        )

    def __repr__(self):
        return f'_Inst({self.name!r}, *{self.args!r})'


class _Ref:
    """Fake MemoryReference; `2 * np.pi * ref` records the factor."""

    def __init__(self, name, index=0):
        self.name = name
        self.index = index

    def __eq__(self, other):
        return (
            isinstance(other, _Ref)
            and (self.name, self.index) == (other.name, other.index)
        )

    def __rmul__(self, factor):
        return ('scaled', factor, self)


class _Program(list):
    """Fake pyquil Program: a list of `_Inst` supporting `+`/`+=`."""

    def __init__(self, *items):
        super().__init__()
        for item in items:
            self += item

    def __iadd__(self, other):
        if isinstance(other, _Inst):
            self.append(other)
        elif isinstance(other, str):
            # Raw Quil text, e.g. a DEFCIRCUIT invocation.
            self.append(_Inst('QUIL', other))
        else:
            self.extend(list(other))
        return self

    def __add__(self, other):
        program = _Program()
        program += self
        program += other
        return program

    def declare(self, name, memory_type, length):
        self.append(_Inst('DECLARE', name, memory_type, length))
        return _Ref(name)

    @property
    def instructions(self):
        return list(self)


class _CallArgument:

    @staticmethod
    def from_identifier(name):
        return name

    @staticmethod
    def from_immediate(value):
        return value


_FAKE_MODULES = (
    'pyquil',
    'pyquil.gates',
    'pyquil.quil',
    'pyquil.quilatom',
    'pyquil.quilbase',
    'pyquil._qpu',
    'pyquil._qpu._classical_computations',
    'quil',
    'quil.instructions',
)


@pytest.fixture
def fake_pyquil(monkeypatch):
    """Install minimal fake pyquil/quil modules for one test."""
    modules = {name: types.ModuleType(name) for name in _FAKE_MODULES}
    gates = modules['pyquil.gates']
    gates.FENCE = lambda *qubits: _Inst('FENCE', *qubits)
    gates.MEASURE = lambda qubit, ref: _Inst('MEASURE', qubit, ref)
    gates.RZ = lambda angle, qubit: _Inst('RZ', angle, qubit)
    modules['pyquil'].Program = _Program
    modules['pyquil.quil'].Program = _Program
    modules['pyquil.quilatom'].FormalArgument = lambda s: f'%{s}'
    modules['pyquil.quilatom'].MemoryReference = _Ref
    quilbase = modules['pyquil.quilbase']
    quilbase.Declare = lambda *args: _Inst('DECLARE', *args)
    quilbase.Call = lambda name, args: _Inst('CALL', name, *args)
    quilbase.DefCircuit = lambda name, **kwargs: _Inst('DEFCIRCUIT', name)
    modules['pyquil._qpu._classical_computations'] \
        .build_extern_function_signatures = lambda: {
            'choose_random_real_sub_regions': _Inst('PRAGMA', 'EXTERN')
        }
    modules['quil'].instructions = modules['quil.instructions']
    modules['quil.instructions'].CallArgument = _CallArgument
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _gate_mapper():
    """Gate mapper whose handlers return recognizable fake insts."""
    return {
        'X': lambda q, **kw: _Inst('X', q),
        'X90': lambda q, **kw: _Inst('X90', q),
        'Y90': lambda q, **kw: _Inst('Y90', q),
        'Rz': lambda q, **kw: _Inst('RZ', kw['phase'], q),
        'CZ': lambda q0, q1, **kw: _Inst('CZ', q0, q1),
        'Meas': lambda q, ref: _Inst('MEASURE', q, ref),
        'MCM': add_MCM,
    }


class _StubTracker:
    """Records every `emit_phase_gate` call it receives."""

    def __init__(self):
        self.calls = []

    def emit_phase_gate(self, gate, *args):
        self.calls.append((gate, args))
        return _Inst('TRACKED', gate.qubits[0])


class TestSingleQubitGateUnitaries:
    """The emitted RZ/RX sequences (in time order) implement the intended
    unitaries, up to a global phase, with RZ(t) = exp(-i t Z / 2).
    """

    @pytest.fixture
    def fake_rx(self, fake_pyquil, monkeypatch):
        sys.modules['pyquil.gates'].RX = (
            lambda angle, qubit: _Inst('RX', angle, qubit)
        )
        # The EF rotations act on the {e, f} subspace exactly as RX/RZ
        # act on {g, e}, so record them under the same names.
        monkeypatch.setattr(
            transpiler_module, 'RX_F12',
            lambda angle, qubit: _Inst('RX', angle, qubit),
        )
        monkeypatch.setattr(
            transpiler_module, 'RZ_F12',
            lambda angle, qubit: _Inst('RZ', angle, qubit),
        )

    @staticmethod
    def _unitary(insts):
        from qcal.gates.single_qubit import rx, rz
        rotation = {'RX': rx, 'RZ': rz}
        unitary = np.eye(2)
        for inst in insts:
            unitary = rotation[inst.name](inst.args[0]) @ unitary
        return unitary

    @pytest.mark.parametrize('fn, gate_cls', [
        (add_H, H),
        (add_SXdag, SXdag),
        (add_SYdag, SYdag),
        (add_X, X),
        (add_X90, X90),
        (add_Y, Y),
        (add_Y90, Y90),
    ])
    @pytest.mark.parametrize('subspace', ['GE', 'EF'])
    def test_matches_gate_matrix(self, fake_rx, fn, gate_cls, subspace):
        unitary = self._unitary(fn(0, subspace=subspace))
        target = gate_cls(0).matrix
        overlap = np.trace(target.conj().T @ unitary) / 2
        assert np.isclose(abs(overlap), 1)


class TestAddMCM:

    def test_yields_nothing_without_pyquil(self):
        assert list(add_MCM((0, 1), [None, None])) == []

    def test_default_gatemapper_routes_mcm_to_add_mcm(self):
        assert DEFAULT_GATEMAPPER['MCM'] is add_MCM
        assert DEFAULT_GATEMAPPER['Meas'] is add_Measure

    def test_one_measure_per_qubit_in_order(self, fake_pyquil):
        assert list(add_MCM((3, 1), ['r3', 'r1'])) == [
            _Inst('MEASURE', 3, 'r3'),
            _Inst('MEASURE', 1, 'r1'),
        ]

    def test_mismatched_lengths_raise(self, fake_pyquil):
        with pytest.raises(ValueError):
            list(add_MCM((0, 1), ['r0']))


class TestTranspileCycle:

    def test_meas_uses_qubit_cref(self, fake_pyquil):
        program = transpile_cycle(
            Cycle({Meas(2)}), _gate_mapper(), {2: 'ro2'}
        )
        assert program == [_Inst('MEASURE', 2, 'ro2')]

    def test_multi_qubit_mcm_measures_every_qubit(self, fake_pyquil):
        program = transpile_cycle(
            Cycle({MCM((0, 1))}), _gate_mapper(), {0: 'ro0', 1: 'ro1'}
        )
        assert program == [
            _Inst('MEASURE', 0, 'ro0'),
            _Inst('MEASURE', 1, 'ro1'),
        ]

    def test_pulse_without_tracker_has_no_fence(self, fake_pyquil):
        program = transpile_cycle(Cycle({X90(0)}), _gate_mapper(), {})
        assert program == [_Inst('X90', 0)]

    def test_pulse_with_tracker_is_fenced_per_qubit(self, fake_pyquil):
        program = transpile_cycle(
            Cycle({X90(0)}), _gate_mapper(), {},
            rc_tracker=_StubTracker(),
        )
        assert program == [_Inst('X90', 0), _Inst('FENCE', 0)]

    def test_phase_gate_goes_to_rc_tracker(self, fake_pyquil):
        rc_tracker = _StubTracker()
        readout_tracker = _StubTracker()
        gate = Rz(1, 0.5)

        program = transpile_cycle(
            Cycle({gate}), _gate_mapper(), {},
            rc_tracker=rc_tracker, readout_tracker=readout_tracker,
        )

        assert program == [_Inst('TRACKED', 1)]
        assert rc_tracker.calls == [(gate, (readout_tracker,))]
        assert readout_tracker.calls == []

    def test_phase_gate_goes_to_readout_tracker_without_rc(
        self, fake_pyquil
    ):
        readout_tracker = _StubTracker()
        gate = Rz(1, 0.5)

        program = transpile_cycle(
            Cycle({gate}), _gate_mapper(), {},
            readout_tracker=readout_tracker,
        )

        assert program == [_Inst('TRACKED', 1)]
        assert readout_tracker.calls == [(gate, ())]

    def test_untracked_phase_gate_is_literal(self, fake_pyquil):
        program = transpile_cycle(
            Cycle({Rz(0, 0.5)}), _gate_mapper(), {}
        )
        assert program == [_Inst('RZ', 0.5, 0)]

    def test_defcircuit_body_uses_formal_arguments(self, fake_pyquil):
        program = transpile_cycle(
            Cycle({CZ(0, 1)}), _gate_mapper(), {},
            cycles_to_defcircuits=True,
        )
        assert program == [_Inst('CZ', '%q0', '%q1')]

    def test_defcircuit_body_with_tracked_phase_gate_raises(
        self, fake_pyquil
    ):
        with pytest.raises(ValueError, match='DEFCIRCUIT'):
            transpile_cycle(
                Cycle({Rz(0, 0.5)}), _gate_mapper(), {},
                cycles_to_defcircuits=True,
                rc_tracker=_StubTracker(),
            )


class TestTranspileCircuitRandomizedReadout:

    @staticmethod
    def _circuit():
        # Final pre-measurement layer: circuit indices 3 and 4.
        return Circuit([
            Cycle({X90(0)}),
            Cycle({X90(0)}),
            Cycle({CZ(0, 1)}),
            Cycle({Rz(0, np.pi / 2)}),
            Cycle({X90(0)}),
            Cycle({Meas(0), Meas(1)}),
        ])

    def test_final_layer_phase_is_recorded_and_referenced(
        self, fake_pyquil
    ):
        _, program = transpile_circuit(
            self._circuit(), _gate_mapper(), randomized_readout=True
        )

        assert program.readout_source_phases == {
            0: [0.25, 0.0, 0.0], 1: [0.0, 0.0, 0.0]
        }
        assert program.source_phases is program.readout_source_phases
        rz = [inst for inst in program if inst.name == 'RZ']
        assert rz == [_Inst(
            'RZ',
            ('scaled', 2 * np.pi, _Ref('measurement_unitary_q0', 0)),
            0,
        )]

    def test_only_final_layer_pulses_are_fenced_per_qubit(
        self, fake_pyquil
    ):
        _, program = transpile_circuit(
            self._circuit(), _gate_mapper(), randomized_readout=True,
            fence_between_cycles=False,
        )
        assert _Inst('FENCE', 0) in program
        assert program.count(_Inst('FENCE', 0)) == 1

    def test_replacing_final_layer_phase_cycle_raises(self, fake_pyquil):
        circuit = self._circuit()
        with pytest.raises(ValueError, match='phase gate'):
            transpile_circuit(
                circuit, _gate_mapper(),
                cycle_replacement={circuit[3]: 'FINAL_RZ'},
                randomized_readout=True,
            )

    def test_replacing_non_final_cycle_is_allowed(self, fake_pyquil):
        circuit = self._circuit()
        _, program = transpile_circuit(
            circuit, _gate_mapper(),
            cycle_replacement={circuit[2]: 'ENTANGLE'},
            randomized_readout=True,
        )
        assert _Inst('QUIL', 'ENTANGLE 0 1') in program

    def test_defcircuits_allow_phase_gates_outside_final_layer(
        self, fake_pyquil
    ):
        circuit = Circuit([
            Cycle({Rz(0, 0.3)}),
            Cycle({X90(0)}),
            Cycle({CZ(0, 1)}),
            Cycle({Rz(0, np.pi / 2)}),
            Cycle({Meas(0), Meas(1)}),
        ])
        _, program = transpile_circuit(
            circuit, _gate_mapper(),
            cycles_to_defcircuits=True, randomized_readout=True,
        )
        # Only the final layer's Rz is randomized.
        assert program.readout_source_phases[0] == [0.25, 0.0, 0.0]

    def test_non_zxzxz_gate_in_final_layer_raises(self, fake_pyquil):
        circuit = Circuit([
            Cycle({CZ(0, 1)}),
            Cycle({Y90(0)}),
            Cycle({Meas(0), Meas(1)}),
        ])
        with pytest.raises(ValueError, match="'Y90'.*ZXZXZ"):
            transpile_circuit(
                circuit, _gate_mapper(), randomized_readout=True
            )

    def test_non_zxzxz_gate_outside_final_layer_is_allowed(
        self, fake_pyquil
    ):
        circuit = Circuit([
            Cycle({Y90(0)}),
            Cycle({CZ(0, 1)}),
            Cycle({Rz(0, np.pi / 2)}),
            Cycle({Meas(0), Meas(1)}),
        ])
        _, program = transpile_circuit(
            circuit, _gate_mapper(), randomized_readout=True
        )
        assert _Inst('Y90', 0) in program

    def test_dd_pulse_in_layer_boundary_is_allowed(self, fake_pyquil):
        # Idle qubits of an RC layer boundary are twirled as identity, so
        # a DD pulse there is not part of a ZXZXZ template.
        circuit = Circuit([
            Cycle({CZ(0, 1), X(2)}),
            Cycle({Rz(0, np.pi / 2)}),
            Cycle({Meas(0), Meas(1), Meas(2)}),
        ])
        _, program = transpile_circuit(
            circuit, _gate_mapper(), randomized_readout=True
        )
        assert _Inst('X', 2) in program
