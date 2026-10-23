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
import pytest

from qcal.interface.pyquil.transpiler import (
    RX_F12,
    RZ_F12,
    PyQuilTranspiler,
    add_CNOT,
    add_CX,
    add_CZ,
    add_Idle,
    add_ISWAP,
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
