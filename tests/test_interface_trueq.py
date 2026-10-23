"""Unit tests for qcal.interface.trueq, exercising its behavior when
the optional `trueq` dependency is NOT installed.

Unlike bqskit/pygsti (hard dependencies in pyproject.toml), `trueq` is
never declared as a dependency anywhere -- every function/class here
lazily does `try: import trueq as tq / except ImportError:
logger.warning(...); return` instead, mirroring the convention used
throughout qcal.interface.pyquil.transpiler: log a warning and bail
out early (returning None) rather than falling through into code that
references the never-bound `tq` name.

This current tests only makes sense in an environment without trueq
installed -- it's skipped outright if trueq happens to be present,
since there'd be nothing left to exercise.
"""
import pytest

from qcal.interface.trueq.compiler import QuditCompiler, TrueqCompiler
from qcal.interface.trueq.transpiler import transpile_cycle
from qcal.interface.trueq.utils import (
    X90_phases,
    serialize_cycles,
    set_cycles,
    set_marker,
)

try:
    import trueq  # noqa: F401
    _HAS_TRUEQ = True
except ImportError:
    _HAS_TRUEQ = False

pytestmark = pytest.mark.skipif(
    _HAS_TRUEQ,
    reason=(
        "trueq is installed; this file only covers the "
        "trueq-not-installed code path"
    ),
)


class _DummyTrueqLike:
    """Stands in wherever a function only needs to get past an early
    attribute access on its trueq-typed argument before reaching the
    line that would otherwise reference `tq`.
    """

    def to_dict(self):
        return {}

    def compile(self, x):
        return x


class TestModuleImport:

    def test_transpiler_imports_successfully(self):
        import qcal.interface.trueq.transpiler  # noqa: F401

    def test_compiler_imports_successfully(self):
        import qcal.interface.trueq.compiler  # noqa: F401

    def test_utils_imports_successfully(self):
        import qcal.interface.trueq.utils  # noqa: F401


class TestFunctionsReturnNoneInsteadOfCrashing:
    """Each of these logs 'Unable to import trueq!' and returns None,
    rather than falling through into a crash on the unbound `tq` name.
    """

    def test_set_cycles(self):
        assert set_cycles(_DummyTrueqLike(), []) is None

    def test_set_marker(self):
        assert set_marker(_DummyTrueqLike(), 0) is None

    def test_serialize_cycles(self):
        assert serialize_cycles([]) is None

    def test_x90_phases(self):
        assert X90_phases([0], _DummyTrueqLike()) is None

    def test_transpile_cycle(self):
        assert transpile_cycle(None, None) is None


class TestClassesInitializeWithoutCrashing:

    def test_trueq_compiler(self):
        # Leaves the instance only partially initialized (no
        # _compiler attribute set), matching the same tradeoff
        # PyQuilTranspiler.__init__ already accepts when pyquil is
        # missing -- but it no longer crashes.
        TrueqCompiler(config='bogus.yaml')

    def test_qudit_compiler(self):
        QuditCompiler()
