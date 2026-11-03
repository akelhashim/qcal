"""Unit tests for qcal.interface.superstaq, exercising its behavior
when the optional `cirq_superstaq`/`qiskit_superstaq` dependencies are
NOT installed.
"""
import pytest

try:
    import cirq_superstaq  # noqa: F401
    _HAS_CIRQ_SUPERSTAQ = True
except ImportError:
    _HAS_CIRQ_SUPERSTAQ = False

try:
    import qiskit_superstaq  # noqa: F401
    _HAS_QISKIT_SUPERSTAQ = True
except ImportError:
    _HAS_QISKIT_SUPERSTAQ = False

pytestmark = pytest.mark.skipif(
    _HAS_CIRQ_SUPERSTAQ or _HAS_QISKIT_SUPERSTAQ,
    reason=(
        "cirq_superstaq/qiskit_superstaq are installed; this file "
        "only covers the not-installed code path"
    ),
)


def test_compiler_module_imports_successfully():
    import qcal.interface.superstaq.compiler  # noqa: F401


def test_cirq_compiler_construction_raises_clean_import_error():
    from qcal.interface.superstaq.compiler import CirqCompiler
    with pytest.raises(ImportError):
        CirqCompiler(api_key='bogus', config=None)


def test_qiskit_compiler_construction_raises_clean_import_error():
    from qcal.interface.superstaq.compiler import QiskitCompiler
    with pytest.raises(ImportError):
        QiskitCompiler(api_key='bogus', config=None)
