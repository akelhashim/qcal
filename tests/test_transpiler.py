"""Unit tests for qcal.transpilation.transpiler.Transpiler."""
import pytest

from qcal.transpilation.transpiler import Transpiler
from qcal.transpilation.utils import GateMapper


class TestTranspiler:

    def test_cannot_be_instantiated_directly(self):
        # transpile() is an abstract method: every real Transpiler
        # subclass implements it, so the base class should not be
        # usable on its own.
        with pytest.raises(TypeError, match='abstract'):
            Transpiler(GateMapper())

    def test_subclass_must_implement_transpile(self):
        class _Incomplete(Transpiler):
            pass

        with pytest.raises(TypeError, match='abstract'):
            _Incomplete(GateMapper())

    def test_concrete_subclass_exposes_gate_mapper(self):
        gate_mapper = GateMapper({'X': lambda q: q})

        class _Concrete(Transpiler):
            def transpile(self, circuits):
                return circuits

        transpiler = _Concrete(gate_mapper)
        assert transpiler.gate_mapper is gate_mapper

    def test_concrete_subclass_transpile_is_used(self):
        class _Concrete(Transpiler):
            def transpile(self, circuits):
                return list(reversed(circuits))

        transpiler = _Concrete(GateMapper())
        assert transpiler.transpile([1, 2, 3]) == [3, 2, 1]
