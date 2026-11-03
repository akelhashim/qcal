"""Unit tests for qcal.transpilation.utils."""
import pytest

from qcal.circuit import Circuit
from qcal.gates.single_qubit import X, Y
from qcal.transpilation.utils import GateMapper, transpilation_error


class TestTranspilationError:

    def test_raises_with_gate_name_in_message(self):
        with pytest.raises(Exception, match="Cannot transpile 'Foo'"):
            transpilation_error('Foo')


class TestGateMapperConstruction:

    def test_empty_construction(self):
        assert dict(GateMapper()) == {}

    def test_from_dict(self):
        mapper = GateMapper({'a': 1, 'b': 2})
        assert dict(mapper) == {'a': 1, 'b': 2}

    def test_from_iterable_of_pairs(self):
        mapper = GateMapper([('a', 1), ('b', 2)])
        assert dict(mapper) == {'a': 1, 'b': 2}

    def test_from_kwargs(self):
        mapper = GateMapper(a=1, b=2)
        assert dict(mapper) == {'a': 1, 'b': 2}


class TestGateMapperMissing:

    def test_missing_key_returns_a_callable(self):
        mapper = GateMapper()
        assert callable(mapper['NotAGate'])

    def test_missing_key_does_not_raise_until_called(self):
        mapper = GateMapper()
        handler = mapper['NotAGate']  # does not raise
        with pytest.raises(Exception, match="Cannot transpile 'NotAGate'"):
            handler()


class TestGateMapperCall:

    def test_gate_returning_function_is_wrapped_in_a_set(self):
        mapper = GateMapper({'X': lambda q: X(q)})
        result = mapper.call('X', 0)
        assert result == {X(0)}

    def test_iterator_returning_function_becomes_a_set(self):
        def gen(q):
            yield X(q)
            yield Y(q)

        mapper = GateMapper({'XY': gen})
        result = mapper.call('XY', 0)
        assert result == {X(0), Y(0)}

    def test_list_returning_function_becomes_a_circuit(self):
        mapper = GateMapper({'XY': lambda q: [X(q), Y(q)]})
        result = mapper.call('XY', 0)
        assert isinstance(result, Circuit)
        assert len(result) == 2
        assert [cycle[0] for cycle in result] == [X(0), Y(0)]

    def test_unsupported_return_type_raises_type_error(self):
        mapper = GateMapper({'bad': lambda q: 42})
        with pytest.raises(TypeError, match='unsupported type'):
            mapper.call('bad', 0)

    def test_string_return_type_raises_type_error(self):
        mapper = GateMapper({'bad': lambda q: 'ab'})
        with pytest.raises(TypeError, match='unsupported type'):
            mapper.call('bad', 0)

    def test_missing_key_raises_transpilation_error(self):
        mapper = GateMapper()
        with pytest.raises(Exception, match="Cannot transpile 'missing'"):
            mapper.call('missing', 0)

    def test_passes_through_args_and_kwargs(self):
        mapper = GateMapper({'X': lambda q, subspace='GE': X(q, subspace)})
        result = mapper.call('X', 0, subspace='EF')
        assert result == {X(0, subspace='EF')}
