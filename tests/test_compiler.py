"""Unit tests for qcal.compilation.compiler.Compiler.

Checks that compiled output never contains a gate type outside the
native set (X, X90, CZ for the example config) plus Rz, for both
non-native input gates (which decompose into the native set, with the
single-qubit case following the canonical 'ZXZXZ' pattern: Rz, X90,
Rz, X90, Rz) and already-native input gates. Also checks that the
compiled circuit implements the same unitary (up to global phase) as
the original.
"""
import numpy as np
import pytest

from qcal.circuit import Circuit, Cycle
from qcal.compilation.compiler import Compiler
from qcal.gates.single_qubit import X90, H, Rz, X
from qcal.gates.two_qubit import CNOT, CZ

NATIVE_GATE_NAMES = {'Rz', 'X90', 'X', 'CZ'}


def _gate_names(circuit):
    return {
        gate.name
        for cycle in circuit if not cycle.is_barrier
        for gate in cycle
    }


def _unitaries_equal_up_to_phase(a, b, atol=1e-2):
    idx = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    phase = b[idx] / a[idx]
    return np.allclose(a * phase, b, atol=atol)


@pytest.fixture
def compiler(config):
    return Compiler(config=config)


class TestNonNativeGatesCompileToZXZXZ:

    def test_single_qubit_non_native_gate(self, compiler):
        circuit = Circuit([Cycle({H(0)})])
        compiled = compiler.compile(circuit)[0]

        assert _gate_names(compiled) <= {'Rz', 'X90'}
        assert [
            gate.name for cycle in compiled for gate in cycle
        ] == ['Rz', 'X90', 'Rz', 'X90', 'Rz']
        assert _unitaries_equal_up_to_phase(
            circuit.unitary, compiled.unitary
        )

    def test_two_qubit_non_native_gate_decomposes_using_cz(
        self, compiler
    ):
        circuit = Circuit([Cycle(CNOT(0, 1))])
        compiled = compiler.compile(circuit)[0]

        assert _gate_names(compiled) <= NATIVE_GATE_NAMES
        assert 'CZ' in _gate_names(compiled)
        assert _unitaries_equal_up_to_phase(
            circuit.unitary, compiled.unitary
        )


class TestNativeGatesStayNative:

    @pytest.mark.parametrize('gate', [X90(0), X(0), Rz(0, 0.3)])
    def test_single_qubit_native_gate(self, compiler, gate):
        circuit = Circuit([Cycle({gate})])
        compiled = compiler.compile(circuit)[0]

        assert _gate_names(compiled) <= NATIVE_GATE_NAMES
        assert _unitaries_equal_up_to_phase(
            circuit.unitary, compiled.unitary
        )

    def test_two_qubit_native_gate(self, compiler):
        circuit = Circuit([Cycle(CZ(0, 1))])
        compiled = compiler.compile(circuit)[0]

        assert _gate_names(compiled) <= NATIVE_GATE_NAMES
        assert _unitaries_equal_up_to_phase(
            circuit.unitary, compiled.unitary
        )
