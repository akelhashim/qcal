"""Unit tests for qcal.interface.pygsti.transpiler.

pygsti is a hard dependency (always installed), so these are direct
functional tests. test_gst.py already exercises this transpiler
end-to-end as part of a real GST run; this file isolates its pieces
directly: the 24 single-qubit Clifford decompositions (add_Clifford_C0
.. add_Clifford_C23), the other gate builders, to_qcal()'s layer-type
handling, and PyGSTiTranspiler's three accepted input forms.
"""
import numpy as np
import pygsti
import pytest
from pygsti.baseobjs import Label
from pygsti.io import write_circuit_list

from qcal.gates.single_qubit import X90, Y90, H
from qcal.interface.pygsti.circuits import load_circuits
from qcal.interface.pygsti.compiler import CLIFFORD_UNITARIES
from qcal.interface.pygsti.transpiler import (
    PyGSTiTranspiler,
    add_Clifford_C0,
    add_Clifford_C1,
    add_Clifford_C2,
    add_Clifford_C3,
    add_Clifford_C4,
    add_Clifford_C5,
    add_Clifford_C6,
    add_Clifford_C7,
    add_Clifford_C8,
    add_Clifford_C9,
    add_Clifford_C10,
    add_Clifford_C11,
    add_Clifford_C12,
    add_Clifford_C13,
    add_Clifford_C14,
    add_Clifford_C15,
    add_Clifford_C16,
    add_Clifford_C17,
    add_Clifford_C18,
    add_Clifford_C19,
    add_Clifford_C20,
    add_Clifford_C21,
    add_Clifford_C22,
    add_Clifford_C23,
    add_global_identity,
    add_Hadamard,
    add_idle,
    add_parallel_X90_Y90,
    add_parallel_X90s,
    add_parallel_Y90_X90,
    add_parallel_Y90s,
    to_qcal,
)
from qcal.units import ns

CLIFFORD_FNS = [
    add_Clifford_C0, add_Clifford_C1, add_Clifford_C2, add_Clifford_C3,
    add_Clifford_C4, add_Clifford_C5, add_Clifford_C6, add_Clifford_C7,
    add_Clifford_C8, add_Clifford_C9, add_Clifford_C10, add_Clifford_C11,
    add_Clifford_C12, add_Clifford_C13, add_Clifford_C14, add_Clifford_C15,
    add_Clifford_C16, add_Clifford_C17, add_Clifford_C18, add_Clifford_C19,
    add_Clifford_C20, add_Clifford_C21, add_Clifford_C22, add_Clifford_C23,
]


def _gates_to_unitary(gates):
    U = np.eye(2, dtype=complex)
    for g in gates:
        U = g.unitary @ U
    return U


def _equal_up_to_phase(a, b, atol=1e-8):
    idx = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    phase = b[idx] / a[idx]
    return np.allclose(a * phase, b, atol=atol)


class TestCliffordDecompositions:
    """Each add_Clifford_C{n} should implement the same unitary (up to
    global phase) as pyGSTi's own 'Gc{n}' gate-name convention -- the
    same CLIFFORD_UNITARIES already used in
    qcal.interface.pygsti.compiler.
    """

    @pytest.mark.parametrize('n, fn', list(enumerate(CLIFFORD_FNS)))
    def test_matches_pygsti_clifford_convention(self, n, fn):
        gates = fn(0)
        assert len(gates) == 3
        assert all(g.qubits == (0,) for g in gates)
        U = _gates_to_unitary(gates)
        assert _equal_up_to_phase(U, CLIFFORD_UNITARIES[f'Gc{n}'])


class TestOtherGateBuilders:

    def test_add_hadamard_matches_h_gate(self):
        U = _gates_to_unitary(add_Hadamard(0))
        assert _equal_up_to_phase(U, H(0).unitary)

    def test_add_idle_is_100ns(self):
        gate = add_idle(0)
        assert gate.properties['params']['duration'] == 100 * ns

    def test_add_global_identity_is_empty_set(self):
        assert add_global_identity((0, 1)) == set()

    def test_add_parallel_x90s(self):
        gates = list(add_parallel_X90s((0, 1)))
        assert [type(g) for g in gates] == [X90, X90]
        assert [g.qubits for g in gates] == [(0,), (1,)]

    def test_add_parallel_y90s(self):
        gates = list(add_parallel_Y90s((0, 1)))
        assert [type(g) for g in gates] == [Y90, Y90]

    def test_add_parallel_x90_y90(self):
        gates = list(add_parallel_X90_Y90((0, 1)))
        assert [type(g) for g in gates] == [X90, Y90]
        assert [g.qubits for g in gates] == [(0,), (1,)]

    def test_add_parallel_y90_x90(self):
        gates = list(add_parallel_Y90_X90((0, 1)))
        assert [type(g) for g in gates] == [Y90, X90]


@pytest.fixture
def gate_mapper():
    return PyGSTiTranspiler()._gate_mapper


class TestToQcal:

    def test_empty_circuit_is_just_a_measurement(self, gate_mapper):
        circuit = pygsti.circuits.Circuit([], line_labels=('Q0',))
        tc = to_qcal(circuit, gate_mapper)

        assert len(tc) == 1
        assert tc[0] == tc[0]  # sanity: measurement cycle present
        assert tc[0].is_barrier is False
        assert list(tc[0])[0].name == 'Meas'

    def test_isolated_gate_layers_get_barriers_between_them(
        self, gate_mapper
    ):
        circuit = pygsti.circuits.Circuit(
            [Label('Gxpi2', 'Q0'), Label('Gzr', 'Q0', args=(0.5,))],
            line_labels=('Q0',),
        )
        tc = to_qcal(circuit, gate_mapper)

        kinds = [(cyc.is_barrier, [g.name for g in cyc] if not cyc.is_barrier
                  else None) for cyc in tc]
        assert kinds == [
            (False, ['X90']), (True, None), (False, ['Rz']),
            (True, None), (False, ['Meas']),
        ]

    def test_parallel_layer_becomes_single_cycle(self, gate_mapper):
        circuit = pygsti.circuits.Circuit(
            [[Label('Gxpi2', 'Q0'), Label('Gypi2', 'Q1')]],
            line_labels=('Q0', 'Q1'),
        )
        tc = to_qcal(circuit, gate_mapper)

        assert {g.name for g in tc[0]} == {'X90', 'Y90'}

    def test_idling_cycle_maps_to_identity(self, gate_mapper):
        circuit = pygsti.circuits.Circuit([[]], line_labels=('Q0',))
        tc = to_qcal(circuit, gate_mapper)

        assert [g.name for g in tc[0]] == ['I']

    def test_clifford_gate_expands_to_three_cycles(self, gate_mapper):
        circuit = pygsti.circuits.Circuit(
            [Label('Gc1', 'Q0')], line_labels=('Q0',)
        )
        tc = to_qcal(circuit, gate_mapper)

        names = [
            [g.name for g in cyc] for cyc in tc if not cyc.is_barrier
        ]
        assert names == [['Rz'], ['X90'], ['Rz'], ['Meas']]

    def test_mixed_layer_merges_simple_gate_with_clifford_expansion(
        self, gate_mapper
    ):
        # Q0 gets a plain gate (Gxpi2); Q1 gets a 3-gate Clifford
        # expansion (Gc1) in the same parallel layer.
        circuit = pygsti.circuits.Circuit(
            [[Label('Gxpi2', 'Q0'), Label('Gc1', 'Q1')]],
            line_labels=('Q0', 'Q1'),
        )
        tc = to_qcal(circuit, gate_mapper)

        non_barrier = [cyc for cyc in tc if not cyc.is_barrier]
        # First cycle merges Q0's X90 with Q1's first Clifford gate.
        assert {(g.name, g.qubits) for g in non_barrier[0]} == {
            ('X90', (0,)), ('Rz', (1,))
        }
        assert [g.name for g in non_barrier[1]] == ['X90']
        assert [g.name for g in non_barrier[2]] == ['Rz']

    def test_final_measurement_covers_all_qubits(self, gate_mapper):
        circuit = pygsti.circuits.Circuit(
            [Label('Gxpi2', 'Q0')], line_labels=('Q0', 'Q1'),
        )
        tc = to_qcal(circuit, gate_mapper)

        meas_cycle = tc[-1]
        assert {g.qubits for g in meas_cycle} == {(0,), (1,)}


class TestPyGSTiTranspilerConstruction:

    def test_default_gate_mapper_has_expected_keys(self):
        t = PyGSTiTranspiler()
        for key in ('Gxpi2', 'Gypi2', 'Gcnot', 'Gcphase', 'Gc0', 'Gc23'):
            assert key in t.gate_mapper

    def test_custom_dict_gate_mapper_is_wrapped(self):
        t = PyGSTiTranspiler(gate_mapper={'Gxpi2': lambda q: None})
        assert 'Gxpi2' in t.gate_mapper
        assert 'Gypi2' not in t.gate_mapper


class TestPyGSTiTranspilerTranspile:

    def _circuits(self):
        return [
            pygsti.circuits.Circuit(
                [Label('Gxpi2', 'Q0')], line_labels=('Q0',)
            ),
            pygsti.circuits.Circuit(
                [Label('Gypi2', 'Q0')], line_labels=('Q0',)
            ),
        ]

    def test_from_list(self):
        circuits = self._circuits()
        result = PyGSTiTranspiler().transpile(circuits)

        assert result.n_circuits == 2
        assert list(result['pygsti_circuit']) == [
            c.str for c in circuits
        ]

    def test_from_circuit_set(self):
        circuits = self._circuits()
        cs = load_circuits(circuits)
        result = PyGSTiTranspiler().transpile(cs)

        assert result.n_circuits == 2
        assert list(result['pygsti_circuit']) == [
            c.str for c in circuits
        ]

    def test_from_file_path(self, tmp_path):
        circuits = self._circuits()
        path = str(tmp_path / 'circuits.txt')
        write_circuit_list(path, circuits)

        result = PyGSTiTranspiler().transpile(path)

        assert result.n_circuits == 2
        assert list(result['pygsti_circuit']) == [
            c.str for c in circuits
        ]
