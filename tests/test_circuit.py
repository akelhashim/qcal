"""Unit tests for qcal.circuit: Barrier, Cycle, Layer, Circuit, CircuitSet."""
import numpy as np
import pandas as pd
import pytest

from qcal.circuit import Barrier, Circuit, CircuitSet, Cycle, Layer
from qcal.gates.single_qubit import Meas, Ry, X, Y, x, y
from qcal.gates.two_qubit import CNOT
from qcal.results import Results


class TestBarrier:

    def test_default_is_empty(self):
        barrier = Barrier()
        assert barrier.is_barrier is True
        assert barrier.qubits == ()
        assert barrier.qudits == ()

    def test_qubits(self):
        barrier = Barrier((0, 1, 2))
        assert barrier.qubits == (0, 1, 2)

    def test_copy_is_independent(self):
        barrier = Barrier((0, 1))
        clone = barrier.copy()
        assert clone is not barrier
        assert clone.qubits == barrier.qubits


class TestCycle:

    def test_single_gate_construction(self, x0):
        cycle = Cycle(x0)
        assert cycle.n_gates == 1
        assert cycle.qubits == (0,)
        assert cycle.is_barrier is False

    def test_iterable_construction(self, x0, y1):
        cycle = Cycle({x0, y1})
        assert cycle.n_gates == 2
        assert cycle.qubits == (0, 1)
        assert cycle.qudits == cycle.qubits

    def test_append(self, x0, y1):
        cycle = Cycle()
        assert cycle.n_gates == 0
        cycle.append(x0)
        cycle.append([y1])
        assert cycle.n_gates == 2
        assert cycle.qubits == (0, 1)

    def test_getitem_and_iter(self, x0, y1):
        cycle = Cycle({x0, y1})
        assert list(cycle) == cycle.gates
        assert cycle[0] == cycle.gates[0]

    def test_gates_sorted_by_qubit(self):
        cycle = Cycle({Y(1), X(0)})
        assert [gate.qubits for gate in cycle.gates] == [(0,), (1,)]

    def test_equality_independent_of_insertion_order(self):
        c1 = Cycle({X(0), Y(1)})
        c2 = Cycle({Y(1), X(0)})
        assert c1 == c2
        assert hash(c1) == hash(c2)

    def test_inequality_different_gates(self, x0, y1):
        assert Cycle(x0) != Cycle(y1)

    def test_unitary_is_kron_of_sorted_gates(self):
        cycle = Cycle({Y(1), X(0)})
        assert np.allclose(cycle.unitary, np.kron(x, y))

    def test_unitary_empty_cycle_is_identity(self):
        cycle = Cycle()
        assert np.allclose(cycle.unitary, np.eye(1))

    def test_unitary_raises_for_non_unitary_gate(self):
        cycle = Cycle(Meas(0))
        with pytest.raises(ValueError):
            _ = cycle.unitary

    def test_copy_is_deep(self, x0):
        cycle = Cycle(x0)
        clone = cycle.copy()
        assert clone is not cycle
        assert clone == cycle
        assert clone.gates[0] is not cycle.gates[0]


class TestLayer:

    def test_is_cycle_subclass(self, x0):
        layer = Layer({x0})
        assert isinstance(layer, Cycle)
        assert layer.n_gates == 1

    def test_str_prefix(self, x0):
        assert str(Layer({x0})).startswith('Layer')
        assert str(Cycle({x0})).startswith('Cycle')

    def test_equal_to_cycle_with_same_gates(self, x0):
        # Layer inherits Cycle.__eq__, which only compares gate sets.
        assert Layer({x0}) == Cycle({x0})


class TestCircuit:

    def test_construction_from_cycles(self, x0, y1):
        circuit = Circuit([Cycle(x0), Cycle(y1)])
        assert len(circuit) == 2
        assert circuit.qubits == (0, 1)
        assert circuit.qudits == circuit.qubits

    def test_construction_wraps_raw_iterables(self, x0):
        circuit = Circuit([[x0]])
        assert isinstance(circuit[0], Cycle)
        assert circuit == Circuit([Cycle(x0)])

    def test_append(self, x0, y1):
        circuit = Circuit()
        circuit.append(Cycle(x0))
        circuit.append([y1])
        assert circuit == Circuit([Cycle(x0), Cycle(y1)])

    def test_append_rejects_bad_type(self):
        circuit = Circuit()
        with pytest.raises(TypeError):
            circuit.append(42)

    def test_circuit_depth_excludes_barriers(self, x0, y1):
        circuit = Circuit([Cycle(x0), Barrier((0,)), Cycle(y1)])
        assert len(circuit) == 3
        assert circuit.circuit_depth == 2
        assert circuit.n_cycles == 2

    def test_circuit_width(self, x0, y1):
        circuit = Circuit([Cycle(x0), Cycle(y1)])
        assert circuit.circuit_width == 2

    def test_pop_and_popleft(self, x0, y1):
        circuit = Circuit([Cycle(x0), Cycle(y1)])
        circuit.pop()
        assert circuit == Circuit([Cycle(x0)])

        circuit = Circuit([Cycle(x0), Cycle(y1)])
        circuit.popleft()
        assert circuit == Circuit([Cycle(y1)])

    def test_prepend(self, x0, y1):
        circuit = Circuit([Cycle(y1)])
        circuit.prepend(Cycle(x0))
        assert circuit == Circuit([Cycle(x0), Cycle(y1)])

    def test_extend_and_prepend_circuit(self, x0, y1):
        c1 = Circuit([Cycle(x0)])
        c2 = Circuit([Cycle(y1)])
        c1.extend(c2)
        assert c1 == Circuit([Cycle(x0), Cycle(y1)])

        c3 = Circuit([Cycle(y1)])
        c3.prepend_circuit(Circuit([Cycle(x0)]))
        assert c3 == Circuit([Cycle(x0), Cycle(y1)])

    def test_reverse(self, x0, y1):
        circuit = Circuit([Cycle(x0), Cycle(y1)])
        circuit.reverse()
        assert circuit == Circuit([Cycle(y1), Cycle(x0)])

    def test_shift(self, x0, y1):
        circuit = Circuit([Cycle(x0), Cycle(y1)])
        circuit.shift(1)
        assert circuit == Circuit([Cycle(y1), Cycle(x0)])

    def test_get_index_and_remove(self, x0, y1):
        circuit = Circuit([Cycle(x0), Cycle(y1)])
        assert circuit.get_index(Cycle(y1)) == 1
        circuit.remove(Cycle(x0))
        assert len(circuit) == 1
        assert circuit == Circuit([Cycle(y1)])

    def test_replace(self, x0, y1):
        circuit = Circuit([Cycle(x0)])
        circuit.replace(Cycle(x0), Cycle(y1))
        assert circuit == Circuit([Cycle(y1)])

    def test_insert(self, x0, y1):
        circuit = Circuit([Cycle(x0)])
        circuit.insert(Cycle(y1), 0)
        assert circuit == Circuit([Cycle(y1), Cycle(x0)])

    def test_measure_default_z_basis(self, x0):
        circuit = Circuit([Cycle(x0)])
        circuit.measure()
        assert circuit == Circuit([Cycle(x0), Cycle({Meas(0)})])

    def test_measure_non_z_basis_adds_rotation_cycle(self, x0):
        circuit = Circuit([Cycle(x0)])
        circuit.measure(qubits=(0,), basis=('X',))
        assert circuit == Circuit([
            Cycle(x0),
            Cycle({Ry(0, -np.pi / 2)}),
            Cycle({Meas(0, 'X')}),
        ])

    def test_results_setter_and_getter(self, x0):
        circuit = Circuit([Cycle(x0)])
        circuit.results = {'0': 10, '1': 90}
        assert isinstance(circuit.results, Results)
        assert circuit.results.counts['0'] == 10

    def test_mcm_results_setter_and_deleter(self, x0):
        circuit = Circuit([Cycle(x0)])
        circuit.mcm_results = {'0': 5, '1': 5}
        assert len(circuit.mcm_results) == 1
        assert isinstance(circuit.mcm_results[0], Results)
        del circuit.mcm_results
        assert circuit.mcm_results == []

    def test_unitary_of_two_x_gates_is_identity(self, x0):
        circuit = Circuit([Cycle(x0), Cycle(X(0))])
        assert np.allclose(circuit.unitary, np.eye(2))

    def test_unitary_empty_circuit_is_identity(self):
        assert np.allclose(Circuit().unitary, np.eye(1))

    def test_unitary_skips_barriers(self, x0):
        circuit = Circuit([Cycle(x0), Barrier((0,))])
        assert np.allclose(circuit.unitary, x)

    def test_two_qubit_gate_cycle(self):
        circuit = Circuit([Cycle(CNOT(0, 1))])
        assert circuit.qubits == (0, 1)
        assert circuit.circuit_width == 2

    def test_join_outer(self, x0, y1):
        c1 = Circuit([Cycle(x0)])
        c2 = Circuit([Cycle(y1), Cycle(X(1))])
        c1.join(c2, how='outer')
        assert c1 == Circuit([Cycle({x0, y1}), Cycle(X(1))])

    def test_join_inner(self, x0, y1):
        c1 = Circuit([Cycle(x0)])
        c2 = Circuit([Cycle(y1), Cycle(X(1))])
        c1.join(c2, how='inner')
        assert c1 == Circuit([Cycle({x0, y1})])

    def test_join_invalid_how_raises(self, x0):
        c1 = Circuit([Cycle(x0)])
        with pytest.raises(ValueError):
            c1.join(Circuit([Cycle(x0)]), how='bad')

    def test_equality_and_hash(self, x0, y1):
        c1 = Circuit([Cycle(x0), Cycle(y1)])
        c2 = Circuit([Cycle(x0), Cycle(y1)])
        assert c1 == c2
        assert hash(c1) == hash(c2)

    def test_copy_is_independent(self, x0):
        circuit = Circuit([Cycle(x0)])
        clone = circuit.copy()
        assert clone is not circuit
        assert clone == circuit
        clone.append(Cycle(Y(1)))
        assert len(circuit) == 1
        assert len(clone) == 2


class TestCircuitSet:

    def test_construction_from_list(self, bell_circuit):
        cs = CircuitSet([bell_circuit, bell_circuit.copy()])
        assert cs.n_circuits == 2
        assert len(cs) == 2
        assert cs.is_empty is False

    def test_construction_empty(self):
        cs = CircuitSet()
        assert cs.n_circuits == 0
        assert cs.is_empty is True

    def test_getitem_int_returns_circuit(self, bell_circuit):
        cs = CircuitSet([bell_circuit])
        assert cs[0] == bell_circuit

    def test_getitem_slice_returns_circuitset(self, bell_circuit):
        cs = CircuitSet([bell_circuit, bell_circuit, bell_circuit])
        sliced = cs[0:2]
        assert isinstance(sliced, CircuitSet)
        assert len(sliced) == 2

    def test_setitem_and_getitem_column(self, bell_circuit):
        cs = CircuitSet([bell_circuit, bell_circuit])
        cs['amplitude'] = [0.1, 0.2]
        assert list(cs['amplitude']) == [0.1, 0.2]

    def test_iter_yields_circuits(self, bell_circuit):
        cs = CircuitSet([bell_circuit, bell_circuit])
        assert list(cs) == [bell_circuit, bell_circuit]

    def test_circuits_property(self, bell_circuit):
        cs = CircuitSet([bell_circuit])
        assert cs.circuits == [bell_circuit]

    def test_append(self, bell_circuit):
        cs = CircuitSet([bell_circuit])
        cs.append([bell_circuit])
        assert cs.n_circuits == 2

    def test_append_rejects_bad_type(self, bell_circuit):
        cs = CircuitSet([bell_circuit])
        with pytest.raises(TypeError):
            cs.append(42)

    def test_batch(self, bell_circuit):
        cs = CircuitSet([bell_circuit] * 5)
        batches = list(cs.batch(2))
        assert [len(b) for b in batches] == [2, 2, 1]
        assert all(isinstance(b, CircuitSet) for b in batches)

    def test_subset_filters_by_column(self, bell_circuit):
        cs = CircuitSet([bell_circuit, bell_circuit])
        cs['amplitude'] = [0.1, 0.2]
        subset = cs.subset(amplitude=0.1)
        assert subset.n_circuits == 1
        assert list(subset['amplitude']) == [0.1]

    def test_subset_unknown_column_raises(self, bell_circuit):
        cs = CircuitSet([bell_circuit])
        with pytest.raises(KeyError):
            cs.subset(not_a_column=1)

    def test_results_setter_list_and_sum_results(self, bell_circuit):
        cs = CircuitSet([bell_circuit, bell_circuit])
        cs.results = [{'00': 10, '11': 5}, {'00': 3, '11': 7}]
        summed = cs.sum_results()
        assert isinstance(summed, Results)
        assert summed.counts['00'] == 13
        assert summed.counts['11'] == 12

    def test_results_setter_length_mismatch_raises(self, bell_circuit):
        cs = CircuitSet([bell_circuit, bell_circuit])
        with pytest.raises(ValueError):
            cs.results = [{'00': 1}]

    def test_copy_is_independent(self, bell_circuit):
        cs = CircuitSet([bell_circuit])
        clone = cs.copy()
        clone.append([bell_circuit])
        assert cs.n_circuits == 1
        assert clone.n_circuits == 2

    def test_head_and_tail(self, bell_circuit):
        cs = CircuitSet([bell_circuit] * 3)
        assert isinstance(cs.head(1), pd.DataFrame)
        assert len(cs.head(1)) == 1
        assert len(cs.tail(1)) == 1

    def test_save_and_load_roundtrip(self, bell_circuit, tmp_path):
        cs = CircuitSet([bell_circuit, bell_circuit])
        path = tmp_path / 'circuit_set.pkl'
        cs.save(str(path))
        loaded_df = CircuitSet.load(str(path))
        assert len(loaded_df) == 2
