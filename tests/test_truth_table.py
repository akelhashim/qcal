"""Unit tests for qcal.characterization.tomography.truth_table.TruthTable.

TruthTable prepares every computational basis state on `qubits` (via
two Id or two X90 pulses per qubit -- a doubled X90 nets a full X
rotation, matching how the rest of the native pulse set is calibrated),
runs the circuit under test, and measures. `analyze()` builds
`truth_table[i, j] = P(output = states[j] | input = states[i])` and,
if `ideal_unitary` is given (despite the name, this is the *ideal
binary truth table* in the same [input, output] layout -- e.g. a
permutation matrix for a classical reversible gate like CNOT or X, not
a complex unitary), computes
`fidelity = Tr(truth_table.T @ ideal_unitary) / 2**n`, which reduces to
the normalized Frobenius inner product of the two matrices. For a
noiseless `StateVectorSimulator` run, truth_table should exactly equal
the gate's ideal permutation matrix, giving fidelity == 1.0 (not just
close to it) with `n_shots=None` exact-probability simulation -- no
need to seed `np.random` or tolerate statistical noise.

`.run()` is used directly throughout.

Known issue: when `qubits` is a strict superset of `circuit.qubits`
(e.g. an untouched ancilla included only for state preparation),
`analyze()` reuses `self._states` -- built from the full `qubits`
list -- as the *output*-state lookup keys against a `Results` object
that was marginalized down to just `circuit.qubits` (a different,
shorter bitstring length). Every lookup therefore misses and silently
returns 0 (`populations` is a zero-defaulting `defaultdict`), so
`truth_table` comes back all zeros instead of the correct,
ancilla-invariant truth table -- with no error raised. See
`TestKnownIssues` below.
"""
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.characterization.tomography.truth_table import TruthTable
from qcal.circuit import Circuit, Cycle
from qcal.gates.single_qubit import X90, Id, Meas, X
from qcal.gates.two_qubit import CNOT
from qcal.simulation.simulators import StateVectorSimulator

CNOT_TRUTH_TABLE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=float)

X_TRUTH_TABLE = np.array([[0, 1], [1, 0]], dtype=float)

IDENTITY_TRUTH_TABLE = np.eye(2)


class TestTruthTableConstruction:

    def test_qubits_defaults_to_circuit_qubits(self, config):
        circuit = Circuit([Cycle({X90(0)})])
        tt = TruthTable(qpu=Emulator, config=config, circuit=circuit)
        assert tt.qubits == circuit.qubits

    def test_custom_qubits_override(self, config):
        circuit = Circuit([Cycle(CNOT(0, 1))])
        tt = TruthTable(
            qpu=Emulator, config=config, circuit=circuit, qubits=[1, 0],
        )
        assert tt.qubits == [1, 0]

    def test_generate_circuits_produces_2n_circuits(self, config):
        circuit = Circuit([Cycle(CNOT(0, 1))])
        tt = TruthTable(qpu=Emulator, config=config, circuit=circuit)
        tt.generate_circuits()
        assert tt.circuits.n_circuits == 4
        assert list(tt.circuits['input state']) == ['00', '01', '10', '11']

    def test_generate_circuits_state_prep_cycles(self, config):
        circuit = Circuit([Cycle({X90(0)})])
        tt = TruthTable(qpu=Emulator, config=config, circuit=circuit)
        tt.generate_circuits()

        assert tt.circuits[0] == Circuit([
            Cycle({Id(0)}), Cycle({Id(0)}), Cycle({X90(0)}), Cycle({Meas(0)})
        ])
        assert tt.circuits[1] == Circuit([
            Cycle({X90(0)}), Cycle({X90(0)}), Cycle({X90(0)}), Cycle({Meas(0)})
        ])


class TestTruthTableSimulatedRun:

    def test_identity_circuit_with_no_ideal_unitary(self, config):
        circuit = Circuit([Cycle({Id(0)})])
        tt = TruthTable(
            qpu=Emulator, config=config, circuit=circuit,
            simulator=StateVectorSimulator(), n_shots=None,
        )
        tt.run()

        assert tt.truth_table == pytest.approx(IDENTITY_TRUTH_TABLE)
        assert tt.fidelity is None

    def test_x_gate_truth_table_and_fidelity(self, config):
        circuit = Circuit([Cycle({X(0)})])
        tt = TruthTable(
            qpu=Emulator, config=config, circuit=circuit,
            ideal_unitary=X_TRUTH_TABLE,
            simulator=StateVectorSimulator(), n_shots=None,
        )
        tt.run()

        assert tt.truth_table == pytest.approx(X_TRUTH_TABLE)
        assert tt.fidelity['val'] == pytest.approx(1.0)

    def test_cnot_truth_table_is_exact_permutation(self, config):
        circuit = Circuit([Cycle(CNOT(0, 1))])
        tt = TruthTable(
            qpu=Emulator, config=config, circuit=circuit,
            ideal_unitary=CNOT_TRUTH_TABLE,
            simulator=StateVectorSimulator(), n_shots=None,
        )
        tt.run()

        assert tt.truth_table == pytest.approx(CNOT_TRUTH_TABLE)
        assert tt.fidelity['val'] == pytest.approx(1.0)

    def test_fidelity_reflects_mismatch_with_wrong_ideal_unitary(
        self, config
    ):
        # The real CNOT truth table only overlaps the identity matrix
        # on the 2 (of 4) input states where the control is |0> --
        # i.e. where CNOT and identity agree -- giving fidelity 2/4.
        circuit = Circuit([Cycle(CNOT(0, 1))])
        tt = TruthTable(
            qpu=Emulator, config=config, circuit=circuit,
            ideal_unitary=np.eye(4),
            simulator=StateVectorSimulator(), n_shots=None,
        )
        tt.run()

        assert tt.truth_table == pytest.approx(CNOT_TRUTH_TABLE)
        assert tt.fidelity['val'] == pytest.approx(0.5)


class TestKnownIssues:

    @pytest.mark.xfail(
        reason=(
            "analyze() reuses self._states (built from the full "
            "`qubits` list) as the output-state lookup keys even when "
            "`qubits` is a strict superset of circuit.qubits, but the "
            "marginalized Results it looks up has shorter bitstring "
            "keys (circuit.qubits-length). Every lookup misses and "
            "silently returns 0 via the zero-defaulting populations "
            "defaultdict, so truth_table comes back all zeros instead "
            "of the correct, ancilla-invariant truth table."
        ),
        strict=True,
    )
    def test_ancilla_superset_qubits_gives_nonzero_truth_table(
        self, config
    ):
        circuit = Circuit([Cycle({X(1)})])
        tt = TruthTable(
            qpu=Emulator, config=config, circuit=circuit, qubits=[0, 1],
            simulator=StateVectorSimulator(), n_shots=None,
        )
        tt.run()

        assert tt.truth_table.sum() > 0
