"""Unit tests for qcal.simulation.simulators.

Known-answer physics checks are adapted from the worked examples in
examples/tutorials/Simulation.ipynb: a Bell state via H+CNOT, a qutrit
Hadamard's equal superposition, an X01->X12 walk from |0> to |2>, and a
partial EF-subspace rotation.
"""
import numpy as np
import pytest
import quax

import qcal as qc
from qcal.gates.single_qubit import MCM
from qcal.simulation.error_models import ErrorModel
from qcal.simulation.simulators import (
    DensityMatrixSimulator,
    StateVectorSimulator,
)


class TestStateVectorSimulator:

    def test_bell_state_exact_probabilities(self, bell_circuit):
        bell_circuit.measure()
        sim = StateVectorSimulator()
        sim.run(bell_circuit)
        results = dict(sim.circuits[0].results.dict)
        assert results == pytest.approx({'00': 0.5, '11': 0.5})

    def test_bell_state_sampled_shots_sum_to_n_shots(self, bell_circuit):
        circuit = bell_circuit
        circuit.measure()
        sim = StateVectorSimulator()
        sim.run(circuit, n_shots=1000)
        results = circuit.results.dict
        assert set(results) <= {'00', '11'}
        assert sum(results.values()) == 1000

    def test_custom_unitary_override_is_used(self):
        # Over-rotated X gate (theta = pi + 0.1): P(1) = sin^2(theta/2).
        gate = qc.X(0)
        theta = np.pi + 0.1
        gate.unitary = np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ])
        circuit = qc.Circuit([qc.Cycle({gate})])
        circuit.measure()

        sim = StateVectorSimulator()
        sim.run(circuit)

        results = dict(circuit.results.dict)
        assert results['1'] == pytest.approx(np.sin(theta / 2) ** 2)

    def test_qutrit_hadamard_equal_superposition(self):
        circuit = qc.Circuit([qc.Cycle({qc.H3(0)})])
        circuit.measure()
        sim = StateVectorSimulator()
        sim.run(circuit)
        results = dict(circuit.results.dict)
        assert results == pytest.approx(
            {'0': 1 / 3, '1': 1 / 3, '2': 1 / 3}
        )

    def test_qutrit_walk_from_ground_to_second_excited_state(self):
        circuit = qc.Circuit([
            qc.Cycle({qc.X01(0)}),
            qc.Cycle({qc.X12(0)}),
        ])
        circuit.measure()
        sim = StateVectorSimulator()
        sim.run(circuit)
        assert dict(circuit.results.dict) == pytest.approx({'2': 1.0})

    def test_partial_ef_rotation_gives_equal_superposition(self):
        circuit = qc.Circuit([
            qc.Cycle({qc.X(0)}),
            qc.Cycle({qc.Rx12(0, np.pi / 2)}),
        ])
        circuit.measure()
        sim = StateVectorSimulator()
        sim.run(circuit)
        results = dict(circuit.results.dict)
        assert results == pytest.approx({'1': 0.5, '2': 0.5})

    def test_states_property_holds_one_statevector_per_circuit(
        self, bell_circuit
    ):
        bell_circuit.measure()
        circuits = [bell_circuit, bell_circuit.copy()]
        sim = StateVectorSimulator()
        sim.run(circuits)
        assert len(sim.states) == 2
        assert all(isinstance(s, quax.StateVector) for s in sim.states)
        assert len(sim.circuits) == 2

    def test_instance_default_n_shots_is_used(self):
        circuit = qc.Circuit([qc.Cycle({qc.Z(0)})])
        circuit.measure()
        sim = StateVectorSimulator(n_shots=50)
        sim.run(circuit)
        assert sum(circuit.results.dict.values()) == 50

    def test_run_n_shots_overrides_instance_default(self):
        circuit = qc.Circuit([qc.Cycle({qc.Z(0)})])
        circuit.measure()
        sim = StateVectorSimulator(n_shots=50)
        sim.run(circuit, n_shots=10)
        assert sum(circuit.results.dict.values()) == 10

    def test_z_gate_leaves_ground_state_population_unchanged(self):
        circuit = qc.Circuit([qc.Cycle({qc.Z(0)})])
        circuit.measure()
        sim = StateVectorSimulator()
        sim.run(circuit)
        assert dict(circuit.results.dict) == pytest.approx({'0': 1.0})


class TestDensityMatrixSimulator:

    def test_default_error_model_is_empty(self):
        sim = DensityMatrixSimulator()
        assert sim.error_model == {}

    def test_no_noise_matches_state_vector_simulator(self, bell_circuit):
        bell_circuit.measure()
        dm_sim = DensityMatrixSimulator()
        dm_sim.run(bell_circuit)
        assert dict(dm_sim.circuits[0].results.dict) == pytest.approx(
            {'00': 0.5, '11': 0.5}
        )

    def test_dict_error_model_is_keyed_by_gate_class_name(self):
        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        error_model = {'X': quax.channels.amplitude_damping(0.2)}
        sim = DensityMatrixSimulator(error_model=error_model)
        sim.run(circuit)
        results = dict(circuit.results.dict)
        # Amplitude damping after X(|0>-> |1>) decays P(1) below 1.
        assert results['1'] < 1.0
        assert results['1'] == pytest.approx(0.8187307485370673)

    def test_states_property_holds_density_matrices(self, bell_circuit):
        bell_circuit.measure()
        sim = DensityMatrixSimulator()
        sim.run([bell_circuit])
        assert len(sim.states) == 1
        assert isinstance(sim.states[0], quax.DensityMatrix)

    def test_single_qudit_channel_broadcasts_over_two_qubit_gate(self):
        # A deterministic single-qubit bit-flip Kraus map assigned to a
        # two-qubit gate's category is applied independently to each
        # of the gate's qudits (not as a single joint channel).
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        import jax.numpy as jnp
        flip = quax.KrausMap.from_matrix(
            jnp.array(X[np.newaxis]), dims=((2,), (2,))
        )
        circuit = qc.Circuit([qc.Cycle({qc.CNOT(0, 1)})])
        circuit.measure()
        sim = DensityMatrixSimulator(error_model={'CNOT': flip})
        sim.run(circuit)
        # |00> --CNOT--> |00> --flip both qubits--> |11>
        assert dict(circuit.results.dict) == pytest.approx({'11': 1.0})

    def test_channel_qudit_count_mismatch_raises_value_error(self):
        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        two_qudit_channel = quax.channels.depolarizing(0.1, dims=(2, 2))
        sim = DensityMatrixSimulator(
            error_model={'X': two_qudit_channel}
        )
        with pytest.raises(ValueError, match='qudits'):
            sim.run(circuit)

    def test_invalid_channel_type_raises_type_error(self):
        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        sim = DensityMatrixSimulator(error_model={'X': np.eye(2)})
        with pytest.raises(TypeError, match='KrausMap or.*SuperOp'):
            sim.run(circuit)

    def test_mcm_without_n_shots_raises_value_error(self):
        circuit = qc.Circuit([
            qc.Cycle({qc.X(0)}),
            qc.Cycle({MCM(0)}),
        ])
        circuit.measure()
        sim = DensityMatrixSimulator()
        with pytest.raises(ValueError, match='MCM'):
            sim.run(circuit)

    def test_mcm_results_are_populated_separately_from_terminal_results(self):
        circuit = qc.Circuit([
            qc.Cycle({qc.X(0)}),
            qc.Cycle({MCM(0)}),
        ])
        circuit.measure()
        sim = DensityMatrixSimulator()
        sim.run(circuit, n_shots=200)

        assert dict(circuit.results.dict) == {'1': 200}
        assert len(circuit.mcm_results) == 1
        assert dict(circuit.mcm_results[0].dict) == {'1': 200}

    def test_readout_confusion_matrix_smears_terminal_results(self):
        class _NoGateNoise(ErrorModel):
            def channel_for(self, gate_name):
                return None

        noise = _NoGateNoise()
        # Row-stochastic: P(report 0 | prepared 1) = 1 -- always flips.
        noise.add_readout_noise('Q0', np.array([[0.0, 1.0], [1.0, 0.0]]))

        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        sim = DensityMatrixSimulator(error_model=noise)
        sim.run(circuit)

        assert dict(circuit.results.dict) == pytest.approx({'0': 1.0})
