"""Unit tests for qcal.backend.emulator.Emulator.

These pin the Emulator's own wiring contract (default config/noise
model construction, custom simulator injection, and how results flow
back into circuits/CircuitSets) rather than the physics of the
underlying simulators, which are exercised elsewhere.
"""
import numpy as np
import quax

from qcal.backend.emulator import Emulator
from qcal.circuit import Circuit, CircuitSet, Cycle
from qcal.gates.single_qubit import Meas, X
from qcal.results import Results
from qcal.simulation.simulators import (
    DensityMatrixSimulator,
    StateVectorSimulator,
)


class TestDefaultConstruction:

    def test_default_config_is_example_config(self):
        emu = Emulator()
        assert emu.config.filename.endswith('config.yaml')

    def test_custom_config_is_used(self, config):
        emu = Emulator(config=config)
        assert emu.config is config

    def test_default_simulator_is_density_matrix(self):
        emu = Emulator()
        assert isinstance(emu.simulator, DensityMatrixSimulator)

    def test_default_n_shots(self):
        emu = Emulator()
        assert emu.simulator.n_shots == 1024

    def test_default_single_qubit_depolarizing_rate(self):
        emu = Emulator()
        channel = emu.simulator.noise_model.channel_for('X')
        expected = quax.channels.depolarizing(0.0005, dims=(2,))
        assert np.allclose(channel.matrix, expected.matrix)

    def test_default_two_qubit_depolarizing_rate(self):
        emu = Emulator()
        channel = emu.simulator.noise_model.channel_for('CZ')
        expected = quax.channels.depolarizing(0.005, dims=(2,))
        assert np.allclose(channel.matrix, expected.matrix)

    def test_default_readout_confusion_matrix_per_qubit(self, config):
        emu = Emulator(config=config)
        expected = np.array([[0.995, 0.02], [0.005, 0.98]])
        for q in config.qubits:
            cmat = emu.simulator.noise_model.confusion_matrix_for(f'Q{q}')
            assert np.allclose(cmat, expected)


class TestCustomSimulator:

    def test_custom_simulator_is_used_as_is(self):
        custom = StateVectorSimulator()
        emu = Emulator(simulator=custom)
        assert emu.simulator is custom


class TestRun:

    def test_run_writes_results_to_circuit(self):
        # Noiseless simulator: X on |0> deterministically measures '1'.
        circuit = Circuit([Cycle(X(0)), Cycle(Meas(0))])
        emu = Emulator(simulator=StateVectorSimulator())
        emu.run(circuit, save=False)

        assert isinstance(circuit.results, Results)
        assert dict(circuit.results.dict) == {'1': 1024}

    def test_run_respects_n_shots_override(self):
        circuit = Circuit([Cycle(X(0)), Cycle(Meas(0))])
        emu = Emulator(simulator=StateVectorSimulator())
        emu.run(circuit, n_shots=10, save=False)

        assert circuit.results.n_shots == 10
        assert dict(circuit.results.dict) == {'1': 10}

    def test_run_writes_results_and_states_columns(self):
        circuit = Circuit([Cycle(X(0)), Cycle(Meas(0))])
        emu = Emulator(simulator=StateVectorSimulator())
        emu.run(CircuitSet([circuit]), save=False)

        assert list(emu.circuits['results']) == [{'1': 1024}]
        assert emu.circuits['states'].iloc[0] == '1|1⟩'
