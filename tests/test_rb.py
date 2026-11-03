"""Unit tests for qcal.benchmarking.rb, run against the Emulator.

CRB is a pyGSTi-backed protocol: construction alone builds a
processor spec and Clifford compilation rules, and a full .run()
generates, executes (via the Emulator), and fits real randomized
circuits. These tests are closer to integration tests than the rest
of the suite and are correspondingly slower.

SRB is not tested here since it requires the (closed-source) True-Q
package, which is not part of the default installation.
"""
import pytest

from qcal.backend.emulator import Emulator
from qcal.benchmarking.rb import CRB
from qcal.simulation.error_models import DepolarizingNoise
from qcal.simulation.simulators import DensityMatrixSimulator


class TestCRBDefaults:

    def test_default_depths_two_qubit(self, config):
        crb = CRB(qpu=Emulator, config=config, qubit_labels=[(0, 1)])
        assert crb.circuit_depths == [2, 4, 8, 32, 64]

    def test_default_depths_single_qubit(self, config):
        crb = CRB(qpu=Emulator, config=config, qubit_labels=[0])
        assert crb.circuit_depths == [2, 8, 32, 128, 256]

    def test_custom_depths_override_default(self, config):
        crb = CRB(
            qpu=Emulator, config=config, qubit_labels=[(0, 1)],
            circuit_depths=[2, 4],
        )
        assert crb.circuit_depths == [2, 4]

    def test_qubit_labels_and_qubits(self, config):
        crb = CRB(qpu=Emulator, config=config, qubit_labels=[(0, 1)])
        assert crb.qubit_labels == [(0, 1)]
        assert crb.qubits == [0, 1]

    def test_randomizeout_default_true(self, config):
        crb = CRB(qpu=Emulator, config=config, qubit_labels=[(0, 1)])
        assert crb.randomizeout is True


class TestCRBRun:

    def test_two_qubit_crb_recovers_realistic_process_infidelity(
        self, config
    ):
        # Uses the Emulator's default noise model (0.5% two-qubit
        # depolarizing plus readout error), so a real decay should be
        # resolved -- a broken pipeline would show up as an
        # infidelity near 0 (no signal) or pegged at 0.5 (no fit).
        crb = CRB(qpu=Emulator, config=config, qubit_labels=[(0, 1)])
        crb.run()

        infidelity = crb.process_infidelity[(0, 1)]
        assert 0. < infidelity['val'] < 0.2
        assert infidelity['err'] > 0.

        assert set(crb.fit_params[(0, 1)].keys()) == {'base', 'a', 'b', 'c'}
        assert crb.circuits.n_circuits == 30 * len(crb.circuit_depths)
        assert (
            sorted(crb.success_probabilities[(0, 1)].keys())
            == crb.circuit_depths
        )


class TestCRBWithCustomNoiseModel:
    """Pin CRB's fit against a known, hand-computed noise model.

    Unlike TestCRBRun (which only checks the fit lands in a broad
    plausible range under the Emulator's default noise), these values
    are specific expected process infidelities for a fixed
    depolarizing rate, confirmed by direct measurement.
    """

    def test_single_qubit_crb_process_infidelity(self, config):
        noise = DepolarizingNoise(single_qubit=0.001)
        sim = DensityMatrixSimulator(noise_model=noise)

        crb = CRB(
            qpu=Emulator, config=config, qubit_labels=[0], simulator=sim
        )
        crb.run()

        infidelity = crb.process_infidelity[0]['val']
        assert infidelity == pytest.approx(4.50e-3, abs=2e-4)

    def test_two_qubit_crb_process_infidelity(self, config):
        noise = DepolarizingNoise(single_qubit=0.001)
        sim = DensityMatrixSimulator(noise_model=noise)

        crb = CRB(
            qpu=Emulator, config=config, qubit_labels=[(0, 1)],
            simulator=sim,
        )
        crb.run()

        infidelity = crb.process_infidelity[(0, 1)]['val']
        assert infidelity == pytest.approx(2.82e-2, abs=1e-3)
