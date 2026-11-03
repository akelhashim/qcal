"""Unit tests for qcal.characterization.phase_estimation.rpe, run
against the Emulator.

Like CRB (see test_rb.py), RPE is a pyGSTi-backed protocol: a full
.run() generates real  circuits, executes them (via the Emulator with a
StateVectorSimulator and real shot noise), and runs them through the actual
pyRPE angle-estimation pipeline.

With no coherent gate error -- an ideal X90/CZ/idle unitary, which is
exactly what the example config's noiseless StateVectorSimulator
circuits apply -- every angle error should be statistically
consistent with zero, bounded only by shot noise at the chosen
n_shots/circuit_depths. np.random.seed is set in each test for
reproducibility, since shot sampling is stochastic.
"""
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.characterization.phase_estimation.rpe import (
    CZRPE,
    RPE,
    X90RPE,
    ZZRPE,
    IdleRPE,
)
from qcal.simulation.simulators import StateVectorSimulator


class TestRPEConstruction:

    def test_invalid_gate_raises(self, config):
        with pytest.raises(ValueError):
            RPE(
                qpu=Emulator, config=config, qubit_labels=[0], gate='bogus',
            )

    def test_idle_generate_circuits(self, config):
        rpe = IdleRPE(
            qpu=Emulator, config=config, qubit_labels=[0],
            circuit_depths=(1, 2, 4),
        )
        rpe.generate_circuits()
        assert rpe.gate == 'I'
        assert rpe._circuits.n_circuits == 6

    def test_x90_generate_circuits(self, config):
        rpe = X90RPE(
            qpu=Emulator, config=config, qubit_labels=[0],
            circuit_depths=(1, 2, 4),
        )
        rpe.generate_circuits()
        assert rpe.gate == 'X90'
        assert rpe._circuits.n_circuits == 11

    def test_cz_generate_circuits(self, config):
        rpe = CZRPE(
            qpu=Emulator, config=config, qubit_labels=[(0, 1)],
            circuit_depths=(1, 2, 4),
        )
        rpe.generate_circuits()
        assert rpe.gate == 'CZ'
        assert rpe._circuits.n_circuits == 18

    def test_zz_generate_circuits(self, config):
        rpe = ZZRPE(
            qpu=Emulator, config=config, qubit_labels=[(0, 1)],
            circuit_depths=(1, 2),
        )
        rpe.generate_circuits()
        assert rpe.gate == 'ZZ'
        # 2 depths x 8 twirled randomizations x 6 (sin/cos x 3 state
        # pairs) circuits.
        assert rpe._circuits.n_circuits == 2 * 8 * 6


class TestIdleRPESimulatedRun:

    def test_no_coherent_error_gives_zero_angle_error(self, config):
        np.random.seed(0)
        rpe = IdleRPE(
            qpu=Emulator, config=config, qubit_labels=[0],
            circuit_depths=(1, 2, 4, 8, 16),
            simulator=StateVectorSimulator(), n_shots=5000,
        )
        rpe.run()

        assert rpe.angle_errors[0]['Z'][-1] == pytest.approx(0.0, abs=5e-3)


class TestX90RPESimulatedRun:

    def test_no_coherent_error_gives_zero_angle_error(self, config):
        np.random.seed(10)
        rpe = X90RPE(
            qpu=Emulator, config=config, qubit_labels=[0],
            circuit_depths=(1, 2, 4, 8, 16),
            simulator=StateVectorSimulator(), n_shots=5000,
        )
        rpe.run()

        # 'X' is the rotation-angle error (true angle - pi/2); 'Z' is
        # the off-axis (rotation-axis tilt) error. Both should vanish
        # for an ideal X90 unitary.
        assert rpe.angle_errors[0]['X'][-1] == pytest.approx(0.0, abs=5e-3)
        assert rpe.angle_errors[0]['Z'][-1] == pytest.approx(0.0, abs=5e-3)

    def test_loss_angle_filters_to_single_angle(self, config):
        np.random.seed(11)
        rpe = X90RPE(
            qpu=Emulator, config=config, qubit_labels=[0],
            circuit_depths=(1, 2, 4, 8),
            simulator=StateVectorSimulator(), n_shots=5000,
            loss_angle='X',
        )
        rpe.run()

        assert len(rpe.loss[0]) == 1

    def test_loss_angle_list_includes_both_angles(self, config):
        np.random.seed(12)
        rpe = X90RPE(
            qpu=Emulator, config=config, qubit_labels=[0],
            circuit_depths=(1, 2, 4, 8),
            simulator=StateVectorSimulator(), n_shots=5000,
            loss_angle=['X', 'Z'],
        )
        rpe.run()

        assert len(rpe.loss[0]) == 2


class TestCZRPESimulatedRun:

    def test_no_coherent_error_gives_zero_angle_error(self, config):
        np.random.seed(20)
        rpe = CZRPE(
            qpu=Emulator, config=config, qubit_labels=[(0, 1)],
            circuit_depths=(1, 2, 4, 8, 16),
            simulator=StateVectorSimulator(), n_shots=5000,
        )
        rpe.run()

        errors = rpe.angle_errors[(0, 1)]
        assert errors['ZZ'][-1] == pytest.approx(0.0, abs=5e-3)
        assert errors['IZ'][-1] == pytest.approx(0.0, abs=5e-3)
        assert errors['ZI'][-1] == pytest.approx(0.0, abs=5e-3)


class TestZZRPESimulatedRun:

    def test_no_coherent_error_gives_zero_angle_error(self, config):
        np.random.seed(30)
        rpe = ZZRPE(
            qpu=Emulator, config=config, qubit_labels=[(0, 1)],
            circuit_depths=(1, 2, 4),
            simulator=StateVectorSimulator(), n_shots=5000,
        )
        rpe.run()

        assert rpe.angle_errors[(0, 1)]['ZZ'][-1] == pytest.approx(
            0.0, abs=5e-3
        )


class TestRPEPlotting:

    def test_plot_and_plot_signal_run_without_error(self, config):
        # IdleRPE defines its own plot()/plot_signal() (shared by every
        # gate type via the parent RPE class), so a single gate type is
        # enough to cover the plotting code path. The Agg backend and
        # 'json' plotly renderer are forced session-wide in conftest.py,
        # so there's no risk of a real window popping open.
        np.random.seed(40)
        rpe = IdleRPE(
            qpu=Emulator, config=config, qubit_labels=[0],
            circuit_depths=(1, 2, 4, 8),
            simulator=StateVectorSimulator(), n_shots=5000,
        )
        rpe.run()
        rpe.plot()
        rpe.plot_signal()
