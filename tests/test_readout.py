"""Unit tests for qcal.benchmarking.readout.ReadoutFidelity.

Uses the Emulator's default per-qubit readout confusion matrix as
ground truth: a correct pipeline should recover values close to that
matrix from simulated measurement counts.
"""
import numpy as np
import pytest

from qcal.backend.emulator import (
    DEFAULT_READOUT_P0,
    DEFAULT_READOUT_P1,
    Emulator,
)
from qcal.benchmarking.readout import ReadoutFidelity

# qcal's row-stochastic convention: C[prep, meas] = P(meas | prep).
EXPECTED_CMAT = np.array([
    [DEFAULT_READOUT_P0, 1 - DEFAULT_READOUT_P0],
    [1 - DEFAULT_READOUT_P1, DEFAULT_READOUT_P1],
])


class TestReadoutFidelityDefaults:

    def test_invalid_gate_raises(self, config):
        with pytest.raises(ValueError):
            ReadoutFidelity(
                qpu=Emulator, config=config, qubits=[0], gate='Y90'
            )

    def test_cmat_is_none_before_run(self, config):
        ro = ReadoutFidelity(qpu=Emulator, config=config, qubits=[0])
        assert ro.cmat is None

    def test_generate_circuits_produces_one_per_level(self, config):
        # n_levels defaults to 2 (qubits, not qutrits): one circuit
        # preparing |0>, one preparing |1>.
        ro = ReadoutFidelity(qpu=Emulator, config=config, qubits=[0, 1])
        ro.generate_circuits()
        assert ro.circuits.n_circuits == 2
        assert list(ro.circuits['prep state']) == [0, 1]


class TestReadoutFidelityRun:

    def test_recovers_emulator_default_confusion_matrix(self, config):
        # A higher shot count than the Emulator's own default (1024)
        # tightens the statistical spread without adding real runtime,
        # since circuit count here doesn't depend on n_shots.
        ro = ReadoutFidelity(
            qpu=Emulator, config=config, qubits=[0, 1], n_shots=8192,
        )
        ro.run()

        for q in (0, 1):
            measured = ro.cmat[f'Q{q}'].to_numpy(dtype=float)
            assert measured == pytest.approx(EXPECTED_CMAT, abs=0.01)
