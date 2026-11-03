"""Unit tests for qcal.calibration.readout (ReadoutCalibration, Fidelity,
Separation).

ReadoutCalibration.analyze() reads real IQ data via per-qubit
'Qn: iq_data' CircuitSet columns to fit its GaussianMixture classifier
-- but the Emulator never populates these columns (it works purely in
terms of qcal.results.Results/populations, not raw I/Q points), so
there's no way to drive this class end-to-end through a real
backend.run(). Instead, generate_circuits() is used only for its
scaffolding (building the right number of circuits), and synthetic
clustered I/Q data -- built the same way as the scikit-learn Gaussian
Mixture Models tutorial
(https://scikit-learn.org/stable/modules/mixture.html): independent
Gaussian clusters per prepared state -- is hand-injected into the
'Qn: iq_data' columns before calling analyze() directly. This recovers
the real GaussianMixture fit/predict/SNR behavior without needing a
simulator that speaks the I/Q representation.

Fidelity and Separation, by contrast, each drive their own internal
ReadoutFidelity/ReadoutCalibration instance via its own .run(), once
per sweep point -- so those are tested by monkeypatching that nested
instance's run() method to inject sweep-dependent synthetic data
instead (a different confusion matrix for Fidelity, a different
cluster separation for Separation), letting the *real* run()/analyze()
of the outer calibration class drive the sweep loop itself.
"""
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.calibration.readout import Fidelity, ReadoutCalibration, Separation
from qcal.simulation.error_models import DepolarizingNoise
from qcal.simulation.simulators import DensityMatrixSimulator


def _iq_cluster(rng, mean, cov, n_shots):
    """n_shots complex I/Q points from an isotropic Gaussian cluster."""
    i = rng.normal(mean[0], cov, n_shots)
    q = rng.normal(mean[1], cov, n_shots)
    return (i + 1j * q).reshape(-1, 1)


class TestReadoutCalibrationConstruction:

    def test_invalid_method_raises(self, config):
        with pytest.raises(ValueError):
            ReadoutCalibration(
                qpu=Emulator, config=config, qubits=[0], method='bogus',
            )

    def test_invalid_gate_raises(self, config):
        with pytest.raises(ValueError):
            ReadoutCalibration(
                qpu=Emulator, config=config, qubits=[0], gate='bogus',
            )

    def test_esp_disabled_by_default_on_example_config(self, config):
        cal = ReadoutCalibration(qpu=Emulator, config=config, qubits=[0])
        assert cal._enable_esp is False

    def test_esp_enabled_is_disabled_during_calibration(self, config):
        config['readout/esp/enable'] = True
        cal = ReadoutCalibration(qpu=Emulator, config=config, qubits=[0])
        assert cal._enable_esp is True
        assert config['readout/esp/enable'] is False

    def test_esp_restored_by_final(self, config):
        config['readout/esp/enable'] = True
        cal = ReadoutCalibration(qpu=Emulator, config=config, qubits=[0])
        cal.final()
        assert config['readout/esp/enable'] is True

    def test_generate_circuits_two_levels(self, config):
        cal = ReadoutCalibration(qpu=Emulator, config=config, qubits=[0])
        cal.generate_circuits()
        assert cal._circuits.n_circuits == 2

    def test_generate_circuits_three_levels(self, config):
        cal = ReadoutCalibration(
            qpu=Emulator, config=config, qubits=[0], n_levels=3,
        )
        cal.generate_circuits()
        assert cal._circuits.n_circuits == 3


class TestReadoutCalibrationSyntheticGmmFit:

    def test_two_well_separated_clusters_are_classified_correctly(
        self, config
    ):
        cal = ReadoutCalibration(qpu=Emulator, config=config, qubits=[0])
        cal.generate_circuits()
        cal._n_shots = 500

        rng = np.random.default_rng(0)
        mean0, mean1 = (0.0, 0.0), (5.0, 5.0)
        cov = 0.3
        cal._circuits['Q0: iq_data'] = [
            _iq_cluster(rng, mean0, cov, 500),
            _iq_cluster(rng, mean1, cov, 500),
        ]
        cal.analyze()

        assert cal._classifier[0].is_fitted
        assert cal._classifier[0].means_[0] == pytest.approx(mean0, abs=0.1)
        assert cal._classifier[0].means_[1] == pytest.approx(mean1, abs=0.1)
        assert cal._classifier[0].predict(
            np.array([mean0])
        )[0] == 0
        assert cal._classifier[0].predict(
            np.array([mean1])
        )[0] == 1
        # Well-separated clusters -> high signal-to-noise ratio.
        assert cal._classifier[0].snr['01'] > 3

    def test_three_level_classification(self, config):
        cal = ReadoutCalibration(
            qpu=Emulator, config=config, qubits=[0], n_levels=3,
        )
        cal.generate_circuits()
        cal._n_shots = 300

        rng = np.random.default_rng(1)
        means = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
        cov = 0.2
        cal._circuits['Q0: iq_data'] = [
            _iq_cluster(rng, m, cov, 300) for m in means
        ]
        cal.analyze()

        assert cal._classifier[0].is_fitted
        for level, mean in enumerate(means):
            assert cal._classifier[0].predict(
                np.array([mean])
            )[0] == level

    def test_overlapping_clusters_have_low_snr(self, config):
        cal = ReadoutCalibration(qpu=Emulator, config=config, qubits=[0])
        cal.generate_circuits()
        cal._n_shots = 500

        rng = np.random.default_rng(2)
        # Means separated by much less than the cluster width.
        cov = 1.0
        cal._circuits['Q0: iq_data'] = [
            _iq_cluster(rng, (0.0, 0.0), cov, 500),
            _iq_cluster(rng, (0.3, 0.0), cov, 500),
        ]
        cal.analyze()

        assert cal._classifier[0].snr['01'] < 1

    def test_plot_runs_against_fitted_classifier(self, config):
        cal = ReadoutCalibration(qpu=Emulator, config=config, qubits=[0])
        cal.generate_circuits()
        cal._n_shots = 300

        rng = np.random.default_rng(0)
        cal._circuits['Q0: iq_data'] = [
            _iq_cluster(rng, (0.0, 0.0), 0.3, 300),
            _iq_cluster(rng, (5.0, 5.0), 0.3, 300),
        ]
        cal.analyze()
        cal.plot()
        cal.plot(raw=True)


class TestFidelitySimulatedSweep:

    def test_fidelity_peaks_at_ideal_readout_param(self, config):
        # Fidelity sweeps a config param and, for each point, drives its
        # internal ReadoutFidelity instance via its own .run(). Since
        # that instance uses the Emulator's DensityMatrixSimulator
        # (results/populations, not I/Q), the readout error itself --
        # not synthetic I/Q clusters -- is what's made to depend on the
        # swept value: a confusion matrix whose off-diagonal error
        # grows linearly away from param_ideal, vanishing exactly at
        # it, so .run()'s fidelity-maximizing argmax should land there.
        param_ideal = 1.0
        param_path = 'readout/0/amp'  # any real, settable float param
        sweep = np.linspace(0.0, 2.0, 11)

        noise = DepolarizingNoise()
        sim = DensityMatrixSimulator(noise_model=noise, n_shots=256)

        cal = Fidelity(
            qpu=Emulator, config=config, qubits=[0],
            params={0: param_path}, param_sweep={0: sweep},
            simulator=sim, n_shots=256,
        )

        real_run = cal._rcal.run

        def _run_with_sweep_dependent_noise():
            val = config[param_path]
            err = min(0.5 * abs(val - param_ideal), 0.49)
            cmat = np.array([[1 - err, err], [err, 1 - err]])
            noise._confusion_matrices = {}
            noise.add_readout_noise('Q0', cmat)
            real_run()

        cal._rcal.run = _run_with_sweep_dependent_noise
        cal.run()

        assert cal.calibrated_values[0] == pytest.approx(param_ideal)


class TestSeparationSimulatedSweep:

    def test_separation_peaks_at_ideal_readout_param(self, config):
        # Separation also sweeps a config param, driving its internal
        # ReadoutCalibration via .run() per point -- but that nested
        # instance IS the I/Q-based classifier, so here the synthetic
        # cluster separation itself (not a confusion matrix) is made
        # to depend on the swept value: cluster distance peaks exactly
        # at param_ideal and falls off on either side, so the
        # SNR-maximizing argmax should land there.
        param_ideal = config['readout/0/freq']
        param_path = 'readout/0/freq'
        sweep = np.linspace(param_ideal - 1e6, param_ideal + 1e6, 11)
        n_shots = 200

        cal = Separation(
            qpu=Emulator, config=config, qubits=[0],
            params={0: param_path}, param_sweep={0: sweep},
        )

        rng = np.random.default_rng(3)

        def _fake_run(plot=False):
            val = config[param_path]
            dist = max(5.0 - 2e-5 * (val - param_ideal) ** 2 / 1e6, 0.1)
            cal._rcal.generate_circuits()
            cal._rcal._n_shots = n_shots
            cal._rcal._circuits['Q0: iq_data'] = [
                _iq_cluster(rng, (0.0, 0.0), 0.2, n_shots),
                _iq_cluster(rng, (dist, 0.0), 0.2, n_shots),
            ]
            cal._rcal.analyze()

        cal._rcal.run = _fake_run
        cal.run()

        assert cal.calibrated_values[0] == pytest.approx(param_ideal)
        assert np.argmax(cal._sep[0]['01']) == len(sweep) // 2
