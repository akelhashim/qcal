"""Unit tests for qcal.characterization.spectroscopy.qubit.TwoTone.

TwoTone.analyze() reads raw per-shot I/Q points via the 'Qn: iq_data'
CircuitSet column -- but the Emulator never populates that column (it
speaks qcal.results.Results/populations, not raw I/Q), so there is no
way to drive this class end-to-end through a real backend.run(). As
in test_calibrate_readout.py, generate_circuits() is used only for its
scaffolding, and synthetic I/Q data is hand-injected into the
'Qn: iq_data' column before calling analyze() directly.

The synthetic trace models a two-tone spectroscopy lineshape: a
Lorentzian dip in |IQ| centered at a known drive frequency, on top of
a flat background. analyze() finds the dip via a Gaussian-smoothed
scipy.signal.find_peaks() call, so recovering the known center
frequency from the fit exercises the same logic a real notch would.
"""
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.characterization.spectroscopy.qubit import TwoTone
from qcal.circuit import Circuit, Cycle
from qcal.gates.single_qubit import X90, Meas
from qcal.units import MHz


def _dip_iq(freqs, f0, width, depth, n_shots, rng, noise_std=0.02):
    """Per-frequency synthetic I/Q points tracing a Lorentzian dip.

    Args:
        freqs (NDArray): frequency sweep points.
        f0 (float): center frequency of the dip.
        width (float): Lorentzian HWHM.
        depth (float): fractional depth of the dip (0 to 1).
        n_shots (int): number of synthetic I/Q points per frequency.
        rng (np.random.Generator): random number generator.
        noise_std (float): standard deviation of the I/Q noise.

    Returns:
        List[NDArray]: one (n_shots, 1) complex array per frequency.
    """
    lorentzian = 1.0 / (1.0 + ((freqs - f0) / width) ** 2)
    mag = 1.0 - depth * lorentzian
    iq_data = []
    for m in mag:
        noise = (
            rng.normal(0, noise_std, n_shots)
            + 1j * rng.normal(0, noise_std, n_shots)
        )
        iq_data.append((m + noise).reshape(-1, 1))
    return iq_data


class TestTwoToneConstruction:

    def test_default_params_path(self, config):
        freqs = {0: np.linspace(-1, 1, 11) * MHz}
        spec = TwoTone(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        assert spec.params[0] == 'single_qubit/0/GE/freq'

    def test_custom_params_path(self, config):
        freqs = {0: np.linspace(-1, 1, 11) * MHz}
        spec = TwoTone(
            qpu=Emulator, config=config, qubits=[0], freqs=freqs,
            params={0: 'some/custom/param'},
        )
        assert spec.params[0] == 'some/custom/param'

    def test_freqs_array_is_broadcast_to_all_qubits(self, config):
        freqs = np.linspace(-1, 1, 11) * MHz
        spec = TwoTone(
            qpu=Emulator, config=config, qubits=[0, 1], freqs=freqs,
        )
        assert spec.frequencies[0] is spec.frequencies[1]

    def test_generate_circuits_produces_one_per_freq_point(self, config):
        freqs = {0: np.linspace(-1, 1, 11) * MHz}
        spec = TwoTone(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        spec.generate_circuits()
        assert spec._circuits.n_circuits == 11
        assert list(spec._circuits['param: single_qubit/0/GE/freq']) == (
            pytest.approx(freqs[0])
        )

    def test_generate_circuits_cycle_count_matches_n_gates(self, config):
        freqs = {0: np.linspace(-1, 1, 5) * MHz}
        spec = TwoTone(
            qpu=Emulator, config=config, qubits=[0], freqs=freqs, n_gates=3,
        )
        spec.generate_circuits()
        assert spec._circuits[0] == Circuit(
            [Cycle({X90(0)})] * 3 + [Cycle({Meas(0)})]
        )


class TestTwoToneSyntheticAnalysis:

    def test_recovers_known_resonance_frequency(self, config):
        f0 = config['single_qubit/0/GE/freq']
        freqs = {0: np.linspace(-20, 20, 201) * MHz + f0}
        spec = TwoTone(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        spec.generate_circuits()

        rng = np.random.default_rng(0)
        spec._circuits['Q0: iq_data'] = _dip_iq(
            freqs[0], f0, width=2 * MHz, depth=0.6, n_shots=100, rng=rng,
        )
        spec.analyze()

        freq_step = freqs[0][1] - freqs[0][0]
        assert spec._char_values[0] == pytest.approx(f0, abs=freq_step)

    def test_recovers_distinct_frequencies_per_qubit(self, config):
        f0 = {0: config['single_qubit/0/GE/freq'],
              1: config['single_qubit/1/GE/freq']}
        freqs = {q: np.linspace(-20, 20, 201) * MHz + f0[q] for q in (0, 1)}
        spec = TwoTone(
            qpu=Emulator, config=config, qubits=[0, 1], freqs=freqs,
        )
        spec.generate_circuits()

        rng = np.random.default_rng(1)
        for q in (0, 1):
            spec._circuits[f'Q{q}: iq_data'] = _dip_iq(
                freqs[q], f0[q], width=2 * MHz, depth=0.6, n_shots=100,
                rng=rng,
            )
        spec.analyze()

        freq_step = freqs[0][1] - freqs[0][0]
        assert spec._char_values[0] == pytest.approx(f0[0], abs=freq_step)
        assert spec._char_values[1] == pytest.approx(f0[1], abs=freq_step)

    def test_two_separated_dips_is_a_fitting_error(self, config):
        f0 = config['single_qubit/0/GE/freq']
        freqs = {0: np.linspace(-20, 20, 201) * MHz + f0}
        spec = TwoTone(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        spec.generate_circuits()

        rng = np.random.default_rng(2)
        lorentzian_1 = 1.0 / (
            1.0 + ((freqs[0] - (f0 - 10 * MHz)) / (1.5 * MHz)) ** 2
        )
        lorentzian_2 = 1.0 / (
            1.0 + ((freqs[0] - (f0 + 10 * MHz)) / (1.5 * MHz)) ** 2
        )
        mag = 1.0 - 0.6 * lorentzian_1 - 0.6 * lorentzian_2
        iq_data = []
        for m in mag:
            noise = (
                rng.normal(0, 0.02, 100) + 1j * rng.normal(0, 0.02, 100)
            )
            iq_data.append((m + noise).reshape(-1, 1))
        spec._circuits['Q0: iq_data'] = iq_data
        spec.analyze()

        assert spec._char_values[0] is None

    def test_final_sets_config_param(self, config):
        f0 = config['single_qubit/0/GE/freq']
        freqs = {0: np.linspace(-20, 20, 201) * MHz + f0}
        spec = TwoTone(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        spec.generate_circuits()

        rng = np.random.default_rng(3)
        spec._circuits['Q0: iq_data'] = _dip_iq(
            freqs[0], f0, width=2 * MHz, depth=0.6, n_shots=100, rng=rng,
        )
        spec.analyze()
        spec.final()

        freq_step = freqs[0][1] - freqs[0][0]
        assert config['single_qubit/0/GE/freq'] == pytest.approx(
            f0, abs=freq_step
        )


class TestTwoTonePlotting:

    def test_plot_runs_without_error(self, config):
        f0 = config['single_qubit/0/GE/freq']
        freqs = {0: np.linspace(-20, 20, 51) * MHz + f0}
        spec = TwoTone(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        spec.generate_circuits()

        rng = np.random.default_rng(4)
        spec._circuits['Q0: iq_data'] = _dip_iq(
            freqs[0], f0, width=2 * MHz, depth=0.6, n_shots=20, rng=rng,
        )
        spec.analyze()
        spec.plot(interactive=False)
        spec.plot(interactive=True)
