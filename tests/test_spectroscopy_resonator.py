"""Unit tests for qcal.characterization.spectroscopy.resonator (Resonator,
Punchout).

Both classes' analyze() read raw per-shot I/Q points via the
'Qn: iq_data' CircuitSet column, which the Emulator never populates
(see test_spectroscopy_qubit.py for the same situation with TwoTone).
As in that file, generate_circuits() is used only for its scaffolding,
and synthetic I/Q data tracing a Lorentzian dip in |IQ| is hand-
injected before calling analyze() directly.
"""
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.characterization.spectroscopy.resonator import Punchout, Resonator
from qcal.circuit import Circuit, Cycle
from qcal.gates.single_qubit import Meas
from qcal.units import MHz


def _dip_iq(freqs, f0, width, depth, n_shots, rng, noise_std=0.02):
    """Per-frequency synthetic I/Q points tracing a Lorentzian dip."""
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


class TestResonatorConstruction:

    def test_default_params_path(self, config):
        freqs = {0: np.linspace(-1, 1, 11) * MHz}
        spec = Resonator(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        assert spec.params[0] == 'readout/0/freq'

    def test_generate_circuits_produces_one_per_freq_point(self, config):
        freqs = {0: np.linspace(-1, 1, 11) * MHz}
        spec = Resonator(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        spec.generate_circuits()
        assert spec._circuits.n_circuits == 11
        assert spec._circuits[0] == Circuit([Cycle({Meas(0)})])


class TestResonatorSyntheticAnalysis:

    def test_recovers_known_resonance_frequency(self, config):
        f0 = config['readout/0/freq']
        freqs = {0: np.linspace(-20, 20, 201) * MHz + f0}
        spec = Resonator(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        spec.generate_circuits()

        rng = np.random.default_rng(0)
        spec._circuits['Q0: iq_data'] = _dip_iq(
            freqs[0], f0, width=2 * MHz, depth=0.6, n_shots=100, rng=rng,
        )
        spec.analyze()

        freq_step = freqs[0][1] - freqs[0][0]
        assert spec._char_values[0] == pytest.approx(f0, abs=freq_step)

    def test_two_separated_dips_is_a_fitting_error(self, config):
        f0 = config['readout/0/freq']
        freqs = {0: np.linspace(-20, 20, 201) * MHz + f0}
        spec = Resonator(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        spec.generate_circuits()

        rng = np.random.default_rng(1)
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
        f0 = config['readout/0/freq']
        freqs = {0: np.linspace(-20, 20, 201) * MHz + f0}
        spec = Resonator(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        spec.generate_circuits()

        rng = np.random.default_rng(2)
        spec._circuits['Q0: iq_data'] = _dip_iq(
            freqs[0], f0, width=2 * MHz, depth=0.6, n_shots=100, rng=rng,
        )
        spec.analyze()
        spec.final()

        freq_step = freqs[0][1] - freqs[0][0]
        assert config['readout/0/freq'] == pytest.approx(f0, abs=freq_step)


class TestResonatorPlotting:

    def test_plot_runs_without_error(self, config):
        f0 = config['readout/0/freq']
        freqs = {0: np.linspace(-20, 20, 51) * MHz + f0}
        spec = Resonator(qpu=Emulator, config=config, qubits=[0], freqs=freqs)
        spec.generate_circuits()

        rng = np.random.default_rng(3)
        spec._circuits['Q0: iq_data'] = _dip_iq(
            freqs[0], f0, width=2 * MHz, depth=0.6, n_shots=20, rng=rng,
        )
        spec.analyze()
        spec.plot(interactive=False)
        spec.plot(interactive=True)


class TestPunchoutConstruction:

    def test_default_params_path(self, config):
        amps = {0: np.linspace(0.01, 0.1, 5)}
        freqs = {0: np.linspace(-1, 1, 11) * MHz}
        spec = Punchout(
            qpu=Emulator, config=config, qubits=[0], amps=amps, freqs=freqs,
        )
        assert spec._params[0] == ('readout/0/amp', 'readout/0/freq')

    def test_generate_circuits_produces_amp_by_freq_grid(self, config):
        amps = {0: np.linspace(0.01, 0.1, 5)}
        freqs = {0: np.linspace(-1, 1, 11) * MHz}
        spec = Punchout(
            qpu=Emulator, config=config, qubits=[0], amps=amps, freqs=freqs,
        )
        spec.generate_circuits()
        assert spec._circuits.n_circuits == 5 * 11
        assert sorted(
            set(spec._circuits['param: readout/0/amp'])
        ) == pytest.approx(sorted(amps[0]))


class TestPunchoutSyntheticAnalysis:

    def test_analyze_produces_amp_by_freq_arrays(self, config):
        f0 = config['readout/0/freq']
        amps = {0: np.linspace(0.01, 0.1, 5)}
        freqs = {0: np.linspace(-10, 10, 41) * MHz + f0}
        spec = Punchout(
            qpu=Emulator, config=config, qubits=[0], amps=amps, freqs=freqs,
        )
        spec.generate_circuits()

        rng = np.random.default_rng(4)
        iq_col = []
        for amp in amps[0]:
            depth = 0.3 + 0.5 * (amp / amps[0].max())
            iq_col.extend(
                _dip_iq(
                    freqs[0], f0, width=1.5 * MHz, depth=depth, n_shots=30,
                    rng=rng,
                )
            )
        spec._circuits['Q0: iq_data'] = iq_col
        spec.analyze()

        assert spec._iq[0].shape == (5, 41)
        assert spec._mag[0].shape == (5, 41)
        assert spec._phase[0].shape == (5, 41)
        # Deeper dip (higher amp) should have a lower minimum magnitude.
        assert spec._mag[0][-1].min() < spec._mag[0][0].min()


class TestPunchoutPlotting:

    def test_plot_runs_without_error(self, config):
        f0 = config['readout/0/freq']
        amps = {0: np.linspace(0.01, 0.1, 3)}
        freqs = {0: np.linspace(-10, 10, 21) * MHz + f0}
        spec = Punchout(
            qpu=Emulator, config=config, qubits=[0], amps=amps, freqs=freqs,
        )
        spec.generate_circuits()

        rng = np.random.default_rng(5)
        iq_col = []
        for _amp in amps[0]:
            iq_col.extend(
                _dip_iq(
                    freqs[0], f0, width=1.5 * MHz, depth=0.5, n_shots=10,
                    rng=rng,
                )
            )
        spec._circuits['Q0: iq_data'] = iq_col
        spec.analyze()
        spec.plot()
