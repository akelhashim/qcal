"""Unit tests for qcal.calibration.cz.Amplitude.

generate_circuits() gives every swept amplitude an identical, ideal CZ
unitary (real hardware compilation -- which would read the
'param: ...' sweep columns back into the pulse amplitude -- is skipped
entirely by the simulator). To simulate the amplitude dependence, each
circuit's CZ gate gets its .unitary hand-overwritten with a
conditional-phase model before running:

    CZ(amp).unitary = diag(1, 1, 1, exp(i * pi * amp / amp_ideal))

i.e. the conditional phase imparted on |11> is modeled as linear in
amp, hitting the ideal pi exactly at amp = amp_ideal (a synthetic
"ideal amplitude" chosen for the test -- this calibration sweeps both
drive lines to the *same* numeric value, unlike the real config, where
they differ -- see test_relative_amp_scales_each_param_by_its_own_
config_value -- so there is no single real config amplitude to recover
here). The *measured* conditionality R is not linear in that phase,
though: near the peak it's quadratic in the phase error (confirmed
empirically, R = 1 - 0.125*(phi - pi)^2 to 6 decimal places over this
sweep), so since phi is linear in amp, R ends up quadratic in amp too
-- which is exactly why Amplitude.analyze() fits it to a parabola, and
why that parabola fit recovers amp_ideal to ~1e-12 precision here.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.calibration.cz import Amplitude
from qcal.fitting.fit import FitParabola
from qcal.simulation.simulators import StateVectorSimulator


@pytest.fixture(autouse=True)
def _close_figures():
    """Prevent matplotlib's open-figure count from growing unbounded."""
    yield
    plt.close('all')


def _set_cz_unitary(circuit, unitary):
    """Overwrite every CZ gate's unitary in a circuit."""
    for cyc in circuit:
        if cyc.is_barrier:
            continue
        for gate in cyc:
            if gate.name == 'CZ':
                gate.unitary = unitary


def _cz_conditional_phase(amp, amp_ideal):
    """diag(1, 1, 1, exp(i pi amp / amp_ideal))."""
    unitary = np.eye(4, dtype=complex)
    unitary[3, 3] = np.exp(1j * np.pi * amp / amp_ideal)
    return unitary


class TestAmplitudeConstruction:

    def test_default_params_path(self, config):
        cal = Amplitude(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            amplitudes=np.linspace(0.9, 1.1, 5),
        )
        assert cal.params[(0, 1)] == (
            'two_qubit/(0, 1)/CZ/pulse/0/kwargs/amp',
            'two_qubit/(0, 1)/CZ/pulse/1/kwargs/amp',
        )

    def test_default_n_gates(self, config):
        cal = Amplitude(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            amplitudes=np.linspace(0.9, 1.1, 5),
        )
        assert cal._n_gates == 1

    def test_uses_two_parabola_fits(self, config):
        cal = Amplitude(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            amplitudes=np.linspace(0.9, 1.1, 5),
        )
        assert [type(f) for f in cal._fit[(0, 1)]] == [
            FitParabola, FitParabola
        ]

    def test_same_amplitudes_applied_to_both_drive_lines(self, config):
        amps = np.linspace(0.9, 1.1, 5)
        cal = Amplitude(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            amplitudes=amps,
        )
        assert cal.amplitudes[(0, 1)][0] is amps
        assert cal.amplitudes[(0, 1)][1] is amps

    def test_relative_amp_scales_each_param_by_its_own_config_value(
        self, config
    ):
        pair = (0, 1)
        amp0 = config['two_qubit/(0, 1)/CZ/pulse/0/kwargs/amp']
        amp1 = config['two_qubit/(0, 1)/CZ/pulse/1/kwargs/amp']
        cal = Amplitude(
            qpu=Emulator, config=config, qubit_pairs=[pair],
            amplitudes=np.array([0.5, 1.0, 1.5]), relative_amp=True,
        )
        assert cal.amplitudes[pair][0] == pytest.approx(
            amp0 * np.array([0.5, 1.0, 1.5])
        )
        assert cal.amplitudes[pair][1] == pytest.approx(
            amp1 * np.array([0.5, 1.0, 1.5])
        )

    def test_generate_circuits_produces_four_sequences(self, config):
        amps = np.linspace(0.9, 1.1, 5)
        cal = Amplitude(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            amplitudes=amps,
        )
        cal.generate_circuits()
        assert cal._circuits.n_circuits == 4 * len(amps)
        assert set(cal._circuits['sequence']) == {
            'C0_X', 'C1_X', 'C0_Y', 'C1_Y'
        }


class TestAmplitudeSimulatedRun:

    @pytest.fixture
    def analyzed_cal(self, config):
        """An Amplitude calibration, run against the synthetic CZ
        conditional-phase model and analyzed.
        """
        pair = (0, 1)
        amp_ideal = 0.3
        amps = np.linspace(0.9 * amp_ideal, 1.1 * amp_ideal, 21)
        cal = Amplitude(
            qpu=Emulator, config=config, qubit_pairs=[pair],
            amplitudes=amps, simulator=StateVectorSimulator(),
            n_shots=None,
        )
        cal.generate_circuits()
        amp_col = cal._circuits[
            'param: two_qubit/(0, 1)/CZ/pulse/0/kwargs/amp'
        ]
        for circuit, amp in zip(cal._circuits, amp_col, strict=True):
            _set_cz_unitary(
                circuit, _cz_conditional_phase(amp, amp_ideal)
            )

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()
        return cal, pair, amp_ideal, amps

    def test_parabola_fit_recovers_ideal_amplitude(self, analyzed_cal):
        cal, pair, amp_ideal, amps = analyzed_cal

        # Conditionality peaks (R=1, maximally conditional) exactly at
        # the ideal amplitude.
        assert cal._R[pair].max() == pytest.approx(1.0, abs=1e-6)
        assert cal._R[pair].argmax() == len(amps) // 2

        assert cal._fit[pair][0].fit_success
        assert cal._fit[pair][1].fit_success
        assert cal.calibrated_values[pair][0] == pytest.approx(
            amp_ideal, abs=1e-6
        )
        assert cal.calibrated_values[pair][1] == pytest.approx(
            amp_ideal, abs=1e-6
        )

    def test_plot_runs_with_amplitude_shaped_data(self, analyzed_cal):
        # cz.Amplitude defines its own plot() (unlike single_qubit.py's
        # Amplitude, which inherits Calibration.plot()), so it isn't
        # covered by test_calibration.py's TestPlot -- this just checks
        # it runs against real conditionality/fit data. The Agg backend
        # is forced session-wide in conftest.py, so there's no risk of
        # a real window popping open.
        cal, *_ = analyzed_cal
        cal.plot()
