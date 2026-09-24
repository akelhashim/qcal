"""Unit tests for qcal.calibration.single_qubit.Amplitude.

The Rabi/parabola fits are exercised end-to-end against a noiseless
StateVectorSimulator. generate_circuits() gives every swept amplitude
an identical, ideal X90 unitary (real hardware compilation -- which
would read the 'param: ...' sweep column back into the pulse amplitude
-- is skipped entirely by the simulator), so each circuit's X90
gate(s) get their .unitary hand-overwritten with an amplitude-
dependent rotation before running. This only works because
Circuit.copy() deep-copies every gate, so each swept circuit's gates
are independent objects that can be mutated without affecting the
others.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.calibration.single_qubit import Amplitude
from qcal.calibration.utils import find_pulse_index
from qcal.fitting.fit import FitCosine, FitParabola
from qcal.gates.single_qubit import rx
from qcal.simulation.simulators import StateVectorSimulator


@pytest.fixture(autouse=True)
def _close_figures():
    """Prevent matplotlib's open-figure count from growing unbounded."""
    yield
    plt.close('all')


def _x90_amp_param(config, qubit, subspace='GE'):
    """The config path to a qubit's X90 pulse amplitude, and its value."""
    param = f'single_qubit/{qubit}/{subspace}/X90/pulse'
    param += f'/{find_pulse_index(config, param)}/kwargs/amp'
    return param, config[param]


def _set_x90_unitary(circuit, theta):
    """Overwrite every X90 gate's unitary in a circuit with rx(theta)."""
    for cyc in circuit:
        if cyc.is_barrier:
            continue
        for gate in cyc:
            if gate.name == 'X90':
                gate.unitary = rx(theta)


class TestAmplitudeConstruction:

    def test_default_params_path_is_derived_from_config(self, config):
        param, _ = _x90_amp_param(config, 0)
        cal = Amplitude(
            qpu=Emulator, config=config, qubits=[0],
            amplitudes=np.linspace(0., 1., 5),
        )
        assert cal.params[0] == param

    def test_invalid_gate_raises(self, config):
        with pytest.raises(ValueError):
            Amplitude(
                qpu=Emulator, config=config, qubits=[0],
                amplitudes=np.linspace(0., 1., 5), gate='Y90',
            )

    def test_invalid_method_raises(self, config):
        with pytest.raises(ValueError):
            Amplitude(
                qpu=Emulator, config=config, qubits=[0],
                amplitudes=np.linspace(0., 1., 5), method='bogus',
            )

    def test_invalid_subspace_raises(self, config):
        with pytest.raises(ValueError):
            Amplitude(
                qpu=Emulator, config=config, qubits=[0],
                amplitudes=np.linspace(0., 1., 5), subspace='bogus',
            )

    def test_x90_n_gates_must_be_multiple_of_four(self, config):
        with pytest.raises(ValueError):
            Amplitude(
                qpu=Emulator, config=config, qubits=[0],
                amplitudes=np.linspace(0., 1., 5), n_gates=2,
            )

    def test_x_n_gates_must_be_multiple_of_two(self, config):
        with pytest.raises(ValueError):
            Amplitude(
                qpu=Emulator, config=config, qubits=[0],
                amplitudes=np.linspace(0., 1., 5), gate='X', n_gates=3,
            )

    def test_relative_amp_scales_by_current_config_value(self, config):
        _, amp_cal = _x90_amp_param(config, 0)
        cal = Amplitude(
            qpu=Emulator, config=config, qubits=[0],
            amplitudes=np.array([0.5, 1.0, 1.5]), relative_amp=True,
        )
        assert cal.amplitudes[0] == pytest.approx(
            amp_cal * np.array([0.5, 1.0, 1.5])
        )

    def test_single_gate_uses_cosine_fit(self, config):
        cal = Amplitude(
            qpu=Emulator, config=config, qubits=[0],
            amplitudes=np.linspace(0., 1., 5), n_gates=1,
        )
        assert isinstance(cal._fit[0], FitCosine)

    def test_multiple_gates_use_parabola_fit(self, config):
        cal = Amplitude(
            qpu=Emulator, config=config, qubits=[0],
            amplitudes=np.linspace(0., 1., 5), n_gates=4,
        )
        assert isinstance(cal._fit[0], FitParabola)


class TestAmplitudeSimulatedRabi:

    def test_single_pulse_cosine_fit_recovers_known_frequency(self, config):
        # amp=0 -> identity (population stays in |0>); the population
        # then completes 2.5 full cosine oscillations across the
        # sweep, a few more than the single quarter-period a real
        # calibration sweep would use -- enough to check the cosine
        # fit itself, independent of any particular config value.
        freq = 2.5
        amps = np.linspace(0., 1., 41)
        cal = Amplitude(
            qpu=Emulator, config=config, qubits=[0], amplitudes=amps,
            n_gates=1, simulator=StateVectorSimulator(), n_shots=None,
        )
        cal.generate_circuits()
        for circuit, amp in zip(
            cal._circuits, cal.amplitudes[0], strict=True
        ):
            _set_x90_unitary(circuit, 2 * np.pi * freq * amp)

        # n_shots=None -> exact probabilities, no shot noise, so the
        # fit below has nothing to converge to but the analytic curve.
        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        assert cal._fit[0].fit_success
        assert cal._fit[0].fit_params['freq'].value == pytest.approx(
            freq, abs=1e-4
        )
        # newvalue = 1/freq * 0.25 (a quarter of the Rabi period).
        assert cal.calibrated_values[0] == pytest.approx(
            0.25 / freq, abs=1e-4
        )

    @pytest.fixture
    def analyzed_parabola_cal(self, config):
        """A four-pulse Amplitude calibration, run and analyzed.

        Repeating a correctly-calibrated pi/2 pulse 4 times returns
        the qubit exactly to |0> (4 x 90 deg = 360 deg), so a small
        sweep around the example config's own X90 amplitude should
        let the parabola fit recover that same value -- a check that
        the whole pipeline (not just the fit function) is wired up
        correctly, since amp_cal comes from the config, not the test.
        """
        _, amp_cal = _x90_amp_param(config, 0)
        amps = np.linspace(0.9 * amp_cal, 1.1 * amp_cal, 21)
        cal = Amplitude(
            qpu=Emulator, config=config, qubits=[0], amplitudes=amps,
            n_gates=4, simulator=StateVectorSimulator(), n_shots=None,
        )
        cal.generate_circuits()
        for circuit, amp in zip(
            cal._circuits, cal.amplitudes[0], strict=True
        ):
            _set_x90_unitary(circuit, (np.pi / 2) * (amp / amp_cal))

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()
        return cal, amp_cal

    def test_four_pulse_parabola_fit_recovers_config_amplitude(
        self, analyzed_parabola_cal
    ):
        cal, amp_cal = analyzed_parabola_cal
        assert cal._fit[0].fit_success
        assert cal.calibrated_values[0] == pytest.approx(amp_cal, rel=1e-3)

    def test_plot_runs_with_amplitude_shaped_data(
        self, analyzed_parabola_cal
    ):
        # Amplitude doesn't override plot() -- it inherits
        # Calibration.plot() (already covered branch-by-branch in
        # test_calibration.py's TestPlot), so this just checks it runs
        # against real Amplitude fit/sweep data, with the same
        # xlabel/ylabel Amplitude.run() would pass in. The Agg backend
        # is forced session-wide in conftest.py, so there's no risk of
        # a real window popping open.
        cal, _ = analyzed_parabola_cal
        cal.plot(xlabel='Amplitude (a.u.)', ylabel=r'$|0\rangle$ Population')
