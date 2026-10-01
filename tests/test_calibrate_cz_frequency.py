"""Unit tests for qcal.calibration.cz.Frequency.

Unlike Amplitude/RelativeAmp/RelativePhase, Frequency's analyze() has
no fit at all (it's commented out in the source) -- the calibrated
value is simply the swept frequency at argmax(conditionality). So the
simulated-recovery test only needs the conditionality to *peak* at the
true frequency, not to trace any particular curve shape: the CZ gate's
.unitary is hand-overwritten with

    CZ(freq).unitary = diag(1, 1, 1, exp(i * (pi - k * |freq - freq_true|)))

which peaks (conditional phase exactly pi) at freq = freq_true and
falls off symmetrically either side. freq_true is chosen to land
exactly on a sampled sweep point (an odd-length, symmetric detuning
sweep via relative_freq=True, centered on 0) so the argmax recovery is
an exact lookup, not subject to any interpolation/fit precision.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.calibration.cz import Frequency
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


def _cz_conditional_phase(freq, freq_true, k):
    """diag(1, 1, 1, exp(i * (pi - k * |freq - freq_true|)))."""
    unitary = np.eye(4, dtype=complex)
    unitary[3, 3] = np.exp(1j * (np.pi - k * abs(freq - freq_true)))
    return unitary


class TestFrequencyConstruction:

    def test_default_params_path(self, config):
        cal = Frequency(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            frequencies=np.linspace(-1e6, 1e6, 5),
        )
        assert cal.params[(0, 1)] == 'two_qubit/(0, 1)/CZ/freq'

    def test_frequencies_used_as_is(self, config):
        freqs = np.linspace(-1e6, 1e6, 5)
        cal = Frequency(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            frequencies=freqs,
        )
        assert cal.frequencies[(0, 1)] is freqs

    def test_relative_freq_adds_current_config_frequency(self, config):
        pair = (0, 1)
        freq_true = config['two_qubit/(0, 1)/CZ/freq']
        detunings = np.linspace(-1e6, 1e6, 5)
        cal = Frequency(
            qpu=Emulator, config=config, qubit_pairs=[pair],
            frequencies=detunings, relative_freq=True,
        )
        assert cal.frequencies[pair] == pytest.approx(
            freq_true + detunings
        )

    def test_generate_circuits_produces_four_sequences(self, config):
        freqs = np.linspace(-1e6, 1e6, 5)
        cal = Frequency(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            frequencies=freqs,
        )
        cal.generate_circuits()
        assert cal._circuits.n_circuits == 4 * len(freqs)
        assert set(cal._circuits['sequence']) == {
            'C0_X', 'C1_X', 'C0_Y', 'C1_Y'
        }


class TestFrequencySimulatedRun:

    @pytest.fixture
    def analyzed_cal(self, config):
        """A Frequency calibration, run against the synthetic CZ
        conditional-phase model and analyzed.
        """
        pair = (0, 1)
        freq_true = config['two_qubit/(0, 1)/CZ/freq']
        # An odd-length, zero-centered detuning sweep puts freq_true
        # exactly on a sampled point.
        detunings = np.linspace(-1e6, 1e6, 21)
        cal = Frequency(
            qpu=Emulator, config=config, qubit_pairs=[pair],
            frequencies=detunings, relative_freq=True,
            simulator=StateVectorSimulator(), n_shots=None,
        )
        cal.generate_circuits()
        freq_col = cal._circuits['param: two_qubit/(0, 1)/CZ/freq']
        k = np.pi / (2 * 1e6)
        for circuit, freq in zip(cal._circuits, freq_col, strict=True):
            _set_cz_unitary(
                circuit, _cz_conditional_phase(freq, freq_true, k)
            )

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()
        return cal, pair, freq_true, detunings

    def test_argmax_recovers_config_frequency(self, analyzed_cal):
        cal, pair, freq_true, detunings = analyzed_cal
        assert cal._R[pair].argmax() == len(detunings) // 2
        assert cal.calibrated_values[pair] == pytest.approx(freq_true)

    def test_plot_runs_with_frequency_shaped_data(self, analyzed_cal):
        # cz.Frequency defines its own plot() (no fit line, since
        # analyze() has none for this class -- see module docstring),
        # so it isn't covered by test_calibration.py's TestPlot. This
        # just checks it runs against real conditionality data. The
        # Agg backend is forced session-wide in conftest.py, so
        # there's no risk of a real window popping open.
        cal, *_ = analyzed_cal
        cal.plot()
