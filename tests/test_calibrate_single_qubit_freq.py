"""Unit tests for qcal.calibration.single_qubit.Frequency.

Unlike Amplitude, no gate-unitary mocking is needed here at all.
generate_circuits() injects each artificial detuning as a purely
virtual Rz phase (phase = 2*pi*detuning*t) rather than any physical
frequency/Hamiltonian evolution -- Idle is a true no-op unitary (see
qcal.gates.single_qubit.Idle), and StateVectorSimulator adds no noise.
So the ideal circuits, run as generated, already represent a qubit
with *zero* real detuning from its configured frequency: the measured
detuning for every artificial detuning point should come back exactly
equal to that artificial value, and the fitted calibrated frequency
should land back on the exact frequency already in the example config.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.calibration.single_qubit import Frequency
from qcal.simulation.simulators import StateVectorSimulator
from qcal.units import MHz, us


@pytest.fixture(autouse=True)
def _close_figures():
    """Prevent matplotlib's open-figure count from growing unbounded."""
    yield
    plt.close('all')


class TestFrequencyConstruction:

    def test_default_params_path_is_derived_from_config(self, config):
        cal = Frequency(qpu=Emulator, config=config, qubits=[0])
        assert cal.params[0] == 'single_qubit/0/GE/freq'

    def test_invalid_subspace_raises(self, config):
        with pytest.raises(ValueError):
            Frequency(
                qpu=Emulator, config=config, qubits=[0], subspace='bogus',
            )

    def test_default_detunings(self, config):
        cal = Frequency(qpu=Emulator, config=config, qubits=[0])
        assert cal.detunings == pytest.approx(
            np.array([-2, -1, 1, 2]) * MHz
        )

    def test_custom_detunings_and_time_sweep(self, config):
        detunings = np.array([-1, 1]) * MHz
        cal = Frequency(
            qpu=Emulator, config=config, qubits=[0], detunings=detunings,
            t_max=1 * us, n_elements=10,
        )
        assert cal.detunings is detunings
        assert cal.times[0].size == 10
        assert cal.times[0][0] == 0.
        assert cal.times[0][-1] == pytest.approx(1 * us)


class TestFrequencySimulatedRamsey:

    def test_recovers_artificial_detunings_and_config_frequency(
        self, config
    ):
        # No unitary injection needed (unlike Amplitude): the
        # "detuning" is entirely the Rz phase already baked into the
        # circuit by generate_circuits(), so a plain cal.run() -- with
        # its own generate_circuits()/analyze()/plot()/final() -- is
        # exactly what we want to exercise here.
        freq_true = config['single_qubit/0/GE/freq']
        cal = Frequency(
            qpu=Emulator, config=config, qubits=[0],
            simulator=StateVectorSimulator(), n_shots=None,
        )
        cal.run()

        # Each per-detuning Ramsey fringe recovers |detuning| exactly.
        assert cal._freqs[0] == pytest.approx(
            np.abs(cal.detunings), abs=1.0
        )
        # The measured-vs-artificial-detuning fit finds zero offset,
        # so the calibrated frequency is exactly the config's own.
        assert cal._fit[0].fit_success
        assert cal.calibrated_values[0] == pytest.approx(
            freq_true, abs=1e-2
        )
