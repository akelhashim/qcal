"""Unit tests for qcal.calibration.cz.RelativeAmp.

Like Amplitude, this sweeps a metadata-only 'param: ...' column that
never reaches a real gate unitary, so the CZ gate's .unitary is
hand-overwritten after generate_circuits() to simulate the conditional
phase's amplitude dependence:

    CZ(amp).unitary = diag(1, 1, 1, exp(i * pi * amp / amp_ideal))

Unlike Amplitude, RelativeAmp's sweep is a *ratio* applied to the
target line's amplitude, always scaled by the *control* line's real
config amplitude (amp0) -- so amp_ideal = amp0 is a real, existing
config value here, not an arbitrary synthetic one, and recovering it
is a meaningful end-to-end check.
"""
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.calibration.cz import RelativeAmp
from qcal.fitting.fit import FitParabola
from qcal.simulation.simulators import StateVectorSimulator


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


class TestRelativeAmpConstruction:

    def test_default_params_path_is_target_line_only(self, config):
        cal = RelativeAmp(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            amplitudes=np.linspace(0.8, 1.2, 5),
        )
        assert cal.params[(0, 1)] == 'two_qubit/(0, 1)/CZ/pulse/1/kwargs/amp'

    def test_amplitudes_scaled_by_control_line_amp(self, config):
        pair = (0, 1)
        amp0 = config['two_qubit/(0, 1)/CZ/pulse/0/kwargs/amp']
        ratios = np.linspace(0.8, 1.2, 5)
        cal = RelativeAmp(
            qpu=Emulator, config=config, qubit_pairs=[pair],
            amplitudes=ratios,
        )
        assert cal.amplitudes[pair] == pytest.approx(amp0 * ratios)

    def test_dict_amplitudes_scaled_per_pair(self, config):
        pair = (0, 1)
        amp0 = config['two_qubit/(0, 1)/CZ/pulse/0/kwargs/amp']
        ratios = np.linspace(0.9, 1.1, 5)
        cal = RelativeAmp(
            qpu=Emulator, config=config, qubit_pairs=[pair],
            amplitudes={pair: ratios},
        )
        assert cal.amplitudes[pair] == pytest.approx(amp0 * ratios)

    def test_uses_single_parabola_fit(self, config):
        cal = RelativeAmp(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            amplitudes=np.linspace(0.8, 1.2, 5),
        )
        assert isinstance(cal._fit[(0, 1)], FitParabola)

    def test_generate_circuits_produces_four_sequences(self, config):
        ratios = np.linspace(0.8, 1.2, 5)
        cal = RelativeAmp(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            amplitudes=ratios,
        )
        cal.generate_circuits()
        assert cal._circuits.n_circuits == 4 * len(ratios)
        assert set(cal._circuits['sequence']) == {
            'C0_X', 'C1_X', 'C0_Y', 'C1_Y'
        }


class TestRelativeAmpSimulatedRun:

    def test_parabola_fit_recovers_control_line_amplitude(self, config):
        pair = (0, 1)
        amp0 = config['two_qubit/(0, 1)/CZ/pulse/0/kwargs/amp']
        ratios = np.linspace(0.9, 1.1, 21)
        cal = RelativeAmp(
            qpu=Emulator, config=config, qubit_pairs=[pair],
            amplitudes=ratios, simulator=StateVectorSimulator(),
            n_shots=None,
        )
        cal.generate_circuits()
        amp_col = cal._circuits[
            'param: two_qubit/(0, 1)/CZ/pulse/1/kwargs/amp'
        ]
        for circuit, amp in zip(cal._circuits, amp_col, strict=True):
            _set_cz_unitary(circuit, _cz_conditional_phase(amp, amp0))

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        assert cal._R[pair].max() == pytest.approx(1.0, abs=1e-6)
        assert cal._fit[pair].fit_success
        assert cal.calibrated_values[pair] == pytest.approx(
            amp0, abs=1e-6
        )
