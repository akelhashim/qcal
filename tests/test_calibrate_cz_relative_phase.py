"""Unit tests for qcal.calibration.cz.RelativePhase.

Same approach as RelativeAmp: the swept 'param: ...' phase column
never reaches a real gate unitary, so the CZ gate's .unitary is
hand-overwritten to simulate the conditional phase's dependence on the
drive phase:

    CZ(phase).unitary = diag(1, 1, 1, exp(i * (pi - (phase - phase_true))))

i.e. the conditional phase is exactly pi (ideal CZ) when phase equals
phase_true, and drifts linearly away from pi on either side. phase_true
is chosen to be the real, existing config value for this pulse's phase
kwarg, so recovering it is a meaningful end-to-end check, not just a
self-consistency check against an arbitrary synthetic target.
"""
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.calibration.cz import RelativePhase
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


def _cz_conditional_phase(phase, phase_true):
    """diag(1, 1, 1, exp(i * (pi - (phase - phase_true))))."""
    unitary = np.eye(4, dtype=complex)
    unitary[3, 3] = np.exp(1j * (np.pi - (phase - phase_true)))
    return unitary


class TestRelativePhaseConstruction:

    def test_default_params_path_is_target_line_only(self, config):
        cal = RelativePhase(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            phases=np.linspace(-0.1, 0.1, 5),
        )
        assert cal.params[(0, 1)] == (
            'two_qubit/(0, 1)/CZ/pulse/1/kwargs/phase'
        )

    def test_phases_used_as_is(self, config):
        phases = np.linspace(-0.1, 0.1, 5)
        cal = RelativePhase(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            phases=phases,
        )
        assert cal.phases[(0, 1)] is phases

    def test_dict_phases_per_pair(self, config):
        pair = (0, 1)
        phases = np.linspace(-0.1, 0.1, 5)
        cal = RelativePhase(
            qpu=Emulator, config=config, qubit_pairs=[pair],
            phases={pair: phases},
        )
        assert cal.phases[pair] is phases

    def test_uses_single_parabola_fit(self, config):
        cal = RelativePhase(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            phases=np.linspace(-0.1, 0.1, 5),
        )
        assert isinstance(cal._fit[(0, 1)], FitParabola)

    def test_generate_circuits_produces_four_sequences(self, config):
        phases = np.linspace(-0.1, 0.1, 5)
        cal = RelativePhase(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            phases=phases,
        )
        cal.generate_circuits()
        assert cal._circuits.n_circuits == 4 * len(phases)
        assert set(cal._circuits['sequence']) == {
            'C0_X', 'C1_X', 'C0_Y', 'C1_Y'
        }


class TestRelativePhaseSimulatedRun:

    def test_parabola_fit_recovers_config_phase(self, config):
        pair = (0, 1)
        phase_true = config['two_qubit/(0, 1)/CZ/pulse/1/kwargs/phase']
        phases = np.linspace(phase_true - 0.1, phase_true + 0.1, 21)
        cal = RelativePhase(
            qpu=Emulator, config=config, qubit_pairs=[pair],
            phases=phases, simulator=StateVectorSimulator(),
            n_shots=None,
        )
        cal.generate_circuits()
        phase_col = cal._circuits[
            'param: two_qubit/(0, 1)/CZ/pulse/1/kwargs/phase'
        ]
        for circuit, p in zip(cal._circuits, phase_col, strict=True):
            _set_cz_unitary(
                circuit, _cz_conditional_phase(p, phase_true)
            )

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        assert cal._R[pair].max() == pytest.approx(1.0, abs=1e-6)
        assert cal._fit[pair].fit_success
        assert cal.calibrated_values[pair] == pytest.approx(
            phase_true, abs=1e-6
        )
