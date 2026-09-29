"""Unit tests for qcal.calibration.single_qubit.Phase.

Both simulated-recovery tests inject a synthetic AC-Stark phase error
by hand-overwriting gate unitaries after generate_circuits() (compiling
the 'param: ...' sweep columns back into real pulse phases is skipped
entirely by the simulator), using the following physical model:

- gate='X': there is only one phase kwarg (pulse segment 1), applied
  *after* the physical X pulse: the compiled pulse is XZ, i.e.
  gate.unitary = Rz(phase + delta_true) @ X, where delta_true is the
  injected AC-Stark error. This is a single-sided dressing (not a
  conjugation), and the cosine fit recovers -delta_true (the phase
  that cancels it) to near machine precision.

- gate='X90': a Stark-shift-induced phase error theta
  accumulated *during* the pulse dresses it symmetrically on both
  sides, X90_physical = Z(theta/2) X90 Z(theta/2) (same sign on both
  sides -- not a conjugation). The software-swept compensation phi is
  applied the same way: X90_compiled(phi) = Z(phi) X90_physical
  Z(phi) = Z(phi + theta/2) X90 Z(phi + theta/2), which reduces to the
  ideal X90 when phi = -theta/2. This symmetric, same-sign dressing is
  what makes the two AllXY-style sequences (Y180_X90/X180_Y90) respond
  linearly with equal and opposite slope, crossing at phi = -theta/2 --
  confirmed empirically (opposite-sign linear fits) and only accurate
  to the model's leading order in phi, so recovery has a small,
  shrinking-with-window systematic bias rather than being exact.
"""
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.calibration.single_qubit import Phase
from qcal.fitting.fit import FitCosine, FitLinear
from qcal.gates.single_qubit import rx, rz
from qcal.simulation.simulators import StateVectorSimulator


def _set_gate_unitary(circuit, gate_name, unitary):
    """Overwrite every gate_name gate's unitary in a circuit."""
    for cyc in circuit:
        if cyc.is_barrier:
            continue
        for gate in cyc:
            if gate.name == gate_name:
                gate.unitary = unitary


class TestPhaseConstruction:

    def test_default_params_path_for_x90(self, config):
        cal = Phase(
            qpu=Emulator, config=config, qubits=[0],
            phases=np.linspace(-1, 1, 5), gate='X90',
        )
        assert cal.params[0] == [
            'single_qubit/0/GE/X90/pulse/0/kwargs/phase',
            'single_qubit/0/GE/X90/pulse/2/kwargs/phase',
        ]

    def test_default_params_path_for_x(self, config):
        cal = Phase(
            qpu=Emulator, config=config, qubits=[0],
            phases=np.linspace(-1, 1, 5), gate='X',
        )
        assert cal.params[0] == 'single_qubit/0/GE/X/pulse/1/kwargs/phase'

    def test_invalid_gate_raises(self, config):
        with pytest.raises(ValueError):
            Phase(
                qpu=Emulator, config=config, qubits=[0],
                phases=np.linspace(-1, 1, 5), gate='Y90',
            )

    def test_invalid_subspace_raises(self, config):
        with pytest.raises(ValueError):
            Phase(
                qpu=Emulator, config=config, qubits=[0],
                phases=np.linspace(-1, 1, 5), subspace='bogus',
            )

    def test_x90_uses_two_linear_fits(self, config):
        cal = Phase(
            qpu=Emulator, config=config, qubits=[0],
            phases=np.linspace(-1, 1, 5), gate='X90',
        )
        assert [type(f) for f in cal._fit[0]] == [FitLinear, FitLinear]

    def test_x_uses_cosine_fit(self, config):
        cal = Phase(
            qpu=Emulator, config=config, qubits=[0],
            phases=np.linspace(-1, 1, 5), gate='X',
        )
        assert isinstance(cal._fit[0], FitCosine)

    def test_x90_generates_two_sequences_per_phase(self, config):
        phases = np.linspace(-1, 1, 5)
        cal = Phase(
            qpu=Emulator, config=config, qubits=[0], phases=phases,
            gate='X90',
        )
        cal.generate_circuits()
        assert cal._circuits.n_circuits == 2 * len(phases)
        assert set(cal._circuits['sequence']) == {'Y180_X90', 'X180_Y90'}

    def test_x_generates_one_sequence_per_phase(self, config):
        phases = np.linspace(-1, 1, 5)
        cal = Phase(
            qpu=Emulator, config=config, qubits=[0], phases=phases,
            gate='X',
        )
        cal.generate_circuits()
        assert cal._circuits.n_circuits == len(phases)
        assert set(cal._circuits['sequence']) == {'X90_X_X90'}


class TestKnownIssues:

    @pytest.mark.xfail(
        reason=(
            "relative_phase=True crashes for gate='X': self._params[q] "
            "is a plain string for gate='X' (only gate='X90' gives a "
            "2-element list), so self._params[q][0] indexes the "
            "string's first character instead of the param path, and "
            "config[...] on that bogus path returns None, which then "
            "fails to add to a float."
        ),
        raises=TypeError,
        strict=True,
    )
    def test_relative_phase_with_x_gate(self, config):
        Phase(
            qpu=Emulator, config=config, qubits=[0],
            phases=np.array([0.1, 0.2]), gate='X', relative_phase=True,
        )

    @pytest.mark.xfail(
        reason=(
            "Phase(gate='X').run() crashes on the example config: the "
            "default params path hardcodes pulse segment index 1 "
            "('single_qubit/.../X/pulse/1/kwargs/phase'), but the example "
            "config's X pulse has only one segment (index 0). final() "
            "then fails to write the calibrated phase back, since "
            "config[bogus_path] resolves to None."
        ),
        raises=TypeError,
        strict=True,
    )
    def test_run_with_x_gate(self, config):
        phases = np.linspace(-np.pi, np.pi, 5)
        cal = Phase(
            qpu=Emulator, config=config, qubits=[0], phases=phases,
            gate='X', simulator=StateVectorSimulator(), n_shots=None,
        )
        cal.run()


class TestPhaseSimulatedRun:

    def test_x_gate_cosine_fit_recovers_known_phase_error(self, config):
        # X90 - X(phase) - X90 - measure. The X gate has only one
        # phase kwarg, applied *after* the physical pulse (XZ), so
        # gate.unitary = Rz(phase + delta_true) @ X -- a single-sided
        # dressing, not a symmetric conjugation. <Z> traces a clean
        # cosine in phase, and the fit recovers the compensating phase
        # -delta_true with no leading-order approximation involved.
        delta_true = 0.3
        phases = np.linspace(-np.pi, np.pi, 21)
        cal = Phase(
            qpu=Emulator, config=config, qubits=[0], phases=phases,
            gate='X', simulator=StateVectorSimulator(), n_shots=None,
        )
        cal.generate_circuits()
        for circuit, p in zip(cal._circuits, cal.phases[0], strict=True):
            _set_gate_unitary(
                circuit, 'X', rz(p + delta_true) @ rx(np.pi)
            )

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        assert cal._fit[0].fit_success
        assert cal.calibrated_values[0] == pytest.approx(
            -delta_true, abs=1e-3
        )

    def test_x90_linear_fit_recovers_stark_phase_error(self, config):
        # A Stark phase error theta dresses the physical X90
        # symmetrically on *both* sides with the *same* sign,
        # X90_physical = Z(theta/2) X90 Z(theta/2). The swept software
        # compensation phi combines with it as Z(phi + theta/2) X90
        # Z(phi + theta/2), which is ideal when phi = -theta/2 --
        # that's the value the two opposite-slope linear fits should
        # cross at.
        half_theta_true = 0.01
        phases = np.linspace(-0.05, 0.05, 21)
        cal = Phase(
            qpu=Emulator, config=config, qubits=[0], phases=phases,
            gate='X90', simulator=StateVectorSimulator(), n_shots=None,
        )
        cal.generate_circuits()
        phase_col = cal._circuits[
            'param: single_qubit/0/GE/X90/pulse/0/kwargs/phase'
        ]
        for circuit, phi in zip(cal._circuits, phase_col, strict=True):
            psi = phi + half_theta_true
            _set_gate_unitary(
                circuit, 'X90', rz(psi) @ rx(np.pi / 2) @ rz(psi)
            )

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        # Equal and opposite first-order sensitivity to the residual
        # phase.
        assert cal._fit[0][0].fit_success
        assert cal._fit[0][1].fit_success
        m0 = cal._fit[0][0].fit_params['m'].value
        m1 = cal._fit[0][1].fit_params['m'].value
        assert m0 == pytest.approx(-m1, rel=1e-3)

        # The crossing point recovers -theta/2; only accurate to the
        # model's leading order in phi, hence the looser tolerance
        # than the (exact) X-gate case above.
        assert cal.calibrated_values[0][0] == pytest.approx(
            -half_theta_true, abs=5e-4
        )
