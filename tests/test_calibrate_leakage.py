"""Unit tests for qcal.calibration.leakage.Leakage.

Unlike the other calibration classes tested so far, Leakage takes an
arbitrary, user-supplied circuit and params/param_sweep dicts rather
than building its own circuits from a fixed gate/subspace -- so the
"gate under test" here is just whatever gate is in the circuit passed
in (an X90 on qubit 0, below). generate_circuits() writes the swept
values into metadata-only 'param: ...' columns that never reach a real
gate unitary (compiling them back into the pulse is skipped entirely
by the simulator), so the X90 gate's .unitary is hand-overwritten with
a synthetic leakage model after generate_circuits():

    X90(theta).unitary = [[cos(theta/2), 0, -sin(theta/2)],
                           [0,            1,  0           ],
                           [sin(theta/2), 0,  cos(theta/2)]]

a GE<->EF leakage rotation by angle theta -- zero leakage (pure
identity on the EF subspace) at theta=0, growing on either side. The
swept param controls theta directly, so the |2> population is a
controllable, deterministic function of the sweep value with a known
minimum, letting argmin/argmax recovery be checked precisely.
"""
import numpy as np
import pytest

import qcal.settings as settings
from qcal.backend.emulator import Emulator
from qcal.calibration.leakage import Leakage
from qcal.circuit import Circuit, Cycle
from qcal.config import Config
from qcal.gates.single_qubit import X90
from qcal.simulation.simulators import StateVectorSimulator


def _leakage_unitary(theta):
    """GE<->EF leakage rotation by angle theta."""
    unitary = np.eye(3, dtype=complex)
    unitary[0, 0] = np.cos(theta / 2)
    unitary[0, 2] = -np.sin(theta / 2)
    unitary[2, 0] = np.sin(theta / 2)
    unitary[2, 2] = np.cos(theta / 2)
    return unitary


def _set_x90_unitary(circuit, unitary):
    """Overwrite every X90 gate's unitary in a circuit."""
    for cyc in circuit:
        if cyc.is_barrier:
            continue
        for gate in cyc:
            if gate.name == 'X90':
                gate.unitary = unitary


@pytest.fixture
def amplifying_circuit():
    circuit = Circuit([Cycle({X90(0)})])
    circuit.measure()
    return circuit


@pytest.fixture(autouse=True)
def _restore_save_data():
    original = settings.Settings.save_data
    yield
    settings.Settings.save_data = original


class TestLeakageConstruction:

    def test_qubits_derived_from_params_keys(self, config, amplifying_circuit):
        cal = Leakage(
            qpu=Emulator, config=config, circuit=amplifying_circuit,
            params={0: 'dummy/param'},
            param_sweep={0: np.linspace(-1, 1, 5)},
        )
        assert cal._qubits == [0]

    def test_circuit_property(self, config, amplifying_circuit):
        cal = Leakage(
            qpu=Emulator, config=config, circuit=amplifying_circuit,
            params={0: 'dummy/param'},
            param_sweep={0: np.linspace(-1, 1, 5)},
        )
        assert cal.circuit is amplifying_circuit

    def test_default_maximize_is_false(self, config, amplifying_circuit):
        cal = Leakage(
            qpu=Emulator, config=config, circuit=amplifying_circuit,
            params={0: 'dummy/param'},
            param_sweep={0: np.linspace(-1, 1, 5)},
        )
        assert cal._maximize is False

    def test_generate_circuits_copies_circuit_per_sweep_point(
        self, config, amplifying_circuit
    ):
        sweep = np.linspace(-1, 1, 5)
        cal = Leakage(
            qpu=Emulator, config=config, circuit=amplifying_circuit,
            params={0: 'dummy/param'}, param_sweep={0: sweep},
        )
        cal.generate_circuits()
        assert cal._circuits.n_circuits == len(sweep)
        assert list(cal._circuits['param: dummy/param']) == list(sweep)

    def test_generate_circuits_writes_each_param_in_a_list(
        self, config, amplifying_circuit
    ):
        sweep = np.linspace(-1, 1, 5)
        cal = Leakage(
            qpu=Emulator, config=config, circuit=amplifying_circuit,
            params={0: ['dummy/param1', 'dummy/param2']},
            param_sweep={0: sweep},
        )
        cal.generate_circuits()
        assert list(cal._circuits['param: dummy/param1']) == list(sweep)
        assert list(cal._circuits['param: dummy/param2']) == list(sweep)


class TestLeakageSimulatedRun:

    def test_argmin_recovers_zero_leakage_point(
        self, config, amplifying_circuit
    ):
        sweep = np.linspace(-1.0, 1.0, 21)
        cal = Leakage(
            qpu=Emulator, config=config, circuit=amplifying_circuit,
            params={0: 'dummy/param'}, param_sweep={0: sweep},
            simulator=StateVectorSimulator(), n_shots=None, n_levels=3,
        )
        cal.generate_circuits()
        param_col = cal._circuits['param: dummy/param']
        for circuit, theta in zip(cal._circuits, param_col, strict=True):
            _set_x90_unitary(circuit, _leakage_unitary(theta))

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        assert cal._sweep_results[0][len(sweep) // 2] == pytest.approx(
            0.0, abs=1e-9
        )
        assert cal.calibrated_values[0] == pytest.approx(0.0, abs=1e-9)
        assert cal.loss[0][0] == pytest.approx(0.0, abs=1e-9)

    def test_maximize_recovers_peak_leakage_point(
        self, config, amplifying_circuit
    ):
        theta_peak = 0.5
        sweep = np.linspace(-1.0, 1.0, 21)
        cal = Leakage(
            qpu=Emulator, config=config, circuit=amplifying_circuit,
            params={0: 'dummy/param'}, param_sweep={0: sweep},
            simulator=StateVectorSimulator(), n_shots=None, n_levels=3,
            maximize=True,
        )
        cal.generate_circuits()
        param_col = cal._circuits['param: dummy/param']
        for circuit, theta in zip(cal._circuits, param_col, strict=True):
            # Peaks (full leakage, theta_eff = pi/2) at theta_peak,
            # falling off linearly on either side.
            theta_eff = (np.pi / 2) * (1 - abs(theta - theta_peak))
            _set_x90_unitary(circuit, _leakage_unitary(theta_eff))

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        assert cal.calibrated_values[0] == pytest.approx(theta_peak)

    def test_grouped_qubits_sum_leakage_across_qubits(self, config):
        circuit = Circuit([Cycle({X90(0), X90(1)})])
        circuit.measure()
        sweep = np.linspace(-1.0, 1.0, 9)
        cal = Leakage(
            qpu=Emulator, config=config, circuit=circuit,
            params={(0, 1): 'dummy/param'}, param_sweep={(0, 1): sweep},
            simulator=StateVectorSimulator(), n_shots=None, n_levels=3,
        )
        cal.generate_circuits()
        param_col = cal._circuits['param: dummy/param']
        for circ, theta in zip(cal._circuits, param_col, strict=True):
            _set_x90_unitary(circ, _leakage_unitary(theta))

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        # Both qubits leak identically, so the summed loss at theta=0
        # should be exactly zero, and strictly positive elsewhere.
        results = cal._sweep_results[(0, 1)]
        zero_idx = len(sweep) // 2
        assert results[zero_idx] == pytest.approx(0.0, abs=1e-9)
        assert all(
            r > 0 for i, r in enumerate(results) if i != zero_idx
        )
        assert cal.calibrated_values[(0, 1)] == pytest.approx(0.0, abs=1e-9)


class TestLeakageFinal:

    def test_final_sets_each_param_in_a_list(self, config, amplifying_circuit):
        cal = Leakage(
            qpu=Emulator, config=config, circuit=amplifying_circuit,
            params={0: ['dummy/param1', 'dummy/param2']},
            param_sweep={0: np.linspace(-1, 1, 5)},
        )
        cal._cal_values[0] = 0.123
        calls = []
        cal.set_param = lambda p, v: calls.append((p, v))
        cal.final()
        assert calls == [
            ('dummy/param1', 0.123), ('dummy/param2', 0.123)
        ]

    def test_final_saves_config_when_save_data_enabled(
        self, config, amplifying_circuit, monkeypatch
    ):
        cal = Leakage(
            qpu=Emulator, config=config, circuit=amplifying_circuit,
            params={0: 'dummy/param'},
            param_sweep={0: np.linspace(-1, 1, 5)},
        )
        saved = []
        monkeypatch.setattr(
            Config, 'save', lambda self, *a, **k: saved.append(True)
        )
        settings.Settings.save_data = True
        cal.final()
        assert saved == [True]

    def test_final_does_not_save_config_when_save_data_disabled(
        self, config, amplifying_circuit, monkeypatch
    ):
        cal = Leakage(
            qpu=Emulator, config=config, circuit=amplifying_circuit,
            params={0: 'dummy/param'},
            param_sweep={0: np.linspace(-1, 1, 5)},
        )
        saved = []
        monkeypatch.setattr(
            Config, 'save', lambda self, *a, **k: saved.append(True)
        )
        settings.Settings.save_data = False
        cal.final()
        assert saved == []
