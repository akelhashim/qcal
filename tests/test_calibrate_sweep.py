"""Unit tests for qcal.calibration.sweep.ParamSweep.

Like Leakage, ParamSweep takes an arbitrary, user-supplied circuit and
params/param_sweep dicts rather than building its own circuits from a
fixed gate/subspace -- so the "gate under test" here is just whatever
gate is in the circuit passed in (an X90 on qubit 0, or on qubits 0
and 1, below). generate_circuits() writes the swept values into
metadata-only 'param: ...' columns that never reach a real gate
unitary (compiling them back into the pulse is skipped entirely by the
simulator), so the X90 gate's .unitary is hand-overwritten after
generate_circuits() with rx(theta), where theta is the swept value
itself. This turns the sweep into a simple Rabi oscillation: the |1>
(or |11>) population traces out a cosine, letting argmax/argmin
recovery of population-based criteria be checked precisely -- e.g. the
population of '1' peaks at exactly theta = pi.
"""
import numpy as np
import pytest

import qcal.settings as settings
from qcal.backend.emulator import Emulator
from qcal.calibration.sweep import ParamSweep
from qcal.circuit import Circuit, Cycle
from qcal.config import Config
from qcal.gates.single_qubit import X90, rx
from qcal.simulation.simulators import StateVectorSimulator


def _set_x90_unitary(circuit, theta):
    """Overwrite every X90 gate's unitary in a circuit with rx(theta)."""
    for cyc in circuit:
        if cyc.is_barrier:
            continue
        for gate in cyc:
            if gate.name == 'X90':
                gate.unitary = rx(theta)


@pytest.fixture
def rabi_circuit():
    circuit = Circuit([Cycle({X90(0)})])
    circuit.measure()
    return circuit


@pytest.fixture(autouse=True)
def _restore_save_data():
    original = settings.Settings.save_data
    yield
    settings.Settings.save_data = original


class TestParamSweepConstruction:

    def test_no_maximize_or_minimize_raises(self, config, rabi_circuit):
        with pytest.raises(ValueError):
            ParamSweep(
                qpu=Emulator, config=config, circuit=rabi_circuit,
                params={0: 'dummy/param'},
                param_sweep={0: np.linspace(-1, 1, 5)},
            )

    def test_qubits_derived_from_params_keys(self, config, rabi_circuit):
        cal = ParamSweep(
            qpu=Emulator, config=config, circuit=rabi_circuit,
            params={0: 'dummy/param'},
            param_sweep={0: np.linspace(-1, 1, 5)}, maximize=['1'],
        )
        assert cal._qubits == [0]

    def test_circuit_property(self, config, rabi_circuit):
        cal = ParamSweep(
            qpu=Emulator, config=config, circuit=rabi_circuit,
            params={0: 'dummy/param'},
            param_sweep={0: np.linspace(-1, 1, 5)}, maximize=['1'],
        )
        assert cal.circuit is rabi_circuit

    def test_generate_circuits_copies_circuit_per_sweep_point(
        self, config, rabi_circuit
    ):
        sweep = np.linspace(-1, 1, 5)
        cal = ParamSweep(
            qpu=Emulator, config=config, circuit=rabi_circuit,
            params={0: 'dummy/param'}, param_sweep={0: sweep},
            maximize=['1'],
        )
        cal.generate_circuits()
        assert cal._circuits.n_circuits == len(sweep)
        assert list(cal._circuits['param: dummy/param']) == list(sweep)

    def test_generate_circuits_writes_each_param_in_a_list(
        self, config, rabi_circuit
    ):
        sweep = np.linspace(-1, 1, 5)
        cal = ParamSweep(
            qpu=Emulator, config=config, circuit=rabi_circuit,
            params={0: ['dummy/param1', 'dummy/param2']},
            param_sweep={0: sweep}, maximize=['1'],
        )
        cal.generate_circuits()
        assert list(cal._circuits['param: dummy/param1']) == list(sweep)
        assert list(cal._circuits['param: dummy/param2']) == list(sweep)


class TestParamSweepSimulatedRun:

    def test_maximize_only_recovers_population_peak(
        self, config, rabi_circuit
    ):
        sweep = np.linspace(0.0, 2 * np.pi, 41)
        cal = ParamSweep(
            qpu=Emulator, config=config, circuit=rabi_circuit,
            params={0: 'dummy/param'}, param_sweep={0: sweep},
            maximize=['1'],
            simulator=StateVectorSimulator(), n_shots=None,
        )
        cal.generate_circuits()
        param_col = cal._circuits['param: dummy/param']
        for circuit, theta in zip(cal._circuits, param_col, strict=True):
            _set_x90_unitary(circuit, theta)

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        # rx(theta) applied to |0> gives Prob(1) = sin^2(theta/2),
        # which peaks at theta = pi -- the sweep midpoint.
        assert np.argmax(
            cal.populations[0]['1']
        ) == len(sweep) // 2
        assert cal.calibrated_values[0] == pytest.approx(np.pi)

    def test_minimize_only_recovers_population_trough(
        self, config, rabi_circuit
    ):
        sweep = np.linspace(0.0, 2 * np.pi, 41)
        cal = ParamSweep(
            qpu=Emulator, config=config, circuit=rabi_circuit,
            params={0: 'dummy/param'}, param_sweep={0: sweep},
            minimize=['0'],
            simulator=StateVectorSimulator(), n_shots=None,
        )
        cal.generate_circuits()
        param_col = cal._circuits['param: dummy/param']
        for circuit, theta in zip(cal._circuits, param_col, strict=True):
            _set_x90_unitary(circuit, theta)

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        # Prob(0) = cos^2(theta/2), which is minimized at theta = pi.
        assert np.argmin(
            cal.populations[0]['0']
        ) == len(sweep) // 2
        assert cal.calibrated_values[0] == pytest.approx(np.pi)

    def test_combined_maximize_and_minimize_average_to_same_point(
        self, config, rabi_circuit
    ):
        # Prob(0) and Prob(1) both single out theta = pi (trough and
        # peak, respectively), so averaging the two criteria should
        # land on exactly the same value as either alone.
        sweep = np.linspace(0.0, 2 * np.pi, 41)
        cal = ParamSweep(
            qpu=Emulator, config=config, circuit=rabi_circuit,
            params={0: 'dummy/param'}, param_sweep={0: sweep},
            maximize=['1'], minimize=['0'],
            simulator=StateVectorSimulator(), n_shots=None,
        )
        cal.generate_circuits()
        param_col = cal._circuits['param: dummy/param']
        for circuit, theta in zip(cal._circuits, param_col, strict=True):
            _set_x90_unitary(circuit, theta)

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        assert cal.calibrated_values[0] == pytest.approx(np.pi)

    def test_grouped_qubits_use_joint_dit_strings(self, config):
        circuit = Circuit([Cycle({X90(0), X90(1)})])
        circuit.measure()
        sweep = np.linspace(0.0, 2 * np.pi, 41)
        cal = ParamSweep(
            qpu=Emulator, config=config, circuit=circuit,
            params={(0, 1): 'dummy/param'}, param_sweep={(0, 1): sweep},
            maximize=['11'],
            simulator=StateVectorSimulator(), n_shots=None,
        )
        cal.generate_circuits()
        param_col = cal._circuits['param: dummy/param']
        for circ, theta in zip(cal._circuits, param_col, strict=True):
            _set_x90_unitary(circ, theta)

        Emulator.run(cal, cal._circuits, save=False)
        cal.analyze()

        # Both qubits rotate identically, so Prob(11) = sin^4(theta/2),
        # still peaking at theta = pi.
        assert np.argmax(
            cal.populations[(0, 1)]['11']
        ) == len(sweep) // 2
        assert cal.calibrated_values[(0, 1)] == pytest.approx(np.pi)


class TestParamSweepFinal:

    def test_final_sets_each_param_in_a_list(self, config, rabi_circuit):
        cal = ParamSweep(
            qpu=Emulator, config=config, circuit=rabi_circuit,
            params={0: ['dummy/param1', 'dummy/param2']},
            param_sweep={0: np.linspace(0, 1, 5)}, maximize=['1'],
        )
        cal._cal_values[0] = 0.123
        calls = []
        cal.set_param = lambda p, v: calls.append((p, v))
        cal.final()
        assert calls == [
            ('dummy/param1', 0.123), ('dummy/param2', 0.123)
        ]

    def test_final_saves_config_when_save_data_enabled(
        self, config, rabi_circuit, monkeypatch
    ):
        cal = ParamSweep(
            qpu=Emulator, config=config, circuit=rabi_circuit,
            params={0: 'dummy/param'},
            param_sweep={0: np.linspace(0, 1, 5)}, maximize=['1'],
        )
        saved = []
        monkeypatch.setattr(
            Config, 'save', lambda self, *a, **k: saved.append(True)
        )
        settings.Settings.save_data = True
        cal.final()
        assert saved == [True]

    def test_final_does_not_save_config_when_save_data_disabled(
        self, config, rabi_circuit, monkeypatch
    ):
        cal = ParamSweep(
            qpu=Emulator, config=config, circuit=rabi_circuit,
            params={0: 'dummy/param'},
            param_sweep={0: np.linspace(0, 1, 5)}, maximize=['1'],
        )
        saved = []
        monkeypatch.setattr(
            Config, 'save', lambda self, *a, **k: saved.append(True)
        )
        settings.Settings.save_data = False
        cal.final()
        assert saved == []
