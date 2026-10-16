"""Unit tests for qcal.characterization.coherence (T1, T2, T2DD,
ParityOscillations).

T1/T2/T2DD each fit an Idle-time sweep to a decay curve, so a
meaningful test has to inject a *duration-dependent* relaxation
channel into the Emulator's DensityMatrixSimulator -- but
qcal.simulation.error_models.RelaxationNoise bakes a single fixed
duration into its per-category channel at construction time
(RelaxationParams.t), so every gate in a category gets identical decay
regardless of its own actual duration. That's unusable here: it would
apply the exact same amount of relaxation to the Idle(duration=0)
circuit and the Idle(duration=t_max) circuit, producing a flat (not
decaying) curve across the whole sweep.

_DurationAwareRelaxation below works around this by overriding
channel_for_gate() (checked before the static channel_for()) to build
a fresh quax.channels.thermal_relaxation() channel from each gate's
own actual time: an Idle gate's own `duration` param, or a fixed
native gate_time for the single-qubit rotation gates making up state
prep/basis change/DD pulses. The gate_time term matters most for
T2DD: a DD sequence with many pulses can spend a large, sweep-
dependent fraction of its nominal total_time actually executing X90/
Y90 pulses rather than idling (confirmed by inspecting a generated
T2DD circuit: at a mid-sweep point, true Idle time was ~3% of the
nominal sweep value), so without gate-time-scaled decay the recovered
decay constant is essentially random. T2DD's default n_pulses=None
also auto-scales the pulse count with each sweep point's own total_time,
which independently breaks the nominal-time/actual-gate-time
correspondence (more pulses at larger t means proportionally more
pulse-time overhead, not just more idle time) -- TestT2DDSimulatedRun
pins n_pulses=4 to remove that confound, recovering tphi as tightly as
T1/T2 do.

ParityOscillations.analyze()'s fit-and-fidelity block is wrapped in a
try/except, so a failed fit leaves self._fidelity as None (logged, not
raised) instead of crashing analyze()/final()/plot().
TestParityOscillationsSimulatedRun checks both the success path (exact
fidelity recovery for a noiseless Bell state) and that a forced fit
failure degrades gracefully.
"""
import matplotlib
import numpy as np
import pytest
import quax

from qcal.backend.emulator import Emulator
from qcal.characterization.coherence import T1, T2, T2DD, ParityOscillations
from qcal.fitting.fit import FitDecayingCosine, FitExponential
from qcal.simulation.error_models import ErrorModel
from qcal.simulation.simulators import DensityMatrixSimulator, StateVectorSimulator
from qcal.units import kHz, us

matplotlib.use('Agg')

GATE_TIME = 30e-9  # native X90 time on the example config


class _DurationAwareRelaxation(ErrorModel):
    """T1 + dephasing relaxation scaled to each gate's own duration.

    See the module docstring for why this is needed instead of
    RelaxationNoise.
    """

    def __init__(self, t1, tphi, gate_time=GATE_TIME, p1=0.0):
        super().__init__()
        self._t1 = t1
        self._tphi = tphi
        self._p1 = p1
        self._gate_time = gate_time

    def channel_for(self, gate_name):
        return None

    def channel_for_gate(self, gate):
        name = type(gate).__name__
        if name == 'Idle':
            duration = gate.properties['params'].get('duration', 0.0)
        elif name in ('X90', 'Y90', 'X', 'Y', 'Z'):
            duration = self._gate_time
        else:
            return None
        if duration <= 0:
            return None
        return quax.channels.thermal_relaxation(
            self._t1, self._tphi, self._p1, duration
        )


class TestT1Construction:

    def test_invalid_gate_raises(self, config):
        with pytest.raises(ValueError):
            T1(qpu=Emulator, config=config, qubits=[0], gate='bogus')

    def test_invalid_subspace_raises(self, config):
        with pytest.raises(ValueError):
            T1(qpu=Emulator, config=config, qubits=[0], subspace='bogus')

    def test_default_params_path(self, config):
        exp = T1(qpu=Emulator, config=config, qubits=[0, 1])
        assert exp.params == {
            0: 'single_qubit/0/GE/T1', 1: 'single_qubit/1/GE/T1'
        }

    def test_times_sweep_from_zero_to_t_max(self, config):
        exp = T1(qpu=Emulator, config=config, qubits=[0], t_max=50 * us)
        assert exp.times[0][0] == 0.0
        assert exp.times[0][-1] == pytest.approx(50 * us)


class TestT1SimulatedRun:

    def test_recovers_known_t1_with_x90_prep(self, config):
        t1_true = 20 * us
        noise = _DurationAwareRelaxation(t1=t1_true, tphi=1000 * t1_true)
        sim = DensityMatrixSimulator(noise_model=noise)

        np.random.seed(0)
        exp = T1(
            qpu=Emulator, config=config, qubits=[0], t_max=100 * us,
            n_elements=30, gate='X90', simulator=sim, n_shots=2000,
        )
        exp.generate_circuits()
        Emulator.run(exp, exp.circuits, save=False)
        exp.analyze()

        assert exp._fit[0].fit_success
        assert exp.characterized_values[0] == pytest.approx(
            t1_true, rel=0.1
        )

    def test_recovers_known_t1_with_x_prep(self, config):
        t1_true = 20 * us
        noise = _DurationAwareRelaxation(t1=t1_true, tphi=1000 * t1_true)
        sim = DensityMatrixSimulator(noise_model=noise)

        np.random.seed(5)
        exp = T1(
            qpu=Emulator, config=config, qubits=[0], t_max=100 * us,
            n_elements=30, gate='X', simulator=sim, n_shots=2000,
        )
        exp.generate_circuits()
        Emulator.run(exp, exp.circuits, save=False)
        exp.analyze()

        assert exp.characterized_values[0] == pytest.approx(
            t1_true, rel=0.1
        )

    def test_final_sets_config_param(self, config):
        t1_true = 20 * us
        noise = _DurationAwareRelaxation(t1=t1_true, tphi=1000 * t1_true)
        sim = DensityMatrixSimulator(noise_model=noise)

        np.random.seed(0)
        exp = T1(
            qpu=Emulator, config=config, qubits=[0], t_max=100 * us,
            n_elements=30, simulator=sim, n_shots=2000,
        )
        exp.generate_circuits()
        Emulator.run(exp, exp.circuits, save=False)
        exp.analyze()
        exp.final()

        assert config['single_qubit/0/GE/T1'] == pytest.approx(
            t1_true, rel=0.1
        )


class TestT2Construction:

    def test_invalid_subspace_raises(self, config):
        with pytest.raises(ValueError):
            T2(qpu=Emulator, config=config, qubits=[0], subspace='bogus')

    def test_default_params_path_non_echo(self, config):
        exp = T2(qpu=Emulator, config=config, qubits=[0], echo=False)
        assert exp.params == {0: 'single_qubit/0/GE/T2*'}
        assert isinstance(exp._fit[0], FitDecayingCosine)

    def test_default_params_path_echo(self, config):
        exp = T2(qpu=Emulator, config=config, qubits=[0], echo=True)
        assert exp.params == {0: 'single_qubit/0/GE/T2e'}
        assert isinstance(exp._fit[0], FitExponential)


class TestT2SimulatedRun:

    def test_recovers_known_t2_star(self, config):
        # T1 >> tphi so that 1/T2* = 1/(2*T1) + 1/tphi ~= 1/tphi.
        tphi_true = 15 * us
        noise = _DurationAwareRelaxation(
            t1=1000 * tphi_true, tphi=tphi_true
        )
        sim = DensityMatrixSimulator(noise_model=noise)

        np.random.seed(1)
        exp = T2(
            qpu=Emulator, config=config, qubits=[0], t_max=60 * us,
            detuning=200 * kHz, echo=False, n_elements=40,
            simulator=sim, n_shots=3000,
        )
        exp.generate_circuits()
        Emulator.run(exp, exp.circuits, save=False)
        exp.analyze()

        assert exp._fit[0].fit_success
        assert exp.characterized_values[0] == pytest.approx(
            tphi_true, rel=0.15
        )

    def test_recovers_known_t2_echo(self, config):
        tphi_true = 15 * us
        noise = _DurationAwareRelaxation(
            t1=1000 * tphi_true, tphi=tphi_true
        )
        sim = DensityMatrixSimulator(noise_model=noise)

        np.random.seed(2)
        exp = T2(
            qpu=Emulator, config=config, qubits=[0], t_max=60 * us,
            echo=True, n_elements=40, simulator=sim, n_shots=3000,
        )
        exp.generate_circuits()
        Emulator.run(exp, exp.circuits, save=False)
        exp.analyze()

        assert exp.characterized_values[0] == pytest.approx(
            tphi_true, rel=0.15
        )


class TestT2DDConstruction:

    def test_invalid_subspace_raises(self, config):
        with pytest.raises(ValueError):
            T2DD(qpu=Emulator, config=config, qubits=[0], subspace='bogus')

    def test_default_params_path(self, config):
        exp = T2DD(qpu=Emulator, config=config, qubits=[0])
        assert exp.params == {0: 'single_qubit/0/GE/T2DD'}


class TestT2DDSimulatedRun:

    def test_recovers_known_decay_time(self, config):
        # n_pulses=4 (rather than the default auto-scaling-with-t) so
        # the pulse-time overhead is constant across the sweep -- see
        # module docstring for why that matters here.
        tphi_true = 10 * us
        noise = _DurationAwareRelaxation(
            t1=1000 * tphi_true, tphi=tphi_true
        )
        sim = DensityMatrixSimulator(noise_model=noise)

        np.random.seed(3)
        exp = T2DD(
            qpu=Emulator, config=config, qubits=[0], t_max=30 * us,
            n_elements=20, n_pulses=4, simulator=sim, n_shots=2000,
        )
        exp.generate_circuits()
        Emulator.run(exp, exp.circuits, save=False)
        exp.analyze()

        assert exp._fit[0].fit_success
        assert exp.characterized_values[0] == pytest.approx(
            tphi_true, rel=0.15
        )


class TestParityOscillationsConstruction:

    def test_generate_circuits_count_and_phase_column(
        self, config, bell_circuit
    ):
        po = ParityOscillations(
            qpu=Emulator, config=config, circuit=bell_circuit, n_elements=11,
        )
        assert po.qubits == bell_circuit.qubits

        po.generate_circuits()

        assert po.circuits.n_circuits == 12
        phases = list(po.circuits['phase'])
        assert np.isnan(phases[0])
        assert phases[1:] == pytest.approx(list(po.phases))


class TestParityOscillationsSimulatedRun:

    def test_analyze_recovers_near_unit_fidelity_for_bell_state(
        self, config, bell_circuit
    ):
        po = ParityOscillations(
            qpu=Emulator, config=config, circuit=bell_circuit,
            simulator=StateVectorSimulator(), n_shots=None, n_elements=21,
        )
        po.generate_circuits()
        Emulator.run(po, po.circuits, save=False)
        po.analyze()

        assert po._fit.fit_success
        assert po.fidelity['val'] == pytest.approx(1.0, abs=1e-3)

    def test_plot_runs_with_sparse_populations(self, config, bell_circuit):
        # A noiseless Bell state only ever measures '00'/'11' -- never
        # '01'/'10' -- so Results.probabilities (sparse: observed
        # states only) has just 2 entries. plot()'s bar chart must
        # look up all 4 basis states via Results.populations (a
        # zero-defaulting defaultdict) instead, or it hits a
        # matplotlib x/height shape mismatch.
        po = ParityOscillations(
            qpu=Emulator, config=config, circuit=bell_circuit,
            simulator=StateVectorSimulator(), n_shots=None, n_elements=21,
        )
        po.generate_circuits()
        Emulator.run(po, po.circuits, save=False)
        po.analyze()

        po.plot()

    def test_final_does_not_crash_when_fit_fails(self, config, bell_circuit):
        po = ParityOscillations(
            qpu=Emulator, config=config, circuit=bell_circuit,
            simulator=StateVectorSimulator(), n_shots=None, n_elements=21,
        )
        po.generate_circuits()
        Emulator.run(po, po.circuits, save=False)
        # Force a fit failure to exercise the graceful-degradation path.
        po._fit.fit = lambda *a, **k: None  # leaves fit_success False

        po.analyze()
        assert po.fidelity is None

        po.final()  # should not raise
