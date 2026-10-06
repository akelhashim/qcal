"""Unit tests for qcal.characterization.phase_estimation.zz.JAZZ.

JAZZ measures a ZZ coupling by comparing two Ramsey experiments on a
target qubit -- one with the control qubit in its "low" state, one in
its "high" state -- echoed with X pulses that cancel ordinary 1-body Z
drift but preserve any genuine ZZ term. The virtual Ramsey detuning is
injected purely via Rz phase on the target qubit
(make_JAZZ_circuits() computes `phase = 2*pi*detuning*t` and applies
it with a single Rz, identically for both branches), and the X90/Idle
gates used elsewhere in the sequence carry plain identity unitaries
with no modeled physics for a static ZZ coupling. So running the
circuits as generated, unmodified, through the Emulator can only ever
recover ZZ = 0 regardless of qubit_pairs or conditional_phase -- not a
meaningful test of the frequency-difference extraction.

To simulate an actual nonzero ZZ rate, the final Rz on the target
qubit is found (via the 'sequence' CircuitSet column, which marks each
circuit's branch as low ('C0 ...') or high ('C1'/'C2 ...')) and its
unitary is overwritten with an extra `2*pi*zz_true*t` added only on
the high branch -- the same "overwrite gate unitaries after
generate_circuits()" technique used throughout the rest of this suite.
This makes freq_high - freq_low equal to -zz_true exactly for a 'GE'
target subspace (confirmed by direct measurement), letting the
recovered char_values be checked against a known, independently-chosen
ZZ rate. For a 'GF' target subspace ('12'/'22'), the Rz is embedded
into the |1>-|2> block of a 3x3 unitary, but the Ramsey coherence
being measured is between |0> and |2> -- which only picks up half of
that embedded phase -- so _inject_zz() injects at twice the intended
rate for those two conditional_phase values. See _inject_zz()'s
docstring for details.

A decaying-cosine fit can land on stderr=None for both branches (e.g.
at small n_elements), in which case uncertainty_of_sum() of the two
None-valued (falling back to 0.0) stderrs is exactly 0.0. That used to
crash round_to_order_error(val, 0.0) on log10(0) -- now fixed, with
regression coverage in test_math_utils.py -- so JAZZ's own default
n_elements=50 is used here rather than pinning a larger value.
"""
import numpy as np
import pytest

from qcal.backend.emulator import Emulator
from qcal.characterization.phase_estimation.zz import JAZZ
from qcal.gates.single_qubit import rz
from qcal.simulation.simulators import StateVectorSimulator
from qcal.units import MHz, kHz, us


def _inject_zz(jazz, zz_true, target_qubit=1):
    """Add an extra zz_true*t phase to the target's Rz, high branch only.

    For a 'GF' target subspace, the Rz is a 2x2 phase gate embedded
    into the |1>-|2> block (see Gate.unitary), but the Ramsey coherence
    being measured is between |0> and |2> -- which only picks up half
    of that phase (|0> is unaffected by an EF-subspace gate; |2> gets
    the full embedded phase/2 convention). So injecting phase at twice
    the intended rate here recovers the intended zz_true in the
    measured char_values. Confirmed empirically: an uninjected 1 MHz
    virtual detuning resolves to ~500 kHz for target_subspace='GF',
    exactly this same halving.
    """
    phase_multiplier = 2 if jazz._ts == 'GF' else 1
    seqs = list(jazz._circuits['sequence'])
    times_arr = jazz._times[jazz._qubits[0]]
    for i, circuit in enumerate(jazz._circuits):
        if not seqs[i].startswith(jazz._seq_high):
            continue
        t = times_arr[i // 4]
        for cyc in circuit:
            if cyc.is_barrier:
                continue
            for gate in cyc:
                if gate.name == 'Rz' and gate.qubits == (target_qubit,):
                    base_phase = gate.properties['params']['phase']
                    gate.unitary = rz(
                        base_phase + phase_multiplier * 2 * np.pi * zz_true * t
                    )


class TestJAZZConstruction:

    def test_invalid_conditional_phase_raises(self, config):
        with pytest.raises(ValueError):
            JAZZ(
                qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
                conditional_phase='bogus',
            )

    def test_default_params_path(self, config):
        jazz = JAZZ(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            conditional_phase='11',
        )
        assert jazz.params[(0, 1)] == 'two_qubit/(0, 1)/ZZ11'

    @pytest.mark.parametrize('conditional_phase,cs,ts', [
        ('11', 'GE', 'GE'),
        ('21', 'GF', 'GE'),
        ('12', 'GE', 'GF'),
        ('22', 'GF', 'GF'),
    ])
    def test_conditional_phase_selects_correct_subspaces(
        self, config, conditional_phase, cs, ts
    ):
        jazz = JAZZ(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            conditional_phase=conditional_phase,
        )
        assert jazz._cs == cs
        assert jazz._ts == ts

    def test_generate_circuits_produces_four_per_time_point(self, config):
        jazz = JAZZ(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            conditional_phase='11', n_elements=10,
        )
        jazz.generate_circuits()
        assert jazz._circuits.n_circuits == 4 * 10


class TestJAZZSimulatedRun:

    @pytest.mark.parametrize('conditional_phase', ['11', '21', '12', '22'])
    def test_no_coherent_zz_gives_zero_characterized_value(
        self, config, conditional_phase
    ):
        np.random.seed(0)
        jazz = JAZZ(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            t_max=5 * us, detuning=1 * MHz,
            conditional_phase=conditional_phase,
            simulator=StateVectorSimulator(), n_shots=1000,
        )
        jazz.run()

        assert jazz._char_values[(0, 1)]['val'] == pytest.approx(
            0.0, abs=5e3
        )

    @pytest.mark.parametrize('conditional_phase', ['11', '21', '12', '22'])
    def test_injected_zz_is_recovered(self, config, conditional_phase):
        np.random.seed(0)
        zz_true = 200 * kHz
        jazz = JAZZ(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            t_max=5 * us, detuning=1 * MHz,
            conditional_phase=conditional_phase,
            simulator=StateVectorSimulator(), n_shots=1000,
        )
        jazz.generate_circuits()
        _inject_zz(jazz, zz_true)

        Emulator.run(jazz, jazz._circuits, save=False)
        jazz.save()
        jazz.analyze()

        assert jazz._char_values[(0, 1)]['val'] == pytest.approx(
            -zz_true, abs=5e3
        )

    def test_final_sets_config_param(self, config):
        np.random.seed(0)
        zz_true = 200 * kHz
        jazz = JAZZ(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            t_max=5 * us, detuning=1 * MHz, conditional_phase='11',
            simulator=StateVectorSimulator(), n_shots=1000,
        )
        jazz.generate_circuits()
        _inject_zz(jazz, zz_true)

        Emulator.run(jazz, jazz._circuits, save=False)
        jazz.save()
        jazz.analyze()
        jazz.final()

        assert config['two_qubit/(0, 1)/ZZ11'] == pytest.approx(
            -zz_true, abs=5e3
        )


class TestJAZZPlotting:

    def test_plot_runs_without_error(self, config):
        np.random.seed(0)
        jazz = JAZZ(
            qpu=Emulator, config=config, qubit_pairs=[(0, 1)],
            t_max=5 * us, detuning=1 * MHz, conditional_phase='11',
            simulator=StateVectorSimulator(), n_shots=1000,
        )
        jazz.run()
        jazz.plot()
