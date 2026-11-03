"""Unit tests for qcal.benchmarking.cb (Cycle Benchmarking).

Covers three layers:

1. Pure helpers (`compute_cycle_infidelity`, `compute_cycle_infidelity1`,
   and the module-private fit/sign helpers) on hand-constructed inputs
   with known answers.
2. Circuit generation (`CB.generate_circuits`): number of circuits,
   metadata columns, Pauli preparation/measurement and per-depth
   structure -- no simulation needed.
3. Small end-to-end runs against the Emulator with a
   DensityMatrixSimulator. Injected noise is restricted to channels
   whose CB answer is known in closed form:

   * quax's depolarizing channel with rate g on a D-dimensional gate
     body has every non-identity Pauli fidelity equal to
     f = exp(-g * D^2 / (D^2 - 1)) (verified numerically against the
     channel's superoperator), so its process infidelity is
     (D^2 - 1) / D^2 * (1 - f).
   * For a stochastic Pauli channel, the process infidelity is the
     total probability of a non-identity Pauli error.
   * Readout (SPAM) error only rescales the decay amplitude, so it
     must not show up as process infidelity.

   Tolerances are a multiple of the reported uncertainty (ufloat .s),
   with the RNGs seeded so the runs are reproducible.

`CB1` is the True-Q-backed protocol and needs `trueq` to run, so it is
not tested here; only the trueq-free `compute_cycle_infidelity1` helper
is, against a stand-in collection.
"""
import contextlib
import io
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from uncertainties import ufloat, unumpy

import qcal.benchmarking.cb as cb_module
import qcal.settings as settings
from qcal.backend.emulator import Emulator
from qcal.benchmarking.cb import (
    CB,
    _decay_array,
    _fit_decay_curve,
    _mean_sem,
    _polarization_and_infidelity,
    _propagate_sign,
    compute_cycle_infidelity,
    compute_cycle_infidelity1,
)
from qcal.circuit import Circuit, Cycle
from qcal.config import Config
from qcal.fitting.fit import FitExponential
from qcal.gates.single_qubit import SX, H, Meas, S, X
from qcal.gates.two_qubit import CNOT, CZ
from qcal.qpu.qpu import QPU
from qcal.simulation.error_models import (
    CustomErrorModel,
    DepolarizingNoise,
)
from qcal.simulation.simulators import DensityMatrixSimulator

EXAMPLE_CONFIG_PATH = str(
    Path(__file__).resolve().parents[1] / 'examples' / 'config' /
    'config.yaml'
)

# Number of reported standard deviations allowed between a recovered
# infidelity and its known value. The reported uncertainty was checked
# to be roughly calibrated (|z| <= ~1.2 over several seeds at the run
# sizes used here), so 4 sigma is a comfortable margin.
N_SIGMA = 4


def _depolarizing_infidelity(rate, dim):
    """Process infidelity of quax's depolarizing channel.

    Args:
        rate (float): depolarizing rate passed to DepolarizingNoise.
        dim (int): Hilbert-space dimension of the gate body (2 or 4).

    Returns:
        float: (dim^2 - 1) / dim^2 * (1 - exp(-rate dim^2 / (dim^2 - 1))).
    """
    d2 = dim**2
    return (d2 - 1) / d2 * (1 - np.exp(-rate * d2 / (d2 - 1)))


@pytest.fixture(autouse=True)
def _seed_rngs():
    """Seed the RNGs used for Pauli sampling and shot sampling."""
    random.seed(1234)
    np.random.seed(1234)


# ---------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------


class TestComputeCycleInfidelity:
    """compute_cycle_infidelity: (d^2-1)/d^2 * (1 - f_D / f_ref)."""

    def test_equal_polarizations_give_zero(self):
        e = compute_cycle_infidelity(
            ufloat(0.97, 0.01), ufloat(0.97, 0.01), n_qubits=2
        )
        assert e.n == pytest.approx(0.0)

    def test_known_single_qubit_value(self):
        e = compute_cycle_infidelity(
            ufloat(0.96, 0.0), ufloat(0.98, 0.0), n_qubits=1
        )
        assert e.n == pytest.approx(0.75 * (1 - 0.96 / 0.98))

    def test_perfect_reference_reduces_to_dressed_infidelity(self):
        f_D = ufloat(0.9, 0.01)
        e = compute_cycle_infidelity(f_D, ufloat(1.0, 0.0), n_qubits=2)
        assert e.n == pytest.approx(15 / 16 * 0.1)
        assert e.s == pytest.approx(15 / 16 * 0.01)


class TestComputeCycleInfidelity1:
    """compute_cycle_infidelity1 only touches `.labels` and
    `.fit(analyze_dim=2)[0].e_F.{val,std}` of a True-Q
    CircuitCollection, so a minimal stand-in suffices.
    """

    def test_single_qubit_known_value(self):
        # F_D = 0.97, F_ref = 0.99, d = 2:
        # f_D = (4 * 0.97 - 1) / 3 = 0.96
        # f_ref = (4 * 0.99 - 1) / 3 = 0.98666...
        # e_C = 3/4 * (1 - f_D / f_ref) = 0.0202702...
        val, err = compute_cycle_infidelity1(
            _FakeCollection((0,), 0.03, 0.0),
            _FakeCollection((0,), 0.01, 0.0),
        )
        assert val == pytest.approx(0.75 * (1 - 0.96 / (2.96 / 3)))
        assert err == 0.0

    def test_two_qubit_matches_compute_cycle_infidelity(self):
        # Rounded to the order of the error, the result must agree
        # with the ufloat-based helper fed the converted polarizations.
        val, err = compute_cycle_infidelity1(
            _FakeCollection((0, 1), 0.05, 0.002),
            _FakeCollection((0, 1), 0.02, 0.001),
        )
        f_D = (16 * ufloat(0.95, 0.002) - 1) / 15
        f_ref = (16 * ufloat(0.98, 0.001) - 1) / 15
        expected = compute_cycle_infidelity(f_D, f_ref, n_qubits=2)
        assert val == pytest.approx(expected.n, abs=err)
        assert err == pytest.approx(expected.s, rel=0.1)
        assert (val, err) == (0.031, 0.0023)

    def test_identical_collections_give_zero(self):
        val, err = compute_cycle_infidelity1(
            _FakeCollection((0, 1), 0.02, 0.001),
            _FakeCollection((0, 1), 0.02, 0.001),
        )
        assert val == 0.0
        assert err > 0.0


class TestMeanSem:

    def test_empty_is_nan(self):
        m = _mean_sem([])
        assert np.isnan(m.n)
        assert np.isnan(m.s)

    def test_single_value_has_zero_error(self):
        m = _mean_sem([0.7])
        assert (m.n, m.s) == (0.7, 0.0)

    def test_mean_and_standard_error(self):
        # std(ddof=1) of [1, 3] is sqrt(2); SEM = sqrt(2) / sqrt(2).
        m = _mean_sem([1.0, 3.0])
        assert m.n == pytest.approx(2.0)
        assert m.s == pytest.approx(1.0)


class TestDecayArray:

    def test_order_follows_depths_and_missing_depth_is_nan(self):
        arr = _decay_array([1, 2, 4], {4: [0.5, 0.7], 1: [0.9]})
        nominal = unumpy.nominal_values(arr)
        assert nominal[0] == pytest.approx(0.9)
        assert np.isnan(nominal[1])
        assert nominal[2] == pytest.approx(0.6)


class TestFitDecayCurve:

    def test_recovers_known_decay_rate(self):
        depths = np.array([1., 2., 4., 8., 16.])
        rng = np.random.default_rng(0)
        evs = 0.9 * 0.97**depths + rng.normal(0, 1e-4, depths.size)
        decay = np.array([ufloat(v, 1e-4) for v in evs])

        fit, f = _fit_decay_curve(depths, decay, 'Pauli X', 'interleaved')

        assert isinstance(fit, FitExponential)
        assert f.n == pytest.approx(0.97, abs=5 * f.s + 1e-4)
        assert 0 < f.s < 1e-3

    def test_nan_entries_are_dropped_before_fitting(self):
        depths = np.array([1., 2., 4., 8.])
        evs = 0.95**depths
        evs[1] += 1e-4
        decay = np.array([
            ufloat(evs[0], 0.01), ufloat(np.nan, np.nan),
            ufloat(evs[2], 0.01), ufloat(evs[3], 0.01),
        ])
        fit, f = _fit_decay_curve(depths, decay, 'Pauli X', 'interleaved')
        assert fit is not None
        assert f.n == pytest.approx(0.95, abs=1e-3)

    def test_fewer_than_two_valid_points_returns_none(self):
        depths = np.array([1., 2., 4.])
        decay = np.array([
            ufloat(0.9, 0.01), ufloat(np.nan, np.nan),
            ufloat(np.nan, np.nan),
        ])
        assert _fit_decay_curve(
            depths, decay, 'Pauli X', 'interleaved'
        ) == (None, None)


class TestPolarizationAndInfidelity:

    def test_single_qubit(self):
        pol, e_F = _polarization_and_infidelity(
            [ufloat(0.97, 0.01), ufloat(0.98, 0.01), ufloat(0.99, 0.01)],
            n_qubits=1,
        )
        assert pol.n == pytest.approx(0.98)
        assert e_F.n == pytest.approx(0.75 * 0.02)

    def test_two_qubit(self):
        _, e_F = _polarization_and_infidelity(
            [ufloat(0.96, 0.0)] * 15, n_qubits=2
        )
        assert e_F.n == pytest.approx(15 / 16 * 0.04)


class TestPropagateSign:
    """Hand-computed signs. For CZ, G(XI) = XZ; with depth 2 the
    effective Pauli at twirl position i is G(S) iff i is odd.
    """

    def test_identity_twirls_give_plus_one(self):
        twirls = [('I', 'I')] * 3
        assert _propagate_sign(
            ('X', 'I'), twirls, Cycle({CZ(0, 1)}), 2
        ) == 1

    def test_leading_twirl_anticommuting_with_decay_flips_sign(self):
        twirls = [('Z', 'I'), ('I', 'I'), ('I', 'I')]
        assert _propagate_sign(
            ('X', 'I'), twirls, Cycle({CZ(0, 1)}), 2
        ) == -1

    def test_middle_twirl_sees_conjugated_pauli(self):
        # Middle twirl sees G(XI) = XZ: IX anticommutes with it ...
        twirls = [('I', 'I'), ('I', 'X'), ('I', 'I')]
        assert _propagate_sign(
            ('X', 'I'), twirls, Cycle({CZ(0, 1)}), 2
        ) == -1
        # ... while IZ commutes with it.
        twirls = [('I', 'I'), ('I', 'Z'), ('I', 'I')]
        assert _propagate_sign(
            ('X', 'I'), twirls, Cycle({CZ(0, 1)}), 2
        ) == 1

    def test_two_anticommuting_twirls_cancel(self):
        twirls = [('Z', 'I'), ('I', 'I'), ('Y', 'I')]
        assert _propagate_sign(
            ('X', 'I'), twirls, Cycle({CZ(0, 1)}), 2
        ) == 1


# ---------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------


class TestCBConstruction:

    def test_depths_are_sorted(self, config):
        cb = CB(Emulator, config, Cycle({CZ(0, 1)}), [8, 2, 4])
        assert cb._circuit_depths == [2, 4, 8]

    def test_non_identity_depth_is_rejected(self, config):
        with pytest.raises(ValueError, match='not the identity'):
            CB(Emulator, config, Cycle({X(0)}), [1])

    def test_non_multiple_depth_is_checked_too(self, config):
        # X^2 = I, but X^3 = X: depth 3 is not a multiple of 2, so it
        # must be explicitly checked and rejected.
        with pytest.raises(ValueError, match='not the identity'):
            CB(Emulator, config, Cycle({X(0)}), [2, 3])

    def test_targeted_decay_of_wrong_length_is_rejected(self, config):
        with pytest.raises(ValueError, match='length'):
            CB(
                Emulator, config, Cycle({CZ(0, 1)}), [2],
                targeted_decays=['XYZ'],
            )

    def test_targeted_decay_with_invalid_character_is_rejected(
        self, config
    ):
        with pytest.raises(ValueError, match='invalid characters'):
            CB(
                Emulator, config, Cycle({CZ(0, 1)}), [2],
                targeted_decays=['XA'],
            )

    def test_targeted_decays_are_upper_cased(self, config):
        cb = CB(
            Emulator, config, Cycle({CZ(0, 1)}), [2],
            targeted_decays=['xz', 'iy'],
        )
        assert cb._targeted_decays == ['XZ', 'IY']

    def test_analyze_subsystems_requires_a_cycle(self, config):
        circuit = Circuit([Cycle({CZ(0, 1)}), Cycle({CZ(0, 1)})])
        with pytest.raises(ValueError, match='requires cycle_or_circuit'):
            CB(Emulator, config, circuit, [1], analyze_subsystems=True)

    def test_analyze_subsystems_requires_two_gates(self, config):
        with pytest.raises(ValueError, match='at least two gates'):
            CB(
                Emulator, config, Cycle({CZ(0, 1)}), [2],
                analyze_subsystems=True,
            )

    def test_subsystems_default_to_full_register(self, config):
        cb = CB(Emulator, config, Cycle({CZ(0, 1)}), [2])
        assert cb.subsystems == [(0, 1)]

    def test_subsystems_include_each_gate_body(self, config):
        cb = CB(
            Emulator, config, Cycle({CZ(0, 1), CZ(2, 3)}), [2],
            analyze_subsystems=True,
        )
        assert cb.subsystems[0] == (0, 1, 2, 3)
        assert sorted(cb.subsystems[1:]) == [(0, 1), (2, 3)]

    def test_experiments_follow_include_ref_cycle(self, config):
        cb = CB(Emulator, config, Cycle({CZ(0, 1)}), [2])
        assert list(cb.fit) == ['interleaved']
        cb = CB(
            Emulator, config, Cycle({CZ(0, 1)}), [2],
            include_ref_cycle=True,
        )
        assert list(cb.fit) == ['interleaved', 'reference']

    def test_reference_infidelity_is_nan_before_analysis(self, config):
        cb = CB(Emulator, config, Cycle({CZ(0, 1)}), [2])
        assert np.isnan(cb.reference_infidelity)
        assert np.isnan(cb.dressed_infidelity)
        assert cb.subsystem_summary is None


# ---------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------


class TestCBGenerateCircuits:

    def test_circuit_count_and_metadata(self, config):
        depths, n_rand = [2, 4], 3
        cb = CB(
            Emulator, config, Cycle({CZ(0, 1)}), depths,
            n_randomizations=n_rand,
            targeted_decays=['ZZ', 'XI', 'IZ'],
        )
        cb.generate_circuits()
        circuits = cb.circuits

        # Greedy QWC grouping (heaviest first): {ZZ, IZ} and {XI}.
        groups = {
            ''.join(g.measurement_basis): sorted(
                ''.join(p) for p in g.paulis
            )
            for g in cb._pauli_groups
        }
        assert groups == {'ZZ': ['IZ', 'ZZ'], 'XI': ['XI']}

        assert circuits.n_circuits == len(groups) * len(depths) * n_rand
        assert set(circuits['experiment']) == {'interleaved'}
        assert set(circuits['measurement_basis']) == {'ZZ', 'XI'}
        assert sorted(set(circuits['depth'])) == depths
        assert sorted(set(circuits['randomization'])) == [0, 1, 2]
        for _, row in circuits._df.iterrows():
            assert len(row['twirl_strings']) == row['depth'] + 1
            assert len(row['group_signs']) == len(row['group_paulis'])
            assert set(row['group_signs']) <= {1, -1}
            assert sorted(row['group_paulis']) == (
                groups[row['measurement_basis']]
            )

    def test_all_paulis_used_when_n_decays_exceeds_4n_minus_1(
        self, config
    ):
        cb = CB(
            Emulator, config, Cycle({X(0)}), [2], n_decays=20,
            n_randomizations=1,
        )
        cb.generate_circuits()
        paulis = sorted(
            p for ps in cb.circuits['group_paulis'] for p in ps
        )
        # The identity is dropped; X, Y, Z each get their own basis.
        assert paulis == ['X', 'Y', 'Z']
        assert sorted(cb.circuits['measurement_basis']) == ['X', 'Y', 'Z']

    def test_interleaved_circuit_structure(self, config):
        depth = 4
        cb = CB(
            Emulator, config, Cycle({CZ(0, 1)}), [depth],
            n_randomizations=2, targeted_decays=['XY'],
        )
        cb.generate_circuits()

        for circuit in cb.circuits['circuit']:
            cycles = circuit.cycles
            barriers = [c for c in cycles if c.is_barrier]
            gates = _gate_names_per_cycle(circuit)

            # prep | B | twirl | B | (CZ | twirl | B) x depth | rot | M
            assert len(barriers) == depth + 2
            assert sum(names == ['CZ'] for names in gates) == depth
            assert gates[-1] == ['Meas', 'Meas']
            # X and Y prep/measurement are non-trivial basis changes
            # on both qubits.
            assert len(gates[0]) == 2
            assert len(gates[-2]) == 2
            assert circuit.qubits == (0, 1)

    def test_reference_circuits_never_contain_the_cycle(self, config):
        depth = 2
        cb = CB(
            Emulator, config, Cycle({CZ(0, 1)}), [depth],
            n_randomizations=2, targeted_decays=['ZX'],
            include_ref_cycle=True,
        )
        cb.generate_circuits()
        df = cb.circuits._df

        assert sorted(set(df['experiment'])) == ['interleaved', 'reference']
        assert (df['experiment'] == 'reference').sum() == 2
        for _, row in df.iterrows():
            gates = _gate_names_per_cycle(row['circuit'])
            n_cz = sum(names == ['CZ'] for names in gates)
            barriers = sum(c.is_barrier for c in row['circuit'].cycles)
            assert barriers == depth + 2
            if row['experiment'] == 'reference':
                assert n_cz == 0
            else:
                assert n_cz == depth

    def test_zxzxz_decomposition_structure(self, config):
        # Each of the depth + 1 twirl layers (with the state prep
        # merged into the first and the basis change merged into the
        # last) becomes one 5-cycle Rz-X90-Rz-X90-Rz block; only the
        # inner twirls are followed by a barrier.
        depth = 4
        cb = CB(
            Emulator, config, Cycle({CZ(0, 1)}), [depth],
            n_randomizations=2, targeted_decays=['XY', 'ZI'],
            decompose_to_zxzxz=True,
        )
        cb.generate_circuits()

        for circuit in cb.circuits['circuit']:
            gates = _gate_names_per_cycle(circuit)
            names = {n for cycle in gates for n in cycle}
            assert names == {'Rz', 'X90', 'CZ', 'Meas'}
            assert sum(c == ['CZ'] for c in gates) == depth
            assert len(gates) == 5 * (depth + 1) + depth + 1
            assert gates[:5] == [
                ['Rz', 'Rz'], ['X90', 'X90'], ['Rz', 'Rz'],
                ['X90', 'X90'], ['Rz', 'Rz'],
            ]
            assert gates[-1] == ['Meas', 'Meas']
            assert sum(
                c.is_barrier for c in circuit.cycles
            ) == depth - 1

    def test_measured_qubits_restrict_decays_to_iz(self, config):
        cb = CB(
            Emulator, config, Cycle({Meas(0)}), [1, 2], n_decays=50,
            n_randomizations=1,
        )
        cb.generate_circuits()
        assert cb._measured_qubits == frozenset({0})
        paulis = {p for ps in cb.circuits['group_paulis'] for p in ps}
        assert paulis == {'Z'}

    def test_identity_is_dropped_and_duplicates_resampled(
        self, config, monkeypatch
    ):
        original = cb_module.generate_random_n_qubit_paulis
        batches = iter([
            [('I', 'I'), ('I', 'I'), ('X', 'Z')],
            [('X', 'Z'), ('I', 'I')],
            [('Y', 'I'), ('Z', 'Z')],
        ])

        def sampler(qubits, measured_qubits=None, n_random_paulis=1):
            # Decay sampling passes measured_qubits; twirl sampling
            # does not.
            if measured_qubits is not None:
                return next(batches)
            return original(qubits, n_random_paulis=n_random_paulis)

        monkeypatch.setattr(
            cb_module, 'generate_random_n_qubit_paulis', sampler
        )
        cb = CB(
            Emulator, config, Cycle({CZ(0, 1)}), [2], n_decays=3,
            n_randomizations=1,
        )
        cb.generate_circuits()
        paulis = {p for ps in cb.circuits['group_paulis'] for p in ps}
        assert paulis == {'XZ', 'YI', 'ZZ'}

    @pytest.mark.parametrize('n_decays', [1, 5, 14])
    def test_random_decays_are_distinct_and_non_identity(
        self, config, n_decays
    ):
        random.seed(n_decays)
        cb = CB(
            Emulator, config, Cycle({CZ(0, 1)}), [2], n_decays=n_decays,
            n_randomizations=1,
        )
        cb.generate_circuits()
        paulis = {p for ps in cb.circuits['group_paulis'] for p in ps}
        assert len(paulis) == n_decays
        assert 'II' not in paulis

    @pytest.mark.parametrize('n_decays', [15, 100])
    def test_all_paulis_used_once_n_decays_reaches_available(
        self, config, n_decays
    ):
        cb = CB(
            Emulator, config, Cycle({CZ(0, 1)}), [2], n_decays=n_decays,
            n_randomizations=1,
        )
        cb.generate_circuits()
        paulis = {p for ps in cb.circuits['group_paulis'] for p in ps}
        assert len(paulis) == 15
        assert 'II' not in paulis

    def test_measured_qubit_caps_available_decays(self, config):
        # Qubit 0 measured mid-cycle: only I/Z there, so 2 * 4 - 1 = 7
        # non-identity decays exist, and asking for more uses them all.
        cb = CB(
            Emulator, config, Cycle({Meas(0), X(1)}), [2], n_decays=7,
            n_randomizations=1,
        )
        cb.generate_circuits()
        paulis = {p for ps in cb.circuits['group_paulis'] for p in ps}
        assert len(paulis) == 7
        assert all(p[0] in 'IZ' for p in paulis)


# ---------------------------------------------------------------------
# Sign propagation, end to end on noiseless data
# ---------------------------------------------------------------------


class TestCBSignsOnNoiselessData:
    """With no noise, every signed, marginalized EV must be exactly +1
    at every depth if the expected eigenstate signs are right.

    These call generate_circuits()/QPU.run()/analyze() directly and
    check the signed EVs rather than the fits: perfectly flat noiseless
    decays leave the exponential fit's uncertainty undetermined.
    """

    @pytest.mark.parametrize(
        ('cycle_or_circuit', 'depths', 'zxzxz'),
        [
            (Cycle({CZ(0, 1)}), [2, 4], False),
            (Cycle({CZ(0, 1)}), [2, 4], True),
            (Cycle({CNOT(0, 1)}), [2, 4], False),
            (Cycle({X(0)}), [2, 4], False),
            (Cycle({SX(0)}), [4, 8], False),
            (Cycle({S(0)}), [4, 8], False),
            (Cycle({CZ(0, 1), X(2)}), [2], False),
            (Circuit([Cycle({CZ(0, 1)}), Cycle({CZ(0, 1)})]), [1, 2],
             False),
        ],
        ids=[
            'CZ', 'CZ-zxzxz', 'CNOT', 'X', 'SX', 'S', 'CZ+X',
            'CZ-CZ-circuit',
        ],
    )
    def test_signed_evs_are_one(
        self, config, cycle_or_circuit, depths, zxzxz
    ):
        cb = _analyze_noiseless(
            config, cycle_or_circuit, depths, decompose_to_zxzxz=zxzxz
        )
        decays = cb.pauli_decays['interleaved']
        assert decays
        for pauli, decay in decays.items():
            assert len(decay) == len(depths)
            assert np.allclose(unumpy.nominal_values(decay), 1.0), pauli

    def test_signed_evs_are_one_for_order_three_cycle(self, config):
        circuit = Circuit([Cycle({S(0)}), Cycle({H(0)})])
        cb = _analyze_noiseless(config, circuit, [3, 6])
        for pauli, decay in cb.pauli_decays['interleaved'].items():
            assert np.allclose(unumpy.nominal_values(decay), 1.0), pauli


class TestCBInjectedPauliNoise:
    """A realistic, non-uniform two-qubit Pauli channel on the CZ.

    For a Pauli channel E(rho) = sum_k p_k P_k rho P_k, the process
    infidelity is exactly the total probability of a non-identity
    Pauli error, regardless of how it is distributed across Paulis.
    """

    PAULI_ERRORS = {'ZI': 0.004, 'IZ': 0.003, 'ZZ': 0.002, 'XX': 0.001}

    @pytest.fixture(scope='class')
    @staticmethod
    def cb():
        random.seed(2)
        np.random.seed(2)
        noise = CustomErrorModel.from_PNR(
            {CZ(0, 1): TestCBInjectedPauliNoise.PAULI_ERRORS}
        )
        cmat = np.array([[0.98, 0.02], [0.05, 0.95]])
        for q in range(2):
            noise.add_readout_noise(q, cmat)
        cb = CB(
            Emulator, Config(EXAMPLE_CONFIG_PATH), Cycle({CZ(0, 1)}),
            [2, 4, 8], n_decays=15, n_randomizations=8, n_shots=1000,
            simulator=DensityMatrixSimulator(error_model=noise),
        )
        _quiet(cb.run)
        return cb

    def test_recovers_process_infidelity(self, cb):
        expected = sum(self.PAULI_ERRORS.values())
        e_F = cb.process_infidelity
        assert e_F.n == pytest.approx(expected, abs=N_SIGMA * e_F.s)
        assert e_F.s < 0.5 * expected

    def test_every_pauli_decay_is_fit(self, cb):
        fids = cb.pauli_fidelities['interleaved']
        assert len(fids) == 15
        assert all(np.isfinite(f.n) and f.s > 0 for f in fids.values())


class TestCBReadoutNoiseOnly:
    """SPAM-only noise: CB must report ~zero process infidelity."""

    @pytest.fixture(scope='class')
    @staticmethod
    def cb():
        random.seed(0)
        np.random.seed(0)
        cb = CB(
            Emulator, Config(EXAMPLE_CONFIG_PATH), Cycle({CZ(0, 1)}),
            [2, 4, 8], n_decays=20, n_randomizations=8, n_shots=1000,
            simulator=_simulator(),
        )
        _quiet(cb.run)
        return cb

    def test_process_infidelity_is_zero_within_uncertainty(self, cb):
        e_F = cb.process_infidelity
        assert 0 < e_F.s < 5e-3
        assert abs(e_F.n) < N_SIGMA * e_F.s

    def test_process_fidelity_is_complement(self, cb):
        assert cb.process_fidelity.n == pytest.approx(
            1 - cb.process_infidelity.n
        )
        assert cb.process_fidelity.s == pytest.approx(
            cb.process_infidelity.s
        )

    def test_all_two_qubit_paulis_are_fit(self, cb):
        # n_decays (20) > 4^2 - 1, so every non-identity Pauli is used.
        fids = cb.pauli_fidelities['interleaved']
        assert len(fids) == 15
        assert 'II' not in fids
        assert set(cb.fit['interleaved']) == set(fids)
        assert all(
            isinstance(f, FitExponential)
            for f in cb.fit['interleaved'].values()
        )

    def test_decay_amplitude_absorbs_readout_error(self, cb):
        # Readout error lowers EV(d) uniformly at every depth (A < 1)
        # rather than making it decay.
        for decay in cb.pauli_decays['interleaved'].values():
            evs = unumpy.nominal_values(decay)
            assert len(evs) == 3
            assert np.all(evs < 0.99)
            assert np.all(evs > 0.8)


class TestCBInjectedDepolarizing:
    """Joint two-qubit depolarizing noise on the CZ only."""

    RATE = 0.02

    @pytest.fixture(scope='class')
    @staticmethod
    def cb():
        random.seed(1)
        np.random.seed(1)
        cb = CB(
            Emulator, Config(EXAMPLE_CONFIG_PATH), Cycle({CZ(0, 1)}),
            [2, 4, 8], n_decays=20, n_randomizations=8, n_shots=1000,
            simulator=_simulator(two_qubit=TestCBInjectedDepolarizing.RATE),
        )
        _quiet(cb.run)
        return cb

    def test_recovers_process_infidelity(self, cb):
        expected = _depolarizing_infidelity(self.RATE, 4)
        e_F = cb.process_infidelity
        assert e_F.n == pytest.approx(expected, abs=N_SIGMA * e_F.s)
        assert e_F.s < 0.25 * expected

    def test_dressed_equals_process_without_reference(self, cb):
        assert cb.dressed_infidelity is cb.process_infidelity
        assert np.isnan(cb.reference_infidelity)

    def test_pauli_fidelities_match_depolarizing_value(self, cb):
        expected = np.exp(-self.RATE * 16 / 15)
        fids = cb.pauli_fidelities['interleaved']
        mean = sum(fids.values()) / len(fids)
        assert mean.n == pytest.approx(expected, abs=N_SIGMA * mean.s)

    def test_pauli_decays_decrease_with_depth(self, cb):
        for decay in cb.pauli_decays['interleaved'].values():
            evs = unumpy.nominal_values(decay)
            assert evs[0] > evs[-1]

    def test_loss_is_keyed_by_the_register(self, cb):
        assert cb.loss == {(0, 1): cb.process_infidelity}

    def test_subsystem_summary_has_whole_cycle_row(self, cb):
        summary = cb.subsystem_summary
        assert list(summary.columns) == [
            'Subset', 'Operation', 'Dressed Infidelity'
        ]
        assert len(summary) == 1
        assert summary['Subset'][0] == (0, 1)
        assert summary['Operation'][0] == 'CZ'
        assert summary['Dressed Infidelity'][0].endswith('%')

    def test_subsystem_infidelities_mirror_process_infidelity(self, cb):
        assert cb.subsystem_infidelities[(0, 1)].n == pytest.approx(
            cb.process_infidelity.n
        )
        assert cb.subsystem_pauli_fidelities == {'interleaved': {}}


class TestCBWithReferenceCycle:
    """Depolarizing noise on every single-qubit gate (including the
    twirls) of an X cycle. Depolarizing noise commutes with every
    gate, so the reference experiment isolates the twirl noise
    exactly and the bare infidelity is that of the X gate alone.
    """

    RATE = 0.01

    @pytest.fixture(scope='class')
    @staticmethod
    def cb():
        random.seed(2)
        np.random.seed(2)
        cb = CB(
            Emulator, Config(EXAMPLE_CONFIG_PATH), Cycle({X(0)}),
            [2, 4, 8], n_decays=20, n_randomizations=10, n_shots=1000,
            include_ref_cycle=True,
            simulator=_simulator(
                single_qubit=TestCBWithReferenceCycle.RATE
            ),
        )
        _quiet(cb.run)
        return cb

    def test_reference_infidelity_is_one_twirl_layer(self, cb):
        expected = _depolarizing_infidelity(self.RATE, 2)
        e = cb.reference_infidelity
        assert e.n == pytest.approx(expected, abs=N_SIGMA * e.s)

    def test_dressed_infidelity_is_gate_plus_twirl(self, cb):
        expected = _depolarizing_infidelity(2 * self.RATE, 2)
        e = cb.dressed_infidelity
        assert e.n == pytest.approx(expected, abs=N_SIGMA * e.s)

    def test_bare_process_infidelity_is_gate_only(self, cb):
        expected = _depolarizing_infidelity(self.RATE, 2)
        e = cb.process_infidelity
        assert e.n == pytest.approx(expected, abs=N_SIGMA * e.s)
        assert e.n < cb.dressed_infidelity.n

    def test_both_experiments_are_fit(self, cb):
        assert set(cb.pauli_fidelities) == {'interleaved', 'reference'}
        assert set(cb.pauli_fidelities['interleaved']) == {'X', 'Y', 'Z'}
        assert set(cb.pauli_fidelities['reference']) == {'X', 'Y', 'Z'}

    def test_loss_uses_integer_label_for_single_qubit(self, cb):
        assert cb.loss == {0: cb.process_infidelity}

    def test_summary_has_reference_and_bare_columns(self, cb):
        assert list(cb.subsystem_summary.columns) == [
            'Subset', 'Operation', 'Dressed Infidelity',
            'Reference Infidelity', 'Bare Infidelity',
        ]
        assert cb.subsystem_summary['Operation'][0] == 'X'


class TestCBAnalyzeSubsystems:
    """Two disjoint CZs, each with joint depolarizing noise."""

    RATE = 0.02

    @pytest.fixture(scope='class')
    @staticmethod
    def cb():
        random.seed(3)
        np.random.seed(3)
        cb = CB(
            Emulator, Config(EXAMPLE_CONFIG_PATH),
            Cycle({CZ(0, 1), CZ(2, 3)}), [2, 4, 8], n_decays=30,
            n_randomizations=6, n_shots=1000, analyze_subsystems=True,
            simulator=_simulator(
                two_qubit=TestCBAnalyzeSubsystems.RATE
            ),
        )
        _quiet(cb.run)
        return cb

    def test_each_subsystem_recovers_its_infidelity(self, cb):
        expected = _depolarizing_infidelity(self.RATE, 4)
        for pair in [(0, 1), (2, 3)]:
            e = cb.subsystem_infidelities[pair]
            assert e.n == pytest.approx(expected, abs=N_SIGMA * e.s)

    def test_whole_cycle_is_product_channel(self, cb):
        # Process fidelity of a tensor product is the product of the
        # individual process fidelities.
        e_pair = _depolarizing_infidelity(self.RATE, 4)
        expected = 1 - (1 - e_pair)**2
        e = cb.process_infidelity
        assert e.n == pytest.approx(expected, abs=N_SIGMA * e.s)
        assert cb.subsystem_infidelities[(0, 1, 2, 3)].n == (
            pytest.approx(e.n)
        )

    def test_loss_excludes_the_whole_cycle(self, cb):
        assert set(cb.loss) == {(0, 1), (2, 3)}

    def test_subsystem_pauli_fidelities_are_local(self, cb):
        local = cb.subsystem_pauli_fidelities['interleaved']
        assert set(local) == {(0, 1), (2, 3)}
        for pair, fids in local.items():
            assert all(len(p) == 2 for p in fids)
            assert 'II' not in fids
            assert set(cb.subsystem_fit['interleaved'][pair]) == set(fids)
            assert set(
                cb.subsystem_pauli_decays['interleaved'][pair]
            ) == set(fids)
        # The whole-cycle set stays exactly the sampled 4-qubit Paulis.
        assert all(
            len(p) == 4 for p in cb.pauli_fidelities['interleaved']
        )

    def test_summary_has_one_row_per_subsystem(self, cb):
        summary = cb.subsystem_summary
        assert list(summary['Subset']) == cb.subsystems
        assert list(summary['Operation']) == ['Cycle', 'CZ', 'CZ']


class TestCBRunSaving:
    """CB.run() must call QPU.run(..., auto_save=False) and then save
    exactly once itself, after analysis.
    """

    def test_run_passes_auto_save_false(self, config, monkeypatch):
        calls = []
        original = Emulator.run

        def spy(self, circuits, *args, **kwargs):
            calls.append(kwargs)
            return original(self, circuits, *args, **kwargs)

        monkeypatch.setattr(Emulator, 'run', spy)
        cb = CB(
            Emulator, config, Cycle({X(0)}), [2], n_decays=1,
            n_randomizations=1, n_shots=10,
        )
        _quiet(cb.run)
        assert calls == [{'auto_save': False}]

    def test_saves_exactly_once_when_save_data_enabled(
        self, config, monkeypatch, tmp_path
    ):
        saves = []
        original = QPU.save

        def spy(self, *args, **kwargs):
            saves.append(self)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(QPU, 'save', spy)
        monkeypatch.setattr(settings.Settings, 'save_data', True)
        monkeypatch.setattr(
            settings.Settings, 'data_save_path', f'{tmp_path}/'
        )

        cb = CB(
            Emulator, config, Cycle({X(0)}), [2, 4], n_decays=3,
            n_randomizations=2, n_shots=50,
        )
        _quiet(cb.run)

        # Once, from CB.save() after analysis -- not also from inside
        # QPU.run() (which auto_save=False suppresses).
        assert saves == [cb]
        assert cb._data_manager._exp_id.endswith('_CBQ0')
        save_path = Path(cb._data_manager._save_path)
        assert save_path.is_relative_to(tmp_path)
        assert (save_path / 'circuits.qc').exists()
        assert (save_path / 'CB_decays_X_0.png').exists()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class _FakeCollection:
    """Minimal stand-in for a trueq.CircuitCollection.

    Args:
        labels (tuple): qubit labels.
        e_F (float): process infidelity reported by the fit.
        std (float): uncertainty on e_F.
    """

    def __init__(self, labels, e_F, std):
        self.labels = labels
        self._e_F = SimpleNamespace(val=e_F, std=std)

    def fit(self, analyze_dim=2):
        """Mimic CircuitCollection.fit.

        Args:
            analyze_dim (int): ignored.

        Returns:
            list: a single estimate exposing `.e_F`.
        """
        return [SimpleNamespace(e_F=self._e_F)]


def _simulator(single_qubit=0.0, two_qubit=0.0):
    """DensityMatrixSimulator with depolarizing and readout noise.

    Args:
        single_qubit (float): depolarizing rate on single-qubit gates.
        two_qubit (float): joint depolarizing rate on two-qubit gates.

    Returns:
        DensityMatrixSimulator: simulator with an asymmetric readout
            confusion matrix on qubits 0-7.
    """
    noise = DepolarizingNoise(
        single_qubit=single_qubit, two_qubit=two_qubit
    )
    cmat = np.array([[0.98, 0.02], [0.05, 0.95]])
    for q in range(8):
        noise.add_readout_noise(q, cmat)
    return DensityMatrixSimulator(error_model=noise)


def _analyze_noiseless(config, cycle_or_circuit, depths, **kwargs):
    """Generate, simulate (noiselessly) and analyze CB circuits.

    Args:
        config (Config): qcal config.
        cycle_or_circuit (Cycle | Circuit): cycle to benchmark.
        depths (list[int]): circuit depths.
        **kwargs: extra CB keyword arguments.

    Returns:
        CB: the analyzed CB instance.
    """
    cb = CB(
        Emulator, config, cycle_or_circuit, depths, n_decays=30,
        n_randomizations=4, n_shots=50,
        simulator=DensityMatrixSimulator(error_model={}), **kwargs,
    )

    def run():
        cb.generate_circuits()
        Emulator.run(cb, cb.circuits, auto_save=False)
        cb.analyze()

    _quiet(run)
    return cb


def _gate_names_per_cycle(circuit):
    """Sorted gate class names in each non-barrier cycle.

    Args:
        circuit (Circuit): circuit to inspect.

    Returns:
        list[list[str]]: one sorted list of names per cycle.
    """
    return [
        sorted(type(g).__name__ for g in cycle.gates)
        for cycle in circuit.cycles if not cycle.is_barrier
    ]


def _quiet(func):
    """Call func with stdout suppressed (runtime tables, plots).

    Args:
        func (Callable): zero-argument callable.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        func()
