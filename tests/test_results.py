"""Unit tests for qcal.results: Results, PerShotResults, and readout
correction (confusion-matrix and quasi-probabilistic).
"""
import dataclasses
import itertools

import numpy as np
import pandas as pd
import pytest

from qcal.results import (
    PerShotResults,
    Results,
    qp_readout_correction,
    readout_correction,
)


class TestConstruction:

    def test_dict_property_matches_input(self):
        results = Results({'1': 10, '0': 90})
        assert dict(results.dict) == {'0': 90, '1': 10}

    def test_empty_construction(self):
        results = Results()
        assert dict(results.dict) == {}
        assert results.states == ()
        assert results.n_shots == 0

    def test_construction_sorts_bitstrings(self):
        results = Results({'11': 1, '00': 2, '01': 3})
        assert list(results.dict.keys()) == ['00', '01', '11']


class TestProperties:

    def test_counts(self):
        results = Results({'0': 90, '1': 10})
        assert results.counts['0'] == 90
        assert results.counts['1'] == 10

    def test_probabilities(self):
        results = Results({'0': 90, '1': 10})
        assert results.probabilities['0'] == pytest.approx(0.9)
        assert results.probabilities['1'] == pytest.approx(0.1)

    def test_populations_matches_probabilities(self):
        results = Results({'0': 90, '1': 10})
        assert results.populations['0'] == pytest.approx(0.9)
        assert results.populations['1'] == pytest.approx(0.1)

    def test_populations_defaults_to_zero_for_unseen_state(self):
        results = Results({'0': 90, '1': 10})
        assert results.populations['bogus'] == 0.

    def test_states(self):
        results = Results({'10': 1, '00': 2, '01': 3})
        assert results.states == ('00', '01', '10')

    def test_n_shots(self):
        results = Results({'0': 90, '1': 10})
        assert results.n_shots == 100

    def test_n_qudits(self):
        results = Results({'000': 1, '111': 1})
        assert results.n_qudits == 3

    def test_levels_for_qubits(self):
        results = Results({'00': 1, '01': 1, '10': 1})
        assert results.levels == (0, 1)
        assert results.dim == 2

    def test_levels_for_qutrits(self):
        results = Results({'00': 1, '02': 1, '21': 1})
        assert results.levels == (0, 1, 2)
        assert results.dim == 3

    def test_ev_all_ground_state(self):
        results = Results({'0': 100})
        assert results.ev == 1.0

    def test_ev_all_excited_state(self):
        results = Results({'1': 100})
        assert results.ev == -1.0

    def test_ev_mixed(self):
        # 60% even-parity ('00', '11'), 40% odd-parity ('01', '10').
        results = Results({'00': 30, '11': 30, '01': 20, '10': 20})
        assert results.ev == pytest.approx(0.2)

    def test_ev_raises_for_empty_results(self):
        with pytest.raises(ValueError, match='empty'):
            Results().ev

    def test_ev_raises_for_leaked_states(self):
        # '02' has even 1-parity but must not be counted as +1.
        results = Results({'00': 50, '02': 50})
        with pytest.raises(ValueError, match='binary'):
            results.ev

    def test_ev_raises_for_zero_total_counts(self):
        results = Results({'0': 5, '1': -5})
        with pytest.raises(ValueError, match='zero total'):
            results.ev

    def test_entropy_of_certain_outcome_is_zero(self):
        results = Results({'0': 100})
        assert results.entropy.nominal_value == pytest.approx(0.)

    def test_entropy_of_uniform_distribution_is_one_bit(self):
        results = Results({'0': 50, '1': 50})
        assert results.entropy.nominal_value == pytest.approx(1.0)


class TestExpectationValue:

    def test_default_matches_ev(self):
        results = Results({'00': 30, '11': 30, '01': 20, '10': 20})
        assert results.expectation_value() == pytest.approx(results.ev)

    def test_bell_state_correlations(self):
        results = Results({'00': 50, '11': 50})
        assert results.expectation_value() == pytest.approx(1.0)
        assert results.expectation_value(0) == pytest.approx(0.0)
        assert results.expectation_value(1) == pytest.approx(0.0)

    def test_single_index(self):
        results = Results({'01': 70, '11': 30})
        assert results.expectation_value(0) == pytest.approx(0.4)
        assert results.expectation_value(1) == pytest.approx(-1.0)

    def test_matches_marginalize(self):
        rng = np.random.default_rng(0)
        states = [''.join(b) for b in itertools.product('01', repeat=3)]
        results = Results(
            dict(zip(states, rng.integers(1, 100, len(states)).tolist()))
        )
        for idx in [0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2), (2, 0)]:
            assert results.expectation_value(idx) == pytest.approx(
                results.marginalize(idx).ev
            )

    def test_quasi_probabilities(self):
        # Negative (quasi-probabilistic) counts are weighted linearly.
        results = Results({'0': 110, '1': -10})
        assert results.expectation_value() == pytest.approx(1.2)

    @pytest.mark.parametrize('idx', [(0, 0), (2,), -1])
    def test_invalid_indices_raise(self, idx):
        results = Results({'00': 50, '11': 50})
        with pytest.raises(ValueError, match='distinct'):
            results.expectation_value(idx)


class TestGetItem:

    def test_getitem_existing_bitstring(self):
        results = Results({'0': 90, '1': 10})
        series = results['0']
        assert series['counts'] == 90
        assert series['probabilities'] == pytest.approx(0.9)

    def test_getitem_missing_bitstring_returns_zeros(self):
        results = Results({'0': 90, '1': 10})
        series = results['11']
        assert series['counts'] == 0
        assert series['probabilities'] == 0.


class TestMarginalize:

    def test_marginalize_single_index(self):
        results = Results({'000': 200, '010': 10, '100': 12, '111': 200})
        marginal = results.marginalize(0)
        assert isinstance(marginal, Results)
        assert dict(marginal.dict) == {'0': 210, '1': 212}

    def test_marginalize_multiple_indices(self):
        results = Results({'000': 200, '010': 10, '100': 12, '111': 200})
        marginal = results.marginalize((0, 2))
        assert dict(marginal.dict) == {'00': 210, '10': 12, '11': 200}

    def test_marginalize_preserves_total_shots(self):
        results = Results({'000': 200, '010': 10, '100': 12, '111': 200})
        marginal = results.marginalize(1)
        assert marginal.n_shots == results.n_shots


class TestTVD:

    def test_tvd_of_identical_distributions_is_zero(self):
        results = Results({'0': 50, '1': 50})
        assert results.tvd(results).nominal_value == pytest.approx(0.)

    def test_tvd_of_disjoint_distributions_is_one(self):
        a = Results({'0': 100})
        b = Results({'1': 100})
        assert a.tvd(b).nominal_value == pytest.approx(1.0)

    def test_tvd_requires_matching_dimension(self):
        a = Results({'0': 10, '1': 10})
        b = Results({'00': 5, '11': 5})
        with pytest.raises(ValueError):
            a.tvd(b)


class TestFidelity:

    def test_fidelity_of_identical_distributions_is_one(self):
        results = Results({'0': 50, '1': 50})
        assert results.fidelity(results).nominal_value == pytest.approx(1.)

    def test_fidelity_of_disjoint_distributions_is_zero(self):
        a = Results({'0': 100})
        b = Results({'1': 100})
        assert a.fidelity(b).nominal_value == pytest.approx(0.)

    def test_fidelity_requires_matching_dimension(self):
        a = Results({'0': 10, '1': 10})
        b = Results({'00': 5, '11': 5})
        with pytest.raises(ValueError):
            a.fidelity(b)


class TestReadoutCorrection:

    def test_apply_readout_correction_recovers_prepared_state(self):
        # Confusion matrix: rows = prepared state, columns = measured state.
        # P(measure 0 | prep 0) = 0.9, P(measure 1 | prep 0) = 0.1, etc.
        cmat = pd.DataFrame(
            [[0.9, 0.1], [0.05, 0.95]],
            index=[0, 1],
            columns=pd.MultiIndex.from_tuples(
                [('Meas State', 0), ('Meas State', 1)]
            ),
        )
        # Raw measured distribution exactly matches the |0> row of cmat.
        results = Results({'0': 900, '1': 100})
        results.apply_readout_correction(cmat)
        assert results.dict['0'] == 1000
        assert results.dict['1'] == 0

    def test_apply_readout_correction_via_constructor(self):
        cmat = pd.DataFrame(
            [[0.9, 0.1], [0.05, 0.95]],
            index=[0, 1],
            columns=pd.MultiIndex.from_tuples(
                [('Meas State', 0), ('Meas State', 1)]
            ),
        )
        results = Results({'0': 900, '1': 100}, confusion_matrix=cmat)
        assert results.dict['0'] == 1000
        assert results.dict['1'] == 0

    def test_mismatched_confusion_matrix_falls_back_to_raw_results(self):
        # A single-qudit confusion matrix applied to 2-qudit results.
        cmat = pd.DataFrame(
            [[0.9, 0.1], [0.05, 0.95]],
            index=[0, 1],
            columns=pd.MultiIndex.from_tuples(
                [('Meas State', 0), ('Meas State', 1)]
            ),
        )
        raw = {'00': 80, '01': 10, '10': 5, '11': 5}
        results = Results(dict(raw))
        results.apply_readout_correction(cmat)
        assert dict(results.dict) == raw


class TestReprAndDf:

    def test_df_has_counts_and_probabilities_rows(self):
        results = Results({'0': 90, '1': 10})
        assert list(results.df.index) == ['counts', 'probabilities']

    def test_repr_and_str_do_not_raise(self):
        results = Results({'0': 90, '1': 10})
        assert 'counts' in repr(results)
        assert 'counts' in str(results)


# ---------------------------------------------------------------------------
# Readout correction with a multi-qubit confusion matrix
# ---------------------------------------------------------------------------

# Per-qubit confusion matrices in qcal's row-stochastic convention:
# C[prep, meas] = P(meas | prep). They differ so that a swapped tensor
# product order would not recover a delta distribution.
C0 = np.array([[0.9, 0.1], [0.2, 0.8]])
C1 = np.array([[0.95, 0.05], [0.1, 0.9]])


def _single_qubit_cmat(cmat: np.ndarray) -> pd.DataFrame:
    """Confusion matrix in the single-qudit ('Meas State', level) format.

    Args:
        cmat (np.ndarray): 2x2 row-stochastic confusion matrix.

    Returns:
        pd.DataFrame: confusion matrix.
    """
    return pd.DataFrame(
        cmat,
        index=[0, 1],
        columns=pd.MultiIndex.from_tuples(
            [('Meas State', 0), ('Meas State', 1)]
        ),
    )


def _multi_qubit_cmat(cmats: dict) -> pd.DataFrame:
    """Confusion matrix in the format produced by ReadoutFidelity.

    Args:
        cmats (dict): mapping of qubit label to 2x2 confusion matrix.

    Returns:
        pd.DataFrame: confusion matrix with (Qq, 'Meas State', level)
            columns.
    """
    columns = pd.MultiIndex.from_tuples([
        (f'Q{q}', 'Meas State', level) for q in cmats for level in (0, 1)
    ])
    index = pd.MultiIndex.from_tuples([('Prep State', 0), ('Prep State', 1)])
    return pd.DataFrame(
        np.hstack(list(cmats.values())), index=index, columns=columns
    )


def _measured_counts(true_state: str, n_shots: int) -> dict:
    """Counts measured for a 2-qubit basis state under C0 (x) C1 noise.

    The leftmost bit is qubit 0, so the measured distribution of a
    prepared basis state is the corresponding row of kron(C0, C1).

    Args:
        true_state (str): prepared 2-qubit basis state, e.g. '01'.
        n_shots (int): number of shots.

    Returns:
        dict: (exact, noiseless-sampling) measured counts.
    """
    row = np.kron(C0, C1)[int(true_state, 2)]
    return {
        s: int(round(p * n_shots))
        for s, p in zip(('00', '01', '10', '11'), row, strict=True)
    }


class TestMultiQubitReadoutCorrection:

    def test_measured_counts_helper_matches_hand_calculation(self):
        # Prep '01': q0 reads 0 w.p. 0.9, q1 reads 1 w.p. 0.9.
        assert _measured_counts('01', 10000) == {
            '00': 900, '01': 8100, '10': 100, '11': 900
        }

    @pytest.mark.parametrize('true_state', ['00', '01', '10', '11'])
    def test_recovers_prepared_basis_state(self, true_state):
        cmat = _multi_qubit_cmat({0: C0, 1: C1})
        results = Results(_measured_counts(true_state, 10000))
        corrected = readout_correction(results, cmat)

        for state, count in corrected.items():
            expected = 10000 if state == true_state else 0
            assert count == pytest.approx(expected, abs=1)

    def test_swapped_qubit_order_does_not_recover_state(self):
        # Sanity check of the bit-ordering convention: assigning C0 to
        # qubit 1 instead of qubit 0 must not give back the delta.
        cmat = _multi_qubit_cmat({0: C1, 1: C0})
        results = Results(_measured_counts('01', 10000))
        corrected = readout_correction(results, cmat)
        assert abs(corrected['00']) > 100

    def test_corrected_counts_are_rounded_not_truncated(self):
        cmat = _multi_qubit_cmat({0: C0, 1: C1})
        results = Results(_measured_counts('01', 10000))
        corrected = readout_correction(results, cmat)
        assert corrected['01'] == 10000

    def test_qubit_order_in_cmat_columns_is_sorted_numerically(self):
        # Q10 must come after Q2 (numeric, not lexicographic, order).
        cmat = _multi_qubit_cmat({10: C1, 2: C0})
        results = Results(_measured_counts('10', 10000))
        corrected = readout_correction(results, cmat)
        assert corrected['10'] == pytest.approx(10000, abs=1)

    def test_partial_correction_matches_hand_inverse(self):
        # 70/30 measured with C0 -> [0.7, 0.3] @ inv(C0) * 1000.
        results = Results({'0': 700, '1': 300})
        corrected = readout_correction(results, _single_qubit_cmat(C0))
        expected = np.array([0.7, 0.3]) @ np.linalg.inv(C0) * 1000
        assert corrected['0'] == pytest.approx(expected[0], abs=1)
        assert corrected['1'] == pytest.approx(expected[1], abs=1)

    def test_mismatched_cmat_returns_input_unchanged(self):
        results = Results({'000': 10, '111': 10})
        cmat = _multi_qubit_cmat({0: C0, 1: C1})
        assert readout_correction(results, cmat) is results.dict

    def test_level_outside_cmat_returns_input_unchanged(self):
        # A leaked '2' outcome cannot be corrected by 2x2 matrices.
        results = Results({'00': 10, '21': 10})
        cmat = _multi_qubit_cmat({0: C0, 1: C1})
        assert readout_correction(results, cmat) is results.dict


class TestApplyReadoutCorrection:

    def test_sets_confusion_matrix_property(self):
        cmat = _multi_qubit_cmat({0: C0, 1: C1})
        results = Results(_measured_counts('10', 10000))
        assert results.confusion_matrix is None
        results.apply_readout_correction(cmat)
        assert results.confusion_matrix is cmat
        assert results.rr_char_results is None

    def test_raw_results_are_preserved(self):
        raw = _measured_counts('10', 10000)
        results = Results(dict(raw))
        results.apply_readout_correction(_multi_qubit_cmat({0: C0, 1: C1}))
        assert results.raw_results == raw
        assert results.dict != raw

    def test_raw_results_returns_a_copy(self):
        results = Results({'0': 90, '1': 10})
        results.raw_results['0'] = 0
        assert results.raw_results == {'0': 90, '1': 10}

    def test_repeated_application_does_not_compound(self):
        cmat = _multi_qubit_cmat({0: C0, 1: C1})
        results = Results(_measured_counts('10', 10000))
        results.apply_readout_correction(cmat)
        once = dict(results.dict)
        results.apply_readout_correction(cmat)
        assert dict(results.dict) == once

    def test_probabilities_reflect_corrected_counts(self):
        cmat = _multi_qubit_cmat({0: C0, 1: C1})
        results = Results(_measured_counts('10', 10000))
        _ = results.probabilities  # build the cached DataFrame first
        results.apply_readout_correction(cmat)
        assert results.probabilities['10'] == pytest.approx(1., abs=1e-3)

    def test_mismatched_cmat_leaves_confusion_matrix_unset(self):
        results = Results({'00': 80, '11': 20})
        results.apply_readout_correction(_single_qubit_cmat(C0))
        assert results.confusion_matrix is None
        assert dict(results.dict) == {'00': 80, '11': 20}

    def test_corrects_results_with_a_single_observed_level(self):
        # [1, 0] @ inv(C0) * 1000 = [1142.86, -142.86]
        results = Results({'0': 1000})
        cmat = _single_qubit_cmat(C0)
        results.apply_readout_correction(cmat)
        assert results.confusion_matrix is cmat
        assert results.dict['0'] == pytest.approx(1000 * 0.8 / 0.7, abs=1)


# ---------------------------------------------------------------------------
# Quasi-probabilistic readout correction (QPRC)
# ---------------------------------------------------------------------------

def _independent_error_dist(eps: list, scale: float = 1e6) -> dict:
    """Bit-flip error distribution for independent per-bit flips.

    Args:
        eps (list): flip probability of each bit, leftmost first.
        scale (float, optional): total weight. Defaults to 1e6.

    Returns:
        dict: error bitstring -> weight.
    """
    dist = {}
    for bits in itertools.product((0, 1), repeat=len(eps)):
        p = np.prod([e if b else 1 - e for b, e in zip(bits, eps,
                                                       strict=True)])
        dist[''.join(map(str, bits))] = p * scale
    return dist


def _xor_shift(dist: dict, state: str) -> dict:
    """Measured distribution of a basis state under a bit-flip channel.

    Args:
        dist (dict): error bitstring -> weight.
        state (str): prepared basis state.

    Returns:
        dict: measured bitstring -> weight.
    """
    n = len(state)
    return {
        format(int(k, 2) ^ int(state, 2), f'0{n}b'): v
        for k, v in dist.items()
    }


class TestQPReadoutCorrection:

    def test_single_qubit_order_one_is_exact(self):
        # q = (0.9 d0 - 0.1 d1) / 0.8; [100, 900] * q -> [0, 1000]
        corrected = qp_readout_correction(
            {'0': 100, '1': 900}, {'0': 900, '1': 100}, order=1
        )
        assert corrected == {'1': 1000}

    def test_recovers_three_qubit_basis_state(self):
        p = _independent_error_dist([0.02, 0.05, 0.1])
        measured = _xor_shift(p, '101')
        assert qp_readout_correction(measured, p) == {'101': 1000000}

    def test_residual_error_decreases_with_order(self):
        p = _independent_error_dist([0.02, 0.05, 0.1])
        measured = _xor_shift(p, '101')
        errors = []
        for order in (1, 2, 3):
            corrected = qp_readout_correction(
                measured, p, order=order, max_iterations=1
            )
            assert set(corrected) == {'101'}
            errors.append(abs(corrected['101'] - 1000000))
        assert errors[0] > errors[1] > errors[2]

    def test_no_clipping_preserves_total_weight(self):
        p = _independent_error_dist([0.02, 0.05, 0.1])
        measured = {'000': 3e5, '011': 7e5}
        measured = {
            s: sum(
                w * p[format(int(s, 2) ^ int(k, 2), '03b')]
                for k, w in measured.items()
            ) / 1e6
            for s in p
        }
        corrected = qp_readout_correction(
            measured, p, clip_to_zero=False, max_iterations=1, order=1
        )
        assert sum(corrected.values()) == pytest.approx(1e6, abs=8)

    def test_recovers_mixture_of_states(self):
        p = _independent_error_dist([0.03, 0.06])
        true = {'00': 0.25, '11': 0.75}
        measured = {
            s: sum(
                w * p[format(int(s, 2) ^ int(t, 2), '02b')]
                for t, w in true.items()
            )
            for s in p
        }
        # The default tol stops once the residual error is <= 1e-3, which
        # leaves a ~1e-5 relative bias; a tighter tol is exact.
        corrected = qp_readout_correction(measured, p)
        assert set(corrected) == {'00', '11'}
        assert corrected['00'] == pytest.approx(250000, rel=1e-3)
        assert corrected['11'] == pytest.approx(750000, rel=1e-3)

        corrected = qp_readout_correction(
            measured, p, tol=1e-8, max_iterations=50
        )
        assert corrected == {'00': 250000, '11': 750000}

    def test_accepts_results_objects(self):
        corrected = qp_readout_correction(
            Results({'0': 100, '1': 900}),
            Results({'0': 900, '1': 100}),
            order=1,
        )
        assert corrected == {'1': 1000}

    def test_no_observed_errors_is_identity(self):
        counts = {'00': 10, '01': 20, '10': 30, '11': 40}
        assert qp_readout_correction(counts, {'00': 500}) == counts

    def test_bitstrings_longer_than_64_bits(self):
        n = 70
        zero = '0' * n
        one = '1' + '0' * (n - 1)
        corrected = qp_readout_correction(
            {one: 900, zero: 100}, {zero: 900, one: 100}, order=1
        )
        assert corrected == {one: 1000}

    def test_default_partitions_handle_low_all_zero_probability(self):
        # p0 = 0.7 ** 2 = 0.49 <= 1/2, so the global inverse alone is
        # undefined; the per-qubit partitions make it invertible.
        p = _independent_error_dist([0.3, 0.3])
        assert p['00'] / 1e6 < 0.5
        corrected = qp_readout_correction(_xor_shift(p, '11'), p)
        assert corrected == {'11': 1000000}

    def test_low_all_zero_probability_without_partitions_raises(self):
        p = _independent_error_dist([0.3, 0.3])
        with pytest.raises(ValueError, match='greater than 0.5'):
            qp_readout_correction(_xor_shift(p, '11'), p, partitions=[])

    def test_explicit_partitions(self):
        p = _independent_error_dist([0.3, 0.3])
        corrected = qp_readout_correction(
            _xor_shift(p, '01'), p, partitions=[[(0,), (1,)]]
        )
        assert corrected == {'01': 1000000}

    @pytest.mark.parametrize('partitions', [
        [[(0,), (0, 1)]],  # not disjoint
        [[(0,), (2,)]],    # out of range
    ])
    def test_invalid_partitions_raise(self, partitions):
        p = _independent_error_dist([0.01, 0.01])
        with pytest.raises(ValueError, match='disjoint'):
            qp_readout_correction(p, p, partitions=partitions)

    def test_order_less_than_one_raises(self):
        with pytest.raises(ValueError, match='order'):
            qp_readout_correction({'0': 1}, {'0': 1}, order=0)

    def test_max_iterations_less_than_one_raises(self):
        with pytest.raises(ValueError, match='max_iterations'):
            qp_readout_correction({'0': 1}, {'0': 1}, max_iterations=0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match='does not match'):
            qp_readout_correction({'00': 1}, {'0': 1})

    def test_non_binary_bitstrings_raise(self):
        with pytest.raises(ValueError, match='binary'):
            qp_readout_correction({'02': 1}, {'00': 1})

    def test_inconsistent_bitstring_lengths_raise(self):
        with pytest.raises(ValueError, match='same length'):
            qp_readout_correction({'0': 1, '00': 1}, {'00': 1})


class TestApplyQPReadoutCorrection:

    def test_constructor_applies_correction(self):
        rr = {'0': 900, '1': 100}
        results = Results({'0': 100, '1': 900}, rr_char_results=rr)
        assert dict(results.dict) == {'1': 1000}
        assert results.raw_results == {'0': 100, '1': 900}
        assert results.rr_char_results is rr
        assert results.confusion_matrix is None

    def test_constructor_rejects_both_corrections(self):
        with pytest.raises(ValueError, match='Only one'):
            Results(
                {'0': 100, '1': 900},
                confusion_matrix=_single_qubit_cmat(C0),
                rr_char_results={'0': 900, '1': 100},
            )

    def test_repeated_application_does_not_compound(self):
        results = Results({'0': 100, '1': 900})
        results.apply_qp_readout_correction({'0': 900, '1': 100})
        results.apply_qp_readout_correction({'0': 900, '1': 100})
        assert dict(results.dict) == {'1': 1000}

    def test_n_shots_and_probabilities_are_updated(self):
        results = Results({'0': 100, '1': 900})
        _ = results.probabilities  # build the cached DataFrame first
        results.apply_qp_readout_correction({'0': 900, '1': 100})
        assert results.n_shots == 1000
        assert results.probabilities['1'] == pytest.approx(1.)

    def test_failure_falls_back_to_raw_results(self):
        # p0 = 0.4 for a single qubit has no inverse.
        results = Results({'0': 100, '1': 900})
        results.apply_qp_readout_correction({'0': 900, '1': 100})
        results.apply_qp_readout_correction({'0': 400, '1': 600})
        assert dict(results.dict) == {'0': 100, '1': 900}
        assert results.rr_char_results is None

    def test_qp_correction_replaces_confusion_matrix_correction(self):
        results = Results({'0': 900, '1': 100})
        results.apply_readout_correction(_single_qubit_cmat(C0))
        assert results.confusion_matrix is not None
        rr = {'0': 900, '1': 100}
        results.apply_qp_readout_correction(rr)
        assert results.confusion_matrix is None
        assert results.rr_char_results is rr
        assert dict(results.dict) == {'0': 1000}

    def test_cmat_correction_replaces_qp_correction(self):
        results = Results({'0': 900, '1': 100})
        results.apply_qp_readout_correction({'0': 900, '1': 100})
        results.apply_readout_correction(_single_qubit_cmat(C0))
        assert results.rr_char_results is None
        assert results.confusion_matrix is not None

    def test_forwards_order_and_iterations(self):
        p = _independent_error_dist([0.02, 0.05, 0.1])
        measured = _xor_shift(p, '101')
        results = Results(measured)
        results.apply_qp_readout_correction(p, order=1, max_iterations=1)
        assert dict(results.dict) == qp_readout_correction(
            measured, p, order=1, max_iterations=1
        )


class TestPlot:

    def test_plot_does_not_raise(self):
        Results({'0': 90, '1': 10}).plot()


# ---------------------------------------------------------------------------
# PerShotResults
# ---------------------------------------------------------------------------

def _qec_cycles() -> list:
    """Per-shot cycles of a circuit where shot 1 failed heralding.

    Ancilla 1 is measured twice mid-circuit, then data qubit 0 is
    measured at the end.

    Returns:
        list: one DataFrame per measurement cycle, indexed by shot.
    """
    index = [0, 2, 3]
    return [
        pd.DataFrame({1: [0, 1, 0]}, index=index),
        pd.DataFrame({1: [1, 0, 0]}, index=index),
        pd.DataFrame({0: [1, 1, 1]}, index=index),
    ]


class TestPerShotResults:

    def test_from_cycles_layout(self):
        per_shot = PerShotResults.from_cycles(_qec_cycles(), n_shots=4)
        assert per_shot.columns == ((0, 1), (1, 1), (2, 0))
        np.testing.assert_array_equal(
            per_shot.outcomes, [[0, 1, 1], [1, 0, 1], [0, 0, 1]]
        )
        assert per_shot.outcomes.dtype == np.int8
        assert per_shot.kept.tolist() == [True, False, True, True]
        assert per_shot.shot_index.tolist() == [0, 2, 3]
        assert per_shot.n_shots == 3
        assert per_shot.n_measurement_cycles == 3

    def test_n_shots_defaults_to_one_past_last_kept_shot(self):
        per_shot = PerShotResults.from_cycles(_qec_cycles())
        assert per_shot.kept.tolist() == [True, False, True, True]

    def test_trailing_discarded_shots_are_recorded(self):
        per_shot = PerShotResults.from_cycles(_qec_cycles(), n_shots=6)
        assert per_shot.kept.tolist() == [
            True, False, True, True, False, False
        ]
        assert per_shot.n_shots == 3

    def test_multi_qubit_cycles_span_multiple_columns(self):
        cycles = [
            pd.DataFrame({1: [0, 1], 3: [1, 1]}),
            pd.DataFrame({0: [0, 0], 1: [1, 0], 2: [1, 1]}),
        ]
        per_shot = PerShotResults.from_cycles(cycles)
        assert per_shot.columns == ((0, 1), (0, 3), (1, 0), (1, 1), (1, 2))
        assert per_shot.outcomes.shape == (2, 5)
        assert per_shot.n_measurement_cycles == 2

    def test_cycle_results_round_trip(self):
        cycles = _qec_cycles()
        per_shot = PerShotResults.from_cycles(cycles, n_shots=4)
        for got, expected in zip(
            per_shot.cycle_results, cycles, strict=True
        ):
            assert list(got.index) == list(expected.index)
            assert list(got.columns) == list(expected.columns)
            np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())

    def test_measurement_record_excluding_final_cycle(self):
        per_shot = PerShotResults.from_cycles(_qec_cycles(), n_shots=4)
        record = per_shot.measurement_record(include_final=False)
        assert list(record.columns) == [(0, 1), (1, 1)]
        assert record.loc[2].tolist() == [1, 0]

    def test_measurement_record_of_single_cycle_without_final(self):
        per_shot = PerShotResults.from_cycles(
            [pd.DataFrame({0: [0, 1, 1]})]
        )
        record = per_shot.measurement_record(include_final=False)
        assert record.shape == (3, 0)
        assert list(record.index) == [0, 1, 2]

    def test_record_reproduces_aggregate_counts(self):
        rng = np.random.default_rng(0)
        index = np.sort(rng.choice(500, size=300, replace=False))
        cycles = [
            pd.DataFrame(
                {q: rng.integers(0, 2, size=300) for q in qubits},
                index=index,
            )
            for qubits in ((1, 3), (1, 3), (0, 1, 2, 3, 4))
        ]
        per_shot = PerShotResults.from_cycles(cycles, n_shots=500)
        assert per_shot.kept.sum() == 300

        for i, cycle in enumerate(cycles):
            expected = Results(
                cycle.astype(str).agg(''.join, axis=1)
                .value_counts().to_dict()
            )
            got = Results(
                per_shot.measurement_record()[i].astype(str)
                .agg(''.join, axis=1).value_counts().to_dict()
            )
            assert dict(got.dict) == dict(expected.dict)
            assert got.n_shots == 300

    def test_values_that_overflow_int8_are_not_compacted(self):
        per_shot = PerShotResults.from_cycles(
            [pd.DataFrame({0: [0, 300]})]
        )
        assert per_shot.outcomes.dtype != np.int8
        assert per_shot.outcomes[1, 0] == 300

    def test_empty_cycles_with_n_shots(self):
        per_shot = PerShotResults.from_cycles([], n_shots=5)
        assert per_shot.kept.tolist() == [False] * 5
        assert per_shot.shot_index.tolist() == []
        assert per_shot.n_measurement_cycles == 0

    def test_non_increasing_shots_raise(self):
        cycles = [pd.DataFrame({0: [0, 1, 1]}, index=[0, 2, 1])]
        with pytest.raises(ValueError, match='increasing'):
            PerShotResults.from_cycles(cycles)

    def test_shots_beyond_n_shots_raise(self):
        with pytest.raises(ValueError, match='indexed in'):
            PerShotResults.from_cycles(_qec_cycles(), n_shots=3)

    def test_negative_shots_raise(self):
        cycles = [pd.DataFrame({0: [0, 1]}, index=[-1, 0])]
        with pytest.raises(ValueError, match='indexed in'):
            PerShotResults.from_cycles(cycles, n_shots=2)

    def test_is_immutable(self):
        per_shot = PerShotResults.from_cycles(_qec_cycles())
        with pytest.raises(dataclasses.FrozenInstanceError):
            per_shot.outcomes = np.zeros((1, 1))


class TestCircuitPerShotResults:

    def test_defaults_to_none(self, bell_circuit):
        assert bell_circuit.per_shot_results is None

    def test_setter_stores_per_shot_results(self, bell_circuit):
        per_shot = PerShotResults.from_cycles(_qec_cycles())
        bell_circuit.per_shot_results = per_shot
        assert bell_circuit.per_shot_results is per_shot
