"""Unit tests for qcal.results: Results and readout_correction."""
import pandas as pd
import pytest

from qcal.results import Results


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

    def test_entropy_of_certain_outcome_is_zero(self):
        results = Results({'0': 100})
        assert results.entropy == 0.

    def test_entropy_of_uniform_distribution_is_one_bit(self):
        results = Results({'0': 50, '1': 50})
        assert results.entropy == pytest.approx(1.0)


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
