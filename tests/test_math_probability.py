"""Unit tests for qcal.math.probability."""
import pytest

from qcal.math.probability import hellinger_fidelity, total_variation_distance
from qcal.results import Results


class TestHellingerFidelity:

    def test_identical_certain_outcomes_give_fidelity_one(self):
        f = hellinger_fidelity(Results({'00': 100}), Results({'00': 100}))
        assert f.nominal_value == pytest.approx(1.0)
        assert f.std_dev == pytest.approx(0.0)

    def test_disjoint_distributions_give_fidelity_zero(self):
        f = hellinger_fidelity(Results({'00': 100}), Results({'11': 100}))
        assert f.nominal_value == pytest.approx(0.0)
        assert f.std_dev == pytest.approx(0.0)

    def test_identical_uniform_distributions_give_fidelity_one(self):
        f = hellinger_fidelity(
            Results({'00': 50, '01': 50}), Results({'00': 50, '01': 50})
        )
        assert f.nominal_value == pytest.approx(1.0)
        assert f.std_dev == pytest.approx(0.1)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match='dimension'):
            hellinger_fidelity(Results({'0': 10}), Results({'00': 10}))

    def test_dict_inputs_match_results_inputs(self):
        counts1, counts2 = {'00': 75, '01': 25}, {'00': 25, '01': 75}
        f_dict = hellinger_fidelity(counts1, Results(counts2))
        f_res = hellinger_fidelity(Results(counts1), Results(counts2))
        assert f_dict.nominal_value == pytest.approx(f_res.nominal_value)
        assert f_dict.std_dev == pytest.approx(f_res.std_dev)

    def test_ideal_distribution_adds_no_uncertainty(self):
        # Probabilities sum to a single "shot", which would otherwise give
        # each population an uncertainty of 0.5.
        noisy, ideal = Results({'00': 50, '01': 50}), {'00': 0.5, '01': 0.5}
        f = hellinger_fidelity(noisy, ideal)
        assert f.nominal_value == pytest.approx(1.0)
        assert f.std_dev == pytest.approx(0.0707, abs=1e-4)
        f_rev = hellinger_fidelity(ideal, noisy)
        assert f_rev.std_dev == pytest.approx(f.std_dev)

    def test_single_shot_distribution_adds_no_uncertainty(self):
        f = hellinger_fidelity(Results({'00': 50, '01': 50}), {'00': 1})
        assert f.nominal_value == pytest.approx(0.5)
        assert f.std_dev == pytest.approx(0.05)

    def test_identical_ideal_distributions_give_exact_fidelity_one(self):
        f = hellinger_fidelity({'00': 0.5, '01': 0.5}, {'00': 0.5, '01': 0.5})
        assert f.nominal_value == pytest.approx(1.0)
        assert f.std_dev == pytest.approx(0.0)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match='Results object or dict'):
            hellinger_fidelity([('00', 10)], {'00': 10})


class TestTotalVariationDistance:

    def test_identical_distributions_give_tvd_zero(self):
        t = total_variation_distance(
            Results({'00': 100}), Results({'00': 100})
        )
        assert t.nominal_value == pytest.approx(0.0)
        assert t.std_dev == pytest.approx(0.0)

    def test_disjoint_distributions_give_tvd_one(self):
        t = total_variation_distance(
            Results({'00': 100}), Results({'11': 100})
        )
        assert t.nominal_value == pytest.approx(1.0)
        assert t.std_dev == pytest.approx(0.0)

    def test_partially_overlapping_distributions(self):
        t = total_variation_distance(
            Results({'00': 75, '01': 25}), Results({'00': 25, '01': 75})
        )
        assert t.nominal_value == pytest.approx(0.5)
        assert t.std_dev == pytest.approx(0.0433, abs=1e-4)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match='dimension'):
            total_variation_distance(
                Results({'0': 10}), Results({'00': 10})
            )

    def test_dict_inputs_match_results_inputs(self):
        counts1, counts2 = {'00': 75, '01': 25}, {'00': 25, '01': 75}
        t_dict = total_variation_distance(counts1, counts2)
        t_res = total_variation_distance(Results(counts1), Results(counts2))
        assert t_dict.nominal_value == pytest.approx(t_res.nominal_value)
        assert t_dict.std_dev == pytest.approx(t_res.std_dev)

    def test_ideal_distribution_adds_no_uncertainty(self):
        noisy, ideal = Results({'00': 75, '01': 25}), {'00': 0.5, '01': 0.5}
        t = total_variation_distance(noisy, ideal)
        assert t.nominal_value == pytest.approx(0.25)
        assert t.std_dev == pytest.approx(0.0306, abs=1e-4)

    def test_identical_ideal_distributions_give_exact_tvd_zero(self):
        t = total_variation_distance(
            {'00': 0.5, '01': 0.5}, {'00': 0.5, '01': 0.5}
        )
        assert t.nominal_value == pytest.approx(0.0)
        assert t.std_dev == pytest.approx(0.0)
