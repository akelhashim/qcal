"""Unit tests for qcal.math.entropy.classical_entropy."""
import warnings

import numpy as np
import pytest

from qcal.math.entropy import classical_entropy
from qcal.results import Results


class TestClassicalEntropy:

    def test_certain_outcome_is_zero(self):
        results = Results({'000': 100})
        entropy = classical_entropy(results)
        assert entropy.nominal_value == pytest.approx(0.)
        assert entropy.std_dev == pytest.approx(0.)

    def test_uniform_two_state_qubit_distribution_is_one_bit(self):
        results = Results({'000': 50, '111': 50})
        entropy = classical_entropy(results)
        assert entropy.nominal_value == pytest.approx(1.0)

    def test_uniform_four_state_qubit_distribution_is_two_bits(self):
        results = Results(
            {'000': 25, '001': 25, '010': 25, '011': 25}
        )
        entropy = classical_entropy(results)
        assert entropy.nominal_value == pytest.approx(2.0)

    def test_uniform_three_state_qutrit_distribution_is_one_dit(self):
        # Generalization to d dimensions: the log base is results.dim
        # (3 for a qutrit), so a uniform 3-outcome distribution gives
        # exactly 1.0, the same way a uniform 2-outcome qubit
        # distribution gives 1.0 bit.
        results = Results({'0': 100, '1': 100, '2': 100})
        entropy = classical_entropy(results)
        assert results.dim == 3
        assert entropy.nominal_value == pytest.approx(1.0)

    def test_matches_manual_calculation_for_mixed_distribution(self):
        results = Results(
            {'000': 200, '010': 10, '100': 12, '111': 200}
        )
        n = 200 + 10 + 12 + 200
        probs = [200 / n, 10 / n, 12 / n, 200 / n]
        expected = -sum(p * np.log2(p) for p in probs)

        entropy = classical_entropy(results)

        assert entropy.nominal_value == pytest.approx(expected)

    def test_uncertainty_is_propagated(self):
        # Each state's binomial shot-noise error
        # sqrt(p * (1 - p) / n) should propagate through log_d() to a
        # nonzero uncertainty on the entropy, consistent with
        # qcal.math.probability.hellinger_fidelity/
        # total_variation_distance.
        results = Results({'000': 50, '111': 50})
        entropy = classical_entropy(results)
        assert entropy.std_dev > 0.

    def test_explicit_zero_probability_state_is_ignored_not_nan(self):
        # Regression test: a state present in the results dict with a
        # count of 0 used to produce `0 * log2(0) == nan` (via
        # 0 * -inf), silently corrupting the whole entropy value
        # instead of being treated as a zero-weight contribution.
        with warnings.catch_warnings():
            warnings.simplefilter('error', category=RuntimeWarning)
            entropy = classical_entropy(Results({'000': 0, '010': 10}))

        assert entropy.nominal_value == pytest.approx(0.)

    def test_single_observed_level_does_not_crash(self):
        # A single observed level (results.dim == 1) has no valid log
        # base (log_1(x) divides by log(1) == 0), but is trivially a
        # certain/deterministic distribution, so entropy is defined as
        # zero rather than raising ZeroDivisionError.
        results = Results({'000': 100})
        assert results.dim == 1

        with warnings.catch_warnings():
            warnings.simplefilter('error', category=RuntimeWarning)
            entropy = classical_entropy(results)

        assert entropy.nominal_value == pytest.approx(0.)
