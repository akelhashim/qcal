"""Unit tests for qcal.managers.classification_manager."""
import numpy as np
import pytest

from qcal.machine_learning.clustering import GMM
from qcal.managers.classification_manager import ClassificationManager


def _two_cluster_data(seed=0, offset=5.0):
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.standard_normal((50, 2)) + np.array([0, 0]),
        rng.standard_normal((50, 2)) + np.array([offset, offset]),
    ])


class TestConstruction:

    def test_builds_one_gmm_per_qubit(self):
        cm = ClassificationManager(qubits=(0, 1, 2))

        assert set(cm.model.keys()) == {0, 1, 2}
        assert all(isinstance(m, GMM) for m in cm.model.values())

    def test_n_levels_sets_n_components(self):
        cm = ClassificationManager(qubits=(0,), n_levels=3)

        assert cm.model[0].n_components == 3

    def test_kwargs_forwarded_to_gmm(self):
        cm = ClassificationManager(qubits=(0,), covariance_type='diag')

        assert cm.model[0].covariance_type == 'diag'

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match='gmm'):
            ClassificationManager(qubits=(0,), model='bogus')

    @pytest.mark.parametrize('bad_model', ['', 'g', 'm', 'gm', 'mm'])
    def test_substrings_of_gmm_are_rejected(self, bad_model):
        # Regression test: the original validation was
        # `assert model in ('gmm')`, where `('gmm')` is just the
        # string 'gmm' (no trailing comma -> no tuple), making it a
        # substring check that silently accepted any of these.
        with pytest.raises(ValueError, match='gmm'):
            ClassificationManager(qubits=(0,), model=bad_model)


class TestGetItem:

    def test_returns_model_for_qubit(self):
        cm = ClassificationManager(qubits=(0, 1))
        assert cm[1] is cm.model[1]


class TestFit:

    def test_fit_delegates_to_underlying_model(self):
        cm = ClassificationManager(qubits=(0,))
        X = _two_cluster_data()

        cm.fit(0, X)

        assert cm[0].is_fitted is True


class TestPredict:

    def test_predict_returns_labels_for_each_sample(self):
        cm = ClassificationManager(qubits=(0,))
        X = _two_cluster_data()
        cm.fit(0, X)

        labels = cm.predict(0, X)

        assert len(labels) == len(X)
        assert set(labels) <= {0, 1}
