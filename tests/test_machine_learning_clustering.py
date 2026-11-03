"""Unit tests for qcal.machine_learning.clustering.GMM."""
import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from qcal.machine_learning.clustering import GMM


def _two_cluster_data(seed=0, offset=5.0):
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.standard_normal((50, 2)) + np.array([0, 0]),
        rng.standard_normal((50, 2)) + np.array([offset, offset]),
    ])


class TestConstruction:

    def test_is_a_gaussian_mixture(self):
        assert isinstance(GMM(), GaussianMixture)

    def test_defaults_before_fitting(self):
        gmm = GMM()
        assert gmm.is_fitted is False
        assert gmm.snr == {}
        assert gmm.X is None
        assert gmm.y is None

    def test_default_n_components_and_covariance_type(self):
        gmm = GMM()
        assert gmm.n_components == 2
        assert gmm.covariance_type == 'spherical'

    def test_custom_kwargs_forwarded_to_base_class(self):
        gmm = GMM(n_components=3, covariance_type='diag', random_state=0)
        assert gmm.n_components == 3
        assert gmm.covariance_type == 'diag'
        assert gmm.random_state == 0


class TestFit:

    def test_fit_sets_fitted_state(self):
        X = _two_cluster_data()
        gmm = GMM(n_components=2)
        gmm.fit(X)

        assert gmm.is_fitted is True
        assert gmm.X is X
        assert gmm.y is None

    def test_fit_stores_labels(self):
        X = _two_cluster_data()
        y = np.array([0] * 50 + [1] * 50)
        gmm = GMM(n_components=2)
        gmm.fit(X, y)

        assert gmm.y is y

    def test_fit_populates_snr(self):
        gmm = GMM(n_components=2)
        gmm.fit(_two_cluster_data())

        assert set(gmm.snr.keys()) == {'01'}

    def test_fit_with_single_component_gives_no_snr_pairs(self):
        gmm = GMM(n_components=1)
        gmm.fit(_two_cluster_data())

        assert gmm.snr == {}


class TestCalculateSnr:
    """calculate_snr() only reads self.n_components/means_/covariances_,
    so its formula can be checked directly without a real fit.
    """

    def test_two_components(self):
        gmm = GMM(n_components=2)
        gmm.means_ = np.array([[0.0], [3.0]])
        gmm.covariances_ = np.array([1.0, 1.0])

        gmm.calculate_snr()

        # dist = |3 - 0| = 3, err = (sqrt(1) + sqrt(1)) * 2 = 4
        assert gmm.snr == {'01': pytest.approx(0.75)}

    def test_three_components_keys_and_values(self):
        gmm = GMM(n_components=3)
        gmm.means_ = np.array([[0.0], [3.0], [6.0]])
        gmm.covariances_ = np.array([1.0, 1.0, 4.0])

        gmm.calculate_snr()

        assert set(gmm.snr.keys()) == {'01', '02', '12'}
        assert gmm.snr['01'] == pytest.approx(3 / 4)
        assert gmm.snr['02'] == pytest.approx(6 / 6)
        assert gmm.snr['12'] == pytest.approx(3 / 6)


class TestRelabel:

    def test_reorders_fitted_parameters(self):
        gmm = GMM(n_components=2)
        gmm.fit(_two_cluster_data())
        means_before = gmm.means_.copy()
        covariances_before = gmm.covariances_.copy()
        weights_before = gmm.weights_.copy()
        precisions_before = gmm.precisions_.copy()
        precisions_chol_before = gmm.precisions_cholesky_.copy()

        gmm.relabel([1, 0])

        np.testing.assert_allclose(gmm.means_, means_before[[1, 0]])
        np.testing.assert_allclose(
            gmm.covariances_, covariances_before[[1, 0]]
        )
        np.testing.assert_allclose(gmm.weights_, weights_before[[1, 0]])
        np.testing.assert_allclose(
            gmm.precisions_, precisions_before[[1, 0]]
        )
        np.testing.assert_allclose(
            gmm.precisions_cholesky_, precisions_chol_before[[1, 0]]
        )


class TestSave:

    def test_save_writes_expected_arrays(self, tmp_path):
        gmm = GMM(n_components=2)
        gmm.fit(_two_cluster_data())

        gmm.save(str(tmp_path) + '/')

        saved_file = tmp_path / 'gmm_model.npz'
        assert saved_file.exists()

        saved = np.load(saved_file, allow_pickle=True)
        assert set(saved.files) == {
            'covariances', 'means', 'params', 'weights'
        }
        np.testing.assert_allclose(saved['means'], gmm.means_)
        np.testing.assert_allclose(saved['covariances'], gmm.covariances_)
        np.testing.assert_allclose(saved['weights'], gmm.weights_)


class TestLoad:

    @pytest.mark.parametrize(
        'covariance_type', ['spherical', 'diag', 'tied', 'full']
    )
    def test_round_trip_recovers_fitted_parameters(
        self, tmp_path, covariance_type
    ):
        gmm = GMM(n_components=2, covariance_type=covariance_type)
        gmm.fit(_two_cluster_data())
        gmm.save(str(tmp_path) + '/')

        loaded = GMM.load(str(tmp_path / 'gmm_model.npz'))

        assert loaded.covariance_type == covariance_type
        np.testing.assert_allclose(loaded.means_, gmm.means_)
        np.testing.assert_allclose(loaded.covariances_, gmm.covariances_)
        np.testing.assert_allclose(loaded.weights_, gmm.weights_)
        np.testing.assert_allclose(
            loaded.precisions_cholesky_, gmm.precisions_cholesky_,
            atol=1e-8,
        )


class TestCalculateSnrAllCovarianceTypes:

    @pytest.mark.parametrize(
        'covariance_type', ['spherical', 'diag', 'tied', 'full']
    )
    def test_snr_is_finite(self, covariance_type):
        gmm = GMM(n_components=2, covariance_type=covariance_type)
        gmm.fit(_two_cluster_data())

        assert np.isfinite(gmm.snr['01'])

    def test_full_covariance_with_negative_correlation_gives_finite_snr(
        self
    ):
        # Regression test: calculate_snr() used to do
        # np.sqrt(self.covariances_[clusters]) directly on full
        # covariance matrices, which silently produced nan whenever a
        # cluster had negative off-diagonal (correlation) terms,
        # instead of extracting the per-feature variances first.
        rng = np.random.default_rng(1)
        corr = np.array([[1, -0.8], [-0.8, 1]])
        X = np.vstack([
            rng.standard_normal((100, 2)) @ corr + np.array([0, 0]),
            rng.standard_normal((100, 2)) @ corr + np.array([5, 5]),
        ])
        gmm = GMM(n_components=2, covariance_type='full')

        gmm.fit(X)

        assert np.isfinite(gmm.snr['01'])
