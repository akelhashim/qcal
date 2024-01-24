"""Submodule for supervised and unsupervised clustering methods.


"""
from __future__ import annotations

from .utils import find_mapping

import logging
import numpy as np

from numpy.typing import ArrayLike
from sklearn.mixture import GaussianMixture
from typing import Dict, List

logger = logging.getLogger(__name__)


class GMM(GaussianMixture):

    def __init__(
            self,
            n_components:    int = 2,
            covariance_type: str = 'spherical',
            **kwargs
        ) -> None:
        """Gaussian Mixture Model

        This class inherits from scikit-learn's GaussianMixture class.

        Args:
            n_components (int, optional): The number of mixture components. 
                Defaults to 2.
            covariance_type (str, optional): String describing the type of 
                covariance parameters to use. Defaults to 'spherical'.
        """
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            **kwargs
        )
        self.__X = None
        self.__y = None
        self._snr = {}
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether or not the model has been fitted.

        Returns:
            bool: fitted or not.
        """
        return self._is_fitted
    
    @property
    def snr(self) -> Dict:
        """Signal to noise ratio.

        Returns:
            Dict: SNR for each pair of states..
        """
        return self._snr

    @property
    def X(self) -> ArrayLike:
        """Data used for fitting.

        Returns:
            ArrayLike: input data.
        """
        return self.__X
    
    @property
    def y(self) -> ArrayLike:
        """Classification labels.

        Returns:
            ArrayLike: labels for data.
        """
        return self.__y

    def fit(self, X: ArrayLike, y: ArrayLike = None) -> None:
        """Fit a Guassian Mixture Model to the input data.

        Args:
            X (ArrayLike): array-like of shape (n_samples, n_features). List of 
                n_features-dimensional data points. Each row corresponds to a 
                single data point.
            y (ArrayLike): fit labels for each data point. Defaults to None.
        """
        super().fit(X)
        self.__X = X
        self.__y = y

        # if y is not None and not np.all(self.predict(X) == y):
        #     mapping = find_mapping(self.predict(X), y)
        #     permutation = np.zeros(self.n_components).astype(int)
        #     for i, j in mapping.items():
        #         permutation[j] = i
            
        #     self.relabel(permutation)

        self._is_fitted = True
        self.calculate_snr()

    def relabel(self, permutation: List[int]) -> None:
        """Relable the clusters.

        Args:
            permutation (List[int]): list of index labels specifying the new
                order.
        """
        self.covariances_ = self.covariances_[permutation]
        self.means_ = self.means_[permutation]
        self.weights_ = self.weights_[permutation]
        self.precisions_ = self.precisions_[permutation]
        self.precisions_cholesky_ = self.precisions_cholesky_[permutation]

    def calculate_snr(self) -> None:
        """Calculate the signal-to-noise between each pair of clusters."""
        for clusters in np.array(np.triu_indices(self.n_components, 1)).T:
            dist = np.sqrt(np.sum(np.diff(self.means_[clusters], axis=0) ** 2))
            err = np.sum(np.sqrt(self.covariances_[clusters]) * 2)
            self._snr["{0}{1}".format(*clusters)] = dist / err
        
    def save(self, filepath: str) -> None:
        """Save the parameters of the GMM.

        Args:
            filepath (str): path where GMM parameters are saved.
        """
        np.savez(
            filepath + 'gmm_model.npz',
            covariances=self.covariances_,
            means=self.means_,
            params=self.get_params(),
            weights=self.weights_
        )

    @staticmethod
    def load(filename: str) -> GMM:
        """Load a GMM model from saved parameters.

        Args:
            filename (str): filename of the saved model.
        """
        model = np.load(filename)
        # gmm = GMM(len(model['means']))
        gmm = GMM(*model['params'][0])
        gmm.precisions_cholesky_ = np.linalg.cholesky(
            np.linalg.inv(model['covariances'])
        )
        gmm.covariances_ = model['covariances']
        gmm.means_ = model['means']
        gmm.weights_ = model['weights']
        
        return gmm
        


