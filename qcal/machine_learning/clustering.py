"""Submodule for supervised and unsupervised clustering methods.


"""
from __future__ import annotations

import logging
import numpy as np

from sklearn.mixture import GaussianMixture

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
        


