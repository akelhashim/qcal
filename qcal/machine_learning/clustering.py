# """Submodule for supervised and unsupervised clustering methods.


# """
# from __future__ import annotations

# import logging
# import numpy as np

# from numpy.random import RandomState
# from sklearn.mixture import GaussianMixture
# from typing import ArrayLike, Float, Int, Literal 

# logger = logging.getLogger(__name__)


# class GMM(GaussianMixture):


#     def __init__(
#             self, 
#             n_components: Int = 1, 
#             *, 
#             covariance_type: Literal[
#                     'full', 'tied', 'diag', 'spherical'
#                 ] = "full", 
#             tol: Float = 0.001, 
#             reg_covar: Float = 0.000001, 
#             max_iter: Int = 100, 
#             n_init: Int = 1, 
#             init_params: Literal[
#                     'kmeans', 'k-means++', 'random', 'random_from_data'
#                 ] = "kmeans", 
#             weights_init: ArrayLike | None = None, 
#             means_init: ArrayLike | None = None, 
#             precisions_init: ArrayLike | None = None, 
#             random_state: Int | RandomState | None = None, 
#             warm_start: bool = False, 
#             verbose: Int = 0, 
#             verbose_interval: Int = 10
#         ) -> None:
#         super().__init__(
#             n_components, 
#             covariance_type=covariance_type, 
#             tol=tol, 
#             reg_covar=reg_covar, 
#             max_iter=max_iter, 
#             n_init=n_init, 
#             init_params=init_params, 
#             weights_init=weights_init, 
#             means_init=means_init, 
#             precisions_init=precisions_init, 
#             random_state=random_state, 
#             warm_start=warm_start, 
#             verbose=verbose, 
#             verbose_interval=verbose_interval)
        
#         self._label_mapper = {
#             n: n for n in range(n_components)
#         }
        
#     def save(self, filepath: str) -> None:
#         """Save the parameters of the GMM.

#         Args:
#             filepath (str): path where GMM parameters are saved.
#         """
#         np.savez(
#             filepath + 'gmm_model.npz',
#             covariances=self.covariances_,
#             means=self.means_,
#             params=self.get_params(),
#             weights=self.weights_
#         )

#     @staticmethod
#     def load(filename: str) -> GMM:
#         """Load a GMM model from saved parameters.

#         Args:
#             filename (str): filename of the saved model.
#         """
#         model = np.load(filename)
#         # gmm = GMM(len(model['means']))
#         gmm = GMM(*model['params'][0])
#         gmm.precisions_cholesky_ = np.linalg.cholesky(
#             np.linalg.inv(model['covariances'])
#         )
#         gmm.covariances_ = model['covariances']
#         gmm.means_ = model['means']
#         gmm.weights_ = model['weights']
        
#         return gmm
        


