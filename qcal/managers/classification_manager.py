"""Submodule for managing the classification of data.

The classification of data is handled by the ClassificationManager class.
"""
from qcal.machine_learning.clustering import GMM

import logging

from numpy.typing import ArrayLike, NDArray
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


class ClassificationManager:
    """Manager for handling how raw IQ results are classified."""

    def __init__(self, 
            qubits:   List | Tuple,
            n_levels: int = 2,
            model:    str = 'gmm',
            **kwargs
        ) -> None:
        """Classification manager.

        This class stores a classification model for each qubit. This model can
        be saved and loaded for future use.

        Args:
            qubits (List | Tuple): list of qubits.
            n_levels (int, optional): number of states to classify. Defaults to 
                2.
            model (str, optional): classification algorithm. Defaults to 'gmm'.
        """
        
        self._qubits = qubits
        self._n_levels = n_levels
        
        assert model in ('gmm'), (
             "Only 'gmm' classifier is currently supported!"
        )

        if model == 'gmm':
            self._model = {
                q: GMM(n_components=n_levels, **kwargs) for q in qubits
            }

    def __getitem__(self, qubit: int) -> Any:
        """Classification model used for each qubit.

        Args:
            qubit (int): qubit label.

        Returns:
            Any: qubit classification model.
        """
        return self._model[qubit]

    @property
    def model(self) -> Any:
        """Model used for clustering and classifying the data.

        Returns:
            Any: model.
        """
        return self._model

    def fit(
            self, qubit: int, X: ArrayLike, y: ArrayLike | None = None
        ) -> None:
        """Fit the data.

        Args:
            qubit (int): qubit label.
            X (ArrayLike): data to fit. Array-like of shape 
                (n_samples, n_features).
            y (ArrayLike | None, optional): classification labels. Defaults to 
                None.
        """
        self._model[qubit].fit(X, y)
        
    def predict(self, qubit: int, X: ArrayLike) -> NDArray: 
        """Predict the classification labels for an array of data.

        Args:
        qubit (int): qubit label.
            X (ArrayLike): data to predict.

        Returns:
            NDArray: classification labels.
        """
        return self._model[qubit].predict(X)
