""" 
Submodule for calculating the entropy of distributions of dit-strings.

"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


# TODO: generalize to d-dimensions
def shannon_entropy(results) -> float:
    """Function for calculating the Shannon entropy of a discrete distribution.

    Args:
        results (Results): qcal Results object.

    Returns:
        float: Shannon entropy.
    """
    entropy = 0.
    for state in results.states:
        entropy -= (
           results[state].probabilities * np.log2(results[state].probabilities)
        )
    
    return entropy