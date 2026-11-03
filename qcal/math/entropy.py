"""
Submodule for calculating the entropy of distributions of dit-strings.

"""
import logging

import numpy as np
from uncertainties import ufloat, umath

logger = logging.getLogger(__name__)


def classical_entropy(results) -> ufloat:
    """Function for calculating the classical entropy of a distribution.

    H_d(p) = -sum_i p_i * log_d(p_i)

    This is the Shannon entropy generalized to d dimensions: the log
    base d is set to results.dim (e.g. 2 for qubits, 3 for qutrits),
    so the result is measured in dits rather than always in bits.

    States with zero population are skipped rather than propagated
    through log_d(), whose derivative diverges at 0 and would
    otherwise turn an exact zero contribution into a NaN uncertainty.
    A dimension of less than 2 (i.e. a single observed level, so the
    distribution is certain) has no valid log base and trivially has
    zero entropy, so it's returned directly.

    Each population p_i is treated as an independent binomial
    proportion with standard error sqrt(p_i * (1 - p_i) / n_shots).

    Args:
        results (Results): qcal Results object.

    Returns:
        ufloat: classical entropy, with propagated shot-noise
            uncertainty.
    """
    d = results.dim
    if d < 2:
        return ufloat(0., 0.)

    n = results.n_shots
    p = np.asarray(results.probabilities, dtype=float)
    p = p[p > 0]

    terms = [
        ufloat(pi, np.sqrt(pi * (1. - pi) / n)) for pi in p
    ]
    if not terms:
        return ufloat(0., 0.)

    return -sum(up * umath.log(up, d) for up in terms)
