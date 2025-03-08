"""Submodule for math on probability distributions.

"""
import itertools
import logging
import numpy as np

logger = logging.getLogger(__name__)


def hellinger_fidelity(results1, results2) -> float:
    """Classical (Hellinger) fidelity.

    fidelity(p, q) = (sum_i sqrt(p_i * q_i))^2

    Args:
        results1 (Results): qcal Results object.
        results2 (Results): qcal Results object.

    Returns:
        float: Hellinger fidelity between results1 and results2.
    """
    assert results1.n_qudits == results2.n_qudits, (
        'The results objects must be the same dimension!'
    )
    n_qudits = results1.n_qudits
    levels = list(set(results1.levels + results2.levels))
    dtstrs = [
        ''.join(i) for i in itertools.product(
            [str(j) for j in levels], repeat=n_qudits
        )
    ]

    fid = 0.
    for dtstr in dtstrs:
        fid += np.sqrt(
            results1.populations[dtstr] * results2.populations[dtstr]
        )
    fid *= fid

    return fid


def total_variation_distance(results1, results2) -> float:
    """Total Variation Distance.

    tvd(p, q) = 0.5 * sum_i |p_i - q_i|

    Args:
        results1 (Results): qcal Results object.
        results2 (Results): qcal Results object.

    Returns:
        float: total variation distance between results1 and results2.
    """
    assert results1.n_qudits == results2.n_qudits, (
        'The results objects must be the same dimension!'
    )
    n_qudits = results1.n_qudits
    levels = list(set(results1.levels + results2.levels))
    dtstrs = [
        ''.join(i) for i in itertools.product(
            [str(j) for j in levels], repeat=n_qudits
        )
    ]

    tvd = 0.
    for dtstr in dtstrs:
        tvd += abs(results1.populations[dtstr] - results2.populations[dtstr])
    tvd /= 2.

    return tvd