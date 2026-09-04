"""Submodule for math on probability distributions.

"""
import itertools
import logging

import numpy as np
from uncertainties import ufloat

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


def total_variation_distance(results1, results2) -> ufloat:
    """Total Variation Distance.

    tvd(p, q) = 0.5 * sum_i |p_i - q_i|

    Only bitstrings observed in either results1 or results2 are
    summed over, rather than every bitstring in the full
    levels ** n_qudits Hilbert space (unobserved bitstrings have
    zero population in both, so they don't contribute). This keeps
    the cost proportional to the number of distinct measured
    outcomes instead of exponential in n_qudits.

    Each population p_i (q_i) is treated as an independent binomial
    proportion with standard error sqrt(p_i * (1 - p_i) / n_shots),
    and these per-bitstring errors are propagated (ignoring the
    typically small covariance between bitstrings induced by the
    multinomial normalization) to give the uncertainty on the
    returned tvd.

    Args:
        results1 (Results): qcal Results object.
        results2 (Results): qcal Results object.

    Returns:
        ufloat: total variation distance between results1 and
            results2, with propagated shot-noise uncertainty.
    """
    assert results1.n_qudits == results2.n_qudits, (
        'The results objects must be the same dimension!'
    )
    pop1, pop2 = results1.populations, results2.populations
    n1, n2 = results1.n_shots, results2.n_shots
    dtstrs = set(pop1.keys()) | set(pop2.keys())

    terms = []
    for dtstr in dtstrs:
        p1, p2 = pop1[dtstr], pop2[dtstr]
        up1 = ufloat(p1, np.sqrt(p1 * (1. - p1) / n1))
        up2 = ufloat(p2, np.sqrt(p2 * (1. - p2) / n2))
        diff = up1 - up2
        # abs()/umath.fabs() on a ufloat are deprecated as of
        # uncertainties>=3.2; negate instead, which still propagates
        # the error correctly since |x| and -x share the same std_dev.
        terms.append(diff if diff.nominal_value >= 0 else -diff)

    return sum(terms) / 2.
