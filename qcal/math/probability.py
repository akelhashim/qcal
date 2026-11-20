"""Submodule for math on probability distributions.

"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict

from uncertainties import ufloat
from uncertainties.umath import sqrt as usqrt

from qcal.math.utils import nominal, population, to_ufloat

if TYPE_CHECKING:
    from qcal.results import Results

logger = logging.getLogger(__name__)


def hellinger_fidelity(
    results1: Results | Dict, results2: Results | Dict
) -> ufloat:
    """Classical (Hellinger) fidelity.

    fidelity(p, q) = (sum_i sqrt(p_i * q_i))^2

    Only bitstrings observed in both results1 and results2 can
    contribute, since sqrt(p_i * q_i) is exactly 0 whenever either
    population is 0 (rather than every bitstring in the full
    levels ** n_qudits Hilbert space). This keeps the cost
    proportional to the number of distinct measured outcomes instead
    of exponential in n_qudits.

    Each population p_i (q_i) is treated as an independent binomial
    proportion with standard error sqrt(p_i * (1 - p_i) / n_shots),
    and these per-bitstring errors are propagated (ignoring the
    typically small covariance between bitstrings induced by the
    multinomial normalization) to give the uncertainty on the
    returned fidelity. Ideal distributions (see
    qcal.results.is_ideal) are treated as exact and contribute no
    uncertainty. Bitstrings with zero population in either
    distribution are skipped rather than propagated through sqrt(),
    whose derivative diverges at 0 and would otherwise turn an exact
    zero contribution into a NaN uncertainty.

    Args:
        results1 (Results | Dict): qcal Results object, or a dictionary of
            bitstring -> count (converted to a Results object).
        results2 (Results | Dict): qcal Results object, or a dictionary of
            bitstring -> count (converted to a Results object).

    Returns:
        ufloat: Hellinger fidelity between results1 and results2,
            with propagated shot-noise uncertainty.
    """
    # Imported here to avoid a circular import, since qcal.results
    # imports this module.
    from qcal.results import is_ideal, to_results

    results1, results2 = to_results(results1), to_results(results2)
    if results1.n_qudits != results2.n_qudits:
        raise ValueError('The results objects must be the same dimension!')

    pop1, pop2 = results1.populations, results2.populations
    ideal1, ideal2 = is_ideal(results1), is_ideal(results2)
    n1, n2 = results1.n_shots, results2.n_shots
    dtstrs = set(pop1.keys()) | set(pop2.keys())

    terms = []
    for dtstr in dtstrs:
        p1, p2 = pop1[dtstr], pop2[dtstr]
        if p1 == 0. or p2 == 0.:
            continue
        up1 = population(p1, n1, ideal1)
        up2 = population(p2, n2, ideal2)
        terms.append(usqrt(up1 * up2))

    if not terms:
        return ufloat(0., 0.)
    return to_ufloat(sum(terms) ** 2)


def total_variation_distance(
    results1: Results | Dict, results2: Results | Dict
) -> ufloat:
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
    returned tvd. Ideal distributions (see qcal.results.is_ideal) are
    treated as exact and contribute no uncertainty.

    Args:
        results1 (Results | Dict): qcal Results object, or a dictionary of
            bitstring -> count (converted to a Results object).
        results2 (Results | Dict): qcal Results object, or a dictionary of
            bitstring -> count (converted to a Results object).

    Returns:
        ufloat: total variation distance between results1 and
            results2, with propagated shot-noise uncertainty.
    """
    # Imported here to avoid a circular import, since qcal.results
    # imports this module.
    from qcal.results import is_ideal, to_results

    results1, results2 = to_results(results1), to_results(results2)
    if results1.n_qudits != results2.n_qudits:
        raise ValueError('The results objects must be the same dimension!')

    pop1, pop2 = results1.populations, results2.populations
    ideal1, ideal2 = is_ideal(results1), is_ideal(results2)
    n1, n2 = results1.n_shots, results2.n_shots
    dtstrs = set(pop1.keys()) | set(pop2.keys())

    terms = []
    for dtstr in dtstrs:
        p1, p2 = pop1[dtstr], pop2[dtstr]
        up1 = population(p1, n1, ideal1)
        up2 = population(p2, n2, ideal2)
        diff = up1 - up2
        # abs()/umath.fabs() on a ufloat are deprecated as of
        # uncertainties>=3.2; negate instead, which still propagates
        # the error correctly since |x| and -x share the same std_dev.
        terms.append(diff if nominal(diff) >= 0 else -diff)

    return to_ufloat(sum(terms) / 2.)
