"""Compilation of qcal circuits to native gate sets.

This subpackage provides the default :class:`Compiler` (a wrapper around
BQSKit), single-qubit gate decompositions (e.g. the ZXZXZ decompositions
in :data:`ZXZXZ_DECOMPOSITIONS`), utilities for merging adjacent gates,
and Pauli conjugation tools used for randomized compiling.
"""
from qcal.compilation.compiler import Compiler
from qcal.compilation.decompositions import ZXZXZ_DECOMPOSITIONS

__all__ = ('Compiler', 'ZXZXZ_DECOMPOSITIONS')
