"""Interfaces between qcal and third-party quantum software packages.

Each subpackage provides compilers and/or transpilers that map qcal
circuits to and from an external framework: BQSKit
(:mod:`qcal.interface.bqskit`), pyGSTi (:mod:`qcal.interface.pygsti`),
PyQuil (:mod:`qcal.interface.pyquil`), Superstaq
(:mod:`qcal.interface.superstaq`), and True-Q
(:mod:`qcal.interface.trueq`). Some of these packages are optional
dependencies and must be installed separately.
"""
