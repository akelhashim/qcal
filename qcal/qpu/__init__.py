"""Quantum Processing Unit (QPU) execution layer.

This subpackage provides the :class:`~qcal.qpu.qpu.QPU` base class, which
compiles, transpiles, sequences, measures, and post-processes circuits.
Hardware backends (see :mod:`qcal.backend`) subclass it and override the
hardware-specific methods.
"""
