"""Calibration protocols for superconducting qubits.

This subpackage provides the abstract
:class:`~qcal.calibration.calibration.Calibration` base class and
concrete calibration routines for single-qubit gates
(:mod:`~qcal.calibration.single_qubit`), two-qubit CZ gates
(:mod:`~qcal.calibration.cz`), readout (:mod:`~qcal.calibration.readout`),
leakage (:mod:`~qcal.calibration.leakage`), and generic parameter sweeps
(:mod:`~qcal.calibration.sweep`). Each calibration is built on top of a
:class:`~qcal.qpu.qpu.QPU` and writes optimal values back to the
:class:`~qcal.config.Config`.
"""
