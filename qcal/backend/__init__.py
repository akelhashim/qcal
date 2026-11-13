"""Hardware and emulated backends for qcal.

This subpackage provides concrete :class:`~qcal.qpu.qpu.QPU` subclasses,
including the simulator-backed :class:`~qcal.backend.emulator.Emulator`
and the QubiC backend (:mod:`qcal.backend.qubic`).
"""
from .emulator import Emulator
