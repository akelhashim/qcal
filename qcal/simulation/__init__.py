"""Quantum circuit simulation for qcal.

This subpackage provides noiseless statevector and noisy density-matrix
simulators (:class:`StateVectorSimulator`,
:class:`DensityMatrixSimulator`) backed by ``quax``, along with
category-based error models (e.g. :class:`DepolarizingNoise`,
:class:`RelaxationNoise`, :class:`CustomErrorModel`) that can be
attached to the density-matrix simulator.
"""
from .error_models import (
    SINGLE_QUBIT_NAMES,
    SINGLE_QUTRIT_NAMES,
    TWO_QUBIT_NAMES,
    TWO_QUTRIT_NAMES,
    AmplitudeDamping,
    BitFlipNoise,
    CustomErrorModel,
    DephasingNoise,
    DepolarizingNoise,
    ErrorModel,
    LeakageNoise,
    PhaseFlipNoise,
    RelaxationNoise,
    RelaxationParams,
    SeepageNoise,
    UnitaryError,
    gate_category,
)
from .simulators import (
    DensityMatrixSimulator,
    Simulator,
    StateVectorSimulator,
)
