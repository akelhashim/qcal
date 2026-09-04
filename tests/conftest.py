"""Shared fixtures for the qcal test suite."""
from pathlib import Path

import pytest

from qcal.circuit import Circuit, Cycle
from qcal.config import Config
from qcal.gates.single_qubit import H, X, Y
from qcal.gates.two_qubit import CNOT

EXAMPLE_CONFIG_PATH = str(
    Path(__file__).resolve().parents[1] / 'examples' / 'config' /
    'config.yaml'
)


@pytest.fixture
def config():
    """A fresh Config loaded from the example config.yaml."""
    return Config(EXAMPLE_CONFIG_PATH)


@pytest.fixture
def x0():
    """A Pauli X gate on qubit 0."""
    return X(0)


@pytest.fixture
def y1():
    """A Pauli Y gate on qubit 1."""
    return Y(1)


@pytest.fixture
def bell_circuit():
    """A 2-qubit circuit: H(0) followed by CNOT(0, 1)."""
    return Circuit([
        Cycle({H(0)}),
        Cycle(CNOT(0, 1)),
    ])
