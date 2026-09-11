"""Shared fixtures for the qcal test suite."""
from pathlib import Path

import pytest

import qcal.settings as settings
from qcal.circuit import Circuit, Cycle
from qcal.config import Config
from qcal.gates.single_qubit import H, X, Y
from qcal.gates.two_qubit import CNOT

EXAMPLE_CONFIG_PATH = str(
    Path(__file__).resolve().parents[1] / 'examples' / 'config' /
    'config.yaml'
)


@pytest.fixture(autouse=True, scope='session')
def _no_plot_popups():
    """Never let a test pop open a browser tab or notebook widget.

    Some protocols (e.g. CRB) call `.plot()` internally as part of
    `.run()`. Plotly auto-detects a renderer based on the environment,
    which resolves to 'browser' in a plain terminal/CI run -- without
    this, running the test suite would silently open a real browser
    tab (or one per plot) on whatever machine runs it. 'json' is inert:
    it just serializes the figure, so it can't open anything or
    require an extra dependency.
    """
    original = settings.Settings.plot_renderer
    settings.Settings.plot_renderer = 'json'
    yield
    settings.Settings.plot_renderer = original


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
