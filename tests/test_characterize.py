"""Unit tests for qcal.characterization.characterize.Characterize.

Characterize is an ABC (analyze() and generate_circuits() are
abstract), so it can't be instantiated directly -- every test below
that needs an instance goes through a trivial concrete subclass
instead. final()/plot()/set_param() and the shared state they operate
on are implemented directly on the base class, so they're tested here
rather than through a real characterization protocol: the per-qubit
dicts they read (`_param_sweep`, `_results`, `_fit`, `_char_values`,
`_errors`) and `_params`/`_qubits` are set directly on the instance,
exactly as a real subclass's generate_circuits()/analyze() would
populate them. `_fit[q]` only needs `.fit_success` and `.predict(x)`,
so a minimal stand-in is used instead of a real Fit subclass.
"""
import numpy as np
import pytest

from qcal.characterization.characterize import Characterize


class _FakeFit:
    """Minimal stand-in for a qcal.fitting.fit.Fit subclass."""

    def __init__(self, fit_success, value=0.5):
        self.fit_success = fit_success
        self._value = value

    def predict(self, x):
        return np.full_like(x, self._value)


class _ConcreteCharacterize(Characterize):
    """Minimal concrete subclass for exercising base-class behavior."""

    def analyze(self) -> None:
        pass

    def generate_circuits(self) -> None:
        pass


@pytest.fixture
def char(config):
    return _ConcreteCharacterize(config)


class TestAbstractness:

    def test_cannot_instantiate_without_implementations(self, config):
        with pytest.raises(TypeError):
            Characterize(config)

    def test_subclass_implementing_methods_is_instantiable(self, config):
        _ConcreteCharacterize(config)  # should not raise


class TestCharacterizeConstruction:

    def test_default_attributes(self, char):
        assert char.qubits is None
        assert char.gate is None
        assert char.params is None
        assert char.subspace is None
        assert char.param_sweep == {}
        assert char.results == {}

    def test_characterized_values_defaults_to_false_for_missing_qubit(
        self, char
    ):
        assert char.characterized_values[0] is False


class TestSetParam:

    def test_set_param_updates_config(self, char, config):
        char.set_param('single_qubit/0/GE/freq', 123.0)
        assert config['single_qubit/0/GE/freq'] == 123.0

    def test_set_param_on_bogus_path_raises_and_logs(self, char, caplog):
        with pytest.raises(TypeError):
            char.set_param('bogus/nested/path', 1.0)
        assert 'Could not set bogus/nested/path' in caplog.text


class TestFinal:

    def test_uses_char_value_when_fit_succeeds(self, char, config):
        char._qubits = [0]
        char._params = {0: 'single_qubit/0/GE/freq'}
        char._fit = {0: _FakeFit(fit_success=True)}
        char._char_values[0] = 999.0

        char.final()

        assert config['single_qubit/0/GE/freq'] == 999.0

    def test_uses_char_value_when_no_fit_dict(self, char, config):
        char._qubits = [0]
        char._params = {0: 'single_qubit/0/GE/freq'}
        char._char_values[0] = 555.0

        char.final()

        assert config['single_qubit/0/GE/freq'] == 555.0

    def test_leaves_config_unchanged_without_fit_or_char_value(
        self, char, config
    ):
        char._qubits = [0]
        char._params = {0: 'single_qubit/0/GE/freq'}
        original = config['single_qubit/0/GE/freq']

        char.final()

        assert config['single_qubit/0/GE/freq'] == original


class TestPlot:

    def test_single_qubit_with_successful_fit(self, char):
        char._qubits = [0]
        char._param_sweep = {0: np.linspace(0, 1, 10)}
        char._results = {0: np.linspace(0, 1, 10) ** 2}
        char._fit = {0: _FakeFit(fit_success=True)}
        char._char_values[0] = 2e-6
        char._errors = {0: 1e-7}

        char.plot()

    def test_single_qubit_without_fit(self, char):
        char._qubits = [0]
        char._param_sweep = {0: np.linspace(0, 1, 10)}
        char._results = {0: np.linspace(0, 1, 10)}
        char._fit = {0: _FakeFit(fit_success=False)}

        char.plot()

    def test_multi_qubit_grid_with_leftover_empty_axes(self, char):
        # 5 qubits -> a 2x4 grid (calculate_nrows_ncols), leaving 3
        # empty subplots to be turned off.
        qubits = [0, 1, 2, 3, 4]
        char._qubits = qubits
        char._param_sweep = {q: np.linspace(0, 1, 10) for q in qubits}
        char._results = {q: np.linspace(0, 1, 10) for q in qubits}
        char._fit = {q: _FakeFit(fit_success=True) for q in qubits}
        for q in qubits:
            char._char_values[q] = 1e-6
        char._errors = dict.fromkeys(qubits, 1e-08)

        char.plot()
