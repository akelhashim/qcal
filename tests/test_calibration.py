"""Unit tests for qcal.calibration.calibration.Calibration.

Calibration is meant to be subclassed (analyze/generate_circuits raise
NotImplementedError here), but final()/plot()/set_param() and the
shared state they operate on are implemented directly on the base
class, so they're tested here rather than through a subclass.
"""
import logging

import matplotlib.pyplot as plt
import pytest

import qcal.settings as settings
from qcal.calibration.calibration import Calibration
from qcal.config import Config


class _FakeFit:
    """Minimal stand-in for a qcal.fitting.fit.Fit subclass."""

    def __init__(self, fit_success: bool) -> None:
        self.fit_success = fit_success

    def predict(self, x):
        return x


class _ConcreteCalibration(Calibration):
    """Minimal concrete subclass for exercising base-class behavior.

    Calibration is an ABC (analyze() is abstract), so it can't be
    instantiated directly -- every test below that needs an instance
    goes through this trivial subclass instead.
    """

    def analyze(self) -> None:
        pass


@pytest.fixture
def cal(config):
    return _ConcreteCalibration(config)


@pytest.fixture(autouse=True)
def _close_figures():
    """Prevent matplotlib's open-figure count from growing unbounded."""
    yield
    plt.close('all')


@pytest.fixture(autouse=True)
def _restore_save_data():
    original = settings.Settings.save_data
    yield
    settings.Settings.save_data = original


class TestInit:

    def test_defaults(self, cal, config):
        assert cal._config is config
        assert cal.gate is None
        assert cal.params is None
        assert cal.subspace is None
        assert cal.qubits is None
        assert cal.param_sweep == {}
        assert cal.sweep_results == {}
        assert cal.calibrated_values == {}

    def test_calibrated_values_defaults_unseen_qubit_to_false(self, cal):
        assert cal.calibrated_values[99] is False

    def test_warns_when_esp_enabled(self, config, caplog):
        config['readout/esp/enable'] = True
        with caplog.at_level(logging.WARNING):
            _ConcreteCalibration(config)
        assert any(
            'Excited State Promotion' in r.message for r in caplog.records
        )

    def test_no_warning_when_esp_disabled(self, config, caplog):
        assert config['readout/esp/enable'] is False
        with caplog.at_level(logging.WARNING):
            _ConcreteCalibration(config)
        assert not any(
            'Excited State Promotion' in r.message for r in caplog.records
        )


class TestAbstractness:

    def test_cannot_instantiate_without_analyze(self, config):
        with pytest.raises(TypeError):
            Calibration(config)

    def test_subclass_implementing_analyze_is_instantiable(self, config):
        _ConcreteCalibration(config)  # should not raise


class TestUnimplementedMethods:

    def test_generate_circuits_raises_not_implemented(self, cal):
        with pytest.raises(NotImplementedError):
            cal.generate_circuits()


class TestSetParam:

    def test_sets_config_value(self, cal, config):
        cal.set_param('single_qubit/0/GE/freq', 123.456789123)
        assert config['single_qubit/0/GE/freq'] == 123.45679

    def test_invalid_param_logs_warning_and_reraises(self, cal, caplog):
        with caplog.at_level(logging.WARNING):
            with pytest.raises(AttributeError):
                cal.set_param(12345, 1.0)  # not a '/'-separated string
        assert any('Could not set' in r.message for r in caplog.records)


class TestFinal:
    """final() drives set_param() based on fit success / cal values.

    These tests replace set_param with a call recorder so the branch
    logic in final() can be checked in isolation from Config.
    """

    @pytest.fixture(autouse=True)
    def _recorder(self, cal, monkeypatch):
        calls = []
        monkeypatch.setattr(
            cal, 'set_param', lambda p, v: calls.append((p, v))
        )
        return calls

    def test_empty_fit_does_nothing(self, cal, _recorder):
        cal._qubits = [0]
        cal._fit = {}
        cal.final()
        assert _recorder == []

    def test_successful_fit_sets_single_param(self, cal, _recorder):
        cal._qubits = [0]
        cal._params = {0: 'single_qubit/0/GE/freq'}
        cal._cal_values[0] = 5.0e9
        cal._fit = {0: _FakeFit(fit_success=True)}
        cal.final()
        assert _recorder == [('single_qubit/0/GE/freq', 5.0e9)]

    def test_failed_fit_with_falsy_cal_value_sets_nothing(
        self, cal, _recorder
    ):
        cal._qubits = [0]
        cal._params = {0: 'single_qubit/0/GE/freq'}
        cal._fit = {0: _FakeFit(fit_success=False)}
        cal.final()
        assert _recorder == []

    def test_failed_fit_with_truthy_cal_value_still_sets(self, cal, _recorder):
        cal._qubits = [0]
        cal._params = {0: 'single_qubit/0/GE/freq'}
        cal._cal_values[0] = 5.0e9
        cal._fit = {0: _FakeFit(fit_success=False)}
        cal.final()
        assert _recorder == [('single_qubit/0/GE/freq', 5.0e9)]

    def test_multiple_params_with_matching_cal_values(self, cal, _recorder):
        cal._qubits = [0]
        cal._params = {0: ['p1', 'p2']}
        cal._cal_values[0] = [1.0, 2.0]
        cal._fit = {0: _FakeFit(fit_success=True)}
        cal.final()
        assert _recorder == [('p1', 1.0), ('p2', 2.0)]

    def test_multiple_params_with_scalar_cal_value_broadcasts(
        self, cal, _recorder
    ):
        cal._qubits = [0]
        cal._params = {0: ['p1', 'p2']}
        cal._cal_values[0] = 3.0
        cal._fit = {0: _FakeFit(fit_success=True)}
        cal.final()
        assert _recorder == [('p1', 3.0), ('p2', 3.0)]

    def test_fit_as_list_sets_per_index(self, cal, _recorder):
        cal._qubits = [0]
        cal._params = {0: ['p1', 'p2']}
        cal._cal_values[0] = [1.0, 0.0]
        cal._fit = {0: [_FakeFit(True), _FakeFit(False)]}
        cal.final()
        # Index 0 succeeds; index 1 fails and its cal value is falsy.
        assert _recorder == [('p1', 1.0)]

    def test_fit_as_list_falsy_entry_overridden_by_truthy_cal_value(
        self, cal, _recorder
    ):
        cal._qubits = [0]
        cal._params = {0: ['p1', 'p2']}
        cal._cal_values[0] = [1.0, 2.0]
        cal._fit = {0: [_FakeFit(True), _FakeFit(False)]}
        cal.final()
        assert _recorder == [('p1', 1.0), ('p2', 2.0)]

    def test_fit_as_dict_all_success_sets_all_params(self, cal, _recorder):
        cal._qubits = [0]
        cal._params = {0: ['p1', 'p2']}
        cal._cal_values[0] = [1.0, 2.0]
        cal._fit = {0: {'a': _FakeFit(True), 'b': _FakeFit(True)}}
        cal.final()
        assert _recorder == [('p1', 1.0), ('p2', 2.0)]

    def test_fit_as_dict_partial_failure_with_truthy_cal_values_sets_all(
        self, cal, _recorder
    ):
        cal._qubits = [0]
        cal._params = {0: ['p1', 'p2']}
        cal._cal_values[0] = [1.0, 2.0]
        cal._fit = {0: {'a': _FakeFit(True), 'b': _FakeFit(False)}}
        cal.final()
        assert _recorder == [('p1', 1.0), ('p2', 2.0)]

    def test_fit_as_dict_partial_failure_with_falsy_cal_values_sets_nothing(
        self, cal, _recorder
    ):
        cal._qubits = [0]
        cal._params = {0: ['p1', 'p2']}
        cal._fit = {0: {'a': _FakeFit(True), 'b': _FakeFit(False)}}
        cal.final()
        assert _recorder == []


class TestFinalSaving:

    def test_saves_config_when_save_data_enabled(self, cal, monkeypatch):
        saved = []
        monkeypatch.setattr(
            Config, 'save', lambda self, *a, **k: saved.append(True)
        )
        settings.Settings.save_data = True
        cal._fit = {}
        cal.final()
        assert saved == [True]

    def test_does_not_save_config_when_save_data_disabled(
        self, cal, monkeypatch
    ):
        saved = []
        monkeypatch.setattr(
            Config, 'save', lambda self, *a, **k: saved.append(True)
        )
        settings.Settings.save_data = False
        cal._fit = {}
        cal.final()
        assert saved == []


class TestPlot:
    """These only check that plot() runs without error and produces
    the side effects it promises (saved files); the Agg backend
    (forced session-wide in conftest.py) makes fig.show() a no-op, so
    there's no risk of a real window popping open.
    """

    def test_basic_single_qubit_plot(self, cal):
        cal._qubits = [0]
        cal._sweep_results = {0: [0.1, 0.4, 0.9]}
        cal._param_sweep = {0: [0, 1, 2]}
        cal._fit = {}
        cal.plot()

    def test_plot_with_successful_fit_draws_fit_line_and_axvline(self, cal):
        cal._qubits = [0]
        cal._sweep_results = {0: [0.1, 0.4, 0.9]}
        cal._param_sweep = {0: [0, 1, 2]}
        cal._cal_values[0] = 1.5
        cal._fit = {0: _FakeFit(fit_success=True)}
        cal.plot()

    def test_plot_with_cal_value_but_no_successful_fit(self, cal):
        cal._qubits = [0]
        cal._sweep_results = {0: [0.1, 0.4, 0.9]}
        cal._param_sweep = {0: [0, 1, 2]}
        cal._cal_values[0] = 1.5
        cal._fit = {0: _FakeFit(fit_success=False)}
        cal.plot()

    def test_plot_with_dict_sweep_results_draws_multiple_series(self, cal):
        cal._qubits = [0]
        cal._sweep_results = {0: {'0': [0.1, 0.2], '1': [0.3, 0.4]}}
        cal._param_sweep = {0: [0, 1]}
        cal._fit = {}
        cal.plot()

    def test_plot_three_qubits_fills_1d_axes_grid(self, cal):
        # ncols = min(3, 4) = 3, nrows = 1 -> a 1-D axes array, fully
        # filled (no 'off' axes).
        qubits = [0, 1, 2]
        cal._qubits = qubits
        cal._sweep_results = {q: [0.1, 0.4, 0.9] for q in qubits}
        cal._param_sweep = {q: [0, 1, 2] for q in qubits}
        cal._fit = {}
        cal.plot()

    def test_plot_five_qubits_uses_2d_axes_grid_with_blank_cells(self, cal):
        # ncols = min(5, 4) = 4, nrows = 2 -> a 2-D axes array with 3
        # unused cells turned off.
        qubits = [0, 1, 2, 3, 4]
        cal._qubits = qubits
        cal._sweep_results = {q: [0.1, 0.4, 0.9] for q in qubits}
        cal._param_sweep = {q: [0, 1, 2] for q in qubits}
        cal._fit = {}
        cal.plot()

    def test_plot_saves_figures_when_save_data_enabled(self, cal, tmp_path):
        cal._qubits = [0]
        cal._sweep_results = {0: [0.1, 0.4, 0.9]}
        cal._param_sweep = {0: [0, 1, 2]}
        cal._fit = {}
        settings.Settings.save_data = True
        cal.plot(save_path=str(tmp_path) + '/')
        assert (tmp_path / 'calibration_results.png').exists()
        assert (tmp_path / 'calibration_results.pdf').exists()
        assert (tmp_path / 'calibration_results.svg').exists()
