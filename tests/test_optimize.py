"""Unit tests for qcal.optimization.optimize.

A fake cost function and stub optimizer stand in for real
calibration/benchmarking classes and real optimizers (Adam, CMA, ...),
so these tests can exercise Optimize/LinearResponse's own
orchestration logic (scaling, config read/write, re-instantiating
cost_func, loss aggregation, the run() loop) in isolation. Config
objects are loaded from a tmp_path copy of the example config so
Optimize.run()'s config.save() (when settings.Settings.save_data is
enabled) never touches the real example file.
"""
import shutil
import types

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

import qcal.settings as settings
from qcal.config import Config
from qcal.optimization.optimize import LinearResponse, NoScaler, Optimize
from qcal.utils import save_init

EXAMPLE_CONFIG_PATH = 'examples/config/config.yaml'
PARAM = 'single_qubit/0/GE/X90/pulse/1/kwargs/amp'


@pytest.fixture
def config(tmp_path):
    path = tmp_path / 'config.yaml'
    shutil.copy(EXAMPLE_CONFIG_PATH, path)
    return Config(str(path))


class FakeCostFunc:
    """Stand-in cost function: loss is M * config[param] + B."""

    @save_init
    def __init__(self, config, param=PARAM, m=1.0, b=0.0):
        self.n_init_calls = getattr(self, 'n_init_calls', 0) + 1
        self._config = config
        self._param = param
        self._m = m
        self._b = b
        self._loss = {0: [self._compute()]}

    def _compute(self):
        return self._m * self._config[self._param] + self._b

    @property
    def loss(self):
        return self._loss

    def run(self):
        self._loss = {0: [self._compute()]}


class StubOptimizer:
    """Deterministic ask-tell stand-in for a real optimizer."""

    def __init__(self, x0, step_size=0.01, **kwargs):
        self._x0 = x0
        self._step_size = step_size
        self._opt_x = x0.copy()
        self._opt_loss = np.array(1e10)
        self._step_count = 0
        self.tell_calls = []

    @property
    def opt_x(self):
        return self._opt_x

    @property
    def opt_loss(self):
        return self._opt_loss

    def tell(self, x, loss):
        self.tell_calls.append((x.copy(), loss))
        loss_val = loss[0] if isinstance(loss, list) else loss
        if np.abs(loss_val) < np.abs(self._opt_loss):
            self._opt_loss = loss_val
            self._opt_x = x.copy()

    def ask(self):
        self._step_count += 1
        return self._x0 + self._step_size * self._step_count


class TestNoScaler:

    def test_fit_transform_is_identity(self):
        x = np.array([[1.0, 2.0]])
        np.testing.assert_array_equal(NoScaler().fit_transform(x), x)

    def test_inverse_transform_is_identity(self):
        x = np.array([[1.0, 2.0]])
        np.testing.assert_array_equal(NoScaler().inverse_transform(x), x)


class TestOptimizeConstruction:

    def test_defaults_to_noscaler_without_bounds(self, config):
        cost_func = FakeCostFunc(config=config)
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func,
        )
        assert isinstance(opt.scaler[0], NoScaler)

    def test_extracts_x0_from_config(self, config):
        cost_func = FakeCostFunc(config=config)
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func,
        )
        assert opt.x0[0].flatten()[0] == pytest.approx(config[PARAM])

    def test_uses_minmaxscaler_with_bounds(self, config):
        cost_func = FakeCostFunc(config=config)
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func,
            lbounds={0: [0.0]}, ubounds={0: [1.0]},
        )
        assert isinstance(opt.scaler[0], MinMaxScaler)
        assert 0.0 <= opt.x0[0].flatten()[0] <= 1.0


class TestOptimizeComputeLoss:

    def test_single_loss_value_is_used_directly(self, config):
        cost_func = FakeCostFunc(config=config)
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func,
        )
        cost_func._loss = {0: [0.25]}

        opt.compute_loss()

        assert opt.loss[0] == pytest.approx(0.25)

    def test_multiple_loss_values_use_quadrature_sum(self, config):
        cost_func = FakeCostFunc(config=config)
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func,
        )
        cost_func._loss = {0: [0.3, 0.4]}

        opt.compute_loss()

        assert opt.loss[0] == pytest.approx(0.5)  # sqrt(0.3^2 + 0.4^2)


class TestOptimizeSetParams:

    def test_writes_value_to_config(self, config):
        cost_func = FakeCostFunc(config=config)
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func,
        )

        opt.set_params({0: np.array([[0.777]])})

        assert config[PARAM] == pytest.approx(0.777)

    def test_reinstantiates_cost_func_with_updated_config(self, config):
        cost_func = FakeCostFunc(config=config)
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func,
        )
        calls_before = cost_func.n_init_calls

        opt.set_params({0: np.array([[0.777]])})

        assert opt.cost_func is cost_func
        assert cost_func.n_init_calls == calls_before + 1
        assert cost_func._init_kwargs['config'] is config


class TestOptimizeRun:

    def test_run_tracks_loss_history_and_writes_best_params(self, config):
        # loss = amp - 0.5, minimized (|loss| smallest) at the first
        # step (smallest positive step moves closest to the root).
        cost_func = FakeCostFunc(config=config, m=1.0, b=-0.5)
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func, n_iters=3,
        )

        opt.run()

        assert len(opt.loss_history[0]) == 3
        np.testing.assert_allclose(
            opt.opt_params[0], opt.optimizer[0].opt_x
        )
        assert config[PARAM] == pytest.approx(
            opt.optimizer[0].opt_x.flatten()[0]
        )

    def test_run_does_not_save_config_by_default(self, config):
        original = config[PARAM]
        cost_func = FakeCostFunc(config=config, m=1.0, b=-0.5)
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func, n_iters=2,
        )

        opt.run()

        reloaded = Config(config._filename)
        assert config[PARAM] != pytest.approx(original)
        assert reloaded[PARAM] == pytest.approx(original)

    def test_run_saves_config_when_save_data_enabled(
        self, config, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(settings.Settings, 'save_data', True)
        cost_func = FakeCostFunc(config=config, m=1.0, b=-0.5)
        # plot() writes figures to the cost function's data manager.
        cost_func.data_manager = types.SimpleNamespace(
            save_path=str(tmp_path) + '/'
        )
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func, n_iters=2,
        )

        opt.run()

        reloaded = Config(config._filename)
        assert reloaded[PARAM] == pytest.approx(config[PARAM])
        assert (tmp_path / 'optimization_convergence.png').exists()

    def test_delta_bootstraps_an_extra_tell_call(self, config):
        cost_func = FakeCostFunc(config=config, m=1.0, b=-0.5)
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func, n_iters=2, delta=0.01,
        )

        opt.run()

        # n_iters tell() calls from the main loop + 1 bootstrap tell().
        assert len(opt.optimizer[0].tell_calls) == 3

    def test_without_delta_no_bootstrap_tell_call(self, config):
        cost_func = FakeCostFunc(config=config, m=1.0, b=-0.5)
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=cost_func, n_iters=2,
        )

        opt.run()

        assert len(opt.optimizer[0].tell_calls) == 2


PARAM_1 = 'single_qubit/1/GE/X90/pulse/1/kwargs/amp'


class TwoLabelFakeCostFunc:
    """Stand-in cost function with independent losses for labels 0 and 1.

    loss[ql] = m[ql] * (config[param_ql] - root[ql]).
    """

    @save_init
    def __init__(self, config, roots, slopes=(1.0, 1.0)):
        self._config = config
        self._roots = roots
        self._slopes = slopes
        self._loss = self._compute()

    def _compute(self):
        return {
            ql: [m * (self._config[p] - r)]
            for ql, p, r, m in zip(
                (0, 1), (PARAM, PARAM_1), self._roots, self._slopes,
                strict=True,
            )
        }

    @property
    def loss(self):
        return self._loss

    def run(self):
        self._loss = self._compute()


def _global_min_optimize(config, global_min):
    """Optimize two labels whose individual and joint optima differ.

    StubOptimizer proposes x0 + 0.01 * k at iteration k = 1, 2, 3. Label
    0's root sits at k = 1, label 1's (10x steeper) root at k = 3, so the
    per-iteration losses are (0, -0.2), (0.01, -0.1), (0.02, 0): label 0
    alone is best at k = 1, but the combined loss is smallest at k = 3.
    """
    x0_0, x0_1 = config[PARAM], config[PARAM_1]
    cost_func = TwoLabelFakeCostFunc(
        config=config, roots=(x0_0 + 0.01, x0_1 + 0.03),
        slopes=(1.0, 10.0),
    )
    opt = Optimize(
        config=config, params={0: PARAM, 1: PARAM_1},
        optimizer=StubOptimizer, cost_func=cost_func, n_iters=3,
        global_min=global_min,
    )
    opt.run()
    return opt, x0_0, x0_1


class TestOptimizeGlobalMin:

    def test_defaults_to_false(self, config):
        opt = Optimize(
            config=config, params={0: PARAM}, optimizer=StubOptimizer,
            cost_func=FakeCostFunc(config=config),
        )
        assert opt.global_min is False
        assert opt.opt_loss_global == np.inf

    def test_independent_selection_uses_each_labels_own_best(self, config):
        opt, x0_0, x0_1 = _global_min_optimize(config, global_min=False)

        assert opt.opt_params[0].flatten()[0] == pytest.approx(x0_0 + 0.01)
        assert opt.opt_params[1].flatten()[0] == pytest.approx(x0_1 + 0.03)

    def test_global_selection_uses_jointly_best_iteration(self, config):
        opt, x0_0, x0_1 = _global_min_optimize(config, global_min=True)

        assert opt.global_min is True
        assert opt.opt_params[0].flatten()[0] == pytest.approx(x0_0 + 0.03)
        assert opt.opt_params[1].flatten()[0] == pytest.approx(x0_1 + 0.03)
        assert opt.opt_loss[0] == pytest.approx(0.02)
        assert opt.opt_loss[1] == pytest.approx(0.0, abs=1e-12)
        assert opt.opt_loss_global == pytest.approx(0.02)
        assert config[PARAM] == pytest.approx(x0_0 + 0.03)
        assert config[PARAM_1] == pytest.approx(x0_1 + 0.03)

    def test_global_min_tracks_combined_loss(self, config):
        opt = Optimize(
            config=config, params={0: PARAM, 1: PARAM_1},
            optimizer=StubOptimizer,
            cost_func=TwoLabelFakeCostFunc(config=config, roots=(0., 0.)),
        )
        opt.update_global_best(
            {0: np.array([1.0]), 1: np.array([2.0])},
            {0: np.array(3.0), 1: np.array(4.0)},
        )
        assert opt.opt_loss_global == pytest.approx(5.0)

        # A worse candidate leaves the stored optimum untouched.
        opt.update_global_best(
            {0: np.array([9.0]), 1: np.array([9.0])},
            {0: np.array(6.0), 1: np.array(0.0)},
        )
        assert opt.opt_loss_global == pytest.approx(5.0)
        np.testing.assert_array_equal(opt._global_opt_params[0], [1.0])

        # A better one replaces it.
        opt.update_global_best(
            {0: np.array([7.0]), 1: np.array([8.0])},
            {0: np.array(0.6), 1: np.array(0.8)},
        )
        assert opt.opt_loss_global == pytest.approx(1.0)
        np.testing.assert_array_equal(opt._global_opt_params[1], [8.0])


class LinearFakeCostFunc:
    """Stand-in cost function with a known-linear response."""

    @save_init
    def __init__(self, config, param=PARAM, m=1.0, b=0.0):
        self._config = config
        self._param = param
        self._m = m
        self._b = b
        self._loss = {0: [self._compute()]}

    def _compute(self):
        return self._m * self._config[self._param] + self._b

    @property
    def loss(self):
        return self._loss

    def run(self):
        self._loss = {0: [self._compute()]}


class TestLinearResponseConstruction:

    def test_initial_opt_params_is_current_config_value(self, config):
        cost_func = LinearFakeCostFunc(config=config)
        lr = LinearResponse(
            config=config, qubit_labels=[0], params={0: PARAM},
            cost_func=cost_func, delta=0.01, n_iters=4,
        )
        assert lr.opt_params[0] == pytest.approx(config[PARAM])

    def test_x_matches_cumulative_sweep_offsets(self, config):
        cost_func = LinearFakeCostFunc(config=config)
        lr = LinearResponse(
            config=config, qubit_labels=[0], params={0: PARAM},
            cost_func=cost_func, delta=0.01, n_iters=4,
        )
        np.testing.assert_allclose(
            lr._x[0], [0.0, 0.01, 0.02, 0.03]
        )


class TestLinearResponseRun:

    def test_recovers_known_slope_and_root(self, config):
        initial = config[PARAM]
        m, b = 2.0, -1.0  # root at amp = 0.5
        cost_func = LinearFakeCostFunc(config=config, m=m, b=b)
        lr = LinearResponse(
            config=config, qubit_labels=[0], params={0: PARAM},
            cost_func=cost_func, delta=0.01, n_iters=4,
        )
        lr.run()

        assert lr.slope[0] == pytest.approx(m)
        assert lr.opt_params[0] == pytest.approx(0.5, abs=1e-6)
        assert config[PARAM] == pytest.approx(0.5, abs=1e-6)
        assert initial != pytest.approx(0.5)  # sanity: actually moved

    def test_falls_back_to_min_abs_loss_when_fit_is_poor(self, config):
        initial = config[PARAM]
        # Non-linear response: a good linear fit is impossible, so
        # analyze() should fall back to the x offset with the smallest
        # |loss| instead of using the (meaningless) fitted root.
        lookup = {0: 5.0, 1: 1.0, 2: 4.0, 3: 0.2}

        class NoisyFakeCostFunc:
            @save_init
            def __init__(self, config, param=PARAM):
                self._config = config
                self._param = param
                self._loss = {0: [self._compute()]}

            def _compute(self):
                offset = round(
                    (self._config[self._param] - initial) / 0.01
                )
                return lookup[offset]

            @property
            def loss(self):
                return self._loss

            def run(self):
                self._loss = {0: [self._compute()]}

        cost_func = NoisyFakeCostFunc(config=config)
        lr = LinearResponse(
            config=config, qubit_labels=[0], params={0: PARAM},
            cost_func=cost_func, delta=0.01, n_iters=4,
        )

        lr.run()

        assert lr._fit[0].fit_success is False
        # min |loss| (0.2) occurs at offset index 3 -> x = 0.03
        assert lr.opt_params[0] == pytest.approx(initial + 0.03)
