"""Unit tests for qcal.optimization.optimizers."""
import numpy as np
import pytest

from qcal.optimization.optimizers import CMA, LQG, LQR, Adam

try:
    import cma  # noqa: F401
    _HAS_CMA = True
except ImportError:
    _HAS_CMA = False

try:
    import control  # noqa: F401
    _HAS_CONTROL = True
except ImportError:
    _HAS_CONTROL = False


class TestAdamConstruction:

    def test_initial_opt_x_is_x0(self):
        opt = Adam(x0=np.array([0.5]), loss=np.array([2.0]))
        np.testing.assert_array_equal(opt.opt_x, np.array([0.5]))

    def test_initial_opt_loss_is_a_large_sentinel(self):
        opt = Adam(x0=np.array([0.5]), loss=np.array([0.0]))
        assert opt.opt_loss == 1e10


class TestAdamTell:

    def test_updates_opt_when_loss_improves(self):
        opt = Adam(x0=np.array([0.0]), loss=np.array([4.0]))
        opt.tell(np.array([0.1]), np.array([1.0]))

        np.testing.assert_array_equal(opt.opt_x, np.array([0.1]))
        assert opt.opt_loss == 1.0

    def test_does_not_update_when_loss_worsens(self):
        opt = Adam(x0=np.array([0.0]), loss=np.array([4.0]))
        opt.tell(np.array([0.1]), np.array([1.0]))
        opt.tell(np.array([0.2]), np.array([5.0]))

        np.testing.assert_array_equal(opt.opt_x, np.array([0.1]))
        assert opt.opt_loss == 1.0

    def test_accepts_list_loss_for_optimize_compatibility(self):
        opt = Adam(x0=np.array([0.0]), loss=np.array([4.0]))
        opt.tell(np.array([0.1]), [np.array(1.0)])

        assert opt.opt_loss == 1.0

    def test_finite_difference_gradient_uses_previous_evaluated_x(self):
        opt = Adam(x0=np.array([0.0]), loss=np.array([4.0]))

        opt.tell(np.array([1.0]), np.array([2.0]))
        np.testing.assert_allclose(opt._grad, [-2.0], atol=1e-6)

        opt.tell(np.array([1.5]), np.array([0.5]))
        np.testing.assert_allclose(opt._grad, [-3.0], atol=1e-6)

    def test_custom_grad_func_is_used_instead_of_finite_difference(self):
        class StubGradFunc:
            def __init__(self):
                self.grad = None

            def tell(self, x, loss):
                self.grad = np.array([42.0])

        grad_func = StubGradFunc()
        opt = Adam(
            x0=np.array([0.0]), loss=np.array([4.0]), grad_func=grad_func
        )
        opt.tell(np.array([0.1]), np.array([1.0]))

        np.testing.assert_array_equal(opt._grad, np.array([42.0]))


class TestAdamStepAndStop:

    def test_converges_to_minimum_over_an_ask_tell_step_loop(self):
        # End-to-end regression test for the finite-difference fix
        # above, using the exact ask-tell-step pattern
        # qcal.optimization.optimize.Optimize.run() drives: step()
        # proposes new params, the loss is measured there, and tell()
        # is called with that (params, loss) pair.
        def loss_fn(x):
            return (x - 2.0) ** 2

        x0 = np.array([0.0])
        opt = Adam(x0=x0, lr=0.1, loss=loss_fn(x0))

        x_pert = x0 + 0.01
        opt.tell(x_pert.copy(), loss_fn(x_pert))
        for _ in range(60):
            x_new = opt.step()
            opt.tell(x_new.copy(), loss_fn(x_new))

        assert opt.opt_x[0] == pytest.approx(2.0, abs=0.05)

    def test_step_moves_parameters_using_internal_grad(self):
        opt = Adam(x0=np.array([0.0]), loss=np.array([4.0]))
        opt.tell(np.array([1.0]), np.array([2.0]))

        x_new = opt.step()

        assert x_new[0] != 0.0

    def test_stop_is_false_when_loss_changes_a_lot(self):
        opt = Adam(x0=np.array([0.0]), loss=np.array([4.0]), tol=1e-4)
        opt.tell(np.array([1.0]), np.array([2.0]))

        assert bool(opt.stop()) is False

    def test_stop_is_true_when_loss_barely_changes(self):
        opt = Adam(x0=np.array([0.0]), loss=np.array([4.0]), tol=1e-2)
        opt.tell(np.array([1.0]), np.array([4.0 - 1e-6]))

        assert bool(opt.stop()) is True


class TestAdamScalarInputs:

    def test_tell_accepts_plain_floats(self):
        opt = Adam(x0=0.0, loss=4.0)
        opt.tell(0.1, 1.0)

        assert opt.opt_x == pytest.approx(0.1)
        assert opt.opt_loss == pytest.approx(1.0)

    def test_step_and_stop_work_after_scalar_tell(self):
        opt = Adam(x0=0.0, loss=4.0)
        opt.tell(0.1, 1.0)

        x_new = opt.step()

        assert x_new != pytest.approx(0.0)
        assert isinstance(opt.stop(), (bool, np.bool_))


@pytest.mark.skipif(_HAS_CMA, reason='cma is installed')
class TestCMAWithoutDependency:

    def test_construction_does_not_crash(self):
        CMA(x0=[0.0], sigma0=0.1)


@pytest.mark.skipif(_HAS_CONTROL, reason='control is installed')
class TestLQRWithoutDependency:

    def test_construction_does_not_crash(self):
        LQR(
            x0=np.array([0.0]), A=np.eye(1), B=np.eye(1), Q=np.eye(1),
            R=np.eye(1),
        )


@pytest.mark.skipif(_HAS_CONTROL, reason='control is installed')
class TestLQGWithoutDependency:

    def test_construction_does_not_crash(self):
        LQG(
            x0=np.array([0.0]), loss=np.array([0.0]), P0=np.eye(1),
            A=np.eye(1), B=np.eye(1), Q=np.eye(1), R=np.eye(1),
            Q_kf=np.eye(1), R_kf=np.eye(1),
        )
