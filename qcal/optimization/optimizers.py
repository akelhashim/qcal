"""Submodule for storing different optimizers.

"""
from qcal.math.utils import uncertainty_of_sum

import logging
import numpy as np

from numpy.typing import ArrayLike, NDArray
from typing import Any

logger = logging.getLogger(__name__)


__all__ = (
    'Adam',
    'CMA', 
    'LQR'
)


class Adam:

    def __init__(self,
            x0:        float | NDArray,
            lr:        float = 0.01, 
            beta_1:    float = 0.9, 
            beta_2:    float = 0.999, 
            eps:       float = 1e-8,
            tol:       float = 1e-4,
            loss:      float | NDArray = 0.,
            grad_func: Any | None = None
        ) -> None:
        """Based on ADAM, a momentum-based stochastic optimizer.
        
        See: https://arxiv.org/abs/1412.6980

        Args:
            x0 (float | NDArray): initial parameter values.
            lr (float, optional): learning rate, can be modified to improve
                convergence. Defaults to 0.01
            beta_1 (float, optional)): hyperparameter 1, [0, 1). In general, 
                this value should not be changed. Defaults to 0.9.
            beta_2 (float, optional)): hyperparameter 2, [0, 1). In general, 
                this value should not be changed. Defaults to 0.999.
            eps (float, optional): small additive factor to ensure we do not 
                divide by zero. In general, this value should not be changed. 
                Defaults to 1e-8.
            tol (float, optional): convergence criteria. Defaults to 1e-4. If 
                the change in the cost is less than this value, the optimization 
                will terminate.
            loss (float | NDArray | None, optional): initial loss corresponding
                to the initial parameters (x). Defaults to 0. This is used to
                compute gradients for the parameters via finite difference
                methods.
            grad_funct (Any | None, optional): function (or class) for computing
                the gradient. If None, finite-difference is used.
        """
        self._x_t = x0  # Parameters at time t = 0
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._eps = eps
        self._tol = tol
        self._loss = loss
        self._grad_func = grad_func

        self._grad = 0.
        self._prev_loss = 0.
        self._opt_loss = loss
        self._opt_x = x0
        self._m_t = 0.     # Aggregate of gradients at time t
        self._m_t_bc = 0.  # Bias-corrected first moment estimate
        self._v_t = 0.     # Sum of the square of past gradients
        self._v_t_bc = 0.  # Bias-corrected second raw moment estimate
        self._t = 0.       # Time step

    @property
    def opt_x(self) -> NDArray:
        """Optimal parameter values.

        Returns:
            NDArray: optimal parameters.
        """
        return self._opt_x
    
    @property
    def opt_loss(self) -> NDArray:
        """Loss correstponding to the optimal parameter values.

        Returns:
            NDArray: loss.
        """
        return self._opt_loss

    def tell(self, x: float | NDArray, loss: float | NDArray) -> None:
        """Pass the parameter values and loss values for computing a gradient.

        If no gradient function is specified, the gradient is computed using 
        finite difference approximation:

        delta = x - x_t
        df(x)/dx = [f(x + delta) - f(x)] / delta

        x (float | NDArray): parameter values corresponding to the loss.
        loss (float | NDArray): loss for each paramter value.
        """
        if isinstance(loss, list):  # Compatibility with Optimize class
            loss = loss[0]

        if np.abs(loss) < np.abs(self._opt_loss):
            self._opt_loss = loss.copy()
            self._opt_x = x.copy()
            
        if self._grad_func:
            self._grad_func.tell(x, loss)
            self._grad = self._grad_func.grad
        else:
            delta = x.copy() - self._x_t
            self._grad = (loss - self._loss) / (delta + self._eps)
        
        self._prev_loss = self._loss.copy()
        self._loss = loss.copy()
    
    def step(self, grad: float | NDArray  = None) -> float | NDArray:
        """Compute the new parameters values based on the gradient for each.

        Args:
            grad (float | NDArray | None, optional): gradient for each paramter.
                Defaults to None. If None, `self._grad` will be used.

        Returns:
            float | NDArray: new parameter values.
        """
        grad = grad if grad is not None else self._grad
        self._t += 1
        self._m_t = (self._beta_1 * self._m_t) + (1 - self._beta_1) * grad
        self._v_t = (self._beta_2 * self._v_t) + (1 - self._beta_2) * grad ** 2
        self._m_t_bc = self._m_t / (1 - self._beta_1 ** self._t)
        self._v_t_bc = self._v_t / (1 - self._beta_2 ** self._t)
        self._x_t = self._x_t - (self._lr * self._m_t_bc) / (
            np.sqrt(self._v_t_bc) + self._eps
        )

        return self._x_t
    
    def stop(self) -> bool:
        """Whether to stop the optimization if convergence has been reached.

        Returns:
            bool: stop the optimization loop or not.
        """
        return abs(self._loss - self._prev_loss) < self._tol


class CMA:

    def __init__(self,
            x0:     float | ArrayLike,
            sigma0: float = 0.15,
            # opts:   Dict = None,
            **kwargs
        ) -> None:
        """CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

        See: 
        - https://github.com/CMA-ES/pycma
        - https://cma-es.github.io/apidocs-pycma/index.html

        Args:
            x0 (float | ArrayLike): initial solution, starting point.
            sigma0 (float, optional): initial standard deviation.  Defaults to 
                0.15. The problem variables should have been scaled, such that a 
                single standard deviation on all variables is useful and the 
                optimum is expected to lie within about `x0` +- 3*sigma0. 
                Often one wants to check for solutions close to the initial 
                point. This allows, for example, for an easier check of 
                consistency of the objective function and its interfacing with 
                the optimizer. In this case, a much smaller sigma0 is advisable.
            **kwargs (Dict, optional): options, a dictionary with optional 
                settings. Defaults to None.
        """
        try:
            import cma
            logger.info(f" cma version: {cma.__version__}")
        except ImportError:
            logger.warning(' Unable to import cma!')
 
        self._es = cma.CMAEvolutionStrategy(x0, sigma0, kwargs)

    @property
    def opt_x(self) -> ArrayLike:
        """Optimal parameter values.

        Returns:
            ArrayLike: optimal parameters.
        """
        return self._es.best.x
    
    @property
    def opt_loss(self) -> ArrayLike:
        """Loss correstponding to the optimal parameter values.

        Returns:
            ArrayLike: loss.
        """
        return self._es.best.f

    def tell(self, x: ArrayLike, loss: ArrayLike) -> None:
        """Pass loss values to prepare for next iteration.

        Args:
            x (ArrayLike): list or array of candidate solution points, most 
                presumably the values delivered by method `ask()`.
            loss (ArrayLike): list or array of objective function loss values 
                corresponding to the respective points.
        """
        self._es.tell(x, loss, copy=True)

    def step(self) -> ArrayLike:
        """Ask for new candidate solutions.

        Returns:
            ArrayLike: new candidate solutions.
        """
        return self._es.ask()
    
    def stop(self) -> bool:
        """Whether to stop the optimization if convergence has been reached.

        Returns:
            bool: stop the optimization loop or not.
        """
        return self._es.stop()
        

class LQR:

    def __init__(self,
            x0:  float | NDArray,
            A:   NDArray,
            B:   NDArray,
            Q:   NDArray,
            R:   NDArray,
            tol: float = 1e-6,
        ) -> None:
        """Linear Quadratic Regulator.

        See:
        https://python-control.readthedocs.io/en/latest/generated/control.dlqr.html

        Args:
            x0 (float | NDArray): initial parameter values.
            A (NDArray): dynamics matrix.
            B (NDArray): input matrix.
            Q (NDArray): state matrix.
            R (NDArray): input weight matrix.
            tol (float, optional): convergence criteria. Defaults to 1e-4. If 
                the change in the cost is less than this value, the optimization 
                will terminate.
        """
        try:
            import control
            from control import dlqr
            logger.info(f" control version: {control.__version__}")
        except ImportError:
            logger.warning(' Unable to import control!')
 
        self._x_t = x0.copy()  # Parameters at time t = 0
        self._B = B
        self._tol = tol

        self._loss = np.zeros(B.shape) # np.array([0.])
        self._prev_loss = 0.
        self._opt_loss = 1e10  # Large value for initial comparison
        self._opt_x = x0.copy()
        self._lqr_gain, _, _ = dlqr(A, B, Q, R)
        self._u = np.zeros(B.shape) # np.array([0.])  # Control TODO

    @property
    def grad(self) -> float | NDArray:
        """Compatibility for using LQR to estimate gradients.

        Returns:
            float | NDArray: control (u).
        """
        return self._u
    
    @property
    def opt_x(self) -> NDArray:
        """Optimal parameter values.

        Returns:
            NDArray: optimal parameters.
        """
        return self._opt_x
    
    @property
    def opt_loss(self) -> NDArray:
        """Loss correstponding to the optimal parameter values.

        Returns:
            NDArray: loss.
        """
        return self._opt_loss

    def tell(self, x: float | NDArray, loss: float | NDArray) -> None:
        """Pass the parameter values and loss values for computing the control.

        The control is computed according the LQR gain and the loss.

        x (float | NDArray): parameter values corresponding to the loss.
        loss (float | NDArray): loss for each paramter value.
        """
        if np.abs(loss) < np.abs(self._opt_loss):
            self._opt_loss = loss.copy()
            self._opt_x = x.copy()
        self._prev_loss = self._loss.copy()
        self._loss = loss.copy()

        # Check proximity to noise floor
        min_distance = np.min(np.abs(loss)) - self._tol
        if min_distance < 0.1:  # Dead zone
            # Reduce control effort near noise floor
            scaling = np.clip(min_distance / 0.1, 0, 1)
        else:
            scaling = 1.

        self._u = -scaling * self._lqr_gain @ loss.copy()  # Control

    def step(self) -> float | NDArray:
        """Compute the new parameters values based on the control.

        Returns:
            float | NDArray: new parameter values.
        """
        self._x_t += self._u

        return self._x_t
    
    def stop(self) -> bool:
        """Whether to stop the optimization if convergence has been reached.

        Returns:
            bool: stop the optimization loop or not.
        """
        return abs(self._loss - self._prev_loss) < self._tol