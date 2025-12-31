"""Submodule for optimizing config parameters based on some objective function.

"""
import logging
from collections.abc import Iterable
from typing import Any, Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

import qcal.settings as settings
from qcal.config import Config
from qcal.fitting.fit import FitLinear
from qcal.math.utils import uncertainty_of_sum
from qcal.plotting.utils import calculate_nrows_ncols

logger = logging.getLogger(__name__)


class NoScaler:
    """This is a dummy class that returns the same values that are input.

    It has the same methods as MinMaxScalar for compatibility with the Optimize
    class, but no rescaling is performed.
    """

    def __init__(self):
        pass

    def fit_transform(self, x: Any) -> Any:
        """Return the values that are passed.

        Args:
            x (Any): input values.

        Returns:
            Any: input values.
        """
        return x

    def inverse_transform(self, x: Any) -> Any:
        """Return the values that are passed.

        Args:
            x (Any): input values.

        Returns:
            Any: input values.
        """
        return x


class LinearResponse:

    def __init__(self,
            config:       Config,
            qubit_labels: Iterable[int | Tuple[int]],
            params:       Dict[int | tuple[int], str | list[str]],
            cost_func:    Any,
            delta:        float | Dict[int | tuple, float],
            n_iters:      int = 3,
            x_label:      str = 'Param Value',
            y_label:      str = 'Loss'
        ) -> None:

        self._config = config
        self._qubit_labels = qubit_labels
        self._params = params
        self._cost_func = cost_func
        self._n_iters = n_iters
        self._x_label = x_label
        self._y_label = y_label

        if not isinstance(delta, Dict):
            self._delta = dict.fromkeys(qubit_labels, delta)
        else:
            self._delta = delta

        self._fit = {ql: FitLinear() for ql in qubit_labels}
        self._x = {
            ql: np.linspace(0, self._delta[ql] * self._n_iters, self._n_iters)
                for ql in qubit_labels
        }
        self._y = {ql: [] for ql in qubit_labels}
        self._opt_params = {
            ql: self._config[self._params[ql][0]]
                if isinstance(self._params[ql], list)
                else self._config[self._params[ql]]
                for ql in self._qubit_labels
        }

    @property
    def config(self) -> Config:
        """Optimized config.

        Returns:
            Config: qcal config object.
        """
        return self._config

    @property
    def cost_func(self) -> Any:
        """Cost function used to measure the loss.

        Returns:
            Any: cost function.
        """
        return self._cost_func

    @property
    def delta(self) -> Dict:
        """Parameter perturbation per iteration.

        Returns:
            Dict: perturbation for each qubit label.
        """
        return self._delta

    @property
    def n_iters(self) -> int:
        """Number of iterations for fitting the linear curve.

        Returns:
            int: number of iterations.
        """
        return self._n_iters

    @property
    def opt_params(self) -> Dict:
        """Optimized parameter values for each qubit label.

        Returns:
            Dict: optimized parameter values.
        """
        return self._opt_params

    @property
    def params(self) -> Dict:
        """Parameters to optimize.

        Returns:
            Dict: dictionary of qubit labels to parameters.
        """
        return self._params

    @property
    def slope(self) -> Dict:
        """Slope of each linear fit."""
        m = {}
        for ql in self._qubit_labels:
            if self._fit[ql].fit_success:
                m[ql] = self._fit[ql].fit_params['m'].value

        return m

    @property
    def y_intercept(self) -> Dict:
        """y-intercept of each linear fit."""
        b = {}
        for ql in self._qubit_labels:
            if self._fit[ql].fit_success:
                b[ql] = self._fit[ql].fit_params['b'].value

        return b

    def analyze(self) -> None:
        """Analyze the data and fit to a linear curve."""
        for ql in self._qubit_labels:
            params = self._fit[ql].model.make_params(m=1, b=self._y[ql][0])
            self._fit[ql].fit(
                self._x[ql], np.array(self._y[ql]), params=params
            )
            if self._fit[ql].result.rsquared > 0.8:
                self._opt_params[ql] += (
                    -self._fit[ql].fit_params['b'].value /
                     self._fit[ql].fit_params['m'].value
                )
            else:
                self._fit[ql]._fit_success = False
                self._opt_params[ql] += (
                    self._x[ql][np.argmin(np.abs(self._y[ql]))]
                )

    def plot(self) -> None:
        """Plot the raw data and the fits."""
        self._config.reload()

        nrows, ncols = calculate_nrows_ncols(len(self._qubit_labels))
        figsize = (5 * ncols, 4 * nrows)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, layout='constrained'
        )

        k = -1
        for i in range(nrows):
            for j in range(ncols):
                k += 1

                if len(self._qubit_labels) == 1:
                    ax = axes
                elif axes.ndim == 1:
                    ax = axes[j]
                elif axes.ndim == 2:
                    ax = axes[i,j]

                if k < len(self._qubit_labels):
                    ql = self._qubit_labels[k]
                    x = self._x[ql] + (
                        self._config[self._params[ql][0]]
                        if isinstance(self._params[ql], list)
                        else self._config[self._params[ql]]
                    )
                    ax.plot(x, self._y[ql], 'o-', label=f'{ql}')
                    ax.plot(
                        x, self._fit[ql].predict(self._x[ql]), '--',
                        label=f'{ql} fit'
                    )
                    ax.axvline(
                        self._opt_params[ql],
                        ls='--',
                        c='k',
                        label='Opt. value',
                    )

                    ax.set_xlabel(f'{self._x_label}', fontsize=15)
                    ax.set_ylabel(f'{self._y_label}', fontsize=15)
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    ax.grid(True)
                    ax.legend(loc=1, fontsize=12)

                else:
                    ax.axis('off')

        fig.set_tight_layout(True)
        if settings.Settings.save_data:
            save_path = self._cost_func.data_manager.save_path
            fig.savefig(save_path + 'linear_response.png', dpi=600)
            fig.savefig(save_path + 'linear_response.pdf')
            fig.savefig(save_path + 'linear_response.svg')
        plt.show()

    def set_params(self) -> None:
        """Set the optimal parameter values."""
        for ql, param in self._params.items():
            if isinstance(param, list):
                for p in param:
                    self._config[p] = self._opt_params[ql]
            else:
                self._config[param] = self._opt_params[ql]

        if settings.Settings.save_data:
            self._config.save()

    def run(self) -> None:
        """Run the sweep."""
        for i in range(self._n_iters):

            if i > 0:
                for ql in self._qubit_labels:
                    if isinstance(self._params[ql], list):
                        for param in self._params[ql]:
                            self._config[param] += self._delta[ql]
                    else:
                        self._config[self._params[ql]] += self._delta[ql]

                self._cost_func._init_kwargs['config'] = self._config
                self._cost_func.__init__(
                    *self._cost_func._init_args,
                    **self._cost_func._init_kwargs
                )

            # Run the cost function to measure the loss
            self._cost_func.run()

            for ql in self._qubit_labels:
                self._y[ql].append(
                    uncertainty_of_sum(self._cost_func.loss[ql])
                    if len(self._cost_func.loss[ql]) > 1
                    else self._cost_func.loss[ql][0]
                )

        self.analyze()
        self.plot()
        self.set_params()


class Optimize:

    def __init__(self,
            config:     Config,
            params:     Dict[int | tuple[int], str | list[str]],
            optimizer:  Callable,
            cost_func:  Any,
            opt_kwargs: Dict[int | tuple[int], Dict] | None = None,
            lbounds:    Dict[float | tuple, list] | None = None,
            ubounds:    Dict[float | tuple, list] | None = None,
            delta:      float | Dict[int | tuple, float] | None = None,
            n_iters:    int = 10,
            tol:        float = 0.1
        ) -> None:
        """Generalized optimization class.

        Args:
            config (Config): qcal config object
            params (Dict[int  |  tuple[int], str  |  list[str]]): dictionary
                mapping qubit labels to config parameters to optimize.
            optimizer (Callable): optimizer.
            cost_func (Any): cost function. This should be a class which, once
                run, should have a `loss` class property for each qubit label.
            opt_kwargs (Dict[int  |  tuple[int], Dict] | None, optional): dict
                mapping qubit labels to optional kwargs for the optimizer.
                Defaults to None.
            lbounds (Dict[float  |  tuple, list] | None, optional): dictionary
                mapping qubit labels to the lower bounds for each parameter
                value specified for those labels. Defaults to None.
            ubounds (Dict[float  |  tuple, list] | None, optional): dictionary
                mapping qubit labels to the upper bounds for each parameter
                value specified for those labels. Defaults to None.
            delta (float | Dict[float  |  tuple, float] | None, optional): this
                kwarg can be used to specify a small initial pertubation for
                gradient-based optimizations. Defaults to None.
            n_iters (int, optional): number of iterations. Defaults to 10.
            tol (float, optional): tolerance, i.e., convergence threshold.
                Defaults to 0.1.
        """
        self._config = config
        self._params = params.copy()
        self._cost_func = cost_func
        self._lbounds = lbounds
        self._ubounds = ubounds
        self._n_iters = n_iters
        self._tol = tol

        self._qubit_labels = list(params.keys())
        for ql, val in self._params.items():
            if not isinstance(val, list):
                self._params[ql] = [val]

        if self._lbounds and self._ubounds:
            self._scaler = {ql: MinMaxScaler() for ql in self._qubit_labels}
            self._x0 = {ql:
                self._scaler[ql].fit_transform(
                    np.vstack([
                        self._lbounds[ql],
                        [self._config[p] for p in self._params[ql]],
                        self._ubounds[ql],
                    ])
                )[1] for ql in self._qubit_labels
            }
        else:
            self._scaler = {ql: NoScaler() for ql in self._qubit_labels}
            self._x0 = {ql:
                self._scaler[ql].fit_transform(
                    np.array([[self._config[p] for p in self._params[ql]]])
                ) for ql in self._qubit_labels
            }
            # self._x0 = {ql:
            #     self._scaler[ql].fit_transform(
            #         np.array(
            #             [self._config[p] for p in self._params[ql]]
            #         ).reshape(-1, 1)
            #     ) for ql in self._qubit_labels
            # }

        opt_kwargs = opt_kwargs if opt_kwargs is not None else {
            ql: {} for ql in self._qubit_labels
        }
        self._optimizer = {
            ql: optimizer(self._x0[ql], **opt_kwargs[ql]) for ql in
            self._qubit_labels
        }

        if delta:
            self._delta = delta if isinstance(delta, dict) else {
                ql: np.array([delta] * len(self._x0[ql]))
                for ql in self._qubit_labels
            }
        else:
            self._delta = {}

        self._x = None
        self._loss = {}
        self._loss_history = {ql: [] for ql in self._qubit_labels}
        self._params_history = {ql: [] for ql in self._qubit_labels}
        self._opt_params = {}
        self._opt_loss = {}

    @property
    def config(self) -> Config:
        """Optimized config.

        Returns:
            Config: qcal config object.
        """
        return self._config

    @property
    def cost_func(self) -> Any:
        """Cost function used to perform the optimization.

        Returns:
            Any: cost function.
        """
        return self._cost_func

    @property
    def delta(self) -> Dict:
        """Initial perturbation for gradient-based optimizers.

        Returns:
            Dict: initial perturbation for each qubit label.
        """
        return self._delta

    @property
    def lbounds(self) -> Dict:
        """Lower bounds on each parameter for each qubit label.

        Returns:
            Dict: lower bounds.
        """
        return self._lbounds

    @property
    def loss(self) -> Dict:
        """Current loss for each qubit label.

        Returns:
            Dict: loss.
        """
        return self._loss

    @property
    def loss_history(self) -> Dict:
        """History of the losses for each iteration for each qubit label.

        Returns:
            Dict: history of losses.
        """
        return self._loss_history

    @property
    def n_iters(self) -> int:
        """Number of optimization iterations.

        Returns:
            int: number of iterations.
        """
        return self._n_iters

    @property
    def opt_loss(self) -> Dict:
        """Optimized loss values for each qubit label.

        Returns:
            Dict: optimized loss values.
        """
        return self._opt_loss

    @property
    def opt_params(self) -> Dict:
        """Optimized parameter values for each qubit label.

        Returns:
            Dict: optimized parameter values.
        """
        return self._opt_params

    @property
    def optimizer(self) -> Dict:
        """Optimizer for each qubit label.

        Returns:
            Dict: optimizer.
        """
        return self._optimizer

    @property
    def params(self) -> Dict:
        """Parameters to optimize.

        Returns:
            Dict: dictionary of qubit labels to parameters.
        """
        return self._params

    @property
    def params_history(self) -> Dict:
        """History of the (unscaled) paramters values for each qubit label.

        Returns:
            Dict: history of parameter values.
        """
        return self._params_history

    @property
    def scaler(self) -> Dict:
        """MinMaxScaler or NoScaler for each qubit label.

        The MinMaxScaler rescales all parameter values to between 0 and 1 for
        improved convergence. NoScaler does not rescale the parameter values.

        Returns:
            Dict: scaler for each qubit label.
        """
        return self._scaler

    @property
    def tol(self) -> float:
        """Tolerance for determining convergence.

        Returns:
            float: tolerance.
        """
        return self._tol

    @property
    def ubounds(self) -> Dict:
        """Upper bounds on each parameter for each qubit label.

        Returns:
            Dict: upper bounds.
        """
        return self._ubounds

    @property
    def x0(self) -> Dict:
        """Initial parameter values for each qubit label.

        Returns:
            Dict: dictionary of qubit labels to parameter values.
        """
        return self._x0

    @property
    def x(self) -> Dict:
        """Current parameter values for each qubit label.

        Parameter values are scaled between 0 and 1.

        Returns:
            Dict: dictionary of qubit labels to parameter values.
        """
        return self._x

    def compute_loss(self) -> None:
        """Compute the cost using the cost function."""
        self._loss = {
            ql: np.array(
                uncertainty_of_sum(self._cost_func.loss[ql])
                if len(self._cost_func.loss[ql]) > 1
                else self._cost_func.loss[ql][0]
            ) for ql in self._qubit_labels
        }

    def set_params(self, values: Dict[float | tuple, NDArray]) -> None:
        """Set the parameters in the config before measuring the cost.

        Args:
            values (Dict[float | tuple, NDArray]): a dictionary mapping qubit
                labels to an array of new param values.
        """
        for ql in self._qubit_labels:
            self._params_history[ql].append(
                self._scaler[ql].inverse_transform(
                    values[ql].reshape(1, -1)
                )[0].copy()
            )
            for i, param in enumerate(self._params[ql]):
                self._config[param] = self._params_history[ql][-1][i]

        # Re-instantiate cost function class with updated config
        self._cost_func._init_kwargs['config'] = self._config
        self._cost_func.__init__(
            *self._cost_func._init_args,
            **self._cost_func._init_kwargs
        )

        if '_param_sweep' in self._cost_func.__dict__.keys():
            self._cost_func._param_sweep = {
                ql: self._scaler[ql].inverse_transform(
                    values[ql].reshape(1, -1)
                )[0] for ql in self._qubit_labels
            }

    def run(self) -> None:
        """Run the optimization loop."""
        # Initial pertubation for computing a gradient
        if self._delta:
            self._x = {}
            for ql in self._qubit_labels:
                self._x[ql] = self._x0[ql] + self._delta[ql]
            self.set_params(self._x)

            self._cost_func.run()
            self.compute_loss()
            for ql in self._qubit_labels:
                self._optimizer[ql].tell(self._x[ql], self._loss[ql])

        for i in range(self._n_iters):  # TODO: add stop()?
            self._x = {
                ql: self._optimizer[ql].step() for ql in self._qubit_labels
            }

            losses = {ql: [] for ql in self._qubit_labels}
            self._n_value_sets = len(self._x[self._qubit_labels[0]])
            for j in range(self._n_value_sets):
                logger.info(f' Iteration: {i+1}/{self._n_iters}')
                self.set_params(
                    {ql: vals[j] for ql, vals in self._x.items()}
                )
                self._cost_func.run()
                self.compute_loss()
                for ql in self._qubit_labels:
                    losses[ql].append(self._loss[ql])

            for ql in self._qubit_labels:
                self._loss_history[ql].append(losses[ql].copy())
                self._optimizer[ql].tell(
                    self._x[ql].copy(), losses[ql].copy()
                )

            self.plot()

        self._opt_params = {
            ql: self._optimizer[ql].opt_x for ql in self._qubit_labels
        }
        self._opt_loss = {
            ql: self._optimizer[ql].opt_loss
            for ql in self._qubit_labels
        }

        logger.info(' Optimization complete!')
        logger.info(' Optimized parameters:')
        self.set_params(self._opt_params)
        self._config.save()

    def plot(self) -> None:
        """Plot the optimization results."""
        nrows, ncols = calculate_nrows_ncols(len(self._qubit_labels))
        figsize = (5 * ncols, 4 * nrows)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, layout='constrained'
        )

        k = -1
        for i in range(nrows):
            for j in range(ncols):
                k += 1

                if len(self._qubit_labels) == 1:
                    ax = axes
                elif axes.ndim == 1:
                    ax = axes[j]
                elif axes.ndim == 2:
                    ax = axes[i,j]

                if k < len(self._qubit_labels):
                    ql = self._qubit_labels[k]

                    ax.plot(np.abs(self._loss_history[ql]), 'o', alpha=0.2)
                    ax.errorbar(
                        np.arange(len(self._loss_history[ql])),
                        np.mean(np.abs(self._loss_history[ql]), 1),
                        yerr=np.std(np.abs(self._loss_history[ql]), 1),
                        color='black', ecolor='blueviolet',
                        elinewidth=3, capsize=0, label='Mean'
                    )

                    ax.text(
                            0.05, 0.95, f'Q{ql}', size=12,
                            transform=ax.transAxes
                        )
                    ax.set_xlabel('Iteration', fontsize=15)
                    ax.set_ylabel('Cost', fontsize=15)
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    ax.grid(True)
                    ax.set_yscale('log')
                    ax.legend(loc=1, fontsize=12)

                else:
                    ax.axis('off')

        fig.set_tight_layout(True)
        if settings.Settings.save_data:
            save_path = self._cost_func.data_manager.save_path
            fig.savefig(save_path + 'optimization_convergence.png', dpi=600)
            fig.savefig(save_path + 'optimization_convergence.pdf')
            fig.savefig(save_path + 'optimization_convergence.svg')
        plt.show()
