"""Submodule handling the main calibration class.

All calibration routines in :mod:`qcal.calibration` derive from the
abstract :class:`Calibration` class, which stores the swept config
parameters, the sweep results, and the fits, and provides the shared
:meth:`~Calibration.final`, :meth:`~Calibration.plot`, and
:meth:`~Calibration.set_param` methods. Concrete calibrations are created
by factory functions (e.g.
:func:`~qcal.calibration.single_qubit.Amplitude`) that combine a
:class:`~qcal.qpu.qpu.QPU` subclass with :class:`Calibration`, and are
run with ``cal.run()``, which generates the circuits, measures them,
analyzes the results, updates the config, and plots the fits.

Example:
    .. code-block:: python

        import numpy as np

        from qcal.backend.emulator import Emulator
        from qcal.calibration.single_qubit import Amplitude
        from qcal.config import EXAMPLE_CONFIG_PATH, Config

        config = Config(str(EXAMPLE_CONFIG_PATH))
        cal = Amplitude(
            Emulator,  # or a hardware QPU class, e.g. QubicQPU
            config,
            qubits=[0, 1],
            amplitudes=np.linspace(0, 1.0, 21),
        )
        cal.run()
        print(cal.calibrated_values)  # new X90 amplitude for each qubit
"""
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import qcal.settings as settings
from qcal.config import Config
from qcal.plotting.utils import calculate_nrows_ncols

logger = logging.getLogger(__name__)


class Calibration(ABC):
    """Main calibration class.

    This abstract class handles basic calibration methods. Subclasses
    must implement :meth:`generate_circuits` and :meth:`analyze`, and
    populate ``self._qubits``, ``self._params``, ``self._param_sweep``,
    ``self._sweep_results``, ``self._fit``, and ``self._cal_values``.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the main Calibration class.

        Args:
            config (Config): qcal Config object. A warning is logged if
                excited state promotion (``readout/esp/enable``) is
                enabled in the config.
        """
        self._config = config
        self._gate = None
        self._params = None
        self._subspace = None
        self._qubits = None
        self._param_sweep = {}
        self._sweep_results = {}
        self._fit = {}
        self._cal_values = defaultdict(lambda: False, {})

        if self._config['readout/esp/enable']:
            logger.warning(
                ' Excited State Promotion is enabled!'
                ' This can cause errors in calibration!'
            )

    @property
    def calibrated_values(self) -> Dict:
        """New values fit by the calibration.

        Returns:
            Dict: qubit to value map.
        """
        return self._cal_values

    @property
    def gate(self) -> str:
        """Gate being calibrated.

        Returns:
            str: name of gate.
        """
        return self._gate

    @property
    def param_sweep(self) -> Dict:
        """Sweep values for each param.

        Returns:
            Dict: qubit to sweep values map.
        """
        return self._param_sweep

    @property
    def params(self) -> Dict:
        """Config parameters to sweep over.

        Returns:
            Dict: qubit to param map.
        """
        return self._params

    @property
    def sweep_results(self) -> Dict:
        """Results from the calibration.

        Returns:
            Dict: calibration results for each qubit.
        """
        return self._sweep_results

    @property
    def subspace(self) -> str:
        """Subspace of the gate being calibrated.

        Returns:
            str: gate subspace.
        """
        return self._subspace

    @property
    def qubits(self) -> List | Tuple:
        """Qubits in the calibration.

        Returns:
            List | Tuple: qubit labels.
        """
        return self._qubits

    @abstractmethod
    def analyze(self) -> None:
        """Analyze the data."""

    @abstractmethod
    def generate_circuits(self) -> None:
        """Generate all calibration circuits."""

    def final(self) -> None:
        """Save and load the config after changing parameters."""
        if self._fit:
            for q in self._qubits:
                if isinstance(self._fit[q], (list, tuple)):
                    for i, fit in enumerate(self._fit[q]):
                        if fit.fit_success:
                            self.set_param(
                                self._params[q][i], self._cal_values[q][i]
                            )
                        elif self._cal_values[q][i]:
                            self.set_param(
                                self._params[q][i], self._cal_values[q][i]
                            )
                elif isinstance(self._fit[q], dict):
                    if all(fit.fit_success for fit in self._fit[q].values()):
                        for i, param in enumerate(self._params[q]):
                            self.set_param(
                                param, self._cal_values[q][i]
                            )
                    elif self._cal_values[q]:
                            for i, param in enumerate(self._params[q]):
                                self.set_param(
                                    param, self._cal_values[q][i]
                                )
                else:
                    if self._fit[q].fit_success or self._cal_values[q]:
                        if (isinstance(self._params[q], (list, tuple)) and
                            isinstance(self._cal_values[q], (list, tuple))):
                            for param, val in zip(
                                self._params[q],
                                self._cal_values[q],
                                strict=False
                            ):
                                self.set_param(param, val)
                        elif (
                            isinstance(self._params[q], (list, tuple)) and not
                            isinstance(self._cal_values[q], (list, tuple))):
                            for param in self._params[q]:
                                self.set_param(param, self._cal_values[q])
                        else:
                            self.set_param(
                                self._params[q], self._cal_values[q]
                            )

        if settings.Settings.save_data:
            self._config.save()
            # self._config.load()

    def plot(
        self, xlabel='Value Sweep', ylabel='Results', save_path=''
    ) -> None:
        """Plot the sweep and fit results.

        Args:
            xlabel (str, optional): x-axis label. Defaults to
                ``'Value Sweep'``.
            ylabel (str, optional): y-axis label. Defaults to ``'Results'``.
            save_path (str, optional): path prefix used when saving the
                figure (only if ``settings.Settings.save_data`` is
                ``True``). Defaults to ``''``.
        """
        nrows, ncols = calculate_nrows_ncols(len(self._qubits))
        figsize = (5 * ncols, 4 * nrows)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, layout='constrained'
        )

        k = -1
        for i in range(nrows):
            for j in range(ncols):
                k += 1

                if len(self._qubits) == 1:
                    ax = axes
                elif axes.ndim == 1:
                    ax = axes[j]
                elif axes.ndim == 2:
                    ax = axes[i,j]

                if k < len(self._qubits):
                    q = self._qubits[k]

                    ax.set_xlabel(xlabel, fontsize=15)
                    ax.set_ylabel(ylabel, fontsize=15)
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    ax.grid(True)

                    if isinstance(self._sweep_results[q], Dict):
                        colors = plt.cm.viridis(
                            np.linspace(0, 1, len(self._sweep_results[q]))
                        )
                        for label_idx, (label, result) in enumerate(
                            self._sweep_results[q].items()
                        ):
                            ax.plot(
                                self._param_sweep[q], result,
                                'o-', c=colors[label_idx], label=label
                            )
                    else:
                        ax.plot(
                            self._param_sweep[q], self._sweep_results[q],
                            'o', c='blue', label='Meas'
                        )
                    if self._fit and self._fit[q].fit_success:
                        x = np.linspace(
                            self._param_sweep[q][0],
                            self._param_sweep[q][-1],
                            100
                        )
                        ax.plot(
                            x, self._fit[q].predict(x),
                            '-', c='orange', label='Fit'
                        )
                        ax.axvline(
                            self._cal_values[q],
                            ls='--', c='k', label='Fit value'
                        )

                    elif self._cal_values[q]:
                        ax.axvline(
                            self._cal_values[q],
                            ls='--', c='k', label='Opt. value'
                        )

                    ax.text(
                            0.05, 0.95, f'Q{q}', size=12,
                            transform=ax.transAxes
                        )
                    ax.legend(loc=1, fontsize=10)

                else:
                    ax.axis('off')

        fig.set_tight_layout(True)
        if settings.Settings.save_data:
            fig.savefig(save_path + 'calibration_results.png', dpi=600)
            fig.savefig(save_path + 'calibration_results.pdf')
            fig.savefig(save_path + 'calibration_results.svg')
        plt.show()

    def set_param(self, param: str, newvalue: Any) -> None:
        """Set a config param to a new value.

        Args:
            param (str): config param.
            newvalue (Any): new value for the param.
        """
        try:
            self._config[param] = (
            float(newvalue) if isinstance(newvalue, float) else newvalue
        )
        except Exception as e:
            logger.warning(f' Could not set {param} in the config!')
            raise e

