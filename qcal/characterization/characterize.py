"""Abstract base class for characterization protocols.

This module defines :class:`Characterize`, the base class shared by the
characterization protocols in :mod:`qcal.characterization` (e.g. coherence
measurements and spectroscopy). Subclasses implement
:meth:`~Characterize.generate_circuits` and :meth:`~Characterize.analyze`;
the base class provides common properties, plotting, and the
:meth:`~Characterize.final` step that writes characterized values back to
the :class:`~qcal.config.Config`.

Characterization protocols are typically created via factory functions
(e.g. :func:`qcal.characterization.coherence.T1`) which return an instance
of a protocol class that subclasses both a :class:`~qcal.qpu.qpu.QPU`
backend and :class:`Characterize`.

Example:
    .. code-block:: python

        from qcal.backend.qubic.qpu import QubicQPU
        from qcal.characterization.coherence import T1
        from qcal.config import Config
        from qcal.units import us

        config = Config('config.yaml')
        t1 = T1(QubicQPU, config, qubits=[0, 1], t_max=300 * us)
        t1.run()  # generate, execute, analyze, save, plot, and final
        print(t1.characterized_values)  # {0: T1 of Q0, 1: T1 of Q1}
"""
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from uncertainties import UFloat

import qcal.settings as settings
from qcal.config import Config
from qcal.plotting.utils import calculate_nrows_ncols

logger = logging.getLogger(__name__)


class Characterize(ABC):
    """Main characterization class.

    This class will handle basic characterization methods.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the main Characterize class.

        Args:
            config (Config): qcal Config object.
        """
        self._config = config
        self._gate = None
        self._params = None
        self._subspace = None
        self._qubits = None
        self._param_sweep = {}
        self._results = {}
        self._fit = {}
        self._char_values = defaultdict(lambda: False, {})

    @property
    def characterized_values(self) -> Dict:
        """Characterized values determined by the fit.

        Returns:
            Dict: qubit to value map. A value is a plain float where no
                uncertainty is computed, or a ``uncertainties.ufloat``
                where one is.
        """
        return self._char_values

    @property
    def gate(self) -> str:
        """Gate being characterized.

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
    def results(self) -> Dict:
        """Results from the characterization.

        Returns:
            Dict: characterization results for each qubit.
        """
        return self._results

    @property
    def subspace(self) -> str:
        """Subspace of the gate being characterized.

        Returns:
            str: gate subspace.
        """
        return self._subspace

    @property
    def qubits(self) -> List | Tuple:
        """Qubits in the characterization.

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
        for q in self._qubits:
            val = self._char_values[q]
            if isinstance(val, UFloat):
                val = val.n
            if self._fit and self._fit[q].fit_success:
                self.set_param(self._params[q], val)
            elif self._char_values[q]:
                self.set_param(self._params[q], val)

        if settings.Settings.save_data:
            self._config.save()

    def plot(
        self, xlabel='Value Sweep', ylabel='Results', flabel='Fit',
        save_path=''
    ) -> None:
        """Plot the sweep and fit results.

        Args:
            xlabel (str, optional): x-axis label. Defaults to 'Value Sweep'.
            ylabel (str, optional): y-axis label. Defaults to 'Results'.
            flabel (str, optional): label of the fitted quantity used in
                the legend. Defaults to 'Fit'.
            save_path (str, optional): directory prefix used when saving
                the figure (only if ``settings.Settings.save_data`` is
                ``True``). Defaults to ''.
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

                    ax.plot(
                        self._param_sweep[q], self._results[q],
                        'o', c='blue', label=f'Meas, Q{q}'
                    )
                    if self._fit[q].fit_success:
                        x = np.linspace(
                            self._param_sweep[q][0],
                            self._param_sweep[q][-1],
                            100
                        )
                        char_val = self._char_values[q]
                        ax.plot(
                            x, self._fit[q].predict(x),
                            '-', c='orange',
                            label=f'Fit: {flabel} = '\
                               f'{round(char_val.n / 1.e-6, 1)} '\
                               rf'({round(char_val.s / 1.e-6, 2)}) $\mu$s'
                        )

                    ax.legend(loc=0, fontsize=12)
                    ax.xaxis.set_major_formatter(
                        lambda x, pos: round(x / 1e-6, 1)
                    )

                else:
                    ax.axis('off')

        fig.set_tight_layout(True)
        if settings.Settings.save_data:
            fig.savefig(save_path + 'characterization_results.png', dpi=600)
            fig.savefig(save_path + 'characterization_results.pdf')
            fig.savefig(save_path + 'characterization_results.svg')
        plt.show()

    def set_param(self, param: str, newvalue: Any) -> None:
        """Set a config param to a new value.

        Args:
            param (str): config param.
            newvalue (Any): new value for the param.
        """
        try:
            self._config[param] = newvalue
        except Exception as e:
            logger.warning(f' Could not set {param} in the config!')
            raise e
