"""Submodule handling the main characterization class.

"""
import qcal.settings as settings

from qcal.plotting.utils import calculate_nrows_ncols
from qcal.config import Config

import logging
import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class Characterize:
    """Main characterization class.
    
    This class will handle basic characterization methods.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the main Calibration class.
        
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
        self._char_values = {}
        self._errors = {}

    @property
    def characterized_values(self) -> Dict:
        """Characterized values determined by the fit.

        Returns:
            Dict: qubit to value map.
        """
        return self._char_values
    
    @property
    def gate(self) -> str:
        """Gate being calibration.

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
        """Qubits in the calibration.

        Returns:
            List | Tuple: qubit labels.
        """
        return self._qubits
    
    def analyze(self) -> None:
        """Analyze the data.
        
        Raises:
            NotImplementedError: this method should be handled in the child 
                class.
        """
        raise NotImplementedError(
            'This method should be handled by the child class!'
        )
    
    def generate_circuits(self) -> None:
        """Generate all calibration circuits.
        
        Raises:
            NotImplementedError: this method should be handled in the child 
                class.
        """
        raise NotImplementedError(
            'This method should be handled by the child class!'
        )

    def final(self) -> None:
        """Save and load the config after changing parameters."""
        for q in self._qubits:
            if self._fit and self._fit[q].fit_success:
                self.set_param(self._params[q], self._char_values[q])
            elif self._char_values[q]:
                self.set_param(self._params[q], self._char_values[q])

        self._config.save()
        self._config.load()

    def plot(
            self, xlabel='Value Sweep', ylabel='Results', flabel='Fit', 
            save_path=''
        ) -> None:
        """Plot the sweep and fit results.

        Args:
            xlabel (str, optional): x-axis label. Defaults to 'Value Sweep'.
            ylabel (str, optional): y-axis label. Defaults to 'Results'.
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
                        ax.plot(
                            x, self._fit[q].predict(x),
                            '-', c='orange', 
                            label=f'Fit: {flabel} = '\
                               f'{round(self._char_values[q] / 1.e-6, 1)} '\
                               rf'({round(self._errors[q] / 1.e-6, 2)}) $\mu$s'
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
        self._config[param] = newvalue
