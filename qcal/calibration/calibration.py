"""Submodule handling the main calibration class.

"""
import qcal.settings as settings

from qcal.plotting.utils import calculate_nrows_ncols
from qcal.config import Config

import logging
import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class Calibration:
    """Main calibration class.
    
    This class will handle basic calibration methods.
    """

    def __init__(self, 
            config:    Config, 
            esp:       bool = False,
            heralding: bool = True
        ) -> None:
        """Initialize the main Calibration class.
        
        Args:
            config (Config): qcal Config object.
            esp (bool, optional): whether to enable excited state
                promotion for the calibration. Defaults to False.
            heralding (bool, optional): whether to enable heralding
                for the calibraion. Defaults to True.
        """
        self._config = config
        self._gate = None
        self._params = None
        self._subspace = None
        self._qubits = None
        self._param_sweep = {}
        self._sweep_results = {}
        self._fit = {}
        self._cal_values = {}
        self._errors = {}

        if esp and not self._config['readout/esp/enable']:
            self.set_param('readout/esp/enable', True)
            self._disable_esp = True
        else:
            self._disable_esp = False

        if heralding and not self._config['readout/herald']:
            self.set_param('readout/herald', True)
            self._disable_heralding = True
        else:
            self._disable_heralding = False

    @property
    def calibrated_values(self) -> Dict:
        """New values fit by the calibration.

        Returns:
            Dict: qubit to value map.
        """
        return self._cal_values
    
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
            if self._fit[q].fit_success:
                self.set_param(self._params[q], self._cal_values[q])

        if self._disable_esp:
            self.set_param('readout/esp/enable', False)

        if self._disable_heralding:
            self.set_param('readout/herald', False)

        self._config.save()
        self._config.load()

    def plot(
            self, xlabel='Value Sweep', ylabel='Results', save_path=''
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
                        self._param_sweep[q], self._sweep_results[q],
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
                            '-', c='orange', label='Fit'
                        )
                        ax.axvline(
                            self._cal_values[q],  
                            ls='--', c='k', label='Fit value'
                        )

                    ax.legend(loc=0, fontsize=12)

                else:
                    ax.axis('off')
            
        fig.set_tight_layout(True)
        if settings.Settings.save_data:
            fig.savefig(save_path + 'calibration_results.png', dpi=300)
        plt.show()
    
    def set_param(self, param: str, newvalue: Any) -> None:
        """Set a config param to a new value.

        Args:
            param (str): config param.
            newvalue (Any): new value for the param.
        """
        self._config[param] = newvalue
