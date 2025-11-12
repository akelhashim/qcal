"""Submodule for estimating the ZZ phase (i.e. rate) between qubits.

"""
import qcal.settings as settings

from qcal.characterization.characterize import Characterize
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.config import Config
from qcal.fitting.fit import FitDecayingCosine
from qcal.gate.single_qubit import Idle, Rz, X90
from qcal.math.utils import (
    uncertainty_of_sum, round_to_order_error
)
from qcal.plotting.utils import calculate_nrows_ncols
from qcal.qpu.qpu import QPU
from qcal.units import kHz, MHz, us
from qcal.utils import flatten, save_init

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.display import clear_output
from lmfit import Parameters
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


def JAZZ(
        qpu:         QPU,
        config:      Config,
        qubit_pairs: List[Tuple],
        t_max:       float = 5*us,
        detuning:    float = 1*MHz,
        subspace:    str = 'GE',
        n_elements:  int = 50,
        params:      Dict | None = None,
        **kwargs
    ) -> Callable:
    """Joint Amplification of ZZ (JAZZ).

    This characterization measures the ZZ rotation angle/phase between qubits.
    To do so, we perform a Ramsey on the target qubit when the control qubit is
    in |0> and |1>, while echoing both qubits with a pi pulse in the middle. The
    difference in frequency between the two Ramsey experiments corresponds to
    the ZZ coupling strength between the two qubits.

    JAZZ is based on Bilinear Rotational Decoupling (BIRD) sequeneces:
    https://www.sciencedirect.com/science/article/pii/0009261482832296

    Basic example useage for initial calibration:

    ```
    cal = LocalPhases(
        CustomQPU, 
        config, 
        qubit_pairs=[(0, 1), (2, 3)])
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_pairs (List[Tuple]): pairs of qubit labels for the two-qubit gate
            calibration.

    Returns:
        Callable: JAZZ characterization class.
    """

    class JAZZ(qpu, Characterize):
        """JAZZ characterization class.
        
        This class inherits a custom QPU from the JAZZ characterization
        function.
        """

        @save_init
        def __init__(self, 
                config:      Config,
                qubit_pairs: List[Tuple],
                t_max:       float = 5*us,
                detuning:    float = 1*MHz,
                subspace:    str = 'GE',
                n_elements:  int = 50,
                params:      Dict | None = None,
                **kwargs
            ) -> None:
            """Initialize the JAZZ class within the function."""
            qpu.__init__(self, config=config, **kwargs)
            Characterize.__init__(self, config)
            
            self._qubits = qubit_pairs
            self._detuning = detuning

            assert subspace in ('GE', 'EF'), (
                "'subspace' must be one of 'GE' or 'EF'!"
            )
            self._subspace = subspace

            self._times = {
                q: np.linspace(0., t_max, n_elements) for q in qubit_pairs
            }
            self._param_sweep = self._times

            if params:
                self._params = params
            else:
                self._params = {}
                for qp in qubit_pairs:
                    self._params[qp] = f'two_qubit/{qp}/ZZ'
                
            self._fit = {
                qp: {
                    'C0': FitDecayingCosine(), 
                    'C1': FitDecayingCosine(), 
                } for qp in qubit_pairs
            }
            
            self._circuits = None
            self._freq_C0 = {q: False for q in qubit_pairs}
            self._freq_C1 = {q: False for q in qubit_pairs}
        
        @property
        def qubit_pairs(self) -> List[Tuple]:
            """Qubit pair labels.

            Returns:
                List[Tuple]: qubit pairs.
            """
            return self._qubits
        
        @property
        def loss(self) -> Dict:
            """Loss for each qubit pair.

            This property can be used for parameter optimization.

            Returns:
                Dict: loss for each qubit pair.
            """
            loss = {}
            for qp in self._qubits:
                if self._char_values[qp]:
                    loss[qp] = [self._char_values[qp]['val']]

            return loss

        def generate_circuits(self) -> None:
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')
            phases = []
            sequence = []
            qubits = sorted(flatten(self._qubits))
            self._circuits = CircuitSet()
            for t in self._times[self._qubits[0]]:
                phase = 2. * np.pi * self._detuning * t  # theta = 2pi*freq*t
                phases.extend([phase] * 2)
                circuit_C0 = Circuit()
                circuit_C1 = Circuit()
                
                if self._subspace == 'EF':
                    cycles = [
                        Cycle({X90(q, subspace='GE') for q in qubits}),
                        Cycle({X90(q, subspace='GE') for q in qubits}),
                        Barrier(qubits)
                    ]
                    circuit_C0.extend(cycles)
                    circuit_C1.extend(cycles)

                # State prepration
                circuit_C0.extend([  # Control in |0>, target in |i->
                    Cycle({
                        X90(p[1], subspace=self._subspace) for p in self._qubits
                    }),
                    Barrier(qubits)
                ])
                circuit_C1.extend([  # Control in |1>, target in |i->
                    Cycle({
                        X90(q, subspace=self._subspace) for q in qubits
                    }),
                    Cycle({
                        X90(p[0], subspace=self._subspace) for p in self._qubits
                    }),
                    Barrier(qubits)
                ])

                # Ramsey on target qubit with echo on both
                cycles = [
                    Cycle({Idle(q, duration=t/2) for q in qubits}),
                    # Cycle({
                    #     Rz(p[1], phase/2, subspace=self._subspace) 
                    #     for p in self._qubits
                    # }),
                    Barrier(qubits),
                    # pi echo on both qubits
                    Cycle({
                        X90(q, subspace=self._subspace) for q in qubits
                    }),
                    Cycle({
                        X90(q, subspace=self._subspace) for q in qubits
                    }),
                    Barrier(qubits),
                    Cycle({Idle(q, duration=t/2) for q in qubits}),
                    Cycle({
                        Rz(p[1], phase, subspace=self._subspace) 
                        for p in self._qubits
                    }),
                    Barrier(qubits),
                ]
                circuit_C0.extend(cycles)
                circuit_C1.extend(cycles)

                # Measurement preparation
                cycle = Cycle(
                    {X90(p[1], subspace=self._subspace) for p in self._qubits}
                )
                circuit_C0.append(cycle)
                circuit_C1.append(cycle)

                # Measurement
                circuit_C0.measure()
                circuit_C1.measure()

                sequence.extend(['C0', 'C1'])
                self._circuits.append(circuit_C0)
                self._circuits.append(circuit_C1)

            self._circuits['sequence'] = sequence
            self._circuits['phase'] = phases         

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            # Fit the frequency of oscillations
            qubits = sorted(flatten(self._qubits))
            for qp in self._qubits:
                t = qubits.index(qp[1])

                prob_C0 = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0'].circuit:
                    prob_C0.append(
                        circuit.results.marginalize(t).populations['0']
                    )

                prob_C1 = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1'].circuit:
                    prob_C1.append(
                        circuit.results.marginalize(t).populations['0']
                    )

                self._results[qp] = {
                    'C0': prob_C0, 'C1': prob_C1
                }

                # a * np.exp(-b * x) * np.cos(2 * np.pi * c * x + d) + e
                e = np.array(prob_C0).min()
                a = np.array(prob_C0).max() - e
                b = np.mean( np.diff(prob_C0) / np.diff(self._times[qp]) ) / a
                params = Parameters()
                params.add('a', value=a)  
                params.add('b', value=b)
                params.add('c', value=self._detuning)
                params.add('d', value=0.)
                params.add('e', value=e)
                self._fit[qp]['C0'].fit(
                    self._times[qp], prob_C0, params=params
                )
                self._fit[qp]['C1'].fit(
                    self._times[qp], prob_C1, params=params
                )

                if self._fit[qp]['C0'].fit_success:
                    self._freq_C0[qp] = (
                        self._fit[qp]['C0'].fit_params['c'].value
                    )

                if self._fit[qp]['C1'].fit_success:
                    self._freq_C1[qp] = (
                        self._fit[qp]['C1'].fit_params['c'].value
                    )

                if self._freq_C0[qp] and self._freq_C1[qp]:
                    val = abs(
                        self._freq_C0[qp] - self._freq_C1[qp]
                    )
                    err = uncertainty_of_sum([
                        self._fit[qp]['C0'].error, 
                        self._fit[qp]['C1'].error
                    ])
                    val, err = round_to_order_error(val, err)
                    self._char_values[qp] = {'val': val, 'err': err}

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_ZZ_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._results]), 'sweep_results'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._char_values]), 'characterized_values'
                )

        def plot(self) -> None:
            """Plot the frequency sweep and fit results."""
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
                        qp = self._qubits[k]

                        ax.set_xlabel(r'Time ($\mu$s)', fontsize=15)
                        ax.set_ylabel( r'$|0\rangle$ Population', fontsize=15)
                        ax.tick_params(axis='both', which='major', labelsize=12)
                        ax.grid(True)

                        ax.plot(
                            self._times[qp], self._results[qp]['C0'],
                            'o', c='blue', label=rf'Q{qp[0]} $|0\rangle$'
                        )
                        ax.plot(
                            self._times[qp], self._results[qp]['C1'],
                            'o', c='red', label=rf'Q{qp[0]} $|1\rangle$'
                        )
                        
                        if self._fit[qp]['C0'].fit_success:
                            freq = (
                                self._fit[qp]['C0'].fit_params['c'].value
                            )
                            x = np.linspace(
                                self._times[qp][0], self._times[qp][-1], 100
                            )
                            ax.plot(
                                x, self._fit[qp]['C0'].predict(x), 'b-',
                                label=f'{freq / kHz:.3f} kHz'
                            )
                            
                        if self._fit[qp]['C1'].fit_success:
                            freq = (
                                self._fit[qp]['C1'].fit_params['c'].value
                            )
                            x = np.linspace(
                                self._times[qp][0], self._times[qp][-1], 100
                            )
                            ax.plot(
                                x, self._fit[qp]['C1'].predict(x), 'r-',
                                label=f'{freq / kHz:.3f} kHz'
                            )

                        title = f'{qp}'
                        if self._char_values[qp]:
                            val = self._char_values[qp]['val']
                            err = self._char_values[qp]['err']
                            title += (
                                f': ZZ = {val / kHz:.3f} ({err / kHz:.3f}) kHz'
                            )
                        ax.set_title(title)
                        
                        ax.legend(loc=0, fontsize=12)

                    else:
                        ax.axis('off')
                    
            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'ZZ_characterization.png', 
                    dpi=300
                )
                fig.savefig(
                    self._data_manager._save_path + 'ZZ_characterization.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'ZZ_characterization.svg'
                )
            plt.show()

        def final(self) -> None:
            """Final calibration method."""
            for qp in self._qubits:
                if self._char_values[qp]:
                    self.set_param(
                        self._params[qp], self._char_values[qp]['val']
                    )
                    print(
                        f"{qp}: ZZ = {self._char_values[qp]['val'] / kHz:.3f} "
                        f"({self._char_values[qp]['err'] / kHz:.3f}) kHz"
                    )

            if settings.Settings.save_data:
                self._config.save()

            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return JAZZ(
        config=config,
        qubit_pairs=qubit_pairs,
        t_max=t_max,
        detuning=detuning,
        subspace=subspace,
        n_elements=n_elements,
        params=params,
        **kwargs
    )