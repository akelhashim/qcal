"""Submodule for calibrating leakage parameters, such as DRAG.

"""
import qcal.settings as settings

from .calibration import Calibration
from qcal.circuit import Circuit, CircuitSet
from qcal.config import Config
from qcal.qpu.qpu import QPU

import logging
import numpy as np
import pandas as pd

from collections.abc import Iterable
from IPython.display import clear_output
from typing import Callable, Dict

logger = logging.getLogger(__name__)


def Leakage(
        qpu:         QPU,
        config:      Config,
        circuit:     Circuit,
        params:      Dict,
        param_sweep: Dict,
        maximize:    bool = False,
        **kwargs
    ) -> Callable:
    """Leakage calibration.

    This calibration finds the param value for which the leakage on
    `leakage_qubit` is minimized.

    Basic example useage:
    ```
    # Amplify the leakage in an X90 gate
    circuit = Circuit([Cycle({X90(0)}) for _ in range(102)])
    circuit.measure()
    
    cal = Leakage(
        CustomQPU, 
        config, 
        circuit=circuit,
        params={0: 'single_qubit/0/GE/X90/pulse/1/kwargs/alpha'},
        param_sweep={0: np.linspace(0.5, 1.5, 21)}
    )
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        circuit (Circuit): circuit which amplifies leakage on some qubit.
        leakage_qubit (int): qubit on which leakage should be minimized.
        params (Dict): dictionary mapping the qubits on which leakage occurs to
            the config parameter which is causing the leakage.
        param_sweep (ArrayLike): dictionary mapping the qubits on which leakage 
            occurs to sweep values for the parameter to be optimized.
        maximize (bool, optional): whether to select the parameter that 
            maximizes leakage, instead of minimze. Defaults to False. This can
            be used to find gate parameters that amplify leakage, such that one
            can better tune other parameters to cancel leakage.

    Returns:
        Callable: Leakage calibration class.
    """

    class Leakage(qpu, Calibration):
        """leakage calibration class.
        
        This class inherits a custom QPU from the Leakage calibration
        function.
        """

        def __init__(self, 
                config:      Config,
                circuit:     Circuit,
                params:      Dict,
                param_sweep: Dict,
                maximize:    bool = False,
                **kwargs
            ) -> None:
            """Initialize the Leakage class within the function."""
            qpu.__init__(self, config=config, **kwargs)
            Calibration.__init__(self, config)

            self._circuit = circuit
            self._params = params
            self._param_sweep = param_sweep
            self._maximize = maximize

            self._qubits = list(self._params.keys())
            self._loss = {}

        @property
        def circuit(self) -> Circuit:
            """Circuit used for calibrating leakage.

            Returns:
                Circuit: circuit.
            """
            return self._circuit
        
        @property
        def params(self) -> Dict:
            """Parameters which are optimized to reducing leakage.

            Returns:
                Dict: dictionary mapping leakage qubits to parameters.
            """
            return self._params
        
        @property
        def param_sweep(self) -> Dict:
            """Value sweeps for the parameters to be optimized.

            Returns:
                Dict: dictionary mapping leakage qubits to parameter sweeps.
            """
            return self._param_sweep
    
        @property
        def loss(self) -> Dict:
            """Loss for each qubit in terms of the |2> state population.

            This property can be used for parameter optimization.

            Returns:
                Dict: loss for each qubit.
            """
            return self._loss
        
        def generate_circuits(self):
            """Generate a CircuitSet that sweeps over all param values."""
            logger.info(' Generating circuits...')

            self._circuits = CircuitSet(
                circuits=[
                    self._circuit.copy() for _ in 
                    range(len(list(self._param_sweep.values())[0]))
                ]
            )

            for q, param in self._params.items():
                if isinstance(param, list):
                    for p in param:
                        self._circuits[
                            f'param: {p}'
                        ] = self._param_sweep[q]
                else:
                    self._circuits[
                            f'param: {param}'
                        ] = self._param_sweep[q]
            
        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            for ql in self._qubits:
                # pop0 = []
                # pop1 = []
                pop2 = []
                if isinstance(ql, Iterable):
                    for circ in self._circuits:
                        # val0 = 0
                        # val1 = 0
                        val2 = 0
                        for q in ql:
                            i = self._circuit.qubits.index(q)
                            # val0 += circ.results.marginalize(i).populations['0']
                            # val1 += circ.results.marginalize(i).populations['1']
                            val2 += circ.results.marginalize(i).populations['2']
                        # pop0.append(val0)
                        # pop1.append(val1)
                        pop2.append(val2)
                        
                else:
                    i = self._circuit.qubits.index(ql)
                    for circ in self._circuits:
                        # pop0.append(
                        #     circ.results.marginalize(i).populations['0']
                        # )
                        # pop1.append(
                        #     circ.results.marginalize(i).populations['1']
                        # )
                        pop2.append(
                            circ.results.marginalize(i).populations['2']
                        )
                
                self._sweep_results[ql] = pop2
                # self._sweep_results[ql] = {
                #     'Prob(0)': pop0,
                #     'Prob(1)': pop1,
                #     'Prob(2)': pop2
                # }
                self._circuits[f'Q{ql}: Prob(2)'] = pop2
                if self._maximize:
                    self._cal_values[ql] = float(
                        self._param_sweep[ql][np.array(pop2).argmax()]
                        )
                else:
                    self._cal_values[ql] = float(
                        self._param_sweep[ql][np.array(pop2).argmin()]
                        )
                
                self._loss[ql] = np.array([np.array(pop2).min()])

        def save(self) -> None:
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_Leakage_{"".join("Q" + str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._sweep_results]), 'sweep_results'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._cal_values]), 'calibrated_values'
                )

        def plot(self) -> None:
            """Plot the sweep results"""
            Calibration.plot(self,
                # ylabel=r'$|2\rangle$ Population',
                ylabel='Probability',
                save_path=self._data_manager._save_path
            )

        def final(self) -> None:
            """Final calibration method."""
            if self._cal_values:
                for q, val in self._cal_values.items():
                    if isinstance(self._params[q], list):
                        for p in self._params[q]:
                            self.set_param(p, val)
                    else:
                        self.set_param(self._params[q], val)
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

    return Leakage(
        config=config,
        circuit=circuit,
        params=params,
        param_sweep=param_sweep,
        maximize=maximize,
        **kwargs
    )
