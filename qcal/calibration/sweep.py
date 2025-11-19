"""Submodule for performing generic parameter sweeps for calibration.

"""
import logging
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from IPython.display import clear_output

import qcal.settings as settings
from qcal.circuit import Circuit, CircuitSet
from qcal.config import Config
from qcal.qpu.qpu import QPU
from qcal.utils import flatten

from .calibration import Calibration

logger = logging.getLogger(__name__)


def ParamSweep(
        qpu:         QPU,
        config:      Config,
        circuit:     Circuit,
        params:      Dict,
        param_sweep: Dict,
        maximize:    List[str] = None,
        minimize:    List[str] = None,
        xlabel:      str = 'Value Sweep',
        **kwargs
    ) -> Callable:
    """Parameter sweep calibration.

    This calibration finds the param value for which the leakage on
    `leakage_qubit` is minimized.

    Basic example useage:
    ```
    # Calibrate the frequency of the 11 <-> 02 transition of a CZ gate
    circuit = qc.Circuit([
        qc.Cycle({qc.X90(0), qc.X90(1)}),
        qc.Cycle({qc.X90(0), qc.X90(1)}),
        qc.Cycle({qc.CZ((0, 1))}),
    ])
    circuit.measure()

    # Sweep +/- 10 MHz around the expected 11 <-> 02 transition frequency
    freqs = np.linspace(-10, 10, 31) * MHz + config['two_qubit/(0, 1)/CZ/freq']

    cal = ParamSweep(
        CustomQPU,
        config,
        circuit=circuit,
        params={(0, 1): 'two_qubit/(0, 1)/CZ/freq'},
        param_sweep={(0, 1): freqs},
        maximize=['02'],
        minimize=['11']
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
        param_sweep (Dict): dictionary mapping the qubits on which leakage
            occurs to sweep values for the parameter to be optimized.
        maximize (List[str], optional): list of dit strings specifying on which
            state(s) to maximize the population. Defaults to None.
        minimize (List[str], optional): list of dit strings specifying on which
            state(s) to minimize the population. Defaults to None.
        xlabel (str, optional): x-axis label. Defaults to 'Value Sweep'.

    Returns:
        Callable: ParamSweep calibration class.
    """

    class ParamSweep(qpu, Calibration):
        """Parameter sweep calibration class.

        This class inherits a custom QPU from the ParamSweep calibration
        function.
        """

        def __init__(self,
                config:      Config,
                circuit:     Circuit,
                params:      Dict,
                param_sweep: Dict,
                maximize:    List[str] = None,
                minimize:    List[str] = None,
                xlabel:      str = 'Value Sweep',
                **kwargs
            ) -> None:
            """Initialize the ParamSweep class within the function."""
            qpu.__init__(self, config=config, **kwargs)
            Calibration.__init__(self, config)

            self._circuit = circuit
            self._params = params
            self._param_sweep = param_sweep
            self._maximize = maximize
            self._minimize = minimize
            self._xlabel = xlabel

            if not maximize and not minimize:
                raise ValueError(
                    'You must specify a dit string for at least one of maximize'
                    ' or minimize!'
                )

            self._qubits = list(self._params.keys())
            self._populations = {}

        @property
        def circuit(self) -> Circuit:
            """Circuit used for calibrating the params.

            Returns:
                Circuit: circuit.
            """
            return self._circuit

        @property
        def params(self) -> Dict:
            """Parameters which are optimized.

            Returns:
                Dict: dictionary mapping qubit labels to parameters.
            """
            return self._params

        @property
        def param_sweep(self) -> Dict:
            """Value sweeps for the parameters to be optimized.

            Returns:
                Dict: dictionary mapping qubit labels to parameter sweeps.
            """
            return self._param_sweep

        @property
        def populations(self) -> Dict:
            """Populations (i.e., probabilities) across the param sweep.

            Returns:
                Dict: dictionary mapping qubit labels to populations.
            """
            return self._populations

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

            qubits = sorted(set(flatten(self._qubits)))
            for ql in self._qubits:
                qindx = qubits.index(ql) if isinstance(ql, int) else (
                    tuple([qubits.index(q) for q in ql])
                )

                # Use middle circuit to find states
                states = set(self._circuits[
                        int(self._circuits.n_circuits / 2)
                    ].results.marginalize(qindx).states
                )
                for state in self._minimize:
                    states.add(state)
                for state in self._maximize:
                    states.add(state)

                self._populations[ql] = {
                    state: [] for state in sorted(states)
                }
                for circ in self._circuits:
                    results = circ.results.marginalize(qindx)
                    for state in states:
                        self._populations[ql][state].append(
                            results.populations[state]
                        )

                cal_values = []
                if self._maximize:
                    for state in self._maximize:
                        cal_values.append(float(
                            self._param_sweep[ql][np.array(
                                self._populations[ql][state]
                            ).argmax()]
                        ))
                if self._minimize:
                    for state in self._minimize:
                        cal_values.append(float(
                            self._param_sweep[ql][np.array(
                                self._populations[ql][state]
                            ).argmin()]
                        ))
                self._cal_values[ql] = np.mean(cal_values)
                self._sweep_results[ql] = self._populations[ql]

        def save(self) -> None:
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_ParamSweep_{"".join("Q" + str(q) for q in self._qubits)}'
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
                ylabel='Population',
                xlabel=self._xlabel,
                save_path=self._data_manager._save_path,
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

    return ParamSweep(
        config=config,
        circuit=circuit,
        params=params,
        param_sweep=param_sweep,
        maximize=maximize,
        minimize=minimize,
        xlabel=xlabel,
        **kwargs
    )
