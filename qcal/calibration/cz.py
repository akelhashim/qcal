"""Submodule for two-qubit CZ gate calibrations.

"""
from __future__ import annotations

import qcal.settings as settings

from .calibration import Calibration
from .utils import in_range # find_pulse_index
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.fitting.fit import (
    # FitAbsoluteValue, FitCosine, FitDecayingCosine, 
    FitParabola
)
# from qcal.math.utils import reciprocal_uncertainty, round_to_order_error
from qcal.gate.single_qubit import Meas, VirtualZ, X90
from qcal.gate.two_qubit import CZ
# from qcal.plotting.utils import calculate_nrows_ncols
from qcal.qpu.qpu import QPU
# from qcal.units import MHz, us

import logging
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from collections.abc import Iterable
from IPython.display import clear_output
from itertools import chain
from typing import Any, Callable, Dict, List, Tuple
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


def tomography_circuits(
        qubit_pairs: List[Tuple], n_elements: int
    ) -> CircuitSet:
    """Partial tomography circuits for CZ sweeps.

    Args:
        qubit_pairs (List[Tuple]): list of qubit pairs.
        n_elements (int):          size of parameter sweep.

    Returns:
        CircuitSet: partial tomography CZ circuits.
    """
    qubits = list(set(chain.from_iterable(qubit_pairs)))

    circuit_C0_X = Circuit([
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        # CZ
        Barrier(qubits),
        Cycle({CZ(pair) for pair in qubit_pairs}),
        Barrier(qubits),
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        # Measure
        Cycle(Meas(q) for q in qubits)
    ])

    circuit_C1_X = Circuit([
        # X on control qubit
        Cycle({X90(p[0]) for p in qubit_pairs}),
        Cycle({X90(p[0]) for p in qubit_pairs}),
        Barrier(qubits),
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        # CZ
        Barrier(qubits),
        Cycle({CZ(pair) for pair in qubit_pairs}),
        Barrier(qubits),
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
    ])

    circuit_C0_Y = Circuit([
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        # CZ
        Barrier(qubits),
        Cycle({CZ(pair) for pair in qubit_pairs}),
        Barrier(qubits),
        # X90 on target qubit
        Cycle({X90(p[1]) for p in qubit_pairs}),
    ])

    circuit_C1_Y = Circuit([
        # X on control qubit
        Cycle({X90(p[0]) for p in qubit_pairs}),
        Cycle({X90(p[0]) for p in qubit_pairs}),
        Barrier(qubits),
        # Y90 on target qubit
        Cycle({VirtualZ(np.pi/2, p[1]) for p in qubit_pairs}),
        Cycle({X90(p[1]) for p in qubit_pairs}),
        Cycle({VirtualZ(-np.pi/2, p[1]) for p in qubit_pairs}),
        # CZ
        Barrier(qubits),
        Cycle({CZ(pair) for pair in qubit_pairs}),
        Barrier(qubits),
        # X90 on target qubit
        Cycle({X90(p[1]) for p in qubit_pairs}),
    ])

    circuits = list()
    circuits.extend([circuit_C0_X.copy() for _ in range(n_elements)])
    circuits.extend([circuit_C1_X.copy() for _ in range(n_elements)])
    circuits.extend([circuit_C0_Y.copy() for _ in range(n_elements)])
    circuits.extend([circuit_C1_Y.copy() for _ in range(n_elements)])
    
    cs = CircuitSet(circuits)
    cs['sequence'] = (
        ['C0_X'] * n_elements + ['C1_X'] * n_elements +
        ['C0_Y'] * n_elements + ['C1_Y'] * n_elements 
    )

    return cs


def RelativePhase(
        qpu:             QPU,
        config:          Config,
        qubit_pairs:     List[Tuple],
        phases:          ArrayLike | NDArray | Dict[ArrayLike | NDArray],
        compiler:        Any | Compiler | None = None, 
        transpiler:      Any | None = None, 
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        esp:             bool = False,
        heralding:       bool = True,
        **kwargs
    ) -> Callable:


    class RelativePhase(qpu, Calibration):
        """Relative phase calibration class.
        
        This class inherits a custom QPU from the RelativePhase calibration
        function.
        """

        def __init__(self, 
                config:          Config,
                qubit_pairs:     List[Tuple],
                phases:          ArrayLike | Dict[ArrayLike],
                compiler:        Any | Compiler | None = None, 
                transpiler:      Any | None = None, 
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2, 
                esp:             bool = False,
                heralding:       bool = True,
                **kwargs
            ) -> None:
            """Initialize the RelativePhase class within the function."""
            qpu.__init__(self,
                config, 
                compiler, 
                transpiler, 
                n_shots, 
                n_batches, 
                n_circs_per_seq, 
                n_levels,
                **kwargs
            )
            Calibration.__init__(self, 
                config, 
                esp=esp,
                heralding=heralding
            )

            self._qubit_pairs = qubit_pairs
            self._qubits = list(set(chain.from_iterable(qubit_pairs)))

            if not isinstance(phases, dict):
                self._phases = {pair: phases for pair in qubit_pairs}
            else:
                self._phases = phases
            self._param_sweep = self._phases

            self._params = {}
            for pair in qubit_pairs:
                # TODO: make compatible w/ pre-pulse
                self._params[pair] = (
                    f'two_qubit/{pair}/CZ/pulse/1/kwargs/phase'
                )
                
            self._R = {}
            self._fit = {pair: FitParabola() for pair in qubit_pairs}

        @property
        def phases(self) -> Dict:
            """Phase sweep for each qubit.

            Returns:
                Dict: qubit to array map.
            """
            return self._phases
        
        @property
        def qubit_pairs(self) -> List[Tuple]:
            """Qubit pair labels.

            Returns:
                List[Tuple]: qubit pairs.
            """
            return self._qubit_pairs
        
        @property
        def conditionality(self) -> Dict[Tuple]:
            """Conditionality computed at each phase value.

            Returns:
                Dict[Tuple]: conditionality for each pair.
            """
            return self._R

        def generate_circuits(self) -> None:
            """Generate all amplitude calibration circuits."""
            logger.info(' Generating circuits...')

            self._circuits = tomography_circuits(
                self._qubit_pairs,
                self._phases[self._qubit_pairs[0]].size
            )

            for pair in self._qubit_pairs:
                self._circuits[f'param: {self._params[pair]}'] = list(
                    self._phases[pair]
                ) * 4

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            # Compute the conditionality and fit to a parabola
            for pair in self._qubit_pairs:
                i = self._qubits.index(pair[1])

                prob_C0_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_X'].circuit:
                    prob_C0_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C1_X = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_X'].circuit:
                    prob_C1_X.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C0_Y = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C0_Y'].circuit:
                    prob_C0_Y.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                prob_C1_Y = []
                for circuit in self._circuits[
                    self._circuits['sequence'] == 'C1_Y'].circuit:
                    prob_C1_Y.append(
                        circuit.results.marginalize(i).populations['0']
                    )

                self._circuits[f'{pair}: pop0'] = (
                    prob_C0_X + prob_C1_X + prob_C0_Y + prob_C1_Y
                )
                self._R[pair] = np.sqrt(
                    (np.array(prob_C1_X) - np.array(prob_C0_X)) ** 2 + 
                    (np.array(prob_C1_Y) - np.array(prob_C0_Y)) ** 2
                )
                self._sweep_results[pair] = self._R[pair]

                self._fit[pair].fit(self._phases[pair], self._R[pair])
                # If the fit was successful, write find the new phase
                if self._fit[pair].fit_success:
                    a, b, _ = self._fit[pair].fit_params
                    newvalue = -b / (2 * a)  # Assume c = 0
                    if a > 0:
                        logger.warning(
                            f'Fit failed for {pair} (positive curvature)!'
                        )
                        self._fit[pair]._fit_success = False
                    elif not in_range(newvalue, self._phases[pair]):
                        logger.warning(
                            f'Fit failed for {pair} (out of range)!'
                        )
                        self._fit[pair]._fit_success = False
                    else:
                        self._cal_values[pair] = newvalue

        def save(self):
            """Save all circuits and data."""
            qpu.save(self)
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._sweep_results]), 'sweep_results'
            )
            self._data_manager.save_to_csv(
                 pd.DataFrame([self._cal_values]), 'calibrated_values'
            )

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CZ_phase_cal_Q{"".join(str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                self.save()
            self.plot(
                xlabel='Phase (rad.)',
                ylabel=(r'$|0\rangle$ Population'),
                save_path=self._data_manager._save_path
            )
            self.final()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

    return RelativePhase(
        config,
        qubit_pairs,
        phases,
        compiler, 
        transpiler, 
        n_shots, 
        n_batches, 
        n_circs_per_seq,
        n_levels, 
        esp,
        heralding,
        **kwargs
    )