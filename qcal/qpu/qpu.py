"""Quantum Processing Unit (QPU)
    
This submodule is used to run all experiments on hardware.

Basic example useage:

    qpu = QPU(config)
    qpu.run(circuit)
"""
from qcal.compilation.compiler import Compiler, DEFAULT_COMPILER
from qcal.config import Config
from qcal.circuit import CircuitSet

import logging
import numpy as np
import pandas as pd
import timeit

from IPython.display import clear_output
from typing import Any, Union

logger = logging.getLogger(__name__)


__all__ = ['QPU']


class QPU:

    __slots__ = (
        '_config',
        '_compiler',
        '_n_shots',
        '_n_batches',
        '_n_circs_per_seq',
        '_circuits',
        '_compiled_circuits',
        '_sequence',
        '_runtime'
    )

    def __init__(
            self,
            config: Config,
            compiler: Union[Compiler, None] = DEFAULT_COMPILER,
            n_shots: int = 1024,
            n_batches: int = 1,
            n_circs_per_seq: int = 1
        ) -> None:
        
        self._config = config
        self._compiler = compiler
        self._n_shots = n_shots
        self._n_batches = n_batches
        self._n_circs_per_seq = n_circs_per_seq

        self._circuits = CircuitSet()
        self._compiled_circuits = CircuitSet()
        self._sequence = None

        self._runtime = pd.DataFrame({
            'Compile':   0.0,
            'Transpile': 0.0,
            'Measure':   0.0,
            'Process':   0.0,
            'Total':     0.0},
          index=['Time (s)']
        )

    @property
    def config(self) -> Config:
        """Returns the config loaded to the QPU.

        Returns:
            Config: experimental config
        """
        return self._config
        
    @property
    def compiler(self) -> Any:
        """Returns the compiler(s) loaded to the QPU.

        Returns:
            Any: circuit compiler
        """
        return self._compiler

    @property
    def circuits(self) -> pd.DataFrame:
        """Returns a DataFrame view of the circuits
        
        Returns:
            pd.DataFrame: DataFrame of circuits and results for all circuits
        """
        return self._circuits._df

    @property
    def compiled_circuits(self) -> pd.DataFrame:
        """Returns a DataFrame view of the compiled circuits
        
        Returns:
            pd.DataFrame: DataFrame of circuits and results for all circuits
        """
        return self._compiled_circuits._df
    
    @property
    def seq(self):
        """Returns the current sequence.
        
        Returns:
            qcal.sequencer.sequencer.Sequence: pulse sequence.
        """
        return self._sequence
        
    def compile(self, circuits) -> None:
        
        if self._compiler is not None:
            compiled_circuits = self._compiler.compile(circuits)
            self._compiled_circuits.append(compiled_circuits)
            return compiled_circuits
        else:
            self._compiled_circuits = self._circuits.__copy__()

    def transpile(self, circuits=None) -> None:

        if circuits is None:
            circuits = self._compiled_circuits.Circuits.to_list()
        
        # TODO

    def measure(self) -> None:
        pass

    def process(self) -> None:
        pass

    def batch(self) -> None:

        n_circ_batches = int(
            np.ceil(self._circuits.n_circuits / self._n_circs_per_seq)
        )
        print((
            f"Dividing {self._circuits.n_circuits} circuits into "
            f"{n_circ_batches} batches of size {self._n_circs_per_seq}..."
        ))
        for i, circuits in enumerate(self._circuits.batch(
            self._n_circs_per_seq)):
            if i > 4:
                clear_output(wait=True)
            print(
                f"Batch {i+1}/{n_circ_batches}: {circuits.n_circuits} circuits"
            )
        
            t0 = timeit.default_timer()
            compiled_circuits = self.compile(circuits)
            self.runtime['Compile'][0] += round(timeit.default_timer() - t0, 1)

            t0 = timeit.default_timer()
            self.transpile(compiled_circuits)
            self.runtime['Transpile'][0] += round(
                timeit.default_timer() - t0, 1
            )

            t0 = timeit.default_timer()
            self.measure()
            self.runtime['Measure'][0] += round(timeit.default_timer() - t0, 1)

            t0 = timeit.default_timer()
            self.process()
            self.runtime['Process'][0] += round(timeit.default_timer() - t0, 1)

    def save(self) -> None:

        filepath = ''  # TODO
        
        self._circuits._df.to_pickle(f'{filepath}circuits.pkl')
        self._compiled_circuits._df.to_pickle(f'{filepath}compiled_circuits.pkl')
        self._runtime.to_csv(f'{filepath}runtime.csv')

        print(f"Data save location: {filepath}*\n")

    def run(
            self,
            circuits: Any,
            n_shots=None,
            n_batches=None,
            save: bool = True) -> None:
        
        if not isinstance(circuits, CircuitSet):
            self._circuits = CircuitSet(circuits=circuits)
        else:
            self._circuits = circuits

        if n_shots is not None:
            self._n_shots = n_shots
        if n_batches is not None:
            self._batches = n_batches
            
        self._runtime = pd.DataFrame({
            'Compile':   0.0,
            'Transpile': 0.0,
            'Measure':   0.0,
            'Process':   0.0,
            'Total':     0.0},
          index=['Time (s)']
        )
        
        t_start = timeit.default_timer()
        if self._circuits.n_circuits <= self._n_circs_per_seq:
            print('No batching...')
            
            t0 = timeit.default_timer()
            self.compile(self._circuits)
            self.runtime['Compile'][0] += round(timeit.default_timer() - t0, 1)

            t0 = timeit.default_timer()
            self.transpile()
            self.runtime['Transpile'][0] += round(
                timeit.default_timer() - t0, 1
            )

            t0 = timeit.default_timer()
            self.measure()
            self.runtime['Measure'][0] += round(timeit.default_timer() - t0, 1)

            t0 = timeit.default_timer()
            self.process()
            self.runtime['Process'][0] += round(timeit.default_timer() - t0, 1)

        else:
            self.batch()

        self._runtime['Total'][0] += round(timeit.default_timer() - t_start, 1)

        clear_output(wait=True)
        print("Done!")
        
        if save:
            self.save()

        print(f"Runtime: {repr(self._runtime)[8:]}\n")
