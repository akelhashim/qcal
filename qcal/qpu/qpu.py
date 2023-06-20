"""Quantum Processing Unit (QPU)
    
This submodule is used to run all experiments on hardware.

Basic example useage:

    qpu = QPU(config)
    qpu.run(circuit)
"""
from qcal.compilation.compiler import Compiler#, DEFAULT_COMPILER
from qcal.config import Config
from qcal.circuit import CircuitSet

import logging
import numpy as np
import pandas as pd
import timeit

from IPython.display import clear_output
from typing import Any, List

logger = logging.getLogger(__name__)


__all__ = ['QPU']


class QPU:

    __slots__ = (
        '_config',
        '_compiler',
        '_n_shots',
        '_n_batches',
        '_n_circs_per_seq',
        '_n_levels',
        '_circuits',
        '_compiled_circuits',
        '_sequence',
        '_runtime'
    )

    def __init__(
            self,
            config: Config,
            compiler: Any | Compiler | None = None,
            n_shots: int = 1024,
            n_batches: int = 1,
            n_circs_per_seq: int = 1,
            n_levels: int = 2
        ) -> None:
        
        self._config = config
        self._compiler = compiler
        self._n_shots = n_shots
        self._n_batches = n_batches
        self._n_circs_per_seq = n_circs_per_seq
        self._n_levels = n_levels

        self._circuits = CircuitSet()
        self._compiled_circuits = CircuitSet()
        self._sequence = None
        self._measurement = None

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
    
    def initialize(self, 
                   circuits: Any | List[Any],
                   n_shots: int | None = None,
                   n_batches: int | None = None
        ) -> None:
        """Initialize the experiment.

        Args:
            circuits (Union[Any, List[Any]]): circuits to measure.
            n_shots (Union[int, None], optional): number of shots per batch. 
                Defaults to None.
            n_batches (Union[int, None], optional): number of batches of shots.
                Defaults to None.
        """

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

    def batch(self) -> None:
        """Circuit batcher."""
        n_circ_batches = int(
                np.ceil(self._circuits.n_circuits / self._n_circs_per_seq)
            )
        logger.info((
                f"Dividing {self._circuits.n_circuits} circuits into "
                f"{n_circ_batches} batches of size {self._n_circs_per_seq}..."
            ))
        
        for i, circuits in enumerate(
            self._circuits.batch(self._n_circs_per_seq)):
            if i > 4:
                clear_output(wait=True)
            logger.info(
                f"Batch {i+1}/{n_circ_batches}: {circuits.n_circuits} circuits"
            )
        
            t0 = timeit.default_timer()
            compiled_circuits = self.compile(circuits)
            self._runtime['Compile'][0] += round(
                    timeit.default_timer() - t0, 1
                )

            t0 = timeit.default_timer()
            self.transpile(compiled_circuits)
            self._runtime['Transpile'][0] += round(
                    timeit.default_timer() - t0, 1
                )

            t0 = timeit.default_timer()
            self.measure()
            self._runtime['Measure'][0] += round(
                    timeit.default_timer() - t0, 1
                )

            t0 = timeit.default_timer()
            self.process()
            self._runtime['Process'][0] += round(
                    timeit.default_timer() - t0, 1
                )

    def compile(self, cs: CircuitSet) -> CircuitSet | List:
        """Compile all circuits using a custom compiler.

        Args:
            cs (CircuitSet): set of circuits to compile

        Returns:
            CircuitSet | List: set of compiled circuits.
        """
        if self._compiler is not None:
            compiled_circuits = self._compiler.compile(cs.circuits)
            self._compiled_circuits.append(compiled_circuits)
            return compiled_circuits
        else:
            self._compiled_circuits.append(cs.circuits.copy())
            return cs

    def transpile(self, circuits: List) -> None:
        """Transpile circuits to a sequence.

        Args:
            circuits (List): circuits to transpile.
        """
        pass # TODO

    def measure(self) -> None:
        pass

    def process(self) -> None:
        pass

    def save(self) -> None:

        pass
        # filepath = ''  # TODO
        
        # self._circuits._df.to_pickle(f'{filepath}circuits.pkl')
        # self._compiled_circuits._df.to_pickle(f'{filepath}compiled_circuits.pkl')
        # self._runtime.to_csv(f'{filepath}runtime.csv')

        # logger.info(f"Data save location: {filepath}*\n")

    def run(self,
            circuits: Any | List[Any],
            n_shots: int | None = None,
            n_batches: int | None = None
        ) -> None:
        """Run all experimental methods.

        Args:
            circuits (Union[Any, List[Any]]): circuits to measure.
            n_shots (Union[int, None], optional): number of shots per batch. 
                Defaults to None.
            n_batches (Union[int, None], optional): number of batches of shots.
                Defaults to None.
        """
        t_start = timeit.default_timer()
        self.initialize(circuits, n_shots, n_batches)
        
        if self._circuits.n_circuits <= self._n_circs_per_seq:
            logger.info('No batching...')
            
            t0 = timeit.default_timer()
            self.compile(self._circuits)
            self._runtime['Compile'][0] += round(timeit.default_timer() - t0, 1)

            t0 = timeit.default_timer()
            self.transpile()
            self._runtime['Transpile'][0] += round(
                timeit.default_timer() - t0, 1
            )

            t0 = timeit.default_timer()
            self.measure()
            self._runtime['Measure'][0] += round(timeit.default_timer() - t0, 1)

            t0 = timeit.default_timer()
            self.process()
            self._runtime['Process'][0] += round(timeit.default_timer() - t0, 1)

        else:
            self.batch()

        self._runtime['Total'][0] += round(timeit.default_timer() - t_start, 1)

        clear_output(wait=True)
        logger.info("Done!")
        print(f"Runtime: {repr(self._runtime)[8:]}\n")
