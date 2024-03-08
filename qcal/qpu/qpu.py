"""Quantum Processing Unit (QPU)
    
This submodule is used to run all experiments on hardware.

Basic example useage:

    qpu = QPU(config)
    qpu.run(circuit)
"""
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.circuit import CircuitSet
from qcal.managers.classification_manager import ClassificationManager
from qcal.managers.data_manager import DataMananger

import qcal.settings as settings

import logging
import numpy as np
import pandas as pd
import timeit

from collections.abc import Iterable
from IPython.display import clear_output
from typing import Any, List

# logger = logging.getLogger(__name__)
logger = logging.getLogger('QPU')
logging.getLogger().setLevel(logging.INFO)


__all__ = ('QPU')


class QPU:
    """Quantum Processing Unit.
    
    This class handles the measurement of circuits, processing of data, and
    saving of results.

    Custom QPUs should inherit this parent class and overwrite the relevant
    methods.
    """
    __slots__ = (
        '_config',
        '_compiler',
        '_transpiler',
        '_n_shots',
        '_n_batches',
        '_n_circs_per_seq',
        '_n_levels',
        '_raster_circuits',
        '_circuits',
        '_compiled_circuits',
        '_transpiled_circuits',
        '_exp_circuits',
        '_sequence',
        '_measurements',
        '_runtime',
        '_classifier',
        '_data_manager'
    )

    def __init__(
            self,
            config:          Config,
            compiler:        Any | Compiler | None = None,
            transpiler:      Any | None = None,
            classifier:      ClassificationManager = None,
            n_shots:         int = 1024,
            n_batches:       int = 1,
            n_circs_per_seq: int = 1,
            n_levels:        int = 2,
            raster_circuits: bool = False
        ) -> None:
        """Initialize an instance of the Quantum Processing Unit.

        Args:
            config (Config): qcal config object.
            compiler (Any | Compiler | None, optional): a custom compiler to
                compile the experimental circuits. Defaults to None.
            transpiler (Any | None, optional): a custom transpiler to 
                transpile the experimental circuits. Defaults to None.
            classifier (ClassificationManager, optional): manager used for
                classifying raw data. Defaults to None.
            n_shots (int, optional): number of measurements per circuit. 
                Defaults to 1024.
            n_batches (int, optional): number of batches of measurements. 
                Defaults to 1.
            n_circs_per_seq (int, optional): maximum number of circuits that
                can be measured per sequence. Defaults to 1.
            n_levels (int, optional): number of energy levels to be measured. 
                Defaults to 2. If n_levels = 3, this assumes that the
                measurement supports qutrit classification.
            raster_circuits (bool, optional): whether to raster through all
                circuits in a batch during measurement. Defaults to False. By
                default, all circuits in a batch will be measured n_shots times
                one by one. If True, all circuits in a batch will be measured
                back-to-back one shot at a time. This can help average out the 
                effects of drift on the timescale of a measurement.
        """
        self._config = config
        self._compiler = compiler
        self._transpiler = transpiler
        self._classifier = classifier
        self._n_shots = n_shots
        self._n_batches = n_batches
        self._n_circs_per_seq = n_circs_per_seq
        self._raster_circuits = raster_circuits

        assert n_levels <= 3, 'n_levels > is not currently supported!'
        self._n_levels = n_levels

        if classifier is not None:
            assert n_levels <= classifier[classifier._qubits[0]].n_components,(
                "'n_levels' is greater than the number of classified states!"
            )

        self._circuits = None
        self._compiled_circuits = None
        self._transpiled_circuits = None
        self._exp_circuits = None
        self._sequence = None
        self._measurements = []
        self._runtime = None
        self._data_manager = DataMananger()

    @property
    def config(self) -> Config:
        """Returns the config loaded to the QPU.

        Returns:
            Config: experimental config
        """
        return self._config
        
    @property
    def compiler(self) -> Any | List[Any]:
        """Returns the compiler(s) loaded to the QPU.

        Returns:
            Any | List[Any]: circuit compiler
        """
        return self._compiler
    
    @property
    def transpiler(self) -> Any | List[Any]:
        """Returns the transpiler(s) loaded to the QPU.

        Returns:
            Any | List[Any]: circuit transpiler
        """
        return self._transpiler

    @property
    def circuits(self) -> Any:
        """Circuits.

        Returns:
            Any: all loaded circuits.
        """
        return self._circuits

    @property
    def compiled_circuits(self) -> Any:
        """Compiled circuits.

        Returns:
            Any: all compiled circuits.
        """
        return self._compiled_circuits
        
    @property
    def transpiled_circuits(self) -> Any:
        """Transpiled circuits.

        Returns:
            Any: all transpiled circuits.
        """
        return self._transpiled_circuits
    
    @property
    def classifier(self) -> ClassificationManager:
        """Classification manager.

        Returns:
            ClassificationManager: current ClassificationManager instance.
        """
        return self._classifier

    @property
    def data_manager(self) -> DataMananger:
        """Data manager.

        Returns:
            DataMananger: current DataManager instance.
        """
        return self._data_manager
    
    @property
    def sequence(self):
        """Returns the current sequence.
        
        Returns:
            qcal.sequencer.sequencer.Sequence: pulse sequence.
        """
        return self._sequence
    
    @property
    def measurements(self) -> List[Any]:
        """Returns the list of measurement objects.

        Returns:
            List[Any]: measurements.
        """
        return self._measurements
    
    def initialize(self, 
            circuits:  Any | List[Any],
            n_shots:   int | None = None,
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
        self._data_manager.generate_exp_id()  # Create a new experimental id
        if isinstance(circuits, List):
            self._circuits = CircuitSet(circuits=circuits)
            self._compiled_circuits = CircuitSet()
        else:
            self._circuits = circuits
            self._compiled_circuits = circuits.__class__()
        self._transpiled_circuits = CircuitSet()
        self._exp_circuits = self._circuits

        if n_shots is not None:
            self._n_shots = n_shots
        if n_batches is not None:
            self._batches = n_batches

        self._measurements = []    
        self._runtime = pd.DataFrame({
            'Compile':    0.0,
            'Transpile':  0.0,
            'Sequencing': 0.0,
            'Write':      0.0,
            'Measure':    0.0,
            'Process':    0.0,
            'Total':      0.0},
          index=['Time (s)']
        )
        
    def compile(self) -> None:
        """Compile the circuits using a custom compiler."""
        if isinstance(self._compiler, Iterable):
            for compiler in self._compiler:
                self._exp_circuits = compiler.compile(self._exp_circuits)
        else:
            self._exp_circuits = self._compiler.compile(self._exp_circuits)
        self._compiled_circuits.append(self._exp_circuits)

    def transpile(self) -> None:
        """Transpile the circuits from some other format to qcal circuits."""
        if isinstance(self._transpiler, Iterable):
            for transpiler in self._transpiler:
                self._exp_circuits = transpiler.transpile(self._exp_circuits)
        else:
            self._exp_circuits = self._transpiler.transpile(self._exp_circuits)
        self._transpiled_circuits.append(self._exp_circuits)
        
    def generate_sequence(self, circuits: Iterable) -> None:
        """Generate a sequence from circuits."""
        pass

    def write(self) -> None:
        """Write the sequence to hardware."""
        pass

    def acquire(self) -> None:
        """Measure the sequence.

        The output of this method should be appended to self._measurements.
        """
        pass

    def process(self) -> None:
        """Post-process the data.
        
        The method should assign a results dictionary as an attribute to each
        circuit.
        """
        pass

    def measure(self) -> None:
        """Measure a set of circuits.

        Args:
            circuits (Any): circuits to measure.
        """
        if self._compiler is not None:
            logger.info(' Compiling circuits...')
            t0 = timeit.default_timer()
            self.compile()
            self._runtime['Compile'].iloc[0] += round(
                    timeit.default_timer() - t0, 1
                )

        if self._transpiler is not None:
            logger.info(' Transpiling circuits...')
            t0 = timeit.default_timer()
            self.transpile()
            self._runtime['Transpile'].iloc[0] += round(
                    timeit.default_timer() - t0, 1
                )
        
        logger.info(' Generating sequences...')
        t0 = timeit.default_timer()
        self.generate_sequence()
        self._runtime['Sequencing'].iloc[0] += round(
            timeit.default_timer() - t0, 1
        )

        logger.info(' Writing sequences...')
        t0 = timeit.default_timer()
        self.write()
        self._runtime['Write'].iloc[0] += round(
            timeit.default_timer() - t0, 1
        )

        logger.info(' Measuring...')
        t0 = timeit.default_timer()
        self.acquire()
        self._runtime['Measure'].iloc[0] += round(
                timeit.default_timer() - t0, 1
            )
        
    def batch_measurements(self) -> None:
        """Measurement batcher."""
        if self._circuits.n_circuits <= self._n_circs_per_seq:
            logger.info(' No batching...')
            self.measure()
        
        else: 
            n_circ_batches = int(
                    np.ceil(self._circuits.n_circuits / self._n_circs_per_seq)
                )
            logger.info(
                f' Dividing {self._circuits.n_circuits} circuits into '
                f'{n_circ_batches} batches of size {self._n_circs_per_seq}...'
            )
            
            for i, circuits in enumerate(
                self._circuits.batch(self._n_circs_per_seq)):
                self._exp_circuits = circuits
                if i > 4:
                    clear_output(wait=True)
                logger.info(
             f' Batch {i+1}/{n_circ_batches}: {circuits.n_circuits} circuit(s)'
                )
                self.measure()

        logger.info(' Processing...')
        t0 = timeit.default_timer()
        self.process()
        self._runtime['Process'][0] += round(
                timeit.default_timer() - t0, 1
            )

    def save(self) -> None:
        """Save all circuits."""
        self._data_manager.create_data_path()

        if len(self._circuits) > 0:
            self._circuits.save(
                self._data_manager._save_path + 'circuits.qc'
            )

        if len(self._compiled_circuits) > 0:
            self._compiled_circuits.save(
                self._data_manager._save_path + 'compiled_circuits.qc'
            )

        if len(self._transpiled_circuits) > 0: 
            self._transpiled_circuits.save(
                self._data_manager._save_path + 'transpiled_circuits.qc'
            )

        self._data_manager.save_to_pickle(self._measurements, 'measurements')
        self._data_manager.save_to_csv(self._runtime, 'runtime')
        self._config.save(self._data_manager._save_path + 'config.yaml')

        if self._classifier:
            self._data_manager.save_to_pickle(
                    self._classifier, 
                    'ClassificationManager'
                )

        logger.info(f" Data save location: {self._data_manager._save_path}\n")

    def run(self,
            circuits:  Any | List[Any],
            n_shots:   int | None = None,
            n_batches: int | None = None,
            save:      bool = True
        ) -> None:
        """Run all experimental methods.

        Args:
            circuits (Union[Any, List[Any]]): circuits to measure.
            n_shots (Union[int, None], optional): number of shots per batch. 
                Defaults to None.
            n_batches (Union[int, None], optional): number of batches of shots.
                Defaults to None.
            save (bool): whether or not to save data at the end of the run
                method. Defaults to True. This should be used for determining
                when the data is saved for custom QPUs which inherit this 
                class.
        """
        t_start = timeit.default_timer()
        self.initialize(circuits, n_shots, n_batches)
        self.batch_measurements()
        self._runtime['Total'][0] += round(timeit.default_timer() - t_start, 1)

        clear_output(wait=True)
        if settings.Settings.save_data and save:
            self.save()
        
        print(f"Runtime: {repr(self._runtime)[8:]}\n")
