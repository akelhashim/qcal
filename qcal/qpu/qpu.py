"""Quantum Processing Unit (QPU)

This module is used to run all experiments on hardware.

Basic example usage::

    qpu = QPU(config)
    qpu.run(circuit)
"""
import logging
import timeit
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import pandas as pd
from IPython.display import clear_output
from numpy.typing import NDArray

import qcal.settings as settings
from qcal.circuit import CircuitSet
from qcal.compilation.compiler import Compiler
from qcal.config import Config
from qcal.managers.classification_manager import ClassificationManager
from qcal.managers.data_manager import DataMananger
from qcal.transpilation.transpiler import Transpiler
from qcal.utils import load_from_pickle

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


__all__ = ('QPU',)


class QPU:
    """Quantum Processing Unit.

    This class handles the measurement of circuits, processing of data, and
    saving of results.

    Custom QPUs should inherit this parent class and overwrite the relevant
    methods.
    """
    __slots__ = (
        '_circuits',
        '_classified_results',
        '_classifier',
        '_compiled_circuits',
        '_compiler',
        '_config',
        '_data_manager',
        '_exp_circuits',
        '_jit',
        '_measurements',
        '_n_batches',
        '_n_circs_per_seq',
        '_n_levels',
        '_n_shots',
        '_raster_circuits',
        '_rcorr_cmat',
        '_runtime',
        '_sequence',
        '_transpiled_circuits',
        '_transpiler',
    )

    def __init__(
            self,
            config:          Config,
            compiler:        Any | Compiler | List[Any] | None = None,
            transpiler:      Any | Transpiler | List[Any] | None = None,
            classifier:      ClassificationManager | None = None,
            n_shots:         int = 1024,
            n_batches:       int = 1,
            n_circs_per_seq: int = 1,
            n_levels:        int = 2,
            jit:             bool = True,
            raster_circuits: bool = False,
            rcorr_cmat:      pd.DataFrame | None = None,
        ) -> None:
        """Initialize an instance of the Quantum Processing Unit.

        Args:
            config (Config): qcal config object.
            compiler (Any | Compiler | List[Any] | None, optional): a custom
                (set of) compilers to compile the experimental circuits.
                Defaults to ``None``.
            transpiler (Any | Transpiler | List[Any] | None, optional): a custom
                (set of) transpiler(s) to transpile the experimental circuits.
                Defaults to ``None``.
            classifier (ClassificationManager | None, optional): manager used
                for classifying raw data. Defaults to ``None``.
            n_shots (int, optional): number of measurements per circuit.
                Defaults to 1024.
            n_batches (int, optional): number of batches of measurements.
                Defaults to 1.
            n_circs_per_seq (int, optional): maximum number of circuits that
                can be measured per sequence. Defaults to 1.
            n_levels (int, optional): number of energy levels to be measured.
                Defaults to 2. If n_levels = 3, this assumes that the
                measurement supports qutrit classification.
            jit (bool, optional): enable just-in-time compilation. Defaults to
                ``True``. When ``True``, the classical prep (compile, transpile,
                generate_sequence) for batch i+1 runs concurrently with the
                hardware acquisition of batch i.
            raster_circuits (bool, optional): whether to raster through all
                circuits in a batch during measurement. Defaults to ``False``.
                By default, all circuits in a batch will be measured ``n_shots``
                times one by one. If ``True``, all circuits in a batch will be
                measured back-to-back one shot at a time. This can help average
                out the effects of drift on the timescale of a measurement.
            rcorr_cmat (pd.DataFrame | None, optional): confusion matrix for
                readout correction. Defaults to ``None``. If passed, the readout
                correction will be applied to the raw bit strings in
                post-processing.
        """
        self._config = config
        self._compiler = compiler
        self._transpiler = transpiler
        self._classifier = classifier
        self._n_shots = n_shots
        self._n_batches = n_batches
        self._n_circs_per_seq = n_circs_per_seq
        self._jit = jit
        self._raster_circuits = raster_circuits
        self._rcorr_cmat = rcorr_cmat

        assert n_levels <= 3, 'n_levels > 3 is not currently supported!'
        self._n_levels = n_levels

        if self._classifier is None:
            try:
                self._classifier = load_from_pickle(
                    settings.Settings.config_path +
                    'ClassificationManager.pkl'
                )
            except Exception:
                logger.warning(' No classifier has been instantiated!')

        if self._classifier is not None:
            assert n_levels <= self._classifier[
                self._classifier._qubits[0]
            ].n_components, (
                "'n_levels' is greater than the number of classified states!"
            )

        self._circuits = None
        self._classified_results = None
        self._compiled_circuits = None
        self._exp_circuits = None
        self._runtime = None
        self._sequence = None
        self._transpiled_circuits = None
        self._measurements = []
        self._data_manager = DataMananger()

    @property
    def circuits(self) -> Any | CircuitSet | None:
        """Circuits.

        Returns:
            Any | CircuitSet | None: all loaded circuits.
        """
        return self._circuits

    @property
    def classified_results(self) -> List[Dict[int, NDArray]] | None:
        """Classified results.

        Returns:
            List[Dict[int, NDArray]] | None: all classified results.
        """
        return self._classified_results

    @property
    def classifier(self) -> ClassificationManager | None:
        """Classification manager.

        Returns:
            ClassificationManager | None: current ClassificationManager
                instance.
        """
        return self._classifier

    @property
    def compiled_circuits(self) -> Any | CircuitSet | None:
        """Compiled circuits.

        Returns:
            Any | CircuitSet | None: all compiled circuits.
        """
        return self._compiled_circuits

    @property
    def exp_circuits(self) -> CircuitSet | None:
        """Experimental circuits.

        Returns:
            CircuitSet | None: experimental circuits of the current batch.
        """
        return self._exp_circuits

    @property
    def jit(self) -> bool:
        """Whether just-in-time compilation is enabled.

        Returns:
            bool: ``True`` if JIT is enabled.
        """
        return self._jit

    @property
    def compiler(self) -> Any | Compiler | List[Any] | None:
        """Compiler(s) loaded to the QPU.

        Returns:
            Any | Compiler | List[Any] | None: circuit compiler(s).
        """
        return self._compiler

    @property
    def config(self) -> Config:
        """Config loaded to the QPU.

        Returns:
            Config: experimental config.
        """
        return self._config

    @property
    def data_manager(self) -> DataMananger:
        """Data manager.

        Returns:
            DataMananger: current DataManager instance.
        """
        return self._data_manager

    @property
    def measurements(self) -> List[Any]:
        """Measurement objects.

        Returns:
            List[Any]: measurements.
        """
        return self._measurements

    @property
    def runtime(self) -> pd.DataFrame:
        """Runtime.

        Returns:
            pd.DataFrame: breakdown of current runtime.
        """
        return self._runtime

    @property
    def sequence(self) -> Any:
        """Current pulse sequence.

        Returns:
            Any: pulse sequence. The format will depend on the hardware backend.
        """
        return self._sequence

    @property
    def transpiled_circuits(self) -> Any | CircuitSet | None:
        """Transpiled circuits.

        Returns:
            Any | CircuitSet | None: all transpiled circuits.
        """
        return self._transpiled_circuits

    @property
    def transpiler(self) -> Any | Transpiler | List[Any] | None:
        """Transpiler(s) loaded to the QPU.

        Returns:
            Any | Transpiler | List[Any] | None: circuit transpiler(s).
        """
        return self._transpiler

    def _initialize(
        self,
        circuits:  CircuitSet | List[Any],
        n_shots:   int | None = None,
        n_batches: int | None = None
    ) -> None:
        """Initialize the experiment.

        Args:
            circuits (CircuitSet | List[Any]): circuits to measure.
            n_shots (int | None, optional): number of shots per batch.
                Defaults to ``None``.
            n_batches (int | None, optional): number of batches of shots.
                Defaults to ``None``.
        """
        self._data_manager.generate_exp_id()  # Create a new experimental id

        if not isinstance(circuits, List) and 'n_circuits' not in dir(circuits):
            circuits = [circuits]
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
            self._n_batches = n_batches

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

    def _prepare(self) -> None:
        """Compile, transpile, and generate sequences (classical CPU tasks)."""
        if self._compiler is not None:
            logger.info(' Compiling circuits...')
            t0 = timeit.default_timer()
            self.compile()
            self._runtime['Compile'] += round(timeit.default_timer() - t0, 1)

        if self._transpiler is not None:
            logger.info(' Transpiling circuits...')
            t0 = timeit.default_timer()
            self.transpile()
            self._runtime['Transpile'] += round(timeit.default_timer() - t0, 1)

        logger.info(' Generating sequences...')
        t0 = timeit.default_timer()
        self.generate_sequence()
        self._runtime['Sequencing'] += round(timeit.default_timer() - t0, 1)

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

    def generate_sequence(self) -> None:
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
        """Measure a set of circuits."""
        self._prepare()

        logger.info(' Writing sequences...')
        t0 = timeit.default_timer()
        self.write()
        self._runtime['Write'] += round(timeit.default_timer() - t0, 1)

        logger.info(' Measuring...')
        t0 = timeit.default_timer()
        self.acquire()
        self._runtime['Measure'] += round(timeit.default_timer() - t0, 1)

    def _batch_measurements(self) -> None:
        """Measurement batcher with JIT classical/quantum overlap.

        For multi-batch runs, the classical prep (compile, transpile,
        generate_sequence) for batch i+1 runs on the CPU while the hardware
        is acquiring batch i, parallelizing the classical and quantum execution
        time.
        """
        def _acquire() -> None:
            """Helper function to acquire in a background thread."""
            t0 = timeit.default_timer()
            self.acquire()
            self._runtime['Measure'] += round(timeit.default_timer() - t0, 1)

        if self._circuits.n_circuits <= self._n_circs_per_seq:
            logger.info(' No batching...')
            self.measure()

        else:
            batches = list(self._circuits.batch(self._n_circs_per_seq))
            n_circ_batches = len(batches)
            logger.info(
                f' Dividing {self._circuits.n_circuits} circuits into '
                f'{n_circ_batches} batches of size {self._n_circs_per_seq}...'
            )

            if self._jit:
                # Prepare the first batch before entering the pipeline loop.
                self._exp_circuits = batches[0]
                logger.info(
                    f' Batch 1/{n_circ_batches}: {batches[0].n_circuits} circuit(s)'
                )
                self._prepare()

                with ThreadPoolExecutor(max_workers=1) as pool:
                    for i in range(n_circ_batches):
                        if i > 4:
                            clear_output(wait=True)

                        logger.info(' Writing sequences...')
                        t0 = timeit.default_timer()
                        self.write()
                        self._runtime['Write'] += round(
                            timeit.default_timer() - t0, 1
                        )

                        # Acquire in a background thread; self._sequence has
                        # already been sent to hardware so the main thread can
                        # safely overwrite it while the hardware runs.
                        logger.info(' Measuring...')
                        future = pool.submit(_acquire)

                        # Prepare the next batch concurrently with acquire.
                        if i + 1 < n_circ_batches:
                            self._exp_circuits = batches[i + 1]
                            logger.info(
                                f' Batch {i+2}/{n_circ_batches}: '
                                f'{batches[i+1].n_circuits} circuit(s)'
                            )
                            self._prepare()

                        future.result()  # re-raises any exception from acquire

            else:
                for i, circuits in enumerate(
                    self._circuits.batch(self._n_circs_per_seq)
                ):
                    self._exp_circuits = circuits
                    if i > 4:
                        clear_output(wait=True)
                    logger.info(
                        f' Batch {i+1}/{n_circ_batches}: '
                        f'{circuits.n_circuits} circuit(s)'
                    )
                    self.measure()

        logger.info(' Processing...')
        t0 = timeit.default_timer()
        self.process()
        self._runtime['Process'] += round(timeit.default_timer() - t0, 1)

    def save(self, create_data_path: bool = True) -> None:
        """Save all circuits and data.

        Args:
            create_data_path (bool, optional): whether to create a data save
                path. Defaults to ``True``.
        """
        if create_data_path:
            self._data_manager.create_data_path()

        if (
            self._circuits is not None
            and len(self._circuits) > 0
        ):
            self._circuits.save(
                self._data_manager._save_path + 'circuits.qc'
            )

        if (
            self._compiled_circuits is not None
            and len(self._compiled_circuits) > 0
        ):
            self._compiled_circuits.save(
                self._data_manager._save_path + 'compiled_circuits.qc'
            )

        if (
            self._transpiled_circuits is not None
            and len(self._transpiled_circuits) > 0
        ):
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
            circuits (Any | List[Any]): circuits to measure.
            n_shots (int | None, optional): number of shots per batch.
                Defaults to ``None``.
            n_batches (int | None, optional): number of batches of shots.
                Defaults to ``None``.
            save (bool): whether or not to save data at the end of the run
                method. Defaults to ``True``. This should be used for determining
                when the data is saved for custom QPUs which inherit this
                class.
        """
        t_start = timeit.default_timer()
        self._initialize(circuits, n_shots, n_batches)
        self._batch_measurements()
        self._runtime['Total'] += round(
            timeit.default_timer() - t_start, 1
        )

        clear_output(wait=True)
        if settings.Settings.save_data and save:
            self.save()

        print(f"Runtime: {repr(self._runtime)[8:]}\n")
