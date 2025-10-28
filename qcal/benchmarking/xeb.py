"""Submodule for cross-entropy benchmarking.

"""
import qcal.settings as settings

from qcal.circuit import CircuitSet
from qcal.config import Config
from qcal.qpu.qpu import QPU

import logging

from IPython.display import clear_output
from typing import Any, Dict, Callable, Iterable


logger = logging.getLogger(__name__)


def XEB(qpu:               QPU,
        config:            Config,
        api_key:           str,
        interleaved_layer: Any,
        circuit_depths:    Iterable[int],
        n_circuits:        int = 30,
        **kwargs
    ) -> Callable:
    """Cross-Entropy Benchmarking

    This is a SupermarQ protocol. See:
    https://github.com/Infleqtion/client-superstaq/blob/main/
    supermarq-benchmarks/supermarq/qcvv/xeb.py

    Args:
        qpu (QPU): custom QPU object.
        api_key (str): Superstaq API key.
        config (Config): qcal Config object.
        interleaved_layer (cirq.Gate | cirq.Moment | cirq.Circuit): gate, 
            moment, or subcircuit to benchmark.
        circuit_depths (Iterable[int]): a list of positive integers 
            specifying how many interleaved cycles of the target layer and 
            random SU(2) gates to generate, for example, [4, 16, 64].
        n_circuits (int, optional): the number of circuits for each circuit 
            depth. Defaults to 30.

    Returns:
        Callable: XEB class instance.
    """

    class XEB(qpu):
        """Supermarq XEB protocol."""

        def __init__(self,
                config:            Config,
                api_key:           str,
                interleaved_layer: Any,
                circuit_depths:    Iterable[int],
                n_circuits:        int = 30,
                **kwargs
            ) -> None:
            from qcal.interface.superstaq.compiler import CirqCompiler
            from qcal.interface.superstaq.transpiler import CirqTranspiler
            
            try:
                import supermarq
                logger.info(f" SupermarQ: {supermarq.__version__}")
            except ImportError:
                logger.warning(' Unable to import supermarq!')

            try:
                import cirq
                logger.info(f" cirq: {cirq.__version__}")
            except ImportError:
                logger.warning(' Unable to import cirq!')
            
            self._interleaved_layer = interleaved_layer
            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._qubits = tuple([q.x for q in interleaved_layer.qubits])
            
            compiler = kwargs.get('compiler', CirqCompiler(api_key, config))
            kwargs.pop('compiler', None)

            transpiler = kwargs.get('transpiler', CirqTranspiler())
            kwargs.pop('transpiler', None)
                
            qpu.__init__(self,
                config=config, 
                compiler=compiler, 
                transpiler=transpiler,
                **kwargs
            )

            self._xeb = None
            self._records = None
            self._qubit_subsets = None

        @property
        def records(self) -> Dict:
            """Results records.

            Map of UUID to results dictionary.

            Returns:
                Dict: results records.
            """
            return self._records
        
        @property
        def results(self) -> Any:
            """QCVV results.

            Returns:
                Any: SupermarQ results object.
            """
            return self._results

        def generate_circuits(self):
            """Generate all SupermarQ XEB circuits."""
            logger.info(' Generating circuits from SupermarQ...')
            from supermarq import qcvv

            self._xeb = qcvv.XEB(
                interleaved_layer=self._interleaved_layer,
                cycle_depths=self._circuit_depths,
                num_circuits=self._n_circuits
            )
            self._circuits = CircuitSet(
                [sample.circuit for sample in self._xeb.samples]
            )
            self._circuits['uuids'] = [
                sample.uuid for sample in self._xeb.samples
            ]

        def analyze(self):
            """Analyze the XEB results."""
            logger.info(' Analyzing the results...')

            self._records = {
                str(uuid): result for uuid, result in zip(
                    self._circuits['uuids'], self._circuits['results']
                )
            }
            self._results = self._xeb.results_from_records(self._records)
            self._qubit_subsets = self._circuits[0].get_independent_qubit_sets()

            print('')
            try:
                for qs in self._qubit_subsets:
                    results = self._results[[q for q in list(qs)]]
                    qubits = [q.x for q in list(qs)]
                    results.analyze(
                        plot_filename=self._data_manager._save_path + 
                        f'XEB_Q{"".join(str(q) for q in qubits)}.png' 
                        if settings.Settings.save_data else None
                    )
                    results.plot_speckle(
                        filename=self._data_manager._save_path + 
                        f'speckle_Q{"".join("Q"+str(q) for q in qubits)}.png' 
                        if settings.Settings.save_data else None
                    )
            except Exception:
                logger.warning(' Unable to analyze the data!')

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_XEB_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self) 

        def final(self) -> None:
            """Final benchmarking method."""
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.save() 
            self.analyze()
            self.final()

    return XEB(
        config=config,
        api_key=api_key,
        interleaved_layer=interleaved_layer,
        circuit_depths=circuit_depths,
        n_circuits=n_circuits,
        **kwargs
    )
