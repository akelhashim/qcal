"""Submodule for RB routines.

"""
from qcal.config import Config
from qcal.qpu.qpu import QPU

import logging

from typing import Any, Callable, List, Tuple, Iterable

logger = logging.getLogger(__name__)


def SRB(qpu:             QPU,
        config:          Config,
        qubit_labels:    Iterable[int, Iterable[int]],
        circuit_depths:  List[int] | Tuple[int],
        tq_config:       str | Any = None,
        compiler:        Any | None = None, 
        transpiler:      Any | None = None,
        n_circuits:      int = 30,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        compiled_pauli:  bool = True,
        include_rcal:    bool = True,
        **kwargs
    ) -> Callable:


    class SRB:
        """True-Q SRB protocol."""
        import trueq as tq

        def __init__(self,
                qpu:             QPU,
                config:          Config,
                qubit_labels:    Iterable[int, Iterable[int]],
                circuit_depths:  List[int] | Tuple[int],
                tq_config:       str | tq.Config = None,
                compiler:        Any | None = None, 
                transpiler:      Any | None = None,
                n_circuits:      int = 30,
                n_shots:         int = 1024,
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2,
                compiled_pauli:  bool = True,
                include_rcal:    bool = True,
                **kwargs
            ) -> None:
            from qcal.compilation.trueq.compiler import Compiler
            from qcal.transpilation.trueq.transpiler import Transpiler
            
            self._qubit_labels = qubit_labels
            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._compiled_pauli = compiled_pauli
            self._include_rcal = include_rcal
            
            if compiler is None:
                compiler = Compiler(config if tq_config is None else tq_config)
            if transpiler is None:
                transpiler = Transpiler()
                
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

        def generate_circuits(self):
            """Generate all True-Q SRB circuits."""
            logger.info(' Generating circuits from True-Q...')
            import trueq as tq

            self._circuits = tq.make_srb(
                self._qubit_labels,
                self._circuit_depths,
                self._n_circuits,
                self._compiled_pauli
            )

            if self._include_rcal:
                self._circuits.append(tq.make_rcal(self._circuits.labels))

        def analyze(self):
            """Analyze the SRB results."""
            logger.info(' Analyzing the results...')
            print(self._circuits.fit())

        def plot(self):
            """Plot the results."""

            

    return SRB(
        qpu,
        config,
        qubit_labels,
        circuit_depths,
        tq_config,
        compiler,
        transpiler,
        n_circuits,
        n_shots,
        n_batches,
        n_circs_per_seq,
        n_levels,
        compiled_pauli,
        include_rcal,
        **kwargs
    )