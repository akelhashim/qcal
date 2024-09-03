"""Submodule for handling compilation of BQSKit circuits.

See: https://github.com/BQSKit/bqskit
"""
from qcal.circuit import Barrier, Circuit, CircuitSet
from qcal.config import Config
from qcal.interface.bqskit.transpiler import Transpiler

import logging
import multiprocessing as mp

from IPython.display import clear_output
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Compiler:
    """BQSKit compiler.

    The compiler is created from a qcal Config object.
    """
    import bqskit
    from bqskit import MachineModel
    from bqskit.ir.gates import (
            CNOTGate, CZGate, RZGate, SqrtXGate, XGate
    )
    from bqskit.compiler import Compiler

    def __init__(
            self, 
            config:             Config, 
            model:              MachineModel | None = None,
            gate_mapper:        Dict = {
                                    'Rz':   RZGate(),
                                    'X':    XGate(),
                                    'X90':  SqrtXGate(),
                                    'CNOT': CNOTGate(),
                                    'CX':   CNOTGate(),
                                    'CZ':   CZGate()
                                },
            optimization_level: int = 1,
            max_synthesis_size: int = 3,
            synthesis_epsilon:  float = 1e-3,
            error_threshold:    float | None = None,
            error_sim_size:     int = 8,
            compiler:           Compiler | None = None,
            seed:               int | None = None,
            with_mapping:       bool = False,
            justify:            bool = False,
            parallelize:        bool = False,
            **compiler_kwargs:  Any,
        ) -> None:
        """Initialize a BQSKit compiler.

        Args:
            config (Config): qcal Config object.
            model (MachineModel | None): a model of the target machine.
                Defaults to an all-to-all connected hardware with CNOTs and U3s
                as native gates. See :class:`MachineModel` for information
                on loading a preset model or creating a custom one.
            gate_mapper (Dict, optional): dictionary mapping the names of the
                natives gates in config.native_gates['set'] to the equivalent
                BQSKit gates. This is only used if a :class:`MachineModel`
                instance is not passed. Defaults to {'Rz': RZGate(), 
                'X': XGate(), 'X90': SqrtXGate(), 'CNOT': CNOTGate(),
                CX': CNOTGate(), 'CZ': CZGate()}.
            optimization_level (int): the degree of optimization in the 
                workflow. The workflow will produce shorter depth circuits at 
                the cost of performance with a higher number. An 
                optimization_level of 0 is not supported due to inherit 
                optimization in any workflow containing synthesis. Valid inputs 
                are 1, 2, 3, or 4. Defaults to 1. 
            max_synthesis_size (int, optional): the maximum size of a unitary 
                to synthesize or instantiate. Larger circuits will be 
                partitioned. Increasing this will most likely lead to better 
                results with an exponential time trade-off. Defaults to 3.
            synthesis_epsilon (float, optional): the maximum distance between 
                target and circuit unitary during any instantiation or 
                synthesis algorithms. Defaults to 1e-3.
            error_threshold (float | None, optional): this parameter controls 
                the verification mechanism in this compile function. By 
                default, it is set to None, so no verification is done. If you 
                set this to a float, the upper bound on compilation error is 
                calculated. If the upper bound is larger than this number, a 
                warning will be logged. Defaults to None.
            error_sim_size (int, optional): if an `error_threshold` is set, the 
                error upper bound is calculated by simulating blocks of this 
                size. As you increase `error_sim_size`, the upper bound on 
                error becomes more accurate. This setting is ignored with 
                direct synthesis compilations, i.e., when a state, system, or 
                unitary are given as input. Defaults to 8.
            compiler (Compiler | None, optional): pass a :class:`Compiler` to 
                prevent creating one. Save on startup time by passing a 
                compiler in when calling `compile` multiple times. Defaults to 
                None.
            seed (int | None, optional): set a seed for the compile function 
                for better reproducibility. If left as None, will not set seed.
                Defaults to None.
            with_mapping (bool, optional): if True, three values will be 
                returned instead of just the compiled circuit. The first value 
                is the compiled circuit, the second value is the initial 
                mapping, and the third value is the final mapping. The initial 
                mapping is a tuple where `initial_mapping[i] = j` implies that 
                logical qudit `i` in the input system starts on the physical 
                qudit `j` in the output circuit. Likewise, the final mapping 
                describes where the logical qudits are in the physical circuit 
                at the end of execution. Defaults to False.
            justify (bool, optional): whether to left justify all of the gates
                in the circuit after compilation. Defaults to False.
            parallelize (bool, optional): whether to use multiprocessing to
                parallelize the circuit compilation. Defaults to False. This
                can speed up the compilation process if there are many
                circuits and/or long circuit depths.
        """
        self._config = config

        if model is None:
            from bqskit import MachineModel
            from bqskit.ir.gates import RZGate
            native_gates = {
                gate_mapper[gate] for gate in config.native_gates['set']
            }
            native_gates.add(RZGate())
            self._gate_mapper = gate_mapper
            self._model = MachineModel(
                config.n_qubits,
                config.qubit_pairs,
                native_gates
            )
        else:
            self._model = model

        self._optimization_level = optimization_level
        self._max_synthesis_size = max_synthesis_size
        self._synthesis_epsilon = synthesis_epsilon
        self._error_threshold = error_threshold
        self._error_sim_size = error_sim_size
        self._compiler = compiler
        self._seed = seed
        self._with_mapping = with_mapping
        self._justify = justify
        self._parallelize = parallelize
        self._compiler_kwargs = compiler_kwargs

        self._to_bqskit = Transpiler(to_bqskit=True)
        self._to_qcal = Transpiler()

    @property
    def config(self) -> Config:
        """qcal config.

        Returns:
            Config: qcal Config object.
        """
        return self._config
    
    def compile(self, circuits: Circuit | CircuitSet | bqskit.Circuit
        ) -> CircuitSet:
        """Compile circuits using the BQSKit compiler.

        Args:
            circuits (Circuit | CircuitSet | bqskit.Circuit): circuits to
                compile.

        Returns:
            CircuitSet: compiled circuits.
        """
        from bqskit import compile
        if not isinstance(circuits, CircuitSet):
            circuits = CircuitSet(circuits=circuits)
        # Copy so we don't mutate the original circuits
        circuits = circuits.copy()

        to_qcal = False
        if isinstance(circuits[0], Circuit):
            to_qcal = True

            # Add a barrier accross entire register to maintain qubit labeling
            for circuit in circuits:
                circuit.prepend(Barrier(self._config.qubits))

            circuits = self._to_bqskit.transpile(circuits)

        if self._parallelize and mp.cpu_count() > 2:
            ccircuits = {}
            logger.info(
                f" Pooling {mp.cpu_count() - 2} processes for compilation..."
            )
            pool = mp.Pool(mp.cpu_count() - 2)
            try:
                for i, circuit in enumerate(circuits):
                    output = pool.apply_async(
                        compile,
                        args=[circuit],
                        kwds={
                            **{
                                'model': self._model,
                                'optimization_level': self._optimization_level,
                                'max_synthesis_size': self._max_synthesis_size,
                                'synthesis_epsilon': self._synthesis_epsilon,
                                'error_threshold': self._error_threshold,
                                'error_sim_size': self._error_sim_size,
                                'compiler': self._compiler,
                                'seed': self._seed,
                                'with_mapping': self._with_mapping,
                            },
                            **self._compiler_kwargs
                        }                      
                    )
                    output = output.get()
                    if self._with_mapping:
                        # ccircuit, input_qubits, output_qubits = output
                        ccircuit, _, _ = output
                    else:
                        ccircuit = output

                    if self._justify:
                        ccircuit.compress()
                        
                    ccircuits[i] = ccircuit
                    
            finally:
                pool.close()
                pool.join()
                ccircuits = list(dict(sorted(ccircuits.items())).values())

        else:
            ccircuits = []
            for circuit in circuits:

                output = compile(
                    circuit,
                    model=self._model,
                    optimization_level=self._optimization_level,
                    max_synthesis_size=self._max_synthesis_size,
                    synthesis_epsilon=self._synthesis_epsilon,
                    error_threshold=self._error_threshold,
                    error_sim_size=self._error_sim_size,
                    compiler=self._compiler,
                    seed=self._seed,
                    with_mapping=self._with_mapping,
                    **self._compiler_kwargs,
                )

                if self._with_mapping:
                    # ccircuit, input_qubits, output_qubits = output
                    ccircuit, _, _ = output
                else:
                    ccircuit = output

                if self._justify:
                    ccircuit.compress()

                ccircuits.append(ccircuit)

        if to_qcal:
            ccircuits = self._to_qcal.transpile(ccircuits)
            # Remove the barrier accross entire register
            for circuit in ccircuits:
                circuit.popleft()
        else:
            ccircuits = CircuitSet(circuits=ccircuits)

        clear_output(wait=True)
        return ccircuits