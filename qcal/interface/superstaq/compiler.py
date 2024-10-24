"""Submodule for True-Q compiler based on qcal config and the basis gates.

"""
from qcal.circuit import CircuitSet
from qcal.config import Config
from qcal.gate.two_qubit import two_qubit_gates

import logging
import numpy as np

from scipy.linalg import block_diag
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# __all__ = ('CirqCompiler', 'QiskitCompiler')


def gateset_from_config(config: Config) -> Dict:
    """Generate a gateset from the qcal config.

    Args:
        config (Config): qcal Config object.

    Returns:
        Dict: gateset.
    """
    gateset = {
        'X90' : [
            [q] for q in tuple(config.native_gates['single_qubit'].keys())
        ]
    }
    for gate in config.native_gates['set']:
        if gate in two_qubit_gates:
            gateset[gate] = [
                list(qp) for qp in config.native_gates['two_qubit'] if gate in
                config.native_gates['two_qubit'][qp]
            ]
    
    return gateset


class CirqCompiler:
    """Superstaq Cirq compiler."""
    try:
        import cirq_superstaq as css
        import cirq
    except ImportError:
        logger.warning(' Unable to import cirq_superstaq!')

    __slots__ = (
        '_service', '_config', '_gateset', '_gate_defs', '_compiler_output'
    )

    def __init__(self, 
            api_key: str, 
            config: Config, 
            gateset: Dict = None, 
            gate_defs: Dict = None     
        ) -> None:
        """Initialize a Cirq Superstaq compiler.

        Args:
            api_key (str): Superstaq API key.
            config (Config): qcal Config object.
            gateset (Dict): gateset for superstaq compiler. Defaults to None. If
                None, it is generated automatically from the native gates in the
                Config object.
            gate_defs (Dict): gate definitions for nonstandard multi-qubit gates 
                in the gateset. Defaults to None.
        """
        import cirq_superstaq as css
        self._service = css.Service(api_key=api_key)
        logger.info(f" Cirq Superstaq version: {css.__version__}")

        self._config = config
        self._gateset = (
            gateset_from_config(config) if gateset is None else gateset
        )
        self._gate_defs = gate_defs
        self._compiler_output = None

    @property
    def compiler(self) -> css.Service:
        """Cirq Superstaq Service.

        Returns:
            css.Service: service object.
        """
        return self.service
    
    @property
    def compiler_output(self) -> Any:
        """Output of compiler.

        Returns:
            Any: compiler output.
        """
        return self._compiler_output
    
    @property
    def config(self) -> Config:
        """qcal config.

        Returns:
            Config: qcal Config object.
        """
        return self._config
    
    @property
    def gateset(self) -> Dict:
        """Gateset for compiler.

        Returns:
            Dict: gateset.
        """
        return self._gateset
    
    @property
    def gate_defs(self) -> Dict:
        """Custom gate definitions.

        Returns:
            Dict: gate definitions.
        """
        return self._gate_defs
    
    @property
    def service(self) -> css.Service:
        """Cirq Superstaq Service.

        Returns:
            css.Service: service object.
        """
        return self._service
    
    def compile(
            self, circuits: cirq.Circuit | List[cirq.Circuit] | CircuitSet,
        ) -> CircuitSet:
        """Compile circuits using the compiler.

        Args:
            circuits (cirq.Circuit | List[cirq.Circuit] | CircuitSet): circuits
                to compile.

        Returns:
            CircuitSet: compiled circuits.
        """
        import cirq
        if isinstance(circuits, cirq.Circuit):
            circuits = [circuits]
        elif isinstance(circuits, CircuitSet):
            circuits = list(circuits.circuit)

        self._compiler_output = self._service.aqt_compile(
            circuits,
            gateset=self._gateset,
            gate_defs=self._gate_defs,
            aqt_configs={}
        )

        return CircuitSet(circuits=self._compiler_output.circuits)
    

class QiskitCompiler:
    """Superstaq Qiskit compiler."""
    try:
        import qiskit_superstaq as qss
        from qiskit import QuantumCircuit
    except ImportError:
        logger.warning(' Unable to import qiskit_superstaq!')

    __slots__ = (
        '_provider', '_backend', '_config', '_gateset', '_gate_defs', 
        '_compiler_output'
    )

    def __init__(self, 
            api_key: str, 
            config: Config, 
            gateset: Dict = None, 
            gate_defs: Dict = None     
        ) -> None:
        """Initialize a Qiskit Superstaq compiler.

        Args:
            api_key (str): Superstaq API key.
            config (Config): qcal Config object.
            gateset (Dict): gateset for superstaq compiler. Defaults to None. If
                None, it is generated automatically from the native gates in the
                Config object.
            gate_defs (Dict): gate definitions for nonstandard multi-qubit gates 
                in the gateset. Defaults to None.
        """
        import qiskit_superstaq as qss
        self._provider = qss.SuperstaqProvider(api_key=api_key)
        self._backend = self._provider.get_backend("aqt_new_qpu")
        logger.info(f" Qiskit Superstaq version: {qss.__version__}")

        self._config = config
        self._gateset = (
            gateset_from_config(config) if gateset is None else gateset
        )
        self._gate_defs = gate_defs
        self._compiler_output = None

    @property
    def compiler(self) -> qss.SuperstaqBackend:
        """Qiskit Superstaq backend.

        Returns:
            qss.SuperstaqBackend: backend object.
        """
        return self._backend
    
    @property
    def compiler_output(self) -> Any:
        """Output of compiler.

        Returns:
            Any: compiler output.
        """
        return self._compiler_output
    
    @property
    def config(self) -> Config:
        """qcal config.

        Returns:
            Config: qcal Config object.
        """
        return self._config
    
    @property
    def gateset(self) -> Dict:
        """Gateset for compiler.

        Returns:
            Dict: gateset.
        """
        return self._gateset
    
    @property
    def gate_defs(self) -> Dict:
        """Custom gate definitions.

        Returns:
            Dict: gate definitions.
        """
        return self._gate_defs
    
    @property
    def provider(self) -> qss.SuperstaqProvider:
        """Qiskit Superstaq Provider.

        Returns:
            qss.SuperstaqProvider: provider object.
        """
        return self._provider
    
    def compile(self, 
            circuits: QuantumCircuit | List[QuantumCircuit] | CircuitSet,
        ) -> CircuitSet:
        """Compile circuits using the compiler.

        Args:
            circuits (QuantumCircuit | List[QuantumCircuit] | CircuitSet): 
                circuits to compile.

        Returns:
            CircuitSet: compiled circuits.
        """
        import qiskit
        if isinstance(circuits, qiskit.QuantumCircuit):
            circuits = [circuits]
        elif isinstance(circuits, CircuitSet):
            circuits = list(circuits.circuit)

        self._compiler_output = self._backend.compile(
            circuits,
            gateset=self._gateset,
            gate_defs=self._gate_defs,
            aqt_configs={}
        )

        return CircuitSet(circuits=self._compiler_output.circuits)
