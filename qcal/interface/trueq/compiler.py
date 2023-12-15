"""Submodule for True-Q compiler based on qcal config and the basis gates.

"""
from qcal.config import Config
from qcal.gate.two_qubit import two_qubit_gates

import logging

logger = logging.getLogger(__name__)


class Compiler:
    """True-Q compiler.

    The compiler can be created from a True-Q config file, or automatically
    from a qcal Config object.
    """
    import trueq as tq

    __slots__ = ('_config', '_tq_config', '_compiler')

    def __init__(
            self, 
            config: Config | tq.Config | str, 
            passes: tq.Compiler.passes = None
        ) -> None:
        """Initialize a True-Q compiler.

        Args:
            config (Config | tq.Config | str): qcal Config object or True-Q yaml.
            passes (tq.Compiler.passes, optional): True-Q compiler passes. 
                Defaults to None. If None, 
        """
        import trueq as tq
        passes = (
            passes if passes is not None else tq.Compiler.HARDWARE_PASSES
        )

        if isinstance(config, Config):
            self._config = config
            self._tq_config = self.from_config(config)
        elif isinstance(config, tq.Config):
            self._config = None
            self._tq_config = config
        elif isinstance(config, str):
            self._config = None
            self._tq_config = tq.Config.from_yaml(config)
        
        self._compiler = tq.Compiler.from_config(
            self._tq_config, passes=passes
        )

    @property
    def compiler(self) -> tq.Compiler:
        """True-Q compiler.

        Returns:
            tq.Compiler: True-Q Compiler object.
        """
        return self._compiler

    @property
    def config(self) -> Config:
        """qcal config.

        Returns:
            Config: qcal Config object.
        """
        return self._config
    
    @property
    def tq_config(self) -> tq.Config:
        """True-Q config.

        Returns:
            tq.Config: True-Q Config object.
        """
        return self._tq_config

    @staticmethod
    def from_config(config: Config):
        """Generate a True-Q compiler from a qcal config.

        Args:
            config (Config): qcal Config object.

        Returns:
            tq.Config: True-Q Config object.
        """
        import trueq as tq

        qubits = {f'({q},)': '()' for q in config.qubits}

        factories_2q = ''
        for gate in config.basis_gates['set']:
            if gate in two_qubit_gates.keys():
                factories_2q += f'\n        - {gate}:'
                factories_2q += '\n            Matrix:'
                for row in two_qubit_gates[gate]().matrix:
                    factories_2q += f'\n            - {list(row)}'
                factories_2q += '\n            Involving:'
            for qubit_pair in config.basis_gates['two_qubit'].keys():
                if gate in config.basis_gates['two_qubit'][qubit_pair]:
                    factories_2q += f'\n                {qubit_pair}: ()'

        _config = f"""
        Dimension: 2
        Mode: ZXZXZ
        Gates:
        - Rz:
            Hamiltonian:
                - ['Z', phi]
            Involving: {qubits}
        - X90(phi=90):
            Hamiltonian:
                - ['X', phi]
            Involving: {qubits}
        """
        _config += factories_2q

        return tq.Config.from_yaml(_config)
    
    def compile(
            self, circuits: tq.Circuit | tq.CircuitCollection
        ) -> tq.Circuit | tq.CircuitCollection:
        """Compile circuits using the compiler.

        Args:
            circuits (tq.Circuit | tq.CircuitCollection): True-Q circuit or
                CircuitCollection.

        Returns:
            tq.Circuit | tq.CircuitCollection: compiled True-Q circuit or
                CircuitCollection.
        """
        return self._compiler.compile(circuits)

        