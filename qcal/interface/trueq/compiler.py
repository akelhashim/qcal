"""Submodule for True-Q compiler based on qcal config and the basis gates.

"""
from qcal.config import Config
from qcal.gate.two_qubit import two_qubit_gates

import logging
import numpy as np

from scipy.linalg import block_diag
from typing import List, Tuple

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
        for gate in config.native_gates['set']:
            if gate in two_qubit_gates.keys():
                factories_2q += f'\n        - {gate}:'
                factories_2q += '\n            Matrix:'
                for row in two_qubit_gates[gate]((0, 1)).matrix:
                    factories_2q += f'\n            - {list(row)}'
                factories_2q += '\n            Involving:'
            for qubit_pair in config.native_gates['two_qubit'].keys():
                if gate in config.native_gates['two_qubit'][qubit_pair]:
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

        
class QuditCompiler:
    """True-Q qudit compiler."""
    import trueq as tq

    # __slots__ = ('_config', '_tq_config', '_compiler')

    def __init__(self) -> None:
        """Initialize a True-Q qudit compiler."""
        import trueq as tq

        factories = [
            tq.config.GateFactory.from_matrix(
                "GEX90", block_diag(tq.Gate.rx(90), 1)
            ),
            tq.config.GateFactory.from_matrix(
                "EFX90", block_diag(1, tq.Gate.rx(90))
            ),
            tq.config.GateFactory.from_matrix(
                "GERz", [["exp(-1j * phi * pi/180)",0,0], [0,1,0], [0,0,1]]
            ),
            tq.config.GateFactory.from_matrix(
                "EFRz", [[1,0,0], [0,1,0], [0,0,"exp(1j * phi * pi/180)"]]
            ),
            # tq.config.GateFactory(
            #     "Rz", 
            #     [tq.math.Rotation(np.pi / 360 * np.diag([1, -1, 0]), "phi")], 
            #     sys_dim=3
            # ),
            # tq.config.GateFactory(
            #     "EFRz", 
            #     [tq.math.Rotation(np.pi / 360 * np.diag([0, 1, -1]), "phi")], 
            #     sys_dim=3
            # ),
        ]

        self._compiler = tq.Compiler([
            tq.compilation.OneQuditDecomp(factories),
            tq.compilation.Parallel(self.DecompEFZ(factories)),
        ])

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

    class DecompEFZ(tq.compilation.base.OperationReplacement):
        r"""
        Decomposes diagonal qutrit unitaries. Assumes that one of the factories 
        produces gates of the form diag(e^{-i phi}, 1, 1) and that another 
        produces gates of the form diag(1, 1, e^{-i phi}). phi _must_ be in 
        radians.
        """
        import trueq as tq

        def __init__(self, factories):
            """_summary_

            Args:
                factories (_type_): _description_
            """
            import trueq as tq

            _IS_Z = lambda x: tq.math.proc_infidelity(
                x, np.diag(np.exp([-1j*np.pi/180,0,0]))
            ) < 1e-10
            _IS_EFZ = lambda x: tq.math.proc_infidelity(
                x, np.diag(np.exp([0,0,1j*np.pi/180]))
            ) < 1e-10

            self.z = None
            self.efz = None
            for factory in factories:
                if factory.n_params == 1 and _IS_Z(factory(1).mat):
                    self.z = factory

                if factory.n_params == 1 and _IS_EFZ(factory(1).mat):
                    self.efz = factory

            assert self.z is not None and self.efz is not None

        @staticmethod
        def is_diag(gate: tq.Gate) -> bool:
            """Determine whether the given gate is unitary.

            Args:
                gate (tq.Gate): True-Q gate

            Returns:
                bool: unitary or not.
            """
            return np.all(
                np.abs(gate.mat[np.triu_indices(gate.width, k=1)]) < 1e-12
            )

        def apply(
                self, labels: List[int] | Tuple[int], operation: tq.Gate
            ) -> List:
            """Apply an operation to a given set of qudit labels.

            Args:
                labels (List[int] | Tuple[int]): qudit labels.
                operation (tq.Gate): True-Q gate.

            Raises:
                RuntimeError: incompatible operation.

            Returns:
                List: list of cycles.
            """
            import trueq as tq

            if (len(labels) != 1 or not isinstance(operation, tq.Gate) or 
                not self.is_diag(operation)
                ):
                return [{labels: operation}]

            diag = operation.mat[1,1].conj() * np.diag(operation.mat)
            return [
                {labels: self.z(-180 / np.pi * np.angle(diag[0]))}, 
                {labels: self.efz(180 / np.pi * np.angle(diag[2]))}
            ]