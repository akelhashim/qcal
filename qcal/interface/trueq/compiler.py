"""Submodule for True-Q compiler based on qcal config and the basis gates.

NOTE: we do not use TYPE_CHECKING for trueq types because this might fail if
trueq is not installed when building docs.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Dict, List

import numpy as np
from scipy.linalg import block_diag

from qcal.config import Config
from qcal.gate.two_qubit import TWO_QUBIT_GATES

logger = logging.getLogger(__name__)


class TrueqCompiler:
    """True-Q compiler.

    The compiler can be created from a True-Q config file, or automatically
    from a qcal Config object.
    """
    __slots__ = ('_config', '_tq_config', '_compiler')

    def __init__(
            self,
            config: Config | trueq.Config | str,  # noqa: F821 # type: ignore
            passes: trueq.Compiler.passes = None  # noqa: F821 # type: ignore
        ) -> None:
        """Initialize a True-Q compiler.

        Args:
            config (Config | trueq.Config | str): qcal Config object or True-Q
                yaml.
            passes (trueq.Compiler.passes, optional): True-Q compiler passes.
                Defaults to None. If None, this will default to the
                HARDWARE_PASSES.
        """
        try:
            import trueq as tq
        except ImportError:
            logger.warning(' Unable to import trueq!')

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
    def compiler(self) -> trueq.Compiler:  # noqa: F821 # type: ignore
        """True-Q compiler.

        Returns:
            trueq.Compiler: True-Q Compiler object.
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
    def tq_config(self) -> trueq.Config:  # noqa: F821 # type: ignore
        """True-Q config.

        Returns:
            trueq.Config: True-Q Config object.
        """
        return self._tq_config

    @staticmethod
    def from_config(config: Config):
        """Generate a True-Q compiler from a qcal config.

        Args:
            config (Config): qcal Config object.

        Returns:
            trueq.Config: True-Q Config object.
        """
        import trueq as tq

        qubits = {f'({q},)': '()' for q in config.qubits}

        factories_2q = ''
        for gate in config.native_gates['set']:
            if gate in TWO_QUBIT_GATES.keys():
                factories_2q += f'\n        - {gate}:'
                factories_2q += '\n            Matrix:'
                for row in TWO_QUBIT_GATES[gate]((0, 1)).matrix:
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

    @staticmethod
    def from_generators(generators: Dict[int, Dict]):
        """Generates a compiler which decomposes single-qubit gates based on the
        true native gate.

        See: trueq.Gate.from_generators

        Args:
            generators (dict): Dictionary of gate generators for each qubit.

        Returns:
            trueq.Compiler: True-Q compiler object
        """
        try:
            import trueq as tq
        except ImportError:
            logger.warning(' Unable to import trueq!')

        class SqCycleDecomp(tq.compilation.base.NCyclePass):

            n_input_cycles = 1
            def __init__(self, native_cycle):
                self._natives = [native_cycle]

            def _apply_cycles(self, cycles):
                cycle = cycles[0]
                if len(cycle) == 0:
                    return cycles

                return tq.backend.local.decompose_sq_gates(
                    cycle, self._natives, True
                )

        native_gate_cycle = tq.Cycle({
            q: tq.Gate.from_generators(
                "X", generators[q]['X'],
                "Z", generators[q]['Z']
            ) for q in generators.keys()
        })

        factories = [
            tq.config.GateFactory(
                "X90",
                layers=[tq.math.FixedRotation(g),],
                involving = {label: ()}
            ) for label, g in native_gate_cycle.gates.items()
        ]
        factories.append(
            tq.config.GateFactory.from_hamiltonian("Rz", [["Z", "phi"]])
        )
        # TODO
        # factories.append(tq.config.GateFactory.from_matrix("cz", tq.Gate.cz))

        # Put everything together into a compiler
        compiler = tq.Compiler([
            tq.compilation.Native2Q(factories),
            tq.compilation.Merge(),
            SqCycleDecomp(native_gate_cycle),
            tq.compilation.Parallel(
                replacements=tq.compilation.NativeExact(factories)
            ),
            tq.compilation.RemoveEmptyCycle()
        ])

        return compiler

    def compile(
            self, circuits: trueq.Circuit | trueq.CircuitCollection  # noqa: F821 # type: ignore
        ) -> trueq.Circuit | trueq.CircuitCollection:  # noqa: F821 # type: ignore
        """Compile circuits using the compiler.

        Args:
            circuits (trueq.Circuit | trueq.CircuitCollection): True-Q circuit
                or CircuitCollection.

        Returns:
            trueq.Circuit | trueq.CircuitCollection: compiled True-Q circuit or
                CircuitCollection.
        """
        return self._compiler.compile(circuits)


class QuditCompiler:
    """True-Q qudit compiler."""
    try:
        import trueq as tq
    except ImportError:
        logger.warning(' Unable to import trueq!')
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
            self, circuits: trueq.Circuit | trueq.CircuitCollection  # noqa: F821 # type: ignore
        ) -> trueq.Circuit | trueq.CircuitCollection:  # noqa: F821 # type: ignore
        """Compile circuits using the compiler.

        Args:
            circuits (trueq.Circuit | trueq.CircuitCollection): True-Q circuit
                or CircuitCollection.

        Returns:
            trueq.Circuit | trueq.CircuitCollection: compiled True-Q circuit or
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

        def __init__(self, factories):
            """_summary_

            Args:
                factories (_type_): _description_
            """
            import trueq as tq

            def _IS_Z(x):
                return tq.math.proc_infidelity(
                            x, np.diag(np.exp([-1j*np.pi/180,0,0]))
                        ) < 1e-10

            def _IS_EFZ(x):
                return tq.math.proc_infidelity(
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
        def is_diag(gate: trueq.Gate) -> bool:  # noqa: F821 # type: ignore
            """Determine whether the given gate is unitary.

            Args:
                gate (trueq.Gate): True-Q gate

            Returns:
                bool: unitary or not.
            """
            return np.all(
                np.abs(gate.mat[np.triu_indices(gate.width, k=1)]) < 1e-12
            )

        def apply(
                self, labels: Sequence[int], operation: trueq.Gate  # noqa: F821 # type: ignore
            ) -> List:
            """Apply an operation to a given set of qudit labels.

            Args:
                labels (Sequence[int]): qudit labels.
                operation (trueq.Gate): True-Q gate.

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
