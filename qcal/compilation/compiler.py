"""Submodule for handling compilation of qcal circuits.

Provides :class:`Compiler`, the default qcal compiler, which compiles
qcal circuits to the native gate set of a :class:`~qcal.config.Config`
using BQSKit (see :class:`~qcal.interface.bqskit.compiler.BQSKitCompiler`).
Single-qubit gates are compiled to the ZXZXZ form (Rz-X90-Rz-X90-Rz).

Example:
    .. code-block:: python

        from qcal.circuit import Circuit, Cycle
        from qcal.compilation import Compiler
        from qcal.config import Config
        from qcal.gates.single_qubit import H
        from qcal.gates.two_qubit import CNOT

        config = Config('examples/config/config.yaml')
        compiler = Compiler(config)

        circuit = Circuit([Cycle({H(0)}), Cycle({CNOT(0, 1)})])
        compiled = compiler.compile(circuit)  # CircuitSet
        compiled_circuit = compiled[0]  # native-gate qcal Circuit
"""
import logging

from qcal.config import Config
from qcal.interface.bqskit.compiler import BQSKitCompiler

logger = logging.getLogger(__name__)


class Compiler(BQSKitCompiler):
    """Default compiler for qcal circuits.

    This compiler is a wrapper around the BQSKit compiler
    (:class:`~qcal.interface.bqskit.compiler.BQSKitCompiler`). It is
    created from a qcal :class:`~qcal.config.Config` object.

    Example:
        .. code-block:: python

            from qcal.circuit import Circuit, Cycle
            from qcal.compilation import Compiler
            from qcal.config import Config
            from qcal.gates.single_qubit import H

            config = Config('examples/config/config.yaml')
            compiler = Compiler(config)
            compiled = compiler.compile(Circuit([Cycle({H(0)})]))
    """

    def __init__(self, config: Config, **kwargs):
        """Initialize a qcal compiler.

        Args:
            config (Config): qcal Config object.
            **kwargs: additional keyword arguments forwarded to
                :class:`~qcal.interface.bqskit.compiler.BQSKitCompiler`
                (e.g. ``optimization_level``, ``seed``).
        """
        super().__init__(config=config, **kwargs)
