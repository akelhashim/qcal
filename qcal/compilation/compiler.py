"""Submodule for handling compilation of qcal circuits.

"""
import logging

from qcal.config import Config
from qcal.interface.bqskit.compiler import BQSKitCompiler

logger = logging.getLogger(__name__)


class Compiler(BQSKitCompiler):
    """qcal compiler.

    This compiler is a wrapper around the BQSKit compiler.
    It is created from a qcal Config object.
    """

    def __init__(self, config: Config, **kwargs):
        """Initialize a qcal compiler.

        Args:
            config (Config): qcal Config object.
        """
        super().__init__(config=config, **kwargs)
