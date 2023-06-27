"""Submodule for constructing a circuit compiler.
"""
from qcal.config import Config

import bqskit
from bqskit import compile
from bqskit import MachineModel
from bqskit.ir import Circuit as BqCircuit
from bqskit.ir.gates import BarrierPlaceholder as BqBarrier
from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.ir.gates import (
    CZGate, XGate, RZGate, SXGate, #ZGate, SGate, SdgGate, TGate, TdgGate
)

import logging
logger = logging.getLogger(__name__)

class Compiler:

    def __init__(self, config: Config) -> None:
        pass

    def compile(self, circuits):
        pass

DEFAULT_COMPILER = Compiler(None)