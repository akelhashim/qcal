"""Custom QPU submodule for QubiC

"""
from qcal.circuit import CircuitSet
from qcal.config import Config
from .qubic import Transpiler
from qcal.qpu.qpu import QPU

import logging
import sys

from collections.abc import Iterable
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

libraries = [
    ('qubic', 'qbc'),
    ('qubitconfig', 'qc')
]
for library, abbrv in libraries:
    try:
        lib = __import__(library)
    except ImportError:
        logger.warning(sys.exc_info())  
    else:
        globals()[abbrv] = lib  


__all__ = ('QubicQPU')


class QubicQPU(QPU):
    """Qubic Quantum Processing Unit.

    QubicQPU inherents from the main QPU class, but overwrites certain methods.

    Args:
        QPU: main QPU class.
    """

    def __init__(
                self,
                config:          Config,
                compiler:        Any | None = None,
                transpiler:      Any | None = Transpiler,
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2
        ) -> None:
        super().__init__(
            config=config,
            compiler=compiler,
            transpiler=transpiler, 
            n_shots=n_shots, 
            n_batches=n_batches,
            n_circs_per_seq=n_circs_per_seq,
            n_levels=n_levels
        )

        # TODO: is this the right way?
        # fpga_config, qchip, channel_config = cfg.load_configs(qchipname)
        self._fpga_config = qbc.rfsoc.hwconfig.FPGAConfig(
            **{'fpga_clk_period': 2.e-9,
               'alu_instr_clks': 5,
               'jump_cond_clks': 5, 
               'jump_fproc_clks': 5, 
               'pulse_regwrite_clks': 3}
        )
        self._channel_config = qbc.rfsoc.hwconfig.load_channel_configs(
            'channel_config.json'
        )
        # TODO: is this just the config?
        self._qchip = qc.qchip.QChip('qubic_cfg.json')
        self._runner = qbc.rpc_client.CircuitRunnerClient(ip='192.168.1.247')
        self._jobman = qbc.job_manager.JobManager(
            self._fpga_config, self._qchip, self._channel_config, self._runner
        )
        self._compiled_program = None

        # TODO: overwrite important config parameters
        
    def generate_sequence(self, circuits: List[Dict]) -> None:
        """Generate a QubiC sequence.

        This occurs in two steps:
        1) Generate the compiled program, which takes into accoun the FPGA
            config and the qchip layout.
        2) Generate the raw ASM code. This is what we are calling the
            "sequence."

        Args:
            circuits (List): _description_
        """
        self._compiled_program = qbc.toolchain.run_compile_stage(
            circuits, self._fpga_config, self._qchip
        )
        self._sequence = qbc.toolchain.run_assemble_stage(
            self._compiled_program, self._channel_config
        )

    def acquire(self) -> None:
        """Measure all circuits."""
        self._runner.load_circuit(self._sequence)
        self._measurement = self._runner.run_circuit(self._n_shots, 4) # What is 4?
        # ([cond_circuit], 10000, ['s11', 'shots'], reads_per_shot=2, fit_gmm=True, delay_per_shot=600.e-6)

    def process(self) -> None:
        pass

        # post-processing for ESP (readout and herald)
        # post-processing for heralding