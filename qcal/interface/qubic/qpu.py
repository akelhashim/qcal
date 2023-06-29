"""Custom QPU submodule for QubiC

"""
from qcal.circuit import CircuitSet
from qcal.config import Config
from .transpiler import Transpiler
from qcal.qpu.qpu import QPU
from qcal.utils import save

import logging
import os
import sys

from collections.abc import Iterable
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# libraries = [
#     ('qubic', 'qbc'),
#     ('qubitconfig', 'qcfg')
# ]
# for library, abbrv in libraries:
#     try:
#         lib = __import__(library)
#     except ImportError:
#         logger.warning(sys.exc_info())  
#     else:
#         globals()[abbrv] = lib  


__all__ = ('QubicQPU')


def calculate_n_reads(config: Config) -> int:
    """Calculate the number of reads from a config.

    The number of reads will depend on:
    1) number of active resets
    2) heralding
    3) readout at the end of the circuit

    This function assumes that there is no mid-circuit measurement, and that
    there is always a readout at the end of a circuit.

    Args:
        config (Config): config object.

    Returns:
        int: number of reads.
    """
    n_reads = 1  # Measurement at the end of the circuit
    
    if config.parameters['reset']['active']['enable']:
        n_reads += config.parameters['reset']['active']['n_resets']
    
    if config.parameters['readout']['herald']:
        n_reads += 1

    return n_reads


def calculate_delay_per_shot(
        config: Config, compiled_program, channel_config,
    ) -> float:
    """Calculate the delay per shot for offloading measurement data.

    This is the maximum sequence length for the handshake from the compiled
    program. This function assumes that all sequences are the same length. If
    this is not true, then the circuits should be batched one-by-one. This
    function also assumes that the last sequence element is a readout pulse.

    Args:
        config (Config): qcal config object.
        compiled_program: QubiC compiled program object.
        channel_config: QubiC channel_config object.

    Returns:
        float: delay per shot in seconds.
    """
    delay = 1.e-6  # Built-in buffer
    delay += (     # Start time of last sequence element
        list(compiled_program[-1].program.values())[-1][-2]['start_time'] / 
        channel_config['fpga_clk_freq']
    )
    delay += config.parameters['readout']['length']  # Readout length
    return delay


class QubicQPU(QPU):
    """Qubic Quantum Processing Unit.

    QubicQPU inherents from the main QPU class, but overwrites certain methods.

    Args:
        QPU: main QPU class.
    """

    def __init__(
                self,
                config:              Config,
                compiler:            Any | None = None,
                transpiler:          Any | None = Transpiler,
                n_shots:             int = 1024, 
                n_batches:           int = 1, 
                n_circs_per_seq:     int = 1,
                n_levels:            int = 2,
                n_reads_per_shot:    int | None = None,
                delay_per_shot:      float | None = None,
                reload_cmd:          bool = True,
                reload_freq:         bool = True,
                reload_env:          bool = True,
                zero_between_reload: bool = True,
                gmm_manager               = None,
                rpc_ip_address:      str = '192.168.1.247'
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

        # import qubic
        # import qubitconfig
        from qubitconfig.qchip import QChip
        from qubic.rfsoc.hwconfig import FPGAConfig, load_channel_configs
        from qubic import rpc_client, job_manager

        self._n_reads_per_shot = (calculate_n_reads(config) 
            if n_reads_per_shot is None else n_reads_per_shot
        )
        self._delay_per_shot = delay_per_shot
        self._reload_cmd = reload_cmd
        self._reload_freq = reload_freq
        self._reload_env = reload_env
        self._zero_between_reload = zero_between_reload
        self._gmm_manager = gmm_manager

        if gmm_manager == None:
            logger.warning(
                'No gmm_manager provided! ' 
                'Measurements will fit to their own gmm.'
            )

        # self._fpga_config = qubic.rfsoc.hwconfig.FPGAConfig(
        self._fpga_config = FPGAConfig(
            **{'fpga_clk_period': 2.e-9,
               'alu_instr_clks': 5,
               'jump_cond_clks': 5, 
               'jump_fproc_clks': 5, 
               'pulse_regwrite_clks': 3}
        )
        # self._channel_config = qubic.rfsoc.hwconfig.load_channel_configs(
        self._channel_config = load_channel_configs(
            os.path.join(os.path.dirname(__file__), 'channel_config.json')
        )
        # self._qchip = qubitconfig.qchip.QChip(
        self._qchip = QChip(
            os.path.join(os.path.dirname(__file__), 'qubic_cfg.json')
        )
        # self._runner = qubic.rpc_client.CircuitRunnerClient(ip='192.168.1.247')
        self._runner = rpc_client.CircuitRunnerClient(ip=rpc_ip_address)
        # self._jobman = qubic.job_manager.JobManager(
        self._jobman = job_manager.JobManager(
            self._fpga_config,
            self._channel_config,
            self._runner,
            self._qchip,
            gmm_manager=self._gmm_manager
        )
        self._compiled_program = None

        # Overwrite qubit and readout frequencies:
        for q in self._config.qubits:
            self._qchip.qubits[f'Q{q}'].freq = (
                self._config.single_qubit[0].GE.freq[0]
            )
            self._qchip.qubits[f'Q{q}'].freq_ef = (
                self._config.single_qubit[0].EF.freq[0]
            )
            self._qchip.qubits[f'Q{q}'].readfreq = (
                self._config.single_qubit[0].GE.freq[0]
            )

    @property
    def fpga_config(self):
        """QubiC FPGA config object."""
        return self._fpga_config
    
    @property
    def channel_config(self):
        """QubiC channel config object."""
        return self._channel_config

    @property
    def qchip(self):
        """QubiC quantum chip object."""
        return self._qchip
    
    @property
    def runner(self):
        """QubiC runner object."""
        return self._runner
    
    @property
    def jobman(self):
        """QubiC job_manager object."""
        return self._jobman
    
    @property
    def compiled_program(self):
        """QubiC compiled_program object."""
        return self._compiled_program
        
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
        from qubic.toolchain import run_compile_stage, run_assemble_stage
        self._compiled_program = run_compile_stage(
            circuits, self._fpga_config, self._qchip
        )
        self._sequence = run_assemble_stage(
            self._compiled_program, self._channel_config
        )

    def acquire(self) -> None:
        """Measure all circuits."""
        if self._delay_per_shot == None:
            self._delay_per_shot = (
                calculate_delay_per_shot(
                    self._config,
                    self._compiled_program,
                    self._channel_config
                )
            )
        
        self._measurement = self._jobman.build_and_run_circuits(
            self._sequence, 
            self._n_shots, 
            ['s11', 'shots', 'counts'], 
            fit_gmm=False if self._gmm_manager is not None else True,
            reads_per_shot=self._n_reads_per_shot,
            delay_per_shot=self._delay_per_shot,
            reload_cmd=self._reload_cmd,
            reload_freq=self._reload_freq,
            reload_env=self._reload_env,
            zero_between_reload=self._zero_between_reload
        )

    def process(self) -> None:
        pass

        # post-processing for ESP (readout and herald)
        # post-processing for heralding

    def save(self) -> None:
        """Save all circuits and data."""
        # save(self._measurement['s11'], 'iq_data')
        # save(self._measurement['s11'], 'iq_data')
        # save(self._measurement['s11'], 'iq_data')
        pass

    def run(self,
            circuits:  Any | List[Any],
            n_shots:   int | None = None,
            n_batches: int | None = None
        ) -> None:
        """Run all experimental methods.

        Args:
            circuits (Union[Any, List[Any]]): circuits to measure.
            n_shots (Union[int, None], optional): number of shots per batch. 
                Defaults to None.
            n_batches (Union[int, None], optional): number of batches of shots.
                Defaults to None.
        """
        super().run(circuits, n_shots, n_batches)