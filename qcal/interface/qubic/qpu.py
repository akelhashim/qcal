"""Custom QPU submodule for QubiC

"""
# from qcal.circuit import CircuitSet
from .post_process import post_process
from .transpiler import Transpiler
from qcal.config import Config
from qcal.qpu.qpu import QPU

import logging
import os
# import sys

from typing import Any, List

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
                delay_per_shot:      float | None = 0,
                reload_cmd:          bool = True,
                reload_freq:         bool = True,
                reload_env:          bool = True,
                zero_between_reload: bool = True,
                gmm_manager               = None,
                rpc_ip_address:      str = '192.168.1.247'
        ) -> None:
        """Initialize a instance of a QPU for QubiC.

        Args:
            config (Config): qcal config object.
            compiler (Any | Compiler | None, optional): a custom compiler to
                compile the experimental circuits. Defaults to None.
            transpiler (Any | None, optional): a custom transpiler to 
                transpile the experimental circuits. Defaults to None.
            n_shots (int, optional): number of measurements per circuit. 
                Defaults to 1024.
            n_batches (int, optional): number of batches of measurements. 
                Defaults to 1.
            n_circs_per_seq (int, optional): maximum number of circuits that
                can be measured per sequence. Defaults to 1.
            n_levels (int, optional): number of energy levels to be measured. 
                Defaults to 2. If n_levels = 3, this assumes that the
                measurement supports qutrit classification.
            n_reads_per_shot (int | None, optional): number of reads per shot
                per circuit. Defaults to None. If None, this will be computed
                from the number of active resets and whether or not heralding
                is used.
            delay_per_shot (float | None, optional): wait time between 
                measuring and offloading the data. Defaults to 0. If 0, this
                will be computed automatically. If None, this is computed by
                the length of the full sequence plus a 1 us buffer.
            reload_cmd (bool, optional): reload pulse command buffer for each
                batched circuit. Defaults to True.
            reload_freq (bool, optional): reload pulse frequencies when loading
                a new circuit. Defaults to True. If the pulse frequencies do
                not change for all circuits in a batch, this can be set to
                False to save execution time.
            reload_env (bool, optional): reload pulse envelopes when loading a
                new circuit. Defaults to True. If the pulse envelopes do not
                change for all circuits in a batch, this can be set to False
                to save execution time.
            zero_between_reload (bool, optional): zero out the pulse command
                buffers before loading a new circuit. Defaults to True. If all
                of the circuits in a batch use the same qubits, this can be set
                to False to save execution time.
            gmm_manager (GMMManager, optional): QubiC GMMManager object.
                Defaults to None. If None, this is loaded from a previously 
                saved manager object: 'gmm_manager.pkl'.
            rpc_ip_address (str, optional): IP address for remote measurements.
                Defaults to '192.168.1.247'.
        """
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
        from qubic import rpc_client, job_manager
        from qubic.rfsoc.hwconfig import FPGAConfig, load_channel_configs
        from qubic.state_disc import GMMManager
        from qubitconfig.qchip import QChip

        self._n_reads_per_shot = (calculate_n_reads(config) 
            if n_reads_per_shot is None else n_reads_per_shot
        )
        self._delay_per_shot = delay_per_shot
        self._reload_cmd = reload_cmd
        self._reload_freq = reload_freq
        self._reload_env = reload_env
        self._zero_between_reload = zero_between_reload
        
        self._gmm_manager = gmm_manager if gmm_manager is not None else (
            os.path.join(os.path.dirname(__file__), 'gmm_manager.pkl')
        )
        self._fpga_config = FPGAConfig(
            **{'fpga_clk_period': 2.e-9,
               'alu_instr_clks': 5,
               'jump_cond_clks': 5, 
               'jump_fproc_clks': 5, 
               'pulse_regwrite_clks': 3}
        )
        self._channel_config = load_channel_configs(
            os.path.join(os.path.dirname(__file__), 'channel_config.json')
        )
        self._qchip = QChip(
            os.path.join(os.path.dirname(__file__), 'qubic_cfg.json')
        )
        self._runner = rpc_client.CircuitRunnerClient(ip=rpc_ip_address)
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
        
    def generate_sequence(self) -> None:
        """Generate a QubiC sequence.

        This occurs in two steps:
        1) Generate the compiled program, which takes into accoun the FPGA
            config and the qchip layout.
        2) Generate the raw ASM code. This is what we are calling the
            "sequence."

        Args:
            circuits (List): TODO
        """
        from qubic.toolchain import run_compile_stage, run_assemble_stage
        self._compiled_program = run_compile_stage(
            self._exp_circuits, self._fpga_config, self._qchip
        )
        self._sequence = run_assemble_stage(
            self._compiled_program, self._channel_config
        )

    def acquire(self) -> None:
        """Measure all circuits."""
        if self._delay_per_shot is None:
            self._delay_per_shot = (
                calculate_delay_per_shot(
                    self._config,
                    self._compiled_program,
                    self._channel_config
                )
            )
        
        measurement = self._jobman.build_and_run_circuits(
            self._sequence, 
            self._n_shots, 
            ['s11', 'shots', 'counts'], 
            fit_gmm=False,
            reads_per_shot=self._n_reads_per_shot,
            delay_per_shot=self._delay_per_shot,
            reload_cmd=self._reload_cmd,
            reload_freq=self._reload_freq,
            reload_env=self._reload_env,
            zero_between_reload=self._zero_between_reload
        )
        self._measurements.append(measurement)

    def process(self) -> None:
        """Process the measurement data."""
        post_process(self._config, self._measurements, self._circuits)

        if not self._compiled_circuits.is_empty:
            post_process(
                self._config, self._measurements, self._compiled_circuits
            )

        if not self._transpiled_circuits.is_empty:
            post_process(
                self._config, self._measurements, self._transpiled_circuits
            )
