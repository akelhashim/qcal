"""Custom QPU submodule for QubiC

"""
import qcal.settings as settings

# from qcal.circuit import CircuitSet
from .post_process import post_process
from .transpiler import Transpiler
from .utils import calculate_n_reads
from qcal.config import Config
from qcal.managers.classification_manager import ClassificationManager
from qcal.qpu.qpu import QPU

import logging
import os
# import sys

from typing import Any, List

logger = logging.getLogger(__name__)


__all__ = ('QubicQPU')


class QubicQPU(QPU):
    """Qubic Quantum Processing Unit.

    QubicQPU inherents from the main QPU class, but overwrites certain methods.

    Args:
        QPU: main QPU class.
    """
    from qubic.state_disc import GMMManager

    def __init__(
                self,
                config:              Config,
                compiler:            Any | None = None,
                transpiler:          Any | None = None,
                classifier:          ClassificationManager = None,
                n_shots:             int = 1024, 
                n_batches:           int = 1, 
                n_circs_per_seq:     int = 1,
                n_levels:            int = 2,
                n_reads_per_shot:    int | dict | None = None,
                raster_circuits:     bool = False,
                outputs:             List[str] = ['s11'],
                hardware_vz_qubits:  List[str] = [],
                measure_qubits:      List[str] | None = None,
                reload_pulse:        bool = True,
                reload_cmd:          bool = True,
                reload_freq:         bool = True,
                reload_env:          bool = True,
                zero_between_reload: bool = True,
                gmm_manager:         GMMManager = None,
                rpc_ip_address:      str = '192.168.1.122',
                port:                int = 9095
        ) -> None:
        """Initialize a instance of a QPU for QubiC.

        Args:
            config (Config): qcal config object.
            compiler (Any | Compiler | None, optional): a custom compiler to
                compile the experimental circuits. Defaults to None.
            transpiler (Any | None, optional): a custom transpiler to 
                transpile the experimental circuits. Defaults to None.
            classifier (ClassificationManager, optional): manager used for 
                classifying raw data. Defaults to None.
            n_shots (int, optional): number of measurements per circuit. 
                Defaults to 1024.
            n_batches (int, optional): number of batches of measurements. 
                Defaults to 1.
            n_circs_per_seq (int, optional): maximum number of circuits that
                can be measured per sequence. Defaults to 1.
            n_levels (int, optional): number of energy levels to be measured. 
                Defaults to 2. If n_levels = 3, this assumes that the
                measurement supports qutrit classification.
            n_reads_per_shot (int | dict | None, optional): number of reads per 
                shot per circuit. Defaults to None. If None, this will be 
                computed from the number of reads per channel in the compiled
                program.
            raster_circuits (bool, optional): whether to raster through all
                circuits in a batch during measurement. Defaults to False. By
                default, all circuits in a batch will be measured n_shots times
                one by one. If True, all circuits in a batch will be measured
                back-to-back one shot at a time. This can help average out the 
                effects of drift on the timescale of a measurement.
            outputs (List[str]): what output data is desired for each
                measurement. Defaults to ['s11', 'shots', 'counts']. 's11'
                is to the integrated IQ data; 'shots' is the classified data
                for each read; and 'counts' is the accumulated statistics
                for each read.
            hardware_vz_qubits (List[str], optional): list of qubit labels
                specifying for which qubits should the virtualz gates be done
                on hardware (as opposed to software). Defaults to None. This is
                necessary if doing conditional phase shifts using mid-
                circuit measurements. Example: ```measure_qubits = ['Q0', 'Q1', 
                'Q3']```.
            measure_qubits (List[str] | None, optional): list of qubit labels 
                for post-processing measurements. Defaults to None. This will
                overwrite the measurement qubits listed in the measurement
                objects. Example: ```measure_qubits = ['Q0', 'Q1', 'Q3']```.
            reload_pulse (bool): reloads the stored pulses when compiling each
                circuit. Defaults to True.
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
            rpc_ip_address (str, optional): IP address for RPC server.
                Defaults to '192.168.1.25'.
            port (int, option): port for RPC server. Defaults to 9096.
        """
        QPU.__init__(self,
            config=config,
            compiler=compiler,
            transpiler=transpiler,
            classifier=classifier,
            n_shots=n_shots, 
            n_batches=n_batches,
            n_circs_per_seq=n_circs_per_seq,
            n_levels=n_levels,
            raster_circuits=raster_circuits
        )
        from qubic import rpc_client, job_manager
        from qubic.rfsoc.hwconfig import FPGAConfig, load_channel_configs
        from qubic.state_disc import GMMManager
        from qubitconfig.qchip import QChip

        # self._n_reads_per_shot = (calculate_n_reads(config) 
        #     if n_reads_per_shot is None else n_reads_per_shot
        # )
        self._n_reads_per_shot = n_reads_per_shot
        self._outputs = outputs
        self._measure_qubits = measure_qubits
        self._reload_cmd = reload_cmd
        self._reload_freq = reload_freq
        self._reload_env = reload_env
        self._zero_between_reload = zero_between_reload
        
        self._qubic_transpiler = Transpiler(
            config, 
            reload_pulse=reload_pulse,
            hardware_vz_qubits=hardware_vz_qubits
        )
        self._gmm_manager = gmm_manager if gmm_manager is not None else (
            os.path.join(settings.Settings.config_path, 'gmm_manager.pkl')
        )
        self._fpga_config = FPGAConfig()
        self._channel_config = load_channel_configs(
            os.path.join(settings.Settings.config_path, 'channel_config.json')
        )
        self._qchip = QChip(
            os.path.join(settings.Settings.config_path, 'qubic_cfg.json')
        )
        self._runner = rpc_client.CircuitRunnerClient(
            ip=rpc_ip_address, port=port
        )
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
                self._config[f'single_qubit/{q}/GE/freq']
            )
            self._qchip.qubits[f'Q{q}'].freq_ef = (
                self._config[f'single_qubit/{q}/EF/freq']
            )
            self._qchip.qubits[f'Q{q}'].readfreq = (
                self._config[f'readout/{q}/freq']
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
        """
        from qubic.toolchain import run_compile_stage, run_assemble_stage

        self._exp_circuits = self._qubic_transpiler.transpile(
            self._exp_circuits
        )

        if self._raster_circuits:
            rastered_circuit = []
            for circuit in self._exp_circuits:
                rastered_circuit.extend(circuit)
            self._exp_circuits = [rastered_circuit]

        self._compiled_program = run_compile_stage(
            self._exp_circuits, self._fpga_config, self._qchip
        )
        self._sequence = run_assemble_stage(
            self._compiled_program, self._channel_config
        )

    def acquire(self) -> None:
        """Measure all circuits."""
        if self._n_reads_per_shot is None:
            self._n_reads_per_shot = calculate_n_reads(
                self._config, self._compiled_program[0]
            )

        measurement = self._jobman.build_and_run_circuits(
            self._sequence, 
            self._n_shots, 
            self._outputs, 
            fit_gmm=False,
            reads_per_shot=self._n_reads_per_shot,
            reload_cmd=self._reload_cmd,
            reload_freq=self._reload_freq,
            reload_env=self._reload_env,
            zero_between_reload=self._zero_between_reload
        )
        self._measurements.append(measurement)

    def process(self) -> None:
        """Process the measurement data."""
        post_process(
            self._config, 
            self._measurements,
            self._circuits,
            measure_qubits=self._measure_qubits,
            n_reads_per_shot=self._n_reads_per_shot,
            classifier=self._classifier, 
            raster_circuits=self._raster_circuits
        )

        if len(self._compiled_circuits) > 1:
            post_process(
                self._config, 
                self._measurements,
                self._compiled_circuits,
                measure_qubits=self._measure_qubits,
                n_reads_per_shot=self._n_reads_per_shot,
                classifier=self._classifier, 
                raster_circuits=self._raster_circuits
            )

        if len(self._transpiled_circuits) > 1:
            post_process(
                self._config,
                self._measurements,
                self._transpiled_circuits,
                measure_qubits=self._measure_qubits,
                n_reads_per_shot=self._n_reads_per_shot,
                classifier=self._classifier, 
                raster_circuits=self._raster_circuits
            )
