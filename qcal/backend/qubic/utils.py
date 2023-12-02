"""Helper functions for QubiC

"""
from qcal.config import Config
from qcal.units import ns, us

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_n_reads(config: Config) -> int:
    """Calculate the number of reads per circuit from a config.

    The number of reads will depend on:
    1) number of active resets
    2) heralding
    3) readout at the end of the circuit

    This function assumes that there is no mid-circuit measurement, and that
    there is always a readout at the end of a circuit.

    Args:
        config (Config): config object.

    Returns:
        int: number of reads per circuit.
    """
    n_reads = 1  # Measurement at the end of the circuit
    
    if config.parameters['reset']['active']['enable']:
        n_reads += config.parameters['reset']['active']['n_resets']
    
    if config.parameters['readout']['herald']:
        n_reads += 1

    return n_reads


def qubic_sequence(compiled_program) -> pd.DataFrame:
    """Write a qubic sequence from a compiled_program.

    Args:
        compiled_program (CompiledProgram): QubiC compiled program.

    Returns:
        pd.DataFrame: QubiC sequence.
    """
    from distproc.compiler import CompiledProgram
    assert isinstance(compiled_program, CompiledProgram)

    df = pd.DataFrame()
    channels = list(compiled_program.program.keys())
    for channel in channels:
        qubit = channel[0].split('.')[0]
        tags = []
        times = []
        for pulse in compiled_program.program[channel]:
            if 'tag' in pulse.keys():
                tags.append(pulse['tag'])
                times.append(pulse['start_time'])
        times = np.array(times) #- times[0]
        times = np.around(times* 2 * ns / us, 3)
        df = df.join(
            pd.DataFrame({qubit: tags}, index=times),
            how='outer'
        )

    df.index.name = '[us]'
    return df.fillna('')