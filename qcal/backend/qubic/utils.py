"""Helper functions for QubiC

"""
from qcal.config import Config
from qcal.units import ns, us

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_n_reads(config: Config, compiled_program) -> dict:
    """Calculate the number of reads per qubit per circuit.

    The number of reads will depend on:
    1) number of active resets
    2) heralding
    3) number of mid-circuit measurements
    3) readout at the end of the circuit

    Args:
        config (Config): qcal config object.
        compiled_program (distproc.compiler.CompiledProgram): QubiC compiled
            program object.

    Returns:
        dict: number of reads per channel.
    """
    seq = qubic_sequence(compiled_program)
    qubits = seq.columns
    
    n_reads = {}
    for q in qubits:
        n_reads[str(int(config[f"readout/{q.strip('Q')}/channel"]))] = (
            sum(seq[q] == 'Readout')
        )

    return n_reads


def qubic_sequence(compiled_program) -> pd.DataFrame:
    """Write a qubic sequence from a compiled_program.

    Args:
        compiled_program (distproc.compiler.CompiledProgram): QubiC compiled 
            program object.

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

    df = df.reindex(sorted(df.columns), axis=1)
    df.index.name = '[us]'
    return df.fillna('')