"""Helper functions for QubiC

"""
from qcal.config import Config
from qcal.sequence.pulse_envelopes import pulse_envelopes
from qcal.units import ns, us

import logging
import numpy as np
import pandas as pd

from numpy.typing import NDArray
from typing import Dict

logger = logging.getLogger(__name__)


def calculate_n_reads(compiled_program) -> dict:
    """Calculate the number of reads per qubit per circuit.

    The number of reads will depend on:
    1) number of active resets
    2) heralding
    3) number of mid-circuit measurements
    3) readout at the end of the circuit

    Args:
        compiled_program (distproc.compiler.CompiledProgram): QubiC compiled
            program object.

    Returns:
        dict: number of reads per channel.
    """
    seq = qubic_sequence(compiled_program)
    qubits = [col for col in seq.columns if 'Q' in col]
    
    n_reads = {}
    for q in qubits:
        n_reads[f'{q}.rdlo'] = sum(seq[q] == 'Readout')

    return n_reads


def generate_pulse(
        config: Config, pulse: Dict, include_amp_phase: bool = True    
    ) -> NDArray[np.complex64]:
    """Generate a pulse in a form compatible with QubiC.

    Args:
        config (Config): qcal Config object.
        pulse (Dict): dictionary object of a pulse returned from indexing a
            config file.
        include_amp_phase (bool): whether to include the pulse amp and phase in
            the generation of the envelope. Defaults to True. By default, QubiC
            uses False because the amp and phase are dynamically included in
            compilation. This lowers the waveform memory requirements when
            generating sequences.

    Returns:
        NDArray[np.complex64]: complex pulse array.
    """
    if not include_amp_phase:
        kwargs = {}
        for key, val in pulse['kwargs'].items():
            if key not in ['amp', 'phase']:
                kwargs[key] = val
    else:
        kwargs = {key: val for key, val in pulse['kwargs'].items()}

    pulse = pulse_envelopes[pulse['env']](
        pulse['length'],
        config['hardware/sample_rate/DAC'] / 
            config['hardware/interpolation_ratio/qdrv'],
        **kwargs
    )
    return pulse


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
        if qubit in df.columns:
            if len(pd.DataFrame({qubit: tags}, index=times)) > 0:
                df = df.combine_first(
                    pd.DataFrame({qubit: tags}, index=times)
                )
        else:
            df = df.join(
                pd.DataFrame({qubit: tags}, index=times),
                how='outer'
            )

    df = df.reindex(sorted(df.columns), axis=1).fillna('')
    df.index.name = '[us]'
    return df.fillna('')