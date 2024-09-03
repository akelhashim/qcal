"""Submodule for generating a qubit processor spec for pyGSTi.

See:
https://github.com/sandialabs/pyGSTi/blob/master/jupyter_notebooks/Tutorials/objects/ProcessorSpec.ipynb
https://github.com/sandialabs/pyGSTi/blob/master/pygsti/processors/processorspec.py
"""
from qcal.config import Config
from qcal.gate.single_qubit import single_qubit_gates
from qcal.gate.two_qubit import two_qubit_gates

import logging

from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


GATE_MAPPER = {
    'X90':  'Gxpi2',
    'Y90':  'Gypi2',
    'Z90':  'Gzpi2',
    'CNOT': 'Gcnot',
    'CX':   'Gcnot',
    'CZ':   'Gcphase'
}


__all__ = 'pygsti_pspec'


def pygsti_pspec(
        config: Config,
        qubits: List[int] | Tuple[int],
        native_gates: List[str] = ['X90', 'Y90'],
        availability: Dict | None = None,
        nonstd_gate_unitaries: Dict | None = None,
        **kwargs
    ):
    """Generates a pyGSTi qubit processor spec.

    Args:
        config (Config): qcal ```Config``` object.
        qubits (List[int] | Tuple[int]): qubit labels.
        native_gates (List[str], optional): native gates. Defaults to 
            ['X90', 'Y90']. These can be formatted in qcal or pyGSTi 
            format.
        availability (Dict | None, optional): a dictionary whose keys are gate
            names and whose values are a tuple of the qubit labels on which the
            gates act. Defaults to None. If None, this will be automatically
            generated based on the ```config.native_gates``` object.
        nonstd_gate_unitaries (Dict | None, optional): a dictionary whose keys 
            are custom gate names and whose values are unitary matrices. 
            Defaults to None.

    Returns:
        QubitProcessorSpec: pyGSTi qubit processor spec object.
    """
    from pygsti.processors import QubitProcessorSpec

    num_qubits = len(qubits)
    qubit_labels = [f'Q{q}' for q in qubits]
    gate_names = [
        GATE_MAPPER[gate] if gate in GATE_MAPPER.keys() else gate 
        for gate in native_gates
    ]
    
    if availability is None:
        availability = {}
        for gate in native_gates:  # This will break if formatted as a pygsti gate
            if gate in single_qubit_gates:
                availability[GATE_MAPPER[gate]] = [(f'Q{q}',) for q in qubits]
            
            elif gate in two_qubit_gates:
                qubit_pairs = []
                for qubit_pair in config.native_gates['two_qubit'].keys():
                    if all(q in qubits for q in qubit_pair):
                        if gate in config.native_gates['two_qubit'][qubit_pair]:
                            qubit_pairs.append((f'Q{q}' for q in qubit_pair))
                availability[GATE_MAPPER[gate]] = qubit_pairs

    pspec = QubitProcessorSpec(
        num_qubits=num_qubits,
        gate_names=gate_names,
        nonstd_gate_unitaries=nonstd_gate_unitaries,
        availability=availability,
        qubit_labels=qubit_labels,
        **kwargs
    )

    return pspec