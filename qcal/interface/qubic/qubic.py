"""Submodule for translating between qcal and QubiC

References:
https://ieeexplore.ieee.org/document/9552516
"""
from qcal.circuit import Circuit
from qcal.config import Config
from qcal.gate.gate import Gate
from qcal.sequencer.pulse_envelopes import pulse_envelopes

import logging

from collections import defaultdict
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


__all__ = ['to_qubic']


def add_reset(
        config: Config, qubits: Union[List, Tuple], circuit: List
    ) -> None:
    """Add active or passive reset to the beginning of a circuit.

    Args:
        config (Config):             config object.
        qubits (Union[List, Tuple]): qubits to reset.
        circuit (List):              qubic circuit.
    """
    if config.reset.reset.active.enable:
        reset_circuit = []
        for _ in range(config.reset.reset.active.n_resets):
            for q in qubits:
                add_measurement(config, q, reset_circuit)
                reset_circuit.append(
                    {'name': 'branch_fproc',
                     'alu_cond': 'eq',
                     'cond_lhs': 1,
                     'func_id': int(config.readout[q].channel),
                     'scope': [f'Q{q}'],  # TODO: replace with qubit channel
                        'true': [  # Why delay?
                            {'name': 'delay', 't': 200.e-9, 'qubit': [f'Q{q}']}
                        ] + [
                            {'name': 'pulse',
                             # TODO: make compatible with qutrit reset
                             'freq': config.single_qubit[q]['GE'].freq[0],
                             'amp': 1.0,
                             'dest': pulse['channel'], 
                             'phase': 0.0,
                             'twidth': pulse['length'],
                             'env': pulse_envelopes[pulse['env']](
                                pulse['length'],
                                config.hardware.loc['DAC_sample_rate'][0],
                                **pulse['kwargs']
                             )}
                        # TODO: add X90 capability
                        for pulse in config.single_qubit[q]['GE']['X180'].pulse
                        ],
                        'false': []
                    },
                )

            reset_circuit.append(
                {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]}
            )
        circuit.extend(reset_circuit)
    
    else:
         circuit.extend((
            {'name': 'delay',
             't': config.reset.reset.passive.delay,
             'qubit': [f'Q{q}' for q in qubits]},
            {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]}
         ))


def add_heralding(
        config: Config, qubits: Union[List, Tuple], circuit: List
    ) -> None:
    """Add heralded readout to the beginning of the circuit.

    Args:
        config (Config):             config object.
        qubits (Union[List, Tuple]): qubits to reset.
        circuit (List):              qubic circuit.
    """
    for q in qubits: # TODO: Pass multiple qubits in the same list?
        add_measurement(config, q, circuit)

    circuit.extend((
            {'name': 'delay',
             't': config.parameters['readout']['reset'],
             'qubit': [f'Q{q}' for q in qubits]},
            {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]}
         ))
         

def add_measurement(
        config: Config, qubit_or_gate: Union[int, Gate], circuit: List
    ) -> None:
    """Add measurement to a circuit.

    Args:
        config (Config):                  config object.
        qubit_or_gate (Union[int, Gate]): qubit label or Gate object.
        circuit (List):                   qubic circuit.
    """
    if isinstance(qubit_or_gate, int):
        qubit = qubit_or_gate
    elif isinstance(qubit_or_gate, Gate):
        qubit = qubit_or_gate.qubits[0]
    circuit.append({'name': 'read', 'qubit': [f'Q{qubit}']})
    # circuit.extend((
    #         {'name': 'pulse',
    #          'dest': f'Q{qubit}.rdrv',
    #          'freq': config.readout[qubit].freq,
    #          'amp': 1.0, 
    #          'phase': 0.0,
    #          'twidth': config.readout[qubit].length,
    #          'env': pulse_envelopes[config.readout[qubit].env](
    #                 config.readout[qubit].length,
    #                 config.readout[qubit].sample_rate,
    #                 amp=config.readout[qubit].amp,
    #                 **config.readout[qubit].kwargs
    #             )
    #         },
    #         {'name': 'pulse', # Is this right?
    #          'dest': f'Q{qubit}.rdlo',
    #          'freq': config.readout[qubit].freq,
    #          'amp': 1.0,
    #          'phase': 1.4806632895920675,  # <-- What is this phase?
    #          'twidth': config.readout[qubit].length,
    #          't0': 6e-07,  # <-- Hardcoded? Readout delay?
    #          'env': pulse_envelopes['square'](
    #                 config.readout[qubit].length,
    #                 config.readout[qubit].sample_rate,
    #             )
    #         }
    #     )
    # )


def add_virtualz_gate(config: Config, gate: Gate, circuit: List):
    """Add a virtual Z gate.

    Args:
        config (Config): config object.
        gate (Gate):     VirtualZ gate.
        circuit (List):  qubic circuit.
    """
    circuit.append(
        {'name': 'virtualz',
         'qubit': [f'Q{gate.qubits[0]}'],
        #  'freq': config.single_qubit[gate.qubits[0]][subspace].freq[0]
         'phase': gate.properties['params']['phase']}
    )


def add_X90_gate(config: Config, gate: Gate, circuit: List) -> None:
    """Add an X90 gate.

    Args:
        config (Config): config object.
        gate (Gate):     X90 gate.
        circuit (List):  qubic circuit.
    """
    subspace = gate.properties['subspace']
    for pulse in (
        config.single_qubit[gate.qubits[0]][subspace]['X90'].pulse):

        if pulse['env'] == 'virtualz':
            circuit.append(
                {'name': 'virtualz',
                 'qubit': [pulse['channel'][:2]],
                # 'freq': config.single_qubit[gate.qubits[0]][subspace].freq[0]
                 'phase': pulse['kwargs']['phase']
                }
            )
            
        else:
            circuit.append(
                {'name': 'pulse',
                 'dest': pulse['channel'],
                 'freq': config.single_qubit[gate.qubits[0]][subspace].freq[0],
                 'amp': 1.0,
                 'phase': 0.0,
                 'twidth': pulse['length'],
                 'env': pulse_envelopes[pulse['env']](
                        pulse['length'],
                        config.hardware.loc['DAC_sample_rate'][0],
                        **pulse['kwargs']
                    )
                }
            )


def add_multi_qubit_gate(config: Config, gate: Gate, circuit: List) -> None:
    """Add a multi-qubit gate.

    Args:
        config (Config): config object.
        gate (Gate):     multi-qubit gate.
        circuit (List):  qubic circuit.
    """
    for pulse in config.two_qubit[gate.qubits][gate.name].pulse[0]:

        if pulse['env'] == 'virtualz':
            circuit.append(
                {'name': 'virtualz',
                 'qubit': [pulse['channel'][:2]],
                # 'freq': config.single_qubit[gate.qubits[0]][subspace].freq[0]
                 'phase': pulse['kwargs']['phase']
                }
            )

        else:
            circuit.append(
                {'name': 'pulse',
                 'dest': pulse['channel'], 
                 'freq': config.two_qubit[gate.qubits][gate.name].freq[0],
                 'amp': 1.0,
                 'phase': 0.0,
                 'twidth': pulse['length'], 
                 'env': pulse_envelopes[pulse['env']](
                        pulse['length'],
                        config.hardware.loc['DAC_sample_rate'][0],
                        **pulse['kwargs']
                    )
                }
            )


def to_qubic(config: Config, circuit: Circuit):

    sq_gate_mapper = defaultdict(lambda: 'Cannot transpile a non-native gate!',
        {'Meas':      add_measurement,
         'VirtualZ':  add_virtualz_gate,
         'X90':       add_X90_gate,
        #  'Y90':       add_Y90_gate  # TODO
        }
    )

    qubic_circuit = []
    
    # Add reset to the beginning of the circuit
    add_reset(config, circuit.qubits, qubic_circuit)

    # Add (optional) readout heralding
    if config.parameters['readout']['herald']:
        add_heralding(config, circuit.qubits, qubic_circuit)

    for cycle in circuit.cycles:

        if not cycle.is_barrier:
            for gate in cycle:
                if gate.is_single_qubit:
                    sq_gate_mapper[gate.name](config, gate, qubic_circuit)
                elif gate.is_multi_qubit:
                    add_multi_qubit_gate(config, gate, qubic_circuit)

        else:
            qubits = cycle.qubits if cycle.qubits else circuit.qubits
            qubic_circuit.append(
                {'name': 'barrier', 
                 'qubit': [f'Q{q}' for q in qubits]},
            )

    return qubic_circuit