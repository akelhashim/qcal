"""Submodule for translating between qcal and QubiC

References:
https://ieeexplore.ieee.org/document/9552516
"""
import qcal
from qcal.circuit import Circuit
from qcal.config import Config
from qcal.gate.gate import Gate
from qcal.sequencer.pulse_envelopes import pulse_envelopes

import logging

from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


__all__ = ['to_qubic']


def add_reset(
        config: Config, qubits: List | Tuple, circuit: List
    ) -> None:
    """Add active or passive reset to the beginning of a circuit.

    Args:
        config (Config):       config object.
        qubits (List | Tuple): qubits to reset.
        circuit (List):        qubic circuit.
    """
    if config.reset.reset.active.enable:
        reset_circuit = []
        for _ in range(config.reset.reset.active.n_resets):
            for q in qubits:
                add_measurement(config, q, reset_circuit)  # TODO: ESP
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
                        for pulse in config.single_qubit[q]['GE']['X'].pulse
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
        config: Config, qubits: List | Tuple, circuit: List
    ) -> None:
    """Add heralded readout to the beginning of the circuit.

    Args:
        config (Config):       config object.
        qubits (List | Tuple): qubits to reset.
        circuit (List):        qubic circuit.
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
        config: Config, qubit_or_gate: int | Gate, circuit: List
    ) -> None:
    """Add measurement to a circuit.

    Args:
        config (Config):            config object.
        qubit_or_gate (int | Gate): qubit label or Gate object.
        circuit (List):             qubic circuit.
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


def add_single_qubit_gate(config: Config, gate: Gate, circuit: List) -> None:
    """Add a single-qubit gate.

    Args:
        config (Config): config object.
        gate (Gate):     single-qubit gate.
        circuit (List):  qubic circuit.
    """
    subspace = gate.properties['subspace']
    for pulse in (
        config.single_qubit[gate.qubits[0]][subspace][gate.name].pulse):

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


def compilation_error(*args):
    """Generic compilation error.

    Raises:
        Exception: compilation error for non-native gate.
    """
    raise Exception(
        f'Cannot compile! {args[1].name} is a non-native gate.'
    ) 


def compile(
        config: Config, circuit: Circuit, gate_mapper: defaultdict
    ) -> List:
    """Compile a qcal circuit to a qubic circuit.

    Args:
        config (Config):           config object.
        circuit (Circuit):         cal Circuit.
        gate_mapper (defaultdict): map between qcal to QubiC gates.

    Returns:
        List: compiled qubic circuit.
    """
    qubic_circuit = []
    
    # Add reset to the beginning of the circuit
    add_reset(config, circuit.qubits, qubic_circuit)

    # Add (optional) readout heralding
    if config.parameters['readout']['herald']:
        add_heralding(config, circuit.qubits, qubic_circuit)

    for cycle in circuit.cycles:

        if not cycle.is_barrier:
            for gate in cycle:
                gate_mapper[gate.name](config, gate, qubic_circuit)

        elif cycle.is_barrier:
            qubits = cycle.qubits if cycle.qubits else circuit.qubits
            qubic_circuit.append(
                {'name': 'barrier', 
                 'qubit': [f'Q{q}' for q in qubits]},
            )

    return qubic_circuit


class Compiler:
    """Generic qubic compiler."""

    __slots__ = ('_config', '_gate_mapper')

    def __init__(self, config: Config) -> None:
        """Initialize with a qcal config object.

        Args:
            config (Config): config object.
        """
        self._config = config
        
        self._gate_mapper = defaultdict(lambda: compilation_error,
            {'Meas':     add_measurement,
             'VirtualZ': add_virtualz_gate
            }
        )
        for gate in config.basis_gates['set']:
            if gate in qcal.gate.single_qubit.__all__:
                self._gate_mapper[gate] = add_single_qubit_gate
            else:
                self._gate_mapper[gate] = add_multi_qubit_gate

    @property
    def config(self) -> Config:
        """Config object.

        Returns:
            Config: config object.
        """
        return self._config
    
    @property
    def gate_mapper(self) -> defaultdict:
        """Gate mapper.

        This dictionary controls how gates are mapped from qcal to QubiC.

        Returns:
            defaultdict: gate mapper.
        """
        return self._gate_mapper

    def compile(self, circuits: List[Circuit]) -> List[Dict]:
        """Compile all circuits.

        Args:
            circuits (List[Circuit]): circuits to compile.

        Returns:
            List[Dict]: compiled circuits.
        """
        compiled_circuits = [
            compile(
                self._config, circuit, self._gate_mapper
            ) for circuit in circuits
        ]
        return compiled_circuits