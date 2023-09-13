"""Submodule for translating between qcal and QubiC

References:
https://ieeexplore.ieee.org/document/9552516
"""
import qcal
from qcal.circuit import Circuit, CircuitSet
from qcal.config import Config
from qcal.gate.gate import Gate
from qcal.sequencer.pulse_envelopes import pulse_envelopes

import logging

from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


__all__ = ('to_qubic', 'Transpiler')


def add_reset(
        config: Config, qubits: List | Tuple, circuit: List
    ) -> None:
    """Add active or passive reset to the beginning of a circuit.

    Args:
        config (Config):       qcal Config object.
        qubits (List | Tuple): qubits to reset.
        circuit (List):        qubic circuit.
    """
    if config['reset/active/enable']:
        reset_circuit = []
        for _ in range(config['reset/active/n_resets']):
            for q in qubits:
                add_measurement(config, q, reset_circuit)
                reset_circuit.append({'name': 'barrier', 'scope': [f'Q{q}']})
                reset_circuit.append(
                    {'name': 'branch_fproc',
                     'alu_cond': 'eq',
                     'cond_lhs': 1,
                     'func_id': int(config[f'readout/{q}/channel']),
                     'scope': [f'Q{q}'],
                        'true': [
                            {'name': 'delay', 
                             't': config['reset/active/feedback_delay'], 
                             'qubit': [f'Q{q}']
                            }
                        ] + 
                        [
                            {'name': 'pulse',
                             'freq': config[f'single_qubit/{q}/GE/freq'],
                             'amp': 1.0,
                             'dest': pulse['channel'], 
                             'phase': 0.0,
                             'twidth': pulse['length'],
                             'env': pulse_envelopes[pulse['env']](
                                pulse['length'],
                                config['hardware/DAC_sample_rate'],
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
             't': config['reset/passive/delay'],
             'qubit': [f'Q{q}' for q in qubits]},
            {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]}
         ))


def add_heralding(
        config: Config, qubits: List | Tuple, circuit: List
    ) -> None:
    """Add heralded readout to the beginning of the circuit.

    Args:
        config (Config):       qcal Config object.
        qubits (List | Tuple): qubits to reset.
        circuit (List):        qubic circuit.
    """
    for q in qubits:
        add_measurement(config, q, circuit)

    circuit.extend((
            {'name': 'delay',
             't': config['readout/reset'],
             'qubit': [f'Q{q}' for q in qubits]},
            {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]}
         ))
         

def add_measurement(
        config: Config, 
        qubit_or_gate: int | Gate,
        circuit: List, 
        pulses: defaultdict
    ) -> None:
    """Add measurement to a circuit.

    Args:
       config (Config):            qcal Config object.
       qubit_or_gate (int | Gate): qubit label or Gate object.
       circuit (List):             qubic circuit.
       pulses (defaultdict):       pulses that have been stored for reuse.
    """
    if isinstance(qubit_or_gate, int):
        qubit = qubit_or_gate
    elif isinstance(qubit_or_gate, Gate):
        qubit = qubit_or_gate.qubits[0]

    meas_pulse = []
    # Excited state promotion
    if config.parameters['readout']['esp']['enable']:
        # TODO: add X90 capability
        if qubit in config.parameters['readout']['esp']['qubits']:
            meas_pulse.extend([
                    {'name': 'pulse',
                     'freq': config[f'single_qubit/{qubit}/EF/freq'],
                     'amp': 1.0,
                     'dest': pulse['channel'], 
                     'phase': 0.0,
                     'twidth': pulse['length'],
                     'env': pulse_envelopes[pulse['env']](
                        pulse['length'],
                        config['hardware/DAC_sample_rate'],
                        **pulse['kwargs']
                        )
                    }
                    for pulse in config[f'single_qubit/{qubit}/EF/X/pulse']
                ]
            )
        
        else:
            length = 0.
            for pulse in config[f'single_qubit/{qubit}/EF/X/pulse']:
                length += pulse['length']
            meas_pulse.append(
                {'name': 'delay',
                 't': length,
                 'qubit': [f'Q{qubit}']
                }
            )
            
    meas_pulse.append({'name': 'barrier', 'qubit': [f'Q{qubit}']})

    meas_pulse.extend([
            {'name': 'pulse',
             'dest': f'Q{qubit}.rdrv',
             'freq': config[f'readout/{qubit}/freq'],
             'amp': 1.0, 
             'phase': 0.0,
             'twidth': config['readout/length'],
             'env': pulse_envelopes[config.readout[qubit].env](
                    config['readout/length'],
                    config['readout/sample_rate'],
                    amp=config[f'readout/{qubit}/amp'],
                    **config['readout/kwargs']
                )
            },
            {'name': 'delay',
             't': config.parameters['readout']['delay'],
             'qubit': [f'Q{qubit}.rdlo']
            },
            {'name': 'pulse',
             'dest': f'Q{qubit}.rdlo',
             'freq': config[f'readout/{qubit}/freq'],
             'amp': 1.0,
             'phase': config[f'readout/{qubit}/phase'],  # Rotation in IQ plane
             'twidth': config['readout/length'],
             'env': pulse_envelopes['square'](
                    config['readout/length'],
                    config['readout/sample_rate'],
                )
            }
        ]
    )

    pulses[f'Meas:{qubit}'] = meas_pulse
    circuit.extend(meas_pulse)


def add_delay(
        config: Config, gate: Gate, circuit: List, pulses: defaultdict
    ) -> None:
    """Add a delay for an idle gate.

    Args:
        config (Config):      qcal Config object. Unused, but included for 
            convention.
        gate (Gate):          single-qubit gate.
        circuit (List):       qubic circuit.
        pulses (defaultdict): pulses that have been stored for reuse.
    """
    qubit = gate.qubits[0]
    duration = gate.properties['params']['duration']
    delay_pulse =  {'name': 'delay', 't': duration, 'qubit': [f'Q{qubit}']}
    if duration > 0.:
        pulses[f'Idle:{qubit}'] = delay_pulse
    else:
        pulses[f'I:{qubit}'] = delay_pulse
    circuit.append(delay_pulse)


def add_virtualz_gate(
        config: Config, gate: Gate, circuit: List, pulses: defaultdict
    ) -> None:
    """Add a virtual Z gate.

    Args:
        config (Config):      qcal Config object.
        gate (Gate):          multi-qubit gate.
        circuit (List):       qubic circuit.
        pulses (defaultdict): pulses that have been stored for reuse.
    """
    qubit = gate.qubits[0]
    phase = gate.properties['params']['phase']
    phase_pulse = {
        'name': 'virtual_z',
        'freq': config[f'single_qubit/{gate.qubits[0]}/{gate.subspace}/freq'],
        'phase': phase
    }
    
    circuit.append(phase_pulse)


def add_single_qubit_gate(
        config: Config, gate: Gate, circuit: List, pulses: defaultdict
    ) -> None:
    """Add a single-qubit gate.

    Args:
        config (Config):      qcal Config object.
        gate (Gate):          multi-qubit gate.
        circuit (List):       qubic circuit.
        pulses (defaultdict): pulses that have been stored for reuse.
    """
    qubit = gate.qubits[0]
    subspace = gate.subspace
    for pulse in (
        config[f'single_qubit/{qubit}/{subspace}/{gate.name}/pulse']):

        if pulse['env'] == 'virtualz':
            circuit.append(
                {'name': 'virtual_z',
                 'freq': config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'phase': pulse['kwargs']['phase']
                }
            )
            
        else:
            circuit.append(
                {'name': 'pulse',
                 'dest': pulse['channel'],
                 'freq': config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'amp': 1.0,
                 'phase': 0.0,
                 'twidth': pulse['length'],
                 'env': pulse_envelopes[pulse['env']](
                        pulse['length'],
                        config['hardware/DAC_sample_rate'],
                        **pulse['kwargs']
                    )
                }
            )


def add_multi_qubit_gate(
        config: Config, gate: Gate, circuit: List, pulses: defaultdict
    ) -> None:
    """Add a multi-qubit gate.

    Args:
        config (Config):      qcal Config object.
        gate (Gate):          multi-qubit gate.
        circuit (List):       qubic circuit.
        pulses (defaultdict): pulses that have been stored for reuse.
    """
    for pulse in config[f'two_qubit/{gate.qubits}/{gate.name}/pulse']:

        if pulse['env'] == 'virtualz':
            circuit.append(
                {'name': 'virtual_z',
                 'freq': config[pulse['freq']],
                 'phase': pulse['kwargs']['phase']
                }
            )

        else:
            circuit.append(
                {'name': 'pulse',
                 'dest': pulse['channel'], 
                 'freq': config[f'two_qubit/{gate.qubits}/{gate.name}/freq'],
                 'amp': 1.0,
                 'phase': 0.0,
                 'twidth': pulse['length'], 
                 'env': pulse_envelopes[pulse['env']](
                        pulse['length'],
                        config['hardware/DAC_sample_rate'],
                        **pulse['kwargs']
                    )
                }
            )


def transpilation_error(*args):
    """Generic transpilation error.

    Raises:
        Exception: transpilation error for non-native gate.
    """
    raise Exception(
        f'Cannot transpile {args[1].name}! is a non-native gate.'
    ) 


def to_qubic(
        config: Config, 
        circuit: Circuit, 
        gate_mapper: defaultdict, 
        pulses: defaultdict
    ) -> List:
    """Compile a qcal circuit to a qubic circuit.

    Args:
        config (Config):           config object.
        circuit (Circuit):         qcal circuit.
        gate_mapper (defaultdict): map between qcal to QubiC gates.
        pulses (defaultdict):      pulses that have been stored for reuse.

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
                gate_mapper[gate.name](config, gate, qubic_circuit, pulses)

        elif cycle.is_barrier:
            qubits = cycle.qubits if cycle.qubits else circuit.qubits
            qubic_circuit.append(
                {'name': 'barrier', 
                 'qubit': [f'Q{q}' for q in qubits]},
            )

    return qubic_circuit


class Transpiler:
    """qcal to QubiC Transpiler."""

    __slots__ = ('_config', '_gate_mapper', '_reset_pulse', '_pulses')

    def __init__(self, 
            config:       Config, 
            gate_mapper:  defaultdict | None = None,
            reload_pulse: bool = True
        ) -> None:
        """Initialize with a qcal Config object.

        Args:
            config (Config): qcal config object.
            gate_mapper (defaultdict | None, optional): dictionary which maps
                circuit gates to QubiC gates. Defaults to None.
            reload_pulse (bool): reloads the stored pulses when compiling each
                circuit. Defaults to True.
        """
        self._config = config
        
        if gate_mapper is None:
            self._gate_mapper = defaultdict(lambda: transpilation_error,
                {'Meas':     add_measurement,
                 'I':        add_delay,
                 'Idle':     add_delay,
                 'Rz':       add_virtualz_gate,
                 'S':        add_virtualz_gate,
                 'Sdag':     add_virtualz_gate,
                 'T':        add_virtualz_gate,
                 'Tdag':     add_virtualz_gate,
                 'VirtualZ': add_virtualz_gate,
                 'Z':        add_virtualz_gate,
                }
            )
            for gate in config.basis_gates['set']:
                if gate in qcal.gate.single_qubit.__all__:
                    self._gate_mapper[gate] = add_single_qubit_gate
                else:
                    self._gate_mapper[gate] = add_multi_qubit_gate
        else:
            self._gate_mapper = gate_mapper

        self._reload_pulse = reload_pulse
        self._pulses = defaultdict(lambda: False, {})

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

    def transpile(self, circuits: CircuitSet) -> List[Dict]:
        """Transpile all circuits.

        Args:
            circuits (CircuitSet): circuits to transpile.

        Returns:
            List[Dict]: transpiled circuits.
        """
        # transpiled_circuits = []
        # # Check for a param sweep
        # params = [col for col in circuits._df.columns if 'param' in col]
        # if params:
        #     for i, circuit in enumerate(circuits):
        #         for param in params:  # [7:] removes the string 'param: '
        #             self._config[param[7:]] = circuits[param][i]
        #         if self._reload_pulse:
        #             self._pulses = defaultdict(lambda: False, {})
        #         transpiled_circuits.append(
        #             to_qubic(
        #                 self._config, circuit, self._gate_mapper, self._pulses
        #             )
        #         )
        #     self._config.reload()  # Reload after making all the changes

        # else:
        #     for circuit in enumerate(circuits):
        #         if self._reload_pulse:
        #             self._pulses = defaultdict(lambda: False, {})
        #         transpiled_circuits.append(
        #             to_qubic(
        #                 self._config, circuit, self._gate_mapper, self._pulses
        #             )
        #         )

        transpiled_circuits = []
        # Check for a param sweep
        params = [col for col in circuits._df.columns if 'param' in col]
        for i, circuit in enumerate(circuits):
            if self._reload_pulse:
                 self._pulses = defaultdict(lambda: False, {})

            if params:
                for param in params:  # [7:] removes the string 'param: '
                    self._config[param[7:]] = circuits[param][i]
            
            transpiled_circuits.append(
                to_qubic(
                    self._config, circuit, self._gate_mapper, self._pulses
                )
            )
            
        if params:
            self._config.reload()  # Reload after making all the changes

        return transpiled_circuits