"""Submodule for translating between qcal and QubiC

References:
https://ieeexplore.ieee.org/document/9552516
"""
import qcal
from qcal.calibration.utils import find_pulse_index
from qcal.circuit import Circuit, CircuitSet
from qcal.config import Config
from qcal.circuit import Cycle
from qcal.gate.gate import Gate
from qcal.gate.single_qubit import Id, Idle, MCM, Reset, Rz, X
from qcal.sequencer.dynamical_decoupling import dd_sequences
from qcal.sequencer.pulse_envelopes import pulse_envelopes
from qcal.sequencer.utils import clip_amplitude

import logging

from collections import defaultdict
from numpy.typing import NDArray
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


__all__ = ('to_qubic', 'Transpiler')


def add_reset(
       config: Config, 
       qubits_or_reset: List | Tuple | Reset, 
       circuit: List, 
       pulses: defaultdict
    ) -> None:
    """Add active or passive reset to the beginning of a circuit.

    Args:
        config (Config):       qcal Config object.
        qubits_or_reset (List | Tuple | Reset): qubits to reset, or Reset
            operation.
        circuit (List):        qubic circuit.
        pulses (defaultdict):  pulses that have been stored for reuse.
    """
    if isinstance(qubits_or_reset, (list, tuple)):
        qubits = qubits_or_reset
    elif isinstance(qubits_or_reset, Reset):
        qubits = qubits_or_reset.qubits

    if config['reset/active/enable']:
        reset_circuit = []
        for _ in range(config['reset/active/n_resets']):
            for q in qubits:
                if isinstance(qubits_or_reset, (list, tuple)):
                    add_measurement(config, q, reset_circuit, pulses)
                elif isinstance(qubits_or_reset, Reset):
                    add_measurement(
                        config, 
                        qubits_or_reset.properties['params']['meas'], 
                        reset_circuit, 
                        pulses
                    )

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
                             'env': clip_amplitude(
                                pulse_envelopes[pulse['env']](
                                    pulse['length'],
                                    config['hardware/DAC_sample_rate'],
                                    **pulse['kwargs']
                                )
                             )
                            }
                        # TODO: add X90 capability
                        for pulse in config[f'single_qubit{q}/GE/X/pulse']
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
       config: Config, qubits: List | Tuple, circuit: List, pulses: defaultdict
    ) -> None:
    """Add heralded readout to the beginning of the circuit.

    Args:
        config (Config):       qcal Config object.
        qubits (List | Tuple): qubits to reset.
        circuit (List):        qubic circuit.
        pulses (defaultdict):  pulses that have been stored for reuse.
    """
    for q in qubits:
        add_measurement(config, q, circuit, pulses)

    circuit.extend((
            {'name': 'delay',
             't': config['readout/reset'],
             'qubit': [f'Q{q}' for q in qubits]},
            {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]}
         ))
         

def add_measurement(
        config: Config, 
        qubit_or_meas: int | Gate,
        circuit: List, 
        pulses: defaultdict
    ) -> None:
    """Add measurement to a circuit.

    Args:
       config (Config):            qcal Config object.
       qubit_or_meas (int | Gate): qubit label or measurement Gate object.
       circuit (List):             qubic circuit.
       pulses (defaultdict):       pulses that have been stored for reuse.
    """
    if isinstance(qubit_or_meas, int):
        qubit = qubit_or_meas
        qubits = (qubit,)
    elif isinstance(qubit_or_meas, Gate):
        qubits = qubit_or_meas.qubits
        qubit = qubits[0]

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
                     'env': clip_amplitude(
                        pulse_envelopes[pulse['env']](
                            pulse['length'],
                            config['hardware/DAC_sample_rate'],
                            **pulse['kwargs']
                        )
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
                {'name': 'delay', 't': length, 'qubit': [f'Q{qubit}']}
            )

    if isinstance(qubit_or_meas, Gate) and qubit_or_meas.name == 'MCM':
        meas_pulse.append({
            'name': 'barrier', 
            'qubit': [
               f'Q{q}' for q in qubit_or_meas.properties['params']['dd_qubits']
            ]
        })

    meas_pulse.append({'name': 'barrier', 'qubit': [f'Q{qubit}']})
    meas_pulse.extend([
            {'name': 'pulse',
             'dest': f'Q{qubit}.rdrv',
             'freq': config[f'readout/{qubit}/freq'],
             'amp': 1.0, 
             'phase': 0.0,
             'twidth': config['readout/length'],
             'env': clip_amplitude(
                pulse_envelopes[config.readout[qubit].env](
                    config['readout/length'],
                    config['readout/sample_rate'],
                    amp=config[f'readout/{qubit}/amp'],
                    **config['readout/kwargs']
                )
             )
            },
            {'name': 'delay',
             't': config[f'readout/{qubit}/delay'],
             'qubit': [f'Q{qubit}.rdlo']
            },
            {'name': 'pulse',
             'dest': f'Q{qubit}.rdlo',
             'freq': config[f'readout/{qubit}/freq'],
             'amp': 1.0,
             'phase': config[f'readout/{qubit}/phase'],  # Rotation in IQ plane
             'twidth': config['readout/length'],
             'env': clip_amplitude(
                pulse_envelopes['square'](
                    config['readout/length'],
                    config['readout/sample_rate'],
                )
             )
            }
        ]
    )

    if isinstance(qubit_or_meas, Gate) and qubit_or_meas.name == 'MCM':
        for q in qubit_or_meas.properties['params']['dd_qubits']:
            add_dynamical_decoupling(
                config,
                qubit_or_meas.properties['params']['dd_method'],
                q,
                config['readout/length'],
                qubit_or_meas.properties['params']['n_dd_pulses'],
                meas_pulse
            )
        add_mcm_apply(config, qubit_or_meas, meas_pulse)
        pulses[f'MCMGE:{qubits}'] = meas_pulse
    
    else:
        pulses[f'MeasGE:{qubits}'] = meas_pulse
    
    circuit.extend(meas_pulse)


def add_dynamical_decoupling(
        config: Config, 
        dd_method: str, 
        qubit: int, 
        length: float, 
        n_dd_pulses: int, 
        pulse: List
    ) -> None:
    """Add dynamical decoupling pulses to a sequence.

    Args:
        config (Config): qcal Config object.
        dd_method (str): dynamical decoupling method.
        qubit (int): qubit label.
        length (float): length of time over which to perform the DD.
        n_dd_pulses (int): number of dynamical decoupling pulses.
        pulse (List): qubic pulse.
    """
    dd_circuit = dd_sequences[dd_method](config, qubit, length, n_dd_pulses)
    for cycle in dd_circuit: 
        if cycle.is_barrier:
            pulse.append({'name': 'barrier', 'qubit': [f'Q{qubit}']})
        else:
            pulse.extend(cycle_pulse(config, cycle))

                
def add_mcm_apply(config: Config, mcm: MCM, pulse: List) -> None:
    """Add a conditional gate depending on the results of a MCM.

    Args:
        config (Config): qcal Config object.
        mcm (MCM): mid-circuit measurement object.
        pulse (List): qubic pulse.
    """
    q_meas = MCM.qubits[0]
    q_cond = MCM['params']['apply']['1'].qubits

    # Initial barrier
    pulse.append(
        {'name': 'barrier', 
         'scope': [f'Q{q}' for q in [q_meas] + [qc for qc in q_cond]]
        }
    )

    # Seqeuence to apply if 1 is measured
    true_apply = [
        {'name': 'delay', 
         't': config['reset/active/feedback_delay'], 
         'qubit': [f'Q{q}' for q in [qc for qc in q_cond]]
        },
    ]
    true_apply.extend(
        cycle_pulse(config, MCM['params']['apply']['1'])
    )

    # Sequence to apply if 0 is measured
    false_apply = cycle_pulse(config, MCM['params']['apply']['0'])

    # Conditional operation
    pulse.append(
        {'name': 'branch_fproc', 
         'alu_cond': 'eq', 
         'cond_lhs': 1, 
         'func_id': int(config[f'readout/{q_meas}/channel']), 
         'scope': [f'Q{q}' for q in [qc for qc in q_cond]],
         'true': true_apply,
         'false': false_apply
        }
    )

    # Final barrier
    pulse.append(
        {'name': 'barrier', 
         'scope': [f'Q{q}' for q in [q_meas] + [qc for qc in q_cond]]
        }
    )


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
    delay_pulse =  [{'name': 'delay', 't': duration, 'qubit': [f'Q{qubit}']}]
    pulses[f'{gate.name}{gate.subspace}:{gate.qubits}'] = delay_pulse
    circuit.extend(delay_pulse)


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
    subspace = gate.subspace
    phase = gate.properties['params']['phase']
    phase_pulse = [{
        'name': 'virtual_z',
        'freq': config[f'single_qubit/{gate.qubits[0]}/{subspace}/freq'],
        'phase': phase
    }]
    pulses[f'{gate.name}{phase}{gate.subspace}:{gate.qubits}'] = phase_pulse
    circuit.extend(phase_pulse)


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
    name = gate.name
    subspace = gate.subspace
    sq_pulse = []
    for pulse in (
        config[f'single_qubit/{qubit}/{subspace}/{name}/pulse']):

        if pulse['env'] == 'virtualz':
            sq_pulse.append(
                {'name': 'virtual_z',
                 'freq': config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'phase': pulse['kwargs']['phase']
                }
            )
            
        else:
            sq_pulse.append(
                {'name': 'pulse',
                 'dest': pulse['channel'],
                 'freq': config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'amp': clip_amplitude(pulse['kwargs']['amp']),
                 'phase': pulse['kwargs']['phase'],
                 'twidth': pulse['length'],
                 'env': pulse_envelopes[pulse['env']](
                        pulse['length'],
                        config['hardware/DAC_sample_rate'],
                        **{key: val for key, val in pulse['kwargs'].items() 
                           if key not in ['amp', 'phase']}
                    )
                }
            )

    pulses[f'{name}{subspace}:{gate.qubits}'] = sq_pulse
    circuit.extend(sq_pulse)


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
    qubits = gate.qubits
    name = gate.name
    mq_pulse = []

    # Add dynamical decoupling
    if config[f'two_qubit/{qubits}/{name}/dynamical_decoupling/enable']:
        sub_config = config[f'two_qubit/{qubits}/{name}/dynamical_decoupling']
        idx = find_pulse_index(config, f'two_qubit/{qubits}/{name}/pulse')
        mq_pulse.append(
            {'name': 'barrier', 
             'scope': [f'Q{q}' for q in sub_config['qubits']]
            }
        )
        for q in sub_config['qubits']:
            add_dynamical_decoupling(
                config,
                sub_config['dd_method'],
                q,
                config[f'two_qubit/{qubits}/{name}/pulse'][idx]['length'],
                sub_config['n_pulses'],
                mq_pulse
            )

    for pulse in config[f'two_qubit/{qubits}/{name}/pulse']:

        if pulse['env'] == 'virtualz':
            mq_pulse.append(
                {'name': 'virtual_z',
                 'freq': config[pulse['freq']],
                 'phase': pulse['kwargs']['phase']
                }
            )

        else:
            mq_pulse.append(
                {'name': 'pulse',
                 'dest': pulse['channel'], 
                 'freq': config[f'two_qubit/{qubits}/{name}/freq'],
                 'amp': clip_amplitude(pulse['kwargs']['amp']),
                 'phase': pulse['kwargs']['phase'],
                 'twidth': pulse['length'], 
                 'env': pulse_envelopes[pulse['env']](
                        pulse['length'],
                        config['hardware/DAC_sample_rate'],
                        **{key: val for key, val in pulse['kwargs'].items() 
                           if key not in ['amp', 'phase']}
                    )
                }
            )

    pulses[f'{name}{gate.subspace}:{qubits}'] = mq_pulse
    circuit.extend(mq_pulse)


def cycle_pulse(config: Config, cycle: Cycle) -> List:
    """Generate a pulse from a cycle of operations.

    This is useful for generating sub-pulses for DD sequences or MCM.

    Args:
        config (Config): qcal Config object.
        cycle (Cycle): cycle of gates.

    Returns:
        List: pulse.
    """
    pulse = []
    for gate in cycle:
        qubit = gate.qubits[0]
        subspace = gate.subspace

        if isinstance(gate, Id):
            continue

        if isinstance(gate, Idle):
            pulse.append(
                {'name': 'delay', 
                 't': gate.properties['params']['duration'], 
                 'qubit': [f'Q{qubit}']
                }
            )

        elif isinstance(gate, Rz):
            pulse.append(
                {'name': 'virtual_z',
                 'freq': config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'phase': gate.properties['params']['phase']
                }
            )

        elif isinstance(gate, X):
            pulse.extend([
                {'name': 'pulse',
                 'dest': p['channel'], 
                 'freq': config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'amp': clip_amplitude(p['kwargs']['amp']),
                 'phase': p['kwargs']['phase'],
                 'twidth': p['length'],
                 'env': pulse_envelopes[p['env']](
                    p['length'],
                    config['hardware/DAC_sample_rate'],
                    **{key: val for key, val in p['kwargs'].items() 
                       if key not in ['amp', 'phase']}
                )} for p in config[f'single_qubit{qubit}/{subspace}/X/pulse']
            ])

    return pulse


def transpilation_error(*args):
    """Generic transpilation error.

    Raises:
        Exception: transpilation error for non-native gate.
    """
    raise Exception(
        f'Cannot transpile {args[1].name}! is a non-native gate.'
    ) 


def to_qubic(
        config:      Config, 
        circuit:     Circuit, 
        gate_mapper: defaultdict,
        pulses:      defaultdict,
        qubit_reset: bool = True
    ) -> List:
    """Compile a qcal circuit to a qubic circuit.

    Args:
        config (Config):           config object.
        circuit (Circuit):         qcal circuit.
        gate_mapper (defaultdict): map between qcal to QubiC gates.
        pulses (defaultdict):      pulses that have been stored for reuse.
        qubit_reset (bool):        whether to reset the qubits before a 
                circuit. Defaults to True. This can be set to False when adding
                subcircuits like mid-circuit measurement.

    Returns:
        List: compiled qubic circuit.
    """
    qubic_circuit = []
    
    # Add reset to the beginning of the circuit
    if qubit_reset:
        add_reset(config, circuit.qubits, qubic_circuit, pulses)

        # Add (optional) readout heralding
        if config.parameters['readout']['herald']:
            add_heralding(config, circuit.qubits, qubic_circuit, pulses)

    for cycle in circuit.cycles:

        if not cycle.is_barrier:
            for gate in cycle:

                name = gate.name
                if 'phase' in gate.properties['params'].keys():
                    name += str(gate.properties['params']['phase'])
                if pulses[f'{name}{gate.subspace}:{gate.qubits}']:
                    qubic_circuit.extend(
                        pulses[f'{name}{gate.subspace}:{gate.qubits}']
                    )

                else:
                    gate_mapper[gate.name](config, gate, qubic_circuit, pulses)

        elif cycle.is_barrier:
            qubits = cycle.qubits if cycle.qubits else circuit.qubits
            qubic_circuit.append(
                {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]},
            )

    return qubic_circuit


class Transpiler:
    """qcal to QubiC Transpiler."""

    # __slots__ = (
    #     '_config', '_gate_mapper', '_reload_pulse', '_qubit_reset', '_pulses'
    # )

    def __init__(self, 
            config:       Config, 
            gate_mapper:  defaultdict | None = None,
            reload_pulse: bool = True,
            qubit_reset:  bool = True
        ) -> None:
        """Initialize with a qcal Config object.

        Args:
            config (Config): qcal config object.
            gate_mapper (defaultdict | None, optional): dictionary which maps
                circuit gates to QubiC gates. Defaults to None.
            reload_pulse (bool): reloads the stored pulses when compiling each
                circuit. Defaults to True.
            qubit_reset (bool): whether to reset the qubits before a circuit.
                Defaults to True. This can be set to False when adding
                subcircuits like mid-circuit measurement.

        """
        self._config = config
        
        if gate_mapper is None:
            self._gate_mapper = defaultdict(lambda: transpilation_error,
                {'Reset':    add_reset,
                 'Meas':     add_measurement,
                 'MCM' :     add_measurement,
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
        self._qubit_reset = qubit_reset
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
                    self._config, 
                    circuit, 
                    self._gate_mapper, 
                    self._pulses,
                    self._qubit_reset
                )
            )
              
        if params:
            self._config.reload()  # Reload after making all the changes

        return transpiled_circuits