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
from qcal.gate.single_qubit import (
    Id, Idle, Meas, MCM, Reset, Rz, VirtualZ, X, Z
)
from qcal.sequencer.dynamical_decoupling import dd_sequences
from qcal.sequencer.pulse_envelopes import pulse_envelopes
from qcal.sequencer.utils import clip_amplitude

import logging

from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


__all__ = ('to_qubic', 'Transpiler')


def readout_time(config: Config, qubit: int) -> float:
    """Compute the total readout time for a given qubit.

    Args:
        config (Config): qcal Config object.
        qubit (int):     qubit label.

    Returns:
        float: total readout time (readout + demod - integration delay)
    """
    time = (
        config[f'readout/{qubit}/delay'] + 
        config[f'readout/{qubit}/demod_time']
    )
    return time


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

    if config['reset/active/enable'] or isinstance(qubits_or_reset, Reset):
        reset_circuit = []
        for _ in range(config['reset/active/n_resets']):
            for q in qubits:
                if isinstance(qubits_or_reset, (list, tuple)):
                    add_measurement(config, q, reset_circuit, None, reset=True)
                elif isinstance(qubits_or_reset, Reset):
                    add_measurement(
                        config, 
                        qubits_or_reset.properties['params']['meas'], 
                        reset_circuit, 
                        None,
                        reset=True
                    )

                # Reset pulse w/ qutrit reset
                reset_q_pulse = []
                # if (config.parameters['readout']['esp']['enable'] and
                #     q in config.parameters['readout']['esp']['qubits']):
                #     reset_q_pulse.extend(
                #         cycle_pulse(config, Cycle({X(q, subspace='EF')}))
                #     )
                #     reset_q_pulse.extend(
                #         cycle_pulse(config, Cycle({X(q, subspace='GE')}))
                #     )
                # else:
                reset_q_pulse.extend(
                    cycle_pulse(config, Cycle({X(q, subspace='GE')}))
                )
                    # reset_q_pulse.extend(
                    #     cycle_pulse(config, Cycle({X(q, subspace='EF')}))
                    # )

                reset_circuit.append({'name': 'barrier', 'scope': [f'Q{q}']})
                reset_circuit.append(
                    {'name':     'branch_fproc',
                     'alu_cond': 'eq',
                     'cond_lhs': 1,
                     'func_id':  f'Q{q}.meas',
                     'scope':    [f'Q{q}'],
                        'true':  reset_q_pulse,
                        'false': []
                    },
                )

            reset_circuit.append(
                {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]}
            )

        if isinstance(qubits_or_reset, Reset):
            pulses[f'ResetGE:{qubits}'] = reset_circuit

        circuit.extend(reset_circuit)
    
    else:
         circuit.extend((
            {'name':  'delay',
             't':     config['reset/passive/delay'],
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
            {'name':  'delay',
             't':     config['readout/reset'],
             'qubit': [f'Q{q}' for q in qubits]},
            {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]}
         ))
         

def add_measurement(
        config: Config, 
        qubit_or_meas: int | Gate,
        circuit: List, 
        pulses: defaultdict | None,
        reset: bool = False
    ) -> None:
    """Add measurement to a circuit.

    Args:
        config (Config):             qcal Config object.
        qubit_or_meas (int | Gate):  qubit label or measurement Gate object.
        circuit (List):              qubic circuit.
        pulses (defaultdict | None): pulses that have been stored for reuse.
        reset (bool):                whether or not the measurement is for a
            reset operation. Defaults to False.
    """
    if isinstance(qubit_or_meas, int):
        qubit = qubit_or_meas
        qubits = (qubit,)
    elif isinstance(qubit_or_meas, Gate):
        qubits = qubit_or_meas.qubits
        qubit = qubits[0]

    meas_pulse = []
    # Excited state promotion
    if config.parameters['readout']['esp']['enable'] and not reset:
        if qubit in config.parameters['readout']['esp']['qubits']:
            meas_pulse.extend(
                cycle_pulse(config, Cycle({X(qubit, subspace='EF')}))
            )
        
        else:
            length = 0.
            for pulse in config[f'single_qubit/{qubit}/EF/X/pulse']:
                length += pulse['length']
            meas_pulse.append(
                {'name':  'delay',
                 't':     length, 
                 'qubit': [f'Q{qubit}']}
            )

    meas_pulse.append({'name': 'barrier', 'qubit': [f'Q{qubit}']})
    meas_pulse.extend(cycle_pulse(config, Cycle({Meas(qubit)})))

    if isinstance(qubit_or_meas, Gate) and qubit_or_meas.name == 'MCM':
        for q in qubit_or_meas.properties['params']['dd_qubits']:
            add_dynamical_decoupling(
                config,
                qubit_or_meas.properties['params']['dd_method'],
                q,
                readout_time(config, qubit),
                qubit_or_meas.properties['params']['n_dd_pulses'],
                meas_pulse
            )
        if qubit_or_meas.properties['params']['apply']:
            add_mcm_apply(config, qubit_or_meas, meas_pulse)
        if pulses is not None:
            pulses[f'MCMGE:{qubits}'] = meas_pulse
    
    else:
        if pulses is not None:
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
    q_meas = mcm.qubits[0]
    q_cond = mcm.properties['params']['apply']['1'].qubits

    # Initial barrier
    pulse.append(
        {'name': 'barrier',
         'scope': [f'Q{q}' for q in [q_meas] + [qc for qc in q_cond]]
        }
    )

    # Seqeuence to apply if 1 is measured
    true_apply = []
    true_apply.extend(
        cycle_pulse(config, mcm.properties['params']['apply']['1'])
    )

    # Sequence to apply if 0 is measured
    false_apply = []
    false_apply.extend(
        cycle_pulse(config, mcm.properties['params']['apply']['0'])
    )

    # Conditional operation
    pulse.append(
        {'name': 'branch_fproc', 
         'alu_cond': 'eq', 
         'cond_lhs': 1, 
         'func_id': f'Q{q_meas}.meas',
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
        'name':  'virtual_z',
        'freq':  config[f'single_qubit/{gate.qubits[0]}/{subspace}/freq'],
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
                {'name':  'virtual_z',
                 'freq':  config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'phase': pulse['kwargs']['phase']
                }
            )
            
        else:
            sq_pulse.append(
                {'name':   'pulse',
                 'tag':    f'{name} {subspace}',
                 'dest':   pulse['channel'],
                 'freq':   config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'amp':    clip_amplitude(pulse['kwargs']['amp']),
                 'phase':  pulse['kwargs']['phase'],
                 'twidth': pulse['length'],
                 'env':    pulse_envelopes[pulse['env']](
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
        for q in sub_config['qubits']:
            add_dynamical_decoupling(
                config,
                sub_config['method'],
                q,
                config[f'two_qubit/{qubits}/{name}/pulse'][idx]['length'],
                sub_config['n_pulses'],
                mq_pulse
            )

    for pulse in config[f'two_qubit/{qubits}/{name}/pulse']:

        if isinstance(pulse, str):
            name = pulse.split('/')[-2] + pulse.split('/')[-3]
            freq = config['/'.join(pulse.split('/')[:-2]) +'/freq']
            for p in config[pulse]:
                if p['env'] == 'virtualz':
                    mq_pulse.append(
                        {'name':  'virtual_z',
                         'freq':  freq,
                         'phase': p['kwargs']['phase']
                        }
                    )
                else:
                    mq_pulse.append(
                        {'name':  'pulse',
                         'tag':    name,
                         'dest':   p['channel'], 
                         'freq':   freq,
                         'amp':    clip_amplitude(p['kwargs']['amp']),
                         'phase':  p['kwargs']['phase'],
                         'twidth': p['length'], 
                         'env':    pulse_envelopes[p['env']](
                                    p['length'],
                                    config['hardware/DAC_sample_rate'],
                                    **{key: val for key, val 
                                       in p['kwargs'].items() 
                                       if key not in ['amp', 'phase']
                                      }
                                )
                        }
                    )
        
        elif pulse['env'] == 'virtualz':
            mq_pulse.append(
                {'name':  'virtual_z',
                 'freq':  config[pulse['freq']],
                 'phase': pulse['kwargs']['phase']
                }
            )

        else:
            mq_pulse.append(
                {'name':  'pulse',
                 'tag':    name,
                 'dest':   pulse['channel'], 
                 'freq':   config[f'two_qubit/{qubits}/{name}/freq'],
                 'amp':    clip_amplitude(pulse['kwargs']['amp']),
                 'phase':  pulse['kwargs']['phase'],
                 'twidth': pulse['length'], 
                 'env':    pulse_envelopes[pulse['env']](
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
                {'name':  'delay',
                 't':     gate.properties['params']['duration'], 
                 'qubit': [f'Q{qubit}']
                }
            )

        elif isinstance(gate, (Rz, VirtualZ, Z)):
            pulse.append(
                {'name':  'virtual_z',
                 'freq':  config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'phase': gate.properties['params']['phase']
                }
            )

        elif isinstance(gate, X):
            pulse.extend([
                {'name':   'pulse',
                 'tag':    f'X {subspace}',
                 'dest':   p['channel'], 
                 'freq':   config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'amp':    clip_amplitude(p['kwargs']['amp']),
                 'phase':  p['kwargs']['phase'],
                 'twidth': p['length'],
                 'env':    pulse_envelopes[p['env']](
                                p['length'],
                                config['hardware/DAC_sample_rate'],
                                **{key: val for key, val in p['kwargs'].items() 
                                if key not in ['amp', 'phase']}
                )} for p in config[f'single_qubit/{qubit}/{subspace}/X/pulse']
            ])

        elif isinstance(gate, Meas):
            pulse.extend([
                {'name':   'pulse',
                 'tag':    'Readout',
                 'dest':   f'Q{qubit}.rdrv',
                 'freq':   config[f'readout/{qubit}/freq'],
                 'amp':    config[f'readout/{qubit}/amp'], 
                 'phase':  0.0,
                 'twidth': config[f'readout/{qubit}/length'],
                 'env':    pulse_envelopes[config.readout[qubit].env](
                                config[f'readout/{qubit}/length'],
                                config['readout/sample_rate']
                           )
                },
                {'name': 'delay',
                 't':     config[f'readout/{qubit}/delay'],
                 'qubit': [f'Q{qubit}.rdlo']
                },
                {'name':   'pulse',
                 'tag':    'Demodulation',
                 'dest':   f'Q{qubit}.rdlo',
                 'freq':   config[f'readout/{qubit}/freq'],
                 'amp':    1.0,
                 'phase':  config[f'readout/{qubit}/phase'],  # Rotation in IQ plane
                 'twidth': config[f'readout/{qubit}/demod_time'],
                 'env':    pulse_envelopes[config.readout[qubit].env](
                                config[f'readout/{qubit}/demod_time'],
                                config['readout/sample_rate']
                           )
                }
            ])

    return pulse


def transpilation_error(*args):
    """Generic transpilation error.

    Raises:
        Exception: transpilation error for non-native gate.
    """
    raise Exception(
        f'Cannot transpile {args[1].name} (non-native gate)!.'
    ) 


def to_qubic(
        config:             Config, 
        circuit:            Circuit, 
        gate_mapper:        defaultdict,
        pulses:             defaultdict,
        qubit_reset:        bool = True,
        hardware_vz_qubits: List[str] = [],
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
        hardware_vz_qubits (List[str], optional): list of qubit labels
            specifying for which qubits should the virtualz gates be done
            on hardware (as opposed to software). Defaults to None. This is
            necessary if doing conditional phase shifts using mid-
            circuit measurements. Example: ```measure_qubits = ['Q0', 'Q1', 
            'Q3']```.

    Returns:
        List: transpiled qubic circuit.
    """
    qubic_circuit = []
    
    # Specify virtual_z gates to be done on hardware
    for q in hardware_vz_qubits:
        qubic_circuit.extend([
            {'name':  'declare', 
             'var':   f'{q}_phase', 
             'dtype': 'phase', 
             'scope': [q]
            },
            {'name': 'set_var', 'value': 0, 'var': f'{q}_phase'},
            {'name': 'bind_phase',
             'freq': config[f'single_qubit/{q[1:]}/GE/freq'],  # TODO
             'var':  f'{q}_phase'
            }
        ])

    # Add reset to the beginning of the circuit
    if qubit_reset:
        add_reset(config, circuit.qubits, qubic_circuit, pulses)

        # Add (optional) readout heralding
        if config.parameters['readout']['herald']:
            add_heralding(config, circuit.qubits, qubic_circuit, pulses)

    for cycle in circuit.cycles:

        if not cycle.is_barrier:
            qubic_circuit.append(
               {'name': 'barrier', 'qubit': [f'Q{q}' for q in circuit.qubits]},
            )
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
            # qubits = cycle.qubits if cycle.qubits else circuit.qubits
            qubic_circuit.append(
               {'name': 'barrier', 'qubit': [f'Q{q}' for q in circuit.qubits]},
            )

    return qubic_circuit


class Transpiler:
    """qcal to QubiC Transpiler."""

    # __slots__ = (
    #     '_config', '_gate_mapper', '_reload_pulse', '_qubit_reset', '_pulses'
    # )

    def __init__(self, 
            config:             Config, 
            gate_mapper:        defaultdict | None = None,
            reload_pulse:       bool = True,
            qubit_reset:        bool = True,
            hardware_vz_qubits: List[str] = [],
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
            hardware_vz_qubits (List[str], optional): list of qubit labels
                specifying for which qubits should the virtualz gates be done
                on hardware (as opposed to software). Defaults to None. This is
                necessary if doing conditional phase shifts using mid-
                circuit measurements. Example: ```measure_qubits = ['Q0', 'Q1', 
                'Q3']```.

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
        self._hardware_vz_qubits = hardware_vz_qubits
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
                    self._config[param[7:]] = circuits[param].iloc[i]
            
            transpiled_circuits.append(
                to_qubic(
                    self._config, 
                    circuit, 
                    self._gate_mapper, 
                    self._pulses,
                    self._qubit_reset,
                    self._hardware_vz_qubits
                )
            )
              
        if params:
            self._config.reload()  # Reload after making all the changes

        return transpiled_circuits