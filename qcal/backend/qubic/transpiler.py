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
    Id, Idle, Meas, MCM, Reset, Rz, VirtualZ, X, X90, Z
)
from qcal.sequence.dynamical_decoupling import dd_sequences
from qcal.sequence.pulse_envelopes import pulse_envelopes
from qcal.sequence.utils import clip_amplitude
from qcal.backend.qubic.utils import generate_pulse_env

import copy
import logging
import numpy as np
import operator

from collections import defaultdict
from functools import reduce
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


__all__ = ('to_qubic', 'Transpiler')


def get_by_path(root: Dict | defaultdict, items: List[Any]) -> Any:
    """Access a nested object in root by item sequence.

    Args:
        root ((Dict | defaultdict): nested dictionary.
        items (List[Any]): list of keys.

    Returns:
        Any: item in the nested dictionary indexed by the list of keys.
    """
    return reduce(operator.getitem, items, root)


def set_by_path(root: Dict | defaultdict, items: List[Any], value: Any):
    """Set a value in a nested object in root by item sequence.

    Args:
        root (Dict | defaultdict): nested dictionary.
        items (List[Any]): list of keys.
        value (Any): value to set for root indexed by the list of keys.
    """
    get_by_path(root, items[:-1])[items[-1]] = value


def readout_time(config: Config, qubit: int) -> float:
    """Compute the total readout time for a given qubit.

    Args:
        config (Config): qcal Config object.
        qubit (int):     qubit label.

    Returns:
        float: total readout time (readout + demod - integration delay)
    """
    time = (
        config[f'readout/{qubit}/demod/delay'] + 
        config[f'readout/{qubit}/demod/time']
    )
    return time


def initialize(config: Config, circuit: List, ) -> None:
    """Initialize DC pulses at the beginning of a sequence.

    Args:
        config (Config): qcal Config object.
        circuit (List):  qubic circuit.
    """
    for ch in config['initialize']:
        circuit.append(
            {'name':   'pulse', 
             'env':    None, 
             'freq':   None, 
             'amp':    config[f'initialize/{ch}/amp'],
             'twidth': config[f'initialize/{ch}/time'], 
             'dest':   ch
            }
        )
    circuit.append({'name': 'barrier'})


def deactivate(config: Config, circuit: List, ) -> None:
    """Deactivate DC pulses at the end of a sequence.

    Args:
        config (Config): qcal Config object.
        circuit (List):  qubic circuit.
    """
    circuit.append({'name': 'barrier'})
    for ch in config['initialize']:
        circuit.append(
            {'name':   'pulse', 
             'env':    None, 
             'freq':   None, 
             'amp':    0.,
             'twidth': 0., 
             'dest':   ch
            }
        )


def add_reset(
       config:          Config, 
       qubits_or_reset: List | Tuple | Reset, 
       circuit:         List, 
       pulses:          defaultdict
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

    reset_circuit = []
    if isinstance(qubits_or_reset, Reset):
        if 'active' in qubits_or_reset.properties['params']['method']:
            for n in range(config['reset/active/n_resets']):
                for q in qubits:
                    if (
                        n==0 and 
                        qubits_or_reset.properties['params']['measure_first']
                    ):
                        add_measurement(
                            config, 
                            qubits_or_reset.properties['params']['meas'], 
                            reset_circuit, 
                            None,
                            reset=True
                        )
                    elif n > 0:
                        add_measurement(
                            config, 
                            qubits_or_reset.properties['params']['meas'], 
                            reset_circuit, 
                            None,
                            reset=True
                        )

                    add_active_reset(config, q, reset_circuit)

                reset_circuit.append(
                    {'name': 'barrier', 
                     'qubit': [f'Q{q}' for q in qubits]
                    }
                )

        if 'unconditional' in qubits_or_reset.properties['params']['method']:
            add_unconditional_reset(config, qubits, reset_circuit)

        # Add the reset pulse to the hash table; TODO: check this works for all
        # pulses[f'ResetGE:{qubits}'] = reset_circuit

    else:
        if config['reset/passive/enable']:
            add_passive_reset(config, qubits, reset_circuit)

        if config['reset/active/enable']:
            for n in range(config['reset/active/n_resets']):
                for q in qubits:
                    add_measurement(config, q, reset_circuit, None, reset=True)
                    add_active_reset(config, q, reset_circuit)

                reset_circuit.append(
                    {'name': 'barrier', 
                     'qubit': [f'Q{q}' for q in qubits]
                    }
                )

        if config[f'reset/unconditional/enable']:
            add_unconditional_reset(config, qubits, reset_circuit)
            
    circuit.extend(reset_circuit)


def add_passive_reset(
       config: Config, qubits: List[int] | Tuple[int], circuit: List,
    ) -> None:
    """Add passive reset.

    Args:
        config (Config):       qcal Config object.
        qubits (List | Tuple): qubits to reset.
        circuit (List):        reset sub-circuit.
    """
    circuit.extend([
        {'name':  'delay',
          't':     config['reset/passive/delay'],
          'qubit': [f'Q{q}' for q in qubits]
        },
        {'name': 'barrier', 
         'qubit': [f'Q{q}' for q in qubits]
        }
    ])


def add_active_reset(
       config: Config, qubit: int, circuit: List,
    ) -> None:
    """Add active reset.

    Args:
        config (Config): qcal Config object.
        qubit (int):     qubits to reset.
        circuit (List):  reset sub-circuit.
    """
    # Reset pulse w/ qutrit reset
    reset_q_pulse = []
    if config['readout/esp/enable']:
        if qubit in config['readout/esp/qubits']:
            reset_q_pulse.extend(
                cycle_pulse(
                    config, Cycle({X(qubit, subspace='EF')})
                )
            )
            reset_q_pulse.extend(
                cycle_pulse(
                    config, Cycle({X(qubit, subspace='GE')})
                )
            )
        else:
            reset_q_pulse.extend(
                cycle_pulse(
                    config, Cycle({X(qubit, subspace='GE')})
                )
            )
            reset_q_pulse.extend(
                cycle_pulse(
                    config, Cycle({X(qubit, subspace='EF')})
                )
            )
    else:
        reset_q_pulse.extend(
            cycle_pulse(config, Cycle({X(qubit, subspace='GE')}))
        )

    circuit.append(
        {'name': 'barrier', 'scope': [f'Q{qubit}']}
    )
    circuit.append(
        {'name':     'branch_fproc',
            'alu_cond': 'eq',
            'cond_lhs': 1,
            'func_id':  f'Q{qubit}.meas',
            'scope':    [f'Q{qubit}'],
            'true':     reset_q_pulse,
            'false':    []
        },
    )


def add_unconditional_reset(
       config: Config, qubits: List[int] | Tuple[int], circuit: List,
    ) -> None:
    """Add unconditional reset.

    Args:
        config (Config):       qcal Config object.
        qubits (List | Tuple): qubits to reset.
        circuit (List):        reset sub-circuit.
    """
    for q in qubits:
        freq = config[f'reset/unconditional/{q}/freq']
        pulse = config[f'reset/unconditional/{q}/pulse']
        for p in pulse:
            circuit.append(
                {'name':   'pulse',
                 'tag':    'Reset',
                 'dest':   p['channel'], 
                 'freq':   freq,
                 'amp':    clip_amplitude(p['kwargs']['amp']),
                 'phase':  p['kwargs']['phase'],
                 'twidth': p['time'],
                 'env':    generate_pulse_env(
                                config=config,
                                pulse=p,
                                include_amp_phase=False
                            )
                }
            )

    circuit.append(
        {'name': 'barrier', 
         'qubit': [f'Q{q}' for q in qubits]
        }
    )


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
             't':     config['readout/resonator_reset'],
             'qubit': [f'Q{q}' for q in qubits]},
            {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]}
         ))
         

def add_measurement(
        config:        Config, 
        qubit_or_meas: int | Gate,
        circuit:       List, 
        pulses:        defaultdict | None,
        reset:         bool = False
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

    meas_pulse = []
    # Excited state promotion
    if config.parameters['readout']['esp']['enable'] and not reset:
        for qubit in qubits:
            if qubit in config.parameters['readout']['esp']['qubits']:
                meas_pulse.extend(
                    cycle_pulse(config, Cycle({X(qubit, subspace='EF')}))
                )
                # meas_pulse.extend(
                #     cycle_pulse(config, Cycle({X90(qubit, subspace='EF')}))
                # )
                # meas_pulse.extend(
                #     cycle_pulse(config, Cycle({X90(qubit, subspace='EF')}))
                # )
            else:
                time = 0.
                for pulse in config[f'single_qubit/{qubit}/EF/X/pulse']:
                    time += pulse['time']
                meas_pulse.append(
                    {'name':  'delay',
                     't':     time, 
                     'qubit': [f'Q{qubit}']
                    }
                )

    # Barrier before readout
    meas_pulse.append(
        {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]}
    )

    for qubit in qubits:
        meas_pulse.extend(cycle_pulse(config, Cycle({Meas(qubit)})))

        # if reset:  # If measurement is for active reset, don't save results
        #     meas_pulse[-1]['save_result'] = False

    # Barrier after readout
    meas_pulse.append(
        {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]}
    )

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
        # if pulses is not None:
        #     pulses[f'MCMGE:{qubits}'] = meas_pulse
    
    elif not reset and pulses is not None:
        pulses[f'MeasGE:{qubits}'] = meas_pulse
    
    circuit.extend(meas_pulse)


def add_dynamical_decoupling(
        config:      Config, 
        dd_method:   str, 
        qubit:       int, 
        time:        float, 
        n_dd_pulses: int, 
        pulse: List
    ) -> None:
    """Add dynamical decoupling pulses to a sequence.

    Args:
        config (Config): qcal Config object.
        dd_method (str): dynamical decoupling method.
        qubit (int): qubit label.
        time (float): time of time over which to perform the DD.
        n_dd_pulses (int): number of dynamical decoupling pulses.
        pulse (List): qubic pulse.
    """
    dd_circuit = dd_sequences[dd_method](config, qubit, time, n_dd_pulses)
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
    q_meas = mcm.qubits
    q_cond = set()
    for btstr in mcm.properties['params']['apply'].keys():
        q_cond.update( mcm.properties['params']['apply'][btstr].qubits)
    q_cond = tuple(sorted(q_cond))
    scope = sorted(list(set([f'Q{q}' for q in q_meas + q_cond])))
    branch_fproc = {
        'name': 'branch_fproc', 
        'alu_cond': 'eq', 
        'cond_lhs': 1,  # 1 == 'Q.meas', 1 = LHS of 'eq'
        'func_id': None,  # f'Q{q_meas}.meas',
        'scope': [f'Q{q}' for q in q_cond],
        'true': [],
        'false': []    
    }
    mapper = {'0': 'false', '1': 'true'}

    # Initial barrier
    pulse.append({'name': 'barrier', 'scope': scope})

    mcm_pulse = copy.deepcopy(branch_fproc)
    for btstr in mcm.properties['params']['apply'].keys():
        depth = []
        for i, bit in enumerate(btstr):
            if get_by_path(mcm_pulse, depth + ['func_id']) is None:
                set_by_path(
                    mcm_pulse, depth + ['func_id'], f'Q{q_meas[i]}.meas'
                )
            depth.append(mapper[bit])
            if i < len(btstr)-1:
                if not isinstance(get_by_path(mcm_pulse, depth), dict):
                    set_by_path(mcm_pulse, depth, copy.deepcopy(branch_fproc))
            else:
                apply = []
                if isinstance(
                    mcm.properties['params']['apply'][btstr], Circuit):
                    for cycle in mcm.properties['params']['apply'][btstr]:
                        apply.extend(cycle_pulse(config, cycle))
                else:
                    apply.extend(
                        cycle_pulse(
                            config, mcm.properties['params']['apply'][btstr]
                        )
                    )
                set_by_path(mcm_pulse, depth, apply)

    # Ensure that each 'true' and 'false' statements are followed by lists
    for n in range(len(btstr), 0, -1):
        for btstr in mcm.properties['params']['apply'].keys():
            depth = []
            for i, bit in enumerate(btstr[:n]):
                depth.append(mapper[bit])
            if not isinstance(get_by_path(mcm_pulse, depth), list):
                set_by_path(
                    mcm_pulse,
                    depth,
                    [get_by_path(mcm_pulse, depth)]
                )

    # Conditional operation
    pulse.append(mcm_pulse)

    # Final barrier
    pulse.append({'name': 'barrier', 'scope': scope})


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

    if gate.name == 'Y90':
        name = 'X90'
        sq_pulse.append(
                {'name':  'virtual_z',
                 'freq':  config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'phase': np.pi/2
                }
            )
    elif gate.name == 'Y':
        name = 'X'
        sq_pulse.append(
                {'name':  'virtual_z',
                 'freq':  config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'phase': np.pi/2
                }
            )  

    for pulse in (
            config[f'single_qubit/{qubit}/{subspace}/{name}/pulse']
        ):

        if isinstance(pulse, str):  # Pre- or post-pulse
            add_pre_post_pulse(config, gate.qubits, pulse, sq_pulse)

        elif pulse['env'] == 'virtualz':
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
                 'twidth': pulse['time'],
                 'env':    generate_pulse_env(
                               config=config,
                               pulse=pulse,
                               include_amp_phase=False
                           )
                }
            )

    if gate.name == 'Y90':
        sq_pulse.append(
                {'name':  'virtual_z',
                 'freq':  config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'phase': -np.pi/2
                }
            )
    elif gate.name == 'Y':
        sq_pulse.append(
                {'name':  'virtual_z',
                 'freq':  config[f'single_qubit/{qubit}/{subspace}/freq'],
                 'phase': -np.pi/2
                }
            )

    pulses[f'{gate.name}{subspace}:{gate.qubits}'] = sq_pulse
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
        mq_pulse.append(
               {'name': 'barrier', 
                'qubit': [f'Q{q}' for q in sub_config['qubits']]
               },
            )
        idx = find_pulse_index(config, f'two_qubit/{qubits}/{name}/pulse')
        for q in sub_config['qubits']:
            add_dynamical_decoupling(
                config,
                sub_config['method'],
                q,
                config[f'two_qubit/{qubits}/{name}/pulse'][idx]['time'],
                sub_config['n_pulses'],
                mq_pulse
            )

    for i, pulse in enumerate(config[f'two_qubit/{qubits}/{name}/pulse']):

        if isinstance(pulse, str):  # Pre- or post-pulse
            add_pre_post_pulse(config, qubits, pulse, mq_pulse)
        
        elif pulse['env'] == 'virtualz':
            mq_pulse.append(
                {'name':  'virtual_z',
                 'freq':  config[pulse['freq']],
                 'phase': pulse['kwargs']['phase']
                }
            )

        else:
            if config[f'two_qubit/{qubits}/{name}/pulse/{i}/freq']:
                freq = config[f'two_qubit/{qubits}/{name}/pulse/{i}/freq']
            else:
                freq = config[f'two_qubit/{qubits}/{name}/freq']
        
            mq_pulse.append(
                {'name':  'pulse',
                 'tag':    name,
                 'dest':   pulse['channel'], 
                 'freq':   freq,
                 'amp':    clip_amplitude(pulse['kwargs']['amp']),
                 'phase':  pulse['kwargs']['phase'],
                 'twidth': pulse['time'], 
                 'env':    generate_pulse_env(
                               config=config,
                               pulse=pulse,
                               include_amp_phase=False
                           )
                }
            )

    pulses[f'{name}{gate.subspace}:{qubits}'] = mq_pulse
    circuit.extend(mq_pulse)


def add_pre_post_pulse(
        config: Config, qubits: Tuple, pulse: str, gate_pulse: List
    ) -> None:
    """Add a pre-pulse or post-pulse to a gate pulse.

    Args:
        config (Config):   qcal Config object.
        qubits (Tuple):    qubits involved in the gate.
        pulse (str):       pulse reference in the config.
        gate_pulse (List): pulse definition of a gate.
    """
    gate_pulse.append(
        {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]},
    )
    pname = pulse.split('/')[-2] + pulse.split('/')[-3]
    freq = config['/'.join(pulse.split('/')[:-2]) +'/freq']
    for p in config[pulse]:
        if p['env'] == 'virtualz':
            gate_pulse.append(
                {'name':  'virtual_z',
                 'freq':  freq,
                 'phase': p['kwargs']['phase']
                }
            )
        else:
            gate_pulse.append(
                {'name':  'pulse',
                    'tag':    pname,
                    'dest':   p['channel'], 
                    'freq':   freq,
                    'amp':    clip_amplitude(p['kwargs']['amp']),
                    'phase':  p['kwargs']['phase'],
                    'twidth': p['time'],
                    'env':    generate_pulse_env(
                                  config=config,
                                  pulse=p,
                                  include_amp_phase=False
                              )
                }
            )
    gate_pulse.append(
        {'name': 'barrier', 'qubit': [f'Q{q}' for q in qubits]},
    )


def cycle_pulse(config: Config, cycle: Cycle) -> List:
    """Generate a pulse from a cycle of operations.

    This is useful for generating sub-pulses for DD sequences or MCM.

    Args:
        config (Config): qcal Config object.
        cycle (Cycle):   cycle of gates.

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

        elif isinstance(gate, X90):
            for p in config[f'single_qubit/{qubit}/{subspace}/X90/pulse']:
                if p['env'] == 'virtualz':
                    pulse.append(
                        {'name':  'virtual_z',
                         'freq':  config[
                                    f'single_qubit/{qubit}/{subspace}/freq'
                                  ],
                         'phase': p['kwargs']['phase']
                        }
                    )
                else:
                    pulse.append(
                        {'name':   'pulse',
                         'tag':    f'X90 {subspace}',
                         'dest':   p['channel'],
                         'freq':  (
                              config[f'single_qubit/{qubit}/{subspace}/freq']
                         ),
                         'amp':    clip_amplitude(p['kwargs']['amp']),
                         'phase':  p['kwargs']['phase'],
                         'twidth': p['time'],
                         'env':    generate_pulse_env(
                                       config=config,
                                       pulse=p,
                                       include_amp_phase=False
                                   )
                        }
                    )

        elif isinstance(gate, X):
             for p in config[f'single_qubit/{qubit}/{subspace}/X/pulse']:
                if p['env'] == 'virtualz':
                    pulse.append(
                        {'name':  'virtual_z',
                         'freq':  config[
                                    f'single_qubit/{qubit}/{subspace}/freq'
                                  ],
                         'phase': p['kwargs']['phase']
                        }
                    )
                else:
                    pulse.append(
                        {'name':   'pulse',
                         'tag':    f'X {subspace}',
                         'dest':   p['channel'], 
                         'freq':   (
                                config[f'single_qubit/{qubit}/{subspace}/freq']
                            ),
                         'amp':    clip_amplitude(p['kwargs']['amp']),
                         'phase':  p['kwargs']['phase'],
                         'twidth': p['time'],
                         'env':    generate_pulse_env(
                                        config=config,
                                        pulse=p,
                                        include_amp_phase=False
                                   )
                        }
                    )

        elif isinstance(gate, Meas):
            pulse.extend([
                {'name':   'pulse',
                 'tag':    'Readout',
                 'dest':   f'Q{qubit}.rdrv',
                 'freq':   config[f'readout/{qubit}/freq'],
                 'amp':    config[f'readout/{qubit}/amp'], 
                 'phase':  0.0,
                 'twidth': config[f'readout/{qubit}/time'],
                 'env':    generate_pulse_env(
                               config=config,
                               pulse=config[f'readout/{qubit}'],
                               channel='rdrv'
                           )
                },
                {'name': 'delay',
                 't':     config[f'readout/{qubit}/demod/delay'],
                 'qubit': [f'Q{qubit}.rdlo']
                },
                {'name':   'pulse',
                 'tag':    'Demodulation',
                 'dest':   f'Q{qubit}.rdlo',
                 'freq':   config[f'readout/{qubit}/freq'],
                 'phase':  config[f'readout/{qubit}/demod/phase'],
                 'twidth': config[f'readout/{qubit}/demod/time'],
                 'env':    generate_pulse_env(
                               config=config,
                               pulse=config[f'readout/{qubit}/demod'],
                               channel='rdlo'
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
        hardware_vz_qubits: List[str] = [],
        circuit_for_loop:   bool = False,
    ) -> List:
    """Compile a qcal circuit to a qubic circuit.

    Args:
        config (Config): config object.
        circuit (Circuit): qcal circuit.
        gate_mapper (defaultdict): map between qcal to QubiC gates.
        pulses (defaultdict): pulses that have been stored for reuse.
        hardware_vz_qubits (List[str], optional): list of qubit labels
            specifying for which qubits should the virtualz gates be done
            on hardware (as opposed to software). Defaults to None. This is
            necessary if doing conditional phase shifts using mid-
            circuit measurements. Example: ```measure_qubits = ['Q0', 'Q1', 
            'Q3']```.
        circuit_for_loop (bool, optional): loops over circuit partitions for 
            circuits with repeated structures. Defaults to False.

    Returns:
        List: transpiled qubic circuit.
    """
    qubic_circuit = []
    
    # Qubic loop instruction
    if circuit_for_loop:
        qubic_circuit.append(
            {'name':  'declare', 
             'var':   'loop_ind',
             'dtype': 'int',
             'scope': [f'Q{q}' for q in circuit.qubits],
            }
        )
        if not hardware_vz_qubits:
            hardware_vz_qubits = [f'Q{q}' for q in circuit.qubits]
    
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
             'freq': config[f'single_qubit/{q[1:]}/GE/freq'],
             'var':  f'{q}_phase'
            }
        ])

        # if config[f'single_qubit/{q[1:]}/EF/freq']:
        #     qubic_circuit.extend([
        #     {'name':  'declare', 
        #      'var':   f'{q}_phase_EF', 
        #      'dtype': 'phase', 
        #      'scope': [q]
        #     },
        #     {'name': 'set_var', 'value': 0, 'var': f'{q}_phase_EF'},
        #     {'name': 'bind_phase',
        #      'freq': config[f'single_qubit/{q[1:]}/EF/freq'],
        #      'var':  f'{q}_phase_EF'
        #     }
        # ])

    if config._parameters['initialize']:
        initialize(config, qubic_circuit)

    # Add reset to the beginning of the circuit
    add_reset(config, circuit.qubits, qubic_circuit, pulses)

    # Add (optional) readout heralding
    if config.parameters['readout']['herald']:
        add_heralding(config, circuit.qubits, qubic_circuit, pulses)

    # Loop instruction for repeated subcircuits
    if circuit_for_loop:
        for sub_circuit, n_reps in circuit.partitions:
            if n_reps == 1:
                for cycle in sub_circuit:
                    if not cycle.is_barrier:
                        qubic_circuit.append(
                            {'name': 'barrier', 
                            #  'qubit': [f'Q{q}' for q in circuit.qubits]
                            }
                        )
                        for gate in cycle:
                            name = gate.name
                            if 'phase' in gate.properties['params'].keys():
                                name += str(gate.properties['params']['phase'])

                            if pulses[f'{name}{gate.subspace}:{gate.qubits}']:
                                qubic_circuit.extend(
                                    pulses[
                                        f'{name}{gate.subspace}:{gate.qubits}'
                                    ]
                                )

                            else:
                                gate_mapper[gate.name](
                                    config, gate, qubic_circuit, pulses
                                )

                    elif cycle.is_barrier:
                        qubits = (
                            cycle.qubits if cycle.qubits else circuit.qubits
                        )
                        qubic_circuit.append(
                            {'name': 'barrier', 
                             'qubit': [f'Q{q}' for q in qubits]
                            }
                        )

            elif n_reps > 1:
                qubic_circuit.append(
                    {'name': 'set_var', 'var': 'loop_ind', 'value': 0}
                )
                loop_circuit = []
                for cycle in sub_circuit:
                    if not cycle.is_barrier:
                        loop_circuit.append(
                            {'name': 'barrier', 
                             'qubit': [f'Q{q}' for q in circuit.qubits]
                            }
                        )
                        for gate in cycle:
                            name = gate.name
                            if 'phase' in gate.properties['params'].keys():
                                name += str(gate.properties['params']['phase'])

                            if pulses[f'{name}{gate.subspace}:{gate.qubits}']:
                                loop_circuit.extend(
                                    pulses[
                                        f'{name}{gate.subspace}:{gate.qubits}'
                                    ]
                                )

                            else:
                                gate_mapper[gate.name](
                                    config, gate, loop_circuit, pulses
                                )

                    elif cycle.is_barrier:
                        qubits = (
                            cycle.qubits if cycle.qubits else circuit.qubits
                        )
                        loop_circuit.append(
                            {'name': 'barrier', 
                             'qubit': [f'Q{q}' for q in qubits]
                            }
                        )

                loop_circuit.append(
                    {'name': 'alu', 
                     'lhs':  1, 
                     'op':   'add', 
                     'rhs':  'loop_ind', 
                     'out':  'loop_ind'
                    }
                )

                qubic_circuit.append(
                    {'name':    'loop', 
                     'cond_lhs': n_reps,
                     'alu_cond': 'ge', 
                     'cond_rhs': 'loop_ind',
                     'scope':    [f'Q{q}' for q in circuit.qubits],
                     'body':     loop_circuit
                    }
                )

    else:
        for cycle in circuit.cycles:

            if not cycle.is_barrier:
                qubic_circuit.append(
                    {'name': 'barrier', 
                    #  'qubit': [f'Q{q}' for q in circuit.qubits]
                    }
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
                        gate_mapper[gate.name](
                            config, gate, qubic_circuit, pulses
                        )

            elif cycle.is_barrier:
                qubits = cycle.qubits if cycle.qubits else circuit.qubits
                qubic_circuit.append(
                    {'name': 'barrier', 
                     'qubit': [f'Q{q}' for q in qubits]
                    }
                )

    if config._parameters['initialize']:
        deactivate(config, qubic_circuit)

    return qubic_circuit


class Transpiler:
    """qcal to QubiC Transpiler"""

    def __init__(self, 
            config:             Config, 
            gate_mapper:        defaultdict | None = None,
            hardware_vz_qubits: List[str] = [],
            circuit_for_loop:   bool = False,
            reload_pulse:       bool = True,
        ) -> None:
        """Initialize with a qcal Config object.

        Args:
            config (Config): qcal config object.
            gate_mapper (defaultdict | None, optional): dictionary which maps
                circuit gates to QubiC gates. Defaults to None.
            hardware_vz_qubits (List[str], optional): list of qubit labels
                specifying for which qubits should the virtualz gates be done
                on hardware (as opposed to software). Defaults to None. This is
                necessary if doing conditional phase shifts using mid-
                circuit measurements. Example: ```measure_qubits = ['Q0', 'Q1', 
                'Q3']```.
            circuit_for_loop (bool, optional): loops over circuit partitions for 
                circuits with repeated structures. Defaults to False.
            reload_pulse (bool, optional): reloads the stored pulses when 
                compiling each circuit. Defaults to True.
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
                 'Z90':      add_virtualz_gate,
                 'Y':        add_single_qubit_gate,
                 'Y90':      add_single_qubit_gate,
                }
            )
            for gate in config.native_gates['set']:
                if gate in qcal.gate.single_qubit.__all__:
                    self._gate_mapper[gate] = add_single_qubit_gate
                else:
                    self._gate_mapper[gate] = add_multi_qubit_gate
        else:
            self._gate_mapper = gate_mapper

        self._hardware_vz_qubits = hardware_vz_qubits
        self._circuit_for_loop = circuit_for_loop
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
        # Check for a param sweep
        params = [col for col in circuits._df.columns if 'param' in col]
        params_reset = {}
        if params:
            for param in params: # [7:] removes the string 'param: '
                 params_reset[param[7:]] = self._config[param[7:]]

        transpiled_circuits = []
        for i, circuit in enumerate(circuits):
            if self._reload_pulse:
                 self._pulses = defaultdict(lambda: False, {})

            if params:
                for param in params:
                    self._config[param[7:]] = circuits[param].iloc[i]
            
            transpiled_circuits.append(
                to_qubic(
                    config=self._config, 
                    circuit=circuit, 
                    gate_mapper=self._gate_mapper, 
                    pulses=self._pulses,
                    hardware_vz_qubits=self._hardware_vz_qubits,
                    circuit_for_loop=self._circuit_for_loop
                )
            )
              
        if params:
            logger.info(' Resetting params...')
            # self._config.reload()  # Reload after making all the changes
            for param, val in params_reset.items():
                self._config[param] = val

        return transpiled_circuits