""""Submodule for phase estimation experiments.

See:
https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.062315
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.190502

Relevant pyRPE code: https://gitlab.com/quapack/pyrpe
"""
import qcal.settings as settings

from qcal.config import Config
from qcal.math.utils import round_to_order_error
from qcal.qpu.qpu import QPU
from qcal.plotting.utils import calculate_nrows_ncols

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from collections.abc import Iterable
from IPython.display import clear_output
from matplotlib.lines import Line2D
from numpy.typing import NDArray, ArrayLike
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


def X90(theta):
    """Definition of an X gate.

    Args:
        theta (angle): angle of rotation.

    Returns:
        pygsti.unitary_to_pauligate: X gate.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    H = theta/2 * pygsti.sigmax
    U = scipy.linalg.expm(-1j * H)
    return pygsti.unitary_to_pauligate(U)


def CZ(theta_iz: float, theta_zi: float, theta_zz: float) -> NDArray:
    """Definition of a CZ gate.

    diag([1,1,1,-1]) == CZ(np.pi/2, np.pi/2, -np.pi/2) (up to phase).

    Args:
        theta_iz (float): IZ angle.
        theta_zi (float): ZI angle
        theta_zz (float): ZZ angle.

    Returns:
        NDArray: matrix exponential of a CZ gate.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    return scipy.linalg.expm(-1j /2 * (theta_iz * pygsti.sigmaiz + 
                                       theta_zi * pygsti.sigmazi + 
                                       theta_zz * pygsti.sigmazz))


def interleaved_circuit_depths(circuit_depths: List[int]) -> List[int]:
    """Circuit depths for interleaved X90 sequences.

    Args:
        circuit_depths (List[int]): circuit depths.

    Returns:
        List[int]: circuit depths with 1/4 the maximum depth.
    """
    max_depth = int(max(circuit_depths) / 4)
    idx = circuit_depths.index(max_depth)
    return circuit_depths[:idx+1]


def make_idle_circuits(circuit_depths: List[int], qubits: Tuple[int]):
    """Generate the idle RPE circuits.

    Args:
        circuit_depths (List[int]): circuit depths.
        qubits (Tuple[int]): qubit labels.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    circuits = (
        [make_idle_cos_circ(d, qubits) for d in circuit_depths] +
        [make_idle_sin_circ(d, qubits) for d in circuit_depths]
    )

    return pygsti.remove_duplicates(circuits)


def make_idle_cos_circ(d: int, qubits: Tuple[int] | List[int]):
    """Make the cosine circuit for idle RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    Gi_prep = pygsti.circuits.Circuit(
        [[('Gypi2', q) for q in qubits]], line_labels=qubits
    )
    Gi_germ = pygsti.circuits.Circuit(
        [[('Gidle', q) for q in qubits]], line_labels=qubits
    ) * d
    Gi_meas = pygsti.circuits.Circuit(
        [[('Gypi2', q) for q in qubits]], line_labels=qubits
    ) * 3

    return Gi_prep + Gi_germ + Gi_meas


def make_idle_sin_circ(d: int, qubits: Tuple[int] | List[int]):
    """Make the cosine circuit for idle RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    Gi_prep = pygsti.circuits.Circuit(
        [[('Gxpi2', q) for q in qubits]], line_labels=qubits
    )
    Gi_germ = pygsti.circuits.Circuit(
        [[('Gidle', q) for q in qubits]], line_labels=qubits
    ) * d
    Gi_meas = pygsti.circuits.Circuit(
        [[('Gypi2', q) for q in qubits]], line_labels=qubits
    ) * 3

    return Gi_prep + Gi_germ + Gi_meas


def make_x90_circuits(circuit_depths: List[int], qubits: Tuple[int]):
    """Generate the axis X90 RPE circuits.

    Args:
        circuit_depths (List[int]): circuit depths.
        qubits (Tuple[int]): qubit labels.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    circuits = (
        [make_x90_cos_circ(d, qubits) for d in circuit_depths] +
        [make_x90_sin_circ(d, qubits) for d in circuit_depths] +
        [make_interleaved_cos_circ(d, qubits) for d in circuit_depths] +
        [make_interleaved_sin_circ(d, qubits) for d in circuit_depths]
    )

    return pygsti.remove_duplicates(circuits)


def make_x90_cos_circ(d: int, qubits: Tuple[int] | List[int]):
    """Make the cosine circuit for X90 RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    # return pygsti.circuits.Circuit([[('Gxpi2', q) for q in qubits]]) * d
    return pygsti.circuits.Circuit(
        [[('Gxpi2', q) for q in qubits]], line_labels=qubits
    ) * d


def make_x90_sin_circ(d: int, qubits: Tuple[int] | List[int]):
    """Make the sine circuit for X90 RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    return pygsti.circuits.Circuit(
        [[('Gxpi2', q) for q in qubits]], line_labels=qubits
    ) * (d + 1)


def make_interleaved_cos_circ(d: int, qubits: Tuple[int] | List[int]):
    """Make the interleaved cosine circuit for X90 axis error RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    Gz_layer = pygsti.circuits.Circuit(
        [[('Gzpi2', q) for q in qubits]], line_labels=qubits
    )
    Gx_layer = pygsti.circuits.Circuit(
        [[('Gxpi2', q) for q in qubits]], line_labels=qubits
    )
    return (
        Gz_layer + Gx_layer + Gx_layer + Gz_layer + Gz_layer + Gx_layer + 
        Gx_layer + Gz_layer
    ) * d


def make_interleaved_sin_circ(d: int, qubits: Tuple[int] | List[int]):
    """Make the interleaved sine circuit for X90 axis error RPE.

    Args:
        d (int): circuit depth.
        qubits (Tuple[int] | List[int]): qubit labels.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    Gz_layer = pygsti.circuits.Circuit(
        [[('Gzpi2', q) for q in qubits]], line_labels=qubits
    )
    Gx_layer = pygsti.circuits.Circuit(
        [[('Gxpi2', q) for q in qubits]], line_labels=qubits
    )
    return (
        Gz_layer + Gx_layer + Gx_layer + Gz_layer + Gz_layer + Gx_layer + 
        Gx_layer + Gz_layer
    ) * d + Gx_layer


def make_cz_circuits(
        circuit_depths: List[int], qubit_pairs: List[Tuple[int]]
    ) -> List:
    """Generate CZ RPE circuits.

    Args:
        circuit_depths (List[int]): circuit depths.
        qubit_pairs (List[Tuple[int]]): pairs of qubits for two-qubit gates.

    Returns:
        List: list of pyGSTi circuits.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    state_pairs = [(0, 1), (2, 3), (3, 1)]
    sin_dict = {
        state_pair: {
            i: make_cz_sin_circ(
                i, state_pair, qubit_pairs
            ) for i in circuit_depths
        } for state_pair in state_pairs
    }
    cos_dict = {
        state_pair: {
            i: make_cz_cos_circ(
                i, state_pair, qubit_pairs
            ) for i in circuit_depths
        } for state_pair in state_pairs
    }

    circuits = []
    for trig_dict in [sin_dict, cos_dict]:
        for state_pair in state_pairs:
            circuits += list(trig_dict[state_pair].values())
    
    return pygsti.remove_duplicates(circuits)


def make_cz_cos_circ(
        d: int, state_pair: Tuple[int], qubit_pairs: List[Tuple[int]]
    ):
    """Make the cosine circuit for CZ RPE.

    Args:
        d (int): circuit depth.
        state_pair (Tuple[int]): state pair.
        qubit_pairs (List[Tuple[int]]): pairs of qubits for two-qubit gates.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    line_labels = []
    for qubit_pair in qubit_pairs:
        line_labels.extend(qubit_pair)

    # <01| for 1/2 (1+cos(1/2 (theta_iz + theta_zz))
    # <00| for 1/2 (1-cos(1/2 (theta_iz + theta_zz))
    if state_pair in [(0, 1), (1, 0)]:
       circ = (
           pygsti.circuits.Circuit(
               [[('Gypi2', qp[1]) for qp in qubit_pairs]], 
               line_labels=line_labels
           ) +
           pygsti.circuits.Circuit(
               [[('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]]
           ) * d +
           pygsti.circuits.Circuit(
               [[('Gypi2', qp[1]) for qp in qubit_pairs]]
           )
       )
    
    # <11| for 1/2 (1+cos(1/2 (theta_iz - theta_zz))
    elif state_pair in [(2, 3), (3, 2)]:
        circ = (
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]], 
                line_labels=line_labels
            ) +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[1]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]]
            ) * d +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[1]) for qp in qubit_pairs]]
            )
        )
    
    # <11| for 1/2 (1+cos(1/2 (theta_zi - theta_zz))
    elif state_pair in [(1, 3), (3, 1)]:
        circ = (
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]], 
                line_labels=line_labels
            ) +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[0]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]]
            ) * d +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[0]) for qp in qubit_pairs]]
            )
        )
    
    else:
        assert False, (
             "state_pair must be in [(0,1), (1,0), (2,3), (3,2), (1,3), (3,1)]"
        )

    return circ

    
def make_cz_sin_circ(
        d: int, state_pair: Tuple[int], qubit_pairs: List[Tuple[int]]
    ):
    """Make the sine circuit for CZ RPE.

    Args:
        d (int): circuit depth.
        state_pair (Tuple[int]): state pair.
        qubit_pairs (List[Tuple[int]]): pairs of qubits for two-qubit gates.

    Returns:
       pygsti.circuits.Circuit: pyGSTi circuit.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')
    
    line_labels = []
    for qubit_pair in qubit_pairs:
        line_labels.extend(qubit_pair)

    # <00| for 1/2 (1+sin(1/2 (theta_iz + theta_zz))
    if state_pair in [(0, 1), (1, 0)]:
        circ = (
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[1]) for qp in qubit_pairs]], 
                line_labels=line_labels
            ) +
            pygsti.circuits.Circuit(
                [[('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]]
            ) * d +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            )
        ) 
    
    # <10| for 1/2 (1+sin(1/2 (theta_iz - theta_zz))
    elif state_pair in [(2, 3),(3, 2)]:
        circ = (
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]], 
                line_labels=line_labels
            ) +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[1]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]]
            ) * d +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            )
        )

    # <01| for 1/2 (1+sin(1/2 (theta_zi - theta_zz))    
    elif state_pair in [(1, 3), (3, 1)]:
        circ = (
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]], 
                line_labels=line_labels
            ) +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[1]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gypi2', qp[0]) for qp in qubit_pairs]]
            ) +
            pygsti.circuits.Circuit(
                [[('Gcphase', qp[0], qp[1]) for qp in qubit_pairs]]
            ) * d +
            pygsti.circuits.Circuit(
                [[('Gxpi2', qp[0]) for qp in qubit_pairs]]
            )
        )
    else:
        assert False, (
            "state_pair must be in [(0,1), (1,0), (2,3), (3,2), (1,3), (3,1)]"
        )
    return circ


def analyze_idle(
        dataset, 
        qubits: List[int], 
        circuit_depths: List[int]
    ) -> Tuple:
    """Analyze idle RPE dataset.

    Args:
        dataset (pygsti.data.dataset): pyGSTi dataset.
        qubits (List[int]): qubit labels.
        circuit_depths (List[int]): circuit depths.

    Returns:
        Tuple: angle estimates, angle errors, and index of last good depth
    """
    try:
        from quapack.pyRPE import RobustPhaseEstimation
        from quapack.pyRPE.quantum import Q
    except ImportError:
        logger.warning(' Unable to import pyRPE!')

    cos_circs = {
        d: make_idle_cos_circ(d, qubits) for d in circuit_depths
    }
    sin_circs = {
        d: make_idle_sin_circ(d, qubits) for d in circuit_depths
    }

    signal = {'ramsey': []}
    # Angle estimate
    experiment = Q()
    for d in circuit_depths:
        cos_counts = dataset[cos_circs[d]].counts
        sin_counts = dataset[sin_circs[d]].counts
        experiment.process_cos(d,
            (int(cos_counts['0']), int(cos_counts['1']))
        )
        experiment.process_sin(d,
            (int(sin_counts['1']), int(sin_counts['0']))
        )
        p_I = int(cos_counts['0']) / (
            int(cos_counts['0']) + int(cos_counts['1'])
        )
        p_Q = int(sin_counts['1']) / (
            int(sin_counts['0']) + int(sin_counts['1'])
        )
        signal['ramsey'].append(1 - 2 * p_I + 1j - 2j * p_Q)

    analysis = RobustPhaseEstimation(experiment)
    angle_estimates = analysis.angle_estimates
    angle_estimates = {'Z': np.array([
        rectify_angle(angle) for angle in angle_estimates
    ])}
    angle_errors = {'Z': angle_estimates['Z']}
    last_good_idx = analysis.check_unif_local(historical=True)

    return (angle_estimates, angle_errors, last_good_idx, signal)


def analyze_x90(
        dataset, 
        qubits: List[int], 
        circuit_depths: List[int], 
        estimator_type: str = 'linearized'
    ) -> Tuple:
    """Analyze RPE dataset for the X90 gate.

    Args:
        dataset (pygsti.data.dataset): pyGSTi dataset.
        qubits (List[int]): qubit labels.
        circuit_depths (List[int]): circuit depths.
        estimator_type (str): type of estimator. Defaults to 'linearized'.

    Returns:
        Tuple: angle estimates, angle errors, and index of last good depth
    """
    try:
        from quapack.pyRPE import RobustPhaseEstimation
        from quapack.pyRPE.quantum import Q
    except ImportError:
        logger.warning(' Unable to import pyRPE!')

    target_x = np.pi / 2

    direct_cos_circs = {
        d: make_x90_cos_circ(d, qubits) for d in circuit_depths
    }
    direct_sin_circs = {
        d: make_x90_sin_circ(d, qubits) for d in circuit_depths
    }

    interleaved_cos_circs = {
        d: make_interleaved_cos_circ(d, qubits) for d in circuit_depths
    }
    interleaved_sin_circs = {
        d: make_interleaved_sin_circ(d, qubits) for d in circuit_depths
    }

    signal = {'direct': [], 'interleaved': []}
    # Direct angle estimates
    experiment = Q()
    for d in circuit_depths:
        direct_cos_counts = dataset[direct_cos_circs[d]].counts
        direct_sin_counts = dataset[direct_sin_circs[d]].counts
        experiment.process_cos(d,
            (int(direct_cos_counts['0']), int(direct_cos_counts['1']))
        )
        experiment.process_sin(d,
            (int(direct_sin_counts['1']), int(direct_sin_counts['0']))
        )
        p_I = int(direct_cos_counts['0']) / (
            int(direct_cos_counts['0']) + int(direct_cos_counts['1'])
        )
        p_Q = int(direct_sin_counts['1']) / (
            int(direct_sin_counts['0']) + int(direct_sin_counts['1'])
        )
        signal['direct'].append(1 - 2 * p_I + 1j - 2j * p_Q)
    analysis = RobustPhaseEstimation(experiment)
    direct_angle_estimates = analysis.angle_estimates
    direct_angle_estimates = np.array([
        rectify_angle(angle) for angle in direct_angle_estimates
    ])
    direct_last_good_idx = analysis.check_unif_local(historical=True)

    # Interleaved angle estimates
    experiment = Q()
    for d in circuit_depths:  #interleaved_circuit_depths(circuit_depths):
        interleaved_cos_counts = dataset[interleaved_cos_circs[d]].counts
        interleaved_sin_counts = dataset[interleaved_sin_circs[d]].counts
        experiment.process_cos(d,
            (int(interleaved_cos_counts['0']), 
             int(interleaved_cos_counts['1'])
            )
        )
        experiment.process_sin(d,
            (int(interleaved_sin_counts['1']), 
             int(interleaved_sin_counts['0'])
            )
        )
        p_I = int(interleaved_cos_counts['0']) / (
            int(interleaved_cos_counts['0']) + int(interleaved_cos_counts['1'])
        )
        p_Q = int(interleaved_sin_counts['1']) / (
            int(interleaved_sin_counts['0']) + int(interleaved_sin_counts['1'])
        )
        signal['interleaved'].append(1 - 2 * p_I + 1j - 2j * p_Q)

    analysis = RobustPhaseEstimation(experiment)
    interleaved_angle_estimates = analysis.angle_estimates
    interleaved_angle_estimates = np.array([
        rectify_angle(angle) for angle in interleaved_angle_estimates
    ])
    interleaved_last_good_idx = analysis.check_unif_local(historical=True)

    if estimator_type == 'linearized':
        epsilon_estimates = direct_angle_estimates / (np.pi/2) - 1
        theta_estimates = np.array([
            np.sin(interleaved_angle_estimates[i]/2) / 
            (2 * np.cos(np.pi * epsilon_estimates[i]/2)) 
            # (2 * np.cos(np.pi * epsilon_estimates[direct_last_good_idx]/2)) 
            for i in range(len(direct_angle_estimates))
        ])

    # angle_estimates = {
    #     'rotation': target_x*(1+epsilon_estimates),
    #     'off axis': theta_estimates
    # }
    angle_estimates = {
        'X': target_x * (1 + epsilon_estimates) * np.cos(theta_estimates),
        'Z': target_x * (1 + epsilon_estimates) * np.sin(theta_estimates),
        # 'X': target_x * (
        #         1 + epsilon_estimates[direct_last_good_idx]
        #     ) * np.cos(theta_estimates),
        # 'Z': target_x * (
        #         1 + epsilon_estimates[direct_last_good_idx]
        #     ) * np.sin(theta_estimates),
    }

    # Extract the last "trusted" RPE angle estimate
    last_good_idx = min([direct_last_good_idx, interleaved_last_good_idx])

    # angle_errors = {
    #     'rotation': target_x * epsilon_estimates,
    #     'off axis': theta_estimates
    # }
    angle_errors = {
        'X': angle_estimates['X'] - target_x,
        'Z': angle_estimates['Z']
    }

    return (angle_estimates, angle_errors, last_good_idx, signal)
        

def analyze_cz(
        dataset, qubit_pairs: List[Tuple[int]], circuit_depths: List[int]
    ) -> Tuple:
    """Analyze RPE dataset for the CZ gate.

    Args:
        dataset (pygsti.data.dataset): pyGSTi dataset.
        qubit_pairs (List[Tuple[int]]): qubit labels.
        circuit_depths (List[int]): circuit depths.

    Returns:
        Tuple: angle estimates, angle errors, and index of last good depth
    """
    try:
        from quapack.pyRPE import RobustPhaseEstimation
        from quapack.pyRPE.quantum import Q
    except ImportError:
        logger.warning(' Unable to import pyRPE!')

    target_zz = -np.pi/2
    target_iz = target_zi = np.pi/2

    state_pairs = [(0, 1), (2, 3), (3, 1)]
    state_pair_lookup = {
        (0, 1):{
            ('cos', '+'): '01', ('cos', '-'): '00',
            ('sin', '+'): '00', ('sin', '-'): '01'
        },
        (2, 3):{
            ('cos', '+'): '11', ('cos', '-'): '10',
            ('sin', '+'): '10', ('sin', '-'): '11'
        },
        (3, 1):{
            ('cos', '+'): '11', ('cos', '-'): '01',
            ('sin', '+'): '01', ('sin', '-'): '11'
        }
    }
    sin_dict = {
        state_pair: {
            d: make_cz_sin_circ(
                d, state_pair, qubit_pairs
            ) for d in circuit_depths
        } for state_pair in state_pairs
    }
    cos_dict = {
        state_pair: {
            d: make_cz_cos_circ(
                d, state_pair, qubit_pairs
            ) for d in circuit_depths
        } for state_pair in state_pairs
    }

    signal = {state_pair: [] for state_pair in state_pairs}
    experiments = {}
    for state_pair in state_pairs:
        experiments[state_pair] = Q()

    for state_pair in state_pairs:
        cos_plus = state_pair_lookup[state_pair]['cos','+']
        cos_minus = state_pair_lookup[state_pair]['cos','-']
        sin_plus = state_pair_lookup[state_pair]['sin','+']
        sin_minus = state_pair_lookup[state_pair]['sin','-']
        for d in circuit_depths:
            experiments[state_pair].process_cos(d,
                (int(dataset[cos_dict[state_pair][d]][cos_plus]),
                 int(dataset[cos_dict[state_pair][d]][cos_minus])
                )
            )
            experiments[state_pair].process_sin(d,
                (int(dataset[sin_dict[state_pair][d]][sin_plus]),
                 int(dataset[sin_dict[state_pair][d]][sin_minus])
                )
            )
            p_I = int(dataset[cos_dict[state_pair][d]][cos_plus]) / (
                int(dataset[cos_dict[state_pair][d]][cos_plus]) + 
                int(dataset[cos_dict[state_pair][d]][cos_minus])
            )
            p_Q = int(dataset[sin_dict[state_pair][d]][sin_plus]) / (
                int(dataset[sin_dict[state_pair][d]][sin_plus]) + 
                int(dataset[sin_dict[state_pair][d]][sin_minus])
            )
            signal[state_pair].append(1 - 2 * p_I + 1j - 2j * p_Q)

    analyses = {}
    for state_pair in state_pairs:
        analyses[state_pair] = RobustPhaseEstimation(experiments[state_pair])
        if state_pair == (0, 1):
            analyses[state_pair].angle_estimates_rectified = [
                rectify_angle(theta) for theta in 
                analyses[state_pair].angle_estimates
            ]
        else:
            analyses[state_pair].angle_estimates_rectified = [
                theta for theta in analyses[state_pair].angle_estimates
            ]    

    # Turn lin. comb. estimates into direct phase estimates
    zz_estimates = 0.5 * (
        np.array(analyses[(0, 1)].angle_estimates_rectified) - 
        np.array(analyses[(2, 3)].angle_estimates_rectified)
    )
    iz_estimates = 0.5 * (
        np.array(analyses[(0, 1)].angle_estimates_rectified) + 
        np.array(analyses[(2, 3)].angle_estimates_rectified)
    )
    zi_estimates = (
        np.array(analyses[(3, 1)].angle_estimates_rectified) + zz_estimates
    )
    angle_estimates = {
        'ZZ': zz_estimates,
        'IZ': iz_estimates,
        'ZI': zi_estimates
    }

    # Extract the last "trusted" RPE angle estimate
    last_good_estimates = {}
    for state_pair in state_pairs:
        last_good_estimates[state_pair] = (
            analyses[(state_pair)].check_unif_local(historical=True)
        )
    last_good_idx = min(list(last_good_estimates.values()))
    # last_good_depth = 2**last_good_idx
    
    angle_errors = {
        'ZZ': zz_estimates - target_zz,
        'IZ': iz_estimates - target_iz,
        'ZI': zi_estimates - target_zi,
    }

    return (angle_estimates, angle_errors, last_good_idx, signal)


def rectify_angle(theta: float) -> float:
    """Rectify the angle to be in [-pi, pi].

    Args:
        theta (float): angle

    Returns:
        float: rectified angle
    """
    # if theta > np.pi:
    #     theta -= 2 * np.pi
    # return theta
    return (theta + np.pi) % (2 * np.pi) - np.pi


def plot_signal(
        signal: Dict, 
        circuit_depths: ArrayLike,
        ax: plt.axes = None, 
        title: str = None
    ) -> None:
    """Plot RPE signal decay.

    Args:
        signal (Dict): signal for each RPE experiment.
        circuit_depths (ArrayLike): circuit depths.
        ax (plt.axes, optional): plot axes. Defaults to None.
        title (str, optional): plot title. Defaults to None.
    """
    if not ax:
        fig, ax = plt.subplots(1, figsize=(5, 4))

    mt = list(Line2D.markers.keys())[2:]
    # Plot the signals on the complex plane with a colormap for the depth
    n = 0
    for label, data in signal.items():
        for i, d in enumerate(circuit_depths):
            ax.scatter(
                data[i].real, 
                data[i].imag,
                marker=mt[n], 
                color=plt.cm.viridis(i / (len(circuit_depths) - 1)),
                label=label if i == 0 else None
            )
        n += 1
    ax.set_title(title)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_aspect('equal')
    ax.grid()
    ax.legend(loc=1)

    # Add colorbar 
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, 
        norm=plt.Normalize(vmin=0, vmax=len(circuit_depths))
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Depth index')

    # Draw the unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    ax.add_artist(circle)

    # Set the axis limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)


def RPE(qpu:             QPU,
        config:          Config,
        qubit_labels:    Iterable[int],
        gate:            str,
        circuit_depths:  List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
        **kwargs
    ) -> Callable:
    """Robust Phase Estimation.

    This protocol requires a valid pyGSTi and pyRPE installation.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[int]): a list specifying sets of system labels 
            on which to perform RPE for a given gate.
        gate (str): gate on which to perform RPE.
        circuit_depths (List[int], optional): a list of positive integers 
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32, 
            64, 128, 256]```.

    Returns:
        Callable: RPE class instance.
    """

    class RPE(qpu):
        """pyRPE protocol."""

        def __init__(self,
                config:          Config,
                qubit_labels:    Iterable[int],
                gate:            str,
                circuit_depths:  List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
                **kwargs
            ) -> None:
            from qcal.interface.pygsti.transpiler import PyGSTiTranspiler

            try:
                import pygsti
                from pygsti.modelpacks import smq2Q_XXYYII
                from pygsti.modelpacks import smq2Q_XYICPHASE
                logger.info(f" pyGSTi version: {pygsti.__version__}\n")
            except ImportError:
                logger.warning(' Unable to import pyGSTi!')

            self._config = config

            assert gate.upper() in ('I', 'X90', 'CZ'), (
                'Only I, X90, and CZ gates are currently supported!'
            )
            self._qubit_labels = qubit_labels
            self._gate = gate.upper()
            self._circuit_depths = circuit_depths

            qubits = []
            for q in qubit_labels:
                if isinstance(q, Iterable):
                    qubits.extend(q)
                else:
                    qubits.append(q)
            self._qubits = sorted(qubits)

            self._gate_model = {
                'I':   None,
                'X90': X90,
                'CZ':  CZ,
            }
            self._target_model = {
                'I':   None,
                'X90': smq2Q_XXYYII.target_model(),
                'CZ':  smq2Q_XYICPHASE.target_model(),
            }
            self._make_circuits = {
                'I':   make_idle_circuits,
                'X90': make_x90_circuits,
                'CZ':  make_cz_circuits
            }
            self._analyze_results = {
                'I':   analyze_idle,
                'X90': analyze_x90,
                'CZ':  analyze_cz
            }

            self._circuits = None
            self._datasets = {}
            self._angle_estimates = {}
            self._angle_errors = {}
            self._last_good_idx = {}
            self._signal = {}

            transpiler = kwargs.get('transpiler', PyGSTiTranspiler())
            kwargs.pop('transpiler', None)
            qpu.__init__(self, config=config, transpiler=transpiler, **kwargs)

        @property
        def angle_estimates(self) -> Dict:
            """Angle estimates for each qubit/qubit pair.

            Returns:
                Dict: angle estimates.
            """
            return self._angle_estimates

        @property
        def angle_errors(self) -> Dict:
            """Angle errors for each qubit/qubit pair.

            Returns:
                Dict: angle errors.
            """
            return self._angle_errors

        @property
        def gate(self) -> str:
            """Gate being characterized.

            Returns:
                str: name of gate.
            """
            return self._gate

        @property
        def last_good_index(self) -> Dict:
            """Last good index for each qubit/qubit pair.

            Returns:
                Dict: last good index.
            """
            return self._last_good_idx
        
        @property
        def signal(self) -> Dict:
            """Signal decay.

            signal = 1 - 2p(I) + i(1 - 2p(Q))

            Returns:
                Dict: signal as a function of cirucit depth for each experiment.
            """
            return self._signal

        def generate_circuits(self):
            """Generate all RPE circuits."""
            from qcal.interface.pygsti.circuits import load_circuits

            logger.info(' Generating circuits from pyGSTi...')
            circuits = self._make_circuits[self._gate](
                self._circuit_depths, self._qubit_labels
            )
            self._circuits = load_circuits(circuits)

        def save(self):
            """Save all circuits and data."""
            self._data_manager._exp_id += (
                f'_RPE_Q{"".join(str(q) for q in self._qubit_labels)}'
            )

            if settings.Settings.save_data:
                logger.info(' Saving the circuits...')
                qpu.save(self)
                
        def generate_pygsti_dataset(self):
            """Generate a pyGSTi dataset for each qubit label."""
            logger.info(' Generating pyGSTi reports...')
            
            circuits = self._transpiled_circuits
            fileloc = self.data_manager.save_path
            for ql in self._qubit_labels:
                if isinstance(ql, Iterable):
                    q_index = tuple([self._qubits.index(q) for q in ql])
                    qs = ''.join(str(q) for q in ql)
                else:
                    q_index = tuple([self._qubits.index(ql)])
                    qs = str(ql)

                results_dfs = []
                for i, circuit in enumerate(circuits):
                    results_dfs.append(
                        pd.DataFrame(
                            [circuit.results.marginalize(q_index).dict], 
                            index=[circuits['pygsti_circuit'][i]]
                        )
                    )
                results_df = pd.concat(results_dfs)
                results_df = results_df.fillna(0).astype(int).rename(
                    columns=lambda col: col + ' count'
                )
                self._df = results_df

                with open(f'{fileloc}dataset_{qs}.txt', 'w') as f:
                    f.write(
                        '## Columns = ' + ', '.join(results_df.columns) + "\n"
                    )
                    f.close()
                results_df.to_csv(
                    f'{fileloc}dataset_{qs}.txt', 
                    sep=' ', 
                    mode='a', 
                    header=False
                )

        def analyze(self):
            """Analyze the RPE results."""
            logger.info(' Analyzing the results...')
            import pygsti
            
            clear_output(wait=True)
            for ql in self._qubit_labels:
                if isinstance(ql, Iterable):
                    qs = ''.join(str(q) for q in ql)
                else:
                    qs = str(ql)
                
                dataset = pygsti.io.read_dataset(
                    self.data_manager.save_path + f'dataset_{qs}.txt'
                )
                self._datasets[ql] = dataset

                results = self._analyze_results[self._gate](
                    dataset, self._qubit_labels, self._circuit_depths
                )
                angle_estimates, angle_errors, last_good_idx, signal = results
                self._angle_estimates[ql] = angle_estimates
                self._angle_errors[ql] = angle_errors
                self._last_good_idx[ql] = last_good_idx
                self._signal[ql] = signal

            for ql in self._qubit_labels:
                if isinstance(ql, Iterable):
                    print(f'\nQubit pair: {ql}')
                else:
                    print(f'\nQubit: {ql}')

                print(f'Last good depth: L = {2**self._last_good_idx[ql]}')
                for angle, errors in self._angle_errors[ql].items():
                    error = errors[self._last_good_idx[ql]]
                    unc = np.pi / (2 * 2**self._last_good_idx[ql])
                    error, unc = round_to_order_error(error, unc)
                    error_deg, unc_deg = round_to_order_error(
                        error * 180 / np.pi, unc * 180 / np.pi
                    )
                    print(
                        f'{angle} error = {error} ({unc}) rad., '
                        f'{error_deg} ({unc_deg}) deg.'
                    )

        def plot(self) -> None:
            """Plot the RPE results."""
            nrows, ncols = calculate_nrows_ncols(len(self._qubit_labels))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            k = -1
            for i in range(nrows):
                for j in range(ncols):
                    k += 1

                    if len(self._qubit_labels) == 1:
                        ax = axes
                    elif axes.ndim == 1:
                        ax = axes[j]
                    elif axes.ndim == 2:
                        ax = axes[i,j]

                    if k < len(self._qubit_labels):
                        ql = self._qubit_labels[k]
                        for angle, errors in self._angle_errors[ql].items():
                            ax.errorbar(
                                self._circuit_depths[:len(errors)], 
                                errors,
                                yerr=np.pi/(2*np.array(
                                    self._circuit_depths[:len(errors)]
                                )),
                                fmt='o-',
                                elinewidth=0.75,
                                capsize=7,
                                label=angle
                            )
                        ax.axvline(
                            2**self._last_good_idx[ql], 
                            ls='--',
                            c='k',
                            label='Last good depth',
                        )

                        maxval = np.abs(np.concatenate(
                            [err for err in self._angle_errors[ql].values()]
                        )).max()
                        ax.set_ylim((-1.1 * maxval, 1.1 * maxval))

                        ax.set_title(f'Q{ql}', fontsize=20)
                        ax.set_xlabel('Circuit Depth', fontsize=15)
                        ax.set_ylabel('Angle Error (rad.)', fontsize=15)
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )
                        ax.set_xscale('log')
                        # ax.set_yscale('log')
                        ax.legend(prop=dict(size=12))
                        ax.grid(True)

                    else:
                        ax.axis('off')
                
            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'RPE.png', 
                    dpi=300
                )
                fig.savefig(
                    self._data_manager._save_path + 'RPE.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'RPE.svg'
                )
            plt.show()

        def plot_signal(self) -> None:
            """Plot signal decay."""
            nrows, ncols = calculate_nrows_ncols(len(self._qubit_labels))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

            k = -1
            for i in range(nrows):
                for j in range(ncols):
                    k += 1

                    if len(self._qubit_labels) == 1:
                        ax = axes
                    elif axes.ndim == 1:
                        ax = axes[j]
                    elif axes.ndim == 2:
                        ax = axes[i,j]

                    if k < len(self._qubit_labels):
                        ql = self._qubit_labels[k]

                        plot_signal(
                            signal=self._signal[ql], 
                            circuit_depths=self._circuit_depths,
                            ax=ax,
                            title=f'Q{ql}'
                        )

                    else:
                        ax.axis('off')

            # fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'RPE_signal.png', 
                    dpi=300
                )
                fig.savefig(
                    self._data_manager._save_path + 'RPE_signal.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'RPE_signal.svg'
                )
            plt.show()

        def final(self) -> None:
            """Final method."""
            if settings.Settings.save_data:
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._angle_estimates]), 'angle_estimates'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._angle_errors]), 'angle_errors'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._last_good_idx]), 'last_good_idx'
                )
            
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.save()
            self.generate_pygsti_dataset()
            self.analyze()
            self.plot()
            self.final()

    return RPE(
        config,
        qubit_labels,
        gate,   
        circuit_depths,
        **kwargs
    )