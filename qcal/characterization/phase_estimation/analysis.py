""""Submodule for analyzing RPE experiments.

"""
from qcal.characterization.phase_estimation.circuits import (
    make_idle_cos_circ, make_idle_sin_circ, make_x90_cos_circ,
    make_x90_sin_circ, make_X90_icos_circ, make_X90_isin_circ,
    make_cz_cos_circ, make_cz_sin_circ
)

import logging
import numpy as np

from typing import List, Tuple

logger = logging.getLogger(__name__)


def analyze_idle(
        dataset, 
        qubits:         List[int], 
        circuit_depths: List[int],
        gate_layer:     List = None,
    ) -> Tuple:
    """Analyze idle RPE dataset.

    Args:
        dataset (pygsti.data.dataset): pyGSTi dataset.
        qubits (List[int]): qubit labels.
        circuit_depths (List[int]): circuit depths.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
        Tuple: angle estimates, angle errors, and index of last good depth
    """
    try:
        from quapack.pyRPE import RobustPhaseEstimation
        from quapack.pyRPE.quantum import Q
    except ImportError:
        logger.warning(' Unable to import pyRPE!')

    cos_circs = {
        d: make_idle_cos_circ(d, qubits, gate_layer) 
        for d in circuit_depths
    }
    sin_circs = {
        d: make_idle_sin_circ(d, qubits, gate_layer) 
        for d in circuit_depths
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
        qubits:         List[int], 
        circuit_depths: List[int], 
        gate_layer:     List = None,
        estimator_type: str = 'linearized',
    ) -> Tuple:
    """Analyze RPE dataset for the X90 gate.

    Args:
        dataset (pygsti.data.dataset): pyGSTi dataset.
        qubits (List[int]): qubit labels.
        circuit_depths (List[int]): circuit depths.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.
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
    eps = 1e-8  # Small additive factor to ensure we do not divide by zero

    direct_cos_circs = {
        d: make_x90_cos_circ(d, qubits, gate_layer) 
        for d in circuit_depths
    }
    direct_sin_circs = {
        d: make_x90_sin_circ(d, qubits, gate_layer) 
        for d in circuit_depths
    }

    interleaved_cos_circs = {
        d: make_X90_icos_circ(d, qubits, gate_layer) 
        for d in circuit_depths
    }
    interleaved_sin_circs = {
        d: make_X90_isin_circ(d, qubits, gate_layer) 
        for d in circuit_depths
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
            int(direct_cos_counts['0']) + int(direct_cos_counts['1']) + eps
        )
        p_Q = int(direct_sin_counts['1']) / (
            int(direct_sin_counts['0']) + int(direct_sin_counts['1']) + eps
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
        dataset, 
        qubit_pairs:    List[Tuple[int]], 
        circuit_depths: List[int], 
        gate_layer:     List = None,
    ) -> Tuple:
    """Analyze RPE dataset for the CZ gate.

    Args:
        dataset (pygsti.data.dataset): pyGSTi dataset.
        qubit_pairs (List[Tuple[int]]): qubit labels.
        circuit_depths (List[int]): circuit depths.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

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
    eps = 1e-8  # Small additive factor to ensure we do not divide by zero

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
                d, state_pair, qubit_pairs, gate_layer
            ) for d in circuit_depths
        } for state_pair in state_pairs
    }
    cos_dict = {
        state_pair: {
            d: make_cz_cos_circ(
                d, state_pair, qubit_pairs, gate_layer
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
                int(dataset[cos_dict[state_pair][d]][cos_minus]) + eps
            )
            p_Q = int(dataset[sin_dict[state_pair][d]][sin_plus]) / (
                int(dataset[sin_dict[state_pair][d]][sin_plus]) + 
                int(dataset[sin_dict[state_pair][d]][sin_minus]) + eps
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


def analyze_zz(
        dataset, 
        qubit_pairs:    List[Tuple[int]], 
        circuit_depths: List[int], 
        gate_layer:     List = None,
    ) -> Tuple:
    """Analyze RPE dataset for the CZ gate.

    Args:
        dataset (pygsti.data.dataset): pyGSTi dataset.
        qubit_pairs (List[Tuple[int]]): qubit labels.
        circuit_depths (List[int]): circuit depths.
        gate_layer (List, optional): custom gate layer for the gate of interest.
            Defaults to None.

    Returns:
        Tuple: angle estimates, angle errors, and index of last good depth
    """
    try:
        from quapack.pyRPE import RobustPhaseEstimation
        from quapack.pyRPE.quantum import Q
    except ImportError:
        logger.warning(' Unable to import pyRPE!')

    target_zz = target_iz = target_zi = 0
    eps = 1e-8  # Small additive factor to ensure we do not divide by zero

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
                d, state_pair, qubit_pairs, gate_layer
            ) for d in circuit_depths
        } for state_pair in state_pairs
    }
    cos_dict = {
        state_pair: {
            d: make_cz_cos_circ(
                d, state_pair, qubit_pairs, gate_layer
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
                int(dataset[cos_dict[state_pair][d]][cos_minus]) + eps
            )
            p_Q = int(dataset[sin_dict[state_pair][d]][sin_plus]) / (
                int(dataset[sin_dict[state_pair][d]][sin_plus]) + 
                int(dataset[sin_dict[state_pair][d]][sin_minus]) + eps
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
    # Wrap angles to [-π/2, π/2] using arcsin(sin(x))
    zz_estimates = np.arcsin(np.sin( 
        0.5 * (
            np.array(analyses[(0, 1)].angle_estimates_rectified) - 
            np.array(analyses[(2, 3)].angle_estimates_rectified)
        )
    ))
    iz_estimates = np.arcsin(np.sin(
        0.5 * (
            np.array(analyses[(0, 1)].angle_estimates_rectified) + 
            np.array(analyses[(2, 3)].angle_estimates_rectified)
        )
    ))
    zi_estimates = np.arcsin(np.sin(
        np.array(analyses[(3, 1)].angle_estimates_rectified) + zz_estimates
    ))
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