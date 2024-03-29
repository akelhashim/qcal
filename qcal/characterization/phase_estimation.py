""""Submodule for phase estimation experiments.

See:
https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.062315
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.190502

Relevant pyRPE code: https://gitlab.com/quapack/pyrpe
"""
import qcal.settings as settings

from qcal.characterization.characterize import Characterize
from qcal.config import Config
from qcal.interface.pygsti.transpiler import Transpiler
from qcal.managers.classification_manager import ClassificationManager
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
from numpy.typing import NDArray
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

    diag([1,1,1,-1]) == CZ(np.pi, np.pi, -np.pi) (up to phase).

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

    return scipy.linalg.expm(-1j /4 * (theta_iz * pygsti.sigmaiz + 
                                       theta_zi * pygsti.sigmazi + 
                                       theta_zz * pygsti.sigmazz))


def make_x90_circuits(circuit_depths: List[int], qubits: Tuple[int]):
    """Generate the X90 RPE circuits.

    Args:
        circuit_depths (List[int]): circuit depths.
        qubits (Tuple[int]): qubit labels.
    """
    try:
        import pygsti
    except ImportError:
        logger.warning('Unable to import pyGSTi!')

    # cos_circs = {k: make_x90_cos_circ(k) for k in max_depths}
    # sin_circs = {k: make_x90_sin_circ(k) for k in max_depths}

    circuits = (
        [make_x90_cos_circ(d, qubits) for d in circuit_depths] +
        [make_x90_sin_circ(d, qubits) for d in circuit_depths]
    )

    return pygsti.remove_duplicates_in_place(circuits)

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

    return pygsti.circuits.Circuit([[('Gxpi2', q) for q in qubits]]) * d


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

    return pygsti.circuits.Circuit([[('Gxpi2', q) for q in qubits]]) * (d + 1)


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
               [[('Gxpi2', qp[1]) for qp in qubit_pairs]], 
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


def analyze_x90(dataset, qubit: int, circuit_depths: List[int]) -> Tuple:
    """Analyze RPE dataset for the X90 gate.

    Args:
        dataset (pygsti.data.dataset): pyGSTi dataset.
        qubit (int): qubit label.
        circuit_depths (List[int]): circuit depths.

    Returns:
        Tuple: angle estimates, angle errors, and last good depth
    """
    try:
        from quapack.pyRPE import RobustPhaseEstimation
        from quapack.pyRPE.quantum import Q
    except ImportError:
        logger.warning(' Unable to import pyRPE!')

    target_x = np.pi/2

    cos_circs = {d: make_x90_cos_circ(d, [qubit]) for d in circuit_depths}
    sin_circs = {d: make_x90_sin_circ(d, [qubit]) for d in circuit_depths}

    experiment = Q()
    for d in circuit_depths:
        cos_circ = cos_circs[d]
        sin_circ = sin_circs[d]
        experiment.process_cos(d,
            (int(dataset[cos_circ]['0']), int(dataset[cos_circ]['1']))
        )
        experiment.process_sin(d,
            (int(dataset[sin_circ]['1']), int(dataset[sin_circ]['0']))
        )
        
    analysis = RobustPhaseEstimation(experiment)
    angle_estimates = {
        'X': analysis.angle_estimates,
    }

    # Extract the last "trusted" RPE angle estimate
    last_good_idx = analysis.check_unif_local(historical=True)
    # last_good_depth = 2**last_good_idx
    
    angle_errors = {
        'X': analysis.angle_estimates - target_x
    }

    return (angle_estimates, angle_errors, last_good_idx)
        

def analyze_cz(dataset, qubit_pair: Tuple, circuit_depths: List[int]) -> Tuple:
    """Analyze RPE dataset for the CZ gate.

    Args:
        dataset (pygsti.data.dataset): pyGSTi dataset.
        qubit_pair (int): qubit labels.
        circuit_depths (List[int]): circuit depths.

    Returns:
        Tuple: angle estimates, angle errors, and last good depth
    """
    try:
        from quapack.pyRPE import RobustPhaseEstimation
        from quapack.pyRPE.quantum import Q
    except ImportError:
        logger.warning(' Unable to import pyRPE!')

    target_zz = -np.pi
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
                d, state_pair, [qubit_pair]
            ) for d in circuit_depths
        } for state_pair in state_pairs
    }
    cos_dict = {
        state_pair: {
            d: make_cz_cos_circ(
                d, state_pair, [qubit_pair]
            ) for d in circuit_depths
        } for state_pair in state_pairs
    }

    experiments = {}
    for state_pair in state_pairs:
        experiments[state_pair] = Q()

    for state_pair in state_pairs:
        cos_plus = state_pair_lookup[state_pair]['cos','+']
        cos_minus = state_pair_lookup[state_pair]['cos','-']
        sin_plus = state_pair_lookup[state_pair]['sin','+']
        sin_minus = state_pair_lookup[state_pair]['sin','-']
        for d in circuit_depths:
            experiments[state_pair].process_sin(d,
                (int(dataset[sin_dict[state_pair][d]][sin_plus]),
                 int(dataset[sin_dict[state_pair][d]][sin_minus])
                )
            )
            experiments[state_pair].process_cos(d,
                (int(dataset[cos_dict[state_pair][d]][cos_plus]),
                 int(dataset[cos_dict[state_pair][d]][cos_minus])
                )
            )

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
    zz_estimates = (
        np.array(analyses[(0, 1)].angle_estimates_rectified) - 
        np.array(analyses[(2, 3)].angle_estimates_rectified)
    )
    iz_estimates = (
        np.array(analyses[(0, 1)].angle_estimates_rectified) + 
        np.array(analyses[(2, 3)].angle_estimates_rectified)
    )
    zi_estimates = (
        2 * np.array(analyses[(3, 1)].angle_estimates_rectified) + 
        zz_estimates
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

    return (angle_estimates, angle_errors, last_good_idx)


def rectify_angle(theta: float) -> float:
    """Constain an angle to be within pi.

    Args:
        theta (float): angle

    Returns:
        float: rectified angle
    """
    if theta > np.pi:
        theta -= 2 * np.pi
    return theta


def RPE(qpu:             QPU,
        config:          Config,
        qubit_labels:    Iterable[int],
        gate:            str,
        circuit_depths:  List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
        compiler:        Any | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        raster_circuits: bool = False,
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
        circuit_depths (List[int]): a list of positive integers specifying the 
            circuit depths.
        compiler (Any | Compiler | None, optional): custom compiler to compile
            the True-Q circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to transpile
            the True-Q circuits to experimental circuits. Defaults to None.
        classifier (ClassificationManager, optional): manager used for 
            classifying raw data. Defaults to None.
        n_shots (int, optional): number of measurements per circuit. 
                Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. Defaults
            to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that can be
            measured per sequence. Defaults to 1.
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.

    Returns:
        Callable: RPE class instance.
    """

    class RPE(qpu, Characterize):
        """pyRPE protocol."""

        def __init__(self,
                qpu:             QPU,
                config:          Config,
                qubit_labels:    Iterable[int],
                gate:            str,
                circuit_depths:  circuit_depths,
                compiler:        Any | None = None,
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_shots:         int = 1024,
                n_batches:       int = 1,
                n_circs_per_seq: int = 1,
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            try:
                import pygsti
                from pygsti.modelpacks import smq2Q_XXYYII
                from pygsti.modelpacks import smq2Q_XYICPHASE
                print(f"pyGSTi version: {pygsti.__version__}\n")
            except ImportError:
                logger.warning(' Unable to import pyGSTi!')

            Characterize.__init__(self, config)

            assert gate.upper() in ('X90', 'CZ'), (
                'Only X90 and CZ gates are currently supported!'
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
                'X90': X90,
                'CZ':  CZ,
            }
            self._target_model = {
                'X90': smq2Q_XXYYII.target_model(),
                'CZ':  smq2Q_XYICPHASE.target_model(),
            }
            self._make_circuits = {
                'X90': make_x90_circuits,
                'CZ':  make_cz_circuits
            }
            self._analyze_results = {
                'X90': analyze_x90,
                'CZ':  analyze_cz
            }

            self._circuits = None
            self._datasets = {}
            self._angle_estimates = {}
            self._angle_errors = {}
            self._last_good_idx = {}

            if transpiler is None:
                transpiler = Transpiler()

            qpu.__init__(self,
                config=config, 
                compiler=compiler, 
                transpiler=transpiler,
                classifier=classifier,
                n_shots=n_shots, 
                n_batches=n_batches, 
                n_circs_per_seq=n_circs_per_seq, 
                raster_circuits=raster_circuits,
                **kwargs
            )

        @property
        def angle_estimates(self) -> Dict:
            """Angle estimates for each qubit/qubit pair.

            Returns:
                Dict: angle estimates.
            """
            self._angle_estimates

        @property
        def angle_errors(self) -> Dict:
            """Angle errors for each qubit/qubit pair.

            Returns:
                Dict: angle errors.
            """
            self._angle_errors

        @property
        def last_good_index(self) -> Dict:
            """Last good index for each qubit/qubit pair.

            Returns:
                Dict: last good index.
            """
            self._last_good_idx

        def generate_circuits(self):
            """Generate all RPE circuits."""
            logger.info(' Generating circuits from pyGSTi...')
            self._circuits = self._make_circuits[self._gate]

        def generate_pygsti_report(self):
            """Generate a pyGSTi report for each qubit label."""
            logger.info(' Generating pyGSTi reports...')
            
            circuits = self._transpiled_circuits
            fileloc = self.data_manager.save_path
            for ql in self._qubit_labels:
                if isinstance(ql, Iterable):
                    q_index = (self._qubits.index(q) for q in ql)
                    qs = ''.join(str(q) for q in ql)
                else:
                    q_index = (self._qubits.index(ql),)
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
            """Analyze the SRB results."""
            logger.info(' Analyzing the results...')
            import pygsti
            
            clear_output(wait=True)
            for ql in self._qubit_labels:
                if isinstance(ql, Iterable):
                    print(f'\nQubit pair: {ql}')
                    qs = ''.join(str(q) for q in ql)
                else:
                    print(f'\nQubit: {ql}')
                    qs = str(ql)
                
                dataset = pygsti.io.read_dataset(
                    self.data_manager.save_path + f'_dataset{qs}.txt'
                )
                self._datasets[ql] = dataset

                results = self._analyze_results[self._gate](
                    dataset, ql, self._circuit_depths
                )
                angle_estimates, angle_errors, last_good_idx = results
                self._angle_estimates[ql] = angle_estimates
                self._angle_errors[ql] = angle_errors
                self._last_good_idx[ql] = last_good_idx

                print(f'Last good depth: L = {2**last_good_idx})')
                for angle, errors in angle_errors.items():
                    error = errors[last_good_idx]
                    unc = np.pi / (2 * 2**last_good_idx)
                    error, unc = round_to_order_error(error, unc)
                    print(
                        f'{angle} error = {error} ({unc}) rad., '
                        f'{error * 180 / np.pi} ({unc * 180 / np.pi}) deg.'
                    )

        def save(self):
            """Save all circuits and data."""
            self._data_manager._exp_id += (
                f'_RPE_Q{"".join(str(q) for q in self._circuits.labels)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._angle_estimates]), 'angle_estimates'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._angle_errors]), 'angle_errors'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._last_good_idx]), 'last_good_idx'
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
                            ax.plot(
                                self._circuit_depths, errors, 'o-', label=angle
                            )
                        ax.axvline(
                            2**self._last_good_depth[ql], 
                            label='Last good depth'
                        )

                        ax.set_title(f'Q{ql}', fontsize=20)
                        ax.set_xlabel('Circuit Depth', fontsize=15)
                        ax.set_ylabel('Angle Error')
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )
                        ax.set_xscale('log')
                        ax.set_yscale('log')
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
            plt.show()

        def final(self) -> None:
            """Final method."""
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.plot()
            self.save()
            self.final()

    return RPE(
        config,
        qubit_labels,
        gate,   
        circuit_depths,
        compiler, 
        transpiler,
        classifier,
        n_shots, 
        n_batches, 
        n_circs_per_seq,
        raster_circuits,
        **kwargs
    )