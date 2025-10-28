"""Submodule for performing Gate Set Tomography (GST)

Relevant literature:
- https://quantum-journal.org/papers/q-2021-10-05-557/

Relevant code repos:
- https://www.pygsti.info/
- https://github.com/sandialabs/pyGSTi
"""
import qcal.settings as settings

from qcal.config import Config
from qcal.qpu.qpu import QPU
from qcal.utils import flatten, save_init

import logging
import numpy as np

from collections.abc import Iterable
from IPython.display import clear_output
from typing import Any, Callable, List, Tuple

logger = logging.getLogger(__name__)


def GST(qpu:            QPU,
        config:         Config,
        qubit_labels:   Iterable[int],
        pspec:          Any | None = None,
        target_model:   Any | None = None,
        prep_fiducials: Any | None = None,
        meas_fiducials: Any | None = None,
        germs:          Any | None = None,
        circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
        fpr:            bool = False,
        **kwargs
    ) -> Callable:
    """Gate Set Tomography.

    This protocol requires a valid pyGSTi installation.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[int]): a list specifying sets of system labels 
            on which to perform RPE for a given gate.
       
        circuit_depths (List[int], optional): a list of positive integers 
            specifying the circuit depths. Defaults to ```[1, 2, 4, 8, 16, 32, 
            64, 128, 256]```.

    Returns:
        Callable: GST class instance.
    """

    class GST(qpu):
        """GST protocol."""
        
        @save_init
        def __init__(self,
                config:         Config,
                qubit_labels:   Iterable[int | Tuple[int]],
                pspec:          Any | None = None,
                target_model:   Any | None = None,
                prep_fiducials: Any | None = None,
                meas_fiducials: Any | None = None,
                germs:          Any | None = None,
                circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
                fpr:            bool = False,
                **kwargs
            ) -> None:
            try:
                import pygsti
                from qcal.interface.pygsti.transpiler import PyGSTiTranspiler
                logger.info(f" pyGSTi version: {pygsti.__version__}\n")
            except ImportError:
                logger.warning(' Unable to import pyGSTi!')

            self._qubit_labels = qubit_labels
            self._qubits = sorted(flatten(qubit_labels))
            self._pspec = pspec
            self._target_model = target_model
            self._prep_fiducials = prep_fiducials
            self._meas_fiducials = meas_fiducials
            self._germs = germs
            self._circuit_depths = circuit_depths
            self._fpr = fpr

            self._protocol = None
            self._edesign = None
            self._data = None
            self._results = None
            self._circuits = None
            self._datasets = {}
            self._report = None

            transpiler = kwargs.get('transpiler', PyGSTiTranspiler())
            kwargs.pop('transpiler', None)
            qpu.__init__(self, config=config, transpiler=transpiler, **kwargs)

        @property
        def pspec(self):
            """pyGSTi processor spec."""
            return self._pspec
        
        @property
        def protocol(self):
            """pyGSTi protocol."""
            return self._protocol
        
        @property
        def edesign(self):
            """pyGSTi edesign."""
            return self._edesign
        
        @property
        def data(self):
            """pyGSTi data object."""
            return self._data
        
        @property
        def results(self):
            """pyGSTi results object."""
            return self._results
        
        def generate_circuits(self):
            """Generate all GST circuits."""
            from pygsti.algorithms.fiducialpairreduction import (
                find_sufficient_fiducial_pairs_per_germ_greedy
            )
            from pygsti.io import write_empty_protocol_data
            from pygsti.protocols import StandardGST, StandardGSTDesign
            from qcal.interface.pygsti.circuits import load_circuits

            print("Prep fiducials:\n", self._prep_fiducials)
            print("Meas fiducials:\n", self._meas_fiducials)
            print("Germs:\n", self._germs)

            self._protocol = StandardGST(
                modes=('full TP','CPTPLND','Target', 'H+S', 'S'),
                target_model=self._target_model
            )

            if self._fpr:
                fiducial_pairs = find_sufficient_fiducial_pairs_per_germ_greedy(
                    target_model=self._target_model, 
                    prep_fiducials=self._prep_fiducials,
                    meas_fiducials=self._meas_fiducials,
                    germs=self._germs,
                    prep_povm_tuples="first", 
                    constrain_to_tp=True,
                    inv_trace_tol= 10, 
                    initial_seed_mode='greedy',
                    evd_tol=1e-5, 
                    sensitivity_threshold=1e-5, 
                    # seed=1222022,
                    verbosity=1,
                    check_complete_fid_set=False
                )
            else: 
                fiducial_pairs = None

            self._edesign = StandardGSTDesign(
                processorspec_filename_or_obj=self._pspec,
                # target_model=self._target_model, 
                prep_fiducial_list_or_filename=self._prep_fiducials, 
                meas_fiducial_list_or_filename=self._meas_fiducials,
                germ_list_or_filename=self._germs, 
                max_lengths=self._circuit_depths,
                fiducial_pairs=fiducial_pairs
            )
            print(
                'Number of circuits: ', 
                len(self._edesign.all_circuits_needing_data)
            )   

            # Save an empty dataset file of all the circuits
            self._data_manager._exp_id += (
                f'_GST_{"".join("Q" + str(q) for q in self._qubits)}'
            )
            self._data_manager.create_data_path()
            write_empty_protocol_data(
                self._data_manager._save_path, 
                self._edesign, 
                sparse=True,
                clobber_ok=True
            )

            self._circuits = load_circuits(
                self._data_manager._save_path + 'data/dataset.txt'
            )

        def save(self):
            """Save all circuits and data."""
            from qcal.interface.pygsti.datasets import generate_pygsti_dataset
            clear_output(wait=True)
            generate_pygsti_dataset(
                self._transpiled_circuits,
                self._data_manager._save_path + 'data/'
            )
            if settings.Settings.save_data:
                qpu.save(self, create_data_path=False)

        def analyze(self):
            """Analyze the GST results."""
            logger.info(' Analyzing the results...')
            import pygsti
            
            self._data = pygsti.io.read_data_from_dir(
                self._data_manager._save_path
            )
            self._results = self._protocol.run(self._data)

            self._report = pygsti.report.construct_standard_report(
                self._results, 
                title="GST Report", 
                verbosity=2
            )
            self._report.write_html(
                self._data_manager._save_path, verbosity=2
            )

        def final(self) -> None:
            """Final method."""
            # if settings.Settings.save_data:
            #     self._data_manager.save_to_csv(
            #         pd.DataFrame([self._angle_estimates]), 'angle_estimates'
            #     )
            #     self._data_manager.save_to_csv(
            #         pd.DataFrame([self._angle_errors]), 'angle_errors'
            #     )
            #     self._data_manager.save_to_csv(
            #         pd.DataFrame([self._last_good_idx]), 'last_good_idx'
            #     )
            
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

    return GST(
        config=config,
        qubit_labels=qubit_labels,
        pspec=pspec,
        target_model=target_model,
        prep_fiducials=prep_fiducials,
        meas_fiducials=meas_fiducials, 
        germs=germs, 
        circuit_depths=circuit_depths,
        fpr=fpr,
        **kwargs
    )


def SingleQubitGST(
        qpu:            QPU,
        config:         Config,
        qubits:         Iterable[int],
        pspec:          Any | None = None,
        target_model:   Any | None = None,
        prep_fiducials: Any | None = None,
        meas_fiducials: Any | None = None,
        germs:          Any | None = None,
        circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
        fpr:            bool = False,
        **kwargs
    ) -> Callable:
    """Single-Qubit Gate Set Tomography.

    This protocol requires a valid pyGSTi installation.

    
    """
    gst = type(GST(
        qpu=qpu,
        config=config,
        qubit_labels=qubits,
        pspec=pspec,
        target_model=target_model,
        prep_fiducials=prep_fiducials,
        meas_fiducials=meas_fiducials, 
        germs=germs, 
        circuit_depths=circuit_depths,
        fpr=fpr,
        **kwargs
    ))

    class SingleQubitGST(gst):
        """GST protocol."""
        
        @save_init
        def __init__(self,
                config:         Config,
                qubits:         Iterable[int],
                pspec:          Any | None = None,
                target_model:   Any | None = None,
                prep_fiducials: Any | None = None,
                meas_fiducials: Any | None = None,
                germs:          Any | None = None,
                circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
                fpr:            bool = False,
                **kwargs
            ) -> None:
            try:
                import pygsti
                from pygsti.modelpacks import smq1Q_XY
                logger.info(f" pyGSTi version: {pygsti.__version__}\n")
            except ImportError:
                logger.warning(' Unable to import pyGSTi!')

            if len(qubits) > 2:
                raise ValueError(
                    'Single-qubit GST is not currently supported for more than '
                    '2 qubits!'
                )

            if len(qubits) == 1:
                pspec = (
                    smq1Q_XY.processor_spec(qubits) if pspec is None 
                    else pspec
                )
                
                target_model = (
                    smq1Q_XY.target_model(qubit_labels=qubits) if target_model
                    is None else target_model
                )

                prep_fiducials = (
                    smq1Q_XY.prep_fiducials(qubits) if prep_fiducials is None 
                    else prep_fiducials
                )

                meas_fiducials = (
                    smq1Q_XY.meas_fiducials(qubits) if meas_fiducials is None 
                    else meas_fiducials
                )

                germs = smq1Q_XY.germs(qubits) if germs is None else germs

            elif len(qubits) == 2:
                from pygsti.algorithms.fiducialselection import find_fiducials
                from pygsti.algorithms.germselection import find_germs
                from pygsti.models.modelconstruction import (
                    create_explicit_model
                )
                from pygsti.processors import QubitProcessorSpec
                from pygsti.tools.internalgates import (
                    standard_gatename_unitaries
                )

                if pspec is None:
                    gate_names = [
                        'Gxpi2', 'Gypi2', 'Gii', 'Gxx', 'Gxy','Gyx', 'Gyy'
                    ]

                    # Define a global 2-qubit idle
                    global_idle = np.eye(4)

                    # Define the unitaries for the parallel gates
                    standard_gate_unitaries= standard_gatename_unitaries()
                    gxpi2 = standard_gate_unitaries['Gxpi2']
                    gypi2 = standard_gate_unitaries['Gypi2']

                    gxx = np.kron(gxpi2, gxpi2)
                    gxy = np.kron(gxpi2, gypi2)
                    gyx = np.kron(gypi2, gxpi2)
                    gyy = np.kron(gypi2, gypi2)

                    nonstd_gate_unitaries = {
                        'Gii': global_idle, 
                        'Gxx': gxx, 
                        'Gxy': gxy,
                        'Gyx': gyx, 
                        'Gyy': gyy
                    }

                    pspec = QubitProcessorSpec(
                        num_qubits=len(qubits), 
                        gate_names=gate_names,
                        nonstd_gate_unitaries= nonstd_gate_unitaries,
                        prep_names=['rho0'], 
                        povm_names=['Mdefault'],
                        availability={
                            'Gxpi2': [(q,) for q in qubits], 
                            'Gzpi2': [(q,) for q in qubits],
                            'Gii':   [tuple(qubits)],
                            'Gxx':   [tuple(qubits)], 
                            'Gxy':   [tuple(qubits)], 
                            'Gyx':   [tuple(qubits)], 
                            'Gyy':   [tuple(qubits)]
                        },
                        qubit_labels=qubits
                    )

                if target_model is None:
                    target_model = create_explicit_model(
                        pspec,  
                        # ideal_gate_type='full TP', 
                        # ideal_spam_type='full TP',  
                        # basis='pp',
                    )

                if prep_fiducials is None and meas_fiducials is None:
                    prep_fiducials, meas_fiducials = find_fiducials(
                        target_model, 
                        candidate_fid_counts={3: 'all upto'}, 
                        assume_clifford=True,
                        verbosity=2
                    )

                if germs is None:
                    germs = find_germs(
                        target_model, 
                        randomize=False, 
                        algorithm='greedy', 
                        assume_real=True, 
                        mode='compactEVD', 
                        float_type=np.double,
                        candidate_germ_counts={4:'all upto'}, 
                        verbosity=2
                    )
            
            gst.__init__(self,
                config=config,
                qubit_labels=qubits,
                pspec=pspec,
                target_model=target_model,
                prep_fiducials=prep_fiducials,
                meas_fiducials=meas_fiducials, 
                germs=germs, 
                circuit_depths=circuit_depths,
                fpr=fpr,
                **kwargs
            )

    return SingleQubitGST(
        config=config,
        qubits=qubits,
        pspec=pspec,
        target_model=target_model,
        prep_fiducials=prep_fiducials,
        meas_fiducials=meas_fiducials, 
        germs=germs, 
        circuit_depths=circuit_depths,
        fpr=fpr,
        **kwargs
    )


def TwoQubitGST(
        qpu:            QPU,
        config:         Config,
        qubits:         Iterable[int],
        pspec:          Any | None = None,
        target_model:   Any | None = None,
        prep_fiducials: Any | None = None,
        meas_fiducials: Any | None = None,
        germs:          Any | None = None,
        circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
        fpr:            bool = False,
        **kwargs
    ) -> Callable:
    """Two-Qubit Gate Set Tomography.

    This protocol requires a valid pyGSTi installation.

    
    """
    gst = type(GST(
        qpu=qpu,
        config=config,
        qubit_labels=qubits,
        pspec=pspec,
        target_model=target_model,
        prep_fiducials=prep_fiducials,
        meas_fiducials=meas_fiducials, 
        germs=germs, 
        circuit_depths=circuit_depths,
        fpr=fpr,
        **kwargs
    ))

    class TwoQubitGST(gst):
        """GST protocol."""
        
        @save_init
        def __init__(self,
                config:         Config,
                qubits:         Iterable[int],
                pspec:          Any | None = None,
                target_model:   Any | None = None,
                prep_fiducials: Any | None = None,
                meas_fiducials: Any | None = None,
                germs:          Any | None = None,
                circuit_depths: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
                fpr:            bool = False,
                **kwargs
            ) -> None:
            try:
                import pygsti
                from pygsti.modelpacks import smq2Q_XYCPHASE
                logger.info(f" pyGSTi version: {pygsti.__version__}\n")
            except ImportError:
                logger.warning(' Unable to import pyGSTi!')

            if len(qubits) != 2:
                raise ValueError(
                    'Two-qubit GST is only support for 2 qubits!'
                )
            
            pspec = (
                smq2Q_XYCPHASE.processor_spec(qubits) if pspec is None 
                else pspec
            )

            target_model = (
                smq2Q_XYCPHASE.target_model(qubit_labels=qubits) if 
                target_model is None else target_model
            )

            prep_fiducials = (
                smq2Q_XYCPHASE.prep_fiducials(qubits) if prep_fiducials is None 
                else prep_fiducials
            )

            meas_fiducials = (
                smq2Q_XYCPHASE.meas_fiducials(qubits) if meas_fiducials is None 
                else meas_fiducials
            )

            germs = smq2Q_XYCPHASE.germs(qubits) if germs is None else germs
            
            gst.__init__(self,
                config=config,
                qubit_labels=qubits,
                pspec=pspec,
                target_model=target_model,
                prep_fiducials=prep_fiducials,
                meas_fiducials=meas_fiducials, 
                germs=germs, 
                circuit_depths=circuit_depths,
                fpr=fpr,
                **kwargs
            )

    return TwoQubitGST(
        config=config,
        qubits=qubits,
        pspec=pspec,
        target_model=target_model,
        prep_fiducials=prep_fiducials,
        meas_fiducials=meas_fiducials, 
        germs=germs, 
        circuit_depths=circuit_depths,
        fpr=fpr,
        **kwargs
    )
