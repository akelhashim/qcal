"""Submodule for RB routines.

For SRB, see:
https://trueq.quantumbenchmark.com/guides/error_diagnostics/srb.html


"""
import qcal.settings as settings

from qcal.config import Config
from qcal.managers.classification_manager import ClassificationManager
from qcal.qpu.qpu import QPU
from qcal.plotting.utils import calculate_nrows_ncols

import logging
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import clear_output
from typing import Any, Callable, List, Tuple, Iterable


logger = logging.getLogger(__name__)


def CRB(qpu:             QPU,
        config:          Config,
        qubits:          List[int] | Tuple[int],
        circuit_depths:  List[int] | Tuple[int],
        n_circuits:      int = 30,
        native_gates:    List[str] = ['X90', 'Y90'],
        pspec:           Any | None = None,
        randomizeout:    bool = True,
        citerations:     int = 20,  
        **kwargs
    ) -> Callable:
    """Clifford Randomized Benchmarking.

    This is a pyGSTi protocol and requires pyGSTi to be installed.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (Iterable[int, Iterable[int]]): a list specifying the qubit
            labels.
        circuit_depths (List[int] | Tuple[int]):  a list of integers >= 0. The 
            CRB depth is the number of Cliffords in the circuit - 2 before 
            each Clifford is compiled into the native gate-set.
        n_circuits (int, optional): The number of (possibly) different CRB 
            circuits sampled at each depth. Defaults to 30.
        native_gates (List[str], optional): a list of the native gates. Defaults 
            to ['X90', 'Y90']. All Cliffords will be decomposed in terms of 
            these gates. If a custom pspec is passed, this list will be ignored.
        pspec (QubitProcessorSpec | None, optional): pyGSTi qubit processor 
            spec. Defaults to None. If None, a processor spec will be 
            automatically generated based on the native_gates and the qubit 
            labels.
        randomizeout (bool, optional): whether or not a random Pauli is compiled
            into the output layer of the circuit. Defaults to True. If False, 
            the ideal output of the circuits (the "success" or "survival" 
            outcome) is always the all-zeros bit string. If True, the ideal 
            output a circuit is randomized to a uniformly random bit-string.
        citerations (int, optional): some of the Clifford compilation algorithms 
            in pyGSTi (including the default algorithm) are randomized, and the 
            lowest-cost circuit is chosen from all the circuit generated in the
            iterations of the algorithm. This is the number of iterations used. 
            Defaults to 20. The time required to generate a CRB circuit is 
            linear in `citerations * (CRB length + 2)`. Lower-depth / lower 
            2-qubit gate count compilations of the Cliffords are important in 
            order to successfully implement CRB on more qubits.

    Returns:
        Callable: CRB class instance.
    """

    class CRB(qpu):
        """pyGSTi Clifford RB protocol."""

        def __init__(self,
                config:          Config,
                qubits:          List[int] | Tuple[int],
                circuit_depths:  List[int] | Tuple[int],
                n_circuits:      int = 30,
                native_gates:    List[str] = ['X90', 'Y90'],
                pspec:           Any | None = None,
                randomizeout:    bool = True,
                citerations:     int = 20,
                **kwargs
            ) -> None:

            try:
                import pygsti
                from pygsti.processors import CliffordCompilationRules as CCR
                from qcal.interface.pygsti.transpiler import Transpiler
                logger.info(f" pyGSTi version: {pygsti.__version__}\n")
            except ImportError:
                logger.warning(' Unable to import pyGSTi!')
            
            self._qubits = qubits
            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._native_gates = native_gates
            self._pspec = pspec
            self._randomizeout = randomizeout
            self._citerations = citerations

            transpiler = kwargs.get('transpiler', Transpiler())
            kwargs.pop('transpiler', None)
            qpu.__init__(self,
                config=config, 
                transpiler=transpiler,
                **kwargs
            )

            if pspec is None:
                from qcal.interface.pygsti.processor_spec import pygsti_pspec
                self._pspec = pygsti_pspec(config, qubits, native_gates)

            self._compilations = {
                'absolute': CCR.create_standard(
                    self._pspec, 'absolute', 
                    ('paulis', '1Qcliffords'), 
                    verbosity=0
                ),            
                'paulieq': CCR.create_standard(
                    self._pspec, 'paulieq', 
                    ('1Qcliffords', 'allcnots'), 
                    verbosity=0
                )
            }

            self._protocol = pygsti.protocols.RB() 
            self._edesign = None
            self._data = None
            self._results = None

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
            """Generate all pyGSTi clifford RB circuits."""
            logger.info(' Generating circuits from pyGSTi...')
            import pygsti
            from qcal.interface.pygsti.circuits import load_circuits

            self._edesign = pygsti.protocols.CliffordRBDesign(
                pspec=self._pspec,
                clifford_compilations=self._compilations, 
                depths=self._circuit_depths, 
                circuits_per_depth=self._n_circuits, 
                qubit_labels=self._pspec.qubit_labels, 
                randomizeout=self._randomizeout,
                citerations=self._citerations
                )
            
            # Save an empty dataset file of all the circuits
            self._data_manager._exp_id += (
                f'_CRB_Q{"".join(str(q) for q in self._qubits)}'
            )
            self._data_manager.create_data_path()
            pygsti.io.write_empty_protocol_data(
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
            clear_output(wait=True)
            from qcal.interface.pygsti.datasets import generate_pygsti_dataset
            generate_pygsti_dataset(
                self._transpiled_circuits,
                self._data_manager._save_path + 'data/'
            )
            if settings.Settings.save_data:
                qpu.save(self, create_data_path=False)

        def analyze(self):
            """Analyze the CRB results."""
            # logger.info(' Analyzing the results...')
            import pygsti
            try:
                self._data = pygsti.io.read_data_from_dir(
                     self._data_manager._save_path
                )
                self._results = self._protocol.run(self._data)
                r = self._results.fits['full'].estimates['r']
                rstd = self._results.fits['full'].stds['r']
                rAfix = self._results.fits['A-fixed'].estimates['r']
                rAfixstd = self._results.fits['A-fixed'].stds['r']
                print(
                    f"e_F = {r:1.2e} ({rstd:1.2e}) (fit with a free asymptote)"
                )
                print(
                    f"e_F = {rAfix:1.2e} ({rAfixstd:1.2e}) (fit with the "
                    "asymptote fixed to 1/2^n)"
                )
            except Exception:
                logger.warning(' Unable to fit the RB data!')

        def plot(self) -> None:
            """Plot the CRB fit results."""
            if self._results is not None:
                if settings.Settings.save_data:
                    self._results.plot(
                        figpath=self._data_manager._save_path + 'CRB_decay.png'
                    )
                else:
                    self._results.plot()

        def final(self) -> None:
            """Final benchmarking method."""
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.save()
            self.analyze() 
            self.plot()
            self.final()

    return CRB(
        config,
        qubits,
        circuit_depths,
        n_circuits,
        native_gates,
        pspec,
        randomizeout,
        citerations,
        **kwargs
    )


def SRB(qpu:             QPU,
        config:          Config,
        qubit_labels:    Iterable[int],
        circuit_depths:  List[int] | Tuple[int],
        tq_config:       str | Any = None,
        compiler:        Any | None = None, 
        transpiler:      Any | None = None,
        classifier:      ClassificationManager = None,
        n_circuits:      int = 30,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        compiled_pauli:  bool = True,
        include_rcal:    bool = False,
        raster_circuits: bool = False,
        **kwargs
    ) -> Callable:
    """Streamlined Randomized Benchmarking.

    This is a True-Q protocol and requires a valid True-Q license.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (Iterable[int, Iterable[int]]): a list specifying sets of 
            system labels to be twirled together by Clifford gates in each 
            circuit. For example, [0, 1, (2, 3)] would perform single-qubit RB
            on 0 and 1, and two-qubit RB on (2, 3).
        circuit_depths (List[int] | Tuple[int]): a list of positive integers 
            specifying how many cycles of random Clifford gates to generate for
            RB, for example, [4, 64, 256].
        tq_config (str | Any, optional): True-Q config yaml file or config
            object. Defaults to None.
        compiler (Any | Compiler | None, optional): custom compiler to compile
            the True-Q circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to transpile
            the True-Q circuits to experimental circuits. Defaults to None.
        classifier (ClassificationManager, optional): manager used for 
            classifying raw data. Defaults to None.
        n_circuits (int, optional): the number of circuits for each circuit 
            depth. Defaults to 30.
        n_shots (int, optional): number of measurements per circuit. 
                Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. Defaults
            to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that can be
            measured per sequence. Defaults to 1.
        n_levels (int, optional): number of energy levels to be measured. 
            Defaults to 2. If n_levels = 3, this assumes that the
            measurement supports qutrit classification.
        compiled_pauli (bool, optional): whether or not to compile a random 
            Pauli gate for each qubit in the cycle preceding a measurement 
            operation. Defaults to True.
        include_rcal (bool, optional): whether to measure RCAL circuits in the
            same circuit collection as the SRB circuit. Defaults to False. If
            True, readout correction will be apply to the fit results 
            automatically.
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.

    Returns:
        Callable: SRB class instance.
    """

    class SRB(qpu):
        """True-Q SRB protocol."""
        import trueq as tq

        def __init__(self,
                config:          Config,
                qubit_labels:    Iterable[int],
                circuit_depths:  List[int] | Tuple[int],
                tq_config:       str | tq.Config = None,
                compiler:        Any | None = None, 
                transpiler:      Any | None = None,
                classifier:      ClassificationManager = None,
                n_circuits:      int = 30,
                n_shots:         int = 1024,
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2,
                compiled_pauli:  bool = True,
                include_rcal:    bool = False,
                raster_circuits: bool = False,
                **kwargs
            ) -> None:
            from qcal.interface.trueq.compiler import Compiler
            from qcal.interface.trueq.transpiler import Transpiler
            
            try:
                import trueq as tq
                logger.info(f" True-Q version: {tq.__version__}")
            except ImportError:
                logger.warning(' Unable to import trueq!')
            
            self._qubit_labels = qubit_labels
            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._compiled_pauli = compiled_pauli
            self._include_rcal = include_rcal
            
            if compiler is None:
                compiler = Compiler(config if tq_config is None else tq_config)
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
                n_levels=n_levels,
                raster_circuits=raster_circuits,
                **kwargs
            )

        def generate_circuits(self):
            """Generate all True-Q SRB circuits."""
            logger.info(' Generating circuits from True-Q...')
            import trueq as tq

            self._circuits = tq.make_srb(
                self._qubit_labels,
                self._circuit_depths,
                self._n_circuits,
                self._compiled_pauli
            )

            if self._include_rcal:
                self._circuits.append(tq.make_rcal(self._circuits.labels))

        def analyze(self):
            """Analyze the SRB results."""
            logger.info(' Analyzing the results...')
            print('')
            try:
                print(self._circuits.fit(analyze_dim=2))
            except Exception:
                logger.warning(' Unable to fit the estimate collection!')

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_SRB_Q{"".join(str(q) for q in self._circuits.labels)}'
            )
            if settings.Settings.save_data:
                qpu.save(self) 

        def plot(self) -> None:
            """Plot the SRB fit results."""

            # Plot the raw curves
            nrows, ncols = calculate_nrows_ncols(len(self._qubit_labels))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )
            
            if isinstance(axes, np.ndarray):
                self._circuits.plot.raw(axes=axes.ravel())
            else:
                self._circuits.plot.raw(axes=axes)

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
                        ax.set_title(ax.get_title(), fontsize=20)
                        ax.xaxis.get_label().set_fontsize(15)
                        ax.yaxis.get_label().set_fontsize(15)
                        ax.tick_params(
                            axis='both', which='major', labelsize=12
                        )
                        ax.legend(prop=dict(size=12))
                        ax.grid(True)

                    else:
                        ax.axis('off')
                
            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'SRB_decays.png', 
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'SRB_decays.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'SRB_decays.svg'
                )
            plt.show()

            # Plot the RB infidelities
            figsize = (5 * ncols, 4)
            fig, ax = plt.subplots(figsize=figsize, layout='constrained')

            self._circuits.plot.compare_rb(axes=ax)
            ax.set_title(ax.get_title(), fontsize=18)
            ax.xaxis.get_label().set_fontsize(15)
            ax.yaxis.get_label().set_fontsize(15)
            ax.tick_params(
                axis='both', which='major', labelsize=12
            )
            ax.legend(prop=dict(size=12))
            ax.grid(True)

            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'SRB_infidelities.png', 
                    dpi=300
                )
            plt.show()

        def final(self) -> None:
            """Final benchmarking method."""
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.save()
            self.analyze() 
            self.plot()
            self.final()

    return SRB(
        config,
        qubit_labels,
        circuit_depths,
        tq_config,
        compiler,
        transpiler,
        classifier,
        n_circuits,
        n_shots,
        n_batches,
        n_circs_per_seq,
        n_levels,
        compiled_pauli,
        include_rcal,
        raster_circuits,
        **kwargs
    )