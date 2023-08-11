"""Submodule for RB routines.

"""
import qcal.settings as settings

from qcal.config import Config
from qcal.qpu.qpu import QPU
from qcal.plotting.utils import calculate_nrows_ncols

import logging
import matplotlib.pyplot as plt

from IPython.display import clear_output
from typing import Any, Callable, List, Tuple, Iterable

logger = logging.getLogger(__name__)


# TODO: add leakage
def SRB(qpu:             QPU,
        config:          Config,
        qubit_labels:    Iterable[int],
        circuit_depths:  List[int] | Tuple[int],
        tq_config:       str | Any = None,
        compiler:        Any | None = None, 
        transpiler:      Any | None = None,
        n_circuits:      int = 30,
        n_shots:         int = 1024, 
        n_batches:       int = 1, 
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        compiled_pauli:  bool = True,
        include_rcal:    bool = True,
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
            same circuit collection as the SRB circuit. Defaults to True. If
            True, readout correction will be apply to the fit results 
            automatically.

    Returns:
        Callable: SRB class instance.
    """

    class SRB(qpu):
        """True-Q SRB protocol."""
        import trueq as tq

        def __init__(self,
                qpu:             QPU,
                config:          Config,
                qubit_labels:    Iterable[int],
                circuit_depths:  List[int] | Tuple[int],
                tq_config:       str | tq.Config = None,
                compiler:        Any | None = None, 
                transpiler:      Any | None = None,
                n_circuits:      int = 30,
                n_shots:         int = 1024,
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                n_levels:        int = 2,
                compiled_pauli:  bool = True,
                include_rcal:    bool = True,
                **kwargs
            ) -> None:
            from qcal.interface.trueq.compiler import Compiler
            from qcal.interface.trueq.transpiler import Transpiler
            
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
                config, 
                compiler, 
                transpiler, 
                n_shots, 
                n_batches, 
                n_circs_per_seq, 
                n_levels,
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
            print(self._circuits.fit())

        def plot(self) -> None:
            """Plot the SRB fit results."""

            # Plot the raw curves
            nrows, ncols = calculate_nrows_ncols(len(self._qubit_labels))
            figsize = (5 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, layout='constrained'
            )

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
                    dpi=300
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

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_SRB_Q{"".join(str(q) for q in self._circuits.labels)}'
            )
            if settings.Settings.save_data:
                self.save()
            self.analyze()
            self.plot()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")


    return SRB(
        qpu,
        config,
        qubit_labels,
        circuit_depths,
        tq_config,
        compiler,
        transpiler,
        n_circuits,
        n_shots,
        n_batches,
        n_circs_per_seq,
        n_levels,
        compiled_pauli,
        include_rcal,
        **kwargs
    )