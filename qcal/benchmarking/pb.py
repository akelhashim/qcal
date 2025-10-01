"""Submodule for purity benchmarking routines.

For XRB, see:
https://trueq.quantumbenchmark.com/guides/error_diagnostics/xrb.html
"""
import qcal.settings as settings

from qcal.analysis.leakage import analyze_leakage
from qcal.config import Config
from qcal.qpu.qpu import QPU
from qcal.plotting.utils import calculate_nrows_ncols

import logging
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import clear_output
from typing import Any, Callable, List, Tuple


logger = logging.getLogger(__name__)


def XRB(qpu:             QPU,
        config:          Config,
        qubit_labels:    List[int | Tuple[int]],
        circuit_depths:  List[int] | Tuple[int],
        n_circuits:      int = 30,
        tq_config:       str | Any = None,
        compiled_pauli:  bool = True,
        include_rcal:    bool = False,
        include_srb:     bool = True,
        **kwargs
    ) -> Callable:
    """eXtended Randomized Benchmarking.

    This is a True-Q protocol and requires a valid True-Q license.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_labels (List[int | Tuple[int]]): a list specifying sets of 
            system labels to be twirled together by Clifford gates in each 
            circuit. For example, [0, 1, (2, 3)] would perform single-qubit RB
            on 0 and 1, and two-qubit RB on (2, 3).
        circuit_depths (List[int] | Tuple[int]): a list of positive integers 
            specifying how many cycles of random Clifford gates to generate for
            RB, for example, [4, 64, 256].
        n_circuits (int, optional): the number of circuits for each circuit 
            depth. Defaults to 30.
        tq_config (str | Any, optional): True-Q config yaml file or config
            object. Defaults to None.
        compiled_pauli (bool, optional): whether or not to compile a random 
            Pauli gate for each qubit in the cycle preceding a measurement 
            operation. Defaults to True. This is only used in SRB, if included.
        include_rcal (bool, optional): whether to measure RCAL circuits in the
            same circuit collection as the SRB circuit. Defaults to False. If
            True, readout correction will be apply to the fit results 
            automatically.
        include_srb (bool, optional): whether to measure SRB circuits in
            addition to XRB circuits. Defaults to True. Together, XRB and SRB 
            can be used to estimate the fraction of the total error due to 
            coherent and stochastic errors in the gate set.

    Returns:
        Callable: XRB class instance.
    """

    class XRB(qpu):
        """True-Q XRB protocol."""
        import trueq as tq

        def __init__(self,
                config:          Config,
                qubit_labels:    List[int | Tuple[int]],
                circuit_depths:  List[int] | Tuple[int],
                n_circuits:      int = 30,
                tq_config:       str | tq.Config = None,
                compiled_pauli:  bool = True,
                include_rcal:    bool = False,
                include_srb:     bool = True,
                **kwargs
            ) -> None:
            from qcal.interface.trueq.compiler import TrueqCompiler
            from qcal.interface.trueq.transpiler import TrueqTranspiler
            
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
            self._include_srb = include_srb

            compiler = kwargs.get(
                'compiler', 
                TrueqCompiler(config if tq_config is None else tq_config)
            )
            kwargs.pop('compiler', None)

            transpiler = kwargs.get('transpiler', TrueqTranspiler())
            kwargs.pop('transpiler', None)
                
            qpu.__init__(self,
                config=config, 
                compiler=compiler, 
                transpiler=transpiler,
                **kwargs
            )

        def generate_circuits(self):
            """Generate all True-Q SRB circuits."""
            logger.info(' Generating circuits from True-Q...')
            import trueq as tq

            self._circuits = tq.make_xrb(
                self._qubit_labels,
                self._circuit_depths,
                self._n_circuits
            )
            
            if self._include_srb:
                self._circuits += tq.make_srb(
                self._qubit_labels,
                self._circuit_depths,
                self._n_circuits,
                self._compiled_pauli
            )

            if self._include_rcal:
                self._circuits += tq.make_rcal(self._circuits.labels)

            self._circuits.shuffle()

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
                f'_XRB_{"".join("Q"+str(q) for q in self._circuits.labels)}'
            )
            if settings.Settings.save_data:
                qpu.save(self) 

        def plot(self) -> None:
            """Plot the XRB fit results."""

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
                    self._data_manager._save_path + 'XRB_decays.png', 
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'XRB_decays.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'XRB_decays.svg'
                )
            plt.show()

            # Plot the XRB infidelities
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
                    self._data_manager._save_path + 'XRB_infidelities.png', 
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'XRB_infidelities.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'XRB_infidelities.svg'
                )
            plt.show()

            if any(
                res.dim == 3 for res in 
                self._circuits.subset(protocol='XRB').results
            ):
                analyze_leakage(
                    self._circuits.subset(protocol='XRB'), 
                    filename=self._data_manager._save_path + 'XRB_'
                )

            if self._include_srb and any(
                res.dim == 3 for res in 
                self._circuits.subset(protocol='SRB').results
            ):
                analyze_leakage(
                    self._circuits.subset(protocol='SRB'), 
                    filename=self._data_manager._save_path + 'SRB_'
                )

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

    return XRB(
        config=config,
        qubit_labels=qubit_labels,
        circuit_depths=circuit_depths,
        n_circuits=n_circuits,
        tq_config=tq_config,
        compiled_pauli=compiled_pauli,
        include_rcal=include_rcal,
        include_srb=include_srb,
        **kwargs
    )