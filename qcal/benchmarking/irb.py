"""Submodule for interleaved randomized benchmarking routines.

For IRB, see:
https://trueq.quantumbenchmark.com/guides/error_diagnostics/irb.html
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
from typing import Any, Callable, Dict, List, Tuple


logger = logging.getLogger(__name__)


def IRB(qpu:                  QPU,
        config:               Config,
        cycle:                Dict,
        qubit_labels:         List[int | Tuple[int]],
        circuit_depths:       List[int] | Tuple[int],
        n_circuits:           int = 30,
        tq_config:            str | Any = None,
        twirl:                str = None, 
        propogate_correction: bool = False,
        compiled_pauli:       bool = True,
        include_rcal:         bool = False,
        include_srb:          bool = True,
        **kwargs
    ) -> Callable:
    """Interleaved Randomized Benchmarking.

    This is a True-Q protocol and requires a valid True-Q license.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        cycle (Dict, tq.Cycle): cycle (or subcircuit) to benchmark.
        circuit_depths (List[int] | Tuple[int]): a list of positive integers 
            specifying how many cycles of random Clifford gates to generate for
            IRB, for example, [4, 64, 256].
        n_circuits (int, optional): the number of circuits for each circuit 
            depth. Defaults to 30.
        tq_config (str | Any, optional): True-Q config yaml file or config
            object. Defaults to None.
        twirl (tq.Twirl, str, optional): The Twirl to use in this protocol. 
            Defaults to 'P'. You can also specify a twirling group that will be 
            used to automatically instantiate a twirl based on the labels in 
            the given cycles.
        propagate_correction (bool, optional): whether to propagate correction 
            gates to the end of the circuit or compile them into neighbouring 
            cycles. Defaults to False. Warning: this can result in arbitrary 
            multi-qubit gates at the end of the circuit!
        compiled_pauli (bool, optional): whether or not to compile a random 
            Pauli gate for each qubit in the cycle preceding a measurement 
            operation. Defaults to True.
        include_rcal (bool, optional): whether to measure RCAL circuits in the
            same circuit collection as the SRB circuit. Defaults to False. If
            True, readout correction will be apply to the fit results 
            automatically.
        include_srb (bool, optional): whether to measure SRB circuits in
            addition to IRB circuits. Defaults to True. Together, IRB and SRB 
            can be used to estimate interleaved gate error.

    Returns:
        Callable: IRB class instance.
    """

    class IRB(qpu):
        """True-Q IRB protocol."""

        def __init__(self,
                config:               Config,
                cycle:                Dict,
                circuit_depths:       List[int] | Tuple[int],
                n_circuits:           int = 30,
                tq_config:            str | Any = None,
                twirl:                str = None,
                qubit_labels:         List[int | Tuple[int]] = None, 
                propogate_correction: bool = False,
                compiled_pauli:       bool = True,
                include_rcal:         bool = False,
                include_srb:          bool = True,
                **kwargs
            ) -> None:
            from qcal.interface.trueq.compiler import TrueqCompiler
            from qcal.interface.trueq.transpiler import TrueqTranspiler
            
            try:
                import trueq as tq
                logger.info(f" True-Q version: {tq.__version__}")
            except ImportError:
                logger.warning(' Unable to import trueq!')

            if include_srb and not qubit_labels:
                raise ValueError(
                    "qubit_labels must be specified if include_srb!"
                )
            
            self._cycle = cycle
            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._twirl = twirl
            self._qubit_labels = qubit_labels
            self._propogate_correction = propogate_correction
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
            """Generate all True-Q IRB circuits."""
            logger.info(' Generating circuits from True-Q...')
            import trueq as tq

            self._circuits = tq.make_irb(
                cycles=self._cycle,
                n_random_cycles=self._circuit_depths,
                n_circuits=self._n_circuits,
                twirl=self._twirl,
                propagate_correction=self._propogate_correction,
                compiled_pauli=self._compiled_pauli
            )
            
            if self._include_srb:
                self._circuits += tq.make_srb(
                    labels=self._qubit_labels,
                    n_random_cycles=self._circuit_depths,
                    n_circuits=self._n_circuits,
                    compiled_pauli=self._compiled_pauli
                )

            if self._include_rcal:
                self._circuits += tq.make_rcal(self._circuits.labels)

            self._circuits.shuffle()

        def analyze(self):
            """Analyze the IRB results."""
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
                f'_IRB_{"".join("Q" + str(q) for q in self._circuits.labels)}'
            )
            if settings.Settings.save_data:
                qpu.save(self) 

        def plot(self) -> None:
            """Plot the IRB fit results."""

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
                    self._data_manager._save_path + 'IRB_decays.png', 
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'IRB_decays.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'IRB_decays.svg'
                )
            plt.show()

            # Plot the IRB summary
            figsize = (5 * ncols, 4)
            fig, ax = plt.subplots(figsize=figsize, layout='constrained')

            self._circuits.plot.irb_summary(axes=ax)
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
                    self._data_manager._save_path + 'IRB_summary.png', 
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'IRB_summary.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'IRB_summary.svg'
                )
            plt.show()

            if any(
                res.dim == 3 for res in 
                self._circuits.subset(protocol='IRB').results
            ):
                analyze_leakage(
                    self._circuits.subset(protocol='IRB'), 
                    filename=self._data_manager._save_path + 'IRB_'
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

    return IRB(
        config=config,
        cycle=cycle,
        circuit_depths=circuit_depths,
        n_circuits=n_circuits,
        tq_config=tq_config,
        twirl=twirl,
        qubit_labels=qubit_labels,
        propogate_correction=propogate_correction,
        compiled_pauli=compiled_pauli,
        include_rcal=include_rcal,
        include_srb=include_srb,
        **kwargs
    )