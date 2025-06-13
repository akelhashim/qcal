"""Submodule for cycle benchmarking.

See:
https://www.nature.com/articles/s41467-019-13068-7
https://trueq.quantumbenchmark.com/guides/error_diagnostics/cb.html
https://trueq.quantumbenchmark.com/api/protocols.html#trueq.make_cb
"""
import qcal.settings as settings

from qcal.analysis.leakage import analyze_leakage
from qcal.config import Config
from qcal.math.utils import round_to_order_error
from qcal.qpu.qpu import QPU

import logging
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import clear_output
from typing import Any, Callable, Dict, List, Tuple, Iterable


logger = logging.getLogger(__name__)


def compute_cycle_infidelity(circs_D, circs_ref) -> Tuple:
    """Compute the infidelity of the interleaved Cycle.

    Args:
        circs_D (tq.CircuitCollection): dressed circuits
        circs_ref (tq.CircuitCollection): reference circuits

    Returns:
        Tuple: (cycle infidelity, error)
    """
    n_qubits = len(circs_D.labels)
    d = 2 ** n_qubits
    F_D = 1 - circs_D.fit(analyze_dim=2)[0].e_F.val
    F_ref = 1 - circs_ref.fit(analyze_dim=2)[0].e_F.val
    err_D = circs_D.fit(analyze_dim=2)[0].e_F.std
    err_ref = circs_ref.fit(analyze_dim=2)[0].e_F.std

    f_D = (d**2 * F_D - 1) / (d**2 - 1)
    f_ref = (d**2 * F_ref - 1) / (d**2 - 1)
    err_f_D = d**2 * err_D / (d**2 - 1)
    err_f_ref = d**2 * err_ref / (d**2 - 1)

    e_C = (d**2 - 1)/d**2 * (1 - f_D/f_ref)
    err_C = np.sqrt( 
        (err_f_D/f_D)**2 + (err_f_ref/f_ref)**2 
    ) * e_C

    e_C, err_C = round_to_order_error(e_C, err_C)

    return (e_C, err_C)


def CB(qpu:                  QPU,
       config:               Config,
       cycle:                Dict,
       circuit_depths:       List[int] | Tuple[int],
       tq_config:            str | Any = None,
       n_circuits:           int = 30,
       n_decays:             int = 20,
       targeted_errors:      Iterable[str] | None = None,
       twirl:                str = 'P',
       propogate_correction: bool = False,
       compiled_pauli:       bool = True,
       include_ref_cycle:    bool = False,
       include_rcal:         bool = False,
       **kwargs
    ) -> Callable:
    """Cycle Benchmarking.

    This is a True-Q protocol and requires a valid True-Q license.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        cycle (Dict, tq.Cycle): cycle (or subcircuit) to benchmark.
        circuit_depths (List[int] | Tuple[int]): a list of positive integers 
            specifying how many interleaved cycles of the target cycle and 
            random Pauli operators to generate, for example, [4, 16, 64].
        tq_config (str | Any, optional): True-Q config yaml file or config
            object. Defaults to None.
        n_circuits (int, optional): the number of circuits for each circuit 
            depth. Defaults to 30.
        n_decays (int, optional): an integer specifying the total number of 
            randomly  chosen Pauli decay strings used to measure the process  
            infidelity or the probability of each error. Defaults to 20. 
            Warning: Setting this value lower than min(20, 4 ** n_qubits - 1) 
            may result in a biased estimate of the process fidelity, and 
            setting this value lower than min(40, 4 ** n_qubits - 1) may result 
            in a biased estimate of the probability for non-identity errors.
        targeted_errors (tq.Weyls, Iterable[str], None, optional): A True-Q  
            Weyls instance, where each row specifies an error to measure. 
            Defaults to None. The identity Pauli will always be added to the 
            list of errors (or be the sole target if None is the argument),  
            which corresponds to measuring the process fidelity of the cycle.  
            For convenience, a list of strings can be given, e.g. ["XII", 
            "ZZY"], which will be used to instantiate a Weyls object.
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
        include_ref_cycle (bool, optional): whether to benchmark the reference
            cycle for the qubits in the target cycle. Defaults to False. This 
            is useful when one wants to benchmark the process fidelity of the 
            interleaved cycle, as opposed to the dressed cycle.
        include_rcal (bool, optional): whether to measure RCAL circuits in the
            same circuit collection as the SRB circuit. Defaults to False. If
            True, readout correction will be apply to the fit results 
            automatically.

    Returns:
        Callable: CB class instance.
    """

    class CB(qpu):
        """True-Q CB protocol."""
        import trueq as tq

        def __init__(self,
                qpu:                  QPU,
                config:               Config,
                cycle:                Dict,
                circuit_depths:       List[int] | Tuple[int],
                tq_config:            str | Any = None,
                n_circuits:           int = 30,
                n_decays:             int = 20,
                targeted_errors:      Iterable[str] | None = None,
                twirl:                str = 'P',
                propogate_correction: bool = False,
                compiled_pauli:       bool = True,
                include_ref_cycle:    bool = False,
                include_rcal:         bool = False,
                **kwargs
            ) -> None:
            from qcal.interface.trueq.compiler import TrueqCompiler
            from qcal.interface.trueq.transpiler import TrueqTranspiler
            
            try:
                import trueq as tq
                logger.info(f" True-Q version: {tq.__version__}")
            except ImportError:
                logger.warning(' Unable to import trueq!')
            
            self._cycle = cycle
            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._n_decays = n_decays
            self._targeted_errors = targeted_errors
            self._twirl = twirl
            self._propagate_correction = propogate_correction
            self._compiled_pauli = compiled_pauli
            self._include_ref_cycle = include_ref_cycle
            self._include_rcal = include_rcal
            
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
            """Generate all True-Q CB circuits."""
            logger.info(' Generating circuits from True-Q...')
            import trueq as tq

            self._circuits = tq.make_cb(
                cycles=self._cycle,
                n_random_cycles=self._circuit_depths,
                n_circuits=self._n_circuits,
                n_decays=self._n_decays, 
                targeted_errors=self._targeted_errors, 
                twirl=self._twirl, 
                propagate_correction=self._propagate_correction, 
                compiled_pauli=self._compiled_pauli
            )

            if self._include_ref_cycle:
               self._circuits += tq.make_cb(
                        cycles=tq.Cycle({}),
                        n_random_cycles=self._circuit_depths,
                        n_circuits=self._n_circuits,
                        n_decays=self._n_decays, 
                        targeted_errors=self._targeted_errors, 
                        twirl=self._twirl,  
                        compiled_pauli=self._compiled_pauli
                    )

            if self._include_rcal:
                self._circuits += tq.make_rcal(self._circuits.labels)

            self._circuits.shuffle()

        def analyze(self):
            """Analyze the CB results."""
            logger.info(' Analyzing the results...')
            import trueq as tq

            print('')
            if self._include_ref_cycle:
                cycle_subset = self._circuits.subset(cycles=[(tq.Cycle({}),)])
                ref_subset = self._circuits.subset(
                        cycles=[(tq.Cycle(self._cycle),)]
                )
                try:
                    print(cycle_subset.fit(analyze_dim=2))
                    print(ref_subset.fit(analyze_dim=2))
                    e_C, err = compute_cycle_infidelity(
                        cycle_subset, ref_subset
                    )
                    print(
                        f'Interleaved cycle infidelity: e_C = {e_C} ({err})\n'
                    )
                except Exception:
                    logger.warning(' Unable to fit the estimate collection!')

            else:
                try:
                    print(self._circuits.fit(analyze_dim=2))
                except Exception:
                    logger.warning(' Unable to fit the estimate collection!')

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CB_Q{"".join(str(q) for q in self._circuits.labels)}'
            )
            if settings.Settings.save_data:
                qpu.save(self) 

        def plot(self) -> None:
            """Plot the CB fit results."""
            # Plot the raw curves
            ncols = 2 if self._include_ref_cycle else 1
            figsize = (6 * ncols, 5)
            fig, axes = plt.subplots(
                1, ncols, figsize=figsize, layout='constrained'
            )
            self._circuits.plot.raw(axes=axes)
            for i in range(ncols):
                if ncols == 1:
                    ax = axes
                elif ncols == 2:
                    ax = axes[i]
                ax.set_title(ax.get_title(), fontsize=20)
                ax.xaxis.get_label().set_fontsize(15)
                ax.yaxis.get_label().set_fontsize(15)
                ax.tick_params(axis='both', which='major', labelsize=12)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[:5], labels[:5], fontsize=12)
                # ax.legend(prop=dict(size=12))
                ax.grid(True)

            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'CB_decays.png', 
                    dpi=300
                )
            plt.show()

            # Plot the CB infidelities
            nrows = 2 if self._include_ref_cycle else 1
            figsize = (8, 5 * nrows)
            fig, axes = plt.subplots(
                nrows, 1, figsize=figsize, layout='constrained'
            )
            self._circuits.plot.compare_pauli_infidelities(axes=axes)
            for i in range(nrows):
                if nrows == 1:
                    ax = axes
                elif nrows == 2:
                    ax = axes[i]
                ax.set_title(ax.get_title(), fontsize=18)
                ax.xaxis.get_label().set_fontsize(15)
                ax.yaxis.get_label().set_fontsize(15)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.legend(prop=dict(size=12))
                ax.grid(True)

            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'CB_infidelities.png', 
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'CB_infidelities.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'CB_infidelities.svg'
                )
            plt.show()

            if any(res.dim == 3 for res in self._circuits.results):
                analyze_leakage(
                    self._circuits, filename=self._data_manager._save_path
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

    return CB(
        qpu,
        config,
        cycle,
        circuit_depths,
        tq_config,
        n_circuits,
        n_decays,
        targeted_errors,
        twirl,
        propogate_correction,
        compiled_pauli,
        include_ref_cycle,
        include_rcal,
        **kwargs
    )