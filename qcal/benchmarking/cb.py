"""Submodule for cycle benchmarking.

See:
https://www.nature.com/articles/s41467-019-13068-7
https://trueq.quantumbenchmark.com/guides/error_diagnostics/cb.html
https://trueq.quantumbenchmark.com/api/protocols.html#trueq.make_cb
"""
import qcal.settings as settings

from qcal.config import Config
from qcal.managers.classification_manager import ClassificationManager
from qcal.math.utils import round_to_order_error
from qcal.qpu.qpu import QPU
from qcal.plotting.utils import calculate_nrows_ncols

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

    e_C = (1 - F_D/F_ref) * (d - 1)/d
    err = np.sqrt( 
        (err_D/F_D)**2 + (err_ref/F_ref)**2 
    ) * (F_D/F_ref) * (d - 1)/d

    e_C, err = round_to_order_error(e_C, err)

    return (e_C, err)


def CB(qpu:                  QPU,
       config:               Config,
       cycle:                Dict,
       circuit_depths:       List[int] | Tuple[int],
       tq_config:            str | Any = None,
       compiler:             Any | None = None, 
       transpiler:           Any | None = None,
       classifier:           ClassificationManager = None,
       n_circuits:           int = 30,
       n_decays:             int = 20,
       n_shots:              int = 1024, 
       n_batches:            int = 1, 
       n_circs_per_seq:      int = 1,
       n_levels:             int = 2,
       targeted_errors:      Iterable[str] | None = None,
       twirl:                str = 'P',
       propogate_correction: bool = False,
       compiled_pauli:       bool = True,
       include_ref_cycle:    bool = False,
       include_rcal:         bool = False,
       raster_circuits:      bool = False,
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
        compiler (Any | Compiler | None, optional): custom compiler to compile
            the True-Q circuits. Defaults to None.
        transpiler (Any | None, optional): custom transpiler to transpile
            the True-Q circuits to experimental circuits. Defaults to None.
        classifier (ClassificationManager, optional): manager used for 
            classifying raw data. Defaults to None.
        n_circuits (int, optional): the number of circuits for each circuit 
            depth. Defaults to 30.
        n_decays (int, optional): an integer specifying the total number of 
            randomly  chosen Pauli decay strings used to measure the process  
            infidelity or the probability of each error. Defaults to 20. 
            Warning: Setting this value lower than min(20, 4 ** n_qubits - 1) 
            may result in a biased estimate of the process fidelity, and 
            setting this value lower than min(40, 4 ** n_qubits - 1) may result 
            in a biased estimate of the probability for non-identity errors.
        n_shots (int, optional): number of measurements per circuit. 
                Defaults to 1024.
        n_batches (int, optional): number of batches of measurements. Defaults
            to 1.
        n_circs_per_seq (int, optional): maximum number of circuits that can be
            measured per sequence. Defaults to 1.
        n_levels (int, optional): number of energy levels to be measured. 
            Defaults to 2. If n_levels = 3, this assumes that the
            measurement supports qutrit classification.
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
        raster_circuits (bool, optional): whether to raster through all
            circuits in a batch during measurement. Defaults to False. By
            default, all circuits in a batch will be measured n_shots times
            one by one. If True, all circuits in a batch will be measured
            back-to-back one shot at a time. This can help average out the 
            effects of drift on the timescale of a measurement.

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
                compiler:             Any | None = None, 
                transpiler:           Any | None = None,
                classifier:           ClassificationManager = None,
                n_circuits:           int = 30,
                n_decays:             int = 20,
                n_shots:              int = 1024, 
                n_batches:            int = 1, 
                n_circs_per_seq:      int = 1,
                n_levels:             int = 2,
                targeted_errors:      Iterable[str] | None = None,
                twirl:                str = 'P',
                propogate_correction: bool = False,
                compiled_pauli:       bool = True,
                include_ref_cycle:    bool = False,
                include_rcal:         bool = False,
                raster_circuits:      bool = False,
                **kwargs
            ) -> None:
            from qcal.interface.trueq.compiler import Compiler
            from qcal.interface.trueq.transpiler import Transpiler
            
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
                self._circuits.append(
                    tq.make_cb(
                        cycles=tq.Cycle({}),
                        n_random_cycles=self._circuit_depths,
                        n_circuits=self._n_circuits,
                        n_decays=self._n_decays, 
                        targeted_errors=self._targeted_errors, 
                        twirl=tq.Twirl('P', self._circuits.labels),  
                        compiled_pauli=self._compiled_pauli
                    )
                )

            if self._include_rcal:
                self._circuits.append(tq.make_rcal(self._circuits.labels))

        def analyze(self):
            """Analyze the SRB results."""
            logger.info(' Analyzing the results...')
            import trueq as tq

            print('')
            if self._include_ref_cycle:
                cycle_subset = self._circuits.subset(cycles=[(tq.Cycle({}),)])
                ref_subset = self._circuits.subset(
                        cycles=[(tq.Cycle(self._cycle),)]
                )
                print(cycle_subset.fit(analyze_dim=2))
                print(ref_subset.fit(analyze_dim=2))

                e_C, err = compute_cycle_infidelity(cycle_subset, ref_subset)
                print(f'Interleaved cycle infidelity: e_C = {e_C} ({err})')

            else:
                print(self._circuits.fit(analyze_dim=2))

        def plot(self) -> None:
            """Plot the SRB fit results."""
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
                        dpi=300
                    )
                plt.show()

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_CB_Q{"".join(str(q) for q in self._circuits.labels)}'
            )
            if settings.Settings.save_data:
                self.save() 
            self.analyze()
            self.plot()
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")


    return CB(
        qpu,
        config,
        cycle,
        circuit_depths,
        tq_config,
        compiler,
        transpiler,
        classifier,
        n_circuits,
        n_decays,
        n_shots,
        n_batches,
        n_circs_per_seq,
        n_levels,
        targeted_errors,
        twirl,
        propogate_correction,
        compiled_pauli,
        include_ref_cycle,
        include_rcal,
        raster_circuits,
        **kwargs
    )