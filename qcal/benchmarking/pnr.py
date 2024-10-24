"""Submodule for Pauli Noise Reconstruction.

"""
import qcal.settings as settings

from qcal.config import Config
from qcal.managers.classification_manager import ClassificationManager
from qcal.qpu.qpu import QPU

import logging
import matplotlib.pyplot as plt

from IPython.display import clear_output
from typing import Any, Callable, Dict, List, Tuple


logger = logging.getLogger(__name__)


def KNR(qpu:                  QPU,
        config:               Config,
        cycle:                Dict,
        circuit_depths:       List[int] | Tuple[int],
        tq_config:            str | Any = None,
        n_circuits:           int = 30,
        n_subsystems:         int = 2,
        twirl:                str = 'P',
        propogate_correction: bool = False,
        compiled_pauli:       bool = True,
        include_rcal:         bool = False,
        **kwargs
    ) -> Callable:
    """K-body Noise Reconstruction.

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
        n_subsystems (int, optional): a positive integer specifying the number
            of body errors to estimate on the subset of qubits in the cycle. 
            Defaults to 2.
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

    Returns:
        Callable: KNR class instance.
    """

    class KNR(qpu):
        """True-Q KNR protocol."""
        import trueq as tq

        def __init__(self,
                qpu:                  QPU,
                config:               Config,
                cycle:                Dict,
                circuit_depths:       List[int] | Tuple[int],
                tq_config:            str | Any = None,
                n_circuits:           int = 30,
                n_subsystems:         int = 2,
                twirl:                str = 'P',
                propogate_correction: bool = False,
                compiled_pauli:       bool = True,
                include_rcal:         bool = False,
                **kwargs
            ) -> None:
            from qcal.interface.trueq.compiler import TrueqCompiler
            from qcal.interface.trueq.transpiler import TrueqTranspiler
            
            import trueq as tq
            print(f"True-Q version: {tq.__version__}\n")

            self._cycle = cycle
            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._n_subsystems = n_subsystems
            self._twirl = twirl
            self._propagate_correction = propogate_correction
            self._compiled_pauli = compiled_pauli
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
            """Generate all True-Q KNR circuits."""
            logger.info(' Generating circuits from True-Q...')
            import trueq as tq

            self._circuits = tq.make_knr(
                cycles=self._cycle,
                n_random_cycles=self._circuit_depths,
                n_circuits=self._n_circuits,
                subsystems=self._n_subsystems,
                twirl=self._twirl, 
                propagate_correction=self._propagate_correction, 
                compiled_pauli=self._compiled_pauli
            )

            if self._include_rcal:
                self._circuits.append(tq.make_rcal(self._circuits.labels))

        def analyze(self):
            """Analyze the KNR results."""
            logger.info(' Analyzing the results...')
            print('')
            try:
                for fit in self._circuits.fit(analyze_dim=2):
                    print(fit)
            except Exception:
                logger.warning(' Unable to fit the estimate collection!')

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_KNR_Q{"".join(str(q) for q in self._circuits.labels)}'
            )
            if settings.Settings.save_data:
                qpu.save(self) 

        def plot(self) -> None:
            """Plot the KNR fit results."""
            # Plot the raw curves
            self._circuits.plot.raw()
            fig = plt.gcf()
            for ax in fig.axes:
                ax.set_title(ax.get_title(), fontsize=20)
                ax.xaxis.get_label().set_fontsize(15)
                ax.yaxis.get_label().set_fontsize(15)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.grid(True)

            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'KNR_decays.png', 
                    dpi=300
                )
            plt.show()

            # Plot the KNR infidelities
            self._circuits.plot.compare_pauli_infidelities()
            fig = plt.gcf()
            for ax in fig.axes:
                ax.set_title(ax.get_title(), fontsize=20)
                ax.xaxis.get_label().set_fontsize(15)
                ax.yaxis.get_label().set_fontsize(15)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.grid(True)

            fig.set_tight_layout(True)
            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'KNR_infidelities.png', 
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + 'KNR_infidelities.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'KNR_infidelities.svg'
                )
            plt.show()

            # Plot the KNR heatmap
            self._circuits.plot.knr_heatmap()
            fig = plt.gcf()
            fig.set_size_inches((8, 8))

            if settings.Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + 'KNR_heatmap.png', 
                    dpi=300
                )
                fig.savefig(
                    self._data_manager._save_path + 'KNR_heatmap.pdf'
                )
                fig.savefig(
                    self._data_manager._save_path + 'KNR_heatmap.svg'
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

    return KNR(
        qpu,
        config,
        cycle,
        circuit_depths,
        tq_config,
        n_circuits,
        n_subsystems,
        twirl,
        propogate_correction,
        compiled_pauli,
        include_rcal,
        **kwargs
    )