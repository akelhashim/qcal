"""Submodule for cycle benchmarking.

See:
https://www.nature.com/articles/s41467-019-13068-7
https://trueq.quantumbenchmark.com/guides/error_diagnostics/cb.html
https://trueq.quantumbenchmark.com/api/protocols.html#trueq.make_cb

For MCMs:
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.020602
https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.010310

"""
import logging
from collections.abc import Iterable, Sequence
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pygsti
import pygsti.algorithms.randomcircuit as random_circuit
from IPython.display import clear_output
from pygsti.baseobjs.label import Label
from pygsti.processors import QubitProcessorSpec
from pygsti.processors.compilationrules import CliffordCompilationRules as CCR

# from scipy.optimize import curve_fit
from qcal.analysis.leakage import analyze_leakage
from qcal.circuit import CircuitSet
from qcal.config import Config
from qcal.gate.two_qubit import TWO_QUBIT_GATES
from qcal.interface.pygsti.compiler import pauli_randomize_clifford_circuit
from qcal.interface.pygsti.processor_spec import pygsti_pspec
from qcal.interface.pygsti.transpiler import PyGSTiTranspiler
from qcal.math.utils import round_to_order_error
from qcal.qpu.qpu import QPU
from qcal.settings import Settings

from .utils import (
    generate_n_qubit_pauli_measurement_map,
    generate_n_qubit_paulis,
    generate_random_n_qubit_paulis,
)

logger = logging.getLogger(__name__)


__all__ = ['CB', 'CBMCM', 'SC']


def compute_cycle_infidelity(circs_D: Any, circs_ref: Any) -> tuple:
    """Compute the infidelity of the interleaved Cycle.

    Args:
        circs_D (tq.CircuitCollection): dressed circuits
        circs_ref (tq.CircuitCollection): reference circuits

    Returns:
        tuple: (cycle infidelity, error)
    """
    n_qubits = len(circs_D.labels)
    d = 2**n_qubits
    F_D = 1 - circs_D.fit(analyze_dim=2)[0].e_F.val
    F_ref = 1 - circs_ref.fit(analyze_dim=2)[0].e_F.val
    err_D = circs_D.fit(analyze_dim=2)[0].e_F.std
    err_ref = circs_ref.fit(analyze_dim=2)[0].e_F.std

    f_D = (d**2 * F_D - 1) / (d**2 - 1)
    f_ref = (d**2 * F_ref - 1) / (d**2 - 1)
    err_f_D = d**2 * err_D / (d**2 - 1)
    err_f_ref = d**2 * err_ref / (d**2 - 1)

    e_C = (d**2 - 1) / d**2 * (1 - f_D / f_ref)
    err_C = np.sqrt((err_f_D / f_D) ** 2 + (err_f_ref / f_ref) ** 2) * e_C

    e_C, err_C = round_to_order_error(e_C, err_C)

    return (e_C, err_C)


def CB(
    qpu:                  QPU,
    config:               Config,
    cycle:                dict,
    circuit_depths:       Iterable[int],
    tq_config:            str | Any = None,
    n_circuits:           int = 30,
    n_decays:             int = 20,
    targeted_errors:      Iterable[str] | None = None,
    twirl:                str = "P",
    propogate_correction: bool = False,
    compiled_pauli:       bool = True,
    include_ref_cycle:    bool = False,
    include_rcal:         bool = False,
    **kwargs,
) -> Callable:
    """Cycle Benchmarking.

    This is a True-Q protocol and requires a valid True-Q license.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        cycle (dict, tq.Cycle): cycle (or subcircuit) to benchmark.
        circuit_depths (Iterable[int]): a list of positive integers specifying
            how many interleaved cycles of the target cycle and
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
        targeted_errors (tq.math.Weyls, Iterable[str], None, optional): A True-Q
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

        # import trueq as tq

        def __init__(
            self,
            config:               Config,
            cycle:                dict,
            circuit_depths:       Iterable[int],
            tq_config:            str | Any = None,
            n_circuits:           int = 30,
            n_decays:             int = 20,
            targeted_errors:      Iterable[str] | None = None,
            twirl:                str = "P",
            propogate_correction: bool = False,
            compiled_pauli:       bool = True,
            include_ref_cycle:    bool = False,
            include_rcal:         bool = False,
            **kwargs,
        ) -> None:
            from qcal.interface.trueq.compiler import TrueqCompiler
            from qcal.interface.trueq.transpiler import TrueqTranspiler

            try:
                import trueq as tq
                logger.info(f" True-Q version: {tq.__version__}")
            except ImportError:
                logger.warning(" Unable to import trueq!")

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
                "compiler", TrueqCompiler(
                    config if tq_config is None else tq_config
                )
            )
            kwargs.pop("compiler", None)

            transpiler = kwargs.get("transpiler", TrueqTranspiler())
            kwargs.pop("transpiler", None)

            qpu.__init__(
                self, config=config, compiler=compiler, transpiler=transpiler,
                **kwargs
            )

        def generate_circuits(self):
            """Generate all True-Q CB circuits."""
            logger.info(" Generating circuits from True-Q...")
            import trueq as tq

            self._circuits = tq.make_cb(
                cycles=self._cycle,
                n_random_cycles=self._circuit_depths,
                n_circuits=self._n_circuits,
                n_decays=self._n_decays,
                targeted_errors=self._targeted_errors,
                twirl=self._twirl,
                propagate_correction=self._propagate_correction,
                compiled_pauli=self._compiled_pauli,
            )

            if self._include_ref_cycle:
                self._circuits += tq.make_cb(
                    cycles=tq.Cycle({}),
                    n_random_cycles=self._circuit_depths,
                    n_circuits=self._n_circuits,
                    n_decays=self._n_decays,
                    targeted_errors=self._targeted_errors,
                    twirl=self._twirl,
                    compiled_pauli=self._compiled_pauli,
                )

            if self._include_rcal:
                self._circuits += tq.make_rcal(self._circuits.labels)

            self._circuits.shuffle()

        def analyze(self):
            """Analyze the CB results."""
            logger.info(" Analyzing the results...")
            import trueq as tq

            print("")
            if self._include_ref_cycle:
                cycle_subset = self._circuits.subset(cycles=[(tq.Cycle({}),)])
                ref_subset = self._circuits.subset(
                    cycles=[
                        (self._cycle,)
                        if isinstance(self._cycle, tq.Cycle)
                        else (tq.Cycle(self._cycle),)
                    ]
                )
                try:
                    print(cycle_subset.fit(analyze_dim=2))
                    print(ref_subset.fit(analyze_dim=2))
                    e_C, err = compute_cycle_infidelity(
                        cycle_subset, ref_subset
                    )
                    print(
                        f"Interleaved cycle infidelity: e_C = {e_C} ({err})\n"
                    )
                except Exception:
                    logger.warning(" Unable to fit the estimate collection!")

            else:
                try:
                    print(self._circuits.fit(analyze_dim=2))
                except Exception:
                    logger.warning(" Unable to fit the estimate collection!")

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f"_CB{''.join('Q' + str(q) for q in self._circuits.labels)}"
            )
            if Settings.save_data:
                qpu.save(self)

        def plot(self) -> None:
            """Plot the CB fit results."""
            # Plot the raw curves
            ncols = 2 if self._include_ref_cycle else 1
            figsize = (6 * ncols, 5)
            fig, axes = plt.subplots(
                1, ncols, figsize=figsize, layout="constrained"
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
                ax.tick_params(axis="both", which="major", labelsize=12)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[:5], labels[:5], fontsize=12)
                # ax.legend(prop=dict(size=12))
                ax.grid(True)

            fig.set_tight_layout(True)
            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + "CB_decays.png", dpi=300
                )
            plt.show()

            # Plot the CB infidelities
            nrows = 2 if self._include_ref_cycle else 1
            figsize = (8, 5 * nrows)
            fig, axes = plt.subplots(
                nrows, 1, figsize=figsize, layout="constrained"
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
                ax.tick_params(axis="both", which="major", labelsize=12)
                ax.legend(prop={"size": 12})
                ax.grid(True)

            fig.set_tight_layout(True)
            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + "CB_infidelities.png",
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + "CB_infidelities.pdf"
                )
                fig.savefig(
                    self._data_manager._save_path + "CB_infidelities.svg"
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
        qpu=qpu,
        config=config,
        cycle=cycle,
        circuit_depths=circuit_depths,
        tq_config=tq_config,
        n_circuits=n_circuits,
        n_decays=n_decays,
        targeted_errors=targeted_errors,
        twirl=twirl,
        propogate_correction=propogate_correction,
        compiled_pauli=compiled_pauli,
        include_ref_cycle=include_ref_cycle,
        include_rcal=include_rcal,
        **kwargs,
    )


def CBMCM(
    qpu:            QPU,
    config:         Config,
    # cycle:          Circuit,
    mcm_qubits:     Sequence[int],
    idle_qubits:    Sequence[int],
    circuit_depths: Iterable[int],
    n_circuits:     int = 30,
    n_decays:       int | None = None,
    pspec:          QubitProcessorSpec | None = None,
    **kwargs
) -> Callable:
    """Circuit Benchmarking of Mid-Circuit Measurements.

    This is a pyGSTi protocol.

    Relevant paper:
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.020602

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        # cycle (Circuit): MCM cycle to be benchmarked.
        mcm_qubits (Sequence[int]): qubits measured in the MCM cycle.
        idle_qubits (Sequence[int]): qubits that are idle in the MCM cycle
            (i.e., have no gates applied).
        circuit_depths (Iterable[int]):  a list of integers >= 0 specifying
            the CB circuit depths.
        n_circuits (int, optional): The number of different CB circuits sampled
            at each depth. Defaults to 30.
        n_decays (int | None, optional): an integer specifying the total number
            of randomly  chosen Pauli decay strings used to measure the process
            infidelity. Defaults to None.
        pspec (QubitProcessorSpec | None, optional): pyGSTi qubit processor
            spec. Defaults to None. If None, a processor spec will be
            automatically generated based on the native_gates and the qubit
            labels.

    Returns:
        Callable: CBMCM class instance.
    """

    class CBMCM(qpu):
        """pyGSTi CB of MCM protocol."""

        def __init__(
            self,
            config:         Config,
            # cycle:          Circuit,
            mcm_qubits:     Sequence[int],
            idle_qubits:    Sequence[int],
            circuit_depths: Iterable[int],
            n_circuits:     int = 30,
            n_decays:       int | None = None,
            pspec:          QubitProcessorSpec | None = None,
            **kwargs
        ) -> None:
            logger.info(f" pyGSTi version: {pygsti.__version__}\n")

            self._qubits = sorted(set(mcm_qubits) | set(idle_qubits))
            self._mcm_qubits = mcm_qubits
            self._idle_qubits = idle_qubits

            cycle_gates = []
            for q in mcm_qubits:
                cycle_gates.append([Label('Iz', f'Q{q}')])
            for q in idle_qubits:
                cycle_gates.append([Label('Gc0', f'Q{q}')])
            self._cycle = pygsti.circuits.Circuit(cycle_gates) # TODO [cycle_gates]
            self._qubit_labels = self._cycle.line_labels

            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._n_decays = (
                n_decays if n_decays is not None else min(
                    20, 4 ** len(self._qubits)
                )
            )

            gate_set = [f'Gc{i}' for i in range(24)]
            for gate in config.native_gates['set']:
                if gate in TWO_QUBIT_GATES:
                    gate_set.append(gate)
            self._pspec = pspec if pspec is not None else pygsti_pspec(
                config, self._qubits, gate_set
            )

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

            transpiler = kwargs.get('transpiler', PyGSTiTranspiler())
            kwargs.pop('transpiler', None)
            qpu.__init__(self, config=config, transpiler=transpiler, **kwargs)

            self._pauli_groups = None
            self._edesign = None
            self._data = None
            self._dataset = None
            self._results = None

            self._error_rates = {}
            self._fit_params = {}
            self._success_probabilities = {}
            self._uncertainties = {}

        @property
        def pspec(self):
            """pyGSTi processor spec."""
            return self._pspec

        @property
        def edesign(self):
            """pyGSTi edesign."""
            return self._edesign

        @property
        def data(self):
            """pyGSTi data object."""
            return self._data

        @property
        def dataset(self):
            """pyGSTi dataset."""
            return self._dataset

        # @property
        # def fit_params(self):
        #     """CRB fit parameters."""
        #     return self._fit_params

        # @property
        # def process_infidelity(self):
        #     """CRB process infidelity."""
        #     process_infidelity = {}
        #     for ql, error_rate in self._error_rates.items():
        #         process_infidelity[ql] = {
        #             'val': error_rate,
        #             'err': self._uncertainties[ql]
        #         }
        #     return process_infidelity

        # @property
        # def success_probabilities(self):
        #     """Success probabilities."""
        #     return self._success_probabilities

        @property
        def results(self):
            """pyGSTi results object."""
            return self._results

        def generate_circuits(self):
            """Generate all pyGSTi circuits."""
            logger.info(' Generating circuits from pyGSTi...')

            if len(self._qubits) <= 2:
                paulis = generate_n_qubit_paulis(
                    qubits=self._qubits,
                    measured_qubits=self._mcm_qubits
                )
            else:
                paulis = generate_random_n_qubit_paulis(
                    qubits=self._qubits,
                    measured_qubits=self._mcm_qubits,
                    n_random_paulis=self._n_decays
                )

            # Remove the all-identity Pauli if it is included
            while tuple(['I'] * len(self._qubits)) in paulis:
                paulis.remove(tuple(['I'] * len(self._qubits)))

            self._pauli_groups = generate_n_qubit_pauli_measurement_map(paulis)
            cs_by_pauli, signs_by_pauli, tbs_by_pauli = _generate_rc_circuits(
                cycle=self._cycle,
                circuit_depths=self._circuit_depths,
                pauli_measurements=self._pauli_groups.keys(),
                compilations=self._compilations,
                n_randomizations=self._n_circuits,
            )

            edesigns = {}
            for pauli, clists in cs_by_pauli.items():
                print(pauli)
                edesigns[pauli] = pygsti.protocols.ByDepthDesign(
                    self._circuit_depths, clists
                )
            self._edesign = pygsti.protocols.CombinedExperimentDesign(edesigns)

            self._circuits = CircuitSet(self._edesign.all_circuits_needing_data)
            self._circuits['pygsti_circuit'] = [
                circ.str for circ in self._edesign.all_circuits_needing_data
            ]

            self._data_manager._exp_id += (
                f'_CBMCM_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if Settings.save_data:
                self._data_manager.create_data_path()
                pygsti.io.write_empty_protocol_data(
                    self._data_manager._save_path,
                    self._edesign,
                    sparse=True,
                    clobber_ok=True
                )

    return CBMCM(
        config=config,
        # cycle=cycle,
        mcm_qubits=mcm_qubits,
        idle_qubits=idle_qubits,
        circuit_depths=circuit_depths,
        n_circuits=n_circuits,
        n_decays=n_decays,
        pspec=pspec,
        **kwargs
    )


def SC(
    qpu:                  QPU,
    config:               Config,
    cycle:                dict,
    circuit_depths:       Iterable[int],
    tq_config:            str | Any = None,
    n_circuits:           int = 30,
    pauli_decays:         Iterable[str] | None = None,
    twirl:                str = "P",
    propogate_correction: bool = False,
    compiled_pauli:       bool = True,
    include_rcal:         bool = False,
    **kwargs,
) -> Callable:
    """Stochastic Calibration.

    SC is nearly identical to CB. The only difference is that in SC measurement
    bases (specified as eigenbases of Pauli operators) are explicitly chosen
    rather than randomly sampled. The measurement bases to be characterized
    should be selected so that they anticommute with some error(s) which are of
    concern so that the error(s) contribute(s) to the element of the process
    matrix corresponding to the Pauli decay.

    This is a True-Q protocol and requires a valid True-Q license.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        cycle (dict, tq.Cycle): cycle (or subcircuit) to benchmark.
        circuit_depths (Iterable[int]): iterable of positive integers
            specifying how many interleaved cycles of the target cycle and
            random Pauli operators to generate, for example, [4, 16, 64].
        tq_config (str | Any, optional): True-Q config yaml file or config
            object. Defaults to None.
        n_circuits (int, optional): the number of circuits for each circuit
            depth. Defaults to 30.
        pauli_decays (tq.math.Weyls, Iterable[str], None, optional): A True-Q
            Weyls instance, where the rows specify which elements of the
            diagonalized error channel should be estimated. These should be
            chosen to anticommute with the Hamiltonian terms of a known noise
            source to be optimized. As a convenience, a list of strings can be
            given, e.g. ["XII", "ZZY"], which will be used to instantiate a
            Weyls object.
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
        Callable: SC class instance.
    """

    class SC(qpu):
        """True-Q SC protocol."""

        def __init__(
            self,
            qpu:                  QPU,
            config:               Config,
            cycle:                dict,
            circuit_depths:       Iterable[int],
            tq_config:            str | Any = None,
            n_circuits:           int = 30,
            pauli_decays:         Iterable[str] | None = None,
            twirl:                str = "P",
            propogate_correction: bool = False,
            compiled_pauli:       bool = True,
            include_rcal:         bool = False,
            **kwargs,
        ) -> None:
            from qcal.interface.trueq.compiler import TrueqCompiler
            from qcal.interface.trueq.transpiler import TrueqTranspiler

            try:
                import trueq as tq
                logger.info(f" True-Q version: {tq.__version__}")
            except ImportError:
                logger.warning(" Unable to import trueq!")

            self._cycle = cycle
            self._circuit_depths = circuit_depths
            self._n_circuits = n_circuits
            self._pauli_decays = pauli_decays
            self._twirl = twirl
            self._propagate_correction = propogate_correction
            self._compiled_pauli = compiled_pauli
            self._include_rcal = include_rcal

            compiler = kwargs.get(
                "compiler", TrueqCompiler(
                    config if tq_config is None else tq_config
                )
            )
            kwargs.pop("compiler", None)

            transpiler = kwargs.get("transpiler", TrueqTranspiler())
            kwargs.pop("transpiler", None)

            qpu.__init__(
                self, config=config, compiler=compiler, transpiler=transpiler,
                **kwargs
            )

        def generate_circuits(self):
            """Generate all True-Q SC circuits."""
            logger.info(" Generating circuits from True-Q...")
            import trueq as tq

            self._circuits = tq.make_sc(
                cycles=self._cycle,
                n_random_cycles=self._circuit_depths,
                n_circuits=self._n_circuits,
                pauli_decays=self._pauli_decays,
                twirl=self._twirl,
                propagate_correction=self._propagate_correction,
                compiled_pauli=self._compiled_pauli,
            )

            if self._include_rcal:
                self._circuits += tq.make_rcal(self._circuits.labels)

            self._circuits.shuffle()

        def analyze(self):
            """Analyze the SC results."""
            logger.info(" Analyzing the results...")

            try:
                print(self._circuits.fit(analyze_dim=2))
            except Exception:
                logger.warning(" Unable to fit the estimate collection!")

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f"_SC{''.join('Q' + str(q) for q in self._circuits.labels)}"
            )
            if Settings.save_data:
                qpu.save(self)

        def plot(self) -> None:
            """Plot the SC fit results."""
            # Plot the raw curves
            ncols = 1
            figsize = (6 * ncols, 5)
            fig, axes = plt.subplots(
                1, ncols, figsize=figsize, layout="constrained"
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
                ax.tick_params(axis="both", which="major", labelsize=12)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[:5], labels[:5], fontsize=12)
                # ax.legend(prop=dict(size=12))
                ax.grid(True)

            fig.set_tight_layout(True)
            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + "SC_decays.png", dpi=300
                )
            plt.show()

            # Plot the SC infidelities
            nrows = 1
            figsize = (8, 5 * nrows)
            fig, axes = plt.subplots(
                nrows, 1, figsize=figsize, layout="constrained"
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
                ax.tick_params(axis="both", which="major", labelsize=12)
                ax.legend(prop={"size": 12})
                ax.grid(True)

            fig.set_tight_layout(True)
            if Settings.save_data:
                fig.savefig(
                    self._data_manager._save_path + "SC_infidelities.png",
                    dpi=600
                )
                fig.savefig(
                    self._data_manager._save_path + "SC_infidelities.pdf"
                )
                fig.savefig(
                    self._data_manager._save_path + "SC_infidelities.svg"
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

    return SC(
        qpu=qpu,
        config=config,
        cycle=cycle,
        circuit_depths=circuit_depths,
        tq_config=tq_config,
        n_circuits=n_circuits,
        pauli_decays=pauli_decays,
        twirl=twirl,
        propogate_correction=propogate_correction,
        compiled_pauli=compiled_pauli,
        include_rcal=include_rcal,
        **kwargs,
    )


def _generate_rc_circuits(
    cycle:              pygsti.circuits.Circuit,
    circuit_depths:     Iterable[int],
    pauli_measurements: Iterable[tuple[str]],
    compilations:       dict[str, CCR],
    n_randomizations:   int = 30,
) -> tuple:
    """Generate randomly compiled circuits.

    Args:
        cycle (pygsti.circuits.Circuit): the base cycle to repeat.
        circuit_depths (Iterable[int]): iterable of cycle depths to test.
        pauli_measurements (Iterable[tuple[str]]): iterable of Pauli strings to
            sample.
        compilations (dict[str, CCR]): compilation rules.
        n_randomizations (int): number of randomizations per depth. Defaults
            to 30.

    Returns:
        tuple: dictionaries containing circuits, signs, and twirl bit-string
            by Pauli.
    """
    cs_by_pauli = {}
    signs_by_pauli = {}
    tbs_by_pauli = {}
    for pauli in pauli_measurements:
        clist = []
        signlist = []
        tbslist = []
        for d in circuit_depths:
            cs = []
            signs = []
            tbss = []
            for _ in range(n_randomizations):
                if not all(i == 'I' for i in pauli):
                    sign = np.random.choice([-1, 1])
                else:
                    sign = 1

                _, _, _, _, prep_layer = random_circuit._sample_stabilizer(
                    pauli, sign, compilations['absolute'], cycle.line_labels
                )
                _, _, meas_layer = random_circuit._stabilizer_to_all_zs(
                    pauli, cycle.line_labels, compilations['absolute']
                )

                circuit = prep_layer.serialize().copy(editable=True)
                circuit.append_circuit_inplace(cycle.repeat(d))
                circuit.append_circuit_inplace(meas_layer.serialize())
                circuit.delete_idle_layers_inplace()
                circuit.done_editing()
                circuit, tbs, _ = pauli_randomize_clifford_circuit(circuit)

                cs.append(circuit)
                tbss.append(tbs)
                signs.append(sign)

            clist.append(cs)
            signlist.append(signs)
            tbslist.append(tbss)

        cs_by_pauli[pauli] = clist
        signs_by_pauli[pauli] = signlist
        tbs_by_pauli[pauli] = tbslist

    return cs_by_pauli, signs_by_pauli, tbs_by_pauli
