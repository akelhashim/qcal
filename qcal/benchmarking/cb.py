"""Submodule for cycle benchmarking.

See:
https://www.nature.com/articles/s41467-019-13068-7
https://trueq.quantumbenchmark.com/guides/error_diagnostics/cb.html
https://trueq.quantumbenchmark.com/api/protocols.html#trueq.make_cb

NOTE: we do not use TYPE_CHECKING for trueq types because this might fail if
trueq is not installed when building docs.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import clear_output, display
from plotly.colors import qualitative
from uncertainties import ufloat, unumpy
from uncertainties.umath import exp as uexp

from qcal.analysis.leakage import analyze_leakage
from qcal.benchmarking.utils import (
    PauliString,
    _pad_pauli,
    generate_n_qubit_pauli_measurement_groups,
    generate_n_qubit_paulis,
    generate_random_n_qubit_paulis,
)
from qcal.circuit import Barrier, Circuit, CircuitSet, Cycle
from qcal.compilation.decompositions import decompose_cycle, pauli_to_cycle
from qcal.compilation.merge import merge_cycles
from qcal.compilation.pauli_conjugation import conjugate_pauli
from qcal.compilation.utils import composes_to_identity
from qcal.config import Config
from qcal.fitting.fit import FitExponential
from qcal.gates.single_qubit import Meas, basis_rotation, prep_rotation
from qcal.math.utils import round_to_order_error
from qcal.plotting.graphs import draw_qpu_heatmap
from qcal.qpu.qpu import QPU
from qcal.results import Results
from qcal.settings import Settings

logger = logging.getLogger(__name__)


__all__ = ['CB', 'CB1', 'SC']


def compute_cycle_infidelity(
    f_D: ufloat, f_ref: ufloat, n_qubits: int
) -> ufloat:
    """Estimate the bare infidelity of an interleaved cycle.

    Divides out the average Pauli fidelity of the reference (empty)
    cycle from that of the interleaved (dressed) cycle, isolating the
    infidelity contributed by cycle_or_circuit itself from that of the
    random Pauli twirling gates.

    Args:
        f_D (ufloat): average Pauli fidelity of the interleaved
            (dressed) cycle.
        f_ref (ufloat): average Pauli fidelity of the reference (empty)
            cycle.
        n_qubits (int): number of qubits the cycle acts on.

    Returns:
        ufloat: estimated bare process infidelity of cycle_or_circuit.
    """
    d = 2 ** n_qubits
    return (d**2 - 1) / d**2 * (1 - f_D / f_ref)


def compute_cycle_infidelity1(
    circs_D: trueq.CircuitCollection, circs_ref: trueq.CircuitCollection  # noqa: F821 # type: ignore
) -> tuple:
    """Compute the infidelity of the interleaved Cycle.

    Args:
        circs_D (trueq.CircuitCollection): dressed circuits
        circs_ref (trueq.CircuitCollection): reference circuits

    Returns:
        tuple: (cycle infidelity, error)
    """
    n_qubits = len(circs_D.labels)
    d = 2**n_qubits
    fit_D = circs_D.fit(analyze_dim=2)[0].e_F
    fit_ref = circs_ref.fit(analyze_dim=2)[0].e_F
    F_D = 1 - ufloat(fit_D.val, fit_D.std)
    F_ref = 1 - ufloat(fit_ref.val, fit_ref.std)

    f_D = (d**2 * F_D - 1) / (d**2 - 1)
    f_ref = (d**2 * F_ref - 1) / (d**2 - 1)

    e_C = (d**2 - 1) / d**2 * (1 - f_D / f_ref)

    e_C_val, e_C_err = round_to_order_error(e_C.n, e_C.s)

    return (e_C_val, e_C_err)


def CB(
    qpu:                QPU,
    config:             Config,
    cycle_or_circuit:   Cycle | Circuit,
    circuit_depths:     Iterable[int],
    n_decays:           int = 20,
    n_randomizations:   int = 30,
    decompose_to_zxzxz: bool = False,
    include_ref_cycle:  bool = False,
    targeted_decays:    Sequence[str] | None = None,
    analyze_subsystems: bool = False,
    **kwargs,
) -> Callable:
    """Cycle Benchmarking (CB) without mirror inversion.

    Estimates per-Pauli decay rates for a target cycle or circuit by twirling
    with random Pauli layers and measuring expectation-value decay vs depth.

    Protocol overview (depth d, Pauli decay string P):
      1. Sample n_decays random n-qubit Pauli strings; group by QWC measurement
         basis.
      2. For each Pauli P, depth d, and randomization index r:
           a. Build a preparation cycle that rotates |0...0> to the +1
           eigenstate of P:
                I, Z  →  identity (|0> is already the +1 eigenstate)
                X     →  Ry(π/2)
                Y     →  Rx(-π/2)
           b. Insert d instances of cycle_or_circuit, with a random Pauli
              twirl cycle before the first, between each adjacent pair,
              and after the last (d+1 twirls total).
           c. Apply the inverse preparation to rotate back to the Z basis.
           d. Measure all qubits in Z.
           e. Compute the expected eigenstate sign ±1 by propagating P through
              the Pauli twirl layers: each twirl Q_i flips the sign if P and Q_i
              anticommute (odd number of qubit-positions with distinct
              non-identity Paulis).
      3. Track per-circuit metadata in the CircuitSet: pauli, depth,
         randomization, sign.
      4. For analysis, group circuits by Pauli, depth, and sign, and then sum
         results, and fit <P>(d) = A * f_P^d to extract the per-Pauli fidelity
         f_P.

    Prerequisite: cycle_or_circuit composed depth times must equal the identity
    (up to global phase) for every depth in circuit_depths.

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        cycle_or_circuit (Cycle | Circuit): the cycle or sub-circuit to
            benchmark.
        circuit_depths (Iterable[int]): number of interleaved cycle_or_circuit
            instances per circuit, e.g. [1, 2, 4, 8, 16].
        n_decays (int): number of randomly sampled Pauli decay strings. Defaults
            to 20. Warning: values below min(20, 4^n - 1) may bias the process
            fidelity estimate.
        n_randomizations (int): number of random Pauli twirl instances per
            (Pauli, depth) pair. Defaults to 30.
        decompose_to_zxzxz (bool): whether to decompose all single-qubit gates
            to ZXZXZ decomposition. Defaults to False. Setting to True can be
            useful when implementing CB using hardware-efficient randomization.
        include_ref_cycle (bool): whether to additionally benchmark the
            empty (all-identity) cycle on the same qubits. Defaults to
            False. The reference circuits are identical to the
            interleaved ones except that cycle_or_circuit is never
            inserted, so they measure the decay due to the random Pauli
            twirls alone. When True, self._e_F reports the estimated
            bare infidelity of cycle_or_circuit (the interleaved decay
            with the reference decay divided out); when False, self._e_F
            reports the dressed infidelity, as before.
        targeted_decays (Sequence[str] | None): an explicit set of Pauli
            decay strings to prepare and measure, e.g. ['XII', 'ZZY'],
            ordered by cycle_or_circuit.qubits. Defaults to None. If given,
            this is used instead of randomly sampling n_decays Pauli decay
            strings, and each string must have length equal to the number
            of qubits in cycle_or_circuit.
        analyze_subsystems (bool): whether to additionally compute a
            marginal process infidelity for each individual gate body
            (subsystem) in cycle_or_circuit, e.g. each disjoint CZ pair
            in a multi-qubit CZ cycle. Defaults to False. Requires
            cycle_or_circuit to be a Cycle (not a multi-cycle Circuit)
            with at least two gates. When True, every sampled decay
            Pauli is marginalized and refit per subsystem: e.g. a decay
            of 'IXYZ' on two disjoint pairs contributes an 'IX'
            measurement of the first pair and a 'YZ' measurement of the
            second, regardless of what the other pair's substring is,
            since the cycle's gates act on disjoint qubits (no new
            circuits are generated for this). Data sharing the same
            local substring on a subsystem, from otherwise-different
            decay strings, is pooled together and refit, kept separate
            from the whole-cycle Pauli fidelities (mixing them would
            bias the whole-cycle average toward the higher-fidelity
            low-weight patterns marginalization manufactures). Results
            are available via self.subsystem_infidelities (bare if
            include_ref_cycle, else dressed, mirroring
            process_infidelity) and self.subsystem_summary (the full
            per-subsystem dressed/reference/bare breakdown). Caveat: a
            marginal fit cannot see errors correlated across subsystems
            (e.g. crosstalk between two CZ pairs); it only reports each
            subsystem's own local Pauli error rate, the same limitation
            as marginal RB/GST.

    Returns:
        Callable: CB class instance.
    """

    class CB(qpu):
        """qcal-native CB protocol."""

        def __init__(
            self,
            config:             Config,
            cycle_or_circuit:   Cycle | Circuit,
            circuit_depths:     Iterable[int],
            n_decays:           int = 20,
            n_randomizations:   int = 30,
            decompose_to_zxzxz: bool = False,
            include_ref_cycle:  bool = False,
            targeted_decays:    Sequence[str] | None = None,
            analyze_subsystems: bool = False,
            **kwargs,
        ) -> None:
            self._cycle_or_circuit = cycle_or_circuit
            self._circuit_depths = sorted(circuit_depths)
            self._n_decays = n_decays
            self._n_randomizations = n_randomizations
            self._decompose_to_zxzxz = decompose_to_zxzxz
            self._qubits = cycle_or_circuit.qubits
            self._include_ref_cycle = include_ref_cycle

            # Qubits undergoing a mid-circuit measurement (Meas/MCM)
            # within cycle_or_circuit. Their decay-Pauli sampling is
            # restricted to {I, Z} below (the only physically
            # meaningful eigenstates to prepare/interrogate on a
            # computational-basis measurement); twirl-layer sampling
            # stays unrestricted over {I, X, Y, Z}.
            cycles = (
                [cycle_or_circuit] if isinstance(cycle_or_circuit, Cycle)
                else [c for c in cycle_or_circuit.cycles if not c.is_barrier]
            )
            self._measured_qubits = frozenset(
                q
                for cycle in cycles
                for gate in cycle.gates
                if gate.is_measurement
                for q in gate.qubits
            )

            if targeted_decays is not None:
                for pauli in targeted_decays:
                    if len(pauli) != len(self._qubits):
                        raise ValueError(
                            f"Pauli decay string '{pauli}' has length "
                            f"{len(pauli)}, but cycle_or_circuit acts on "
                            f"{len(self._qubits)} qubits."
                        )
                    if not set(pauli.upper()) <= {'I', 'X', 'Y', 'Z'}:
                        raise ValueError(
                            f"Pauli decay string '{pauli}' contains "
                            "invalid characters; each character must be "
                            "one of 'I', 'X', 'Y', 'Z'."
                        )
                targeted_decays = [p.upper() for p in targeted_decays]
            self._targeted_decays = targeted_decays

            # If cycle_or_circuit^base_depth = I, then for any multiple
            # k*base_depth, cycle_or_circuit^(k*base_depth) =
            # (cycle_or_circuit^base_depth)^k = I automatically, so only
            # depths that are not multiples of the smallest depth need an
            # explicit check.
            base_depth = self._circuit_depths[0]
            depths_to_check = [base_depth] + [
                depth for depth in self._circuit_depths[1:]
                if depth % base_depth != 0
            ]
            for depth in depths_to_check:
                if not composes_to_identity(cycle_or_circuit, depth):
                    raise ValueError(
                        f"cycle_or_circuit^{depth} is not the identity "
                        "(up to global phase). CB requires that "
                        "cycle_or_circuit composed `depth` times equals "
                        "the identity for every depth in circuit_depths."
                    )

            self._experiments = (
                ['interleaved', 'reference']
                if self._include_ref_cycle else ['interleaved']
            )
            self._fit = {experiment: {} for experiment in self._experiments}
            self._pauli_fidelities = {
                experiment: {} for experiment in self._experiments
            }
            self._pauli_decays = {
                experiment: {} for experiment in self._experiments
            }
            self._process_infidelities = {}

            self._analyze_subsystems = analyze_subsystems
            # self._subsystems always includes the full register (the
            # whole cycle) as its first entry, so the whole-cycle and
            # per-gate-body analyses/plots can share the same
            # machinery (_compute_subsystem_infidelities,
            # _build_subsystem_summary, plot) via one uniform loop.
            # Gate-body subsystems are appended only when
            # analyze_subsystems is True; the extra marginalize-and-
            # refit work in _accumulate_subsystem_evs/
            # _fit_subsystem_decays is skipped for the full register,
            # since it's already covered directly by the main per-Pauli
            # fit above.
            self._subsystems: list[tuple] = [self._qubits]
            whole_cycle_name = (
                cycle_or_circuit.gates[0].name
                if isinstance(cycle_or_circuit, Cycle)
                and cycle_or_circuit.n_gates == 1
                else 'Cycle'
            )
            self._subsystem_gate_names: dict[tuple, str] = {
                self._qubits: whole_cycle_name
            }
            self._subsystem_positions: dict[tuple, list[int]] = {
                self._qubits: list(range(len(self._qubits)))
            }
            self._subsystem_infidelities = {
                experiment: {} for experiment in self._experiments
            }
            self._subsystem_bare_infidelities = {}
            self._subsystem_summary = None
            # Marginalized/pooled per-subsystem Pauli decays/fidelities,
            # keyed by (experiment, subsystem, local Pauli string, e.g.
            # 'IX'). Kept separate from self._pauli_fidelities/
            # _pauli_decays -- which must stay exactly the originally
            # sampled full-register Pauli set -- because the whole-cycle
            # process infidelity is a *uniform* average over that
            # sample; mixing in subsystem-local reconstructions would
            # over-represent low-weight (single-subsystem-only) Paulis,
            # which are systematically higher-fidelity than full-weight
            # ones, biasing the whole-cycle infidelity low.
            self._subsystem_pauli_decays: dict[
                str, dict[tuple, dict[str, np.ndarray]]
            ] = {experiment: {} for experiment in self._experiments}
            self._subsystem_pauli_fidelities: dict[
                str, dict[tuple, dict[str, ufloat]]
            ] = {experiment: {} for experiment in self._experiments}
            self._subsystem_fit: dict[
                str, dict[tuple, dict[str, FitExponential]]
            ] = {experiment: {} for experiment in self._experiments}

            if analyze_subsystems:
                if not isinstance(cycle_or_circuit, Cycle):
                    raise ValueError(
                        "analyze_subsystems requires cycle_or_circuit to "
                        "be a Cycle, since subsystems are derived from "
                        "its constituent gates."
                    )
                if cycle_or_circuit.n_gates < 2:
                    raise ValueError(
                        "analyze_subsystems requires cycle_or_circuit to "
                        "contain at least two gates; there is nothing to "
                        "marginalize with a single gate body."
                    )
                gate_subsystems = [
                    gate.qubits for gate in cycle_or_circuit.gates
                ]
                self._subsystems += gate_subsystems
                self._subsystem_gate_names.update({
                    gate.qubits: gate.name
                    for gate in cycle_or_circuit.gates
                })
                self._subsystem_positions.update({
                    s: [self._qubits.index(q) for q in s]
                    for s in gate_subsystems
                })

            qpu.__init__(self, config=config, **kwargs)

        @property
        def pauli_decays(self) -> dict[str, dict[str, np.ndarray]]:
            """Estimated per-Pauli decays.

            Returns:
                dict[str, dict[str, np.ndarray]]: experiment ('interleaved'
                    or 'reference') to (Pauli string to decay array) map.
                    Each decay array is a 1D np.ndarray of ufloats (mean EV
                    ± SEM over randomizations), one per entry of
                    self._circuit_depths, in the same order; a missing
                    (depth, pauli) pair is stored as ufloat(nan, nan).
            """
            return self._pauli_decays

        @property
        def pauli_fidelities(self) -> dict[str, dict[str, ufloat]]:
            """Estimated per-Pauli fidelities.

            Returns:
                dict[str, dict[str, ufloat]]: experiment ('interleaved'
                    or 'reference') to (Pauli string to fidelity) map.
            """
            return self._pauli_fidelities

        @property
        def process_fidelity(self) -> ufloat | float:
            """Estimated process fidelity of the cycle_or_circuit.

            This is the bare (interleaved-cycle-only) fidelity if
            include_ref_cycle was True, or the dressed fidelity
            otherwise.

            Returns:
                ufloat | float: process fidelity e_F, or NaN if no valid
                fidelity estimates were obtained.
            """
            return 1 - self.process_infidelity

        @property
        def process_infidelity(self) -> ufloat | float:
            """Estimated process infidelity of the cycle_or_circuit.

            This is the bare (interleaved-cycle-only) infidelity if
            include_ref_cycle was True, or the dressed infidelity
            otherwise.

            Returns:
                ufloat | float: process infidelity e_F, or NaN if no valid
                fidelity estimates were obtained.
            """
            return self._e_F

        @property
        def dressed_infidelity(self) -> ufloat | float:
            """Estimated dressed process infidelity of cycle_or_circuit.

            This is the interleaved experiment's process infidelity
            (it still includes the decay contributed by the random
            Pauli twirls), regardless of include_ref_cycle -- unlike
            process_infidelity, which is the bare infidelity when
            include_ref_cycle is True.

            Returns:
                ufloat | float: dressed process infidelity, or NaN if
                    no valid fidelity estimates were obtained.
            """
            return self._process_infidelities.get('dressed', np.nan)

        @property
        def reference_infidelity(self) -> ufloat | float:
            """Estimated process infidelity of the reference experiment.

            Only populated when include_ref_cycle was True.

            Returns:
                ufloat | float: reference process infidelity, or NaN if
                    include_ref_cycle was False or no valid fidelity
                    estimates were obtained.
            """
            return self._process_infidelities.get('reference', np.nan)

        @property
        def loss(self) -> dict[tuple, ufloat]:
            """Loss for the cycle_or_circuit.

            This property can be used for parameter optimization.

            Returns:
                dict[tuple, ufloat]: process infidelity of
                    cycle_or_circuit, keyed by its qubits.
            """
            return {tuple(self._qubits): self.process_infidelity}

        @property
        def subsystem_infidelities(self) -> dict[tuple, ufloat]:
            """Estimated marginal process infidelity of each subsystem.

            Mirrors process_infidelity: this is the bare (reference-
            corrected) infidelity per subsystem if include_ref_cycle
            was True, or the dressed infidelity otherwise. Only
            populated when analyze_subsystems was True. A subsystem is
            the qubit tuple acted on by one gate in cycle_or_circuit
            (e.g. one CZ pair). The full dressed/reference/bare
            breakdown per subsystem is available via subsystem_summary.

            Returns:
                dict[tuple, ufloat]: subsystem qubits to process
                    infidelity map.
            """
            if self._include_ref_cycle:
                return self._subsystem_bare_infidelities
            return self._subsystem_infidelities.get('interleaved', {})

        @property
        def subsystem_summary(self) -> pd.DataFrame | None:
            """Summary table of the whole-cycle and per-subsystem
            process infidelities.

            Always populated after analyze() runs. If
            analyze_subsystems is False (or no subsystems were
            found), the table only contains the whole-cycle row.

            Returns:
                pd.DataFrame | None: table with columns 'Subset',
                    'Operation', 'Dressed Infidelity', and (if
                    include_ref_cycle) 'Reference Infidelity' and 'Bare
                    Infidelity'; or None if analyze() has not yet been
                    called.
            """
            return self._subsystem_summary

        @property
        def fit(self) -> dict[str, dict[str, FitExponential]]:
            """Exponential fits underlying each per-Pauli decay.

            Returns:
                dict[str, dict[str, FitExponential]]: experiment
                    ('interleaved' or 'reference') to (Pauli string to
                    FitExponential) map.
            """
            return self._fit

        @property
        def subsystems(self) -> list[tuple]:
            """Subsystems analyzed for marginal process infidelity.

            The first entry is always the full register (the whole
            cycle). If analyze_subsystems was True, this is followed
            by one entry per constituent gate body in
            cycle_or_circuit (e.g. each disjoint CZ pair).

            Returns:
                list[tuple]: qubit tuples, one per subsystem.
            """
            return self._subsystems

        @property
        def subsystem_pauli_decays(
            self
        ) -> dict[str, dict[tuple, dict[str, np.ndarray]]]:
            """Marginalized, pooled per-subsystem Pauli decays.

            Only populated when analyze_subsystems was True. Keyed by
            the local (not zero-padded) Pauli string on the
            subsystem's own qubits (e.g. 'IX' on a CZ pair), pooling
            data from every sampled decay Pauli that shares that local
            substring. Kept separate from self.pauli_decays, which
            covers only the originally sampled full-register Paulis.

            Returns:
                dict[str, dict[tuple, dict[str, np.ndarray]]]:
                    experiment to (subsystem to (local Pauli string to
                    decay array)) map.
            """
            return self._subsystem_pauli_decays

        @property
        def subsystem_pauli_fidelities(
            self
        ) -> dict[str, dict[tuple, dict[str, ufloat]]]:
            """Marginalized, pooled per-subsystem Pauli fidelities.

            Only populated when analyze_subsystems was True. See
            subsystem_pauli_decays for how entries are keyed and
            pooled; these fidelities are merged into
            self.pauli_fidelities under their zero-padded full-length
            key as a side effect of the marginal fit.

            Returns:
                dict[str, dict[tuple, dict[str, ufloat]]]: experiment
                    to (subsystem to (local Pauli string to fidelity))
                    map.
            """
            return self._subsystem_pauli_fidelities

        @property
        def subsystem_fit(
            self
        ) -> dict[str, dict[tuple, dict[str, FitExponential]]]:
            """Exponential fits underlying each per-subsystem Pauli
            decay.

            Only populated when analyze_subsystems was True.

            Returns:
                dict[str, dict[tuple, dict[str, FitExponential]]]:
                    experiment to (subsystem to (local Pauli string to
                    FitExponential)) map.
            """
            return self._subsystem_fit

        def generate_circuits(self) -> None:
            """Generate all CB circuits and store them in self._circuits.

            Pauli decay strings are grouped into qubit-wise-commuting (QWC)
            sets so that every Pauli in a group can be estimated from a
            single shared circuit: preparing the +1 eigenstate of the
            group's combined measurement basis is simultaneously a +1
            eigenstate of every member Pauli, and marginalizing a single
            measurement in that basis yields the expectation value of each
            member. So circuits are generated per (group, depth,
            randomization), not per (Pauli, depth, randomization):

              prepare(basis) | twirl | barrier | [cycle_or_circuit | twirl
                             | barrier]^d | measure(basis)

            If include_ref_cycle is True, this is repeated for a
            'reference' experiment in which cycle_or_circuit is never
            inserted, so the circuit reduces to d+1 random Pauli twirl
            layers with no interleaved gate:

              prepare(basis) | twirl | barrier | [twirl
                             | barrier]^d | measure(basis)

            CircuitSet metadata columns:
              'measurement_basis' — joined basis string, e.g. 'XZI'
              'depth'             — number of cycle_or_circuit instances
              'randomization'     — integer index 0..n_randomizations-1
              'group_paulis'      — joined Pauli strings of every member
                                    of this group, e.g. ['XZI', 'XZZ']
              'group_signs'       — ±1 expected eigenstate sign for each
                                    entry in 'group_paulis', under ideal
                                    evolution
              'twirl_strings'     — list of d+1 joined twirl strings
                                    applied during the circuit, kept for
                                    reference
              'experiment'        — 'interleaved' or 'reference'
            """
            logger.info(" Generating circuits...")

            # Generate the Pauli decays grouped by simultaneous measurements
            if self._targeted_decays is not None:
                sampled_paulis = [
                    tuple(pauli) for pauli in self._targeted_decays
                ]
            elif self._n_decays > 4**len(self._qubits) - 1:
                sampled_paulis = generate_n_qubit_paulis(
                    self._qubits, measured_qubits=self._measured_qubits
                )
            else:
                sampled_paulis = generate_random_n_qubit_paulis(
                    self._qubits,
                    measured_qubits=self._measured_qubits,
                    n_random_paulis=self._n_decays,
                )

            # Drop the all-identity Pauli from the sampled set
            if ('I',) * len(self._qubits) in sampled_paulis:
                sampled_paulis.remove(('I',) * len(self._qubits))
            self._pauli_groups = generate_n_qubit_pauli_measurement_groups(
                set(sampled_paulis)
            )

            circuits: list[Circuit] = []
            basis_labels: list[str] = []
            depths: list[int] = []
            randomizations: list[int] = []
            group_paulis_list: list[list[str]] = []
            group_signs_list: list[list[int]] = []
            twirl_strs_list: list[list[str]] = []
            experiment_labels: list[str] = []

            for experiment in self._experiments:
                insert_cycle = experiment == 'interleaved'
                # For sign propagation, the reference experiment's
                # interleaved gate is the identity: an empty Cycle
                # conjugates every decay Pauli to itself.
                interleaved_op = (
                    self._cycle_or_circuit if insert_cycle else Cycle()
                )

                for group in self._pauli_groups:
                    # Positions where every Pauli in the group is 'I' are
                    # unconstrained (marginalized away later), and are
                    # prepared/measured as identities.
                    basis = group.measurement_basis
                    measurement_basis = ''.join(basis)

                    for depth in self._circuit_depths:
                        for r in range(self._n_randomizations):
                            # Sample d+1 random Pauli twirl strings,
                            # shared by every Pauli in the group for
                            # this circuit.
                            twirl_strings = generate_random_n_qubit_paulis(
                                self._qubits, n_random_paulis=depth + 1
                            )
                            signs = [
                                _propagate_sign(
                                    pauli,
                                    twirl_strings,
                                    interleaved_op,
                                    depth
                                )
                                for pauli in group.paulis
                            ]

                            circuit = Circuit()
                            if self._decompose_to_zxzxz:
                                # State prep: +1 eigenstate of the group's
                                # shared basis, which is simultaneously a +1
                                # eigenstate of every Pauli in the group.
                                prep_cycle = Cycle(
                                    prep_rotation(q, b) for q, b in zip(
                                        self._qubits, basis, strict=True
                                    )
                                )

                                # Initial twirl
                                initial_twirl = pauli_to_cycle(
                                    twirl_strings[0],
                                    self._qubits,
                                    False
                                )

                                # Combine prep + initial twirl layer into zxzxz
                                circuit.extend(
                                    decompose_cycle(
                                        merge_cycles(
                                            prep_cycle,
                                            initial_twirl.cycles[0]
                                        )
                                    )
                                )

                            else:
                                circuit.prepare(
                                    measurement_basis,
                                    qubits=list(self._qubits),
                                )
                                circuit.append(Barrier(self._qubits))

                                circuit.extend(
                                    pauli_to_cycle(
                                        twirl_strings[0],
                                        self._qubits,
                                        self._decompose_to_zxzxz
                                    )
                                )
                                circuit.append(Barrier(self._qubits))

                            for i in range(depth):
                                # Interleaved gate cycle
                                if insert_cycle:
                                    if isinstance(
                                        self._cycle_or_circuit, Cycle
                                    ):
                                        circuit.append(
                                            self._cycle_or_circuit
                                        )
                                    else:
                                        circuit.extend(
                                            self._cycle_or_circuit
                                        )

                                # Twirling layer
                                if (
                                    self._decompose_to_zxzxz
                                    and i == depth - 1
                                ):
                                    # Combine the final twirl with the
                                    # final basis-change rotation into
                                    # zxzxz before appending it.
                                    final_twirl = pauli_to_cycle(
                                        twirl_strings[i + 1],
                                        self._qubits,
                                        False
                                    ).cycles[0]
                                else:
                                    circuit.extend(
                                        pauli_to_cycle(
                                            twirl_strings[i + 1],
                                            self._qubits,
                                            self._decompose_to_zxzxz
                                        )
                                    )
                                    circuit.append(Barrier(self._qubits))

                            # Combine final twirl with final basis-change
                            # rotation into zxzxz before appending
                            if self._decompose_to_zxzxz:
                                basis_cycle = Cycle({
                                    basis_rotation(Meas(q, b))
                                    for q, b in zip(
                                        self._qubits, basis, strict=True
                                    )
                                })
                                circuit.extend(
                                    decompose_cycle(
                                        merge_cycles(
                                            final_twirl, basis_cycle
                                        )
                                    )
                                )
                                circuit.measure()
                            else:
                                circuit.measure(
                                    qubits=list(self._qubits),
                                    basis=list(basis),
                                )

                            circuits.append(circuit)
                            basis_labels.append(measurement_basis)
                            depths.append(depth)
                            randomizations.append(r)
                            group_paulis_list.append(
                                [''.join(p) for p in group.paulis]
                            )
                            group_signs_list.append(signs)
                            twirl_strs_list.append(
                                [''.join(ts) for ts in twirl_strings]
                            )
                            experiment_labels.append(experiment)

            self._circuits = CircuitSet(circuits)
            self._circuits['measurement_basis'] = basis_labels
            self._circuits['depth'] = depths
            self._circuits['randomization'] = randomizations
            self._circuits['group_paulis'] = group_paulis_list
            self._circuits['group_signs'] = group_signs_list
            self._circuits['twirl_strings'] = twirl_strs_list
            self._circuits['experiment'] = experiment_labels

        def analyze(self) -> None:
            """Fit per-Pauli decay curves and estimate the cycle infidelity.

            For each experiment ('interleaved', and 'reference' if
            include_ref_cycle), each Pauli Q in a QWC group, and each
            depth d, collects the parity expectation value from every
            circuit sharing Q's group (a single measurement in the
            group's basis yields Q's expectation value via
            marginalization), and averages over randomizations:

              EV(Q, d) = mean_r [ sign(r) * marginalized_ev(result, Q) ]

            Fits EV(Q, d) = A * f_Q^d per Pauli using FitExponential with
            offset fixed to zero (f_Q = exp(-b)), via the module-level
            _fit_decay_curve helper. Reports the process infidelity of
            each experiment (via _polarization_and_infidelity), where
            the polarization is the mean f_Q over sampled Paulis:

              e_F = (d^2 - 1) / d^2 * (1 - polarization)

            If include_ref_cycle is True, the estimated bare infidelity
            of cycle_or_circuit is additionally computed from the
            interleaved and reference polarizations (see
            compute_cycle_infidelity), and reported as self._e_F;
            otherwise self._e_F is the interleaved (dressed) process
            infidelity.

            Results are stored in self._pauli_fidelities and
            self._pauli_decays (each keyed by experiment, then Pauli
            string), and self._e_F. self._pauli_decays[experiment][pauli]
            is an array of ufloats (mean EV ± SEM over randomizations),
            one per entry of self._circuit_depths, in the same order;
            a missing (depth, pauli) pair is stored as ufloat(nan, nan).

            self._subsystems always includes the full register (the
            whole cycle) as its first entry, so self.subsystem_summary
            and plot() can treat the whole-cycle and per-gate-body
            results uniformly; that entry is skipped by
            _accumulate_subsystem_evs/_fit_subsystem_decays (it would
            just redundantly refit what's already computed directly
            above) but is still used by _compute_subsystem_infidelities/
            _compute_subsystem_bare_infidelities, which for that entry
            simply reproduce self._process_infidelities/self._e_F.

            If analyze_subsystems is True, every sampled decay Pauli is
            additionally marginalized down to each gate-body subsystem
            in cycle_or_circuit: for a decay like 'IXYZ' on two disjoint
            CZ pairs, the 'IX' substring is pooled with every other
            sampled decay sharing that same substring on that pair
            (regardless of what the other pair's substring is), the
            marginalized/signed EV is refit the same way (see
            _accumulate_subsystem_evs / _fit_subsystem_decays), and the
            resulting fidelity is stored in
            self._subsystem_pauli_fidelities/_subsystem_pauli_decays,
            keyed by the local (not zero-padded) Pauli string (e.g.
            'IX'). This is deliberately kept separate from
            self._pauli_fidelities/_pauli_decays, which must remain
            exactly the originally sampled full-register set: the
            whole-cycle infidelity is a uniform average over that set,
            and mixing in subsystem-local reconstructions -- which
            skew low-weight (single-subsystem-only, and so
            systematically higher-fidelity) -- would bias it low. The
            local sign is obtained by calling the existing
            _propagate_sign on the zero-padded local pattern: 'I' never
            anticommutes, so positions outside the subsystem trivially
            drop out of the parity computation, leaving exactly the
            local sign. The per-subsystem dressed (and, if
            include_ref_cycle, bare) infidelity is then computed the
            same way as the whole-cycle one (see
            _compute_subsystem_infidelities /
            _compute_subsystem_bare_infidelities), and reported via
            self.subsystem_infidelities (bare if include_ref_cycle,
            else dressed) / self.subsystem_summary (the full
            dressed/reference/bare breakdown). Because this only
            marginalizes already-collected data, it generates no
            additional circuits;
            it also cannot detect errors correlated across subsystems
            (e.g. crosstalk between two CZ pairs), the same limitation
            shared by marginal RB/GST.
            """
            logger.info(" Analyzing the results...")

            depths = np.array(self._circuit_depths, dtype=float)
            polarizations: dict[str, ufloat] = {}
            subsystem_polarizations: dict[str, dict[tuple, ufloat]] = {
                experiment: {} for experiment in self._experiments
            }

            # Analyze the fits for each experiment
            for experiment in self._experiments:
                # subsystem -> local Pauli pattern -> depth -> list of
                # signed, marginalized EVs pooled across every sampled
                # decay Pauli sharing that local pattern on the
                # subsystem (see analyze_subsystems).
                subsystem_local_evs: dict[
                    tuple, dict[tuple, dict[int, list[float]]]
                ] = {s: {} for s in self._subsystems}
                # Sign propagation depends on which gate was actually
                # interleaved for this experiment (the empty Cycle for
                # 'reference'), matching generate_circuits().
                interleaved_op = (
                    self._cycle_or_circuit if experiment == 'interleaved'
                    else Cycle()
                )

                for group in self._pauli_groups:
                    measurement_basis = ''.join(group.measurement_basis)

                    for pauli_tuple in group.paulis:
                        pauli = ''.join(pauli_tuple)
                        active = [
                            i for i, p in enumerate(pauli_tuple)
                            if p != 'I'
                        ]

                        # EV (per Pauli) at each depth, as a ufloat
                        # of the mean ± SEM over randomizations, in
                        # self._circuit_depths order.
                        pauli_evs: list[ufloat] = []
                        for depth in self._circuit_depths:
                            evs: list[float] = []

                            for r in range(self._n_randomizations):
                                subset = self._circuits.subset(
                                    experiment=experiment,
                                    measurement_basis=measurement_basis,
                                    depth=depth,
                                    randomization=r,
                                )
                                if len(subset) == 0:
                                    continue
                                results = subset.circuit.iloc[0].results
                                member_idx = (
                                    subset['group_paulis'].iloc[0].index(
                                        pauli
                                    )
                                )
                                sign = (
                                    subset['group_signs'].iloc[0][
                                        member_idx
                                    ]
                                )
                                evs.append(
                                    sign
                                    * results.marginalize(tuple(active)).ev
                                )

                                if self._analyze_subsystems:
                                    twirl_tuples = [
                                        tuple(ts) for ts in
                                        subset['twirl_strings'].iloc[0]
                                    ]
                                    self._accumulate_subsystem_evs(
                                        pauli_tuple, depth, results,
                                        twirl_tuples, interleaved_op,
                                        subsystem_local_evs,
                                    )

                            pauli_evs.append(_mean_sem(evs))

                        decay = np.array(pauli_evs)
                        self._pauli_decays[experiment][pauli] = decay
                        fit, f = _fit_decay_curve(
                            depths, decay, f"Pauli {pauli}", experiment
                        )
                        if fit is not None:
                            self._fit[experiment][pauli] = fit
                            self._pauli_fidelities[experiment][pauli] = f

                if self._analyze_subsystems:
                    self._fit_subsystem_decays(
                        experiment, depths, subsystem_local_evs
                    )

                # Analyze the fidelity for each experiment
                # The interleaved experiment's process infidelity is the
                # dressed infidelity of cycle_or_circuit (it still
                # includes the decay contributed by the random Pauli
                # twirls), as opposed to the bare infidelity self._e_F
                # computed below when include_ref_cycle is True.
                infidelity_key = (
                    'dressed' if experiment == 'interleaved' else experiment
                )
                fidelities = list(
                    self._pauli_fidelities[experiment].values()
                )
                if fidelities:
                    polarization, e_F = _polarization_and_infidelity(
                        fidelities, len(self._qubits)
                    )
                    polarizations[experiment] = polarization
                    self._process_infidelities[infidelity_key] = e_F
                else:
                    logger.warning(
                        f" No valid fidelity estimates ({experiment})."
                    )
                    self._process_infidelities[infidelity_key] = np.nan

                # Always run, even when analyze_subsystems is False:
                # self._subsystems still holds the whole-cycle entry,
                # and this is what populates its dressed infidelity
                # (and, via subsystem_polarizations, its bare
                # infidelity below) for the summary table.
                self._compute_subsystem_infidelities(
                    experiment, subsystem_polarizations
                )

            # Compute the bare cycle infidelity from the interleaved
            # and reference polarizations
            if self._include_ref_cycle:
                f_D = polarizations.get('interleaved')
                f_ref = polarizations.get('reference')
                if f_D is not None and f_ref is not None:
                    self._e_F = compute_cycle_infidelity(
                        f_D, f_ref, len(self._qubits)
                    )
                else:
                    logger.warning(
                        " Unable to estimate the bare cycle infidelity."
                    )
                    self._e_F = np.nan
                self._compute_subsystem_bare_infidelities(
                    subsystem_polarizations
                )
            else:
                self._e_F = self._process_infidelities.get(
                    'dressed', np.nan
                )

            self._build_subsystem_summary()

        def _accumulate_subsystem_evs(
            self,
            pauli_tuple:          PauliString,
            depth:                int,
            results:              Results,
            twirl_tuples:         list[PauliString],
            interleaved_op:       Cycle,
            subsystem_local_evs:  dict[
                tuple, dict[tuple, dict[int, list[float]]]
            ],
        ) -> None:
            """Marginalize one circuit's result onto every subsystem
            pauli_tuple has non-identity overlap with.

            For each subsystem S, the local pattern is pauli_tuple's
            substring at S's positions. Its sign is obtained by calling
            _propagate_sign on that pattern zero-padded to full length:
            'I' never anticommutes, so positions outside S trivially
            drop out of the parity computation, leaving exactly S's
            local sign (see the analyze_subsystems note on `analyze`).

            Args:
                pauli_tuple (PauliString): the full decay Pauli this
                    circuit was sampled for.
                depth (int): circuit depth.
                results (Results): this circuit's measurement results.
                twirl_tuples (list[PauliString]): this circuit's d+1
                    Pauli twirl layers.
                interleaved_op (Cycle): the gate actually interleaved
                    for this experiment (cycle_or_circuit, or an empty
                    Cycle for the reference experiment).
                subsystem_local_evs (dict): accumulator, mutated in
                    place: subsystem -> local pattern -> depth -> list
                    of signed EVs.
            """
            for s in self._subsystems:
                positions_s = self._subsystem_positions[s]
                if len(positions_s) == len(self._qubits):
                    # The full register: already fit directly above,
                    # marginalizing to "all positions" would just
                    # redundantly refit the same full-length Pauli.
                    continue

                local_pattern = tuple(pauli_tuple[i] for i in positions_s)
                if all(p == 'I' for p in local_pattern):
                    continue

                local_active = [
                    i for i in positions_s if pauli_tuple[i] != 'I'
                ]
                local_sign = _propagate_sign(
                    _pad_pauli(
                        local_pattern, positions_s, len(self._qubits)
                    ),
                    twirl_tuples, interleaved_op, depth,
                )
                local_ev = (
                    local_sign * results.marginalize(tuple(local_active)).ev
                )
                subsystem_local_evs[s].setdefault(
                    local_pattern, {}
                ).setdefault(depth, []).append(local_ev)

        def _fit_subsystem_decays(
            self,
            experiment:           str,
            depths:               np.ndarray,
            subsystem_local_evs:  dict[
                tuple, dict[tuple, dict[int, list[float]]]
            ],
        ) -> None:
            """Fit each subsystem's pooled local-pattern decays and
            store them in self._subsystem_pauli_decays/
            _subsystem_pauli_fidelities, keyed by the local (not
            zero-padded) Pauli string (e.g. 'IX' on a subsystem's
            positions).

            This pools data from every sampled decay Pauli that shares
            that local substring, however they differ outside the
            subsystem. Deliberately kept separate from
            self._pauli_decays/_pauli_fidelities: those must remain
            exactly the originally sampled full-register Pauli set,
            since the whole-cycle infidelity is a uniform average over
            it -- merging in subsystem-local reconstructions would
            over-represent low-weight (higher-fidelity) Paulis and bias
            that average low (see self._subsystem_pauli_fidelities).
            """
            for s in self._subsystems:
                positions_s = self._subsystem_positions[s]
                if len(positions_s) == len(self._qubits):
                    continue  # already fit directly; see analyze()
                for local_pattern, depth_evs in (
                    subsystem_local_evs[s].items()
                ):
                    local_key = ''.join(local_pattern)
                    decay = _decay_array(self._circuit_depths, depth_evs)
                    fit, f = _fit_decay_curve(
                        depths, decay,
                        f"subsystem {s} pattern '{local_key}'", experiment,
                    )
                    if fit is None:
                        continue
                    self._subsystem_pauli_decays[experiment].setdefault(
                        s, {}
                    )[local_key] = decay
                    self._subsystem_fit[experiment].setdefault(
                        s, {}
                    )[local_key] = fit
                    self._subsystem_pauli_fidelities[experiment].setdefault(
                        s, {}
                    )[local_key] = f

        def _compute_subsystem_infidelities(
            self,
            experiment:               str,
            subsystem_polarizations:  dict[str, dict[tuple, ufloat]],
        ) -> None:
            """Average each subsystem's Pauli fidelities into a
            polarization and compute its dressed process infidelity,
            mirroring the whole-cycle calculation in `analyze`.

            The full register uses self._pauli_fidelities directly
            (the originally sampled set, unmodified); every other
            subsystem uses its own marginalize-and-refit-pooled
            self._subsystem_pauli_fidelities (see
            _fit_subsystem_decays) -- these are intentionally not
            mixed (see self._subsystem_pauli_fidelities).
            """
            for s in self._subsystems:
                if len(self._subsystem_positions[s]) == len(self._qubits):
                    local_fidelities = list(
                        self._pauli_fidelities[experiment].values()
                    )
                else:
                    local_fidelities = list(
                        self._subsystem_pauli_fidelities[experiment]
                        .get(s, {}).values()
                    )
                if not local_fidelities:
                    logger.warning(
                        f" No local Pauli decays found for subsystem "
                        f"{s} ({experiment})."
                    )
                    continue

                polarization, e_F_sub = _polarization_and_infidelity(
                    local_fidelities, len(s)
                )
                subsystem_polarizations[experiment][s] = polarization
                self._subsystem_infidelities[experiment][s] = e_F_sub

        def _compute_subsystem_bare_infidelities(
            self,
            subsystem_polarizations: dict[str, dict[tuple, ufloat]],
        ) -> None:
            """Compute each subsystem's bare (reference-corrected)
            infidelity from its interleaved and reference
            polarizations, mirroring the whole-cycle self._e_F
            calculation in `analyze`.
            """
            for s in self._subsystems:
                f_D_sub = subsystem_polarizations['interleaved'].get(s)
                f_ref_sub = subsystem_polarizations['reference'].get(s)
                if f_D_sub is None or f_ref_sub is None:
                    logger.warning(
                        " Unable to estimate the bare infidelity for "
                        f"subsystem {s}."
                    )
                    continue
                self._subsystem_bare_infidelities[s] = (
                    compute_cycle_infidelity(f_D_sub, f_ref_sub, len(s))
                )

        def _build_subsystem_summary(self) -> None:
            """Build and display the per-subsystem infidelity summary.

            One row per entry of self._subsystems -- the whole cycle
            (always first) plus one row per gate body when
            analyze_subsystems is True -- with columns 'Subset',
            'Operation', 'Dressed Infidelity', and (if
            include_ref_cycle) 'Reference Infidelity' and 'Bare
            Infidelity'. Stored in self._subsystem_summary and
            displayed via IPython.display.display.
            """
            def format_infidelity(e_F) -> str:
                if e_F is None or (
                    isinstance(e_F, float) and np.isnan(e_F)
                ):
                    return 'N/A'
                val, err = round_to_order_error(e_F.n * 100, e_F.s * 100)
                return f"{val}% ± {err}%"

            rows = []
            for s in self._subsystems:
                row = {
                    'Subset': s,
                    'Operation': self._subsystem_gate_names[s],
                    'Dressed Infidelity': format_infidelity(
                        self._subsystem_infidelities['interleaved'].get(s)
                    ),
                }
                if self._include_ref_cycle:
                    row['Reference Infidelity'] = format_infidelity(
                        self._subsystem_infidelities['reference'].get(s)
                    )
                    row['Bare Infidelity'] = format_infidelity(
                        self._subsystem_bare_infidelities.get(s)
                    )
                rows.append(row)

            self._subsystem_summary = pd.DataFrame(rows)
            print("\nCB Process Infidelity Summary:")
            display(self._subsystem_summary)
            print("\n")

        def plot(self) -> None:
            """Plot per-Pauli decay curves and Pauli infidelities.

            Generates two figures per (experiment, subsystem) pair --
            self._subsystems always includes the whole cycle (the full
            register) as its first entry, plus one entry per gate body
            when analyze_subsystems is True:
              1. Raw decays: one subplot per Pauli string supported on
                 that subsystem, showing the mean EV per depth
                 (markers) and the fitted A * f_P^depth curve (line),
                 with f_P ± uncertainty in the legend.
              2. Pauli infidelities: a bar plot of 1 - f_P per Pauli
                 string supported on that subsystem, with that
                 subsystem's process infidelity drawn as a horizontal
                 line.

            If include_ref_cycle is True, the estimated bare infidelity
            (self.subsystem_infidelities, already displayed by
            analyze() in the summary table) is not drawn on either
            experiment's own plot, since it is not a per-Pauli quantity
            of either curve alone.

            If analyze_subsystems is True, a QPU heatmap of the
            marginalized per-subsystem infidelities (self.
            subsystem_infidelities) is drawn first, before the raw
            per-subsystem plots described above.

            Plotly figures are shown for interactive use; matplotlib
            equivalents are saved to disk when Settings.save_data is
            True.
            """
            if self._analyze_subsystems:
                self._plot_subsystem_heatmap()

            for experiment in self._experiments:
                for s in self._subsystems:
                    self._plot_subsystem(experiment, s)

        def _plot_subsystem_heatmap(self) -> None:
            """Plot a QPU heatmap of the marginalized subsystem
            infidelities.

            Draws self.subsystem_infidelities (dressed infidelity by
            default, or bare/reference-corrected infidelity if
            include_ref_cycle was True) over the gate-body subsystems
            only, i.e. self._subsystems excluding the whole-cycle
            entry, since the latter isn't a single qubit/pair node or
            edge on the QPU graph.
            """
            values = {}
            for s in self._subsystems[1:]:
                e_F = self.subsystem_infidelities.get(s)
                if e_F is None:
                    continue
                values[s[0] if len(s) == 1 else s] = e_F.n

            if not values:
                return

            draw_qpu_heatmap(
                self.config,
                values,
                label='Cycle Benchmarking',
                cbar_label=(
                    'Bare Gate Infidelity' if self._include_ref_cycle else
                    'Dressed Gate Infidelity'
                ),
            )

        def _plot_subsystem(self, experiment: str, s: tuple) -> None:
            """Plot one subsystem's raw decays and Pauli infidelities
            for one experiment (see `plot`).
            """
            colors = qualitative.Plotly
            all_depths = np.array(self._circuit_depths, dtype=float)

            positions_s = self._subsystem_positions[s]
            if len(positions_s) == len(self._qubits):
                # The full register: plot the originally sampled set
                # directly, exactly as the pre-subsystem-feature plot
                # did.
                pauli_fidelities = self._pauli_fidelities[experiment]
                pauli_decays = self._pauli_decays[experiment]
                fits = self._fit[experiment]
            else:
                pauli_fidelities = self._subsystem_pauli_fidelities[
                    experiment
                ].get(s, {})
                pauli_decays = self._subsystem_pauli_decays[
                    experiment
                ].get(s, {})
                fits = self._subsystem_fit[experiment].get(s, {})
            e_F = self._subsystem_infidelities[experiment].get(s, np.nan)

            gate_n = self._subsystem_gate_names[s]
            qubit_str = ', '.join(str(q) for q in s)
            title_prefix = f"{experiment.capitalize()} {gate_n} ({qubit_str})"
            qubit_suffix = '_'.join(str(q) for q in s)
            file_suffix = (
                f'_{gate_n}_{qubit_suffix}'
                if experiment == 'interleaved'
                else f'_{experiment}_{gate_n}_{qubit_suffix}'
            )
            paulis = sorted(pauli_decays.keys())

            # ---- Plot 1: raw decays --------------------------
            if paulis:
                pfig = go.Figure()

                if Settings.save_data:
                    mfig, ax = plt.subplots(
                        figsize=(6, 5), layout='constrained'
                    )

                for k, pauli in enumerate(paulis):
                    color = colors[k % len(colors)]
                    nominal = unumpy.nominal_values(
                        pauli_decays[pauli]
                    )
                    all_errs = unumpy.std_devs(pauli_decays[pauli])
                    valid = ~np.isnan(nominal)
                    depths = all_depths[valid]
                    evs = nominal[valid]
                    errs = all_errs[valid]

                    fit = fits.get(pauli)
                    f_p = pauli_fidelities.get(pauli)
                    has_fit = (
                        fit is not None and fit.fit_success
                        and depths.size
                    )
                    label = (
                        f'{pauli}: f={f_p.n:.4f} ({f_p.s:.4f})'
                        if has_fit and f_p is not None
                        else f'{pauli}: Fit' if has_fit else pauli
                    )

                    pfig.add_trace(
                        go.Scatter(
                            x=depths, y=evs, mode='markers',
                            marker={'size': 8, 'color': color},
                            error_y={
                                'type': 'data', 'array': errs,
                                'visible': True,
                            },
                            name=label, legendgroup=pauli,
                            showlegend=not has_fit,
                        ),
                    )
                    if Settings.save_data:
                        ax.errorbar(
                            depths, evs, yerr=errs, fmt='o',
                            color=color, markersize=6, capsize=3,
                            label=None if has_fit else label,
                        )

                    if has_fit:
                        xfit = np.linspace(
                            0, 1.1 * depths.max(), 200
                        )
                        yfit = fit.predict(xfit)
                        pfig.add_trace(
                            go.Scatter(
                                x=xfit, y=yfit, mode='lines',
                                line={'color': color, 'width': 2},
                                name=label, legendgroup=pauli,
                                showlegend=True,
                            ),
                        )
                        if Settings.save_data:
                            ax.plot(
                                xfit, yfit, '-', color=color,
                                label=label,
                            )

                pfig.update_xaxes(
                    title_text='Cycle Depth', showgrid=True,
                )
                pfig.update_yaxes(
                    title_text='Expectation Value', showgrid=True,
                )
                pfig.update_layout(
                    height=500,
                    width=750,
                    template='plotly_white',
                    paper_bgcolor='white',
                    plot_bgcolor='#fbfbfd',
                    title_text=f'{title_prefix} Pauli Decays',
                    margin={'t': 50},
                )
                pfig.update_xaxes(
                    showline=True, mirror=True, linecolor='#c7c7c7',
                    linewidth=1, gridcolor='#e5e7eb', zeroline=False,
                    ticks='outside',
                )
                pfig.update_yaxes(
                    showline=True, mirror=True, linecolor='#c7c7c7',
                    linewidth=1, gridcolor='#e5e7eb', zeroline=False,
                    ticks='outside',
                )
                pfig.show()

                if Settings.save_data:
                    ax.set_xlabel('Cycle Depth')
                    ax.set_ylabel('Expectation Value')
                    ax.grid(True)
                    ax.legend(fontsize=8)
                    mfig.suptitle(f'{title_prefix} Pauli Decays')
                    mfig.savefig(
                        self._data_manager._save_path
                        + f'CB_decays{file_suffix}.png',
                        dpi=300,
                    )
                    mfig.savefig(
                        self._data_manager._save_path
                        + f'CB_decays{file_suffix}.pdf'
                    )
                    mfig.savefig(
                        self._data_manager._save_path
                        + f'CB_decays{file_suffix}.svg'
                    )
                    plt.close(mfig)

            # ---- Plot 2: Pauli infidelities ------------------
            if pauli_fidelities and not (
                isinstance(e_F, float) and np.isnan(e_F)
            ):
                pauli_labels = sorted(pauli_fidelities.keys())
                infidelities = [
                    1 - pauli_fidelities[p] for p in pauli_labels
                ]
                y = [inf.n for inf in infidelities]
                yerr = [inf.s for inf in infidelities]
                e_F_label = f'e_F = {e_F.n:.2e} ({e_F.s:.2e})'

                pfig2 = go.Figure()
                pfig2.add_trace(
                    go.Bar(
                        x=pauli_labels, y=y,
                        error_y={
                            'type': 'data', 'array': yerr,
                            'visible': True
                        },
                        marker_color='#1f77b4',
                        name='Pauli infidelity',
                        showlegend=False,
                    )
                )
                pfig2.add_hrect(
                    y0=e_F.n - e_F.s, y1=e_F.n + e_F.s,
                    fillcolor='#F26C8C', opacity=0.3, line_width=0,
                )
                pfig2.add_hline(
                    y=e_F.n, line={'color': '#F26C8C', 'dash': 'dash'},
                )
                pfig2.add_annotation(
                    text=e_F_label,
                    xref='paper', yref='paper',
                    x=0.02, y=0.98,
                    xanchor='left', yanchor='top',
                    showarrow=False,
                    font={'color': '#F26C8C'},
                    bgcolor='white',
                    bordercolor='#F26C8C',
                    borderwidth=1,
                    borderpad=4,
                )
                pfig2.update_layout(
                    height=450,
                    width=min(150 * len(pauli_labels) + 150, 1000),
                    template='plotly_white',
                    paper_bgcolor='white',
                    plot_bgcolor='#fbfbfd',
                    title_text=f'{title_prefix} Pauli Infidelities',
                    margin={'t': 50},
                )
                pfig2.update_xaxes(
                    title_text='Pauli Decay Term', type='category',
                    showgrid=True, showline=True, mirror=True,
                    linecolor='#c7c7c7', linewidth=1,
                    gridcolor='#e5e7eb', zeroline=False,
                    ticks='outside', tickangle=-45,
                )
                pfig2.update_yaxes(
                    title_text='Infidelity', showgrid=True,
                    showline=True, mirror=True, linecolor='#c7c7c7',
                    linewidth=1, gridcolor='#e5e7eb', zeroline=False,
                    ticks='outside',
                )
                pfig2.show()

                if Settings.save_data:
                    mfig2 = plt.figure(
                        figsize=(min(0.6 * len(pauli_labels) + 3, 12), 5)
                    )
                    x = np.arange(len(pauli_labels))
                    plt.bar(x, y, yerr=yerr, color='#1f77b4', capsize=3)
                    plt.axhspan(
                        e_F.n - e_F.s, e_F.n + e_F.s,
                        color='#F26C8C', alpha=0.3,
                    )
                    plt.axhline(
                        e_F.n, color='#F26C8C', linestyle='--'
                    )
                    plt.text(
                        0.02, 0.98, e_F_label,
                        transform=plt.gca().transAxes,
                        ha='left', va='top',
                        color='#F26C8C', fontsize=12,
                        bbox={
                            'facecolor': 'white',
                            'edgecolor': '#F26C8C',
                            'boxstyle': 'round,pad=0.3',
                        },
                    )
                    plt.xticks(x, pauli_labels, rotation=45, ha='right')
                    plt.xlabel('Pauli Decay Term', fontsize=15)
                    plt.ylabel('Infidelity', fontsize=15)
                    plt.grid(True)
                    mfig2.set_tight_layout(True)
                    mfig2.savefig(
                        self._data_manager._save_path
                        + f'CB_infidelities{file_suffix}.png',
                        dpi=600,
                    )
                    mfig2.savefig(
                        self._data_manager._save_path
                        + f'CB_infidelities{file_suffix}.pdf'
                    )
                    mfig2.savefig(
                        self._data_manager._save_path
                        + f'CB_infidelities{file_suffix}.svg'
                    )
                    plt.close(mfig2)

        def save(self) -> None:
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f"_CB{''.join('Q' + str(q) for q in self._qubits)}"
            )
            if Settings.save_data:
                qpu.save(self)

        def final(self) -> None:
            """Final benchmarking method."""
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self) -> None:
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.save()
            self.analyze()
            self.plot()
            self.final()

    return CB(
        config=config,
        cycle_or_circuit=cycle_or_circuit,
        circuit_depths=circuit_depths,
        n_decays=n_decays,
        n_randomizations=n_randomizations,
        decompose_to_zxzxz=decompose_to_zxzxz,
        include_ref_cycle=include_ref_cycle,
        targeted_decays=targeted_decays,
        analyze_subsystems=analyze_subsystems,
        **kwargs,
    )


def CB1(
    qpu:                  QPU,
    config:               Config,
    cycle:                dict | trueq.Cycle,  # noqa: F821 # type: ignore
    circuit_depths:       Iterable[int],
    tq_config:            str | trueq.Config | None = None,  # noqa: F821 # type: ignore
    n_circuits:           int = 30,
    n_decays:             int = 20,
    targeted_errors:      Iterable[str] | trueq.math.Weyls | None = None,  # noqa: F821 # type: ignore
    twirl:                str | trueq.Twirl = "P",  # noqa: F821 # type: ignore
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
        cycle (dict, trueq.Cycle): cycle (or subcircuit) to benchmark.
        circuit_depths (Iterable[int]): a list of positive integers specifying
            how many interleaved cycles of the target cycle and
            random Pauli operators to generate, for example, [4, 16, 64].
        tq_config (str | trueq.Config | None, optional): True-Q config yaml file
            or config object. Defaults to None.
        n_circuits (int, optional): the number of circuits for each circuit
            depth. Defaults to 30.
        n_decays (int, optional): an integer specifying the total number of
            randomly  chosen Pauli decay strings used to measure the process
            infidelity or the probability of each error. Defaults to 20.
            Warning: Setting this value lower than min(20, 4 ** n_qubits - 1)
            may result in a biased estimate of the process fidelity, and
            setting this value lower than min(40, 4 ** n_qubits - 1) may result
            in a biased estimate of the probability for non-identity errors.
        targeted_errors (Iterable[str] | trueq.math.Weyls | None, optional): A
            True-Q Weyls instance, where each row specifies an error to measure.
            Defaults to None. The identity Pauli will always be added to the
            list of errors (or be the sole target if None is the argument),
            which corresponds to measuring the process fidelity of the cycle.
            For convenience, a list of strings can be given, e.g. ["XII",
            "ZZY"], which will be used to instantiate a Weyls object.
        twirl (str | trueq.Twirl, optional): The Twirl to use in this protocol.
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

        def __init__(
            self,
            config:               Config,
            cycle:                dict | trueq.Cycle,  # noqa: F821 # type: ignore
            circuit_depths:       Iterable[int],
            tq_config:            str | trueq.Config = None,  # noqa: F821 # type: ignore
            n_circuits:           int = 30,
            n_decays:             int = 20,
            targeted_errors:      Iterable[str] | trueq.math.Weyls | None = None,  # noqa: F821 # type: ignore
            twirl:                str | trueq.Twirl = "P",  # noqa: F821 # type: ignore
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
                    e_C, err = compute_cycle_infidelity1(
                        cycle_subset, ref_subset
                    )
                    print(
                        f"Bare cycle infidelity: e_C = {e_C} ({err})\n"
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
                f"_CB1{''.join('Q' + str(q) for q in self._circuits.labels)}"
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


def SC(
    qpu:                  QPU,
    config:               Config,
    cycle:                dict | trueq.Cycle,  # noqa: F821 # type: ignore
    circuit_depths:       Iterable[int],
    tq_config:            str | trueq.Config = None,  # noqa: F821 # type: ignore
    n_circuits:           int = 30,
    pauli_decays:         Iterable[str] | trueq.math.Weyls | None = None,  # noqa: F821 # type: ignore
    twirl:                str | trueq.Twirl = "P",  # noqa: F821 # type: ignore
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
        cycle (dict, trueq.Cycle): cycle (or subcircuit) to benchmark.
        circuit_depths (Iterable[int]): iterable of positive integers
            specifying how many interleaved cycles of the target cycle and
            random Pauli operators to generate, for example, [4, 16, 64].
        tq_config (str | trueq.Config, optional): True-Q config yaml file or
            config object. Defaults to None.
        n_circuits (int, optional): the number of circuits for each circuit
            depth. Defaults to 30.
        pauli_decays (Iterable[str] | trueq.math.Weyls | None, optional): A
            True-Q Weyls instance, where the rows specify which elements of the
            diagonalized error channel should be estimated. These should be
            chosen to anticommute with the Hamiltonian terms of a known noise
            source to be optimized. As a convenience, a list of strings can be
            given, e.g. ["XII", "ZZY"], which will be used to instantiate a
            Weyls object.
        twirl (str | trueq.Twirl, optional): The Twirl to use in this protocol.
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
            config:               Config,
            cycle:                dict | trueq.Cycle,  # noqa: F821 # type: ignore
            circuit_depths:       Iterable[int],
            tq_config:            str | trueq.Config = None,  # noqa: F821 # type: ignore
            n_circuits:           int = 30,
            pauli_decays:         Iterable[str] | trueq.math.Weyls | None = None,  # noqa: F821 # type: ignore
            twirl:                str | trueq.Twirl = "P",  # noqa: F821 # type: ignore
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


def _mean_sem(evs: list[float]) -> ufloat:
    """Mean ± SEM of a list of signed expectation values.

    Args:
        evs (list[float]): signed expectation values from repeated
            randomizations of the same (Pauli, depth) pair.

    Returns:
        ufloat: ufloat(nan, nan) if evs is empty; ufloat(mean, 0) if
            there is exactly one value; otherwise ufloat(mean, SEM).
    """
    if not evs:
        return ufloat(np.nan, np.nan)
    mean = float(np.mean(evs))
    sem = (
        float(np.std(evs, ddof=1)) / np.sqrt(len(evs))
        if len(evs) > 1 else 0.0
    )
    return ufloat(mean, sem)


def _decay_array(
    circuit_depths: Sequence[int],
    evs_by_depth:   dict[int, list[float]],
) -> np.ndarray:
    """Build a per-depth decay array of ufloats (mean ± SEM).

    Args:
        circuit_depths (Sequence[int]): depths, in output order.
        evs_by_depth (dict[int, list[float]]): depth to list of signed
            EVs from repeated randomizations; a missing depth is
            treated as having no data.

    Returns:
        np.ndarray: 1D array of ufloats, one per entry of
            circuit_depths, in the same order.
    """
    return np.array([
        _mean_sem(evs_by_depth.get(depth, [])) for depth in circuit_depths
    ])


def _fit_decay_curve(
    depths:     np.ndarray,
    decay:      np.ndarray,
    label:      str,
    experiment: str,
) -> tuple[FitExponential | None, ufloat | None]:
    """Fit EV(d) = A * f^d (offset fixed to 0) to a decay curve.

    Args:
        depths (np.ndarray): circuit depths (float array).
        decay (np.ndarray): ufloat array of EV(d), same length/order as
            depths; NaN entries are dropped before fitting.
        label (str): human-readable identifier for warning messages,
            e.g. "Pauli XYI" or "subsystem (1, 2) pattern 'IXII'".
        experiment (str): experiment name, for warning messages.

    Returns:
        tuple[FitExponential | None, ufloat | None]: (fit, fidelity) on
            success, where fidelity = exp(-b); (None, None) if there
            wasn't enough valid data, the fit failed, or its
            uncertainty couldn't be estimated (a warning is logged for
            whichever case applies).
    """
    evs_arr = unumpy.nominal_values(decay)
    valid = ~np.isnan(evs_arr)
    if valid.sum() < 2:
        logger.warning(f" Not enough data for {label} ({experiment}).")
        return None, None

    fit = FitExponential()
    params = fit.model.make_params(a=1.0, b=0.01, c=0)
    params['c'].vary = False
    fit.fit(depths[valid], evs_arr[valid], params=params)
    if not fit.fit_success:
        logger.warning(f" Fit failed for {label} ({experiment}).")
        return None, None

    b = fit.fit_params['b']
    if b.stderr is None:
        logger.warning(
            f" Unable to estimate fit uncertainty for {label} "
            f"({experiment})."
        )
        return None, None

    return fit, uexp(-ufloat(b.value, b.stderr))


def _polarization_and_infidelity(
    fidelities: list[ufloat],
    n_qubits:   int,
) -> tuple[ufloat, ufloat]:
    """Average a list of Pauli fidelities into a polarization, and
    convert that to a process infidelity.

    Args:
        fidelities (list[ufloat]): per-Pauli fidelities to average.
        n_qubits (int): number of qubits the average is over.

    Returns:
        tuple[ufloat, ufloat]: (polarization, process infidelity),
            where e_F = (d^2 - 1) / d^2 * (1 - polarization) and
            d = 2**n_qubits.
    """
    polarization = sum(fidelities) / len(fidelities)
    d = 2 ** n_qubits
    e_F = (d**2 - 1) / d**2 * (1 - polarization)
    return polarization, e_F


def _propagate_sign(
    decay_pauli:      PauliString,
    twirl_paulis:     list[PauliString],
    cycle_or_circuit: Cycle | Circuit,
    depth:            int,
) -> int:
    """Determine the ±1 eigenstate sign after ideal evolution.

    In the Heisenberg picture, the measurement observable S is propagated
    backward through the circuit.  Each application of the cycle G
    transforms S → G†SG, so the effective Pauli that twirl layer i sees
    alternates between S and G(S) = GSG†.

    For 0-indexed twirl position i (i = 0 is the leading twirl, before the
    first cycle application), the effective Pauli is:
      G(S)  if (i + depth) % 2 == 1
      S     otherwise

    The sign flips whenever the effective Pauli and the twirl anticommute
    (odd number of qubit positions where both are non-identity and differ).

    Args:
        decay_pauli (PauliString): the Pauli decay string, ordered by
            `cycle_or_circuit.qubits`, e.g. ('X', 'Z', 'I').
        twirl_paulis (list[PauliString]): all d+1 Pauli twirl layers.
        cycle_or_circuit (Cycle | Circuit): the benchmarked cycle or
            circuit.
        depth (int): number of cycle applications d.

    Returns:
        int: +1 or -1.
    """
    g_pauli, _ = conjugate_pauli(decay_pauli, cycle_or_circuit)
    sign = 1
    for i, twirl in enumerate(twirl_paulis):
        effective = g_pauli if (i + depth) % 2 == 1 else decay_pauli
        n_anticommuting = sum(
            1
            for p, q in zip(effective, twirl, strict=True)
            if p != 'I' and q != 'I' and p != q
        )
        if n_anticommuting % 2 == 1:
            sign *= -1
    return sign
