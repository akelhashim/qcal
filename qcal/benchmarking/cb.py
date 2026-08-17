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
from collections.abc import Iterable
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

from qcal.analysis.leakage import analyze_leakage
from qcal.benchmarking.utils import (
    PauliString,
    generate_n_qubit_pauli_measurement_groups,
    generate_random_n_qubit_paulis,
)
from qcal.circuit import Barrier, Circuit, CircuitSet, Cycle
from qcal.config import Config
from qcal.fitting.fit import FitExponential
from qcal.gate.single_qubit import SINGLE_QUBIT_PAULIS, X, Y, Z
from qcal.math.utils import round_to_order_error
from qcal.qpu.qpu import QPU
from qcal.settings import Settings

logger = logging.getLogger(__name__)


__all__ = ['CB', 'CB1', 'SC']


def _composes_to_identity(
    cycle_or_circuit: Cycle | Circuit, depth: int
) -> bool:
    """Assert that cycle_or_circuit^d ≈ I (up to global phase).

    Strategy: use np.linalg.matrix_power to raise the unitary matrix of the
    cycle_or_circuit to the specified depth. A matrix is proportional to
    identity iff M/M[0,0] ≈ I when M[0,0] ≠ 0, or equivalently
    |tr(M^d)|^2 == dim^2.

    Args:
        cycle_or_circuit (Cycle | Circuit): the cycle or circuit to check.
        depth (int): the number of times to compose cycle_or_circuit.

    Returns:
        bool: True if cycle_or_circuit^depth is identity (up to global phase),
            False otherwise.
    """
    M = cycle_or_circuit.unitary
    dim = M.shape[0]
    Md = np.linalg.matrix_power(M, depth)
    # Remove global phase via the first non-zero diagonal element
    diag = np.diag(Md)
    pivot = next(v for v in diag if abs(v) > 1e-10)
    phase = pivot / abs(pivot)
    if np.allclose(Md / phase, np.eye(dim), atol=1e-6):
        return True
    else:
        return False
        # assert np.allclose(Md / phase, np.eye(dim), atol=1e-6), (
        #     f"cycle_or_circuit^{d} is not identity (up to global phase). "
        #     f"Max residual: {np.max(np.abs(Md / phase - np.eye(dim))):.2e}"
        # )


def _pauli_str_to_matrix(pauli: PauliString) -> np.ndarray:
    """Build the n-qubit Pauli matrix for a PauliString."""
    mat = SINGLE_QUBIT_PAULIS[pauli[0]](0).matrix.astype(complex)
    for p in pauli[1:]:
        mat = np.kron(mat, SINGLE_QUBIT_PAULIS[p](0).matrix)
    return mat


def _identify_pauli_str(M: np.ndarray, n: int) -> PauliString:
    """Return Pauli labels for an n-qubit Pauli matrix, ignoring phase.

    Uses recursive block decomposition: for σ_k ⊗ rest, the 2x2 block
    structure of M in the first qubit's index uniquely identifies σ_k.
    """
    if n == 0:
        return ()
    half = M.shape[0] // 2
    M00, M01 = M[:half, :half], M[:half, half:]
    M10, M11 = M[half:, :half], M[half:, half:]

    if np.allclose(M01, 0) and np.allclose(M10, 0):
        if np.allclose(M00, M11):
            label, rest = 'I', M00
        else:
            label, rest = 'Z', M00
    elif np.allclose(M01, M10):
        label, rest = 'X', M01
    else:
        label, rest = 'Y', -1j * M10

    return (label,) + _identify_pauli_str(rest, n - 1)


def _conjugated_pauli(
    pauli: PauliString, U: np.ndarray
) -> PauliString:
    """Return the Pauli string of U P U†, ignoring global phase."""
    P = _pauli_str_to_matrix(pauli)
    return _identify_pauli_str(U @ P @ U.conj().T, len(pauli))


def _pauli_to_cycle(pauli: PauliString, qubits: tuple) -> Cycle | None:
    """Convert a PauliString to a Cycle of Pauli gates.

    Identity components are omitted (no gate on that qubit). Returns None
    if the entire string is identity (caller should skip appending).
    """
    gates = []
    for p, q in zip(pauli, qubits, strict=True):
        if p == 'X':
            gates.append(X(q))
        elif p == 'Y':
            gates.append(Y(q))
        elif p == 'Z':
            gates.append(Z(q))
    return Cycle(gates) if gates else None


def _propagate_sign(
    decay_pauli:   PauliString,
    twirl_paulis:  list[PauliString],
    cycle_unitary: np.ndarray,
    depth:         int,
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
        decay_pauli (PauliString): the Pauli decay string,
            e.g. ('X', 'Z', 'I').
        twirl_paulis (list[PauliString]): all d+1 Pauli twirl layers.
        cycle_unitary (np.ndarray): unitary matrix of the benchmarked
            cycle.
        depth (int): number of cycle applications d.

    Returns:
        int: +1 or -1.
    """
    g_pauli = _conjugated_pauli(decay_pauli, cycle_unitary)
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


def _is_sub_pauli(q: PauliString, p: PauliString) -> bool:
    """Return True if q is a sub-Pauli of p (q_i == p_i or q_i == 'I').

    A sub-Pauli has the same non-identity operators as p at a subset of
    positions, and identity everywhere else.
    """
    return all(
        qi == 'I' or qi == pi
        for qi, pi in zip(q, p, strict=True)
    )



def CB(
    qpu:                QPU,
    config:             Config,
    cycle_or_circuit:   Cycle | Circuit,
    circuit_depths:     Iterable[int],
    n_decays:           int = 20,
    n_randomizations:   int = 30,
    decompose_to_zxzxz: bool = False,
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
            **kwargs,
        ) -> None:
            self._cycle_or_circuit = cycle_or_circuit
            self._circuit_depths = sorted(circuit_depths)
            self._n_decays = n_decays
            self._n_randomizations = n_randomizations
            self._decompose_to_zxzxz = decompose_to_zxzxz
            self._qubits = (
                cycle_or_circuit.qubits
                if isinstance(cycle_or_circuit, Cycle)
                else cycle_or_circuit.labels
            )

            for depth in self._circuit_depths:
                if not _composes_to_identity(cycle_or_circuit, depth):
                    raise ValueError(
                        f"cycle_or_circuit^{depth} is not the identity "
                        "(up to global phase). CB requires that "
                        "cycle_or_circuit composed `depth` times equals "
                        "the identity for every depth in circuit_depths."
                    )

            qpu.__init__(self, config=config, **kwargs)

        def generate_circuits(self) -> None:
            """Generate all CB circuits and store them in self._circuits.

            For each (Pauli, depth, randomization) triple the circuit is:

              prepare(P) | twirl | barrier | [cycle_or_circuit | twirl
                         | barrier]^d | measure(P)

            CircuitSet metadata columns:
              'pauli'          — joined Pauli string, e.g. 'XZI'
              'depth'          — number of cycle_or_circuit instances
              'randomization'  — integer index 0..n_randomizations-1
              'sign'           — ±1 expected eigenstate sign under ideal
                                 evolution
              'twirl_strings'  — list of d+1 joined twirl strings; used
                                 by analyze() to compute per-Pauli signs
                                 for marginalized Paulis
            """
            logger.info(" Generating circuits...")

            sampled_paulis = generate_random_n_qubit_paulis(
                self._qubits, n_random_paulis=self._n_decays
            )
            pauli_groups = generate_n_qubit_pauli_measurement_groups(
                sampled_paulis
            )
            self._pauli_groups = pauli_groups

            circuits: list[Circuit] = []
            pauli_labels: list[str] = []
            depths: list[int] = []
            randomizations: list[int] = []
            signs: list[int] = []
            twirl_strs_list: list[list[str]] = []

            for paulis in pauli_groups.values():
                for pauli in paulis:
                    pauli_str = ''.join(pauli)
                    meas_basis = [
                        'Z' if p == 'I' else p for p in pauli
                    ]

                    for depth in self._circuit_depths:
                        for r in range(self._n_randomizations):
                            # Sample d+1 random Pauli twirl strings
                            twirl_strings = generate_random_n_qubit_paulis(
                                self._qubits, n_random_paulis=depth + 1
                            )
                            sign = _propagate_sign(
                                pauli,
                                twirl_strings,
                                self._cycle_or_circuit.unitary,
                                depth
                            )

                            all_cycles: list[Cycle] = []

                            # if self._decompose_to_zxzxz:
                                
                            leading = _pauli_to_cycle(
                                twirl_strings[0], self._qubits
                            )
                            if leading is not None:
                                all_cycles.append(leading)
                            all_cycles.append(Barrier(self._qubits))

                            for i in range(depth):
                                if isinstance(
                                    self._cycle_or_circuit, Cycle
                                ):
                                    all_cycles.append(
                                        self._cycle_or_circuit
                                    )
                                else:
                                    all_cycles.extend(
                                        self._cycle_or_circuit.cycles
                                    )
                                twirl_cycle = _pauli_to_cycle(
                                    twirl_strings[i + 1], self._qubits
                                )
                                if twirl_cycle is not None:
                                    all_cycles.append(twirl_cycle)
                                all_cycles.append(Barrier(self._qubits))

                            circ = Circuit(all_cycles)
                            circ.prepare(
                                pauli_str,
                                qubits=list(self._qubits),
                            )
                            circ.measure(
                                qubits=list(self._qubits),
                                basis=meas_basis,
                            )
                            circuits.append(circ)
                            pauli_labels.append(pauli_str)
                            depths.append(depth)
                            randomizations.append(r)
                            signs.append(sign)
                            twirl_strs_list.append(
                                [''.join(ts) for ts in twirl_strings]
                            )

            self._circuits = CircuitSet(circuits)
            self._circuits['pauli'] = pauli_labels
            self._circuits['depth'] = depths
            self._circuits['randomization'] = randomizations
            self._circuits['sign'] = signs
            self._circuits['twirl_strings'] = twirl_strs_list

        # ------------------------------------------------------------------
        # Analysis and plotting
        # ------------------------------------------------------------------

        def analyze(self) -> None:
            """Fit per-Pauli decay curves and estimate the cycle error rate.

            For each Pauli Q in a QWC group and each depth d, collects
            mean parity expectation values from ALL circuits whose
            preparation Pauli P has Q as a sub-Pauli (Q_i = P_i or
            Q_i = I). This includes the directly prepared circuits (P = Q)
            and any parent circuits (P ⊋ Q), where EV for Q is recovered
            by marginalizing out the identity positions of Q.

            For each contributing circuit the sign for Q is recomputed
            from the stored twirl strings via _propagate_sign, giving:

              EV(Q, d) = mean_r [ sign_Q(r) * _marginalized_ev(result, Q) ]

            Fits EV(Q, d) = A * f_Q^d per Pauli using FitExponential with
            offset fixed to zero (f_Q = exp(-b)). Reports the process
            infidelity:

              e_F = (d^2 - 1) / d^2 * (1 - mean f_Q over sampled Paulis)

            Results are stored in self._pauli_infidelities (keyed by Pauli
            string) and self._e_F.
            """
            logger.info(" Analyzing the results...")

            depths = np.array(self._circuit_depths, dtype=float)
            self._pauli_infidelities = {}

            for paulis in self._pauli_groups.values():
                for q_tuple in paulis:
                    q_str = ''.join(q_tuple)
                    active = [
                        i for i, p in enumerate(q_tuple)
                        if p != 'I'
                    ]
                    if not active:
                        continue

                    mean_evs: list[float] = []
                    for depth in self._circuit_depths:
                        evs: list[float] = []

                        for p_tuple in paulis:
                            if not _is_sub_pauli(q_tuple, p_tuple):
                                continue
                            p_str = ''.join(p_tuple)

                            for r in range(self._n_randomizations):
                                sub = self._circuits.subset(
                                    pauli=p_str,
                                    depth=depth,
                                    randomization=r,
                                )
                                if len(sub) == 0:
                                    continue
                                result = sub.results.iloc[0]
                                twirl_strs = (
                                    sub['twirl_strings'].iloc[0]
                                )
                                twirl_tuples = [
                                    tuple(ts) for ts in twirl_strs
                                ]
                                sign_q = _propagate_sign(
                                    q_tuple, twirl_tuples,
                                    self._cycle_or_circuit.unitary,
                                    depth,
                                )
                                evs.append(
                                    sign_q * result.marginalize(
                                        tuple(active)
                                    ).ev
                                )

                        mean_evs.append(
                            float(np.mean(evs)) if evs else np.nan
                        )

                    evs_arr = np.array(mean_evs)
                    valid = ~np.isnan(evs_arr)

                    A, f_P, f_err = np.nan, np.nan, np.nan
                    if valid.sum() >= 2:
                        fitter = FitExponential()
                        params = fitter.model.make_params(
                            a=1.0, b=0.01, c=0
                        )
                        params['c'].vary = False
                        fitter.fit(
                            depths[valid], evs_arr[valid],
                            params=params,
                        )
                        if fitter.fit_success:
                            b = fitter.fit_params['b']
                            A = float(fitter.fit_params['a'].value)
                            f_P = float(np.exp(-b.value))
                            f_err = (
                                float(b.stderr * f_P)
                                if b.stderr is not None else np.nan
                            )
                        else:
                            logger.warning(
                                f" Fit failed for Pauli {q_str}."
                            )
                    else:
                        logger.warning(
                            f" Not enough data for Pauli {q_str}."
                        )

                    self._pauli_infidelities[q_str] = {
                        'A': A,
                        'f': f_P,
                        'f_err': f_err,
                        'mean_evs': mean_evs,
                    }

            d_dim = 2 ** len(self._qubits)
            fidelities = [
                v['f'] for v in self._pauli_infidelities.values()
                if not np.isnan(v['f'])
            ]
            if fidelities:
                avg_f = float(np.mean(fidelities))
                self._e_F = (d_dim**2 - 1) / d_dim**2 * (1 - avg_f)
                print(
                    f"\nProcess infidelity: e_F = {self._e_F:.4e}\n"
                )
            else:
                logger.warning(" No valid fidelity estimates.")
                self._e_F = np.nan

        def plot(self) -> None:
            """Plot per-Pauli decay curves (mean EV vs depth) with fitted exponentials.

            One subplot per sampled Pauli string showing:
              - Scatter: mean EV per depth
              - Line: fitted A * f_P^depth
              - Legend entry with f_P ± uncertainty

            TODO: implement after analyze() is complete.
            """
            raise NotImplementedError

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
        **kwargs,
    )


def compute_cycle_infidelity(
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
