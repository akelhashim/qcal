"""Submodule for handling dynamical decoupling sequences.


Two-qubit syncopation matrix — Table I of Evert et al., "Syncopated
Dynamical Decoupling for Suppressing Crosstalk in Quantum Circuits",
Phys. Rev. Applied 24, 044025 (2025) [arXiv:2403.07836].

Each cell shows which static two-qubit Pauli couplings (out of XX, YY,
ZZ) are averaged out to first order when the row-labeled sequence is
applied to one qubit and the column-labeled sequence to the other,
both spanning the same total idle time t. X / Y in sequence names
refer to pi_x / pi_y pulses.

Pulse-time conventions:
    Length-2 sequence  P1P2:           (t/2, t)
    Length-2 CPMG      P1P2-CPMG:      (t/4, 3t/4)
    Length-4 sequence  P1P2P3P4:       (t/4, t/2, 3t/4, t)
    Length-4 CPMG      P1P2P3P4-CPMG:  (t/8, 3t/8, 5t/8, 7t/8)

Cell codes (subset of {XX, YY, ZZ} that is decoupled):
    .   none -- diagonal: identical synchronized sequences
    Z   ZZ
    XY  XX, YY
    XZ  XX, ZZ
    YZ  YY, ZZ
    *   all three (XX, YY, ZZ)

Sequence index legend:
    1  XX            7  YXYX
    2  XX-CPMG       8  YXYX-CPMG
    3  XXXX          9  YY
    4  XXXX-CPMG    10  YY-CPMG
    5  XYXY         11  YYYY
    6  XYXY-CPMG    12  YYYY-CPMG

                            Qubit 1
            1   2   3   4   5   6   7   8   9  10  11  12
        +-----------------------------------------------------
      1 |   .  YZ  YZ  YZ   *  XZ  XZ  XZ  XY   *   *   *
      2 |  YZ   .  YZ  YZ  XZ  XZ   *  XZ   *  XY   *   *
      3 |  YZ  YZ   .  YZ  XY   *  XY   *   *   *  XY   *
      4 |  YZ  YZ  YZ   .   *  XY   *  XY   *   *   *  XY
Qubit 5 |   *  XZ  XY   *   .   Z  XY   Z  YZ   *  XY   *
  0   6 |  XZ  XZ   *  XY   Z   .   Z  XY  YZ  YZ   *  XY
      7 |  XZ   *  XY   *  XY   Z   .   Z   *  YZ  XY   *
      8 |  XZ  XZ   *  XY   Z  XY   Z   .  YZ  YZ   *  XY
      9 |  XY   *   *   *  YZ  YZ   *  YZ   .  XZ  XZ  XZ
     10 |   *  XY   *   *   *  YZ  YZ  YZ  XZ   .  XZ  XZ
     11 |   *   *  XY   *  XY   *  XY   *  XZ  XZ   .  XZ
     12 |   *   *   *  XY   *  XY   *  XY  XZ  XZ  XZ   .

Headline takeaways from the paper:
- ZZ coupling can ONLY be cancelled when the two sequences are
  syncopated -- either timing-shifted (one is the -CPMG variant of
  the other, e.g. (XX, XX-CPMG)) or frequency-doubled (one has twice
  as many pulses, e.g. (XXXX, XX)). Synchronized identical sequences
  on coupled qubits leave ZZ untouched.
- (XX, XX-CPMG) and (XXXX, XX) are the minimum-pulse pairs that
  decouple ZZ on the two qubits while still suppressing single-qubit
  dephasing; they additionally cancel YY.
- Operator-alternation alone (e.g. (XYXY, YXYX)) cancels XX and YY
  but NOT ZZ -- timing or frequency syncopation is required for ZZ.
- The matrix is symmetric in (Q0, Q1).
"""
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

from qcal.circuit import Barrier, Circuit, Cycle
from qcal.gate.single_qubit import X90, Y90, Idle

# from typing import Any, List

logger = logging.getLogger(__name__)


__all__ = ('XY_N', 'DD_SEQUENCES')


def XY_N(
    qubits:     Sequence[int],
    total_time: float,
    gate_time:  float,
    n_pulses:   int | None = None,
    subspace:   str = 'GE',
) -> Circuit:
    """Subcircuit for performing an XY-type dynamical decoupling (DD) sequence.

    This circuit implements a syncopated XY-type DD sequence for more than one
    qubit, but assumes even and odd qubits are nearest neighbors.

    See:
    https://aws.amazon.com/blogs/quantum-computing/suppressing-errors-with-
    dynamical-decoupling-using-pulse-control-on-amazon-braket/

    Args:
        qubits (Sequence[int]): qubit labels.
        total_time (float): total time over which to perform the DD.
        gate_time (float): time of the native X90 gate.
        n_pulses (int | None, optional): number of pulses. Defaults to None. For
            example, for n_pulses = 4, this performs an XY4 DD sequence. If
            None, the number of pulses is set to the maximum number of pulses
            that can fit within the specified time, with a minimum of 4.
        subspace (str, optional): qubit subspace for the DD sequence. Defaults
            to 'GE'. 'EF' is also supported.

    Returns:
        Circuit: qcal Circuit.
    """
    if n_pulses is None:
        n_pulses = max(4, int(total_time / (gate_time * 8)) * 4)
    else:
        if n_pulses % 4 != 0:
            raise ValueError("'n_pulses' must be a multiple of 4!")

    qubits_even = [q for i, q in enumerate(qubits) if i % 2 == 0]
    qubits_odd = [q for i, q in enumerate(qubits) if i % 2 == 1]
    syncopated = bool(qubits_even and qubits_odd)

    # Account for the duration of the X90 gates, which are assumed to be
    # centered on either side of the interval given by tau
    tau = (
        total_time - gate_time if syncopated else total_time
    ) / (2 * n_pulses)

    min_tau = 2 * gate_time if syncopated else gate_time
    if tau < min_tau:
        min_total_time = (
            2 * n_pulses * 2 * gate_time if syncopated
            else 2 * n_pulses * gate_time
        )
        if syncopated:
            logger.warning(
                ' Syncopated XY%d DD sequence will exceed total_time=%.3g: '
                'the fixed gate_time=%.3g idles in the interleaved cycles '
                'cannot be reduced, requiring a minimum total_time of %.3g.',
                n_pulses, total_time, gate_time, min_total_time
            )
        else:
            logger.warning(
                ' XY%d DD sequence will exceed total_time=%.3g: '
                'requires a minimum total_time of %.3g for gate_time=%.3g.',
                n_pulses, total_time, min_total_time, gate_time
            )

    # gate_time for X90 = 0.5 * gate_time for X180
    # The idle_time = tau - 2*gate_time for syncopated sequences, because the
    # X90 gates are assumed to be centered on either side of the interval given
    # by tau. Only the first idle time should be (tau - gate_time)
    idle_time = max(0.0, tau - 2 * gate_time)
    first_idle_time = max(0.0, tau - gate_time)

    if syncopated:
        DD_sequence = Circuit([
            # Idle
            Cycle({Idle(q, duration=first_idle_time) for q in qubits}),

            # X (even)
            Cycle(
                {
                    X90(q, subspace=subspace) for q in qubits_even
                } | {
                    Idle(q, duration=gate_time) for q in qubits_odd
                }
            ),
            # NOTE: time = tau
            Cycle(
                {
                    X90(q, subspace=subspace) for q in qubits_even
                } | {
                    Idle(q, duration=gate_time) for q in qubits_odd
                }
            ),

            # Idle
            Cycle({Idle(q, duration=idle_time) for q in qubits}),

            # X (odd)
            Cycle(
                {
                    X90(q, subspace=subspace) for q in qubits_odd
                } | {
                    Idle(q, duration=gate_time) for q in qubits_even
                }
            ),
            # NOTE: time = 2*tau
            Cycle(
                {
                    X90(q, subspace=subspace) for q in qubits_odd
                } | {
                    Idle(q, duration=gate_time) for q in qubits_even
                }
            ),

            # Idle
            Cycle({Idle(q, duration=idle_time) for q in qubits}),

            # Y (even)
            Cycle(
                {
                    Y90(q, subspace=subspace) for q in qubits_even
                } | {
                    Idle(q, duration=gate_time) for q in qubits_odd
                }
            ),
            # NOTE: time = 3*tau
            Cycle(
                {
                    Y90(q, subspace=subspace) for q in qubits_even
                } | {
                    Idle(q, duration=gate_time) for q in qubits_odd
                }
            ),

            # Idle
            Cycle({Idle(q, duration=idle_time) for q in qubits}),

            # Y (odd)
            Cycle(
                {
                    Y90(q, subspace=subspace) for q in qubits_odd
                } | {
                    Idle(q, duration=gate_time) for q in qubits_even
                }
            ),
            # NOTE: time = 4*tau
            Cycle(
                {
                    Y90(q, subspace=subspace) for q in qubits_odd
                } | {
                    Idle(q, duration=gate_time) for q in qubits_even
                }
            ),

            # Idle
            Cycle({Idle(q, duration=idle_time) for q in qubits}),

            # X (even)
            Cycle(
                {
                    X90(q, subspace=subspace) for q in qubits_even
                } | {
                    Idle(q, duration=gate_time) for q in qubits_odd
                }
            ),
            # NOTE: time = 5*tau
            Cycle(
                {
                    X90(q, subspace=subspace) for q in qubits_even
                } | {
                    Idle(q, duration=gate_time) for q in qubits_odd
                }
            ),

            # Idle
            Cycle({Idle(q, duration=idle_time) for q in qubits}),

            # X (odd)
            Cycle(
                {
                    X90(q, subspace=subspace) for q in qubits_odd
                } | {
                    Idle(q, duration=gate_time) for q in qubits_even
                }
            ),
            # NOTE: time = 6*tau
            Cycle(
                {
                    X90(q, subspace=subspace) for q in qubits_odd
                } | {
                    Idle(q, duration=gate_time) for q in qubits_even
                }
            ),

            # Idle
            Cycle({Idle(q, duration=idle_time) for q in qubits}),

            # Y (even)
            Cycle(
                {
                    Y90(q, subspace=subspace) for q in qubits_even
                } | {
                    Idle(q, duration=gate_time) for q in qubits_odd
                }
            ),
            # NOTE: time = 7*tau
            Cycle(
                {
                    Y90(q, subspace=subspace) for q in qubits_even
                } | {
                    Idle(q, duration=gate_time) for q in qubits_odd
                }
            ),

            # Idle
            Cycle({Idle(q, duration=idle_time) for q in qubits}),

            # Y (odd)
            Cycle(
                {
                    Y90(q, subspace=subspace) for q in qubits_odd
                } | {
                    Idle(q, duration=gate_time) for q in qubits_even
                }
            ),
            # NOTE: time = 8*tau
            Cycle(
                {
                    Y90(q, subspace=subspace) for q in qubits_odd
                } | {
                    Idle(q, duration=gate_time) for q in qubits_even
                }
            ),

            Barrier((q for q in qubits))
        ])

    else:
        inter_idle = max(0.0, 2 * tau - 2 * gate_time)
        DD_sequence = Circuit([
            # Idle
            Cycle({Idle(q, duration=first_idle_time) for q in qubits}),
            # X
            Cycle({X90(q, subspace=subspace) for q in qubits}),
            Cycle({X90(q, subspace=subspace) for q in qubits}),
            # Idle
            Cycle({Idle(q, duration=inter_idle) for q in qubits}),
            # Y
            Cycle({Y90(q, subspace=subspace) for q in qubits}),
            Cycle({Y90(q, subspace=subspace) for q in qubits}),
            # Idle
            Cycle({Idle(q, duration=inter_idle) for q in qubits}),
            # X
            Cycle({X90(q, subspace=subspace) for q in qubits}),
            Cycle({X90(q, subspace=subspace) for q in qubits}),
            # Idle
            Cycle({Idle(q, duration=inter_idle) for q in qubits}),
            # Y
            Cycle({Y90(q, subspace=subspace) for q in qubits}),
            Cycle({Y90(q, subspace=subspace) for q in qubits}),
            # Idle
            Cycle({Idle(q, duration=first_idle_time) for q in qubits}),
            Barrier((q for q in qubits))
        ])

    DD_circuit = Circuit()
    for _ in range(int(n_pulses / 4)):
        DD_circuit.extend(DD_sequence.copy())

    return DD_circuit


DD_SEQUENCES: Mapping[str, Callable] = defaultdict(
    lambda: 'DD sequence not currently supported!', {
        'XY_N': XY_N,
    }
)
