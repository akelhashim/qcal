"""Helper functions for PyQuil interface.

NOTE: we do not use TYPE_CHECKING for PyQuil types because this might fail if
PyQuil is not installed when building docs.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Sequence, Tuple

from qcal.sequence.dynamical_decoupling import DD_SEQUENCES

from .transpiler import PyQuilTranspiler, add_X90

logger = logging.getLogger(__name__)


DEFAULT_SYMMETRIC_GATES: FrozenSet[str] = frozenset(
    {"CZ", "SWAP", "ISWAP", "XX", "YY", "ZZ"}
)


@dataclass(frozen=True)
class CycleSignature:
    name: str
    formal_qubits: Tuple[int, ...]
    active_multiset: FrozenSet[Tuple[Tuple, int]]
    idle_qubits: FrozenSet[int]


def _classify(
    instr: AbstractInstruction, symmetric_gates: FrozenSet[str] # type: ignore # noqa: F821
) -> Tuple[str, object]:
    """Classify an instruction for parallel-cycle matching.

    Sorts each instruction into one of three roles the matcher cares about:

    - ``"idle"``: an ``I q`` gate, treated as an idle marker rather than an
      active operation. The payload is the integer qubit index. Idles are
      collected into a separate set per block and checked against the
      DefCircuit's idle qubits as a subset constraint, so manual programs
      can omit them entirely.
    - ``"active"``: a real gate or a measurement that must appear in any
      matching block. The payload is a hashable canonical key suitable for
      use in a ``Counter`` / ``frozenset`` multiset:

      * For ``Gate``: ``("GATE", name, modifiers, params, qubits)``, where
        ``qubits`` is sorted for gates listed in ``symmetric_gates`` (so
        ``CZ 12 13`` and ``CZ 13 12`` produce equal keys) and left in order
        otherwise (preserving control/target distinction for e.g. ``CNOT``).
        Modifiers (``DAGGER``, ``CONTROLLED``, ``FORKED``) and gate
        parameters are part of the key, so a ``DAGGER CZ`` won't match a
        plain ``CZ``.
      * For ``Measurement``: ``("MEASURE", qubit_index)``. The classical
        target is intentionally dropped so that ``MEASURE 27 ro[0]`` in a
        user program matches ``MEASURE q27`` in a DefCircuit body.
    - ``"boundary"``: anything else (``Pragma``, ``Fence``, ``Delay``,
      ``Reset``, frame / pulse-level operations, etc.). The payload is
      ``None``. Boundary instructions terminate the in-progress block and
      are preserved verbatim in the rewritten program.

    Args:
        instr (AbstractInstruction): the pyquil instruction to classify.
        symmetric_gates (FrozenSet[str]): Names of two-qubit gates whose
            argument order is irrelevant. Qubits in active keys for these gates
            are sorted before being placed in the key, so that argument order
            does not affect matching.

    Returns:
        A ``(kind, payload)`` tuple where ``kind`` is one of ``"idle"``,
        ``"active"``, or ``"boundary"``, and ``payload`` is a qubit index,
        a canonical instruction key, or ``None`` respectively.
    """
    try:
        from pyquil.quilbase import Gate, Measurement
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    if _is_identity(instr):
        return ("idle", _qubit_index(instr.qubits[0]))

    elif isinstance(instr, Gate):
        qubits = tuple(_qubit_index(q) for q in instr.qubits)
        if instr.name in symmetric_gates and len(qubits) == 2:
            qubits = tuple(sorted(qubits))
        key = (
            "GATE",
            instr.name,
            tuple(instr.modifiers) if instr.modifiers else (),
            tuple(instr.params),
            qubits,
        )
        return ("active", key)

    elif isinstance(instr, Measurement):
        return ("active", ("MEASURE", _qubit_index(instr.qubit)))

    else:
        return ("boundary", None)


def _get_template_object(instr: Pulse | Capture) -> Any | None:  # type: ignore  # noqa: F821
    """Return the template object carried by a Pulse or Capture instruction.

    For a ``Pulse``, this returns ``instr.waveform``. For a ``Capture``, this
    returns ``instr.kernel``.

    Args:
        instr (Pulse | Capture): a Quil-T instruction that may carry a
            waveform-like template object.

    Returns:
        Any | None: The waveform-like object attached to the instruction,
        or ``None`` if the instruction type is unsupported.
    """
    try:
        from pyquil.quilbase import Capture, Pulse
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    if isinstance(instr, Pulse):
        return instr.waveform
    elif isinstance(instr, Capture):
        return instr.kernel
    else:
        return None


def _is_identity(instr: AbstractInstruction) -> bool:  # type: ignore  # noqa: F821
    """Check whether an instruction is a plain identity gate (``I q``).

    Identity gates have a special role in the matcher: inside a ``DEFCIRCUIT``
    body they are treated as *idle markers* — declarations that a qubit
    participates in the parallel cycle but performs no operation — rather
    than as ordinary single-qubit gates. The matcher therefore needs to
    classify them separately from active gates so it can build the
    signature's ``idle_qubits`` set and apply the looser subset check at
    match time, instead of demanding that manual programs reproduce every
    ``I q`` line verbatim.

    The check requires the gate to be exactly ``I`` with no modifiers
    (e.g. ``DAGGER I q``) and no parameters. A modified or parameterized
    ``I`` is unusual but not impossible, and treating it as a plain idle
    would silently drop information; such instructions fall through to be
    handled as regular active gates instead.

    Args:
        instr (AbstractInstruction): any PyQuil instruction.

    Returns:
        ``True`` if ``instr`` is an unmodified, unparameterized ``I`` gate,
        ``False`` otherwise (including for non-``Gate`` instructions such as
        ``Measurement``, ``Pragma``, or ``Fence``).
    """
    try:
        from pyquil.quilbase import Gate
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    return (
        isinstance(instr, Gate)
        and instr.name == "I"
        and not instr.modifiers
        and not instr.params
    )


def _make_invocation(name: str, qubits: Sequence[int]) -> Gate: # type: ignore  # noqa: F821
    """Construct a DefCircuit invocation as a parameterless ``Gate``.

    A DefCircuit call in Quil is syntactically indistinguishable from a
    gate application — ``CZ_3 1 2 7 ...`` parses the same way as a
    multi-qubit gate with no parameters — so pyquil represents it with the
    same ``Gate`` class. This helper centralizes that fact: when a block
    matches a signature, the matcher emits one of these in place of the
    block's instructions.

    Args:
        name: The DefCircuit name (used as the gate name in the emitted
            instruction), e.g. ``"CZ_3"`` or ``"MEASURE_ANCILLA"``.
        qubits: Concrete physical qubit indices, in the order declared by
            the DefCircuit's formal parameter list. These are wrapped in
            ``Qubit`` objects to form the gate's qubit arguments.

    Returns:
        A ``Gate`` whose ``out()`` renders as ``"<name> q0 q1 ..."``,
        suitable for direct insertion into a ``Program``'s instruction
        stream.
    """
    try:
        from pyquil.quilatom import Qubit
        from pyquil.quilbase import Gate
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    return Gate(name=name, params=[], qubits=[Qubit(q) for q in qubits])


def _set_template_object(instr: Pulse | Capture, obj: Any) -> None:  # type: ignore  # noqa: F821
    """Write a template object back to a ``Pulse`` or ``Capture`` instruction.

    Args:
        instr (Pulse | Capture): a Quil-T instruction that carries a
            waveform-like template object.
        obj (Any): the updated waveform-like object to assign.

    Raises:
        TypeError: if ``instr`` is not a supported instruction type.
    """
    try:
        from pyquil.quilbase import Capture, Pulse
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    if isinstance(instr, Pulse):
        instr.waveform = obj
    elif isinstance(instr, Capture):
        instr.kernel = obj
    else:
        raise TypeError(f"Unsupported instruction type: {type(instr)!r}")


def _qubit_index(q: Qubit | FormalArgument | int) -> int:  # type: ignore  # noqa: F821
    """Coerce a PyQuil qubit reference to its integer index.

    Normalizes the several representations a qubit can take in a parsed PyQuil
    program so the rest of the matcher can compare qubits with ``==`` and use
    them as dict / set keys. The three accepted inputs are:

    - ``Qubit(n)``: a concrete physical qubit, as it appears in user-written
      instructions like ``CZ 12 13``. Returns ``n``.
    - ``FormalArgument("qN")``: a formal parameter from a ``DEFCIRCUIT`` body
      (e.g. ``CZ q12 q13``). Returns ``N``. This relies on the convention
      that formal names have the form ``q<int>`` and that the integer denotes
      the physical qubit the calibration is written for, which holds for
      every DefCircuit shipped with the calibration files. Formals that don't
      match this pattern raise ``ValueError``, since a truly parametric
      template would need a different matching strategy (binding formals to
      concretes) than this matcher implements.
    - ``int``: returned unchanged, as a defensive passthrough for callers
      that have already unwrapped a qubit.

    Without this normalization, ``Qubit(13)`` from the user's program and
    ``FormalArgument("q13")`` from a DefCircuit body would hash and compare
    as distinct objects, and the multiset-based block matching would never
    find a match.

    Args:
        q (Qubit | FormalArgument | int): a qubit reference of type ``Qubit``,
            ``FormalArgument``, or ``int``.

    Returns:
        int: the integer index of the qubit.

    Raises:
        ValueError: if ``q`` is a ``FormalArgument`` whose name is not of
            the form ``q<int>``.
        TypeError: if ``q`` is not one of the three accepted types.
    """
    try:
        from pyquil.quilatom import FormalArgument, Qubit
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    if isinstance(q, Qubit):
        return q.index
    elif isinstance(q, FormalArgument):
        if q.name.startswith("q") and q.name[1:].isdigit():
            return int(q.name[1:])
        raise ValueError(
            f"Formal argument {q.name!r} is not of the form 'qN'; this matcher "
            "assumes DefCircuit formals correspond to physical qubits by name."
        )
    elif isinstance(q, int):
        return q
    else:
        raise TypeError(f"Unrecognized qubit object: {type(q).__name__}")


def add_active_reset(
    program:    Program,  # type: ignore  # noqa: F821
    n_resets:   int = 1,
    esp:        bool = False,
    esp_qubits: Sequence[int] | None = None,
) -> Program:  # type: ignore  # noqa: F821
    """Prepend active resets to a PyQuil program using Quil jump statements.

    For each reset cycle, every qubit in the program is measured into a
    dedicated ``ro_reset`` register. A ``JUMP-UNLESS`` branch skips the
    correction pulses when the qubit is already in |0⟩. Successive reset
    cycles are separated by global FENCEs across all program qubits; a final
    global FENCE follows the last reset before the main program body.

    The correction sequence applied when a qubit is measured as |1⟩ depends
    on whether excited state promotion (ESP) is enabled for that qubit:

    - If ``esp=True`` and the qubit is in ``esp_qubits``: two
      ``RX_F12(π/2)`` pulses (EF subspace) then two ``RX(π/2)`` pulses
      (GE subspace).
    - Otherwise: two ``RX(π/2)`` pulses (GE subspace) then two
      ``RX_F12(π/2)`` pulses (EF subspace).

    Args:
        program (Program): the program to prepend active reset to.
        n_resets (int, optional): number of active reset cycles to perform.
            Defaults to 1.
        esp (bool, optional): whether to use excited state promotion for
            qubits listed in ``esp_qubits``. Defaults to ``False``.
        esp_qubits (Sequence[int] | None, optional): qubit indices that
            receive the ESP-ordered correction sequence. Only relevant when
            ``esp=True``. Defaults to ``None``.

    Returns:
        Program: a new program with active reset cycles prepended before the
        original instructions.
    """
    try:
        from pyquil import Program
        from pyquil.gates import DELAY, FENCE, MEASURE
        from pyquil.quilatom import MemoryReference, Qubit
        from pyquil.quilbase import Jump, JumpTarget, JumpUnless, Label
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    all_qubits = sorted(program.get_qubit_indices())
    esp_set = set(esp_qubits) if esp_qubits is not None else set()

    new_program: Program = program.copy_everything_except_instructions()
    for r in range(n_resets):
        new_program += FENCE(*all_qubits)

        for q in all_qubits:
            label_false = Label(f"ar_{r}_{q}_false")
            label_done  = Label(f"ar_{r}_{q}_done")
            cref = MemoryReference(f"ro{q}", 0)
            qubit_obj = Qubit(q)

            new_program += MEASURE(q, cref)
            new_program += JumpUnless(label_false, cref)
            # $when_true: qubit measured |1⟩, apply correction pulses
            if esp and q in esp_set:
                new_program += add_X90(qubit_obj, **{'subspace': 'EF'})
                new_program += add_X90(qubit_obj, **{'subspace': 'EF'})
                new_program += add_X90(qubit_obj, **{'subspace': 'GE'})
                new_program += add_X90(qubit_obj, **{'subspace': 'GE'})
            else:
                new_program += add_X90(qubit_obj, **{'subspace': 'GE'})
                new_program += add_X90(qubit_obj, **{'subspace': 'GE'})
                new_program += add_X90(qubit_obj, **{'subspace': 'EF'})
                new_program += add_X90(qubit_obj, **{'subspace': 'EF'})
            new_program += Jump(label_done)
            # $when_false: qubit measured |0⟩, nothing to do
            new_program += JumpTarget(label_false)
            new_program += JumpTarget(label_done)

    new_program += FENCE(*all_qubits)

    for instr in program.instructions:
        new_program += instr

    return new_program


def add_dd_sequence_during_operation(
    program:    Program,  # type: ignore  # noqa: F821
    operation:  Gate | Measurement,  # type: ignore  # noqa: F821
    dd_qubits:  Sequence[int],
    total_time: float,
    gate_time:  float,
    n_pulses:   int | None = None,
    dd_method:  str = 'XX_N',
    subspace:   str = 'GE',
) -> Program:  # type: ignore  # noqa: F821
    """Add a dynamical decoupling sequence concurrent with a matching operation.

    Walks the program's instruction stream and, alongside every instruction
    that matches ``operation``, inserts a dynamical decoupling (DD) sequence on
    ``dd_qubits``. Because this function assumes the program uses fences to
    delimit parallel cycles, the DD instructions are placed in the same fenced
    block as the matched operation and therefore execute concurrently with it.

    Matching rules:

    - ``Gate``: name and qubit indices must both match, in order.
    - ``Measurement``: qubit index must match (classical register is ignored).

    Basic example usage:

    ```python
    from pyquil import Program
    from pyquil.gates import CZ, H

    from qcal.interface.pyquil.utils import add_dd_sequence_during_operation
    from qcal.units import ns

    # Program with a CZ on qubits 0 and 1; qubit 2 is idle during the CZ.
    p = Program()
    p += FENCE(0, 1, 2)
    p += CZ(0, 1)
    p += FENCE(0, 1, 2)
    p += H(2)

    # Apply an XX_N DD sequence on qubit 2 while CZ(0, 1) executes.
    p = add_dd_sequence_during_operation(
        program=p,
        operation=CZ(0, 1),
        dd_qubits=[2],
        total_time=100 * ns,
        gate_time=20 * ns,
        dd_method='XX_N',
    )
    print(p)
    # DECLARE ro BIT[1]
    # FENCE 0 1 2
    # CZ 0 1
    # DELAY 2 4e-9
    # RX(1.5707963267948966) 2
    # RX(1.5707963267948966) 2
    # DELAY 2 1.2000000000000002e-8
    # RX(1.5707963267948966) 2
    # RX(1.5707963267948966) 2
    # DELAY 2 4e-9
    # FENCE 2
    # FENCE 0 1 2
    # H 2
    ```

    Args:
        program (Program): the program to update.
        operation (Gate | Measurement): the PyQuil instruction to match against.
        dd_qubits (Sequence[int]): qubit indices on which the DD sequence
            should be applied concurrently with ``operation``.
        total_time (float): total time over which to perform the DD.
        gate_time (float): time of the native X90 gate.
        n_pulses (int | None, optional): number of pulses. Defaults to ``None``.
            If ``None``, the number of pulses is set to the maximum number of
            pulses that can fit within the total time, with a minimum that
            depends on the DD method.
        dd_method (str, optional): the DD method to use. Defaults to 'XX_N'.
        subspace (str, optional): qubits subspace for the DD sequence. Defaults
            'GE'.

    Returns:
        Program: a new program with DD sequences inserted alongside each
            matching instruction.
    """
    try:
        from pyquil import Program
        from pyquil.quilbase import Gate, Measurement
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    transpiler = PyQuilTranspiler(fence_between_cycles=False)

    if isinstance(operation, Gate):
        target_qubits = tuple(_qubit_index(q) for q in operation.qubits)

        def _matches(instr: object) -> bool:
            return (
                isinstance(instr, Gate)
                and instr.name == operation.name
                and tuple(
                    _qubit_index(q) for q in instr.qubits
                ) == target_qubits
            )
    elif isinstance(operation, Measurement):
        target_qubit = _qubit_index(operation.qubit)

        def _matches(instr: object) -> bool:
            return (
                isinstance(instr, Measurement)
                and _qubit_index(instr.qubit) == target_qubit
            )
    else:
        raise TypeError(
            f"operation must be a Gate or Measurement, got {type(operation)!r}"
        )

    new_program: Program = program.copy_everything_except_instructions()
    for instr in program.instructions:
        new_program += instr

        if _matches(instr):
            dd_circuit = DD_SEQUENCES[dd_method](
                qubits=dd_qubits,
                total_time=total_time,
                gate_time=gate_time,
                n_pulses=n_pulses,
                subspace=subspace
            )
            dd_program = transpiler.transpile(dd_circuit)[0]
            new_program += dd_program

    return new_program


def add_delay_after_measurements(
        program: Program, delay: float  # type: ignore  # noqa: F821
) -> Program:  # type: ignore  # noqa: F821
    """Add a delay after each measurement in a PyQuil program.

    Args:
        program (Program): the program to update.
        delay (float): the delay time in seconds.

    Returns:
        pyquil.Program: the updated program.
    """
    try:
        from pyquil import Program
        from pyquil.gates import DELAY
        from pyquil.quilbase import Measurement
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    new_program: Program = program.copy_everything_except_instructions()
    for instr in program.instructions:
        new_program += instr
        if isinstance(instr, Measurement):
            new_program += DELAY(instr.qubit, delay)
    return new_program


def get_defcal_param(
    defcal: DefCalibration | DefMeasureCalibration,  # type: ignore  # noqa: F821
    param: str,
    *,
    instr_idx: int | None = None,
) -> Any:
    """Get a template parameter from a calibration instruction.

    This function searches the instructions in a ``DefCalibration`` or
    ``DefMeasureCalibration`` for the first supported instruction containing
    a matching parameter and returns the parameter value.

    Supported instruction/template pairs are:
        - ``Pulse.waveform``
        - ``Capture.kernel``

    Args:
        defcal (DefCalibration | DefMeasureCalibration): the calibration object
            to inspect.
        param (str): the name of the template parameter to retrieve.
        instr_idx (int | None, optional): optional specific instruction index
            to inspect. If not provided, all instructions are searched in order.

    Returns:
        Any: the retrieved template parameter value.

    Raises:
        IndexError: if ``instr_idx`` is out of range.
        ValueError: if no supported instruction contains the requested
            parameter.
    """
    try:
        from pyquil.quilbase import Capture, Pulse
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    instrs = list(defcal.instructions)
    indices = [instr_idx] if instr_idx is not None else range(len(instrs))

    for i in indices:
        instr = instrs[i]
        if not isinstance(instr, (Pulse, Capture)):
            continue

        tmpl = _get_template_object(instr)
        if tmpl is None:
            continue

        try:
            return getattr(tmpl, param)
        except Exception:
            continue

    raise ValueError(
        f"Could not find parameter {param!r} in any supported "
        f"instruction in calibration {getattr(defcal, 'name', '<unnamed>')!r}."
    )


def prepend_delay_to_program(program: Program, delay: float) -> Program:  # type: ignore  # noqa: F821
    """Prepend a delay to a PyQuil program.

    Args:
        program (Program): the program to update.
        delay (float): the delay time in seconds.

    Returns:
        Program: the updated program.
    """
    try:
        from pyquil.gates import DELAY
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    qubits = program.get_qubit_indices()
    # p = Program()
    # p += Delay(*qubits, delay)
    return program.prepend_instructions([DELAY(q, delay) for q in qubits])


def prepend_esp_to_measure(
    program:              Program,  # type: ignore  # noqa: F821
    qubits:               Sequence[int] | None = None,
    fence_between_cycles: bool = False,
) -> Program:  # type: ignore  # noqa: F821
    """Prepend an EF Pi pulse (ESP) to each measurement in a PyQuil program.

    When ``fence_between_cycles=True`` (default), the function operates on
    fenced cycles: for each block of instructions between a pair of FENCE
    instructions that contains measurements, a single EF pi cycle is inserted
    immediately before the original measurement cycle. The EF pi cycle is
    bounded on each side by a global FENCE across all program qubits, so
    qubits that do not receive an EF pi pulse simply idle for the duration.
    Measurements appearing outside a fenced block are handled with the same
    global FENCE delimiters.

    When ``fence_between_cycles=False``, the function iterates over
    instructions directly and inserts two EF X90 pulses immediately before
    each matching measurement, without any surrounding FENCE instructions.

    Args:
        program (Program): the program to update.
        qubits (Sequence[int] | None, optional): qubit indices to which the EF
            pi pulse should be applied. When ``None``, the pulse is prepended
            before every measurement. Defaults to ``None``.
        fence_between_cycles (bool, optional): if ``True``, groups instructions
            by fenced cycle before inserting EF pi pulses, surrounding the EF
            pi cycle with global FENCEs. If ``False``, inserts EF pi pulses
            inline before each matching measurement with no FENCEs added.
            Defaults to ``False``.

    Returns
        Program: the updated program.
    """
    try:
        from pyquil import Program
        from pyquil.gates import FENCE
        from pyquil.quilbase import Fence, Measurement
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    all_qubits = program.get_qubit_indices()
    new_program: Program = program.copy_everything_except_instructions()

    if not fence_between_cycles:
        for instr in program.instructions:
            if isinstance(instr, Measurement):
                if qubits is None or _qubit_index(instr.qubit) in qubits:
                    new_program += add_X90(instr.qubit, **{'subspace': 'EF'})
                    new_program += add_X90(instr.qubit, **{'subspace': 'EF'})
            new_program += instr
        return new_program

    instrs = list(program.instructions)
    i = 0
    while i < len(instrs):
        instr = instrs[i]

        if isinstance(instr, Fence):
            opening_fence = instr
            i += 1

            block: List = []
            while i < len(instrs) and not isinstance(instrs[i], Fence):
                block.append(instrs[i])
                i += 1

            closing_fence = instrs[i] if i < len(instrs) else None
            if i < len(instrs):
                i += 1

            measure_instrs = [b for b in block if isinstance(b, Measurement)]
            if measure_instrs:
                new_program += FENCE(*all_qubits)
                for meas in measure_instrs:
                    if qubits is None or _qubit_index(meas.qubit) in qubits:
                        new_program += add_X90(meas.qubit, **{'subspace': 'EF'})
                        new_program += add_X90(meas.qubit, **{'subspace': 'EF'})
                new_program += FENCE(*all_qubits)

            new_program += opening_fence
            for b in block:
                new_program += b
            if closing_fence is not None:
                new_program += closing_fence

        else:
            if isinstance(instr, Measurement):
                # Collect all consecutive unfenced measurements as a virtual
                # cycle so they share a single EF pi cycle.
                meas_run: List = []
                while i < len(instrs) and isinstance(instrs[i], Measurement):
                    meas_run.append(instrs[i])
                    i += 1
                esp_meas = [
                    m for m in meas_run
                    if qubits is None or _qubit_index(m.qubit) in qubits
                ]
                if esp_meas:
                    new_program += FENCE(*all_qubits)
                    for meas in esp_meas:
                        new_program += add_X90(meas.qubit, **{'subspace': 'EF'})
                        new_program += add_X90(meas.qubit, **{'subspace': 'EF'})
                    new_program += FENCE(*all_qubits)
                for m in meas_run:
                    new_program += m
            else:
                new_program += instr
                i += 1

    return new_program


def prepend_measure_to_program(
    program: Program,  # type: ignore  # noqa: F821
    delay:   float = 2e-6,
) -> Program:  # type: ignore  # noqa: F821
    """Prepend a measure to a PyQuil program.

    Args:
        program (Program): the program to update.
        delay (float, optional): delay in seconds to insert after each
            measurement. Defaults to 2 µs.

    Returns:
        Program: the updated program.
    """
    try:
        from pyquil.gates import DELAY, MEASURE
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    qubits = sorted(program.get_qubit_indices())
    instrs = []
    for q in qubits:
        instrs.append(MEASURE(q, (f'ro{q}', 0)))
        instrs.append(DELAY(q, delay))
    return program.prepend_instructions(instrs)


def replace_parallel_cycles(
    program:     Program, # type: ignore  # noqa: F821
    defcircuits: Optional[Iterable[DefCircuit]] = None, # type: ignore  # noqa: F821
    *,
    symmetric_gates: FrozenSet[str] = DEFAULT_SYMMETRIC_GATES
) -> Program: # type: ignore  # noqa: F821
    """Rewrite parallel-cycle blocks in a program as ``DefCircuit`` invocations.

    Walks the program's instruction stream, partitioning it into blocks of
    consecutive gate-level instructions (``Gate`` and ``Measurement``)
    separated by boundary instructions (``Fence``, ``Pragma``, ``Delay``,
    ``Reset``, frame ops, ...). Each block whose contents match a known
    ``DefCircuit`` signature — the active multiset must equal the
    ``DefCircuit``'s, and any explicit ``I q`` lines must be on qubits the
    ``DefCircuit`` also marks idle — is collapsed into a single invocation of
    that ``DefCircuit``. Boundary instructions and non-matching blocks are
    preserved verbatim.

    Signatures come from two sources, merged in this order:

    1. Every ``DefCircuit`` already attached to ``program`` (the program is
       assumed to have been parsed alongside its calibration file, so
       ``program.circuits`` is already populated).
    2. Any additional ``DefCircuit`` objects passed via ``defcircuits``.

    When a passed-in ``DefCircuit`` shares a name with one on the program, the
    passed-in version wins — both for matching and in the returned program.
    This makes it convenient to swap in a refreshed or locally-modified
    ``DefCircuit`` without first stripping the old one.

    The returned program preserves all of the original program's metadata
    (frames, waveforms, ``DefCal``s, measure calibrations, ``DefCircuit``s, ...)
    and additionally includes every ``DefCircuit`` passed via ``defcircuits``,
    so the result is self-contained and can be handed straight back to the
    compiler.

    Args:
        program (Program): the PyQuil ``Program`` whose instruction stream will
            be rewritten. Its calibrations, frames, waveforms, and
            ``DefCircuit``s are carried through to the result.
        defcircuits (Iterable[DefCircuit], optional): optional extra
            ``DefCircuit`` objects to consider for matching, in addition to
            those already on the program. Defaults to ``None``. When ``None``
            (the default), only the program's own ``DefCircuit``s are used.
            Names that collide with existing program ``DefCircuit``s override
            the program's version.
        symmetric_gates: Names of two-qubit gates whose argument order is
            irrelevant for matching. Forwarded to
            :func:`signature_from_defcircuit`. Defaults to
            :data:`DEFAULT_SYMMETRIC_GATES`.

    Returns:
        Program: a new ``Program`` with parallel-cycle blocks replaced by
        ``DefCircuit`` invocations, with all original metadata and any merged-in
        ``DefCircuit``s attached.

    Raises:
        ValueError: if two DefCircuits in the merged set share the same
            active-instruction multiset (the matcher cannot disambiguate
            them), or if any DefCircuit body is malformed; see
            :func:`signature_from_defcircuit`.
    """
    try:
        from pyquil.quilbase import AbstractInstruction, DefCircuit
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    def flush() -> None:
        if not block:
            return
        sig = by_active.get(frozenset(Counter(block_active).items()))
        if sig is not None and set(block_idles).issubset(sig.idle_qubits):
            new_instructions.append(
                _make_invocation(sig.name, sig.formal_qubits)
            )
        else:
            new_instructions.extend(block)
        block.clear()
        block_active.clear()
        block_idles.clear()

    # Merge DefCircuits: program's own first, then passed-in (which wins on
    # collision).
    program_circuits: Dict[str, DefCircuit] = {
        instr.name: instr
        for instr in program.instructions
        if isinstance(instr, DefCircuit)
    }
    extra_circuits: Dict[str, DefCircuit] = {}
    if defcircuits is not None:
        for dc in defcircuits:
            extra_circuits[dc.name] = dc
    all_circuits = {**program_circuits, **extra_circuits}

    signatures = [
        signature_from_defcircuit(dc, symmetric_gates=symmetric_gates)
        for dc in all_circuits.values()
    ]

    defined_calibrations = {
        (cal.name, tuple(_qubit_index(q) for q in cal.qubits))
        for cal in (
            list(program.calibrations) + list(program.measure_calibrations)
        )
    }
    for sig in signatures:
        if (sig.name, sig.formal_qubits) not in defined_calibrations:
            logger.warning(
                " DefCircuit %r on qubits %s has no matching DefCal in the "
                "program; the rewritten call will be unresolvable at compile "
                "time.",
                sig.name,
                sig.formal_qubits,
            )

    by_active: Dict[FrozenSet, CycleSignature] = {}
    for sig in signatures:
        if sig.active_multiset in by_active:
            raise ValueError(
                f"DefCircuits {sig.name!r} and "
                f"{by_active[sig.active_multiset].name!r} "
                "share the same active-instruction multiset; matcher cannot "
                "disambiguate."
            )
        by_active[sig.active_multiset] = sig

    new_instructions: List[AbstractInstruction] = []
    block: List[AbstractInstruction] = []
    block_active: List[Tuple] = []
    block_idles: List[int] = []

    for instr in program.instructions:
        kind, payload = _classify(instr, symmetric_gates)
        if kind == "boundary":
            flush()
            new_instructions.append(instr)
        else:
            block.append(instr)
            if kind == "active":
                block_active.append(payload)
            else:
                block_idles.append(payload)
    flush()

    # Preserve all original metadata (frames, waveforms, calibrations, etc.).
    new_program = program.copy_everything_except_instructions()

    # Apply passed-in DefCircuit overrides on top of what was copied.
    for dc in extra_circuits.values():
        new_program.inst(dc)

    for instr in new_instructions:
        new_program += instr

    return new_program


def set_defcal_param(
    defcal: DefCalibration | DefMeasureCalibration,  # type: ignore  # noqa: F821
    param:  str,
    value:  Any,
    *,
    instr_idx: int | None = None,
) -> DefCalibration | DefMeasureCalibration:  # type: ignore  # noqa: F821
    """Set a template parameter in a calibration instruction.

    This function searches the instructions in a ``DefCalibration`` or
    ``DefMeasureCalibration`` for the first supported instruction containing
    a matching parameter and updates that parameter in place.

    Supported instruction/template pairs are:
        - ``Pulse.waveform``
        - ``Capture.kernel``

    Args:
        defcal (DefCalibration | DefMeasureCalibration): the calibration object
            to modify.
        param (str): the name of the template parameter to update.
        value (Any): the new value to assign to the parameter.
        instr_idx (int | None, optional): optional specific instruction
            index to inspect. If not provided, all instructions are searched in
            order.

    Returns:
        DefCalibration | DefMeasureCalibration: the mutated calibration object.

    Raises:
        IndexError: if ``instr_idx`` is out of range.
        ValueError: if no supported instruction contains the requested
            parameter.
    """
    try:
        from pyquil.quilbase import Capture, Pulse
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    instrs = list(defcal.instructions)
    indices = [instr_idx] if instr_idx is not None else range(len(instrs))

    for i in indices:
        instr = instrs[i]
        if not isinstance(instr, (Pulse, Capture)):
            continue

        tmpl = _get_template_object(instr)
        if tmpl is None:
            continue

        if not hasattr(tmpl, param):
            continue

        setattr(tmpl, param, value)
        _set_template_object(instr, tmpl)
        instrs[i] = instr
        defcal.instructions = instrs
        return defcal

    raise ValueError(
        f"Could not find parameter {param!r} in any supported "
        f"instruction in calibration {getattr(defcal, 'name', '<unnamed>')!r}."
    )


def set_defcal_param_in_program(
    program: Program,  # type: ignore  # noqa: F821
    defcal:  DefCalibration | DefMeasureCalibration,  # type: ignore  # noqa: F821
    param:   str,
    value:   Any,
    *,
    instr_idx: int | None = None,
) -> DefCalibration | DefMeasureCalibration:  # type: ignore  # noqa: F821
    """Set a calibration template parameter and write it back to a program.

    This function reads a calibration snapshot from ``program.calibrations``,
    updates a matching parameter, and re-inserts the calibration into the
    program with ``program.inst(...)``.

    Args:
        program (Program): the program containing the calibration.
        defcal (DefCalibration | DefMeasureCalibration): the calibration to
            update.
        param (str): the name of the template parameter to update.
        value (Any): the new value to assign to the parameter.
        instr_idx (int | None, optional): optional specific instruction
            index to inspect. If not provided, all instructions are searched in
            order.

    Returns:
        DefCalibration | DefMeasureCalibration: the updated calibration object
            that was written back to the program.

    Raises:
        IndexError: if ``instr_idx`` is provided and out of range.
        ValueError: if no supported instruction contains the requested
            parameter.
    """
    updated_cal = set_defcal_param(
        defcal,
        param,
        value,
        instr_idx=instr_idx,
    )
    program.inst(updated_cal)

    return updated_cal


def set_waveform(name: str, array: Any) -> DefWaveform:  # type: ignore  # noqa: F821
    """Create a new ``DefWaveform`` from a name and sample array.

    Args:
        name (str): the waveform name.
        array (Any): the waveform samples.

    Returns:
        DefWaveform: a new ``DefWaveform`` instance.
    """
    try:
        from pyquil.quilbase import DefWaveform
    except ImportError as exc:
        raise ImportError("Unable to import PyQuil!") from exc

    return DefWaveform(name, [], list(array))


def set_waveform_in_program(
    program: Program,  # type: ignore  # noqa: F821
    name:    str,
    array:   Any,
) -> DefWaveform:  # type: ignore  # noqa: F821
    """Create a ``DefWaveform`` and insert it into a PyQuil program.

    This rewrites the waveform definition in the program if a waveform with the
    same name already exists.

    Args:
        program (Program): the program to update.
        name (str): the waveform name.
        array (Any): the waveform samples.

    Returns:
        DefWaveform: the newly created ``DefWaveform`` that was inserted into
            the program.
    """
    waveform = set_waveform(name, array)
    program.inst(waveform)
    return waveform


def signature_from_defcircuit(
    defcircuit: DefCircuit, # type: ignore  # noqa: F821
    *,
    symmetric_gates: FrozenSet[str] = DEFAULT_SYMMETRIC_GATES,
) -> CycleSignature:
    """Build a matchable ``CycleSignature`` from a ``DefCircuit``.

    Walks the DefCircuit body once, classifying each instruction with
    :func:`_classify` and partitioning it into two structures:

    - The active multiset (gates and measurements that a candidate block
      must reproduce exactly), stored as a ``frozenset`` of
      ``(key, count)`` pairs so the signature can serve as a dict key for
      O(1) lookup during matching.
    - The set of idle qubits declared via ``I q`` lines, against which a
      candidate block's explicit idles are checked as a subset (manual
      programs are not required to write these out, but if they do, the
      qubits must be ones the DefCircuit also marks idle).

    The DefCircuit's formal parameter list is captured in declaration order
    and used to construct the invocation when a block matches: a DefCircuit
    declared ``DEFCIRCUIT CZ_3 q1 q2 q7 ...`` produces calls of the form
    ``CZ_3 1 2 7 ...``. This relies on the formals-as-physical-qubits
    convention enforced by :func:`_qubit_index`.

    Args:
        defcircuit: The ``DefCircuit`` to summarize. Must have no classical
            parameters (parametric DefCircuits are out of scope for this
            matcher) and a non-empty body of gate-level instructions.
        symmetric_gates: Names of two-qubit gates whose argument order is
            irrelevant when building active keys. Forwarded to
            :func:`_classify`. Defaults to :data:`DEFAULT_SYMMETRIC_GATES`.

    Returns:
        A ``CycleSignature`` containing the DefCircuit's name, formal qubit
        tuple, active-instruction multiset, and idle-qubit set.

    Raises:
        ValueError: If the DefCircuit declares classical parameters, has an
            empty active body, or contains a body instruction that is
            neither a ``Gate`` nor a ``Measurement`` (e.g. a stray
            ``Pragma`` or frame operation, which would indicate the
            DefCircuit is doing something the matcher isn't designed for).
    """
    if getattr(defcircuit, "parameters", None):
        raise ValueError(
            f"DefCircuit {defcircuit.name!r} has classical parameters."
        )

    formals = tuple(_qubit_index(q) for q in defcircuit.qubit_variables)
    active_keys: List[Tuple] = []
    idle_qubits: List[int] = []
    for body_instr in defcircuit.instructions:
        kind, payload = _classify(body_instr, symmetric_gates)
        if kind == "active":
            active_keys.append(payload)
        elif kind == "idle":
            idle_qubits.append(payload)
        else:
            raise ValueError(
                f"DefCircuit {defcircuit.name!r} body contains "
                f"{type(body_instr).__name__}; only Gate and Measurement allowed."
            )

    if not active_keys:
        raise ValueError(
            f"DefCircuit {defcircuit.name!r} has no active instructions."
        )

    return CycleSignature(
        name=defcircuit.name,
        formal_qubits=formals,
        active_multiset=frozenset(Counter(active_keys).items()),
        idle_qubits=frozenset(idle_qubits),
    )
