"""Helper functions for PyQuil interface.

"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _get_template_object(instr: Any) -> Any | None:
    """Return the template object carried by a Pulse or Capture instruction.

    For a ``Pulse``, this returns ``instr.waveform``. For a ``Capture``, this
    returns ``instr.kernel``.

    Args:
        instr (Any): A Quil-T instruction that may carry a waveform-like
            template object.

    Returns:
        Any | None: The waveform-like object attached to the instruction,
        or ``None`` if the instruction type is unsupported.
    """
    try:
        from pyquil.quilbase import Capture, Pulse
    except ImportError as exc:
        raise ImportError("Unable to import pyquil!") from exc

    if isinstance(instr, Pulse):
        return instr.waveform
    if isinstance(instr, Capture):
        return instr.kernel
    return None


def _set_template_object(instr: Any, obj: Any) -> None:
    """Write a template object back to a Pulse or Capture instruction.

    Args:
        instr (Any): A Quil-T instruction that carries a waveform-like template
            object.
        obj (Any): The updated waveform-like object to assign.

    Raises:
        TypeError: If ``instr`` is not a supported instruction type.
    """
    try:
        from pyquil.quilbase import Capture, Pulse
    except ImportError as exc:
        raise ImportError("Unable to import pyquil!") from exc

    if isinstance(instr, Pulse):
        instr.waveform = obj
    elif isinstance(instr, Capture):
        instr.kernel = obj
    else:
        raise TypeError(f"Unsupported instruction type: {type(instr)!r}")


def get_defcal_param(
    defcal: Any,
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
        defcal (pyquil.quilbase.DefCalibration): The calibration object to
            inspect.
        param (str): The name of the template parameter to retrieve.
        instr_idx (int | None, optional): Optional specific instruction index
            to inspect. If not provided, all instructions are searched in order.

    Returns:
        Any: The retrieved template parameter value.

    Raises:
        IndexError: If ``instr_idx`` is out of range.
        ValueError: If no supported instruction contains the requested
            parameter.
    """
    try:
        from pyquil.quilbase import Capture, Pulse
    except ImportError as exc:
        raise ImportError("Unable to import pyquil!") from exc

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
            # return tmpl.get_parameter(param)
            return getattr(tmpl, param)
        except Exception:
            continue

    raise ValueError(
        f"Could not find parameter {param!r} in any supported "
        f"instruction in calibration {getattr(defcal, 'name', '<unnamed>')!r}."
    )


def set_defcal_param(
    defcal: Any,
    param:  str,
    value:  Any,
    *,
    instr_idx: int | None = None,
) -> Any:
    """Set a template parameter in a calibration instruction.

    This function searches the instructions in a ``DefCalibration`` or
    ``DefMeasureCalibration`` for the first supported instruction containing
    a matching parameter and updates that parameter in place.

    Supported instruction/template pairs are:
        - ``Pulse.waveform``
        - ``Capture.kernel``

    Args:
        defcal (pyquil.quilbase.DefCalibration): The calibration object to
            modify.
        param (str): The name of the template parameter to update.
        value (Any): The new value to assign to the parameter.
        instr_idx (int | None, optional): Optional specific instruction
            index to inspect. If not provided, all instructions are searched in
            order.

    Returns:
        pyquil.quilbase.DefCalibration: The mutated calibration object.

    Raises:
        IndexError: If ``instr_idx`` is out of range.
        ValueError: If no supported instruction contains the requested
            parameter.
    """
    try:
        from pyquil.quilbase import Capture, Pulse
    except ImportError as exc:
        raise ImportError("Unable to import pyquil!") from exc

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
            tmpl.get_parameter(param)
        except Exception:
            continue

        tmpl.set_parameter(param, value)
        _set_template_object(instr, tmpl)
        instrs[i] = instr
        defcal.instructions = instrs
        return defcal

    raise ValueError(
        f"Could not find parameter {param!r} in any supported "
        f"instruction in calibration {getattr(defcal, 'name', '<unnamed>')!r}."
    )


def set_defcal_param_in_program(
    program: Any,
    defcal:  Any,
    param:   str,
    value:   Any,
    *,
    instr_idx: int | None = None,
) -> Any:
    """Set a calibration template parameter and write it back to a program.

    This function reads a calibration snapshot from ``program.calibrations``,
    updates a matching parameter, and re-inserts the calibration into the
    program with ``program.inst(...)``.

    Args:
        program (pyquil.quil.Program): The program containing the calibration.
        defcal (pyquil.quilbase.DefCalibration): The calibration to update.
        param (str): The name of the template parameter to update.
        value (Any): The new value to assign to the parameter.
        instr_idx (int | None, optional): Optional specific instruction
            index to inspect. If not provided, all instructions are searched in
            order.

    Returns:
        pyquil.quilbase.DefCalibration: The updated calibration object that
            was written back to the program.

    Raises:
        IndexError: If ``instr_idx`` is provided and out of range.
        ValueError: If no supported instruction contains the requested
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


def set_waveform(name: str, array: Any) -> Any:
    """Create a new ``DefWaveform`` from a name and sample array.

    Args:
        name (str): The waveform name.
        array (Any): The waveform samples.

    Returns:
        Any: A new ``DefWaveform`` instance.
    """
    try:
        from pyquil.quilbase import DefWaveform
    except ImportError as exc:
        raise ImportError("Unable to import pyquil!") from exc

    return DefWaveform(name, [], list(array))


def set_waveform_in_program(
    program: Any,
    name:    str,
    array:   Any,
) -> Any:
    """Create a ``DefWaveform`` and insert it into a pyQuil program.

    This rewrites the waveform definition in the program if a waveform with the
    same name already exists.

    Args:
        program (Any): The program to update.
        name (str): The waveform name.
        array (Any): The waveform samples.

    Returns:
        Any: The newly created ``DefWaveform`` that was inserted into the
            program.
    """
    waveform = set_waveform(name, [], array)
    program.inst(waveform)
    return waveform


def prepend_delay_to_program(program: Any, delay: float) -> Any:
    """Prepend a delay to a pyQuil program.

    Args:
        program (pyquil.Program): The program to update.
        delay (float): The delay time in seconds.

    Returns:
        pyquil.Program: The updated program.
    """
    try:
        from pyquil.gates import DELAY
    except ImportError as exc:
        raise ImportError("Unable to import pyquil!") from exc

    qubits = program.get_qubit_indices()
    # p = Program()
    # p += Delay(*qubits, delay)
    return program.prepend_instructions([DELAY(*qubits, delay)])


def prepend_measure_to_program(program: Any) -> Any:
    """Prepend a measure to a pyQuil program.

    Args:
        program (pyquil.Program): The program to update.

    Returns:
        pyquil.Program: The updated program.
    """
    try:
        from pyquil.gates import MEASURE
    except ImportError as exc:
        raise ImportError("Unable to import pyquil!") from exc

    qubits = program.get_qubit_indices()
    measure = [MEASURE(q, ('ro', i)) for i, q in enumerate(sorted(qubits))]
    return program.prepend_instructions(measure)
