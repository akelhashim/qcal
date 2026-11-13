"""Helper functions for calibration scripts."""
import numpy as np
from numpy.typing import NDArray

from qcal.config import Config


def find_pulse_index(config: Config, param: str):
    """Find the first index of a physical pulse of a gate in a composite pulse.

    This function will ignore any pre-pulses that are listed as string objects
    which reference other pulses defined in the config.

    Args:
        config (Config): qcal Config object.
        param (str): config path to the list of pulses of a gate (e.g.
            ``'single_qubit/0/GE/X90/pulse'``).

    Returns:
        int: index of the first pulse that is not a string reference or a
            ``virtualz`` pulse, or None if there is no such pulse.
    """
    for i, pulse in enumerate(config[param]):
        if isinstance(pulse, str):
            continue
        else:
            if pulse['env'] != 'virtualz':
                return i
            else:
                continue


def in_range(number: int | float, array: NDArray) -> bool:
    """Check whether a number is within the range defined by an array.

    Args:
        number (int | float): number to check within a range.
        array (NDArray): array whose max and min values define the range.

    Returns:
        bool: whether the number is within the range.
    """
    min, max = np.array(array).min(), np.array(array).max()
    is_in_range = (min <= number) & (number <= max)

    return is_in_range
