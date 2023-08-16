"""Helper functions for calibration scripts.

"""
from qcal.config import Config

import numpy as np

from numpy.typing import NDArray


def find_pulse_index(config: Config, param: str):
    """Find the first index of a physical pulse of a gate in a composite pulse.

    This function will ignore any pre-pulses that are listed as string objects
    which reference other pulses defined in the config.

    Args:
        config (Config): qcal Config object.
        param (str): parameter path.
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