"""Helper functions for calibration scripts.

"""
from qcal.config import Config


def find_pulse_index(config: Config, param: str):
    """Find the first index of a physical pulse in a composite pulse.

    Args:
        config (Config): qcal Config object.
        param (str): parameter path.
    """
    for i, pulse in enumerate(config[param]):
        if pulse['env'] != 'virtualz':
            return i
        else:
            continue