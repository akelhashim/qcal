"""Helper functions for sequencer.

"""
import logging
import numpy as np

from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def clip_amplitude(
        amp: float | NDArray, min_amp: float = -1.0, max_amp: float = 1.0
    ) -> float | NDArray:
    """Clip the amplitude of a pulse to a maximum value.

    Args:
        amp (float | NDArray): amplitude value or pulse arrray.
        min_amp (float, optional): minimum amplitude. Defaults to -1.0.
        max_amp (float, optional): maximum amplitude. Defaults to 1.0.

    Returns:
        NDArray: pulse array with clipped amplitude.
    """
    return np.clip(amp, min_amp, max_amp)