"""Helper functions for sequencer.

"""
import logging
import numpy as np

from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def clip_amplitude(
        amp: float | NDArray, max_amp: float = 1.0
    ) -> float | NDArray:
    """Clip the amplitude of a pulse to a maximum value.

    Args:
        amp (float | NDArray): amplitude value or pulse arrray.
        max_amp (float, optional): maximum amplitude. Defaults to 1.0.

    Returns:
        NDArray: pulse array with clipped amplitude.
    """
    if isinstance(amp, float):
        if amp > 1.0:
            return 1.0
        else:
            return amp
    else:
        amp[amp > max_amp] = max_amp
        return amp