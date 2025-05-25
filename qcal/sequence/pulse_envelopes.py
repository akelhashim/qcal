"""Submodule for storing various definitions of pulse envelopes.

"""
from qcal.sequence.utils import cosine_basis_function, solve_coefficients

import logging
import numpy as np

from collections import defaultdict
from numpy.typing import NDArray
from typing import List, Union

logger = logging.getLogger(__name__)


__all__ = [
    'cosine_square',
    'custom',
    'DRAG',
    'FAST',
    'FAST_DRAG',
    'linear',
    'gaussian',
    'sine',
    'square',
    'virtualz',
]


def cosine_square(
        length:        float, 
        sample_rate:   float, 
        amp:           float = 1.0,
        phase:         float = 0.0, 
        ramp_fraction: float = 0.25
    ) -> NDArray[np.complex64]:
    """Pulse envelope with cosine ramps and a flat top.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        phase (float, optional): pulse phase. Defaults to 0.0.
        ramp_fraction (float, optional): fraction of ramp for rising and 
            falling edge. Defaults to 0.25.

    Returns:
        NDArray[np.complex64]: envelope with cosine ramps and a flat top.
    """
    assert ramp_fraction <= 0.5, 'ramp_fraction cannot be more than 0.5!'
    n_points = int(round(length * sample_rate))
    n_points_cos = int(round(n_points * ramp_fraction))
    n_points_square = n_points - 2 * n_points_cos
    freq = 1. / (2 * n_points_cos)

    cos = (
        np.cos(2. * np.pi * freq * np.arange(0, 2*n_points_cos) - np.pi ) + 1
    ) / 2.
    square = np.ones([n_points_square])
    cos_square = np.concatenate(
        (cos[:n_points_cos], square, cos[n_points_cos:])
    )
    return np.array(amp * cos_square * np.exp(1j * phase)).astype(np.complex64)


def custom(length: float, sample_rate: float, filename: str) -> NDArray:
    """Load a custom pulse envelope.

    Args:
        length (float): pulse length in seconds. This argument is unused.
        sample_rate (float): sample rate in Hz. This argument is unused.
        filename (str): filename where the custom pulse envelope is saved.

    Returns:
        NDArray: custom pulse envelope.
    """
    return np.load(filename).astype(np.complex64)


def DRAG(
        length:      float, 
        sample_rate: float, 
        alpha:       float = 0.0, 
        amp:         float = 1.0, 
        anh:         float = -200e6, 
        df:          float = 0.0,
        n_sigma:     int = 3, 
        phase:       float = 0.0
    ) -> NDArray[np.complex64]:
    """Derivative Removal by Adiabatic Gate (DRAG) pulse envelope.

    References:
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.110501
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.83.012308
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.020501

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        alpha (float, optional): DRAG parameter. Defaults to 0.0. For phase
            errors, alpha should be 0.5. For leakage errors, alpha should be 1.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        anh (float, optional): qubit anharmonicity. Defaults to -200e6.
        df (float, optional): frequency detuning. Defaults to 0.0.
        n_sigma (int, optional): number of standard deviations. Defaults to 3.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray[np.complex64]: DRAG envelope.
    """
    df /= sample_rate
    delta = 2 * np.pi * (anh / sample_rate + df)
    n_points = int(round(length * sample_rate))
    sigma = n_points / (2. * n_sigma)
    x = np.arange(0, n_points)
    det = np.exp(2 * np.pi * 1j * df * x)

    gauss = gaussian(length, sample_rate, amp, n_sigma, phase)
    dgauss_dx = -(x - n_points / 2.) / (sigma ** 2) * gauss
    return np.array(
            det * (gauss - 1j * alpha * dgauss_dx / delta)
        ).astype(np.complex64)


def FAST(
        length:         float, 
        sample_rate:    float, 
        freq_intervals: List, 
        weights:        List, 
        amp:            float = 1.0, 
        N:              int = 4, 
        phase:          float = 0.0, 
        theta:          float = np.pi/2 
    ) -> NDArray[np.complex64]:
    """Fourier Ansatz Spectrum Tuning (FAST) pulse envelope.

    Reference:
    https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.030353

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        freq_intervals (List): list of [f_low, f_high] pairs.
        weights (List): weights for each frequency interval.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        N (int, optional): number of Fourier terms.
        phase (float, optional): pulse phase. Defaults to 0.0.
        theta (float, optional): rotation angle. Defaults to pi/2.

    Returns:
        NDArray[np.complex64]: FAST envelope.
    """
    # Setup time array
    n_points = int(round(length * sample_rate))
    t = np.linspace(0, length, n_points)

    # Solve for coefficients
    coefficients = solve_coefficients(
        N=N, 
        t_p=length, 
        theta=theta, 
        freq_intervals=freq_intervals, 
        weights=weights,
    )
    
    # Generate envelope
    fast = np.zeros_like(t)
    for n in range(N):
        fast += coefficients[n] * cosine_basis_function(n+1, t, length)
    fast /= np.max(np.abs(fast))
    
    return np.array(amp * fast * np.exp(1j*phase)).astype(np.complex64)


def FAST_DRAG(
        length:         float, 
        sample_rate:    float, 
        freq_intervals: List = None, 
        weights:        List = None, 
        alpha:          float = 0.0, 
        amp:            float = 1.0, 
        anh:            float = -200e6, 
        N:              int = 4, 
        phase:          float = 0.0, 
        theta:          float = np.pi/2 
    ) -> NDArray[np.complex64]:
    """Fourier Ansatz Spectrum Tuning (FAST) DRAG pulse envelope.

    Reference:
    https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.030353

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        freq_intervals (List): list of [f_low, f_high] pairs.
        weights (List): weights for each frequency interval.
        alpha (float, optional): DRAG parameter. Defaults to 0.0. For phase
            errors, alpha should be 0.5. For leakage errors, alpha should be 1.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        anh (float, optional): qubit anharmonicity. Defaults to -200e6.
        N (int, optional): number of Fourier terms.
        phase (float, optional): pulse phase. Defaults to 0.0.
        theta (float): rotation angle. Defaults to pi/2.

    Returns:
        NDArray[np.complex64]: FAST DRAG envelope.
    """
    # Setup time array
    dt = 1.0 / sample_rate
    delta = 2 * np.pi * anh
    
    # Default frequency intervals based on anharmonicity if not provided
    if freq_intervals is None:
        freq_intervals = [
            [abs(anh) * 0.8, abs(anh) * 1.2],  # Around ef-transition
            [3 * abs(anh), 5 * abs(anh)]       # High frequency cutoff
        ]
    
    if weights is None:
        weights = [1.0, 0.1]  # Higher weight for ef-transition suppression
    
    # Generate I component using FAST
    I = FAST(
        length=length, 
        sample_rate=sample_rate, 
        freq_intervals=freq_intervals, 
        weights=weights, 
        amp=amp, 
        N=N, 
        phase=phase, 
        theta=theta
    )

    # Generate Q component using DRAG
    Q = alpha / delta * np.gradient(I, dt)

    fast_drag = I + 1j * Q
    
    return np.array(fast_drag).astype(np.complex64)


def linear(
        length: float, sample_rate: float, amp: float = 1.0, phase: float = 0.0
    ) -> NDArray[np.complex64]:
    """Linear ramp pulse envelope.

    The linear ramp starts at 0, and ends at amp.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray[np.complex64]: linear ramp envelop.
    """
    n_points = int(round(length * sample_rate))
    return np.array(
            amp * np.linspace(0, 1, n_points) * np.exp(1j*phase)
        ).astype(np.complex64)


def gaussian(
        length:      float, 
        sample_rate: float, 
        amp:         float = 1.0,
        n_sigma:     int = 3, 
        phase:       float = 0.0
    ) -> NDArray[np.complex64]:
    """Gaussian pulse envelope.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        n_sigma (int, optional): number of standard deviations. Defaults to 3.
        phase (float, optional): pulse phase. Defaults to 0.0.
        
    Returns:
        NDArray[np.complex64]: Gaussian envelope.
    """
    n_points = int(round(length * sample_rate))
    sigma = n_points / (2. * n_sigma)
    x = np.arange(0, n_points)
    gauss = np.exp(-0.5 * (x - n_points / 2.) ** 2 / sigma ** 2)
    return np.array(amp * gauss * np.exp(1j * phase)).astype(np.complex64)


def gaussian_square(
        length:        float, 
        sample_rate:   float, 
        amp:           float = 1.0,
        n_sigma:       int = 3, 
        phase:         float = 0.0, 
        ramp_fraction: float = 0.25
    ) -> NDArray[np.complex64]:
    """Pulse envelope with Gaussian ramps and a flat top.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        phase (float, optional): pulse phase. Defaults to 0.0.
        n_sigma (int, optional): number of standard deviations. Defaults to 3.
        ramp_fraction (float, optional): fraction of ramp for rising and 
            falling edge. Defaults to 0.25.

    Returns:
        NDArray[np.complex64]: Gaussian square envelope.
    """
    assert ramp_fraction <= 0.5, 'ramp_fraction cannot be more than 0.5!'
    n_points = int(round(length * sample_rate))
    n_points_gauss = int(round(n_points * ramp_fraction))
    n_points_square = n_points - 2 * n_points_gauss
    sigma = n_points_gauss / n_sigma
    x = np.arange(0, 2 * n_points_gauss)

    gauss = np.exp(-0.5 * (x - n_points_gauss) ** 2 / sigma ** 2)
    square = np.ones([n_points_square])
    gauss_square = np.concatenate(
        (gauss[:n_points_gauss], square, gauss[n_points_gauss:])
    )
    return np.array(
            amp * gauss_square * np.exp(1j * phase)
        ).astype(np.complex64)


def sine(
        length:      float, 
        sample_rate: float, 
        amp:         float = 1.0, 
        freq:        float = 0.0, 
        phase:       float= 0.0
    ) -> NDArray[np.complex64]:
    """Sine pulse envelope.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        freq (float, optional): frequency of the sine wave. Defaults to 0.0.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray[np.complex64]: sine envelope.
    """
    n_points = int(round(length * sample_rate))
    t = np.linspace(0, length, n_points)
    return np.array(
        amp * np.exp(1j * (2.0 * np.pi * freq * t + phase)).astype('complex64')
    )


def square(
        length: float, sample_rate: float, amp: float = 1.0, phase: float = 0.0
    ) -> NDArray[np.complex64]:
    """Square pulse envelope.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray[np.complex64]: square envelope.
    """
    n_points = int(round(length * sample_rate))
    return np.array(
            amp * np.ones(n_points) * np.exp(1j*phase)
        ).astype('complex64')


def virtualz(
        length:      float = 0, 
        sample_rate: float = 0, 
        amp:         float = 1.0, 
        phase:       float= 0.0
    ) -> np.complex64:
    """Virtual-z phase.

    Args:
        length (float): pulse length in seconds. Defaults to 0. Unused, but
            included for consistency.
        sample_rate (float): sample rate in Hz. Defaults to 0. Unused, but
            included for consistency.
        amp (float, optional): pulse amplitude. Defaults to 1.0. Unused, but
            included for consistency.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        np.complex64: phase.
    """
    return np.exp(1j*phase).astype(np.complex64)


pulse_envelopes = defaultdict(lambda: 'Pulse envelope not available!', {
    'cosine_square': cosine_square,
    'custom':        custom,
    'DRAG':          DRAG, 
    'FAST':          FAST,
    'FAST_DRAG':     FAST_DRAG,
    'linear':        linear, 
    'gaussian':      gaussian, 
    'sine':          sine, 
    'square':        square,
    'virtualz':      virtualz,
})