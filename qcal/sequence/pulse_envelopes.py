"""Submodule for storing various definitions of pulse envelopes.

"""
from qcal.units import MHz
from qcal.sequence.utils import cosine_basis_function, solve_coefficients

import logging
import numpy as np

from collections import defaultdict
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typing import List

logger = logging.getLogger(__name__)


__all__ = [
    'cosine_square',
    'custom',
    'DRAG',
    'FAST',
    'FAST_DRAG',
    'gaussian',
    'gaussian_square',
    'linear',
    'RAP',
    'STARAP',
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
    drag_pulse = np.array(
            det * (gauss + 1j * alpha * dgauss_dx / delta)
        ).astype(np.complex64)

    drag_pulse /= max(abs(drag_pulse)) # Normalize

    return amp*drag_pulse


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
            [0.9 * abs(anh), 1.1 * abs(anh)],  # Around ef-transition
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

    fast_drag = np.array(I + 1j * Q).astype(np.complex64)

    fast_drag /= max(abs(fast_drag)) # Normalize

    return amp*fast_drag


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


def RAP(
       length: float, sample_rate: float, env: str, f0: float, f1: float, 
        **kwargs
    ) -> NDArray[np.complex64]:
    """Rapid Adiabatic Passage pulse.

    This pulse implements a frequency sweep (about the carrier frequency) by
    advancing the phase of the pulse in time.

    Args:
        env (str): envelopoe of the underlying pulse.
        f0 (float): start frequency in Hz.
        f1 (float): end frequency in Hz.
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.

    Returns:
        NDArray[np.complex64]: RAP pulse.
    """
    n_points = int(round(length * sample_rate))
    t = np.linspace(0, length, n_points)
    dt = t[1] - t[0]

    # Linear frequency sweep: f(t) = f0 + (f1-f0)*t/T
    # Phase: phi(t) = 2*pi*f0*t + pi*(f1-f0)*t^2/T
    # phi_t = 2 * np.pi * f0 * t + np.pi * (f1 - f0) * t**2 / length

    # Method 2: Numerical integration (more general)
    f_t = f0 + (f1 - f0) * t / length  # instantaneous frequency, linear chirp
    phi_t = 2 * np.pi * np.cumsum(f_t) * dt
    phi_t = phi_t - phi_t[0]  # start at zero phase

    # Get amplitude envelope
    pulse = pulse_envelopes[env](length, sample_rate, **kwargs)

    return pulse * np.exp(1j * phi_t)


def STARAP(
        length: float, sample_rate: float, env: str, f0: float, f1: float, 
        omega0: float = 10 * MHz, cd_weight: float = 0.5, **kwargs
    ) -> NDArray[np.complex64]:
    """Shortcut to Adiabaticity enhanced Rapid Adiabatic Passage pulse.

    This pulse implements a frequency sweep with counter-diabatic driving
    to enable faster adiabatic evolution.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        env (str): envelope of the underlying pulse.
        f0 (float): start frequency in Hz.
        f1 (float): end frequency in Hz.
        omega0: Rabi frequency in Hz. Defaults to 10 MHz.
        cd_weight: weight of counter-diabatic term. Defaults to 1.0, which is
            the theoretical optimum.

    Returns:
        NDArray[np.complex64]: STA-RAP pulse.
    """
    n_points = int(round(length * sample_rate))
    t = np.linspace(0, length, n_points)
    dt = t[1] - t[0]

    # Get amplitude envelope
    envelope = pulse_envelopes[env](length, sample_rate, **kwargs)

    # Frequency sweep
    # tanh frequency sweep to avoid singularity at the center
    a = 6                     # Steepness parameter
    s = 2 * (t / length) - 1  # Normalize t to [-1, 1]
    f_t = f0 + (f1 - f0) * (np.tanh(a * s) + 1) / 2

    # Phase calculation
    phi_t = 2 * np.pi * np.cumsum(f_t) * dt
    phi_t = phi_t - phi_t[0]

    # STA calculations
    # Convert to detuning from center frequency
    # Add epsilon to avoid division issues  
    eps = 1e-12
    f_center = (f0 + f1) / 2
    delta_t = 2 * np.pi * (f_t - f_center)
    delta_t = delta_t + eps * (np.abs(delta_t) < eps)

    # Rabi frequency
    omega_t = omega0 * 2 * np.pi * np.abs(envelope)

    # Calculate mixing angle
    theta_t = -np.arctan2(omega_t, delta_t) # Minus to go from 0 to pi

    # Calculate derivative with smoothing to reduce noise
    theta_smooth = gaussian_filter1d(theta_t, sigma=2.0)
    theta_dot = np.gradient(theta_smooth, dt)

    # Construct the STA pulse
    # X component (original drive)
    # Y component (counter-diabatic drive) 
    sta_pulse = (omega_t/2 + cd_weight * 1j * theta_dot/2) * np.exp(1j * phi_t)

    # Apply envelope phase safely
    # envelope_abs = np.abs(envelope)
    # envelope_phase = np.where(
    #     envelope_abs > eps, 
    #     envelope / envelope_abs, 
    #     1.0 + 0j
    # )
    # sta_pulse *= envelope_phase

    sta_pulse /= np.max(np.abs(sta_pulse))  # Normalize

    return sta_pulse.astype(np.complex64)


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
    'cosine_square':        cosine_square,
    'custom':               custom,
    'DRAG':                 DRAG, 
    'FAST':                 FAST,
    'FAST_DRAG':            FAST_DRAG,
    'gaussian':             gaussian, 
    'gaussian_square':      gaussian_square,
    'linear':               linear, 
    'RAP':                  RAP,
    'STARAP':               STARAP,
    'sine':                 sine, 
    'square':               square,
    'virtualz':             virtualz,
})