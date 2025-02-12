"""Submodule for storing various definitions of pulse envelopes.

"""
import numpy as np

from collections import defaultdict
from numpy.typing import NDArray
from typing import Union


__all__ = [
    'cosine_square',
    'custom',
    'DRAG',
    'linear',
    'gaussian',
    'sine',
    'square',
    'virtualz',
    # 'zz_DRAG'
]


def cosine_square(
        length: float, sample_rate: float, amp: float = 1.0,
        phase: float = 0.0, ramp_fraction: float = 0.25
    ) -> NDArray:
    """Pulse envelope with cosine ramps and a flat top.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        phase (float, optional): pulse phase. Defaults to 0.0.
        ramp_fraction (float, optional): fraction of ramp for rising and 
            falling edge. Defaults to 0.25.

    Returns:
        NDArray: envelope with cosine ramps and a flat top.
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
        length: float, sample_rate: float, alpha: float = 0.0, 
        amp: float = 1.0, anh: float = -270e6, df: float = 0.0,
        n_sigma: int = 3, phase: float = 0.0
    ) -> NDArray:
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
        anh (float, optional): qubit anharmonicity. Defaults to -270e6.
        df (float, optional): frequency detuning. Defaults to 0.0.
        n_sigma (int, optional): number of standard deviations. Defaults to 3.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray: DRAG envelope.
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


def linear(
        length: float, sample_rate: float, amp: float = 1.0, phase: float = 0.0
    ) -> NDArray:
    """Linear ramp pulse envelope.

    The linear ramp starts at 0, and ends at amp.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray: linear ramp envelop.
    """
    n_points = int(round(length * sample_rate))
    return np.array(
            amp * np.linspace(0, 1, n_points) * np.exp(1j*phase)
        ).astype(np.complex64)


def gaussian(
        length: float, sample_rate: float, amp: float = 1.0,
        n_sigma: Union[float, int] = 3, phase: float = 0.0
    ) -> NDArray:
    """Gaussian pulse envelope.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        n_sigma (int, optional): number of standard deviations. Defaults to 3.
        phase (float, optional): pulse phase. Defaults to 0.0.
        
    Returns:
        NDArray: Gaussian envelope.
    """
    n_points = int(round(length * sample_rate))
    sigma = n_points / (2. * n_sigma)
    x = np.arange(0, n_points)
    gauss = np.exp(-0.5 * (x - n_points / 2.) ** 2 / sigma ** 2)
    return np.array(amp * gauss * np.exp(1j * phase)).astype(np.complex64)


def gaussian_square(
        length: float, sample_rate: float, amp: float = 1.0,
        n_sigma: int = 3, phase: float = 0.0, ramp_fraction: float = 0.25
    ) -> NDArray:
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
        NDArray: _description_
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
        length: float, sample_rate: float, amp: float = 1.0, 
        freq: float = 0.0, phase: float= 0.0
    ) -> NDArray:
    """Sine pulse envelope.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        freq (float, optional): frequency of the sine wave. Defaults to 0.0.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray: sine envelope.
    """
    n_points = int(round(length * sample_rate))
    t = np.linspace(0, length, n_points)
    return np.array(
        amp * np.exp(1j * (2.0 * np.pi * freq * t + phase)).astype('complex64')
    )


def square(
        length: float, sample_rate: float, amp: float = 1.0, phase: float = 0.0
    ) -> NDArray:
    """Square pulse envelope.

    Args:
        length (float): pulse length in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray: square envelope.
    """
    n_points = int(round(length * sample_rate))
    return np.array(
            amp * np.ones(n_points) * np.exp(1j*phase)
        ).astype('complex64')


def virtualz(
        length: float = 0, sample_rate: float = 0, 
        amp: float = 1.0, phase: float= 0.0
    ) -> np.complex64:
    """Virtual-z phase.

    Args:
        length (float): pulse length in seconds. Defaults to 0.
        sample_rate (float): sample rate in Hz. Defaults to 0.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        np.complex64: phase.
    """
    return np.exp(1j*phase).astype(np.complex64)


# def zz_DRAG(
#         length: float, sample_rate: float, alpha: float = 0.0, amp: float = 1.0, 
#         df: float = 0.0, phase: float = 0.0
#     ) -> NDArray:
#     """ZZ-Interaction-Free pulse.

#     The a0 and a2 parameters are optimized for a 40 ns pulse.
    
#     Reference: https://arxiv.org/pdf/2309.13927

#     Args:
#         length (float): pulse length in seconds.
#         sample_rate (float): sample rate in Hz.
#         alpha (float, optional): DRAG parameter. Defaults to 0.0. For phase
#             errors, alpha should be 0.5. For leakage errors, alpha should be 1.
#         amp (float, optional): pulse amplitude. Defaults to 1.0.
#         df (float, optional): frequency detuning. Defaults to 0.0.
#         phase (float, optional): pulse phase. Defaults to 0.0.

#     Returns:
#         NDArray: ZZ DRAG envelope.
#     """
#     a0 = 0.31831
#     a2 = -0.00515
#     amp /= a0  # Rescale to a maximum of 1.0

#     n_points = int(round(length * sample_rate))
#     # t = np.arange(0, n_points).astype(np.complex64)
#     t = np.linspace(0, length, n_points).astype(np.complex64)
#     cos = np.cos((np.pi/length) * (t - length/2))
#     env = a0 * cos**2 + a2 * (t - length/2)**2 * cos**2

#     # DRAG correction
#     sin = np.sin((np.pi/length) * (t - length/2))
#     c = -2 * np.pi / length
#     env_DRAG = 1.j * alpha * (
#         c * a0 * cos * sin + 
#         2 * a2 * (t - length/2) * cos**2 + 
#         c * a2 * (t - length/2)**2 * cos * sin
#     )
#     env += env_DRAG

#     # Detuning df offset
#     df /= sample_rate
#     x = np.linspace(0, length, n_points).astype(np.complex64)
#     env *= np.exp(2 * np.pi * 1j * df * x)

#     return np.array(amp * env * np.exp(1j * phase))


pulse_envelopes = defaultdict(lambda: 'Pulse envelope not available!', {
    'cosine_square': cosine_square,
    'custom':        custom,
    'DRAG':          DRAG, 
    'linear':        linear, 
    'gaussian':      gaussian, 
    'sine':          sine, 
    'square':        square,
    'virtualz':      virtualz,
    # 'zz_DRAG':       zz_DRAG
})