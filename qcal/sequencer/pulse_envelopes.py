"""Submodule for storing various definitions of pulse envelopes.

"""
import numpy as np

from numpy.typing import NDArray
from typing import Union


all = ['cosine_square', 'DRAG', 'linear', 'gaussian', 'sine', 'square']


def cosine_square(
        width: float, sample_rate: float, amp: float = 1.0,
        phase: float = 0.0, ramp_fraction: float = 0.25
    ) -> NDArray:
    """Pulse envelope with cosine ramps and a flat top.

    Args:
        width (float): pulse width in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        phase (float, optional): pulse phase. Defaults to 0.0.
        ramp_fraction (float, optional): fraction of ramp for rising and 
            falling edge. Defaults to 0.25.

    Returns:
        NDArray: envelope with cosine ramps and a flat top.
    """
    assert ramp_fraction <= 0.5, 'ramp_fraction cannot be more than 0.5!'
    n_points = int(round(width * sample_rate))
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


def DRAG(
        width: float, sample_rate: float, alpha: float = 1.0, 
        amp: float = 1.0, anh: float = -270e6, df: float = 0.0,
        n_sigma: int = 3, phase: float = 0.0
    ) -> NDArray:
    """Derivative Removal by Adiabatic Gate (DRAG) pulse envelope.

    References:
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.110501
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.83.012308
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.020501

    Args:
        width (float): pulse width in seconds.
        sample_rate (float): sample rate in Hz.
        alpha (float, optional): DRAG parameter. Defaults to 1.0. For phase
            errors, alpha should be 0.5. For leakage errors, alpha should be 1.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        anh (float, optional): qubit anharmonicity. Defaults to -270e6.
        df (float, optional): frequency detuning. Defaults to 0.0.
        n_sigma (int, optional): number of standard deviations. Defaults to 3.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray: DRAG envelope.
    """
    df = 2 * np.pi * df / sample_rate
    delta = 2 * np.pi * anh / sample_rate - df
    n_points = int(round(width * sample_rate))
    sigma = n_points / (2. * n_sigma)
    x = np.arange(0, n_points)
    det = np.exp(2 * np.pi * 1j * df * x)

    gauss = gaussian(width, sample_rate, amp, n_sigma, phase)
    dgauss_dx = -(x - n_points / 2.) / (sigma ** 2) * gauss
    return np.array(
            det * (gauss - 1j * alpha * dgauss_dx / delta)
        ).astype(np.complex64)


def linear(
        width: float, sample_rate: float, amp: float = 1.0, phase: float = 0.0
    ) -> NDArray:
    """Linear ramp pulse envelope.

    The linear ramp starts at 0, and ends at amp.

    Args:
        width (float): pulse width in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray: linear ramp envelop.
    """
    n_points = int(round(width * sample_rate))
    return np.array(
            amp * np.linspace(0, 1, n_points) * np.exp(1j*phase)
        ).astype(np.complex64)


def gaussian(
        width: float, sample_rate: float, amp: float = 1.0,
         n_sigma: Union[float, int] = 3, phase: float = 0.0
    ) -> NDArray:
    """Gaussian pulse envelope.

    Args:
        width (float): pulse width in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        n_sigma (int, optional): number of standard deviations. Defaults to 3.
        phase (float, optional): pulse phase. Defaults to 0.0.
        
    Returns:
        NDArray: Gaussian envelope.
    """
    n_points = int(round(width * sample_rate))
    sigma = n_points / (2. * n_sigma)
    x = np.arange(0, n_points)
    gauss = np.exp(-0.5 * (x - n_points / 2.) ** 2 / sigma ** 2)
    return np.array(amp * gauss * np.exp(1j * phase)).astype(np.complex64)


def gaussian_square(
        width: float, sample_rate: float, amp: float = 1.0,
        n_sigma: int = 3, phase: float = 0.0, ramp_fraction: float = 0.25
    ) -> NDArray:
    """Pulse envelope with Gaussian ramps and a flat top.

    Args:
        width (float): pulse width in seconds.
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
    n_points = int(round(width * sample_rate))
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


def sin(
        width: float, sample_rate: float, amp: float = 1.0, 
        freq: float = 0.0, phase: float= 0.0
    ) -> NDArray:
    """Sine pulse envelope.

    Args:
        width (float): pulse width in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        freq (float, optional): frequency of the sine wave. Defaults to 0.0.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray: sine envelope.
    """
    n_points = int(round(width * sample_rate))
    t = np.linspace(0, width, n_points)
    return np.array(
        amp * np.exp(1j * (2.0 * np.pi * freq * t + phase)).astype('complex64')
    )


def square(
        width: float, sample_rate: float, amp: float = 1.0, phase: float = 0.0
    ) -> NDArray:
    """Square pulse envelope.

    Args:
        width (float): pulse width in seconds.
        sample_rate (float): sample rate in Hz.
        amp (float, optional): pulse amplitude. Defaults to 1.0.
        phase (float, optional): pulse phase. Defaults to 0.0.

    Returns:
        NDArray: square envelope.
    """
    n_points = int(round(width * sample_rate))
    return np.array(
            amp * np.ones(n_points) * np.exp(1j*phase)
        ).astype('complex64')