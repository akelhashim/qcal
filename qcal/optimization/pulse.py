"""Optimization of pulse envelopes.

Provides :func:`optimize_FAST_DRAG`, which tunes the hyperparameters of a
FAST DRAG pulse to suppress its spectral weight at the EF transition
frequency (i.e. to reduce leakage).
"""
import logging
from typing import Callable, Dict

import numpy as np
from IPython.display import clear_output

import qcal.settings as settings
from qcal.config import Config
from qcal.plotting.sequence import plot_pulse

logger = logging.getLogger(__name__)


def optimize_FAST_DRAG(
    config:               Config,
    pulse_param:          str,
    pulse_envelope_func:  Callable,
    param_sweep:          Dict | None = None
) -> None:
    """Optimize the FAST DRAG pulse hyperparameters for reducing leakage.

    The hyperparameters are optimized by minimizing the magnitude of the
    pulse's frequency spectrum at the EF transition frequency. First, the
    number of Fourier components ``N`` is swept and set to the value which
    minimizes the spectral magnitude; then, each weight in ``weights`` is
    swept in turn. The optimized values are written to the config (under
    ``{pulse_param}/kwargs``), the optimized pulse is plotted, and the
    config is saved if ``settings.Settings.save_data`` is True.

    Args:
        config (Config): qcal Config object.
        pulse_param (str): config parameter for the pulse to optimize, e.g.
            ``'single_qubit/0/GE/FAST_DRAG'``. The qubit label is parsed
            from the second element of the path.
        pulse_envelope_func (Callable): function which generates the pulse
            envelope. It is called as
            ``pulse_envelope_func(config, config[pulse_param])`` and must
            return the complex pulse envelope as an array.
        param_sweep (Dict | None, optional): dictionary with keys ``'N'``
            and ``'w'`` specifying the values of ``N`` and of each weight to
            sweep. Defaults to None, in which case ``N`` is swept from 2 to
            10 and each weight over
            ``[0.1, 0.3, 1.0, 2.0, ..., 10.0]``.
    """
    qubit = int(pulse_param.split('/')[1])
    freq_GE = config[f'single_qubit/{qubit}/GE/freq']
    freq_EF = config[f'single_qubit/{qubit}/EF/freq']

    duration = config[f'{pulse_param}/length']
    pulse_envelope = pulse_envelope_func(config, config[pulse_param])
    t = np.linspace(0, duration, len(pulse_envelope))
    dt = t[1] - t[0]

    if param_sweep is None:
        param_sweep = {
            'N': np.arange(2, 11).astype(int),
            'w': [0.1, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }

    opt_params = {}
    mags = []
    for N in param_sweep['N']:
        config[f'{pulse_param}/kwargs/N'] = N
        pulse_envelope = pulse_envelope_func(config, config[pulse_param])
        padded_pulse_envelope = np.pad(
            pulse_envelope,
            (1000, 1000),
            mode='constant',
            constant_values=0.0 + 0.0j
        )
        freq_spectrum = np.fft.fftshift(np.fft.fft(padded_pulse_envelope))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(padded_pulse_envelope), dt))
        freqs += freq_GE
        spectrum_magnitude = np.abs(freq_spectrum)
        idx = (np.abs(freqs - freq_EF)).argmin()
        mag = spectrum_magnitude[idx]
        mags.append(mag)
    opt_params['N'] = param_sweep['N'][np.argmin(mags)]
    config[f'{pulse_param}/kwargs/N'] = int(param_sweep['N'][np.argmin(mags)])

    for i in range(len(config[f'{pulse_param}/kwargs/weights'])):
        mags = []
        for w in param_sweep['w']:
            config[f'{pulse_param}/kwargs/weights/{i}'] = w
            pulse_envelope = pulse_envelope_func(config, config[pulse_param])
            padded_pulse_envelope = np.pad(
                pulse_envelope,
                (1000, 1000),
                mode='constant',
                constant_values=0.0 + 0.0j
            )
            freq_spectrum = np.fft.fftshift(np.fft.fft(padded_pulse_envelope))
            freqs = np.fft.fftshift(
                np.fft.fftfreq(len(padded_pulse_envelope), dt)
            )
            freqs += freq_GE
            spectrum_magnitude = np.abs(freq_spectrum)
            idx = (np.abs(freqs - freq_EF)).argmin()
            mag = spectrum_magnitude[idx]
            mags.append(mag)
        opt_params[f'w{i}'] = param_sweep['w'][np.argmin(mags)]
        config[f'{pulse_param}/kwargs/weights/{i}'] = (
            param_sweep['w'][np.argmin(mags)]
        )

    clear_output(wait=True)
    logger.info(' Optimzation complete!')
    print('Optimized params: ', opt_params)

    plot_pulse(
        config=config,
        pulse_duration=duration,
        pulse_envelope=pulse_envelope_func(config, config[pulse_param]),
        pulse_qubit=qubit
    )

    if settings.Settings.save_data:
        config.save()
