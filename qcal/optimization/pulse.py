"""Submodule for optimizing pulse envelopes.

"""
import qcal.settings as settings

from qcal.config import Config
from qcal.plotting.sequence import plot_pulse

import logging
import numpy as np

from IPython.display import clear_output
from numpy.typing import NDArray
from typing import Callable, Dict

logger = logging.getLogger(__name__)


def optimize_FAST_DRAG(
        config:               Config,
        pulse_param:          str,
        pulse_envelope_func:  Callable,
        param_sweep:          Dict | None = None
    ) -> NDArray:
    """Optimize the hyperparameters for the FAST DRAG pulse for reducing
    leakage.

    Args:
        config (Config): qcal Config object.
        pulse_param (str): config parameter for the pulse to optimize
        pulse_duration (float): duration of pulse in seconds.
        pulse_envelope (NDArray[np.complex64]): pulse envelope.
        pulse_qubit (int): qubit on which the pulse is to be played. Defaults to
            None. If not None, the qubit's frequencies will be plotted.
        neighbor_qubits (List[int] | None): neighboring qubits on which the 
            pulse is not being played. Defaults to None. If not None, the
            frequencies of the qubits will be plotted. This is currently
            unused.
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
        